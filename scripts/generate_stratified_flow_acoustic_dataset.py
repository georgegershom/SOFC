#!/usr/bin/env python3
"""
Stratified Two-Phase Flow Acoustics Dataset Generator

Generates a physically-informed synthetic dataset for experiments studying
attenuation mechanisms in stratified gas-liquid flows.

Outputs per run:
- metadata.json: run configuration, geometry, instrumentation
- fluid_properties.csv: densities, viscosities, temperature, pressure
- flow_regime.csv: U_SG, U_SL, alpha, flow pattern, interface height, wave amplitude
- interface_height.csv: time series of interface height
- signals/source_signal.csv: time, pressure (Pa)
- signals/received_signal_pos_{i}.csv: time, pressure (Pa) per hydrophone
- psd/source_psd.csv and psd/received_psd_pos_{i}.csv: frequency, PSD
- attenuation_profiles/tl_pos_{i}.csv: frequency, transmission loss (dB)
- attenuation_profiles/gamma_model.csv: frequency, attenuation coefficient gamma (1/m)
- turbulence/velocity_profile_{phase}.csv: r/R, U (m/s)
- turbulence/tke_epsilon_{phase}.csv: r/R, k (m^2/s^2), epsilon (m^2/s^3)
- turbulence/shear_stress.csv: tau_wall_L, tau_wall_G, tau_interface (Pa)
- instrumentation.csv: sensor layout and basic calibration data
- A top-level dataset_manifest.csv is created in the root output directory summarizing runs.

This script is designed to be fast and deterministic (given a seed) while
producing plausible signals and metrics for algorithm development and validation.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from scipy.signal import welch
except Exception:
    welch = None  # Will fallback to numpy FFT if scipy is unavailable


# ----------------------------- Data Classes ----------------------------- #

@dataclass
class Geometry:
    pipe_inner_diameter_m: float = 0.05  # 5 cm
    pipe_length_m: float = 10.0
    hydrophone_positions_m: Tuple[float, ...] = (1.0, 3.0, 5.0, 7.0, 9.0)


@dataclass
class DAQ:
    sampling_rate_hz: int = 10_000
    duration_s: float = 2.0


@dataclass
class FluidProperties:
    rho_g_kg_per_m3: float
    rho_l_kg_per_m3: float
    mu_g_pa_s: float
    mu_l_pa_s: float
    temperature_c: float
    pressure_bar: float


@dataclass
class FlowRegime:
    U_SG_m_per_s: float
    U_SL_m_per_s: float
    void_fraction_alpha: float
    flow_pattern: str  # "smooth" or "wavy"
    interface_height_m: float
    wave_amplitude_m: float


@dataclass
class Instrumentation:
    acoustic_source: str
    hydrophones: int
    hydrophone_sensitivity_v_per_pa: float
    pressure_transmitters: int
    conductance_probes: int
    impedance_sensors: int
    thermocouples: int
    viscometer: str
    hydrometer: str
    daq_model: str


@dataclass
class RunConfig:
    run_id: str
    seed: int
    geometry: Geometry
    daq: DAQ
    fluids: FluidProperties
    flow: FlowRegime
    instrumentation: Instrumentation


# ----------------------------- Utilities ----------------------------- #


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def save_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------- Physics Models ----------------------------- #


def classify_flow_pattern(U_SG: float, U_SL: float, rho_g: float, rho_l: float, D: float) -> str:
    """Simple classifier: wavy if gas inertia overcomes interfacial restoring forces.
    Uses a heuristic based on superficial velocities and Froude-like scaling.
    """
    g = 9.81
    Fr_g = U_SG / math.sqrt(g * D + 1e-9)
    Re_l = rho_l * max(U_SL, 1e-3) * D / max(1e-6, 1e-3)
    # Heuristic thresholds
    if Fr_g > 1.0 or Re_l > 4000:
        return "wavy"
    return "smooth"


def estimate_void_fraction(U_SG: float, U_SL: float) -> float:
    """Estimate void fraction alpha from superficial velocities for stratified flow.
    This uses a bounded ratio model where alpha tends to the gas flow share.
    """
    alpha = U_SG / (U_SG + U_SL + 1e-9)
    # Keep within 5%..95% to avoid degenerate layers
    return float(np.clip(alpha, 0.05, 0.95))


def interface_height_from_alpha(alpha: float, D: float) -> float:
    """Approximate interface height (liquid depth) from void fraction.
    For simplicity, use h = (1 - alpha) * D.
    """
    return (1.0 - alpha) * D


def wave_amplitude(flow_pattern: str, U_SG: float, U_SL: float, D: float) -> float:
    base = 0.0005 if flow_pattern == "smooth" else 0.003
    scale = 0.001 * (U_SG + 0.5 * U_SL) / max(0.05, D)
    return float(np.clip(base + scale, 0.0002, 0.01))  # 0.2 mm .. 10 mm


def friction_factor(Re: float) -> float:
    if Re < 2300:
        return 64.0 / max(Re, 1e-6)
    # Blasius for smooth pipes
    return 0.3164 * (Re ** -0.25)


def compute_turbulence_profiles(
    rho: float,
    mu: float,
    D: float,
    U_bulk: float,
    n_points: int = 64,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Compute mean velocity, k, epsilon profiles using simple turbulence models.
    Returns (velocity_df, tke_eps_df, tau_wall).
    """
    Re = rho * U_bulk * D / mu
    cf = friction_factor(Re)
    tau_wall = 0.5 * cf * rho * U_bulk ** 2
    u_star = math.sqrt(max(tau_wall / rho, 1e-12))
    kappa = 0.41
    # Radial coordinate normalized by radius (0 at center, 1 at wall)
    r_by_R = np.linspace(0.0, 1.0, n_points)
    y_plus_min = 30.0  # outer layer start

    # Use 1/7th power law as a simple profile approximation
    U_max = U_bulk * 7.0 / 6.0
    U_profile = U_max * (1.0 - r_by_R) ** (1.0 / 7.0)

    # TKE and epsilon approximations
    # k ~ (u_star)^2 * C_k, epsilon ~ (u_star^3) / (kappa * y)
    C_k = 1.5
    k_profile = np.full_like(r_by_R, fill_value=C_k * u_star ** 2)
    # Avoid singularity at center; set y = (1 - r_by_R) * R, R cancels in normalization scale
    y_norm = np.maximum(1e-3, (1.0 - r_by_R))
    epsilon_profile = (u_star ** 3) / (kappa * y_norm * (D / 2))

    vel_df = pd.DataFrame({
        "r_by_R": r_by_R,
        "U_m_per_s": U_profile,
    })
    tke_eps_df = pd.DataFrame({
        "r_by_R": r_by_R,
        "k_m2_per_s2": k_profile,
        "epsilon_m2_per_s3": epsilon_profile,
    })
    return vel_df, tke_eps_df, tau_wall


def acoustic_attenuation_model(
    freqs: NDArray[np.floating],
    alpha: float,
    U_SG: float,
    U_SL: float,
    rho_g: float,
    rho_l: float,
    mu_g: float,
    mu_l: float,
    D: float,
) -> NDArray[np.floating]:
    """Frequency-dependent attenuation coefficient gamma(f) [1/m].
    Model includes viscous absorption (∝ f^2), scattering by interfacial waves (∝ f^(4/3)),
    and wall losses (∝ sqrt(f)). Coefficients depend on layer properties.
    """
    # Layer effective properties (very rough mixture rules)
    rho_eff = alpha * rho_g + (1 - alpha) * rho_l
    mu_eff = alpha * mu_g + (1 - alpha) * mu_l

    # Base coefficients scaled by flow intensities
    C_visc = 1e-12 * (mu_eff / 1e-3) * (rho_eff / 1000.0)
    C_scatt = 5e-6 * (alpha * (1 - alpha)) * (0.2 + 0.8 * min(1.0, (U_SG + U_SL) / 10.0))
    C_wall = 2e-4 * (1.0 + 0.5 * (D / 0.05))

    gamma = C_visc * (freqs ** 2) + C_scatt * (np.maximum(freqs, 1.0) ** (4.0 / 3.0)) + C_wall * np.sqrt(np.maximum(freqs, 1.0))
    return gamma.astype(np.float64)


# ----------------------------- Signal Generation ----------------------------- #


def generate_source_signal(daq: DAQ, seed: int) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    rng = np.random.default_rng(seed)
    fs = daq.sampling_rate_hz
    T = daq.duration_s
    t = np.arange(0, T, 1.0 / fs, dtype=np.float64)

    # Mixed content: multi-tone + chirp for broadband assessment
    f_tones = np.array([200.0, 500.0, 1000.0, 2000.0, 3500.0])
    amps = np.array([1.0, 0.8, 0.6, 0.5, 0.4])
    signal_tones = np.sum([a * np.sin(2 * np.pi * f * t) for a, f in zip(amps, f_tones)], axis=0)

    f0, f1 = 150.0, min(0.45 * fs, 4500.0)
    chirp = np.sin(2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / T * t ** 2))

    signal = 250.0 * (0.6 * signal_tones + 0.4 * chirp)  # pressure in Pa

    # Add small source noise
    noise = rng.normal(0.0, 2.0, size=t.shape)
    p_src = (signal + noise).astype(np.float64)
    return t, p_src


def apply_attenuation(
    t: NDArray[np.floating],
    p_src: NDArray[np.floating],
    fs: int,
    distance_m: float,
    gamma_f: NDArray[np.floating],
) -> NDArray[np.floating]:
    n = len(t)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    P = np.fft.rfft(p_src)
    # Transmission over distance: H(f) = exp(-gamma(f) * x)
    H = np.exp(-gamma_f * distance_m)
    P_out = P * H
    p_out = np.fft.irfft(P_out, n=n)
    return p_out.astype(np.float64)


def compute_psd(
    signal: NDArray[np.floating], fs: int, nperseg: int = 2048
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if welch is not None:
        f, Pxx = welch(signal, fs=fs, nperseg=min(nperseg, len(signal)))
    else:
        # Simple periodogram fallback
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        S = np.fft.rfft(signal)
        Pxx = (1.0 / (fs * n)) * (np.abs(S) ** 2)
        f = freqs
    return f.astype(np.float64), Pxx.astype(np.float64)


def compute_snr(
    f: NDArray[np.floating], Pxx_signal: NDArray[np.floating], band: Tuple[float, float]
) -> float:
    fmin, fmax = band
    in_band = (f >= fmin) & (f <= fmax)
    out_band = ~in_band
    signal_power = float(np.trapz(Pxx_signal[in_band], f[in_band]))
    noise_power = float(np.trapz(Pxx_signal[out_band], f[out_band]))
    if noise_power <= 0.0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)


# ----------------------------- Dataset Generation ----------------------------- #


def generate_run(
    out_dir: Path,
    run_index: int,
    base_seed: int,
    geometry: Geometry,
    daq: DAQ,
    fluids: FluidProperties,
    U_SG: float,
    U_SL: float,
) -> Dict[str, str]:
    rng = np.random.default_rng(base_seed + run_index * 97)

    # Flow regime
    alpha = estimate_void_fraction(U_SG, U_SL)
    pattern = classify_flow_pattern(U_SG, U_SL, fluids.rho_g_kg_per_m3, fluids.rho_l_kg_per_m3, geometry.pipe_inner_diameter_m)
    h = interface_height_from_alpha(alpha, geometry.pipe_inner_diameter_m)
    A_w = wave_amplitude(pattern, U_SG, U_SL, geometry.pipe_inner_diameter_m)

    flow = FlowRegime(
        U_SG_m_per_s=U_SG,
        U_SL_m_per_s=U_SL,
        void_fraction_alpha=alpha,
        flow_pattern=pattern,
        interface_height_m=h,
        wave_amplitude_m=A_w,
    )

    # Instrumentation
    instr = Instrumentation(
        acoustic_source="Programmable piezoelectric transducer",
        hydrophones=len(geometry.hydrophone_positions_m),
        hydrophone_sensitivity_v_per_pa=1e-3,
        pressure_transmitters=2,
        conductance_probes=2,
        impedance_sensors=1,
        thermocouples=2,
        viscometer="Rotational viscometer",
        hydrometer="Precision hydrometer",
        daq_model="NI-USB-6343",
    )

    run_id = f"run_{run_index:03d}"
    run_path = out_dir / run_id

    # Save metadata
    cfg = RunConfig(
        run_id=run_id,
        seed=int(base_seed + run_index * 97),
        geometry=geometry,
        daq=daq,
        fluids=fluids,
        flow=flow,
        instrumentation=instr,
    )
    save_json(run_path / "metadata.json", {
        **asdict(cfg),
        "version": 1,
    })

    # Save fluid properties
    save_csv(run_path / "fluid_properties.csv", pd.DataFrame([
        {
            "rho_g_kg_per_m3": fluids.rho_g_kg_per_m3,
            "rho_l_kg_per_m3": fluids.rho_l_kg_per_m3,
            "mu_g_pa_s": fluids.mu_g_pa_s,
            "mu_l_pa_s": fluids.mu_l_pa_s,
            "temperature_c": fluids.temperature_c,
            "pressure_bar": fluids.pressure_bar,
        }
    ]))

    # Save flow regime summary
    save_csv(run_path / "flow_regime.csv", pd.DataFrame([
        {
            "U_SG_m_per_s": flow.U_SG_m_per_s,
            "U_SL_m_per_s": flow.U_SL_m_per_s,
            "void_fraction_alpha": flow.void_fraction_alpha,
            "flow_pattern": flow.flow_pattern,
            "interface_height_m": flow.interface_height_m,
            "wave_amplitude_m": flow.wave_amplitude_m,
        }
    ]))

    # Interface height time-series (simple stationary wave)
    t_int = np.arange(0.0, daq.duration_s, 1.0 / daq.sampling_rate_hz, dtype=np.float64)
    f_wave = 2.0 + 3.0 * rng.random()  # 2..5 Hz waves
    interface_ht = h + A_w * np.sin(2 * np.pi * f_wave * t_int)
    save_csv(run_path / "interface_height.csv", pd.DataFrame({
        "time_s": t_int,
        "interface_height_m": interface_ht,
    }))

    # Source signal
    t, p_src = generate_source_signal(daq, seed=int(base_seed + 13 * run_index))
    save_csv(run_path / "signals" / "source_signal.csv", pd.DataFrame({
        "time_s": t,
        "pressure_Pa": p_src,
    }))

    # Frequency grid for attenuation model (match rFFT freqs)
    n = len(t)
    f_fft = np.fft.rfftfreq(n, d=1.0 / daq.sampling_rate_hz)
    gamma_f = acoustic_attenuation_model(
        f_fft,
        alpha=alpha,
        U_SG=U_SG,
        U_SL=U_SL,
        rho_g=fluids.rho_g_kg_per_m3,
        rho_l=fluids.rho_l_kg_per_m3,
        mu_g=fluids.mu_g_pa_s,
        mu_l=fluids.mu_l_pa_s,
        D=geometry.pipe_inner_diameter_m,
    )
    save_csv(run_path / "attenuation_profiles" / "gamma_model.csv", pd.DataFrame({
        "frequency_hz": f_fft,
        "gamma_per_m": gamma_f,
    }))

    # Received signals at each hydrophone, add measurement noise
    band = (100.0, min(0.45 * daq.sampling_rate_hz, 4500.0))
    received_paths: List[Path] = []
    psd_paths: List[Path] = []
    tl_paths: List[Path] = []

    f_src_psd, Pxx_src = compute_psd(p_src, daq.sampling_rate_hz)
    save_csv(run_path / "psd" / "source_psd.csv", pd.DataFrame({
        "frequency_hz": f_src_psd,
        "psd": Pxx_src,
    }))

    # Hydrophone sensitivity to translate noise floor (arbitrary scale)
    noise_sigma_pa = 1.0 + 0.5 * rng.random()

    for i, x in enumerate(geometry.hydrophone_positions_m, start=1):
        p_x = apply_attenuation(t, p_src, daq.sampling_rate_hz, x, gamma_f)
        # Additive Gaussian noise
        p_x_noisy = p_x + rng.normal(0.0, noise_sigma_pa, size=p_x.shape)

        path_sig = run_path / "signals" / f"received_signal_pos_{i}.csv"
        save_csv(path_sig, pd.DataFrame({
            "time_s": t,
            "pressure_Pa": p_x_noisy,
        }))
        received_paths.append(path_sig)

        f_rec_psd, Pxx_rec = compute_psd(p_x_noisy, daq.sampling_rate_hz)
        path_psd = run_path / "psd" / f"received_psd_pos_{i}.csv"
        save_csv(path_psd, pd.DataFrame({
            "frequency_hz": f_rec_psd,
            "psd": Pxx_rec,
        }))
        psd_paths.append(path_psd)

        # Transmission loss TL(f) = 10 log10(P_src / P_rec)
        # Align on common frequency grid (they should match)
        TL = 10.0 * np.log10(np.maximum(Pxx_src, 1e-18) / np.maximum(Pxx_rec, 1e-18))
        path_tl = run_path / "attenuation_profiles" / f"tl_pos_{i}.csv"
        save_csv(path_tl, pd.DataFrame({
            "frequency_hz": f_src_psd,
            "TL_dB": TL,
        }))
        tl_paths.append(path_tl)

    # SNR at furthest sensor for manifest
    snr_db = compute_snr(f_rec_psd, Pxx_rec, band)

    # Turbulence and shear metrics
    # Convert superficial to layer velocities using area fractions
    area_frac_g = alpha
    area_frac_l = 1.0 - alpha
    U_g_layer = U_SG / max(area_frac_g, 1e-3)
    U_l_layer = U_SL / max(area_frac_l, 1e-3)

    vel_L, tkeeps_L, tau_w_L = compute_turbulence_profiles(
        rho=fluids.rho_l_kg_per_m3,
        mu=fluids.mu_l_pa_s,
        D=geometry.pipe_inner_diameter_m,
        U_bulk=U_l_layer,
    )
    vel_G, tkeeps_G, tau_w_G = compute_turbulence_profiles(
        rho=fluids.rho_g_kg_per_m3,
        mu=fluids.mu_g_pa_s,
        D=geometry.pipe_inner_diameter_m,
        U_bulk=U_g_layer,
    )
    # Interfacial shear (very rough): proportional to relative velocity
    rho_avg = 0.5 * (fluids.rho_g_kg_per_m3 + fluids.rho_l_kg_per_m3)
    tau_interface = 0.01 * rho_avg * (U_g_layer - U_l_layer) ** 2

    save_csv(run_path / "turbulence" / "velocity_profile_L.csv", vel_L)
    save_csv(run_path / "turbulence" / "tke_epsilon_L.csv", tkeeps_L)

    save_csv(run_path / "turbulence" / "velocity_profile_G.csv", vel_G)
    save_csv(run_path / "turbulence" / "tke_epsilon_G.csv", tkeeps_G)

    save_csv(run_path / "turbulence" / "shear_stress.csv", pd.DataFrame([
        {
            "tau_wall_L_Pa": tau_w_L,
            "tau_wall_G_Pa": tau_w_G,
            "tau_interface_Pa": tau_interface,
        }
    ]))

    # Instrumentation table
    save_csv(run_path / "instrumentation.csv", pd.DataFrame([
        {
            "sensor": "hydrophone",
            "count": instr.hydrophones,
            "sensitivity_V_per_Pa": instr.hydrophone_sensitivity_v_per_pa,
        },
        {"sensor": "pressure_transmitter", "count": instr.pressure_transmitters, "sensitivity_V_per_Pa": 5e-3},
        {"sensor": "conductance_probe", "count": instr.conductance_probes, "sensitivity_V_per_Pa": None},
        {"sensor": "impedance_sensor", "count": instr.impedance_sensors, "sensitivity_V_per_Pa": None},
        {"sensor": "thermocouple", "count": instr.thermocouples, "sensitivity_V_per_Pa": None},
    ]))

    # Return summary row for manifest
    return {
        "run_id": run_id,
        "U_SG_m_per_s": U_SG,
        "U_SL_m_per_s": U_SL,
        "alpha": alpha,
        "flow_pattern": pattern,
        "interface_height_m": h,
        "wave_amplitude_m": A_w,
        "snr_db_at_last_sensor": snr_db,
        "temperature_c": fluids.temperature_c,
        "pressure_bar": fluids.pressure_bar,
    }


def build_param_grid(num_runs: int, base_seed: int) -> List[Tuple[float, float, float, float, float, float]]:
    """Return a list of tuples (U_SG, U_SL, rho_g, rho_l, mu_g, mu_l, T, P)
    by sampling plausible stratified-flow operating points.
    """
    rng = np.random.default_rng(base_seed + 12345)
    params = []

    # Base fluid properties around air-water; small variations with T and P
    for i in range(num_runs):
        temperature_c = float(rng.uniform(15.0, 30.0))
        pressure_bar = float(rng.uniform(1.0, 3.0))

        # Approximate property adjustments with temperature
        rho_l = 998.0 - 0.3 * (temperature_c - 20.0)
        mu_l = 1.002e-3 * math.exp(-0.033 * (temperature_c - 20.0))
        rho_g = 1.2 * (pressure_bar / 1.0) * (293.15 / (273.15 + temperature_c))
        mu_g = 1.8e-5 * math.pow((273.15 + temperature_c) / 293.15, 0.7)

        # Stratified regime superficial velocities
        U_SL = float(rng.uniform(0.05, 0.5))  # m/s
        U_SG = float(rng.uniform(2.0, 12.0))  # m/s

        params.append((U_SG, U_SL, rho_g, rho_l, mu_g, mu_l, temperature_c, pressure_bar))

    return params


def generate_dataset(
    out_root: Path,
    num_runs: int,
    base_seed: int,
    geometry: Geometry,
    daq: DAQ,
) -> None:
    ensure_dir(out_root)
    manifest_rows: List[Dict[str, str]] = []

    param_grid = build_param_grid(num_runs, base_seed)

    for idx, (U_SG, U_SL, rho_g, rho_l, mu_g, mu_l, T_c, P_bar) in enumerate(param_grid):
        fluids = FluidProperties(
            rho_g_kg_per_m3=rho_g,
            rho_l_kg_per_m3=rho_l,
            mu_g_pa_s=mu_g,
            mu_l_pa_s=mu_l,
            temperature_c=T_c,
            pressure_bar=P_bar,
        )
        manifest_rows.append(
            generate_run(
                out_dir=out_root,
                run_index=idx,
                base_seed=base_seed,
                geometry=geometry,
                daq=daq,
                fluids=fluids,
                U_SG=U_SG,
                U_SL=U_SL,
            )
        )

    save_csv(out_root / "dataset_manifest.csv", pd.DataFrame(manifest_rows))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stratified flow acoustics dataset")
    p.add_argument("--out-dir", type=str, default="data/stratified_flow_acoustics", help="Output directory")
    p.add_argument("--num-runs", type=int, default=12, help="Number of experimental runs to generate")
    p.add_argument("--sample-rate", type=int, default=10_000, help="DAQ sampling rate (Hz)")
    p.add_argument("--duration", type=float, default=2.0, help="Signal duration (s)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    geometry = Geometry()
    daq = DAQ(sampling_rate_hz=args.sample_rate, duration_s=args.duration)

    out_root = Path(args.out_dir)
    generate_dataset(
        out_root=out_root,
        num_runs=args.num_runs,
        base_seed=args.seed,
        geometry=geometry,
        daq=daq,
    )
    print(f"Dataset generated at: {out_root.resolve()}")


if __name__ == "__main__":
    main()
