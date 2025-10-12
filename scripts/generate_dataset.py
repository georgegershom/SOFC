#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional CoolProp imports for fluid properties
try:
    from CoolProp.CoolProp import PropsSI  # type: ignore
    HAVE_COOLPROP = True
except Exception:
    HAVE_COOLPROP = False


def _welch_psd(x: np.ndarray, fs: int, nperseg: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    nperseg = min(nperseg, n)
    if nperseg <= 1:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.fft.rfft(x)
        psd = (np.abs(spec) ** 2) / (fs * max(n, 1))
        return freqs.astype(np.float32), psd.astype(np.float32)
    noverlap = nperseg // 2
    step = max(1, nperseg - noverlap)
    window = np.hanning(nperseg)
    scale = np.sum(window ** 2)
    acc = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = x[start:start + nperseg]
        segw = seg * window
        spec = np.fft.rfft(segw)
        psd = (np.abs(spec) ** 2) / (fs * max(scale, 1e-12))
        if acc is None:
            acc = psd
        else:
            acc += psd
        count += 1
    if count == 0:
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = np.fft.rfft(x * np.hanning(n))
        psd = (np.abs(spec) ** 2) / (fs * np.sum(np.hanning(n) ** 2))
        return freqs.astype(np.float32), psd.astype(np.float32)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    Pxx = acc / count
    return freqs.astype(np.float32), Pxx.astype(np.float32)


def _chirp(t: np.ndarray, f0: float, f1: float, t1: float) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    beta = math.log(max(f1, 1e-6) / max(f0, 1e-6)) / max(t1, 1e-9)
    # instantaneous phase for logarithmic frequency sweep
    phase = 2.0 * math.pi * f0 * (np.expm1(beta * t) / max(beta, 1e-12))
    return np.sin(phase)


@dataclass
class PipeConfig:
    diameter_m: float = 0.05
    length_m: float = 10.0


@dataclass
class AcousticsConfig:
    sample_rate_hz: int = 10000
    duration_s: float = 0.25
    n_sensors: int = 3
    sensor_positions_m: Tuple[float, float, float] = (1.0, 3.0, 5.0)
    base_source_rms_pa: float = 500.0
    min_snr_db: float = 10.0
    max_snr_db: float = 40.0
    fmin_hz: float = 200.0
    fmax_hz: float = 5000.0


@dataclass
class RunParams:
    run_id: int
    temperature_k: float
    pressure_pa: float
    u_sg: float
    u_sl: float
    flow_pattern: str
    alpha: float
    interface_height_m: float
    wave_amplitude_m: float
    wave_frequency_hz: float
    rho_g: float
    rho_l: float
    mu_g: float
    mu_l: float
    c_g: float
    c_l: float
    c_eff: float
    k_g: float
    k_l: float
    epsilon_g: float
    epsilon_l: float
    tau_w_g: float
    tau_w_l: float
    tau_interface: float


R_AIR = 287.058  # J/(kg·K)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sample_operating_conditions(rng: np.random.Generator) -> Tuple[float, float]:
    # Temperature 15-40 C, Pressure 1-3 bar abs
    T_k = rng.uniform(288.15, 313.15)
    P_pa = rng.uniform(1.0e5, 3.0e5)
    return T_k, P_pa


def compute_fluid_properties(T_k: float, P_pa: float) -> Tuple[float, float, float, float, float, float]:
    # Returns rho_g, rho_l, mu_g, mu_l, c_g, c_l
    if HAVE_COOLPROP:
        try:
            rho_g = PropsSI("D", "T", T_k, "P", P_pa, "Air")
            mu_g = PropsSI("V", "T", T_k, "P", P_pa, "Air")
            c_g = PropsSI("A", "T", T_k, "P", P_pa, "Air")
            rho_l = PropsSI("D", "T", T_k, "P", P_pa, "Water")
            mu_l = PropsSI("V", "T", T_k, "P", P_pa, "Water")
            c_l = PropsSI("A", "T", T_k, "P", P_pa, "Water")
            return float(rho_g), float(rho_l), float(mu_g), float(mu_l), float(c_g), float(c_l)
        except Exception:
            pass
    # Fallback simple models
    T_c = T_k - 273.15
    rho_g = P_pa / (R_AIR * T_k)
    # Sutherland's law for air viscosity
    C1 = 1.458e-6
    S = 110.4
    mu_g = C1 * T_k ** 1.5 / (T_k + S)
    # Water density and viscosity approx near ambient
    rho_l = 1000.0 - 0.3 * (T_c - 20.0)
    mu_l = 1e-3 * math.exp(-0.033 * (T_c - 20.0))
    c_g = 331.0 + 0.6 * T_c
    c_l = 1480.0 - 3.0 * (T_c - 20.0)
    return float(rho_g), float(rho_l), float(mu_g), float(mu_l), float(c_g), float(c_l)


def drift_flux_alpha(u_sg: float, u_sl: float) -> float:
    j_g = max(u_sg, 1e-6)
    j_l = max(u_sl, 1e-6)
    j = j_g + j_l
    C0 = 1.2  # distribution parameter typical for stratified
    Vgj = 0.0  # drift velocity ~ 0 for large bubbles / stratified interface
    alpha = max(0.01, min(0.99, C0 * j_g / j + Vgj / j))
    return alpha


def interface_height_from_alpha(alpha: float, D: float) -> float:
    # For a horizontal circular pipe, area fraction equals gas holdup alpha.
    # Find liquid level height h that gives area_g/A_total = alpha.
    # Use bisection on h in [0, D]
    A_total = math.pi * (D ** 2) / 4.0

    def area_gas(h: float) -> float:
        # h is liquid height from bottom; gas occupies above
        if h <= 0.0:
            return A_total
        if h >= D:
            return 0.0
        R = D / 2.0
        y = h - R  # interface height relative to center
        theta = 2.0 * math.acos(max(-1.0, min(1.0, y / R)))  # central angle of liquid segment
        A_liq = 0.5 * (R ** 2) * (theta - math.sin(theta))
        return A_total - A_liq

    lo, hi = 0.0, D
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        a = area_gas(mid) / A_total
        if a > alpha:
            # need more liquid (higher h) to reduce gas area
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def friction_factor_blasius(Re: float) -> float:
    if Re < 1e-8:
        return 0.0
    if Re < 2300.0:
        return 64.0 / max(Re, 1e-6)
    return 0.3164 / (Re ** 0.25)


def approximate_phase_hydraulic_diameter(alpha: float, D: float) -> Tuple[float, float]:
    # crude but stable approximation
    Dg = max(1e-4, D * math.sqrt(alpha))
    Dl = max(1e-4, D * math.sqrt(1.0 - alpha))
    return Dg, Dl


def compute_turbulence_and_shear(u: float, rho: float, mu: float, D_h: float) -> Tuple[float, float, float]:
    Re = rho * max(u, 1e-6) * D_h / max(mu, 1e-9)
    f = friction_factor_blasius(Re)
    tau_w = 0.5 * f * rho * u ** 2
    u_tau = math.sqrt(max(tau_w, 0.0) / max(rho, 1e-12))
    # Empirical scaling
    k = 1.5 * (0.2 * u) ** 2  # assuming turbulence intensity ~20%
    epsilon = (u_tau ** 3) / max(0.41 * 0.05 * D_h, 1e-6)  # using kappa*yp ~ 0.05D
    return k, epsilon, tau_w


def wood_mixture_speed_of_sound(alpha: float, rho_g: float, rho_l: float, c_g: float, c_l: float) -> float:
    denom = alpha / (rho_g * c_g ** 2) + (1.0 - alpha) / (rho_l * c_l ** 2)
    if denom <= 0.0:
        return min(c_g, c_l)
    rho_mix = alpha * rho_g + (1.0 - alpha) * rho_l
    c_eff = math.sqrt(max(1e-9, 1.0 / (rho_mix * denom)))
    return c_eff


def attenuation_coefficient_profile(
    freqs: np.ndarray,
    alpha: float,
    rho_g: float,
    rho_l: float,
    mu_g: float,
    mu_l: float,
    c_eff: float,
    D: float,
    wave_amp: float,
    u_sg: float,
    u_sl: float,
    flow_pattern: str,
) -> np.ndarray:
    # Simplified model capturing trends: increases with f, viscosity, void fraction; reduced by diameter.
    f0 = 1000.0
    mu_mix = alpha * mu_g + (1.0 - alpha) * mu_l
    rho_mix = alpha * rho_g + (1.0 - alpha) * rho_l
    slip = max(0.1, min(10.0, u_sg / max(u_sl, 1e-3)))
    base = 2.0 * np.sqrt(np.maximum(freqs, 1.0) / f0) * (mu_mix / (rho_mix * max(c_eff, 1.0) ** 2 * max(D, 1e-6) ** 2))
    tp = 0.25 * alpha * (1.0 - alpha) * (freqs / f0) * (1.0 + 0.2 * (slip - 1.0))
    iface = 0.0
    if flow_pattern == "wavy":
        iface = 0.15 * (wave_amp / max(D, 1e-6)) * (freqs / f0)
    gamma = base + tp + iface
    return gamma


def synthesize_source_signal(cfg: AcousticsConfig, rng: np.random.Generator) -> np.ndarray:
    n = int(cfg.sample_rate_hz * cfg.duration_s)
    t = np.arange(n) / cfg.sample_rate_hz
    # Sum of random tones + chirp for broadband
    n_tones = rng.integers(5, 10)
    phases = rng.uniform(0.0, 2.0 * math.pi, size=n_tones)
    freqs = rng.uniform(cfg.fmin_hz, cfg.fmax_hz, size=n_tones)
    amps = rng.uniform(0.5, 1.0, size=n_tones)
    sig = np.zeros_like(t)
    for a, f, ph in zip(amps, freqs, phases):
        sig += a * np.sin(2.0 * math.pi * f * t + ph)
    sig += 0.8 * _chirp(t, f0=cfg.fmin_hz, f1=cfg.fmax_hz, t1=cfg.duration_s)
    # Normalize to target RMS
    rms = np.sqrt(np.mean(sig ** 2)) + 1e-12
    sig = sig * (cfg.base_source_rms_pa / rms)
    return sig.astype(np.float32)


def apply_attenuation_and_noise(
    source: np.ndarray,
    distance_m: float,
    gamma_profile: np.ndarray,
    freqs: np.ndarray,
    cfg: AcousticsConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    # Approximate as frequency-dependent filtering via magnitude response exp(-gamma x)
    n = len(source)
    # FFT-based filtering
    spec = np.fft.rfft(source)
    # Build magnitude response on FFT bins
    fft_freqs = np.fft.rfftfreq(n, d=1.0 / cfg.sample_rate_hz)
    # Interpolate gamma at FFT freqs
    gamma_interp = np.interp(fft_freqs, freqs, gamma_profile, left=gamma_profile[0], right=gamma_profile[-1])
    mag = np.exp(-gamma_interp * distance_m)
    spec_filtered = spec * mag
    received = np.fft.irfft(spec_filtered, n=n).astype(np.float32)
    # Additive white noise based on random SNR target
    target_snr_db = float(rng.uniform(cfg.min_snr_db, cfg.max_snr_db))
    sig_pow = float(np.mean(received ** 2)) + 1e-12
    noise_pow = sig_pow / (10.0 ** (target_snr_db / 10.0))
    noise = rng.normal(scale=math.sqrt(noise_pow), size=n).astype(np.float32)
    noisy = received + noise
    return noisy, target_snr_db


def compute_psd(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    return _welch_psd(x, fs=fs, nperseg=min(1024, len(x)))


def velocity_profile_loglaw(n_points: int, D: float, u_bulk: float, rho: float, mu: float) -> Tuple[np.ndarray, np.ndarray]:
    # Very rough: use log-law scaled by friction velocity to match bulk velocity
    # y from wall to center for each phase thickness fraction; here we apply over radius
    R = D / 2.0
    y = np.linspace(1e-4 * R, R, n_points, dtype=np.float64)
    # Iteratively find u_tau such that mean ~ u_bulk
    kappa = 0.41
    B = 5.2
    def mean_velocity(u_tau: float) -> float:
        nu = mu / max(rho, 1e-12)
        y_plus = y * u_tau / max(nu, 1e-12)
        u = (u_tau / kappa) * np.log(np.maximum(1.0, y_plus)) + B * u_tau
        return float(np.trapz(u, y) / (R - y[0]))
    u_tau = max(1e-3, 0.05 * u_bulk)
    for _ in range(20):
        m = mean_velocity(u_tau)
        if m <= 0:
            break
        u_tau *= (u_bulk / m)
    nu = mu / max(rho, 1e-12)
    y_plus = y * u_tau / max(nu, 1e-12)
    u_profile = (u_tau / kappa) * np.log(np.maximum(1.0, y_plus)) + B * u_tau
    # Mirror to full diameter coordinate
    y_full = np.linspace(0.0, D, n_points, dtype=np.float64)
    # Simple mapping: assign near-wall values symmetrically
    u_full = np.interp(y_full, np.concatenate([np.array([0.0]), y, np.array([D])]),
                       np.concatenate([np.array([0.0]), u_profile, np.array([0.0])]))
    return y_full.astype(np.float32), u_full.astype(np.float32)


def generate_dataset(
    out_dir: str,
    n_runs: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    pipe = PipeConfig()
    acoust = AcousticsConfig()

    # Precompute frequencies for attenuation profile
    freqs = np.linspace(acoust.fmin_hz, acoust.fmax_hz, 128, dtype=np.float64)

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "acoustics"))
    ensure_dir(os.path.join(out_dir, "profiles", "velocity", "gas"))
    ensure_dir(os.path.join(out_dir, "profiles", "velocity", "liquid"))

    runs_summary: List[Dict] = []

    for run_id in range(1, n_runs + 1):
        T_k, P_pa = sample_operating_conditions(rng)

        # Sample superficial velocities (m/s)
        u_sg = float(rng.uniform(0.2, 8.0))
        u_sl = float(rng.uniform(0.1, 2.0))

        rho_g, rho_l, mu_g, mu_l, c_g, c_l = compute_fluid_properties(T_k, P_pa)

        alpha = drift_flux_alpha(u_sg, u_sl)
        h = interface_height_from_alpha(alpha, pipe.diameter_m)

        # Flow pattern classification (very rough)
        we_g = rho_g * u_sg ** 2 * pipe.diameter_m / max(mu_g * (1.0 / (rho_g + 1e-12)), 1e-12)
        fr_g = u_sg / math.sqrt((pipe.diameter_m) * 9.81)
        flow_pattern = "wavy" if (fr_g > 0.5 and alpha > 0.2) else "smooth"

        # Interface wave params for wavy regime
        if flow_pattern == "wavy":
            wave_amp = float(min(0.4 * h, rng.uniform(0.2e-3, 1.5e-3) * (1.0 + 5.0 * alpha)))
            wave_freq = float(rng.uniform(1.0, 20.0) * u_sg / max(pipe.diameter_m, 1e-6))
        else:
            wave_amp = 0.0
            wave_freq = 0.0

        c_eff = wood_mixture_speed_of_sound(alpha, rho_g, rho_l, c_g, c_l)

        Dg, Dl = approximate_phase_hydraulic_diameter(alpha, pipe.diameter_m)
        k_g, eps_g, tau_w_g = compute_turbulence_and_shear(u_sg, rho_g, mu_g, Dg)
        k_l, eps_l, tau_w_l = compute_turbulence_and_shear(u_sl, rho_l, mu_l, Dl)
        tau_interface = 0.01 * (alpha * rho_g + (1 - alpha) * rho_l) * (u_sg - u_sl) ** 2

        gamma_profile = attenuation_coefficient_profile(
            freqs=freqs,
            alpha=alpha,
            rho_g=rho_g,
            rho_l=rho_l,
            mu_g=mu_g,
            mu_l=mu_l,
            c_eff=c_eff,
            D=pipe.diameter_m,
            wave_amp=wave_amp,
            u_sg=u_sg,
            u_sl=u_sl,
            flow_pattern=flow_pattern,
        )

        # Synthesize acoustics
        source = synthesize_source_signal(acoust, rng)
        # Save source signal
        run_dir = os.path.join(out_dir, "acoustics", f"run_{run_id:04d}")
        ensure_dir(run_dir)
        t = np.arange(len(source)) / acoust.sample_rate_hz
        src_df = pd.DataFrame({"time_s": t, "pressure_pa": source})
        src_df.to_csv(os.path.join(run_dir, "source_signal.csv"), index=False)

        # Frequency-domain data for metrics
        src_f, src_psd = compute_psd(source, acoust.sample_rate_hz)
        pd.DataFrame({"frequency_hz": src_f, "psd": src_psd}).to_csv(
            os.path.join(run_dir, "source_psd.csv"), index=False
        )
        src_rms = float(np.sqrt(np.mean(source ** 2)))
        src_peak = float(np.max(np.abs(source)))

        # Received signals at sensors
        received_signals: List[np.ndarray] = []
        received_snrs: List[float] = []
        tl_profiles: List[np.ndarray] = []
        metrics_rows: List[Dict] = []
        metrics_rows.append({
            "signal": "source",
            "distance_m": 0.0,
            "rms_pa": src_rms,
            "peak_pa": src_peak,
            "snr_db": float("nan"),
        })

        for i, x in enumerate(acoust.sensor_positions_m, start=1):
            received, snr_db = apply_attenuation_and_noise(source, x, gamma_profile, freqs, acoust, rng)
            received_signals.append(received)
            received_snrs.append(snr_db)
            rec_f, rec_psd = compute_psd(received, acoust.sample_rate_hz)
            # Transmission loss per frequency: TL = 10*log10(Psrc/Prec)
            # Interpolate PSDs to common freqs (Welch returns same bins given same length)
            TL = 10.0 * np.log10((src_psd + 1e-15) / (rec_psd + 1e-15))
            tl_profiles.append(TL)
            df = pd.DataFrame({"time_s": t, "pressure_pa": received})
            df.to_csv(os.path.join(run_dir, f"received_sensor{i}.csv"), index=False)
            pd.DataFrame({"frequency_hz": rec_f, "psd": rec_psd}).to_csv(
                os.path.join(run_dir, f"received_psd_sensor{i}.csv"), index=False
            )
            rec_rms = float(np.sqrt(np.mean(received ** 2)))
            rec_peak = float(np.max(np.abs(received)))
            metrics_rows.append({
                "signal": f"sensor{i}",
                "distance_m": float(x),
                "rms_pa": rec_rms,
                "peak_pa": rec_peak,
                "snr_db": float(snr_db),
            })

        # Save attenuation profile used (per-meter gamma) and derived TL at sensor 1
        att_df = pd.DataFrame({
            "frequency_hz": freqs,
            "gamma_per_m": gamma_profile,
        })
        att_df.to_csv(os.path.join(run_dir, "attenuation_profile.csv"), index=False)

        tl_df = pd.DataFrame({
            "frequency_hz": rec_f,
            "TL_dB_sensor1": tl_profiles[0],
            "TL_dB_sensor2": tl_profiles[1],
            "TL_dB_sensor3": tl_profiles[2],
        })
        tl_df.to_csv(os.path.join(run_dir, "transmission_loss.csv"), index=False)

        # Save per-run metrics (amplitudes and SNRs)
        pd.DataFrame(metrics_rows).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

        # Velocity profiles (very approximate) for each phase
        y_g, u_g = velocity_profile_loglaw(64, pipe.diameter_m, max(u_sg, 0.01), rho_g, mu_g)
        y_l, u_l = velocity_profile_loglaw(64, pipe.diameter_m, max(u_sl, 0.01), rho_l, mu_l)
        pd.DataFrame({"y_m": y_g, "u_g_m_per_s": u_g}).to_csv(
            os.path.join(out_dir, "profiles", "velocity", "gas", f"run_{run_id:04d}.csv"), index=False
        )
        pd.DataFrame({"y_m": y_l, "u_l_m_per_s": u_l}).to_csv(
            os.path.join(out_dir, "profiles", "velocity", "liquid", f"run_{run_id:04d}.csv"), index=False
        )

        runs_summary.append(asdict(RunParams(
            run_id=run_id,
            temperature_k=T_k,
            pressure_pa=P_pa,
            u_sg=u_sg,
            u_sl=u_sl,
            flow_pattern=flow_pattern,
            alpha=alpha,
            interface_height_m=h,
            wave_amplitude_m=wave_amp,
            wave_frequency_hz=wave_freq,
            rho_g=rho_g,
            rho_l=rho_l,
            mu_g=mu_g,
            mu_l=mu_l,
            c_g=c_g,
            c_l=c_l,
            c_eff=c_eff,
            k_g=k_g,
            k_l=k_l,
            epsilon_g=eps_g,
            epsilon_l=eps_l,
            tau_w_g=tau_w_g,
            tau_w_l=tau_w_l,
            tau_interface=tau_interface,
        )))

    # Save runs summary
    pd.DataFrame(runs_summary).to_csv(os.path.join(out_dir, "runs.csv"), index=False)

    # Metadata
    meta = {
        "description": "Synthetic dataset for stratified flow acoustic attenuation studies",
        "n_runs": n_runs,
        "pipe": asdict(pipe),
        "acoustics": asdict(acoust),
        "frequencies_hz": freqs.tolist(),
        "notes": "Fluid properties by CoolProp when available; otherwise empirical fallbacks.",
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate stratified flow acoustic attenuation dataset")
    p.add_argument("--out", default="data/stratified_acoustic_attenuation", help="Output directory")
    p.add_argument("--runs", type=int, default=150, help="Number of runs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_dir(args.out)
    generate_dataset(args.out, args.runs, args.seed)
    # Zip the dataset
    zip_path = args.out.rstrip("/") + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    shutil.make_archive(args.out, 'zip', args.out)
    print(f"Dataset generated at {args.out} and zipped at {zip_path}")
