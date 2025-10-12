#!/usr/bin/env python3
"""
Synthetic dataset generator for stratified flow attenuation study.
Generates CFD-like fields, acoustic signals, and mathematical model outputs.
"""
from __future__ import annotations
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------
# Configuration data structures
# ----------------------------

@dataclass
class Geometry:
    domain: str  # "2D" or "3D"
    length_m: float
    height_m: float
    width_m: float | None = None

@dataclass
class Flow:
    liquid_density: float
    gas_density: float
    liquid_viscosity: float
    gas_viscosity: float
    superficial_velocity_liquid: float
    superficial_velocity_gas: float
    gravity_m_s2: float

@dataclass
class Acoustics:
    frequency_hz: float
    amplitude_pa: float
    source_location: Tuple[float, float, float]
    duration_s: float
    sampling_rate_hz: float

@dataclass
class CaseConfig:
    case_id: str
    description: str
    geometry: Geometry
    flow: Flow
    acoustics: Acoustics
    turbulence_model: str  # "k-epsilon" | "k-omega-SST" | "LES"
    multiphase_model: str  # "VOF" | "Euler-Euler"

# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_npy(path: Path, array: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def save_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# Synthetic field generators
# ----------------------------

def generate_grid(nx: int, ny: int, nz: int | None, geom: Geometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    x = np.linspace(0.0, geom.length_m, nx)
    y = np.linspace(0.0, geom.height_m, ny)
    z = None
    if geom.domain == "3D" and nz is not None and geom.width_m is not None:
        z = np.linspace(0.0, geom.width_m, nz)
    return x, y, z


def turbulent_noise(shape: Tuple[int, ...], intensity: float, rng: np.random.Generator) -> np.ndarray:
    # Colored noise via 1/f^alpha shaping in Fourier domain (alpha ~ 5/3 for Kolmogorov)
    alpha = 5.0 / 3.0
    noise = rng.normal(0.0, 1.0, size=shape)
    noise_k = np.fft.fftn(noise)
    grid = [np.fft.fftfreq(n) for n in shape]
    k2 = 0
    for g in np.meshgrid(*grid, indexing='ij'):
        k2 = k2 + g**2
    k_mag = np.sqrt(k2) + 1e-6
    shaping = 1.0 / (k_mag**(alpha / 2.0))
    shaped = np.fft.ifftn(noise_k * shaping).real
    shaped = shaped / np.std(shaped)
    return intensity * shaped


def generate_velocity_pressure_fields(cfg: CaseConfig, nx: int = 128, ny: int = 64, nz: int | None = None, seed: int | None = None) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    geom = cfg.geometry
    x, y, z = generate_grid(nx, ny, nz, geom)

    if geom.domain == "2D":
        shape = (ny, nx)
        base_u = np.tile(np.linspace(cfg.flow.superficial_velocity_gas, cfg.flow.superficial_velocity_liquid, ny)[:, None], (1, nx))
        base_v = np.zeros(shape)
        u = base_u + turbulent_noise(shape, intensity=0.1 * np.max(base_u + 1e-6), rng=rng)
        v = base_v + turbulent_noise(shape, intensity=0.05 * np.max(base_u + 1e-6), rng=rng)
        pressure = (1.0 - y[:, None] / geom.height_m) * 1000.0
        pressure = pressure + turbulent_noise(shape, intensity=5.0, rng=rng)
        fields = {
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
            "p": pressure.astype(np.float32),
        }
    else:
        assert nz is not None
        shape = (ny, nx, nz)
        base_u = np.repeat(np.linspace(cfg.flow.superficial_velocity_gas, cfg.flow.superficial_velocity_liquid, ny)[:, None, None], nx, axis=1)
        base_u = np.repeat(base_u, nz, axis=2)
        base_v = np.zeros(shape)
        base_w = np.zeros(shape)
        u = base_u + turbulent_noise(shape, intensity=0.1 * float(np.max(base_u) + 1e-6), rng=rng)
        v = base_v + turbulent_noise(shape, intensity=0.05 * float(np.max(base_u) + 1e-6), rng=rng)
        w = base_w + turbulent_noise(shape, intensity=0.05 * float(np.max(base_u) + 1e-6), rng=rng)
        yy = np.linspace(0.0, geom.height_m, ny)[:, None, None]
        pressure = (1.0 - yy / geom.height_m) * 1000.0
        pressure = np.repeat(pressure, nx, axis=1)
        pressure = np.repeat(pressure, nz, axis=2)
        pressure = pressure + turbulent_noise(shape, intensity=5.0, rng=rng)
        fields = {
            "u": u.astype(np.float32),
            "v": v.astype(np.float32),
            "w": w.astype(np.float32),
            "p": pressure.astype(np.float32),
        }

    return {**fields, "x": x.astype(np.float32), "y": y.astype(np.float32), "z": None if z is None else np.asarray(z, dtype=np.float32)}


def generate_vof_contours(cfg: CaseConfig, shape: Tuple[int, ...], seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if len(shape) == 2:
        ny, nx = shape
        y = np.linspace(0.0, cfg.geometry.height_m, ny)[:, None]
        interface_y = 0.4 * cfg.geometry.height_m + 0.05 * cfg.geometry.height_m * np.sin(np.linspace(0, 2*np.pi, nx)[None, :])
        vof = (y > interface_y).astype(np.float32)  # 1 = gas, 0 = liquid
        vof = vof + 0.05 * rng.normal(size=(ny, nx))
        vof = np.clip(vof, 0.0, 1.0)
    else:
        ny, nx, nz = shape
        y = np.linspace(0.0, cfg.geometry.height_m, ny)[:, None, None]
        interface_y = 0.4 * cfg.geometry.height_m + 0.05 * cfg.geometry.height_m * np.sin(
            np.linspace(0, 2*np.pi, nx)[None, :, None])
        interface_y = interface_y + 0.05 * cfg.geometry.height_m * np.sin(np.linspace(0, 4*np.pi, nz)[None, None, :])
        vof = (y > interface_y).astype(np.float32)
        vof = vof + 0.05 * rng.normal(size=(ny, nx, nz))
        vof = np.clip(vof, 0.0, 1.0)
    return vof.astype(np.float32)


def generate_turbulence_quantities(cfg: CaseConfig, velocity_fields: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # Simple surrogates: k ~ 1/2 * (u'²+v'²+w'²), epsilon ~ C_mu^(3/4) * k^(3/2) / l
    C_mu = 0.09
    length_scale = 0.1 * cfg.geometry.height_m
    u = velocity_fields["u"]
    v = velocity_fields.get("v")
    w = velocity_fields.get("w")

    def variance(a: np.ndarray) -> np.ndarray:
        return (a - np.mean(a)) ** 2

    if w is None:
        k = 0.5 * (variance(u) + variance(v))
    else:
        k = 0.5 * (variance(u) + variance(v) + variance(w))
    epsilon = (C_mu ** 0.75) * (np.maximum(k, 1e-6) ** 1.5) / max(length_scale, 1e-6)
    nu_t = C_mu * np.maximum(k, 0.0) ** 2 / np.maximum(epsilon, 1e-6)
    return {"k": k.astype(np.float32), "epsilon": epsilon.astype(np.float32), "nu_t": nu_t.astype(np.float32)}


# ----------------------------
# Acoustic propagation and math model
# ----------------------------

def compute_effective_sound_speed(cfg: CaseConfig) -> float:
    # Wood's mixture formula as a surrogate: 1/(rho*c^2) = sum(alpha_i/(rho_i*c_i^2))
    # Estimate phase fractions from average VOF later, but here approximate by density ratio
    rho_l = cfg.flow.liquid_density
    rho_g = cfg.flow.gas_density
    c_l = 1480.0  # m/s water-like
    c_g = 340.0   # m/s air-like
    alpha_g = 0.4
    alpha_l = 1.0 - alpha_g
    inv_bulk = alpha_l / (rho_l * c_l**2) + alpha_g / (rho_g * c_g**2)
    c_eff = math.sqrt(1.0 / ((rho_l * alpha_l + rho_g * alpha_g) * inv_bulk))
    return float(c_eff)


def compute_attenuation_coefficient(cfg: CaseConfig) -> float:
    # Simple frequency-dependent attenuation due to scattering and viscosity
    f = cfg.acoustics.frequency_hz
    mu_l = cfg.flow.liquid_viscosity
    mu_g = cfg.flow.gas_viscosity
    rho_mix = 0.6 * cfg.flow.liquid_density + 0.4 * cfg.flow.gas_density
    nu_mix = (0.6 * mu_l + 0.4 * mu_g) / max(rho_mix, 1e-6)
    boundary_layer_thickness = math.sqrt(2 * nu_mix / (2 * math.pi * f + 1e-6))
    scatter_term = 0.5 * (boundary_layer_thickness / max(cfg.geometry.height_m, 1e-6))
    visc_term = 2e-6 * f
    return float(scatter_term + visc_term)


def simulate_acoustic_waveform(cfg: CaseConfig, distance_m: float, n_receivers: int = 3) -> Dict[str, np.ndarray]:
    sr = int(cfg.acoustics.sampling_rate_hz)
    t = np.arange(0, cfg.acoustics.duration_s, 1.0 / sr)
    f = cfg.acoustics.frequency_hz
    c_eff = compute_effective_sound_speed(cfg)
    alpha = compute_attenuation_coefficient(cfg)

    signals = []
    for i in range(n_receivers):
        d = distance_m * (1 + 0.2 * i)
        tau = d / c_eff
        amplitude = cfg.acoustics.amplitude_pa * math.exp(-alpha * d) / max(d, 1e-6)
        phase = 2 * math.pi * f * (t - tau)
        signal = amplitude * np.sin(phase)
        # Add flow-induced noise and dispersion-like envelope
        envelope = np.exp(-((t - tau) ** 2) / (2 * (0.002 + 0.0005 * i) ** 2))
        signal = signal * envelope
        noise = 0.02 * cfg.acoustics.amplitude_pa * np.random.default_rng(1234 + i).normal(size=t.shape)
        signals.append((signal + noise).astype(np.float32))

    return {"time": t.astype(np.float32), "signals": np.stack(signals, axis=0)}


# ----------------------------
# Validation data fabrication
# ----------------------------

def fabricate_experimental_waveform(sim: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    time = sim["time"]
    signals = sim["signals"]
    rng = np.random.default_rng(42)
    exp = signals * (1.0 + 0.05 * rng.normal(size=signals.shape))
    exp = exp + 0.01 * np.max(np.abs(signals)) * rng.normal(size=signals.shape)
    return {"time": time.astype(np.float32), "signals": exp.astype(np.float32)}


# ----------------------------
# Additional mathematical outputs
# ----------------------------

def transfer_matrix_reflection_transmission(cfg: CaseConfig, frequencies_hz: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute normal-incidence reflection and transmission at a single gas-liquid interface.
    Uses plane-wave impedances Z = rho*c.
    Returns amplitude coefficients over the provided frequency array.
    """
    rho_l = cfg.flow.liquid_density
    rho_g = cfg.flow.gas_density
    c_l = 1480.0
    c_g = 340.0
    Z_l = rho_l * c_l
    Z_g = rho_g * c_g
    # Reflection (from liquid to gas) and transmission amplitude at interface
    R = (Z_g - Z_l) / (Z_g + Z_l)
    T = 2 * Z_g / (Z_g + Z_l)
    # Frequency independence at normal incidence in this simplified model
    R_arr = np.full_like(frequencies_hz, fill_value=R, dtype=np.float64)
    T_arr = np.full_like(frequencies_hz, fill_value=T, dtype=np.float64)
    return {"frequencies_hz": frequencies_hz.astype(np.float32), "R": R_arr.astype(np.float32), "T": T_arr.astype(np.float32)}


def estimate_time_delay_T0(cfg: CaseConfig) -> float:
    """Baseline time of flight across the domain centerline as a proxy for T0."""
    c_eff = compute_effective_sound_speed(cfg)
    return float(cfg.geometry.length_m / max(c_eff, 1e-6))


def derive_validation_metrics(distances_m: np.ndarray, time: np.ndarray, sim_signals: np.ndarray, exp_signals: np.ndarray) -> Dict[str, float]:
    """Derive effective sound speed and attenuation from signals.

    - c_eff_exp: from cross-correlation delay between adjacent receivers.
    - alpha_exp: from slope of ln(A*d) vs distance (A = RMS amplitude).
    """
    # Cross-correlation based c_eff
    c_estimates: List[float] = []
    for i in range(1, sim_signals.shape[0]):
        s0 = exp_signals[i - 1]
        s1 = exp_signals[i]
        corr = np.correlate(s1 - np.mean(s1), s0 - np.mean(s0), mode="full")
        lag = np.argmax(corr) - (len(s0) - 1)
        dt = time[1] - time[0]
        delta_t = abs(lag * dt)
        dd = distances_m[i] - distances_m[i - 1]
        if delta_t > 0 and dd > 0:
            c_estimates.append(dd / delta_t)
    c_eff_exp = float(np.median(c_estimates)) if c_estimates else float("nan")

    # Attenuation from amplitude decay (1/d * exp(-alpha d))
    A = np.sqrt(np.mean(exp_signals**2, axis=1))  # RMS per receiver
    # avoid log(0)
    eps = 1e-9
    y = np.log((A + eps) * distances_m)
    X = np.vstack([np.ones_like(distances_m), distances_m]).T
    # Least squares fit: y = b0 + b1 * d, alpha = -b1
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha_exp = float(-b[1])

    return {"c_eff_exp": c_eff_exp, "alpha_exp": alpha_exp}


# ----------------------------
# Main generation workflow
# ----------------------------

def default_cases() -> List[CaseConfig]:
    cases: List[CaseConfig] = []
    for case_idx, (turb, multi, domain) in enumerate([
        ("k-epsilon", "VOF", "2D"),
        ("k-omega-SST", "VOF", "2D"),
        ("LES", "Euler-Euler", "2D"),
        ("LES", "VOF", "3D"),
    ], start=1):
        if domain == "2D":
            geom = Geometry(domain="2D", length_m=0.5, height_m=0.1)
        else:
            geom = Geometry(domain="3D", length_m=0.3, height_m=0.1, width_m=0.1)
        flow = Flow(
            liquid_density=998.0,
            gas_density=1.2,
            liquid_viscosity=1.0e-3,
            gas_viscosity=1.8e-5,
            superficial_velocity_liquid=0.5,
            superficial_velocity_gas=2.0,
            gravity_m_s2=9.81,
        )
        acoustics = Acoustics(
            frequency_hz=4000.0 if case_idx < 3 else 12000.0,
            amplitude_pa=100.0,
            source_location=(0.0, 0.05, 0.0),
            duration_s=0.05,
            sampling_rate_hz=200000.0,
        )
        cases.append(CaseConfig(
            case_id=f"case_{case_idx:02d}",
            description=f"Stratified flow surrogate {domain} with {turb} and {multi}",
            geometry=geom,
            flow=flow,
            acoustics=acoustics,
            turbulence_model=turb,
            multiphase_model=multi,
        ))
    return cases


def generate_case(cfg: CaseConfig, out_root: Path) -> Dict[str, float]:
    # Grid and velocity/pressure fields
    if cfg.geometry.domain == "2D":
        fields = generate_velocity_pressure_fields(cfg, nx=192, ny=96, seed=123)
        vof = generate_vof_contours(cfg, fields["u"].shape, seed=456)
    else:
        fields = generate_velocity_pressure_fields(cfg, nx=96, ny=64, nz=48, seed=123)
        vof = generate_vof_contours(cfg, fields["u"].shape, seed=456)

    turb = generate_turbulence_quantities(cfg, fields)

    # Acoustic simulation (3 receivers at increasing distances)
    base_distance = 0.2
    receiver_distances = base_distance * (1 + 0.2 * np.arange(3))
    sim = simulate_acoustic_waveform(cfg, distance_m=base_distance)
    exp = fabricate_experimental_waveform(sim)

    # Math outputs
    c_eff = compute_effective_sound_speed(cfg)
    alpha = compute_attenuation_coefficient(cfg)
    T0 = estimate_time_delay_T0(cfg)
    freqs = np.linspace(0.5 * cfg.acoustics.frequency_hz, 1.5 * cfg.acoustics.frequency_hz, 25)
    rt = transfer_matrix_reflection_transmission(cfg, freqs)

    # Save
    case_dir = out_root / cfg.case_id
    ensure_dir(case_dir)

    # CFD outputs
    cfd_dir = case_dir / "CFD"
    ensure_dir(cfd_dir)
    for name in ["u", "v", "w", "p", "x", "y", "z"]:
        if name in fields and fields[name] is not None:
            save_npy(cfd_dir / f"{name}.npy", fields[name])
    save_npy(cfd_dir / "vof.npy", vof)
    for name, arr in turb.items():
        save_npy(cfd_dir / f"{name}.npy", arr)

    # Mathematical outputs
    math_dir = case_dir / "Mathematical"
    ensure_dir(math_dir)
    save_json(math_dir / "predictions.json", {
        "effective_sound_speed_m_s": c_eff,
        "attenuation_coefficient_np_per_m": alpha,
        "T0_s": T0,
    })
    save_json(math_dir / "wave_propagation.json", {
        "frequencies_hz": rt["frequencies_hz"].tolist(),
        "reflection_amplitude": rt["R"].tolist(),
        "transmission_amplitude": rt["T"].tolist(),
    })

    # Acoustic signals
    save_npy(math_dir / "acoustic_time.npy", sim["time"])
    save_npy(math_dir / "acoustic_signals_sim.npy", sim["signals"])

    # Validation
    val_dir = case_dir / "Validation"
    ensure_dir(val_dir)
    save_npy(val_dir / "acoustic_time.npy", exp["time"])
    save_npy(val_dir / "acoustic_signals_exp.npy", exp["signals"])
    metrics = derive_validation_metrics(receiver_distances, sim["time"], sim["signals"], exp["signals"])
    save_json(val_dir / "comparison.json", {
        "c_eff_model_m_s": c_eff,
        "alpha_model_np_per_m": alpha,
        "c_eff_exp_m_s": metrics["c_eff_exp"],
        "alpha_exp_np_per_m": metrics["alpha_exp"],
        "receiver_distances_m": receiver_distances.tolist(),
    })

    # Summary for manifest
    return {
        "c_eff": c_eff,
        "alpha": alpha,
        "num_grid_points": int(np.prod(fields["u"].shape)),
        "duration_s": float(cfg.acoustics.duration_s),
        "sampling_rate_hz": float(cfg.acoustics.sampling_rate_hz),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "datasets" / "stratified_flow_simulation"
    ensure_dir(root)
    cases = default_cases()

    manifest: Dict[str, Dict] = {"cases": []}
    for cfg in cases:
        summary = generate_case(cfg, root)
        entry = asdict(cfg)
        entry["summary"] = summary
        manifest["cases"].append(entry)

    save_json(root / "metadata" / "manifest.json", manifest)
    print(f"Generated {len(cases)} cases at {root}")


if __name__ == "__main__":
    main()
