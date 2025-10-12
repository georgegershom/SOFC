#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import csv
import time
import zipfile
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class PhysicalProperties:
    density_layer1: float  # kg/m^3
    density_layer2: float  # kg/m^3
    viscosity_layer1: float  # Pa*s
    viscosity_layer2: float  # Pa*s
    sound_speed_layer1: float  # m/s
    sound_speed_layer2: float  # m/s


@dataclass
class Domain2D:
    length_x: float
    length_y: float
    num_x: int
    num_y: int


@dataclass
class Domain3D:
    length_x: float
    length_y: float
    length_z: float
    num_x: int
    num_y: int
    num_z: int


@dataclass
class AcousticSetup:
    length_x: float
    num_x: int
    dt: float
    num_steps: int
    sensor_positions: List[float]  # positions in meters along x
    attenuation_base: float  # base attenuation factor per second


@dataclass
class TurbulenceModelConfig:
    name: str  # 'k-epsilon', 'k-omega-sst', 'les'
    c_mu: float
    kappa: float
    mixing_length_scale: float
    intensity_base: float


DEFAULT_TURBULENCE_MODELS: List[TurbulenceModelConfig] = [
    TurbulenceModelConfig(name="k-epsilon", c_mu=0.09, kappa=0.41, mixing_length_scale=0.07, intensity_base=0.05),
    TurbulenceModelConfig(name="k-omega-sst", c_mu=0.07, kappa=0.41, mixing_length_scale=0.05, intensity_base=0.04),
    TurbulenceModelConfig(name="les", c_mu=0.10, kappa=0.41, mixing_length_scale=0.10, intensity_base=0.03),
]


# ---------- Utility ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, **arrays) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)


def save_json(path: str, data: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def save_csv(path: str, header: List[str], rows: List[List[float]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


# ---------- VOF fields ----------

def generate_vof_2d(domain: Domain2D, interface_y_fraction: float, smoothing_length: float,
                     perturbation_amplitude: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, domain.length_x, domain.num_x)
    y = np.linspace(0.0, domain.length_y, domain.num_y)
    X, Y = np.meshgrid(x, y, indexing='xy')

    y0 = interface_y_fraction * domain.length_y
    mode_x = rng.integers(1, 4)
    interface = y0 + perturbation_amplitude * np.sin(2.0 * np.pi * mode_x * X / domain.length_x)

    # Smooth Heaviside to represent volume fraction of layer1 below interface
    vof = 1.0 / (1.0 + np.exp((Y - interface) / max(1e-6, smoothing_length)))
    return X, Y, vof


def generate_vof_3d(domain: Domain3D, interface_y_fraction: float, smoothing_length: float,
                     perturbation_amplitude: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, domain.length_x, domain.num_x)
    y = np.linspace(0.0, domain.length_y, domain.num_y)
    z = np.linspace(0.0, domain.length_z, domain.num_z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='xy')

    y0 = interface_y_fraction * domain.length_y
    mode_x = rng.integers(1, 4)
    mode_z = rng.integers(1, 4)
    interface = y0 \
        + perturbation_amplitude * np.sin(2.0 * np.pi * mode_x * X / domain.length_x) \
        + perturbation_amplitude * 0.8 * np.sin(2.0 * np.pi * mode_z * Z / domain.length_z)

    vof = 1.0 / (1.0 + np.exp((Y - interface) / max(1e-6, smoothing_length)))
    return X, Y, Z, vof


# ---------- Velocity and pressure ----------

def blend_by_vof(q1: np.ndarray, q2: np.ndarray, vof: np.ndarray) -> np.ndarray:
    return vof * q1 + (1.0 - vof) * q2


def generate_velocity_pressure_2d(X: np.ndarray, Y: np.ndarray, vof: np.ndarray,
                                   props: PhysicalProperties, g: float = 9.81,
                                   max_centerline_velocity_layer1: float = 0.4,
                                   max_centerline_velocity_layer2: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ly = Y.max() - Y.min()
    # Poiseuille-like base profile in each layer (along x), centered at each layer
    y_norm = Y / max(1e-9, Ly)

    # Layer centers (below and above interface ~0.5)
    center1 = 0.25
    center2 = 0.75

    u1 = max_centerline_velocity_layer1 * (1.0 - ((y_norm - center1) / 0.25) ** 2)
    u2 = max_centerline_velocity_layer2 * (1.0 - ((y_norm - center2) / 0.25) ** 2)
    u1 = np.clip(u1, 0.0, None)
    u2 = np.clip(u2, 0.0, None)

    # Gentle secondary flow driven by interface curvature
    dy = Y[1, 0] - Y[0, 0]
    dx = X[0, 1] - X[0, 0]
    grad_vof_y, grad_vof_x = np.gradient(vof, dy, dx)
    v_secondary = 0.02 * (-grad_vof_x)  # indicative cross-stream motion

    u = blend_by_vof(u1, u2, vof)
    v = v_secondary

    # Hydrostatic + dynamic pressure
    rho_mix = blend_by_vof(np.full_like(vof, props.density_layer1), np.full_like(vof, props.density_layer2), vof)
    p_hydro = rho_mix * g * (Ly - Y)
    p_dyn = 0.5 * rho_mix * u ** 2
    p = p_hydro + p_dyn

    return u, v, p


def generate_velocity_pressure_3d(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, vof: np.ndarray,
                                   props: PhysicalProperties, g: float = 9.81,
                                   max_centerline_velocity_layer1: float = 0.4,
                                   max_centerline_velocity_layer2: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Ly = Y.max() - Y.min()
    y_norm = Y / max(1e-9, Ly)
    center1 = 0.25
    center2 = 0.75

    u1 = max_centerline_velocity_layer1 * (1.0 - ((y_norm - center1) / 0.25) ** 2)
    u2 = max_centerline_velocity_layer2 * (1.0 - ((y_norm - center2) / 0.25) ** 2)
    u1 = np.clip(u1, 0.0, None)
    u2 = np.clip(u2, 0.0, None)

    # Add weak spanwise modulation to mimic 3D structures
    span_mod = 1.0 + 0.05 * np.sin(2.0 * np.pi * Z / (Z.max() - Z.min() + 1e-9))
    u = blend_by_vof(u1, u2, vof) * span_mod

    # Cross-stream vortical motion around interface
    dy = Y[1, 0, 0] - Y[0, 0, 0]
    dx = X[0, 1, 0] - X[0, 0, 0]
    dz = Z[0, 0, 1] - Z[0, 0, 0]
    grad_vof_y, grad_vof_x, grad_vof_z = np.gradient(vof, dy, dx, dz)
    v = 0.02 * (-grad_vof_x)
    w = 0.02 * (grad_vof_z)

    rho_mix = blend_by_vof(np.full_like(vof, props.density_layer1), np.full_like(vof, props.density_layer2), vof)
    p_hydro = rho_mix * g * (Ly - Y)
    p_dyn = 0.5 * rho_mix * u ** 2
    p = p_hydro + p_dyn

    return u, v, w, p


# ---------- Turbulence surrogate ----------

def compute_turbulence_2d(u: np.ndarray, v: np.ndarray, vof: np.ndarray, Y: np.ndarray,
                           props: PhysicalProperties, model: TurbulenceModelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dy = Y[1, 0] - Y[0, 0]
    dx = Y[0, 1] - Y[0, 0]  # approximate spacing in x using Y grid shape

    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)

    strain_magnitude = np.sqrt(2.0 * (du_dx ** 2 + dv_dy ** 2) + (du_dy + dv_dx) ** 2)

    # Local turbulent intensity surrogate
    velocity_magnitude = np.sqrt(u ** 2 + v ** 2)
    intensity = model.intensity_base + 0.02 * np.tanh(strain_magnitude)

    k = 1.5 * (intensity * (velocity_magnitude + 1e-6)) ** 2

    # Mixing length, damped near interface based on VOF gradient
    grad_vof_y, grad_vof_x = np.gradient(vof, dy, dx)
    interface_indicator = np.tanh(10.0 * np.sqrt(grad_vof_x ** 2 + grad_vof_y ** 2))
    distance_to_wall = np.minimum(Y, Y.max() - Y)
    mixing_length = model.mixing_length_scale * np.maximum(1e-6, distance_to_wall) * (1.0 - 0.3 * interface_indicator)

    epsilon = (model.c_mu ** 0.75) * (k ** 1.5) / np.maximum(1e-6, mixing_length)

    rho_mix = blend_by_vof(np.full_like(vof, props.density_layer1), np.full_like(vof, props.density_layer2), vof)
    mu_t = rho_mix * model.c_mu * (k ** 2) / np.maximum(1e-6, epsilon)
    return k, epsilon, mu_t


def compute_turbulence_3d(u: np.ndarray, v: np.ndarray, w: np.ndarray, vof: np.ndarray, Y: np.ndarray,
                           props: PhysicalProperties, model: TurbulenceModelConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dy = Y[1, 0, 0] - Y[0, 0, 0]
    # Approximate dx, dz using Y grid shape assumptions; safer to infer from array shapes via indices
    # We'll create dummy spacing values assuming consistent grid
    dx = dy
    dz = dy

    du_dy, du_dx, du_dz = np.gradient(u, dy, dx, dz)
    dv_dy, dv_dx, dv_dz = np.gradient(v, dy, dx, dz)
    dw_dy, dw_dx, dw_dz = np.gradient(w, dy, dx, dz)

    strain_magnitude = np.sqrt(
        2.0 * (du_dx ** 2 + dv_dy ** 2 + dw_dz ** 2)
        + (du_dy + dv_dx) ** 2
        + (du_dz + dw_dx) ** 2
        + (dv_dz + dw_dy) ** 2
    )

    velocity_magnitude = np.sqrt(u ** 2 + v ** 2 + w ** 2)
    intensity = model.intensity_base + 0.02 * np.tanh(strain_magnitude)
    k = 1.5 * (intensity * (velocity_magnitude + 1e-6)) ** 2

    grad_vof_y, grad_vof_x, grad_vof_z = np.gradient(vof, dy, dx, dz)
    interface_indicator = np.tanh(10.0 * np.sqrt(grad_vof_x ** 2 + grad_vof_y ** 2 + grad_vof_z ** 2))
    distance_to_wall = np.minimum(Y, Y.max() - Y)
    mixing_length = model.mixing_length_scale * np.maximum(1e-6, distance_to_wall) * (1.0 - 0.3 * interface_indicator)

    epsilon = (model.c_mu ** 0.75) * (k ** 1.5) / np.maximum(1e-6, mixing_length)

    rho_mix = blend_by_vof(np.full_like(vof, props.density_layer1), np.full_like(vof, props.density_layer2), vof)
    mu_t = rho_mix * model.c_mu * (k ** 2) / np.maximum(1e-6, epsilon)
    return k, epsilon, mu_t


# ---------- Acoustic simulation ----------

def simulate_acoustic_1d_layered(setup: AcousticSetup, props: PhysicalProperties,
                                 interface_x_fraction: float, seed: int,
                                 source_frequency: float = 1500.0,
                                 source_width: float = 0.002) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nx = setup.num_x
    dx = setup.length_x / (nx - 1)
    x = np.linspace(0.0, setup.length_x, nx)

    # Piecewise sound speed and density for two layers (interface perpendicular to propagation)
    x0 = interface_x_fraction * setup.length_x
    layer_mask = x[:, None] <= x0
    c = np.where(layer_mask, props.sound_speed_layer1, props.sound_speed_layer2).astype(np.float64)
    rho = np.where(layer_mask, props.density_layer1, props.density_layer2).astype(np.float64)

    # CFL-safe dt if not provided
    dt = setup.dt
    c_max = max(props.sound_speed_layer1, props.sound_speed_layer2)
    if dt * c_max / dx > 0.9:
        dt = 0.9 * dx / c_max

    nt = setup.num_steps

    # FDTD 1D wave eq with simple damping
    p_prev = np.zeros(nx, dtype=np.float64)
    p_curr = np.zeros(nx, dtype=np.float64)
    p_next = np.zeros(nx, dtype=np.float64)

    # Source at left boundary index 2 (avoid boundary)
    src_index = 2
    t_array = np.arange(nt) * dt

    # Gaussian-modulated sinusoid
    src = np.exp(-((t_array - 6.0 / source_frequency) ** 2) / (2.0 * source_width ** 2)) * np.sin(2.0 * np.pi * source_frequency * t_array)

    # Damping coefficient per time-step
    damping = np.exp(-setup.attenuation_base * dt)

    # Sensor indices
    sensor_indices = [max(1, min(nx - 2, int(pos / setup.length_x * (nx - 1)))) for pos in setup.sensor_positions]
    sensor_signals = {f"sensor_{i}_x{setup.sensor_positions[i]:.3f}": np.zeros(nt, dtype=np.float64) for i in range(len(sensor_indices))}

    # Courant numbers per cell
    courant = (c * dt / dx) ** 2

    # Boundary: simple absorbing (first-order)
    absorb_coeff = 0.98

    for n in range(nt):
        # Internal update
        p_next[1:-1] = (
            2.0 * p_curr[1:-1] - p_prev[1:-1]
            + courant[1:-1, 0] * (p_curr[2:] - 2.0 * p_curr[1:-1] + p_curr[:-2])
        )

        # Source term
        p_next[src_index] += src[n]

        # Absorbing boundaries
        p_next[0] = absorb_coeff * p_curr[1]
        p_next[-1] = absorb_coeff * p_curr[-2]

        # Damping
        p_next *= damping

        # Rotate time levels
        p_prev, p_curr, p_next = p_curr, p_next, p_prev

        # Record sensors
        for i, idx in enumerate(sensor_indices):
            key = f"sensor_{i}_x{setup.sensor_positions[i]:.3f}"
            sensor_signals[key][n] = p_curr[idx]

    result = {
        "time": t_array,
        "x": x,
    }
    result.update(sensor_signals)
    return result


# ---------- Mathematical models ----------

def effective_sound_speed_series(phi1: float, c1: float, c2: float) -> float:
    # Series (layers along propagation); harmonic average by slowness
    return 1.0 / (phi1 / c1 + (1.0 - phi1) / c2 + 1e-12)


def attenuation_model(freqs: np.ndarray, base_alpha: float, n: float = 1.2, corner: float = 2000.0) -> np.ndarray:
    # Simple power-law with high-frequency soft saturation
    return base_alpha * (freqs / corner) ** n / np.sqrt(1.0 + (freqs / (2.0 * corner)) ** 2)


def transfer_matrix_layer(k: complex, thickness: float, Z: float) -> np.ndarray:
    # Acoustic transfer matrix for a single layer at normal incidence
    cos_ = np.cos(k * thickness)
    j_sin = 1j * np.sin(k * thickness)
    return np.array([[cos_, j_sin * Z], [j_sin / Z, cos_]], dtype=complex)


def compute_transfer_response(freqs: np.ndarray, rho_list: List[float], c_list: List[float], thickness_list: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    # Compute reflection and transmission magnitude at normal incidence for a stack
    Z_list = [rho * c for rho, c in zip(rho_list, c_list)]
    R = np.zeros_like(freqs, dtype=float)
    T = np.zeros_like(freqs, dtype=float)
    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        M = np.eye(2, dtype=complex)
        for rho, c, d, Z in zip(rho_list, c_list, thickness_list, Z_list):
            k = omega / c
            M = M @ transfer_matrix_layer(k, d, Z)
        Z0 = Z_list[0]
        ZN = Z_list[-1]
        Zin = (M[0, 0] * ZN + M[0, 1]) / (M[1, 0] * ZN + M[1, 1])
        r = (Zin - Z0) / (Zin + Z0)
        t = 2.0 * ZN / (M[0, 0] * ZN + M[0, 1] + M[1, 0] * ZN + M[1, 1])
        R[i] = np.abs(r)
        T[i] = np.abs(t)
    return R, T


# ---------- Dataset generation ----------

def generate_dataset(output_dir: str, seed: int, include_zip: bool,
                     turbulence_models: List[TurbulenceModelConfig]) -> str:
    rng = np.random.default_rng(seed)

    # Physical properties representative of oil-water or saline-freshwater stratification
    props = PhysicalProperties(
        density_layer1=1025.0,  # seawater
        density_layer2=850.0,   # light oil or lighter fluid
        viscosity_layer1=1.2e-3,
        viscosity_layer2=6.0e-3,
        sound_speed_layer1=1500.0,
        sound_speed_layer2=1200.0,
    )

    # Domains
    domain2d = Domain2D(length_x=0.5, length_y=0.2, num_x=160, num_y=96)
    domain3d = Domain3D(length_x=0.3, length_y=0.2, length_z=0.15, num_x=96, num_y=64, num_z=48)

    acoustic = AcousticSetup(
        length_x=1.0, num_x=2000, dt=1.0e-6, num_steps=4000,
        sensor_positions=[0.2, 0.5, 0.8], attenuation_base=40.0  # Np/s
    )

    interface_fraction_y = 0.55
    interface_fraction_x = 0.45
    smoothing_length = 0.002
    perturbation_amplitude = 0.01

    timestamp = int(time.time())
    manifest: Dict[str, any] = {
        "generated_at_unix": timestamp,
        "seed": seed,
        "physical_properties": asdict(props),
        "domains": {
            "2d": asdict(domain2d),
            "3d": asdict(domain3d),
        },
        "acoustic_setup": asdict(acoustic),
        "interface_fractions": {"y": interface_fraction_y, "x": interface_fraction_x},
        "smoothing_length": smoothing_length,
        "perturbation_amplitude": perturbation_amplitude,
        "turbulence_models": [asdict(m) for m in turbulence_models],
    }

    base_dir = os.path.abspath(output_dir)
    cfd2d_dir = os.path.join(base_dir, "cfd", "2d")
    cfd3d_dir = os.path.join(base_dir, "cfd", "3d")
    acoustics_dir = os.path.join(base_dir, "acoustics")
    models_dir = os.path.join(base_dir, "models")
    validation_dir = os.path.join(base_dir, "validation")
    for d in [cfd2d_dir, cfd3d_dir, acoustics_dir, models_dir, validation_dir]:
        ensure_dir(d)

    # --- CFD-like 2D ---
    X2, Y2, vof2 = generate_vof_2d(domain2d, interface_fraction_y, smoothing_length, perturbation_amplitude, seed)
    u2, v2, p2 = generate_velocity_pressure_2d(X2, Y2, vof2, props)

    # Save base fields
    save_npz(os.path.join(cfd2d_dir, "fields_base.npz"), x=X2, y=Y2, vof=vof2, u=u2, v=v2, p=p2)

    # Turbulence per model
    turbulence_meta = []
    for model in turbulence_models:
        k2, eps2, mu_t2 = compute_turbulence_2d(u2, v2, vof2, Y2, props, model)
        model_slug = model.name.replace(' ', '_')
        save_npz(os.path.join(cfd2d_dir, f"turbulence_{model_slug}.npz"), k=k2, epsilon=eps2, mu_t=mu_t2)
        turbulence_meta.append({"model": model.name, "file": f"cfd/2d/turbulence_{model_slug}.npz"})

    # --- CFD-like 3D ---
    X3, Y3, Z3, vof3 = generate_vof_3d(domain3d, interface_fraction_y, smoothing_length, perturbation_amplitude, seed + 1)
    u3, v3, w3, p3 = generate_velocity_pressure_3d(X3, Y3, Z3, vof3, props)
    save_npz(os.path.join(cfd3d_dir, "fields_base.npz"), x=X3, y=Y3, z=Z3, vof=vof3, u=u3, v=v3, w=w3, p=p3)

    for model in turbulence_models:
        k3, eps3, mu_t3 = compute_turbulence_3d(u3, v3, w3, vof3, Y3, props, model)
        model_slug = model.name.replace(' ', '_')
        save_npz(os.path.join(cfd3d_dir, f"turbulence_{model_slug}.npz"), k=k3, epsilon=eps3, mu_t=mu_t3)

    # --- Acoustics ---
    acoustics = simulate_acoustic_1d_layered(acoustic, props, interface_fraction_x, seed + 2)
    # Save time series per sensor and combined CSV
    sensors = [k for k in acoustics.keys() if k.startswith("sensor_")]
    rows = []
    header = ["time"] + sensors
    for i in range(acoustic.num_steps):
        row = [float(acoustics["time"][i])] + [float(acoustics[s][i]) for s in sensors]
        rows.append(row)
    save_csv(os.path.join(acoustics_dir, "pressure_timeseries.csv"), header, rows)
    # Also save numpy format
    save_npz(os.path.join(acoustics_dir, "pressure_timeseries.npz"), **acoustics)

    # --- Mathematical model outputs ---
    # Effective sound speed and attenuation across volume fractions
    freqs = np.linspace(200.0, 8000.0, 128)
    phi1_values = np.linspace(0.05, 0.95, 19)
    c_eff = np.array([effective_sound_speed_series(phi, props.sound_speed_layer1, props.sound_speed_layer2) for phi in phi1_values])
    alpha_eff = attenuation_model(freqs, base_alpha=2.0)

    save_csv(os.path.join(models_dir, "effective_sound_speed.csv"), ["phi1", "c_eff"], [[float(phi), float(c)] for phi, c in zip(phi1_values, c_eff)])
    save_csv(os.path.join(models_dir, "attenuation_vs_frequency.csv"), ["frequency_hz", "alpha_np_per_m"], [[float(f), float(a)] for f, a in zip(freqs, alpha_eff)])

    # Transfer matrix for two layers and surrounding medium (air-water-oil-air)
    rho_list = [1.2, props.density_layer1, props.density_layer2, 1.2]
    c_list = [343.0, props.sound_speed_layer1, props.sound_speed_layer2, 343.0]
    thickness_list = [0.01, 0.05, 0.04, 0.01]
    R, T = compute_transfer_response(freqs, rho_list, c_list, thickness_list)
    save_csv(os.path.join(models_dir, "transfer_matrix_response.csv"), ["frequency_hz", "R_mag", "T_mag"], [[float(f), float(r), float(t)] for f, r, t in zip(freqs, R, T)])

    # Time-delay estimates assuming propagation across acoustic.length_x
    # Use c_eff at phi from 2D interface fraction as a proxy
    phi1_proxy = interface_fraction_y
    c_proxy = effective_sound_speed_series(phi1_proxy, props.sound_speed_layer1, props.sound_speed_layer2)
    T0 = acoustic.length_x / c_proxy
    save_json(os.path.join(models_dir, "time_delay.json"), {"distance_m": acoustic.length_x, "c_proxy_m_per_s": c_proxy, "T0_s": T0})

    # --- Validation data: fabricate experimental variants ---
    # Waveforms: add noise and slight dispersion to simulated pressure
    exp_rows = []
    header_exp = ["time"] + sensors
    noise_scale = 0.03
    dispersion = 1.005
    for i in range(acoustic.num_steps):
        t = float(acoustics["time"][i]) * dispersion
        row = [t]
        for s in sensors:
            sim_val = float(acoustics[s][i])
            exp_val = 1.02 * sim_val + noise_scale * rng.normal()
            row.append(exp_val)
        exp_rows.append(row)
    save_csv(os.path.join(validation_dir, "acoustic_waveforms_simulated.csv"), header, rows)
    save_csv(os.path.join(validation_dir, "acoustic_waveforms_experimental.csv"), header_exp, exp_rows)

    # Attenuation values: add multiplicative and additive bias
    atten_sim = alpha_eff
    atten_exp = 1.05 * atten_sim + 0.1 * (freqs / freqs.max())
    save_csv(os.path.join(validation_dir, "attenuation_comparison.csv"), ["frequency_hz", "alpha_sim", "alpha_exp"], [[float(f), float(a1), float(a2)] for f, a1, a2 in zip(freqs, atten_sim, atten_exp)])

    # Sound speeds: measured vs predicted at sampled phi
    c_meas = c_eff * (1.0 + 0.01 * (2.0 * rng.random(c_eff.shape) - 1.0))
    save_csv(os.path.join(validation_dir, "sound_speed_comparison.csv"), ["phi1", "c_pred", "c_meas"], [[float(phi), float(cp), float(cm)] for phi, cp, cm in zip(phi1_values, c_eff, c_meas)])

    # Manifest/meta
    manifest.update({
        "files": {
            "cfd_2d_base": "cfd/2d/fields_base.npz",
            "cfd_3d_base": "cfd/3d/fields_base.npz",
            "acoustics_timeseries_csv": "acoustics/pressure_timeseries.csv",
            "acoustics_timeseries_npz": "acoustics/pressure_timeseries.npz",
            "models_effective_sound_speed": "models/effective_sound_speed.csv",
            "models_attenuation_vs_frequency": "models/attenuation_vs_frequency.csv",
            "models_transfer_matrix": "models/transfer_matrix_response.csv",
            "models_time_delay": "models/time_delay.json",
            "validation_waveforms_sim": "validation/acoustic_waveforms_simulated.csv",
            "validation_waveforms_exp": "validation/acoustic_waveforms_experimental.csv",
            "validation_attenuation": "validation/attenuation_comparison.csv",
            "validation_sound_speed": "validation/sound_speed_comparison.csv",
            "turbulence_files": turbulence_meta,
        }
    })
    save_json(os.path.join(base_dir, "manifest.json"), manifest)

    # Zip packaging
    zip_path = None
    if include_zip:
        zip_path = base_dir.rstrip(os.sep) + ".zip"
        with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(base_dir):
                for file in files:
                    fp = os.path.join(root, file)
                    arcname = os.path.relpath(fp, os.path.dirname(base_dir))
                    z.write(fp, arcname)
    return zip_path or base_dir


def parse_models_arg(models_arg: Optional[str]) -> List[TurbulenceModelConfig]:
    if not models_arg:
        return DEFAULT_TURBULENCE_MODELS
    requested = [m.strip().lower() for m in models_arg.split(',') if m.strip()]
    mapping = {m.name: m for m in DEFAULT_TURBULENCE_MODELS}
    models = []
    for name in requested:
        if name in mapping:
            models.append(mapping[name])
        else:
            # Accept a few synonyms
            key = name.replace('-', ' ')
            if key in mapping:
                models.append(mapping[key])
            elif key == 'k omega sst':
                models.append(mapping['k-omega-sst'])
            else:
                raise ValueError(f"Unsupported turbulence model: {name}")
    return models


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic stratified flow simulation dataset (CFD-like and mathematical models)")
    parser.add_argument('--output', required=True, help='Output directory for the dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--zip', action='store_true', help='Package the dataset directory into a zip file')
    parser.add_argument('--models', type=str, default=None, help='Comma-separated turbulence models: k-epsilon,k-omega-sst,les')
    args = parser.parse_args()

    models = parse_models_arg(args.models)

    out = generate_dataset(args.output, args.seed, args.zip, models)
    print(out)


if __name__ == '__main__':
    main()
