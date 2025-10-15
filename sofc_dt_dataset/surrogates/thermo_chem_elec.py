from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

F = 96485.33212  # Faraday constant [C/mol]
R = 8.314462618  # gas constant [J/mol/K]


def _make_grid(shape: Tuple[int, int, int]):
    nx, ny, nz = shape
    x = np.linspace(0.0, 1.0, nx, dtype=np.float64)
    y = np.linspace(0.0, 1.0, ny, dtype=np.float64)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float64)
    return x, y, z


def _smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    """Lightweight separable smoothing with uniform kernel; shape-preserving."""
    if sigma <= 0.0:
        return field
    if field.ndim == 1:
        return _smooth_along_axis(field[np.newaxis, :, np.newaxis], sigma, axis=1)[0, :, 0]
    if field.ndim == 2:
        out = _smooth_along_axis(field[np.newaxis, ...], sigma, axis=1)[0]
        out = _smooth_along_axis(out[np.newaxis, ...], sigma, axis=2)[0]
        return out
    if field.ndim == 3:
        out = field
        out = _smooth_along_axis(out, sigma, axis=0)
        out = _smooth_along_axis(out, sigma, axis=1)
        out = _smooth_along_axis(out, sigma, axis=2)
        return out
    return field


def _smooth_along_axis(arr: np.ndarray, sigma: float, axis: int) -> np.ndarray:
    k = max(1, int(2 * sigma + 1)) | 1  # ensure odd
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    arr_moved = np.moveaxis(arr, axis, 0)
    pad_width = [(0, 0)] * arr_moved.ndim
    pad_width[0] = (pad_left, pad_right)
    x = np.pad(arr_moved, pad_width, mode="edge")
    # prefix zero for correct windowed sums
    c = np.cumsum(x, axis=0, dtype=np.float64)
    c0 = np.take(c, indices=range(0, c.shape[0]), axis=0)
    c_pad = np.concatenate([np.zeros_like(c0[:1]), c0], axis=0)
    window_sums = c_pad[k:] - c_pad[:-k]
    res = window_sums / k
    res = np.moveaxis(res, 0, axis)
    return res


def generate_current_density_field(shape: Tuple[int, int, int], i_avg: float, Uf: float, seed: int) -> np.ndarray:
    # Spatial distribution: decay along x due to fuel depletion; channelization in y; reaction zone near z=0.5
    nx, ny, nz = shape
    x, y, z = _make_grid(shape)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Base field with exponential decay along x (stronger with higher Uf)
    kx = 0.5 * Uf + 0.05
    base = np.exp(-kx * X)

    # Channel stripes along y
    n_channels = 6
    stripe = 1.0 + 0.25 * np.sin(2.0 * np.pi * n_channels * Y + 0.8)

    # Reaction preference near mid-thickness
    depth = 1.0 + 0.3 * np.exp(-((Z - 0.4) ** 2) / (2 * 0.05 ** 2))

    rng = np.random.default_rng(seed)
    noise = rng.normal(1.0, 0.05, size=shape)

    field = i_avg * base * stripe * depth * noise
    field = _smooth(field, sigma=1.2)

    # Normalize to preserve i_avg
    scale = i_avg / field.mean()
    return field * scale


def generate_temperature_field(shape: Tuple[int, int, int], T_in: float, i_field: np.ndarray) -> np.ndarray:
    # Simple thermal surrogate: inlet temperature + ohmic + activation heating components
    nx, ny, nz = shape
    x, y, z = _make_grid(shape)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Effective resistivity (decreases with T). Approximate using T_in for simplicity
    rho0 = 1.5e-5  # [Ohm*m]
    Ea = 0.6 * 96485 / 1000  # pseudo activation energy scale in J/mol-like units
    rho_eff = rho0  # keep simple for stability

    # Heat source ~ i^2 * rho
    q = (i_field ** 2) * rho_eff

    # Conduction effect: cooler near inlet x=0, warmer downstream, cosine variation across y
    kx = 0.12
    base_grad = 1.0 + 0.5 * (1.0 - np.exp(-kx * X)) * (1.0 + 0.1 * np.cos(2 * np.pi * Y))

    # Convert source to temperature rise (scaled surrogate)
    c_scale = 1.0e-6
    dT = c_scale * _smooth(q, sigma=1.5) * base_grad

    # Slight through-thickness gradient
    dT += 3.0 * (Z - 0.5)

    T = T_in + dT
    return T


def generate_species_fields(shape: Tuple[int, int, int], i_field: np.ndarray, pressure: float, y_H2_in: float, y_H2O_in: float, y_O2_in: float, Uf: float) -> Dict[str, np.ndarray]:
    x, y, z = _make_grid(shape)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Effective decay rates tied to utilization and average current
    i_mean = float(i_field.mean())
    beta_fuel = 0.4 * Uf + 1e-4 * i_mean / 1e4
    beta_ox = 0.35 * Uf + 8e-5 * i_mean / 1e4

    p_H2_in = y_H2_in * pressure
    p_H2O_in = y_H2O_in * pressure
    p_O2_in = y_O2_in * pressure

    # H2 decreases downstream (x), H2O increases; O2 decreases from cathode side (we model along x as well)
    p_H2 = p_H2_in * np.exp(-beta_fuel * X) * (1.0 + 0.05 * np.sin(2 * np.pi * 5 * Y))
    p_H2O = (p_H2O_in + (p_H2_in - p_H2)) * (1.0 + 0.02 * np.cos(2 * np.pi * 3 * Y))
    p_O2 = p_O2_in * np.exp(-beta_ox * (1.0 - X)) * (1.0 + 0.03 * np.sin(2 * np.pi * 4 * (1 - Y)))

    # Smooth to mimic diffusion
    p_H2 = _smooth(p_H2, 1.0)
    p_H2O = _smooth(p_H2O, 1.0)
    p_O2 = _smooth(p_O2, 1.0)

    return {"p_H2": p_H2, "p_H2O": p_H2O, "p_O2": p_O2}


def generate_overpotentials(shape: Tuple[int, int, int], T: np.ndarray, i_field: np.ndarray) -> Dict[str, np.ndarray]:
    # Activation overpotential via simplified Butler-Volmer
    n = 2  # electrons
    alpha = 0.5
    i00 = 2.0  # A/m^2 pre-exponential for exchange current
    Ea = 6.0e4  # J/mol
    i0 = i00 * np.exp(-Ea / (R * np.maximum(T, 1.0)))

    eta_act = (R * T / (alpha * n * F)) * np.arcsinh(0.5 * i_field / (np.maximum(i0, 1e-6)))

    # Ohmic overpotential
    rho0 = 2.0e-5  # Ohm*m
    Er = 2.0e4
    rho = rho0 * np.exp(Er / np.maximum(T, 1.0))
    eta_ohm = rho * i_field

    # Concentration overpotential (log of species depletion proxy based on i)
    # use normalized i to avoid negative/invalid logs
    i_norm = i_field / (np.maximum(i_field.max(), 1.0))
    eta_conc = (R * T / (n * F)) * np.log(1.0 / np.maximum(1.0 - 0.85 * i_norm, 1e-3))

    # Smooth for stability
    eta_act = _smooth(eta_act, 1.0)
    eta_ohm = _smooth(eta_ohm, 1.0)
    eta_conc = _smooth(eta_conc, 1.0)

    return {"eta_act": eta_act, "eta_ohm": eta_ohm, "eta_conc": eta_conc}


def generate_lf_temperature_1d(nx: int, T3d: np.ndarray) -> np.ndarray:
    # Average over y,z to form 1D profile along x
    T1d = T3d.mean(axis=(1, 2))
    # Optionally smooth
    return _smooth(T1d, 1.0)
