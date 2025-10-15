from __future__ import annotations
import numpy as np
from typing import Dict

R = 8.314462618


def _smooth(x: np.ndarray, s: float) -> np.ndarray:
    # lightweight smoothing; preserve shape exactly
    if s <= 0:
        return x
    if x.ndim == 1:
        return _smooth_axis(x[np.newaxis, :, np.newaxis], s, axis=1)[0, :, 0]
    if x.ndim == 2:
        out = _smooth_axis(x[np.newaxis, ...], s, axis=1)[0]
        out = _smooth_axis(out[np.newaxis, ...], s, axis=2)[0]
        return out
    if x.ndim == 3:
        out = _smooth_axis(x, s, axis=0)
        out = _smooth_axis(out, s, axis=1)
        out = _smooth_axis(out, s, axis=2)
        return out
    return x


def _smooth_axis(arr: np.ndarray, s: float, axis: int) -> np.ndarray:
    k = max(1, int(2 * s + 1)) | 1
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    arr_moved = np.moveaxis(arr, axis, 0)
    pad_width = [(0, 0)] * arr_moved.ndim
    pad_width[0] = (pad_left, pad_right)
    x = np.pad(arr_moved, pad_width, mode="edge")
    c = np.cumsum(x, axis=0, dtype=np.float64)
    c_pad = np.concatenate([np.zeros_like(c[:1]), c], axis=0)
    window_sums = c_pad[k:] - c_pad[:-k]
    res = window_sums / k
    res = np.moveaxis(res, 0, axis)
    return res


def compute_mechanical_fields(T: np.ndarray, i_field: np.ndarray, E: float, nu: float, alpha_cte: float, hours: float) -> Dict[str, np.ndarray]:
    # Thermal stress surrogate: sigma_th = E * alpha * (T - T_ref) / (1 - nu)
    T_ref = np.median(T)
    sigma_th = E * alpha_cte * (T - T_ref) / max(1e-6, (1.0 - nu))

    # Add gradient-induced stress contribution
    # approximate |grad T|
    gx = np.gradient(T, axis=0)
    gy = np.gradient(T, axis=1)
    gz = np.gradient(T, axis=2)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    sigma_grad = 5.0e6 * grad_mag  # scale factor

    sigma_vm = np.abs(sigma_th) + sigma_grad
    sigma_vm = _smooth(sigma_vm, 1.0)

    # Elastic strain (von Mises-equivalent proxy)
    epsilon_eq = sigma_vm / max(E, 1e3)

    # Creep strain surrogate: Norton-Bailey epsilon_c = A * sigma^n * t * exp(-Q/RT)
    Q = 1.2e5
    A = 1e-25
    n = 3.0
    T_eff = np.maximum(T, 1.0)
    creep = A * (np.maximum(sigma_vm, 0.0) ** n) * (hours * 3600.0) * np.exp(-Q / (R * T_eff))

    epsilon_total = epsilon_eq + creep

    # Damage indicator: combine stress and creep via a smooth threshold
    sigma_crit = 80e6  # 80 MPa threshold-like
    damage_cont = 1.0 / (1.0 + np.exp(-(sigma_vm - sigma_crit) / (0.15 * sigma_crit)))
    damage = np.clip(damage_cont + 5.0 * creep / (1e-3 + np.percentile(creep, 99) + 1e-12), 0.0, 1.0)

    fields = {
        "sigma_vm": sigma_vm,
        "epsilon_eq": epsilon_total,
        "damage": damage,
    }
    return fields


def estimate_time_to_failure_hours(sigma_vm: np.ndarray, T: np.ndarray) -> float:
    # Basquin-like life model with Arrhenius temperature term
    sigma_mean = float(np.mean(sigma_vm)) + 1e-6
    T_mean = float(np.mean(T))
    m = 4.5
    t0 = 2.0e4  # scaling hours
    Q = 8.0e4
    life = t0 * (80e6 / sigma_mean) ** m * np.exp(Q / (R * T_mean))
    life = float(np.clip(life, 10.0, 1e6))
    return life
