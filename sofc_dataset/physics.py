from __future__ import annotations
from typing import Dict, Tuple
import json
import math
import numpy as np


def generate_grid(nx: int, ny: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    return x, y, z


def _smooth_noise(shape: Tuple[int, int, int], rng: np.random.Generator, num_terms: int = 4) -> np.ndarray:
    nx, ny, nz = shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    field = np.zeros(shape, dtype=np.float64)
    for _ in range(num_terms):
        fx = rng.integers(1, 4)
        fy = rng.integers(1, 4)
        fz = rng.integers(1, 4)
        phase = rng.uniform(0, 2 * math.pi)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.sin(2 * math.pi * (fx * X + fy * Y + fz * Z) + phase)
    field /= num_terms
    return field


def _clamp(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


def _compute_von_mises(sxx: np.ndarray, syy: np.ndarray, szz: np.ndarray, sxy: np.ndarray, syz: np.ndarray, szx: np.ndarray) -> np.ndarray:
    term1 = (sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2
    term2 = 6 * (sxy ** 2 + syz ** 2 + szx ** 2)
    return np.sqrt(0.5 * term1 + term2)


def _finite_difference_gradient(u: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    grad = np.zeros_like(u)
    # central differences interior
    slc_center = [slice(None)] * u.ndim
    slc_prev = [slice(None)] * u.ndim
    slc_next = [slice(None)] * u.ndim
    slc_center[axis] = slice(1, -1)
    slc_prev[axis] = slice(0, -2)
    slc_next[axis] = slice(2, None)
    grad[tuple(slc_center)] = (u[tuple(slc_next)] - u[tuple(slc_prev)]) / (2.0 * spacing)
    # forward/backward at boundaries
    slc0 = [slice(None)] * u.ndim; slc0[axis] = 0
    slc1 = [slice(None)] * u.ndim; slc1[axis] = 1
    slc_end = [slice(None)] * u.ndim; slc_end[axis] = -1
    slc_endm1 = [slice(None)] * u.ndim; slc_endm1[axis] = -2
    grad[tuple(slc0)] = (u[tuple(slc1)] - u[tuple(slc0)]) / spacing
    grad[tuple(slc_end)] = (u[tuple(slc_end)] - u[tuple(slc_endm1)]) / spacing
    return grad


def _nernst_voltage(T_K: float, pH2: float, pH2O: float, pO2: float) -> float:
    # Simplified Nernst equation for SOFC: E = E0 + (RT/2F) ln(pH2 * sqrt(pO2) / pH2O)
    # constants
    R = 8.314
    F = 96485.3329
    E0 = 1.1  # V, nominal
    pH2 = max(pH2, 1e-6)
    pH2O = max(pH2O, 1e-6)
    pO2 = max(pO2, 1e-6)
    return E0 + (R * T_K / (2.0 * F)) * math.log(pH2 * math.sqrt(pO2) / pH2O)


def generate_fields(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    params: Dict[str, float],
    seed: int | None = None,
) -> Dict[str, np.ndarray | float | Dict[str, float]]:
    rng = np.random.default_rng(seed if seed is not None else int(params.get("operating_hours", 0)) + 12345)
    nx, ny, nz = len(x), len(y), len(z)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Temperatures in Kelvin
    T_in_fuel = params["inlet_fuel_temp_C"] + 273.15
    T_in_air = params["inlet_air_temp_C"] + 273.15
    base_T = 0.5 * (T_in_fuel + T_in_air)

    # Simple heat generation scaling from current density and resistivity proxy
    current_density = params["current_density_A_per_cm2"]  # A/cm^2
    ionic_cond = params["ionic_conductivity_S_per_m"]
    electronic_cond = params["electronic_conductivity_S_per_m"]
    effective_resistivity = 1.0 / max(ionic_cond + electronic_cond * 1e-3, 1e-6)
    heat_scale = 30.0 + 200.0 * current_density * effective_resistivity

    temp_noise = _smooth_noise((nx, ny, nz), rng, num_terms=5)
    T = base_T + 40.0 * (X - 0.5) + 20.0 * (Y - 0.5) + 10.0 * (Z - 0.5) + heat_scale * temp_noise
    T = _clamp(T, 500.0, 1400.0)

    # Species fields along flow x-direction
    H2_in = params["fuel_H2_frac"]
    H2O_in = params["fuel_H2O_frac"]
    CO_in = params["fuel_CO_frac"]
    CH4_in = params["fuel_CH4_frac"]
    air_util = params["air_utilization_frac"]

    k_consume = 1.0 + 2.0 * params["fuel_utilization_frac"]
    H2 = H2_in * np.exp(-k_consume * X)
    H2O = 1.0 - (1.0 - H2O_in) * np.exp(-0.8 * k_consume * X)
    # O2 proportional to air utilization consumption over x
    O2_in = 0.21
    O2 = O2_in * np.exp(-1.5 * air_util * X)

    # Current density field perturbed by species and temperature
    i_field = current_density * (1.0 + 0.2 * (T - T.mean()) / (T.std() + 1e-6))
    i_field *= (0.8 + 0.4 * H2 / (H2.max() + 1e-6))
    i_field = _clamp(i_field, 0.05, 3.0)

    # Overall cell voltage from Nernst minus losses
    T_bulk = float(T.mean())
    E_nernst = _nernst_voltage(T_bulk, float(H2_in), float(H2O_in), float(O2_in))
    R_area = 0.1 + 0.2 * effective_resistivity  # ohm*cm^2 proxy
    a_tafel = 0.04  # V scaling
    V_cell = E_nernst - a_tafel * math.log(max(current_density, 1e-4)) - current_density * R_area
    V_cell = float(max(min(V_cell, 1.2), 0.2))

    # Displacements from thermal expansion and compliance
    alpha = params["interconnect_CTE_per_K"]
    E_GPa = params["interconnect_YoungsModulus_GPa"]
    E = E_GPa * 1e9
    nu = 0.3
    thermal_strain = alpha * (T - base_T)
    compliance = 1.0 / E
    Ux = compliance * (thermal_strain * (X - 0.5))
    Uy = compliance * (thermal_strain * (Y - 0.5))
    Uz = 0.5 * compliance * (thermal_strain * (Z - 0.5))

    # Strain tensor from displacement gradients
    dx = 1.0 / max(nx - 1, 1)
    dy = 1.0 / max(ny - 1, 1)
    dz = 1.0 / max(nz - 1, 1)

    dux_dx = _finite_difference_gradient(Ux, axis=0, spacing=dx)
    dux_dy = _finite_difference_gradient(Ux, axis=1, spacing=dy)
    dux_dz = _finite_difference_gradient(Ux, axis=2, spacing=dz)
    duy_dx = _finite_difference_gradient(Uy, axis=0, spacing=dx)
    duy_dy = _finite_difference_gradient(Uy, axis=1, spacing=dy)
    duy_dz = _finite_difference_gradient(Uy, axis=2, spacing=dz)
    duz_dx = _finite_difference_gradient(Uz, axis=0, spacing=dx)
    duz_dy = _finite_difference_gradient(Uz, axis=1, spacing=dy)
    duz_dz = _finite_difference_gradient(Uz, axis=2, spacing=dz)

    exx = dux_dx
    eyy = duy_dy
    ezz = duz_dz
    exy = 0.5 * (dux_dy + duy_dx)
    eyz = 0.5 * (duy_dz + duz_dy)
    ezx = 0.5 * (duz_dx + dux_dz)

    # Remove thermal expansion to get mechanical strain
    eth = thermal_strain
    exx_m = exx - eth
    eyy_m = eyy - eth
    ezz_m = ezz - eth

    # Isotropic linear elasticity: stress = C : strain_m
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    tr_e = exx_m + eyy_m + ezz_m
    sxx = 2 * mu * exx_m + lam * tr_e
    syy = 2 * mu * eyy_m + lam * tr_e
    szz = 2 * mu * ezz_m + lam * tr_e
    sxy = 2 * mu * exy
    syz = 2 * mu * eyz
    szx = 2 * mu * ezx

    von_mises = _compute_von_mises(sxx, syy, szz, sxy, syz, szx)

    # Fracture metrics at an approximate crack location
    a_m = params["initial_crack_length_mm"] * 1e-3
    cx = params["initial_crack_x_frac"]
    cy = params["initial_crack_y_frac"]
    ix = int(round(cx * (nx - 1)))
    iy = int(round(cy * (ny - 1)))
    iz = nz // 2
    s_tip_yy = float(syy[ix, iy, iz])
    s_tip_xy = float(sxy[ix, iy, iz])
    s_tip_yz = float(syz[ix, iy, iz])
    a_eff = max(a_m, 1e-6)
    K_I = s_tip_yy * math.sqrt(math.pi * a_eff)
    K_II = s_tip_xy * math.sqrt(math.pi * a_eff)
    K_III = s_tip_yz * math.sqrt(math.pi * a_eff)

    # Energy release rate (approximate isotropic)
    G = ((1 - nu ** 2) / E) * (K_I ** 2 + K_II ** 2) + ((1 + nu) / E) * (K_III ** 2)

    # Creep proxy (steady-state Norton's law-like)
    A_creep = 1e-25
    n_creep = 3.0
    Q_by_R = 30000.0  # K
    creep_strain_rate = A_creep * (von_mises ** n_creep) * np.exp(-Q_by_R / T)
    hours = params.get("operating_hours", 0.0)
    creep_strain = creep_strain_rate * hours * 3600.0

    fields: Dict[str, np.ndarray | float | Dict[str, float]] = {
        "temperature_K": T.astype(np.float32),
        "current_density_A_per_cm2": i_field.astype(np.float32),
        "H2_molfrac": H2.astype(np.float32),
        "H2O_molfrac": H2O.astype(np.float32),
        "O2_molfrac": O2.astype(np.float32),
        "Ux_m": Ux.astype(np.float32),
        "Uy_m": Uy.astype(np.float32),
        "Uz_m": Uz.astype(np.float32),
        "strain_xx": exx.astype(np.float32),
        "strain_yy": eyy.astype(np.float32),
        "strain_zz": ezz.astype(np.float32),
        "strain_xy": exy.astype(np.float32),
        "strain_yz": eyz.astype(np.float32),
        "strain_zx": ezx.astype(np.float32),
        "stress_xx_Pa": sxx.astype(np.float32),
        "stress_yy_Pa": syy.astype(np.float32),
        "stress_zz_Pa": szz.astype(np.float32),
        "stress_xy_Pa": sxy.astype(np.float32),
        "stress_yz_Pa": syz.astype(np.float32),
        "stress_zx_Pa": szx.astype(np.float32),
        "von_mises_Pa": von_mises.astype(np.float32),
        "creep_strain": creep_strain.astype(np.float32),
        "cell_voltage_V": float(V_cell),
        "fracture_metrics": {
            "K_I_Pa_sqrt_m": float(K_I),
            "K_II_Pa_sqrt_m": float(K_II),
            "K_III_Pa_sqrt_m": float(K_III),
            "G_J_per_m2": float(G),
        },
    }

    return fields
