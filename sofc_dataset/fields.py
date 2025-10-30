from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int
    nz: int


def _make_grid(nx: int, ny: int, nz: int, thickness_um: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create regular grid coordinates (meters)."""
    # Assume square in-plane domain; set side length so that area scales with active area later
    # Here we use unit length 1.0 in arbitrary meters and scale fields relative to area in formulas.
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, thickness_um * 1e-6, nz)
    return x, y, z


def _smooth_random_field(shape: Tuple[int, int, int], rng: np.random.Generator, corr: float = 0.1) -> np.ndarray:
    """Generate a smooth random field by convolving white noise with a small kernel."""
    nz, ny, nx = shape
    noise = rng.standard_normal(shape)
    # Build separable kernel
    def kernel1d(n: int, c: float) -> np.ndarray:
        r = np.arange(n) - (n - 1) / 2.0
        k = np.exp(-(r ** 2) / (2.0 * (c * n) ** 2 + 1e-12))
        k /= k.sum()
        return k

    kx = kernel1d(9, corr)
    ky = kernel1d(9, corr)
    kz = kernel1d(7, corr)

    def conv1d_along(arr: np.ndarray, axis: int, k: np.ndarray) -> np.ndarray:
        pad = len(k) // 2
        arr_pad = np.pad(
            arr,
            [(pad, pad) if a == axis else (0, 0) for a in range(arr.ndim)],
            mode="reflect",
        )
        out = np.zeros_like(arr)
        # roll-sum convolution
        for i in range(len(k)):
            slc = [slice(None)] * arr_pad.ndim
            slc[axis] = slice(i, i + arr.shape[axis])
            out += k[i] * arr_pad[tuple(slc)]
        return out

    f = noise
    f = conv1d_along(f, 2, kx)
    f = conv1d_along(f, 1, ky)
    f = conv1d_along(f, 0, kz)
    f = (f - f.mean()) / (f.std() + 1e-12)
    return f


def _layer_index(z: np.ndarray, t_an: float, t_el: float, t_ca: float, t_ic: float, t_se: float) -> np.ndarray:
    """Return integer layer index per z: 0=anode,1=electrolyte,2=cathode,3=interconnect,4=sealant."""
    bounds = np.cumsum([t_an, t_el, t_ca, t_ic, t_se])
    idx = np.zeros_like(z, dtype=int)
    idx[z >= bounds[0]] = 1
    idx[z >= bounds[1]] = 2
    idx[z >= bounds[2]] = 3
    idx[z >= bounds[3]] = 4
    return idx


def generate_fields(inputs: Dict[str, float | str], grid: GridSpec, seed: int | None = None) -> Dict[str, np.ndarray]:
    """Synthesize coupled multi-physics 3D fields for SOFC.

    Shapes use (nz, ny, nx). Units annotated in keys for clarity.
    """
    rng = np.random.default_rng(seed)

    nx, ny, nz = grid.nx, grid.ny, grid.nz

    # Thickness per layer (m)
    t_an = float(inputs["thickness_anode_um"]) * 1e-6
    t_el = float(inputs["thickness_electrolyte_um"]) * 1e-6
    t_ca = float(inputs["thickness_cathode_um"]) * 1e-6
    t_ic = float(inputs["thickness_interconnect_um"]) * 1e-6
    t_se = float(inputs["thickness_sealant_um"]) * 1e-6
    t_tot = t_an + t_el + t_ca + t_ic + t_se

    x, y, z = _make_grid(nx, ny, nz, thickness_um=(t_tot * 1e6))
    # 3D meshgrid, with Z major
    X = np.broadcast_to(x[np.newaxis, np.newaxis, :], (nz, ny, nx))
    Y = np.broadcast_to(y[np.newaxis, :, np.newaxis], (nz, ny, nx))
    Z = np.broadcast_to(z[:, np.newaxis, np.newaxis], (nz, ny, nx))

    # Mean operating points
    jd_mean = float(inputs["current_density_A_per_cm2"])  # A/cm^2
    V_cell = float(inputs["voltage_V"])  # V

    # Flow/channel influence (spatial modulation)
    design = str(inputs["flow_channel_design"]).lower()
    pitch = float(inputs["flow_channel_pitch_mm"]) / 1000.0
    # Frequency approx inversely proportional to pitch (normalized by domain length=1m here)
    freq = max(1.0, min(50.0, 1.0 / (pitch + 1e-6)))
    if design == "serpentine":
        pattern = 0.5 * (np.sin(2 * math.pi * freq * X) * np.cos(2 * math.pi * 0.5 * freq * Y))
    elif design == "interdigitated":
        pattern = 0.5 * (np.sign(np.sin(2 * math.pi * freq * X)) * np.cos(2 * math.pi * freq * Y))
    else:
        pattern = 0.4 * np.cos(2 * math.pi * freq * X)

    texture = 0.15 * _smooth_random_field((nz, ny, nx), rng, corr=0.12)

    # Current density distribution (A/cm^2)
    jd = jd_mean * (1.0 + 0.25 * pattern + 0.1 * texture)
    jd = np.clip(jd, 0.02, None)

    # Electrolyte ASR estimate ~ thickness / conductivity (Ohm*cm^2)
    sigma_el = float(inputs["electrolyte_ionic_sigma_S_m"])  # S/m
    asr_el = (t_el) / (sigma_el + 1e-9)  # Ohm*m -> scale to cm^2 below
    asr_el_cm2 = asr_el * 1e4

    # Overpotential roughly proportional to current density * ASR
    overpotential = jd * asr_el_cm2

    # Temperature field (K) – base on inlet temps and ohmic heat q ~ jd*V_loss
    Tin_f = float(inputs["inlet_temp_fuel_C"]) + 273.15
    Tin_a = float(inputs["inlet_temp_air_C"]) + 273.15
    T_base = 0.5 * (Tin_f + Tin_a)

    V_oc = 1.1  # nominal OCV
    V_loss = max(0.02, V_oc - V_cell)
    q_norm = jd * V_loss  # W/cm^2 proxy (scaled)
    # Normalize and distribute heat mostly in electrolyte and active layers
    layer_idx_1d = _layer_index(z, t_an, t_el, t_ca, t_ic, t_se)
    w_z = np.interp(Z[:, 0, 0], z, (layer_idx_1d == 1) * 0.6 + (layer_idx_1d == 0) * 0.2 + (layer_idx_1d == 2) * 0.2)
    w_z = w_z[:, np.newaxis, np.newaxis]
    # Temperature rise scaled with q, attenuated by convective cooling (via flow)
    fuel_flow = float(inputs["fuel_flow_rate_slpm"])  # SLPM
    air_flow = float(inputs["air_flow_rate_slpm"])  # SLPM
    cooling = 0.2 + 0.01 * (fuel_flow + 0.5 * air_flow)
    dT = (8.0 + 150.0 * q_norm / (1.0 + cooling)) * w_z
    # Add smooth variation along x (flow direction)
    dT *= (0.8 + 0.2 * (1.0 - X))
    T = T_base + dT + 2.0 * _smooth_random_field((nz, ny, nx), rng, corr=0.08)

    # Species fields (mole fractions): H2 decreases along x, H2O increases
    h2_in = 0.8  # nominal inlet mol fraction (dry basis simplified)
    utilization = np.clip(0.3 + 0.25 * (jd_mean / 2.0), 0.1, 0.8)
    decay_rate = 1.5 * utilization
    H2 = h2_in * np.exp(-decay_rate * X) * (1.0 - 0.05 * pattern)
    H2 = np.clip(H2, 0.02, 0.95)
    H2O = np.clip(1.0 - H2 - 0.1, 0.01, 0.97)

    # Thermal-mechanical response: simple thermoelastic estimate
    # Effective modulus (Pa) varying by layer, convert GPa->Pa
    E_an = float(inputs["anode_E_GPa"]) * 1e9
    E_el = float(inputs["electrolyte_E_GPa"]) * 1e9
    E_ca = float(inputs["cathode_E_GPa"]) * 1e9
    E_ic = float(inputs["interconnect_E_GPa"]) * 1e9
    E_se = float(inputs["sealant_E_GPa"]) * 1e9

    alpha_an = float(inputs["anode_cte_1e6_per_K"]) * 1e-6
    alpha_el = float(inputs["electrolyte_cte_1e6_per_K"]) * 1e-6
    alpha_ca = float(inputs["cathode_cte_1e6_per_K"]) * 1e-6
    alpha_ic = float(inputs["interconnect_cte_1e6_per_K"]) * 1e-6
    alpha_se = float(inputs["sealant_cte_1e6_per_K"]) * 1e-6

    # Map layer-dependent properties over z
    layers = _layer_index(z, t_an, t_el, t_ca, t_ic, t_se)
    E_z = np.choose(layers, [E_an, E_el, E_ca, E_ic, E_se]).astype(float)
    alpha_z = np.choose(layers, [alpha_an, alpha_el, alpha_ca, alpha_ic, alpha_se]).astype(float)
    E = np.broadcast_to(E_z[:, np.newaxis, np.newaxis], (nz, ny, nx))
    alpha = np.broadcast_to(alpha_z[:, np.newaxis, np.newaxis], (nz, ny, nx))

    T_ref = 298.15
    deltaT = T - T_ref

    # Strain (small): epsilon ~ alpha * deltaT, with slight anisotropy
    exx = 0.8 * alpha * deltaT
    eyy = 0.8 * alpha * deltaT
    ezz = 1.2 * alpha * deltaT
    exy = 0.05 * alpha * deltaT * np.sin(2 * math.pi * freq * X)
    exz = 0.03 * alpha * deltaT * np.cos(2 * math.pi * 0.7 * freq * X)
    eyz = 0.03 * alpha * deltaT * np.sin(2 * math.pi * 0.7 * freq * Y)

    # Von Mises stress (Pa) approximate from E * alpha * deltaT / (1 - nu)
    nu = 0.28
    sigma_equiv = (E * alpha * deltaT) / (1.0 - nu)
    # Add contribution from gradients (thermal + species)
    gx, gy = np.gradient(T.mean(axis=0))
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_term = np.broadcast_to(grad_mag[np.newaxis, :, :], (nz, ny, nx))
    sigma_von_mises = np.abs(sigma_equiv) * (0.6 + 0.4 * (grad_term / (grad_term.max() + 1e-9)))

    # Displacement (m): crude integration of strain over dimension
    ux = np.cumsum(exx.mean(axis=0), axis=1) / nx
    uy = np.cumsum(eyy.mean(axis=0), axis=0) / ny
    uz = np.cumsum(ezz[:, ny // 2, nx // 2]) / nz
    UX = np.broadcast_to(ux[np.newaxis, :, :], (nz, ny, nx)) * 1e-6
    UY = np.broadcast_to(uy[np.newaxis, :, :], (nz, ny, nx)) * 1e-6
    UZ = np.broadcast_to(uz[:, np.newaxis, np.newaxis], (nz, ny, nx)) * 1e-6

    fields: Dict[str, np.ndarray] = {
        # Electrochemical
        "current_density_A_per_cm2": jd.astype(np.float32),
        "overpotential_V": overpotential.astype(np.float32),
        # Thermal
        "temperature_K": T.astype(np.float32),
        # Species
        "H2_molfrac": H2.astype(np.float32),
        "H2O_molfrac": H2O.astype(np.float32),
        # Mechanical
        "von_mises_stress_Pa": sigma_von_mises.astype(np.float32),
        "strain_exx": exx.astype(np.float32),
        "strain_eyy": eyy.astype(np.float32),
        "strain_ezz": ezz.astype(np.float32),
        "strain_exy": exy.astype(np.float32),
        "strain_exz": exz.astype(np.float32),
        "strain_eyz": eyz.astype(np.float32),
        "displacement_ux_m": UX.astype(np.float32),
        "displacement_uy_m": UY.astype(np.float32),
        "displacement_uz_m": UZ.astype(np.float32),
        # Coordinates (for reference consumers)
        "coord_x_m": np.broadcast_to(x[np.newaxis, np.newaxis, :], (nz, ny, nx)).astype(np.float32),
        "coord_y_m": np.broadcast_to(y[np.newaxis, :, np.newaxis], (nz, ny, nx)).astype(np.float32),
        "coord_z_m": np.broadcast_to(z[:, np.newaxis, np.newaxis], (nz, ny, nx)).astype(np.float32),
        # Layer index map
        "layer_index": np.broadcast_to(layers[:, np.newaxis, np.newaxis], (nz, ny, nx)).astype(np.int16),
    }

    return fields
