from __future__ import annotations
import numpy as np
from ..config import GeneratorConfig
from ..sampling import InputSample
from .utils import random_field, smooth_nd, clamp


def area_specific_resistance(T: np.ndarray, cfg: GeneratorConfig) -> np.ndarray:
    # Simple temp dependence: ASR(T) = asr_ref * (1 + asr_temp_coeff * dT/100K)
    dT = T - cfg.T_ref
    return cfg.asr_ref * (1.0 + cfg.asr_temp_coeff * (dT / 100.0))


def synthesize_temperature_fields(sample: InputSample, cfg: GeneratorConfig, rng: np.random.Generator) -> dict:
    # LF: 1D axial profile (length discretization = lf_t_points)
    n1 = cfg.grid.lf_t_points
    x = np.linspace(0.0, 1.0, n1)

    base_T = sample.inlet_temperature_K
    heat_gen = 120.0 * sample.average_current_Apcm2  # K scale per A/cm^2 (toy)
    axial_drop = 25.0 * (sample.fuel_utilization - 0.6)  # more utilization -> more drop

    t1d = base_T + heat_gen * (1 - np.exp(-3.0 * x)) - axial_drop * x

    # Add correlated perturbations
    noise1d = random_field((n1,), rng, corr_iters=12, alpha=0.6)
    t1d = t1d + 3.0 * noise1d
    t1d = smooth_nd(t1d, iters=2, alpha=0.5)

    # MF: 2D coarse grid
    nx, ny = cfg.grid.mf_nx, cfg.grid.mf_ny
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    T2 = base_T + heat_gen * (1 - np.exp(-3.0 * X)) - axial_drop * X
    edge_cool = 10.0 * (X * (1 - X) + Y * (1 - Y))
    T2 = T2 - edge_cool
    T2 += 2.0 * random_field((nx, ny), rng, corr_iters=10, alpha=0.55)
    T2 = smooth_nd(T2, iters=3, alpha=0.5)

    # HF: 3D fine grid
    Nx, Ny, Nz = cfg.grid.hf_nx, cfg.grid.hf_ny, cfg.grid.hf_nz
    X3, Y3, Z3 = np.meshgrid(
        np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), np.linspace(0, 1, Nz), indexing="ij"
    )
    T3 = base_T + heat_gen * (1 - np.exp(-3.0 * X3)) - axial_drop * X3
    # Through-thickness gradient (electrolyte thinner -> more gradient)
    z_grad = (sample.electrolyte_thickness_um - 5.0) / 15.0
    T3 = T3 - 6.0 * z_grad * (Z3 - 0.5)
    T3 += 1.5 * random_field((Nx, Ny, Nz), rng, corr_iters=8, alpha=0.55)
    T3 = smooth_nd(T3, iters=4, alpha=0.55)

    return {"lf_1d": t1d.astype(np.float32), "mf_2d": T2.astype(np.float32), "hf_3d": T3.astype(np.float32)}


def synthesize_current_fields(sample: InputSample, cfg: GeneratorConfig, T2: np.ndarray, T3: np.ndarray, rng: np.random.Generator) -> dict:
    # LF constant
    i_lf = np.full((1,), sample.average_current_Apcm2, dtype=np.float32)

    # MF 2D: mildly higher near inlet, reduced near edges
    nx, ny = T2.shape
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    i2 = sample.average_current_Apcm2 * (1.0 + 0.1 * np.exp(-4.0 * X))
    i2 *= 1.0 - 0.08 * (X * (1 - X) + Y * (1 - Y))
    i2 += 0.02 * random_field((nx, ny), rng)
    i2 = smooth_nd(i2, iters=2, alpha=0.5)

    # HF 3D: similar, plus mild through-thickness modulation
    Nx, Ny, Nz = T3.shape
    X3, Y3, Z3 = np.meshgrid(
        np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), np.linspace(0, 1, Nz), indexing="ij"
    )
    i3 = sample.average_current_Apcm2 * (1.0 + 0.12 * np.exp(-4.2 * X3))
    i3 *= 1.0 - 0.06 * (X3 * (1 - X3) + Y3 * (1 - Y3))
    i3 *= 1.0 + 0.02 * (Z3 - 0.5)
    i3 += 0.02 * random_field((Nx, Ny, Nz), rng)
    i3 = smooth_nd(i3, iters=2, alpha=0.5)

    return {"lf_scalar": i_lf, "mf_2d": i2.astype(np.float32), "hf_3d": i3.astype(np.float32)}
