from __future__ import annotations
import numpy as np
from ..config import GeneratorConfig
from ..sampling import InputSample
from .utils import random_field, smooth_nd, clamp


def species_fields(sample: InputSample, cfg: GeneratorConfig, shape2d: tuple[int, int], shape3d: tuple[int, int, int], rng: np.random.Generator) -> dict:
    nx, ny = shape2d
    Nx, Ny, Nz = shape3d

    X2, Y2 = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny), indexing="ij")
    # Anode channel along +x: H2 consumed, H2O produced
    h2_2d = sample.anode_inlet_h2_fraction * (1.0 - 0.6 * X2)
    h2o_2d = (1.0 - sample.anode_inlet_h2_fraction) * (0.4 + 0.6 * X2)

    X3, Y3, Z3 = np.meshgrid(
        np.linspace(0, 1, Nx), np.linspace(0, 1, Ny), np.linspace(0, 1, Nz), indexing="ij"
    )
    h2_3d = sample.anode_inlet_h2_fraction * (1.0 - 0.65 * X3)
    h2o_3d = (1.0 - sample.anode_inlet_h2_fraction) * (0.35 + 0.65 * X3)

    # Cathode: O2 drops along flow
    o2_2d = sample.cathode_inlet_o2_fraction * (1.0 - 0.5 * (X2))
    o2_3d = sample.cathode_inlet_o2_fraction * (1.0 - 0.55 * (X3))

    # Add mild heterogeneity
    for A in (h2_2d, h2o_2d, o2_2d):
        A += 0.01 * random_field(A.shape, rng, corr_iters=8, alpha=0.6)
        A[:] = clamp(A, 1e-3, 0.999)
    for A in (h2_3d, h2o_3d, o2_3d):
        A += 0.01 * random_field(A.shape, rng, corr_iters=6, alpha=0.6)
        A[:] = clamp(A, 1e-3, 0.999)

    return {
        "h2_2d": h2_2d.astype(np.float32),
        "h2o_2d": h2o_2d.astype(np.float32),
        "o2_2d": o2_2d.astype(np.float32),
        "h2_3d": h2_3d.astype(np.float32),
        "h2o_3d": h2o_3d.astype(np.float32),
        "o2_3d": o2_3d.astype(np.float32),
    }


def overpotentials(sample: InputSample, cfg: GeneratorConfig, i2: np.ndarray, i3: np.ndarray, T2: np.ndarray, T3: np.ndarray, species: dict) -> dict:
    R, F = cfg.R, cfg.F

    # Exchange current density i0 depends on T and reactant fraction
    def i0_from_T_c(T: np.ndarray, c: np.ndarray) -> np.ndarray:
        # Arrhenius-like boost with T, proportional to concentration
        return cfg.i0_ref * (c + 1e-3) * np.exp(0.5 * (T - cfg.T_ref) / 100.0)

    # 2D
    i0_2d = i0_from_T_c(T2, species["h2_2d"])  # anode kinetics proxy
    eta_act_2d = (R * T2) / (2.0 * F) * np.log((i2 + 1e-6) / (i0_2d + 1e-6))

    asr_2d = np.maximum(1e-4, cfg.asr_ref * (1.0 + cfg.asr_temp_coeff * (T2 - cfg.T_ref) / 100.0))
    eta_ohm_2d = i2 * asr_2d

    i_lim_2d = 2.0 * i0_2d + 0.8  # A/cm^2
    i_ratio_2d = np.clip(i2 / (i_lim_2d + 1e-6), 1e-6, 0.99)
    eta_conc_2d = -(R * T2) / (2.0 * F) * np.log(1.0 - i_ratio_2d)

    # 3D
    i0_3d = i0_from_T_c(T3, species["h2_3d"])  # anode proxy
    eta_act_3d = (R * T3) / (2.0 * F) * np.log((i3 + 1e-6) / (i0_3d + 1e-6))

    asr_3d = np.maximum(1e-4, cfg.asr_ref * (1.0 + cfg.asr_temp_coeff * (T3 - cfg.T_ref) / 100.0))
    eta_ohm_3d = i3 * asr_3d

    i_lim_3d = 2.0 * i0_3d + 0.8
    i_ratio_3d = np.clip(i3 / (i_lim_3d + 1e-6), 1e-6, 0.99)
    eta_conc_3d = -(R * T3) / (2.0 * F) * np.log(1.0 - i_ratio_3d)

    return {
        "eta_act_2d": eta_act_2d.astype(np.float32),
        "eta_ohm_2d": eta_ohm_2d.astype(np.float32),
        "eta_conc_2d": eta_conc_2d.astype(np.float32),
        "eta_act_3d": eta_act_3d.astype(np.float32),
        "eta_ohm_3d": eta_ohm_3d.astype(np.float32),
        "eta_conc_3d": eta_conc_3d.astype(np.float32),
    }
