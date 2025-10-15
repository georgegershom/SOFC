from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GridConfig:
    # Low-fidelity
    lf_t_points: int = 128  # 1D axial points for T

    # Mid-fidelity (2D)
    mf_nx: int = 48
    mf_ny: int = 48

    # High-fidelity (3D)
    hf_nx: int = 48
    hf_ny: int = 48
    hf_nz: int = 16


@dataclass(frozen=True)
class GeneratorConfig:
    # Counts per phase (MF/HF must be <= LF)
    num_lf: int = 200
    num_mf: int = 40
    num_hf: int = 8

    # Grids
    grid: GridConfig = GridConfig()

    # Random seeds
    seed: int | None = 1234

    # Output directory
    out_dir: Path = Path("/workspace/data/sofc_mf_dataset")

    # Physical constants (simplified)
    R: float = 8.314  # J/(mol*K)
    F: float = 96485.3329  # C/mol

    # Reference values
    T_ref: float = 1023.15  # K (~750 C)
    E_ref: float = 150e9  # Pa (effective stack stiffness)
    alpha_th_ref: float = 10e-6  # 1/K (effective CTE)

    # Model parameters
    asr_ref: float = 0.3  # Ohm*cm^2, area-specific resistance at T_ref
    asr_temp_coeff: float = -0.6  # fractional per 100K change
    i0_ref: float = 1e-3  # A/cm^2, exchange current density at ref

    # Creep parameters (toy model)
    creep_A: float = 1e-15
    creep_m: float = 1.2
    creep_Q: float = 3.0e5  # J/mol

    # Damage thresholds
    sigma_vm_thresh: float = 60e6  # Pa
    Gc_interface: float = 500.0  # J/m^2

    # Life model
    weibull_k: float = 2.0
    weibull_lambda_hours: float = 10000.0
