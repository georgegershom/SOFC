from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

# Microstructural descriptors at 20°C used to inform macro properties
@dataclass
class Microstructure:
    mix_id: str
    rubber_volume_fraction: float  # volume fraction of rubber replacement [0-1]
    rubber_particle_size_mm: float  # characteristic size
    aggregate_volume_fraction: float  # non-rubber aggregate fraction by volume
    binder_volume_fraction: float  # paste fraction by volume
    initial_porosity: float  # connected porosity at 20°C
    water_mass_fraction: float  # mass fraction of evaporable water in composite
    matrix_elastic_modulus_pa: float  # cement paste effective E at 20°C
    aggregate_elastic_modulus_pa: float  # mineral aggregate E
    rubber_elastic_modulus_pa: float  # rubber E at 20°C
    matrix_thermal_conductivity_wmk: float
    aggregate_thermal_conductivity_wmk: float
    rubber_thermal_conductivity_wmk: float
    matrix_density_kgm3: float
    aggregate_density_kgm3: float
    rubber_density_kgm3: float
    matrix_cte_1k: float  # coefficient of thermal expansion
    aggregate_cte_1k: float
    rubber_cte_1k: float
    base_permeability_m2: float
    base_biot_coefficient: float
    base_poisson_ratio: float


def build_microstructure(mix_id: str) -> Microstructure:
    # Defaults for HPC control mix
    rubber_vf = 0.0
    size_mm = 1.0
    # Rubber volume fractions by mix id
    if mix_id == "C":
        rubber_vf = 0.0
        size_mm = 0.0
    elif mix_id == "R5S":
        rubber_vf = 0.05
        size_mm = 1.0
    elif mix_id == "R10S":
        rubber_vf = 0.10
        size_mm = 1.0
    elif mix_id == "R15S":
        rubber_vf = 0.15
        size_mm = 1.0
    elif mix_id == "R20S":
        rubber_vf = 0.20
        size_mm = 1.0
    elif mix_id == "R10L":
        rubber_vf = 0.10
        size_mm = 5.0
    else:
        raise ValueError(f"Unknown mix_id: {mix_id}")

    # Assume total solid volume fraction ~ 0.9; paste ~ 0.5 of total volume for control
    binder_vf_control = 0.50
    aggregate_vf_control = 0.40

    # Rubber replaces fine aggregates by volume
    aggregate_vf = max(0.0, aggregate_vf_control - rubber_vf)
    binder_vf = binder_vf_control

    initial_porosity = 0.10 + 0.05 * rubber_vf  # rubber increases connected porosity
    water_mass_fraction = 0.05 + 0.01 * rubber_vf  # evaporable water fraction

    # Constituent properties at 20°C
    matrix_E = 22e9  # Pa, effective hardened paste
    aggregate_E = 70e9  # granite/basalt range
    rubber_E = 15e6  # soft rubber crumbs

    matrix_k = 1.2  # W/mK
    aggregate_k = 3.0  # W/mK
    rubber_k = 0.2  # W/mK

    matrix_rho = 2100.0
    aggregate_rho = 2650.0
    rubber_rho = 1100.0

    matrix_cte = 11e-6
    aggregate_cte = 8.5e-6
    rubber_cte = 80e-6

    base_perm = 1e-18 * (1 + 8 * rubber_vf)  # higher with rubber
    base_biot = min(0.95, 0.65 + 0.25 * initial_porosity + 0.10 * rubber_vf)
    base_nu = 0.20 + 0.02 * rubber_vf

    return Microstructure(
        mix_id=mix_id,
        rubber_volume_fraction=rubber_vf,
        rubber_particle_size_mm=size_mm,
        aggregate_volume_fraction=aggregate_vf,
        binder_volume_fraction=binder_vf,
        initial_porosity=initial_porosity,
        water_mass_fraction=water_mass_fraction,
        matrix_elastic_modulus_pa=matrix_E,
        aggregate_elastic_modulus_pa=aggregate_E,
        rubber_elastic_modulus_pa=rubber_E,
        matrix_thermal_conductivity_wmk=matrix_k,
        aggregate_thermal_conductivity_wmk=aggregate_k,
        rubber_thermal_conductivity_wmk=rubber_k,
        matrix_density_kgm3=matrix_rho,
        aggregate_density_kgm3=aggregate_rho,
        rubber_density_kgm3=rubber_rho,
        matrix_cte_1k=matrix_cte,
        aggregate_cte_1k=aggregate_cte,
        rubber_cte_1k=rubber_cte,
        base_permeability_m2=base_perm,
        base_biot_coefficient=base_biot,
        base_poisson_ratio=base_nu,
    )
