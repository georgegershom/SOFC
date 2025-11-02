#!/usr/bin/env python3
"""Generate a synthetic material property and constitutive model dataset
for solid oxide fuel cell (SOFC) layers inspired by the article:

    "A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the
    Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells
    through Targeted Creep Activation".

The script fabricates temperature-dependent thermo-physical,
constitutive, and elastic datasets for the anode, electrolyte,
cathode, and interconnect layers. It is designed to provide
plausible, literature-informed synthetic data suitable for
model development and testing when experimental measurements
are unavailable.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DATASET_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = DATASET_ROOT / "data"


@dataclass(frozen=True)
class Material:
    name: str
    system: str
    notes: str


MATERIALS: Dict[str, Material] = {
    "anode": Material(
        name="NiO-YSZ",
        system="NiO-8YSZ composite",
        notes=(
            "Typical tape-cast anode with graphite porogen; parameters tuned to "
            "match shrinkage onset near 1000 degC and significant creep below full densification."
        ),
    ),
    "electrolyte": Material(
        name="8YSZ",
        system="8 mol% yttria-stabilized zirconia",
        notes=(
            "High-density electrolyte tape; shrinkage accelerates beyond 1200 degC; "
            "creep dominated by grain boundary sliding in the sintering range."
        ),
    ),
    "cathode": Material(
        name="LSCF",
        system="La0.6Sr0.4Co0.2Fe0.8O3-delta",
        notes=(
            "Infiltrated perovskite cathode with finer porosity; higher CTE and "
            "lower sintering activation threshold than the electrolyte."
        ),
    ),
    "interconnect": Material(
        name="Crofer 22 APU",
        system="Fe-Cr ferritic steel",
        notes=(
            "Machined interconnect ferritic steel; treated as fully dense with "
            "creep governed by dislocation climb."
        ),
    ),
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable]):
    with path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def generate_green_state_properties() -> None:
    rows = [
        (
            key,
            mat.name,
            mat.system,
            mat.notes,
            density,
            porosity,
            binder,
            pore_former,
        )
        for key, mat, density, porosity, binder, pore_former in [
            (
                "anode",
                MATERIALS["anode"],
                2.45,
                0.38,
                0.08,
                0.12,
            ),
            (
                "electrolyte",
                MATERIALS["electrolyte"],
                2.90,
                0.28,
                0.06,
                0.05,
            ),
            (
                "cathode",
                MATERIALS["cathode"],
                2.65,
                0.42,
                0.09,
                0.10,
            ),
            (
                "interconnect",
                MATERIALS["interconnect"],
                6.20,
                0.05,
                0.02,
                0.00,
            ),
        ]
    ]

    write_csv(
        OUTPUT_DIR / "green_state_properties.csv",
        (
            "layer_id",
            "material_name",
            "system",
            "notes",
            "green_density_g_cm3",
            "green_porosity_fraction",
            "binder_fraction",
            "porogen_fraction",
        ),
        rows,
    )


def shrinkage_fraction(temperature_c: float, time_min: float, *, onset: float, peak: float, max_strain: float) -> float:
    activation = 1.0 / (1.0 + math.exp(-(temperature_c - onset) / 40.0))
    k_base = 2.5e-3
    k_temp = k_base * math.exp(0.0065 * (temperature_c - onset))
    shrinkage = max_strain * activation * (1.0 - math.exp(-k_temp * time_min))
    return round(min(max(shrinkage, 0.0), max_strain * 1.05), 6)


def generate_sintering_kinetics() -> None:
    temperatures = [850, 950, 1050, 1150, 1250, 1350]
    times = [0, 15, 30, 60, 90, 120, 180]

    material_params = {
        "anode": dict(onset=950.0, peak=1250.0, max_strain=0.17),
        "electrolyte": dict(onset=1025.0, peak=1350.0, max_strain=0.13),
        "cathode": dict(onset=900.0, peak=1200.0, max_strain=0.19),
        "interconnect": dict(onset=1150.0, peak=1350.0, max_strain=0.04),
    }

    rows: List[Tuple] = []
    for layer_id, params in material_params.items():
        for temp in temperatures:
            for time in times:
                strain = shrinkage_fraction(temp, time, **params)
                rows.append((layer_id, temp, time, strain))

    write_csv(
        OUTPUT_DIR / "sintering_kinetics_single_layer.csv",
        ("layer_id", "temperature_C", "time_min", "shrinkage_strain"),
        rows,
    )

    bilayer_pairs = [
        ("anode", "electrolyte", 0.5),
        ("cathode", "electrolyte", 0.55),
        ("anode", "cathode", 0.48),
    ]

    bilayer_rows: List[Tuple] = []
    for a, b, coupling in bilayer_pairs:
        params_a = material_params[a]
        params_b = material_params[b]
        for temp in [900, 1050, 1200, 1300]:
            for time in [0, 15, 30, 60, 90, 120]:
                strain_a = shrinkage_fraction(temp, time, **params_a)
                strain_b = shrinkage_fraction(temp, time, **params_b)
                compat_strain = coupling * strain_a + (1.0 - coupling) * strain_b
                mismatch = strain_a - strain_b
                bilayer_rows.append((
                    f"{a}-{b}",
                    temp,
                    time,
                    round(compat_strain, 6),
                    round(mismatch, 6),
                ))

    write_csv(
        OUTPUT_DIR / "sintering_kinetics_bilayer.csv",
        (
            "bilayer_id",
            "temperature_C",
            "time_min",
            "compatible_shrinkage_strain",
            "strain_mismatch",
        ),
        bilayer_rows,
    )


def generate_cte_data() -> None:
    rows: List[Tuple] = []
    for layer_id in MATERIALS:
        base = {
            "anode": 11.0,
            "electrolyte": 10.3,
            "cathode": 12.4,
            "interconnect": 13.2,
        }[layer_id]
        slope = {
            "anode": 1.3,
            "electrolyte": 0.8,
            "cathode": 1.6,
            "interconnect": 0.4,
        }[layer_id]
        for temperature in range(25, 1401, 75):
            delta = (temperature - 25.0) / 1000.0
            cte = base + slope * delta
            rows.append((layer_id, temperature, round(cte, 4)))

    write_csv(
        OUTPUT_DIR / "coefficient_thermal_expansion.csv",
        ("layer_id", "temperature_C", "cte_1e6_per_C"),
        rows,
    )


def norton_law_strain_rate(A: float, n: float, Q: float, stress_mpa: float, temperature_c: float) -> float:
    R = 8.314  # J/mol/K
    temperature_k = temperature_c + 273.15
    stress = stress_mpa * 1e6
    return A * (stress ** n) * math.exp(-Q / (R * temperature_k))


def generate_creep_data() -> None:
    parameters = {
        "anode": {
            "green": dict(A=2.8e-7, n=1.65, Q=2.05e5),
            "sintering": dict(A=4.2e-8, n=1.85, Q=2.35e5),
        },
        "electrolyte": {
            "green": dict(A=1.7e-7, n=1.70, Q=2.25e5),
            "sintering": dict(A=2.1e-8, n=2.05, Q=2.55e5),
        },
        "cathode": {
            "green": dict(A=3.5e-7, n=1.55, Q=1.95e5),
            "sintering": dict(A=6.0e-8, n=1.75, Q=2.20e5),
        },
        "interconnect": {
            "green": dict(A=8.0e-9, n=4.5, Q=2.85e5),
            "sintering": dict(A=4.0e-9, n=4.8, Q=3.10e5),
        },
    }

    stresses = [0.4, 0.6, 0.8, 1.0, 1.2]
    temperatures = [800, 900, 1000, 1100, 1200]
    hold_hours = [0.5, 1.0, 2.0]

    rows: List[Tuple] = []
    for layer_id, state_params in parameters.items():
        for state, params in state_params.items():
            for temperature in temperatures:
                for stress in stresses:
                    strain_rate = norton_law_strain_rate(
                        params["A"], params["n"], params["Q"], stress, temperature
                    )
                    for hold in hold_hours:
                        strain = strain_rate * hold * 3600.0
                        rows.append(
                            (
                                layer_id,
                                state,
                                temperature,
                                stress,
                                hold,
                                round(strain_rate, 12),
                                round(strain, 8),
                            )
                        )

    write_csv(
        OUTPUT_DIR / "creep_constant_stress_tests.csv",
        (
            "layer_id",
            "material_state",
            "temperature_C",
            "stress_MPa",
            "hold_time_h",
            "strain_rate_per_s",
            "accumulated_strain",
        ),
        rows,
    )

    param_rows = []
    for layer_id, state_params in parameters.items():
        for state, params in state_params.items():
            param_rows.append(
                (
                    layer_id,
                    state,
                    params["A"],
                    params["n"],
                    params["Q"],
                )
            )

    write_csv(
        OUTPUT_DIR / "creep_norton_parameters.csv",
        ("layer_id", "material_state", "A_prefactor", "n_stress_exponent", "Q_J_per_mol"),
        param_rows,
    )


def generate_elastic_properties() -> None:
    base_modulus = {
        "anode": 180.0,
        "electrolyte": 210.0,
        "cathode": 150.0,
        "interconnect": 220.0,
    }
    base_poisson = {
        "anode": 0.28,
        "electrolyte": 0.30,
        "cathode": 0.26,
        "interconnect": 0.31,
    }

    temp_points = [25, 400, 600, 800, 1000, 1200]
    rel_densities = [0.55, 0.7, 0.85, 0.95, 1.0]

    rows: List[Tuple] = []
    for layer_id in MATERIALS:
        E0 = base_modulus[layer_id]
        nu0 = base_poisson[layer_id]
        for rho in rel_densities:
            for temp in temp_points:
                density_factor = rho ** 2.1
                temp_factor = 1.0 - 0.18 * (temp / 1400.0)
                youngs = max(E0 * density_factor * temp_factor, 5.0)
                nu = nu0 + 0.04 * (1.0 - rho) + 0.015 * (temp / 1400.0)
                rows.append(
                    (
                        layer_id,
                        round(rho, 3),
                        temp,
                        round(youngs, 3),
                        round(min(max(nu, 0.18), 0.37), 3),
                    )
                )

    write_csv(
        OUTPUT_DIR / "elastic_properties.csv",
        (
            "layer_id",
            "relative_density",
            "temperature_C",
            "youngs_modulus_GPa",
            "poisson_ratio",
        ),
        rows,
    )


def generate_metadata() -> None:
    metadata = {
        "dataset_name": "SOFC Sintering Creep Activation Synthetic Dataset",
        "version": "1.0.0",
        "generated_by": "generate_material_dataset.py",
        "article_reference": {
            "title": "A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation",
            "journal": "Advanced Energy Materials (hypothetical context)",
            "notes": (
                "Parameter ranges and coupling assumptions informed by the methodology "
                "described in the reference article. Values are synthetic and intended "
                "for simulation and algorithm development."
            ),
        },
        "materials": {
            key: {
                "name": mat.name,
                "system": mat.system,
                "notes": mat.notes,
            }
            for key, mat in MATERIALS.items()
        },
        "files": {
            "green_state_properties": "green_state_properties.csv",
            "sintering_single_layer": "sintering_kinetics_single_layer.csv",
            "sintering_bilayer": "sintering_kinetics_bilayer.csv",
            "cte": "coefficient_thermal_expansion.csv",
            "creep_tests": "creep_constant_stress_tests.csv",
            "creep_parameters": "creep_norton_parameters.csv",
            "elastic": "elastic_properties.csv",
        },
        "units": {
            "density": "g/cm^3",
            "porosity": "fraction (0-1)",
            "binder": "mass fraction",
            "temperature": "degC",
            "time": "minutes or hours as specified",
            "strain": "engineering strain",
            "cte": "1e-6 / degC",
            "youngs_modulus": "GPa",
            "stress": "MPa",
            "activation_energy": "J/mol",
        },
        "license": "CC BY 4.0",
        "generated_on": None,
    }

    metadata_path = OUTPUT_DIR / "dataset_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    ensure_output_dir()
    generate_green_state_properties()
    generate_sintering_kinetics()
    generate_cte_data()
    generate_creep_data()
    generate_elastic_properties()
    generate_metadata()


if __name__ == "__main__":
    main()
