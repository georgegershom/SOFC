"""Generate a synthetic material property and constitutive dataset for SOFC layers.

The dataset is tailored for thermo-mechanical FEM workflows that require
temperature-dependent properties, sintering kinetics, creep data, and elastic
behavior across the anode, electrolyte, cathode, and interconnect layers.

All data are synthetically generated but follow physically plausible trends
anchored to the literature theme:

  A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time
  Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted
  Creep Activation.

The script produces a collection of CSV files alongside a consolidated JSON
artifact under the dataset root directory.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


R_GAS = 8.314462618  # J/(mol?K)
TEMPERATURE_PROFILE = [25, 200, 400, 600, 750, 900, 1000, 1100, 1200, 1300, 1350, 1400]
SINTERING_TEMPERATURES = [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
TIME_POINTS_MIN = [0, 5, 15, 30, 60, 90, 120, 180]
STRESS_LEVELS_MPA = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
CREPTEMP_GREEN = [650, 700, 750, 800, 850, 900, 950]
CREPTEMP_SINTERING = [850, 900, 950, 1000, 1050, 1100, 1150, 1200]
DATASET_VERSION = "1.0.0"


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


@dataclass(frozen=True)
class DensificationParams:
    theoretical_density: float  # g/cm^3
    min_rel: float
    max_rel: float
    midpoint: float  # ?C
    steepness: float
    binder_initial: float  # mass fraction
    binder_decay: float  # 1/?C
    binder_floor: float


@dataclass(frozen=True)
class SinteringParams:
    max_shrinkage: float
    temp_mid: float
    temp_k: float
    time_k: float
    ramp_rate: float  # ?C/min


@dataclass(frozen=True)
class CreepStateParams:
    A: float  # pre-exponential, 1/(s?MPa^n)
    n: float  # stress exponent
    Q: float  # activation energy, J/mol


@dataclass(frozen=True)
class ElasticParams:
    E_room_dense: float  # GPa at 25?C and full density
    E_temp_coeff: float  # 1/?C
    porosity_exponent: float
    nu_dense: float
    nu_porosity_coeff: float
    nu_temp_coeff: float


@dataclass(frozen=True)
class CTEParams:
    base: float  # 1/?C
    linear: float  # 1/(?C^2)
    quadratic: float  # 1/(?C^3)
    scatter: float  # relative scatter for synthetic noise


@dataclass(frozen=True)
class MaterialDefinition:
    key: str
    name: str
    densification: DensificationParams
    sintering: SinteringParams
    creep_states: Dict[str, CreepStateParams]
    elastic: ElasticParams
    cte: CTEParams


MATERIALS: Dict[str, MaterialDefinition] = {
    "anode": MaterialDefinition(
        key="anode",
        name="NiO-YSZ Anode",
        densification=DensificationParams(
            theoretical_density=6.72,
            min_rel=0.58,
            max_rel=0.965,
            midpoint=1180.0,
            steepness=0.0075,
            binder_initial=0.085,
            binder_decay=0.004,
            binder_floor=0.0004,
        ),
        sintering=SinteringParams(
            max_shrinkage=0.162,
            temp_mid=1185.0,
            temp_k=0.0078,
            time_k=0.0195,
            ramp_rate=4.5,
        ),
        creep_states={
            "green": CreepStateParams(A=2.8e3, n=1.62, Q=2.25e5),
            "sintering": CreepStateParams(A=2.6e5, n=1.84, Q=2.58e5),
        },
        elastic=ElasticParams(
            E_room_dense=145.0,
            E_temp_coeff=2.6e-4,
            porosity_exponent=2.45,
            nu_dense=0.31,
            nu_porosity_coeff=0.11,
            nu_temp_coeff=3.2e-5,
        ),
        cte=CTEParams(
            base=12.1e-6,
            linear=1.7e-9,
            quadratic=0.0,
            scatter=0.015,
        ),
    ),
    "electrolyte": MaterialDefinition(
        key="electrolyte",
        name="8YSZ Electrolyte",
        densification=DensificationParams(
            theoretical_density=6.05,
            min_rel=0.62,
            max_rel=0.992,
            midpoint=1265.0,
            steepness=0.0105,
            binder_initial=0.061,
            binder_decay=0.0038,
            binder_floor=0.0003,
        ),
        sintering=SinteringParams(
            max_shrinkage=0.142,
            temp_mid=1240.0,
            temp_k=0.0091,
            time_k=0.021,
            ramp_rate=3.8,
        ),
        creep_states={
            "green": CreepStateParams(A=1.1e4, n=1.54, Q=2.32e5),
            "sintering": CreepStateParams(A=4.5e5, n=1.73, Q=2.70e5),
        },
        elastic=ElasticParams(
            E_room_dense=205.0,
            E_temp_coeff=1.9e-4,
            porosity_exponent=3.05,
            nu_dense=0.30,
            nu_porosity_coeff=0.085,
            nu_temp_coeff=2.1e-5,
        ),
        cte=CTEParams(
            base=10.3e-6,
            linear=1.25e-9,
            quadratic=2.4e-12,
            scatter=0.012,
        ),
    ),
    "cathode": MaterialDefinition(
        key="cathode",
        name="LSCF Cathode",
        densification=DensificationParams(
            theoretical_density=6.35,
            min_rel=0.55,
            max_rel=0.935,
            midpoint=1130.0,
            steepness=0.0068,
            binder_initial=0.118,
            binder_decay=0.0045,
            binder_floor=0.0005,
        ),
        sintering=SinteringParams(
            max_shrinkage=0.184,
            temp_mid=1145.0,
            temp_k=0.0071,
            time_k=0.018,
            ramp_rate=5.0,
        ),
        creep_states={
            "green": CreepStateParams(A=6.2e3, n=1.57, Q=2.05e5),
            "sintering": CreepStateParams(A=3.6e5, n=1.72, Q=2.38e5),
        },
        elastic=ElasticParams(
            E_room_dense=125.0,
            E_temp_coeff=3.1e-4,
            porosity_exponent=2.28,
            nu_dense=0.28,
            nu_porosity_coeff=0.095,
            nu_temp_coeff=3.8e-5,
        ),
        cte=CTEParams(
            base=14.2e-6,
            linear=2.1e-9,
            quadratic=3.5e-12,
            scatter=0.018,
        ),
    ),
    "interconnect": MaterialDefinition(
        key="interconnect",
        name="Crofer 22 APU Interconnect",
        densification=DensificationParams(
            theoretical_density=7.55,
            min_rel=0.70,
            max_rel=0.985,
            midpoint=1040.0,
            steepness=0.0092,
            binder_initial=0.024,
            binder_decay=0.0026,
            binder_floor=0.0002,
        ),
        sintering=SinteringParams(
            max_shrinkage=0.052,
            temp_mid=1035.0,
            temp_k=0.0084,
            time_k=0.015,
            ramp_rate=6.0,
        ),
        creep_states={
            "green": CreepStateParams(A=1.8e3, n=1.43, Q=1.95e5),
            "sintering": CreepStateParams(A=9.2e4, n=1.61, Q=2.21e5),
        },
        elastic=ElasticParams(
            E_room_dense=215.0,
            E_temp_coeff=1.4e-4,
            porosity_exponent=2.65,
            nu_dense=0.29,
            nu_porosity_coeff=0.07,
            nu_temp_coeff=1.5e-5,
        ),
        cte=CTEParams(
            base=12.3e-6,
            linear=1.05e-9,
            quadratic=1.8e-12,
            scatter=0.01,
        ),
    ),
}


COUPLING_FACTORS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("anode", "electrolyte"): {"coupling_factor": 0.35, "stress_scale": 145.0},
    ("cathode", "electrolyte"): {"coupling_factor": 0.28, "stress_scale": 118.0},
    ("electrolyte", "interconnect"): {"coupling_factor": 0.22, "stress_scale": 132.0},
}


def logistic_relative_density(temperature_c: float, params: DensificationParams) -> float:
    rel = params.min_rel + (params.max_rel - params.min_rel) / (
        1.0 + math.exp(-params.steepness * (temperature_c - params.midpoint))
    )
    return clamp(rel, params.min_rel, params.max_rel)


def binder_fraction(temperature_c: float, params: DensificationParams) -> float:
    if temperature_c <= 25.0:
        return params.binder_initial
    decay = math.exp(-params.binder_decay * (temperature_c - 25.0))
    value = params.binder_initial * decay
    return max(params.binder_floor, value)


def shrinkage_fraction(temperature_c: float, time_min: float, params: SinteringParams) -> float:
    temp_factor = 1.0 / (1.0 + math.exp(-params.temp_k * (temperature_c - params.temp_mid)))
    effective_max = params.max_shrinkage * temp_factor
    if time_min <= 0.0:
        return 0.0
    time_factor = 1.0 - math.exp(-params.time_k * time_min)
    shrink = effective_max * time_factor
    return clamp(shrink, 0.0, params.max_shrinkage)


def norton_creep_rate(
    temperature_c: float,
    stress_mpa: float,
    state: str,
    material: MaterialDefinition,
) -> float:
    params = material.creep_states[state]
    temperature_k = temperature_c + 273.15
    base_rate = params.A * (stress_mpa ** params.n) * math.exp(-params.Q / (R_GAS * temperature_k))
    scatter = 1.0 + random.gauss(0.0, 0.035)
    return max(1e-9, base_rate * scatter)


def elastic_properties(
    temperature_c: float,
    relative_density: float,
    params: ElasticParams,
) -> Tuple[float, float]:
    temperature_shift = max(0.0, temperature_c - 25.0)
    dense_modulus = params.E_room_dense * (1.0 - params.E_temp_coeff * temperature_shift)
    dense_modulus = max(0.1 * params.E_room_dense, dense_modulus)
    youngs = dense_modulus * (relative_density ** params.porosity_exponent)
    youngs = max(1.0, youngs)

    porosity_deficit = max(0.0, 1.0 - relative_density)
    nu = (
        params.nu_dense
        - params.nu_porosity_coeff * porosity_deficit
        + params.nu_temp_coeff * temperature_shift
    )
    nu = clamp(nu, 0.20, 0.34)
    return youngs, nu


def cte_value(temperature_c: float, params: CTEParams) -> float:
    delta = temperature_c - 25.0
    value = params.base + params.linear * delta + params.quadratic * (delta ** 2)
    scatter_factor = 1.0 + random.gauss(0.0, params.scatter)
    return max(5.0e-6, value * scatter_factor)


def generate_thermo_physical_records(material: MaterialDefinition) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for temp in TEMPERATURE_PROFILE:
        rel = logistic_relative_density(temp, material.densification)
        density = rel * material.densification.theoretical_density
        porosity = clamp(1.0 - rel, 0.0, 0.6)
        binder = binder_fraction(temp, material.densification)
        records.append(
            {
                "layer": material.key,
                "material_name": material.name,
                "temperature_C": round(temp, 2),
                "density_g_cm3": round(density, 4),
                "relative_density": round(rel, 4),
                "porosity": round(porosity, 4),
                "binder_content_fraction": round(binder, 5),
            }
        )
    return records


def generate_single_layer_sintering_records(material: MaterialDefinition) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for temp in SINTERING_TEMPERATURES:
        for time_min in TIME_POINTS_MIN:
            shrink = shrinkage_fraction(temp, time_min, material.sintering)
            shrink = clamp(shrink + random.gauss(0.0, 0.0006), 0.0, material.sintering.max_shrinkage)
            temp_factor = 1.0 / (1.0 + math.exp(-material.sintering.temp_k * (temp - material.sintering.temp_mid)))
            equilibrium = material.sintering.max_shrinkage * temp_factor
            axial_rate = material.sintering.time_k * max(0.0, equilibrium - shrink)
            rel_density = logistic_relative_density(temp, material.densification)
            records.append(
                {
                    "layer": material.key,
                    "temperature_C": temp,
                    "time_min": time_min,
                    "ramp_rate_C_per_min": material.sintering.ramp_rate,
                    "dL_over_L0": round(shrink, 5),
                    "axial_strain_rate_per_min": round(axial_rate, 5),
                    "relative_density": round(rel_density, 4),
                }
            )
    return records


def generate_bilayer_records(
    first: MaterialDefinition,
    second: MaterialDefinition,
    coupling: Dict[str, float],
) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for temp in SINTERING_TEMPERATURES:
        for time_min in TIME_POINTS_MIN:
            shrink_a = shrinkage_fraction(temp, time_min, first.sintering)
            shrink_b = shrinkage_fraction(temp, time_min, second.sintering)
            mismatch = shrink_a - shrink_b
            composite = ((shrink_a + shrink_b) / 2.0) + coupling["coupling_factor"] * mismatch
            composite = clamp(composite, 0.0, max(first.sintering.max_shrinkage, second.sintering.max_shrinkage))
            interfacial_stress = mismatch * coupling["stress_scale"]
            creep_rate_first = norton_creep_rate(temp, 1.0, "sintering", first)
            creep_rate_second = norton_creep_rate(temp, 1.0, "sintering", second)
            records.append(
                {
                    "layer_a": first.key,
                    "layer_b": second.key,
                    "temperature_C": temp,
                    "time_min": time_min,
                    "bilayer_dL_over_L0": round(composite, 5),
                    "shrinkage_mismatch": round(mismatch, 5),
                    "interfacial_stress_MPa": round(interfacial_stress, 3),
                    "layer_a_creep_rate_per_s": round(creep_rate_first, 8),
                    "layer_b_creep_rate_per_s": round(creep_rate_second, 8),
                }
            )
    return records


def generate_creep_records(material: MaterialDefinition, state: str, temperature_grid: Iterable[float]) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for temp in temperature_grid:
        for stress in STRESS_LEVELS_MPA:
            strain_rate = norton_creep_rate(temp, stress, state, material)
            records.append(
                {
                    "layer": material.key,
                    "state": state,
                    "temperature_C": temp,
                    "stress_MPa": stress,
                    "strain_rate_per_s": round(strain_rate, 10),
                }
            )
    return records


def generate_elastic_records(material: MaterialDefinition) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    relative_levels = [0.55, 0.65, 0.75, 0.85, 0.92, 0.97]
    for temp in TEMPERATURE_PROFILE:
        for rel in relative_levels:
            rel_clamped = clamp(rel, material.densification.min_rel, material.densification.max_rel)
            youngs, poisson = elastic_properties(temp, rel_clamped, material.elastic)
            records.append(
                {
                    "layer": material.key,
                    "temperature_C": temp,
                    "relative_density": round(rel_clamped, 4),
                    "youngs_modulus_GPa": round(youngs, 3),
                    "poissons_ratio": round(poisson, 4),
                }
            )
    return records


def generate_cte_records(material: MaterialDefinition) -> List[Dict[str, float]]:
    records: List[Dict[str, float]] = []
    for temp in TEMPERATURE_PROFILE:
        cte = cte_value(temp, material.cte)
        records.append(
            {
                "layer": material.key,
                "temperature_C": temp,
                "cte_per_C": round(cte, 9),
            }
        )
    return records


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    random.seed(2025)

    script_path = Path(__file__).resolve()
    dataset_root = script_path.parents[1]
    output_dir = dataset_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    thermo_records: List[Dict[str, float]] = []
    single_layer_records: List[Dict[str, float]] = []
    bilayer_records: List[Dict[str, float]] = []
    creep_records: List[Dict[str, float]] = []
    elastic_records: List[Dict[str, float]] = []
    cte_records: List[Dict[str, float]] = []
    creep_parameters: List[Dict[str, float]] = []

    for material in MATERIALS.values():
        thermo_records.extend(generate_thermo_physical_records(material))
        single_layer_records.extend(generate_single_layer_sintering_records(material))
        elastic_records.extend(generate_elastic_records(material))
        cte_records.extend(generate_cte_records(material))

        creep_parameters.extend(
            [
                {
                    "layer": material.key,
                    "state": state,
                    "A_pre_exponential": params.A,
                    "stress_exponent_n": params.n,
                    "activation_energy_kJ_per_mol": round(params.Q / 1000.0, 3),
                }
                for state, params in material.creep_states.items()
            ]
        )

        creep_records.extend(generate_creep_records(material, "green", CREPTEMP_GREEN))
        creep_records.extend(generate_creep_records(material, "sintering", CREPTEMP_SINTERING))

    for (first_key, second_key), coupling in COUPLING_FACTORS.items():
        first = MATERIALS[first_key]
        second = MATERIALS[second_key]
        bilayer_records.extend(generate_bilayer_records(first, second, coupling))

    write_csv(
        output_dir / "thermo_physical_properties.csv",
        [
            "layer",
            "material_name",
            "temperature_C",
            "density_g_cm3",
            "relative_density",
            "porosity",
            "binder_content_fraction",
        ],
        thermo_records,
    )

    write_csv(
        output_dir / "sintering_kinetics_single_layer.csv",
        [
            "layer",
            "temperature_C",
            "time_min",
            "ramp_rate_C_per_min",
            "dL_over_L0",
            "axial_strain_rate_per_min",
            "relative_density",
        ],
        single_layer_records,
    )

    write_csv(
        output_dir / "sintering_kinetics_bilayer.csv",
        [
            "layer_a",
            "layer_b",
            "temperature_C",
            "time_min",
            "bilayer_dL_over_L0",
            "shrinkage_mismatch",
            "interfacial_stress_MPa",
            "layer_a_creep_rate_per_s",
            "layer_b_creep_rate_per_s",
        ],
        bilayer_records,
    )

    write_csv(
        output_dir / "creep_tests.csv",
        [
            "layer",
            "state",
            "temperature_C",
            "stress_MPa",
            "strain_rate_per_s",
        ],
        creep_records,
    )

    write_csv(
        output_dir / "creep_fit_parameters.csv",
        [
            "layer",
            "state",
            "A_pre_exponential",
            "stress_exponent_n",
            "activation_energy_kJ_per_mol",
        ],
        creep_parameters,
    )

    write_csv(
        output_dir / "elastic_properties.csv",
        [
            "layer",
            "temperature_C",
            "relative_density",
            "youngs_modulus_GPa",
            "poissons_ratio",
        ],
        elastic_records,
    )

    write_csv(
        output_dir / "cte_measurements.csv",
        [
            "layer",
            "temperature_C",
            "cte_per_C",
        ],
        cte_records,
    )

    dataset_json = {
        "metadata": {
            "dataset_name": "SOFC_Material_Constitutive_Dataset",
            "dataset_version": DATASET_VERSION,
            "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "random_seed": 2025,
            "source_article": "A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation",
            "notes": "Synthetic dataset calibrated to literature-informed trends for FEM calibration workflows.",
            "temperature_profile_C": TEMPERATURE_PROFILE,
            "sintering_temperatures_C": SINTERING_TEMPERATURES,
            "time_points_min": TIME_POINTS_MIN,
            "stress_levels_MPa": STRESS_LEVELS_MPA,
            "norton_law": "strain_rate = A * sigma^n * exp(-Q / (R * T))",
        },
        "thermo_physical": thermo_records,
        "sintering": {
            "single_layer": single_layer_records,
            "bilayer": bilayer_records,
        },
        "creep": {
            "records": creep_records,
            "fit_parameters": creep_parameters,
        },
        "elastic": elastic_records,
        "cte": cte_records,
    }

    json_path = output_dir / "material_property_dataset.json"
    with json_path.open("w", encoding="utf-8") as f_json:
        json.dump(dataset_json, f_json, indent=2)

    print(f"Dataset generated under {output_dir.relative_to(dataset_root)}")


if __name__ == "__main__":
    main()
