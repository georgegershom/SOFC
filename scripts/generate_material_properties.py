#!/usr/bin/env python3
import json
import math
import os
import random
import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple

random.seed(42)

OUTPUT_DIR = "/workspace/datasets/material_properties"
T_MIN_C = 25
T_MAX_C = 1000
T_STEP_C = 25
TEMPERATURE_C_GRID = list(range(T_MIN_C, T_MAX_C + 1, T_STEP_C))


@dataclass
class ElasticParams:
    E0_GPa: float
    poisson_0: float
    degrade_fraction_over_range: float
    poisson_delta_over_range: float


@dataclass
class CTEParams:
    alpha_rt_1_per_K: float
    alpha_highT_1_per_K: float


@dataclass
class Material:
    name: str
    phase: str
    elastic: ElasticParams
    cte: CTEParams


# Base material parameters (plausible, literature-inspired but fabricated)
BASE_MATERIALS: Dict[str, Material] = {
    "YSZ": Material(
        name="YSZ",
        phase="ceramic",
        elastic=ElasticParams(
            E0_GPa=205.0,
            poisson_0=0.30,
            degrade_fraction_over_range=0.12,  # ~12% reduction by 1000 C
            poisson_delta_over_range=0.01,
        ),
        cte=CTEParams(
            alpha_rt_1_per_K=10.5e-6,
            alpha_highT_1_per_K=11.5e-6,
        ),
    ),
    "Ni": Material(
        name="Ni",
        phase="metal",
        elastic=ElasticParams(
            E0_GPa=200.0,
            poisson_0=0.31,
            degrade_fraction_over_range=0.18,  # ~18% reduction by 1000 C
            poisson_delta_over_range=0.02,
        ),
        cte=CTEParams(
            alpha_rt_1_per_K=13.5e-6,
            alpha_highT_1_per_K=16.0e-6,
        ),
    ),
    "NiO": Material(
        name="NiO",
        phase="ceramic",
        elastic=ElasticParams(
            E0_GPa=180.0,
            poisson_0=0.22,
            degrade_fraction_over_range=0.20,
            poisson_delta_over_range=0.02,
        ),
        cte=CTEParams(
            alpha_rt_1_per_K=12.0e-6,
            alpha_highT_1_per_K=14.0e-6,
        ),
    ),
}


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def interpolate_linear(x0: float, x1: float, t: float) -> float:
    return x0 * (1.0 - t) + x1 * t


def compute_elastic_T_series(material: Material) -> Tuple[List[float], List[float]]:
    E_series = []
    nu_series = []
    for T in TEMPERATURE_C_GRID:
        fraction = (T - T_MIN_C) / (T_MAX_C - T_MIN_C)
        E_T = material.elastic.E0_GPa * (1.0 - material.elastic.degrade_fraction_over_range * fraction)
        nu_T = material.elastic.poisson_0 + material.elastic.poisson_delta_over_range * fraction
        E_series.append(E_T)
        nu_series.append(nu_T)
    return E_series, nu_series


def compute_cte_T_series(material: Material) -> List[float]:
    alpha_series = []
    for T in TEMPERATURE_C_GRID:
        fraction = (T - T_MIN_C) / (T_MAX_C - T_MIN_C)
        alpha_T = interpolate_linear(material.cte.alpha_rt_1_per_K, material.cte.alpha_highT_1_per_K, fraction)
        alpha_series.append(alpha_T)
    return alpha_series


def compute_vrh_composite_series(E1: List[float], E2: List[float], v1: List[float], v2: List[float], vol_fraction_phase1: float) -> Tuple[List[float], List[float]]:
    E_vrh = []
    v_mix = []
    f = vol_fraction_phase1
    for E1_T, E2_T, v1_T, v2_T in zip(E1, E2, v1, v2):
        E_voigt = f * E1_T + (1.0 - f) * E2_T
        E_reuss = 1.0 / (f / E1_T + (1.0 - f) / E2_T)
        E_vrh_T = 0.5 * (E_voigt + E_reuss)
        v_mix_T = f * v1_T + (1.0 - f) * v2_T
        E_vrh.append(E_vrh_T)
        v_mix.append(v_mix_T)
    return E_vrh, v_mix


def compute_mixed_cte_series(alpha1: List[float], alpha2: List[float], vol_fraction_phase1: float) -> List[float]:
    f = vol_fraction_phase1
    return [f * a1 + (1.0 - f) * a2 for a1, a2 in zip(alpha1, alpha2)]


def compute_plane_strain_modulus(E_GPa: float, nu: float) -> float:
    E_Pa = E_GPa * 1e9
    return E_Pa / (1.0 - nu * nu)


def compute_Gc_from_Kic(Kic_MPa_sqrt_m: float, E_plane_strain_Pa: float) -> float:
    K_pa_sqrt_m = Kic_MPa_sqrt_m * 1e6
    return (K_pa_sqrt_m ** 2) / E_plane_strain_Pa


def compute_interface_equivalent_E_plane_strain(E1_GPa: float, v1: float, E2_GPa: float, v2: float) -> float:
    # Harmonic average of plane-strain moduli as a simple surrogate for interface compliance
    E1p = compute_plane_strain_modulus(E1_GPa, v1)
    E2p = compute_plane_strain_modulus(E2_GPa, v2)
    return 2.0 * E1p * E2p / (E1p + E2p)


def write_elastic_rt_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "material", "phase", "composition_note", "porosity_vol_frac",
        "temperature_C", "E_GPa", "poisson_ratio",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_fracture_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "type",  # bulk | interface
        "material_or_interface",
        "description",
        "temperature_C",
        "environment",
        "E_ref_GPa",
        "poisson_ref",
        "K_IC_MPa_sqrt_m",
        "G_c_J_per_m2",
        "uncertainty_note",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_chemical_expansion_csv(path: str, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "material",
        "family",
        "parameter",
        "value",
        "unit",
        "reference_temperature_C",
        "std_dev",
        "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)

    # Compute base series for pure phases
    series_by_material: Dict[str, Dict[str, List[float]]] = {}
    for key, mat in BASE_MATERIALS.items():
        E_series, nu_series = compute_elastic_T_series(mat)
        alpha_series = compute_cte_T_series(mat)
        series_by_material[key] = {
            "E_GPa": E_series,
            "nu": nu_series,
            "alpha_1_per_K": alpha_series,
        }

    # Composites: Ni-YSZ at various Ni volume fractions
    composite_entries = []
    composite_fractions = [0.30, 0.40, 0.50]
    for f_ni in composite_fractions:
        E_mix, nu_mix = compute_vrh_composite_series(
            series_by_material["Ni"]["E_GPa"],
            series_by_material["YSZ"]["E_GPa"],
            series_by_material["Ni"]["nu"],
            series_by_material["YSZ"]["nu"],
            vol_fraction_phase1=f_ni,
        )
        alpha_mix = compute_mixed_cte_series(
            series_by_material["Ni"]["alpha_1_per_K"],
            series_by_material["YSZ"]["alpha_1_per_K"],
            vol_fraction_phase1=f_ni,
        )
        composite_entries.append({
            "name": "Ni-YSZ",
            "phase": "composite",
            "composition_note": f"{int(round(f_ni*100))}% Ni vol.",
            "porosity_vol_frac": 0.10,  # fabricated assumption
            "E_GPa": E_mix,
            "nu": nu_mix,
            "alpha_1_per_K": alpha_mix,
            "f_Ni": f_ni,
        })

    # 1) Elastic RT CSV
    elastic_rt_rows: List[Dict[str, object]] = []
    # Pure phases
    for key, mat in BASE_MATERIALS.items():
        elastic_rt_rows.append({
            "material": key,
            "phase": mat.phase,
            "composition_note": "pure",
            "porosity_vol_frac": 0.0,
            "temperature_C": 25,
            "E_GPa": series_by_material[key]["E_GPa"][0],
            "poisson_ratio": series_by_material[key]["nu"][0],
        })
    # Composites at RT
    for comp in composite_entries:
        elastic_rt_rows.append({
            "material": "Ni-YSZ",
            "phase": "composite",
            "composition_note": comp["composition_note"],
            "porosity_vol_frac": comp["porosity_vol_frac"],
            "temperature_C": 25,
            "E_GPa": comp["E_GPa"][0],
            "poisson_ratio": comp["nu"][0],
        })
    write_elastic_rt_csv(os.path.join(OUTPUT_DIR, "elastic_properties_rt.csv"), elastic_rt_rows)

    # 2) Elastic T JSON (includes nu)
    elastic_T_payload = []
    # Pure phases
    for key, mat in BASE_MATERIALS.items():
        elastic_T_payload.append({
            "material": key,
            "phase": mat.phase,
            "composition_note": "pure",
            "porosity_vol_frac": 0.0,
            "temperature_C": TEMPERATURE_C_GRID,
            "E_GPa": series_by_material[key]["E_GPa"],
            "poisson_ratio": series_by_material[key]["nu"],
        })
    # Composites
    for comp in composite_entries:
        elastic_T_payload.append({
            "material": "Ni-YSZ",
            "phase": "composite",
            "composition_note": comp["composition_note"],
            "porosity_vol_frac": comp["porosity_vol_frac"],
            "temperature_C": TEMPERATURE_C_GRID,
            "E_GPa": comp["E_GPa"],
            "poisson_ratio": comp["nu"],
        })
    with open(os.path.join(OUTPUT_DIR, "elastic_properties_T.json"), "w") as f:
        json.dump(elastic_T_payload, f, indent=2)

    # 3) CTE T JSON
    cte_T_payload = []
    for key, mat in BASE_MATERIALS.items():
        cte_T_payload.append({
            "material": key,
            "phase": mat.phase,
            "composition_note": "pure",
            "temperature_C": TEMPERATURE_C_GRID,
            "alpha_1_per_K": series_by_material[key]["alpha_1_per_K"],
        })
    for comp in composite_entries:
        cte_T_payload.append({
            "material": "Ni-YSZ",
            "phase": "composite",
            "composition_note": comp["composition_note"],
            "temperature_C": TEMPERATURE_C_GRID,
            "alpha_1_per_K": comp["alpha_1_per_K"],
        })
    with open(os.path.join(OUTPUT_DIR, "cte_T.json"), "w") as f:
        json.dump(cte_T_payload, f, indent=2)

    # 4) Fracture datasets (bulk and interfaces)
    fracture_rows: List[Dict[str, object]] = []

    # Bulk materials at RT
    bulk_specs = [
        {"key": "YSZ", "K_mean": 3.6, "K_std": 0.4},
        {"key": "Ni", "K_mean": 100.0, "K_std": 10.0},
        {"key": "NiO", "K_mean": 2.2, "K_std": 0.4},
    ]
    for spec in bulk_specs:
        key = spec["key"]
        E_GPa = series_by_material[key]["E_GPa"][0]
        nu = series_by_material[key]["nu"][0]
        E_plane = compute_plane_strain_modulus(E_GPa, nu)
        Kic = random.gauss(spec["K_mean"], spec["K_std"])  # MPa*sqrt(m)
        Gc = compute_Gc_from_Kic(Kic, E_plane)
        fracture_rows.append({
            "type": "bulk",
            "material_or_interface": key,
            "description": "fully dense, fabricated from literature ranges",
            "temperature_C": 25,
            "environment": "ambient",
            "E_ref_GPa": E_GPa,
            "poisson_ref": nu,
            "K_IC_MPa_sqrt_m": round(Kic, 3),
            "G_c_J_per_m2": round(Gc, 3),
            "uncertainty_note": "+/- {:.2f} MPa√m on K_IC".format(spec["K_std"]),
        })

    # Bulk composites at RT (porous anodes are weaker — we fabricate moderate values)
    for comp in composite_entries:
        f_Ni = comp["f_Ni"]
        E_GPa = comp["E_GPa"][0]
        nu = comp["nu"][0]
        E_plane = compute_plane_strain_modulus(E_GPa, nu)
        # fabricate K_IC scaling modestly with Ni fraction
        K_base = 1.8 + 1.5 * (f_Ni - 0.30) / 0.20  # 0.30->1.8, 0.50->3.3
        Kic = random.gauss(K_base, 0.2)
        Gc = compute_Gc_from_Kic(Kic, E_plane)
        fracture_rows.append({
            "type": "bulk",
            "material_or_interface": "Ni-YSZ",
            "description": f"porous anode, {comp['composition_note']}, fabricated",
            "temperature_C": 25,
            "environment": "ambient",
            "E_ref_GPa": E_GPa,
            "poisson_ref": nu,
            "K_IC_MPa_sqrt_m": round(Kic, 3),
            "G_c_J_per_m2": round(Gc, 3),
            "uncertainty_note": "+/- 0.25 MPa√m on K_IC",
        })

    # Interface fracture: anode (Ni-YSZ) / electrolyte (YSZ)
    interface_envs = [
        ("as-sintered", 25, 7.0),
        ("reduced", 800, 9.0),
        ("oxidized", 800, 5.0),
    ]
    for comp in composite_entries:
        f_Ni = comp["f_Ni"]
        # Reference YsZ and anode properties at the environment temperature for E'
        for env_name, T_ref, Gc_mean in interface_envs:
            idx = TEMPERATURE_C_GRID.index(T_ref)
            E_ysz = series_by_material["YSZ"]["E_GPa"][idx]
            v_ysz = series_by_material["YSZ"]["nu"][idx]
            E_an = comp["E_GPa"][idx]
            v_an = comp["nu"][idx]
            E_plane_eq = compute_interface_equivalent_E_plane_strain(E_an, v_an, E_ysz, v_ysz)
            # fabricate distribution of Gc around mean
            Gc_i = max(1.0, random.gauss(Gc_mean + 3.0 * (f_Ni - 0.40), 1.0))  # J/m^2
            Kic_i = math.sqrt(Gc_i * E_plane_eq) / 1e6  # back to MPa*sqrt(m)
            fracture_rows.append({
                "type": "interface",
                "material_or_interface": "anode( Ni-YSZ ) / electrolyte( YSZ )",
                "description": f"interface fracture, {comp['composition_note']}",
                "temperature_C": T_ref,
                "environment": env_name,
                "E_ref_GPa": round(E_plane_eq / 1e9, 3),  # equivalent plane-strain in GPa
                "poisson_ref": None,
                "K_IC_MPa_sqrt_m": round(Kic_i, 3),
                "G_c_J_per_m2": round(Gc_i, 3),
                "uncertainty_note": "+/- 1.0 J/m^2 on G_c (fabricated)",
            })

    write_fracture_csv(os.path.join(OUTPUT_DIR, "fracture_properties.csv"), fracture_rows)

    # 5) Chemical expansion coefficients
    chem_rows: List[Dict[str, object]] = []

    # Ni -> NiO oxidation (phase change); report linear expansion per unit oxidation extent
    # Pilling-Bedworth ratio for Ni->NiO ~ 1.69 (volumetric); linear ~ 1.19
    chem_rows.append({
        "material": "Ni -> NiO",
        "family": "phase-change",
        "parameter": "beta_linear_per_oxidation_fraction",
        "value": 0.19,
        "unit": "strain per unit oxidation fraction",
        "reference_temperature_C": 800,
        "std_dev": 0.02,
        "notes": "Derived from PB ratio ~1.69 (fabricated mapping)",
    })
    chem_rows.append({
        "material": "Ni -> NiO",
        "family": "phase-change",
        "parameter": "Pilling_Bedworth_ratio",
        "value": 1.69,
        "unit": "-",
        "reference_temperature_C": 800,
        "std_dev": 0.05,
        "notes": "Volumetric ratio upon oxidation (fabricated range)",
    })

    # Perovskites (examples used widely in SOFC cathodes)
    chem_rows.append({
        "material": "LSCF",
        "family": "perovskite",
        "parameter": "beta_chem_per_delta_O",
        "value": 0.018,
        "unit": "strain per oxygen nonstoichiometry delta",
        "reference_temperature_C": 800,
        "std_dev": 0.004,
        "notes": "Fabricated within literature-like range",
    })
    chem_rows.append({
        "material": "LSM",
        "family": "perovskite",
        "parameter": "beta_chem_per_delta_O",
        "value": 0.006,
        "unit": "strain per oxygen nonstoichiometry delta",
        "reference_temperature_C": 800,
        "std_dev": 0.002,
        "notes": "Fabricated within literature-like range",
    })

    write_chemical_expansion_csv(os.path.join(OUTPUT_DIR, "chemical_expansion.csv"), chem_rows)

    # Save a small manifest to help discovery
    manifest = {
        "files": [
            "elastic_properties_rt.csv",
            "elastic_properties_T.json",
            "cte_T.json",
            "fracture_properties.csv",
            "chemical_expansion.csv",
        ],
        "temperature_C_grid": TEMPERATURE_C_GRID,
        "notes": "Fabricated dataset for simulation and modeling; values are plausible but not measured.",
        "version": 1,
    }
    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
