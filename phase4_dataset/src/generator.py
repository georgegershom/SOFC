from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import random

from microstructure import build_microstructure
from temperature_models import temperature_grid_c, compute_properties_vs_temperature
from stochastic import StatValue, compute_stochastic_bounds
from exporters import (
    export_json,
    export_csv,
    export_comsol_material_table,
    export_ansys_engdat,
    export_abaqus_material,
    ensure_dir,
)

MIX_IDS = ["C", "R5S", "R10S", "R15S", "R20S", "R10L"]

@dataclass
class Row:
    Temperature_C: float
    density_kgm3: float
    specific_heat_jkgk: float
    thermal_conductivity_wmk: float
    elastic_modulus_pa: float
    poisson_ratio: float
    cte_1k: float
    free_thermal_strain: float
    compressive_strength_pa: float
    permeability_m2: float
    biot_coefficient: float
    moisture_diffusivity_m2s: float
    porosity: float
    rubber_integrity: float
    dehydration_fraction: float
    damage_index: float


def split_calibration_validation(temperatures: List[float]) -> Tuple[List[float], List[float]]:
    # Interleave: even steps for calibration, odd steps for validation to maintain coverage
    calib, valid = [], []
    for idx, t in enumerate(temperatures):
        (calib if (idx % 2 == 0) else valid).append(t)
    return calib, valid


def compute_rows_for_mix(mix_id: str, temperatures: List[float]) -> List[Dict[str, float]]:
    micro = build_microstructure(mix_id)
    rows: List[Dict[str, float]] = []
    for t in temperatures:
        props = compute_properties_vs_temperature(micro, t)
        row = {"Temperature_C": t}
        row.update(props)
        rows.append(row)
    return rows


def add_stochastic_bounds(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    # Apply relative coefficient of variation per property for dataset bounds
    cvs = {
        "density_kgm3": 0.01,
        "specific_heat_jkgk": 0.05,
        "thermal_conductivity_wmk": 0.05,
        "elastic_modulus_pa": 0.08,
        "poisson_ratio": 0.03,
        "cte_1k": 0.07,
        "free_thermal_strain": 0.07,
        "compressive_strength_pa": 0.10,
        "permeability_m2": 0.25,
        "biot_coefficient": 0.03,
        "moisture_diffusivity_m2s": 0.25,
        "porosity": 0.05,
        "rubber_integrity": 0.05,
        "dehydration_fraction": 0.05,
        "damage_index": 0.07,
    }
    abs_bounds = {
        "density_kgm3": (1100.0, 2700.0),
        "specific_heat_jkgk": (400.0, 4000.0),
        "thermal_conductivity_wmk": (0.10, 3.50),
        "elastic_modulus_pa": (0.5e9, 70e9),
        "poisson_ratio": (0.15, 0.40),
        "cte_1k": (4e-6, 90e-6),
        "free_thermal_strain": (-0.01, 0.02),
        "compressive_strength_pa": (2e6, 150e6),
        "permeability_m2": (1e-21, 1e-13),
        "biot_coefficient": (0.5, 1.0),
        "moisture_diffusivity_m2s": (1e-13, 1e-7),
        "porosity": (0.02, 0.60),
        "rubber_integrity": (0.0, 1.0),
        "dehydration_fraction": (0.0, 1.0),
        "damage_index": (0.0, 0.99),
    }
    out: List[Dict[str, float]] = []
    for r in rows:
        new_r = dict(r)
        for k, cv in cvs.items():
            mean = r[k]
            vmin, vmax = abs_bounds[k]
            stat = compute_stochastic_bounds(mean, cv, vmin, vmax)
            new_r[k+"_mean"] = stat.mean
            new_r[k+"_std"] = stat.std
        out.append(new_r)
    return out


def export_all_for_mix(mix_id: str, calib_rows: List[Dict[str, float]], valid_rows: List[Dict[str, float]], outdir: str) -> None:
    ensure_dir(outdir)
    # Core CSV with mean/std
    fields = [
        "Temperature_C",
        "density_kgm3_mean","density_kgm3_std",
        "specific_heat_jkgk_mean","specific_heat_jkgk_std",
        "thermal_conductivity_wmk_mean","thermal_conductivity_wmk_std",
        "elastic_modulus_pa_mean","elastic_modulus_pa_std",
        "poisson_ratio_mean","poisson_ratio_std",
        "cte_1k_mean","cte_1k_std",
        "free_thermal_strain_mean","free_thermal_strain_std",
        "compressive_strength_pa_mean","compressive_strength_pa_std",
        "permeability_m2_mean","permeability_m2_std",
        "biot_coefficient_mean","biot_coefficient_std",
        "moisture_diffusivity_m2s_mean","moisture_diffusivity_m2s_std",
        "porosity_mean","porosity_std",
        "rubber_integrity_mean","rubber_integrity_std",
        "dehydration_fraction_mean","dehydration_fraction_std",
        "damage_index_mean","damage_index_std",
    ]
    export_csv(os.path.join(outdir, f"{mix_id}_calibration.csv"), calib_rows, fields)
    export_csv(os.path.join(outdir, f"{mix_id}_validation.csv"), valid_rows, fields)

    # FEA format exports use deterministic means (drop std)
    def strip_stats(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        out = []
        for r in rows:
            base = {"Temperature_C": r["Temperature_C"]}
            for key in list(r.keys()):
                if key.endswith("_mean"):
                    base[key[:-5]] = r[key]
            out.append(base)
        return out

    calib_means = strip_stats(calib_rows)
    valid_means = strip_stats(valid_rows)

    export_comsol_material_table(os.path.join(outdir, f"{mix_id}_calibration_comsol.csv"), calib_means)
    export_comsol_material_table(os.path.join(outdir, f"{mix_id}_validation_comsol.csv"), valid_means)

    export_ansys_engdat(os.path.join(outdir, f"{mix_id}_calibration_ansys.csv"), calib_means)
    export_ansys_engdat(os.path.join(outdir, f"{mix_id}_validation_ansys.csv"), valid_means)

    export_abaqus_material(os.path.join(outdir, f"{mix_id}_calibration_abaqus.inp"), calib_means)
    export_abaqus_material(os.path.join(outdir, f"{mix_id}_validation_abaqus.inp"), valid_means)


def generate_dataset(out_root: str = "/workspace/phase4_dataset/outputs", step_c: float = 10.0) -> None:
    temps = temperature_grid_c(20.0, 800.0, step_c)
    calib_t, valid_t = split_calibration_validation(temps)

    for mix_id in MIX_IDS:
        all_rows = compute_rows_for_mix(mix_id, temps)
        # Keep provenance fields
        for r in all_rows:
            r["Mix_ID"] = mix_id
        # Split and add stats
        calib_rows = [r for r in all_rows if r["Temperature_C"] in calib_t]
        valid_rows = [r for r in all_rows if r["Temperature_C"] in valid_t]
        calib_rows = add_stochastic_bounds(calib_rows)
        valid_rows = add_stochastic_bounds(valid_rows)
        # Export
        outdir = os.path.join(out_root, mix_id)
        export_all_for_mix(mix_id, calib_rows, valid_rows, outdir)


if __name__ == "__main__":
    generate_dataset()
