#!/usr/bin/env python3
"""
Synthetic dataset package generator for:
Failure Mechanism of underground structure in sandy and clay soils.

Outputs:
- Multiple CSV datasets (by category)
- Multiple PNG figures
- A ZIP archive containing only CSV files
"""

from __future__ import annotations

import csv
import math
import zipfile
from collections import Counter, deque
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RNG = np.random.default_rng(20260217)

OUTPUT_ROOT = Path(__file__).resolve().parent / "underground_failure_dataset"
CSV_DIR = OUTPUT_ROOT / "csv"
FIG_DIR = OUTPUT_ROOT / "figures"
META_DIR = OUTPUT_ROOT / "metadata"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def ensure_dirs() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def generate_physical_model_dataset() -> list[dict]:
    rows: list[dict] = []
    flow_conditions = [5, 10, 15, 20, 30, 40, 50, 60]
    strata_options = [
        "SC_top + silty_sand_mid + dense_sand_bottom",
        "SC_top + sandy_clay_mid + weathered_mudstone_bottom",
        "SC_top + loose_sand_mid + sandy_clay_bottom",
        "SC_top + silty_clay_mid + dense_sand_bottom",
    ]

    for flow in flow_conditions:
        for run in range(1, 3):
            exp_id = f"PM-{flow:02d}-{run:02d}"
            clay_fraction = float(RNG.uniform(24, 49))
            void_ratio = float(RNG.uniform(0.52, 0.78))
            groundwater_depth = float(RNG.uniform(0.8, 2.6))
            strata = strata_options[int(RNG.integers(0, len(strata_options)))]
            target_settlement = 9.0 + 1.05 * flow + float(RNG.normal(0.0, 1.6))
            time_constant = clamp(84.0 - 0.75 * flow + float(RNG.normal(0, 2.0)), 26.0, 95.0)

            for t in range(0, 181, 6):
                progress = 1.0 - math.exp(-(t / time_constant))
                settlement = target_settlement * progress + float(RNG.normal(0.0, 0.65))
                settlement = max(settlement, 0.0)
                sinkhole_diameter = (
                    0.09
                    + 0.0042 * settlement
                    + 0.0015 * flow
                    + float(RNG.normal(0.0, 0.012))
                )
                sinkhole_diameter = max(sinkhole_diameter, 0.04)
                pore_pressure = (
                    12.0
                    + 0.95 * flow
                    + 0.07 * t
                    + float(RNG.normal(0.0, 2.1))
                )
                failure_flag = int(settlement > 38.0 or sinkhole_diameter > 0.42)

                rows.append(
                    {
                        "experiment_id": exp_id,
                        "soil_profile": strata,
                        "soil_type": "Sandy clay",
                        "flow_condition_lpm": round(flow, 2),
                        "time_min": t,
                        "settlement_mm": round(settlement, 3),
                        "sinkhole_diameter_m": round(sinkhole_diameter, 4),
                        "pore_pressure_kpa": round(pore_pressure, 3),
                        "clay_fraction_pct": round(clay_fraction, 2),
                        "initial_void_ratio": round(void_ratio, 3),
                        "groundwater_depth_m": round(groundwater_depth, 3),
                        "failure_flag": failure_flag,
                        "source_reference": "NIH / Mendeley Data (synthetic reconstruction)",
                    }
                )

    fields = [
        "experiment_id",
        "soil_profile",
        "soil_type",
        "flow_condition_lpm",
        "time_min",
        "settlement_mm",
        "sinkhole_diameter_m",
        "pore_pressure_kpa",
        "clay_fraction_pct",
        "initial_void_ratio",
        "groundwater_depth_m",
        "failure_flag",
        "source_reference",
    ]
    write_csv(CSV_DIR / "physical_model_pipeline_sinkholes.csv", fields, rows)
    return rows


def generate_field_monitoring_dataset() -> list[dict]:
    rows: list[dict] = []
    start = date(2023, 1, 1)
    depths = [2.0, 5.0, 8.0, 11.0]

    for idx in range(8):
        monitor_id = f"EMB-{idx + 1:02d}"
        chainage = 25.0 * idx
        for depth in depths:
            baseline_p = 1400.0 + 16.0 * depth + float(RNG.normal(0, 12))
            baseline_s = 285.0 + 9.0 * depth + float(RNG.normal(0, 7))
            creep = float(RNG.uniform(0.0, 0.8))

            for day in range(0, 361, 10):
                obs_date = start + timedelta(days=day)
                load_stage = min(220.0, 22.0 + 0.55 * day)
                seasonal = 1.0 + 0.11 * math.sin((2 * math.pi * day) / 365.0)
                p_wave = baseline_p + 0.34 * load_stage * seasonal + float(RNG.normal(0, 8))
                s_wave = baseline_s + 0.18 * load_stage * seasonal + float(RNG.normal(0, 5))

                creep += max(0.0, float(RNG.normal(0.05, 0.03)))
                ext_disp = (
                    0.052 * load_stage * math.exp(-depth / 16.0)
                    + creep
                    + float(RNG.normal(0, 0.35))
                )
                pressure_cell = 0.72 * load_stage + 1.4 * depth + float(RNG.normal(0, 2.4))
                pore_pressure = (
                    47.0
                    + 0.22 * load_stage
                    + 6.0 * math.sin((2 * math.pi * (day + 8.0 * depth)) / 365.0)
                    + float(RNG.normal(0, 3.2))
                )
                creep_rate = max(0.0, ext_disp / (day + 1.0) * 4.8 + float(RNG.normal(0.01, 0.005)))

                if ext_disp > 26 or pore_pressure > 110:
                    alert_level = "high"
                elif ext_disp > 17 or pore_pressure > 95:
                    alert_level = "medium"
                else:
                    alert_level = "low"

                rows.append(
                    {
                        "monitor_point_id": monitor_id,
                        "chainage_m": round(chainage, 2),
                        "depth_m": round(depth, 2),
                        "date": obs_date.isoformat(),
                        "load_stage_kpa": round(load_stage, 3),
                        "p_wave_velocity_mps": round(p_wave, 3),
                        "s_wave_velocity_mps": round(s_wave, 3),
                        "extensometer_displacement_mm": round(ext_disp, 3),
                        "pressure_cell_kpa": round(pressure_cell, 3),
                        "pore_pressure_kpa": round(pore_pressure, 3),
                        "creep_rate_mm_per_day": round(creep_rate, 4),
                        "formation": "Clay (Mudstone Formation)",
                        "alert_level": alert_level,
                        "source_reference": "University of Bath (synthetic reconstruction)",
                    }
                )

    fields = [
        "monitor_point_id",
        "chainage_m",
        "depth_m",
        "date",
        "load_stage_kpa",
        "p_wave_velocity_mps",
        "s_wave_velocity_mps",
        "extensometer_displacement_mm",
        "pressure_cell_kpa",
        "pore_pressure_kpa",
        "creep_rate_mm_per_day",
        "formation",
        "alert_level",
        "source_reference",
    ]
    write_csv(CSV_DIR / "field_monitoring_embankment_clay.csv", fields, rows)
    return rows


def generate_numerical_simulation_dataset() -> list[dict]:
    rows: list[dict] = []
    case = 0
    depths = [12, 16, 20, 24, 28]
    widths = [10, 15, 20, 25]
    prop_spacings = [2.5, 3.5, 4.5]
    wall_thicknesses = [0.6, 0.8, 1.0]
    stiffness_factors = [0.8, 1.0, 1.2]

    for d in depths:
        for w in widths:
            for prop in prop_spacings:
                for wall_t in wall_thicknesses:
                    for stiff in stiffness_factors:
                        case += 1
                        max_wall_disp = (
                            1.65 * d
                            + 0.72 * w
                            + 6.8 * (prop - 2.5)
                            - 24.0 * wall_t
                            + 18.0 / stiff
                            + float(RNG.normal(0, 2.3))
                        )
                        max_wall_disp = max(max_wall_disp, 5.0)
                        toe_heave = (
                            0.42 * max_wall_disp
                            + 0.3 * d
                            - 6.0 * wall_t
                            + float(RNG.normal(0, 1.8))
                        )
                        fos = (
                            2.45
                            - 0.033 * d
                            - 0.019 * w
                            - 0.07 * (prop - 2.5)
                            + 0.46 * wall_t
                            + 0.14 * stiff
                            + float(RNG.normal(0, 0.04))
                        )

                        if fos < 1.1:
                            failure_mode = "Global instability"
                        elif toe_heave > 40:
                            failure_mode = "Basal heave"
                        elif max_wall_disp > 70:
                            failure_mode = "Excessive wall deflection"
                        else:
                            failure_mode = "Serviceable"

                        rows.append(
                            {
                                "case_id": f"LC-{case:04d}",
                                "soil_type": "London Clay",
                                "excavation_depth_m": d,
                                "excavation_width_m": w,
                                "prop_spacing_m": prop,
                                "wall_thickness_m": wall_t,
                                "stiffness_reduction_factor": stiff,
                                "max_wall_displacement_mm": round(max_wall_disp, 3),
                                "toe_heave_mm": round(toe_heave, 3),
                                "factor_of_safety": round(fos, 4),
                                "predicted_failure_mode": failure_mode,
                                "source_reference": "Zenodo / Imperial College London (synthetic reconstruction)",
                            }
                        )

    fields = [
        "case_id",
        "soil_type",
        "excavation_depth_m",
        "excavation_width_m",
        "prop_spacing_m",
        "wall_thickness_m",
        "stiffness_reduction_factor",
        "max_wall_displacement_mm",
        "toe_heave_mm",
        "factor_of_safety",
        "predicted_failure_mode",
        "source_reference",
    ]
    write_csv(CSV_DIR / "numerical_simulation_london_clay_excavation.csv", fields, rows)
    return rows


def generate_geohazard_dataset() -> list[dict]:
    rows: list[dict] = []
    cell = 0

    for ix in range(20):
        for iy in range(18):
            cell += 1
            easting = 530000 + 250 * ix
            northing = 160000 + 250 * iy
            spatial_term = 0.55 + 0.45 * math.sin(ix / 3.0) + 0.35 * math.cos(iy / 4.0)

            swelling = clamp(35 + 18 * spatial_term + float(RNG.normal(0, 12)), 0, 100)
            compressible = clamp(30 + 16 * (1 - spatial_term) + float(RNG.normal(0, 11)), 0, 100)
            running_sand = clamp(28 + 14 * math.sin((ix + iy) / 5.0) + float(RNG.normal(0, 10)), 0, 100)
            corroded_density = max(0.0, float(RNG.normal(6 + 0.03 * swelling + 0.02 * compressible, 2.1)))

            movement_potential = clamp(
                0.38 * swelling + 0.34 * compressible + 0.22 * running_sand + 1.8 * corroded_density,
                0,
                100,
            )

            if movement_potential > 75:
                risk_class = "Very High"
            elif movement_potential > 60:
                risk_class = "High"
            elif movement_potential > 45:
                risk_class = "Moderate"
            elif movement_potential > 30:
                risk_class = "Low"
            else:
                risk_class = "Very Low"

            if swelling + compressible >= running_sand + 25:
                dominant_soil = "Clay"
            elif running_sand > swelling and running_sand > compressible:
                dominant_soil = "Sand"
            else:
                dominant_soil = "Clay-Sand Mix"

            rows.append(
                {
                    "cell_id": f"BGS-{cell:04d}",
                    "easting_m": easting,
                    "northing_m": northing,
                    "swelling_clay_index": round(swelling, 3),
                    "compressible_ground_index": round(compressible, 3),
                    "running_sand_index": round(running_sand, 3),
                    "corroded_asset_density_per_km2": round(corroded_density, 3),
                    "ground_movement_potential": round(movement_potential, 3),
                    "risk_class": risk_class,
                    "dominant_soil": dominant_soil,
                    "source_reference": "British Geological Survey (BGS) style (synthetic reconstruction)",
                }
            )

    fields = [
        "cell_id",
        "easting_m",
        "northing_m",
        "swelling_clay_index",
        "compressible_ground_index",
        "running_sand_index",
        "corroded_asset_density_per_km2",
        "ground_movement_potential",
        "risk_class",
        "dominant_soil",
        "source_reference",
    ]
    write_csv(CSV_DIR / "geohazard_susceptibility_bgs_style.csv", fields, rows)
    return rows


def generate_meteorological_dataset() -> list[dict]:
    rows: list[dict] = []
    start = date(2022, 1, 1)
    end = date(2024, 12, 31)
    rolling = deque(maxlen=7)
    soil_moisture = 42.0
    day_idx = 0
    current = start

    while current <= end:
        doy = current.timetuple().tm_yday
        temperature = 11.0 + 8.0 * math.sin((2 * math.pi * (doy - 80)) / 365.0) + float(RNG.normal(0, 1.8))

        base_rain = max(0.0, float(RNG.gamma(shape=1.8, scale=2.5) - 1.4))
        seasonal_rain = 1.0 + 0.35 * math.sin((2 * math.pi * (doy + 20)) / 365.0)
        storm = float(RNG.uniform(12, 38)) if float(RNG.random()) < 0.04 else 0.0
        precip = max(0.0, base_rain * seasonal_rain + storm)

        evap = max(0.0, 0.42 * temperature + float(RNG.normal(0.6, 0.8)))
        water_balance = precip - evap

        rolling.append(precip)
        antecedent_7d = sum(rolling)
        soil_moisture = clamp(soil_moisture + 0.45 * precip - 0.34 * evap + float(RNG.normal(0, 1.1)), 5.0, 100.0)
        warning_index = clamp(0.55 * antecedent_7d + 0.65 * soil_moisture - 35.0 + float(RNG.normal(0, 4)), 0.0, 100.0)

        if warning_index >= 78:
            warning_level = "critical"
        elif warning_index >= 62:
            warning_level = "high"
        elif warning_index >= 45:
            warning_level = "moderate"
        else:
            warning_level = "low"

        rows.append(
            {
                "date": current.isoformat(),
                "precipitation_mm": round(precip, 3),
                "temperature_c": round(temperature, 3),
                "evapotranspiration_mm": round(evap, 3),
                "antecedent_rain_7d_mm": round(antecedent_7d, 3),
                "soil_moisture_index": round(soil_moisture, 3),
                "water_balance_mm": round(water_balance, 3),
                "failure_warning_index": round(warning_index, 3),
                "warning_level": warning_level,
                "context_soil_type": "Clay",
                "source_reference": "Newcastle University style weather context (synthetic reconstruction)",
            }
        )

        current += timedelta(days=1)
        day_idx += 1

    fields = [
        "date",
        "precipitation_mm",
        "temperature_c",
        "evapotranspiration_mm",
        "antecedent_rain_7d_mm",
        "soil_moisture_index",
        "water_balance_mm",
        "failure_warning_index",
        "warning_level",
        "context_soil_type",
        "source_reference",
    ]
    write_csv(CSV_DIR / "meteorological_context_clay_embankment.csv", fields, rows)
    return rows


def generate_soil_properties_dataset() -> list[dict]:
    rows: list[dict] = []

    for i in range(1, 251):
        rel_density = float(RNG.uniform(35, 95))
        d10 = float(RNG.uniform(0.08, 0.32))
        d50 = d10 * float(RNG.uniform(2.2, 4.4))
        d90 = d50 * float(RNG.uniform(1.7, 3.0))
        friction = 29.0 + 0.12 * rel_density + float(RNG.normal(0, 1.2))
        dilation = max(0.0, -4.0 + 0.13 * rel_density + float(RNG.normal(0, 1.0)))
        youngs = 12.0 + 0.85 * rel_density + float(RNG.normal(0, 4.0))
        nu = clamp(float(RNG.normal(0.29, 0.035)), 0.2, 0.38)

        rows.append(
            {
                "sample_id": f"SAND-{i:04d}",
                "soil_type": "Sandy soil (cohesionless)",
                "relative_density_pct": round(rel_density, 3),
                "D10_mm": round(d10, 4),
                "D50_mm": round(d50, 4),
                "D90_mm": round(d90, 4),
                "friction_angle_deg": round(friction, 3),
                "dilation_angle_deg": round(dilation, 3),
                "undrained_shear_strength_cu_kpa": "",
                "plasticity_index_PI": "",
                "liquid_limit_pct": "",
                "plastic_limit_pct": "",
                "compression_index_Cc": "",
                "swelling_index_Cs": "",
                "youngs_modulus_mpa": round(youngs, 3),
                "poisson_ratio": round(nu, 4),
                "source_reference": "SoilModels / Mendeley style calibration set (synthetic)",
            }
        )

    for i in range(1, 251):
        plasticity_index = float(RNG.uniform(12, 45))
        liquid_limit = plasticity_index + float(RNG.uniform(22, 38))
        plastic_limit = liquid_limit - plasticity_index
        cu = 18.0 + 2.7 * plasticity_index + float(RNG.normal(0, 10))
        cu = max(cu, 18.0)
        friction = float(RNG.uniform(18, 31))
        cc = clamp(0.12 + 0.007 * plasticity_index + float(RNG.normal(0, 0.02)), 0.08, 0.55)
        cs = clamp(0.02 + 0.0025 * plasticity_index + float(RNG.normal(0, 0.01)), 0.01, 0.2)
        youngs = 5.0 + 0.22 * cu + float(RNG.normal(0, 3.0))
        nu = clamp(float(RNG.normal(0.38, 0.03)), 0.3, 0.48)

        rows.append(
            {
                "sample_id": f"CLAY-{i:04d}",
                "soil_type": "Clay soil (cohesive)",
                "relative_density_pct": "",
                "D10_mm": "",
                "D50_mm": "",
                "D90_mm": "",
                "friction_angle_deg": round(friction, 3),
                "dilation_angle_deg": 0.0,
                "undrained_shear_strength_cu_kpa": round(cu, 3),
                "plasticity_index_PI": round(plasticity_index, 3),
                "liquid_limit_pct": round(liquid_limit, 3),
                "plastic_limit_pct": round(plastic_limit, 3),
                "compression_index_Cc": round(cc, 4),
                "swelling_index_Cs": round(cs, 4),
                "youngs_modulus_mpa": round(youngs, 3),
                "poisson_ratio": round(nu, 4),
                "source_reference": "SoilModels / Mendeley style calibration set (synthetic)",
            }
        )

    fields = [
        "sample_id",
        "soil_type",
        "relative_density_pct",
        "D10_mm",
        "D50_mm",
        "D90_mm",
        "friction_angle_deg",
        "dilation_angle_deg",
        "undrained_shear_strength_cu_kpa",
        "plasticity_index_PI",
        "liquid_limit_pct",
        "plastic_limit_pct",
        "compression_index_Cc",
        "swelling_index_Cs",
        "youngs_modulus_mpa",
        "poisson_ratio",
        "source_reference",
    ]
    write_csv(CSV_DIR / "soil_properties_input_parameters.csv", fields, rows)
    return rows


def generate_structural_monitoring_dataset() -> list[dict]:
    rows: list[dict] = []
    structures = ["Tunnel", "Basement wall", "Shaft lining"]
    soil_options = ["Sand", "Clay", "Clay-Sand Mix"]
    groundwater_options = ["Dry", "Perched", "Saturated"]

    for record in range(1, 1201):
        structure = structures[int(RNG.integers(0, len(structures)))]
        soil = str(RNG.choice(soil_options, p=[0.45, 0.45, 0.10]))
        groundwater = str(RNG.choice(groundwater_options, p=[0.34, 0.28, 0.38]))
        cover_depth = float(RNG.uniform(4.0, 35.0))
        diameter = float(RNG.uniform(4.0, 15.0))
        wall_t = float(RNG.uniform(0.25, 1.2))

        soil_factor = 1.0 if soil == "Sand" else 1.25 if soil == "Clay" else 1.15
        water_factor = 1.0 if groundwater == "Dry" else 1.15 if groundwater == "Perched" else 1.35

        settlement = (
            ((0.55 * cover_depth + 1.15 * diameter) / (wall_t * 3.2))
            * soil_factor
            * water_factor
            + float(RNG.normal(0, 2.4))
        )
        settlement = max(0.0, settlement)
        lateral = max(0.0, 0.62 * settlement + 0.4 * diameter + float(RNG.normal(0, 1.8)))
        earth_pressure = (
            18.0
            * cover_depth
            * water_factor
            * (1.0 if soil == "Sand" else 1.08)
            + float(RNG.normal(0, 25.0))
        )
        pore_pressure = (
            (35.0 if groundwater == "Dry" else 90.0 if groundwater == "Perched" else 130.0)
            + 2.4 * cover_depth
            + float(RNG.normal(0, 10.0))
        )

        failure_label = int(
            settlement > 38.0 or lateral > 28.0 or (soil == "Clay" and pore_pressure > 180.0)
        )
        if failure_label == 0:
            failure_mode = "Stable"
        elif soil == "Sand" and groundwater == "Saturated" and settlement > 30.0:
            failure_mode = "Piping/erosion"
        elif lateral > 32.0:
            failure_mode = "Structural buckling"
        elif soil == "Clay" and settlement > 35.0:
            failure_mode = "Consolidation settlement"
        else:
            failure_mode = "Serviceability exceedance"

        rows.append(
            {
                "record_id": f"STR-{record:05d}",
                "structure_type": structure,
                "soil_type": soil,
                "cover_depth_m": round(cover_depth, 3),
                "diameter_or_width_m": round(diameter, 3),
                "wall_thickness_m": round(wall_t, 3),
                "groundwater_state": groundwater,
                "surface_settlement_mm": round(settlement, 3),
                "lateral_deflection_mm": round(lateral, 3),
                "earth_pressure_kpa": round(earth_pressure, 3),
                "pore_pressure_kpa": round(pore_pressure, 3),
                "failure_mode": failure_mode,
                "failure_label": failure_label,
                "source_reference": "Kaggle/BGS style infrastructure dataset (synthetic)",
            }
        )

    fields = [
        "record_id",
        "structure_type",
        "soil_type",
        "cover_depth_m",
        "diameter_or_width_m",
        "wall_thickness_m",
        "groundwater_state",
        "surface_settlement_mm",
        "lateral_deflection_mm",
        "earth_pressure_kpa",
        "pore_pressure_kpa",
        "failure_mode",
        "failure_label",
        "source_reference",
    ]
    write_csv(CSV_DIR / "structural_monitoring_performance.csv", fields, rows)
    return rows


def generate_synthetic_fem_dataset() -> list[dict]:
    rows: list[dict] = []
    sim = 0
    embedment_ratios = [round(x, 1) for x in np.arange(0.8, 3.01, 0.2)]

    for soil in ["Sand", "Clay"]:
        for ratio in embedment_ratios:
            for groundwater in ["Dry", "Saturated"]:
                for _ in range(35):
                    sim += 1
                    if soil == "Sand":
                        cohesion = float(RNG.uniform(1.0, 10.0))
                        friction = float(RNG.uniform(31.0, 42.0))
                        modulus = float(RNG.uniform(20.0, 120.0))
                    else:
                        cohesion = float(RNG.uniform(30.0, 160.0))
                        friction = float(RNG.uniform(17.0, 31.0))
                        modulus = float(RNG.uniform(8.0, 60.0))

                    loading_rate = float(RNG.uniform(5.0, 90.0))
                    water_factor = 1.35 if groundwater == "Saturated" else 1.0
                    soil_factor = 1.25 if soil == "Clay" else 1.0

                    max_settlement = (
                        (58.0 / ratio)
                        * soil_factor
                        * water_factor
                        * (35.0 / (modulus + 10.0))
                        * (1 + loading_rate / 140.0)
                        + float(RNG.normal(0, 2.0))
                    )
                    max_settlement = max(max_settlement, 0.0)
                    max_lateral = max(0.0, 0.72 * max_settlement + float(RNG.normal(0, 3.0)))
                    peak_pore = (
                        (55.0 if groundwater == "Dry" else 150.0)
                        + 0.7 * loading_rate
                        + (30.0 / ratio)
                        + float(RNG.normal(0, 8.0))
                    )

                    if groundwater == "Saturated" and soil == "Sand" and ratio < 1.4 and peak_pore > 170:
                        mode = "Piping"
                    elif soil == "Clay" and max_settlement > 35 and ratio < 1.6:
                        mode = "Basal heave"
                    elif max_lateral > 42:
                        mode = "Structural buckling"
                    elif max_settlement > 45:
                        mode = "Excessive settlement"
                    else:
                        mode = "Stable"

                    failed = int(mode != "Stable")
                    rows.append(
                        {
                            "sim_id": f"FEM-{sim:05d}",
                            "soil_type": soil,
                            "embedment_ratio_H_D": ratio,
                            "groundwater_condition": groundwater,
                            "cohesion_kpa": round(cohesion, 3),
                            "friction_angle_deg": round(friction, 3),
                            "elastic_modulus_mpa": round(modulus, 3),
                            "loading_rate_kpa_per_day": round(loading_rate, 3),
                            "max_settlement_mm": round(max_settlement, 3),
                            "max_lateral_displacement_mm": round(max_lateral, 3),
                            "peak_pore_pressure_kpa": round(peak_pore, 3),
                            "failure_mode": mode,
                            "failed": failed,
                            "source_reference": "FEM parametric study (synthetic)",
                        }
                    )

    fields = [
        "sim_id",
        "soil_type",
        "embedment_ratio_H_D",
        "groundwater_condition",
        "cohesion_kpa",
        "friction_angle_deg",
        "elastic_modulus_mpa",
        "loading_rate_kpa_per_day",
        "max_settlement_mm",
        "max_lateral_displacement_mm",
        "peak_pore_pressure_kpa",
        "failure_mode",
        "failed",
        "source_reference",
    ]
    write_csv(CSV_DIR / "synthetic_fem_failure_modes.csv", fields, rows)
    return rows


def generate_catalog(row_counts: dict[str, int]) -> None:
    catalog_rows = []
    for name, count in sorted(row_counts.items()):
        catalog_rows.append(
            {
                "dataset_file": name,
                "row_count": count,
                "topic": "Failure mechanism of underground structures in sandy/clay soils",
                "fabrication_type": "Synthetic, statistically plausible",
            }
        )
    write_csv(
        CSV_DIR / "dataset_catalog.csv",
        ["dataset_file", "row_count", "topic", "fabrication_type"],
        catalog_rows,
    )


def write_readme(row_counts: dict[str, int]) -> None:
    text = f"""# Underground Structure Failure Dataset Package (Synthetic)

This package was generated automatically and is fully synthetic/fabricated for
research prototyping and ML workflow testing on the topic:

> Failure Mechanism of underground structure in sandy and clay soils

## Contents

- `csv/`: tabular datasets by category
- `figures/`: visual summaries derived from generated data
- `metadata/`: notes and packaging artifacts

## Important Note

All values are simulated and do **not** represent real monitored assets.
Source names in dataset columns are thematic references to the requested
organizations/repositories but this package is a fabricated reconstruction.

## Row counts

"""
    for name, count in sorted(row_counts.items()):
        text += f"- {name}: {count} rows\n"

    (META_DIR / "README.md").write_text(text, encoding="utf-8")


def make_figures(
    physical_rows: list[dict],
    field_rows: list[dict],
    numerical_rows: list[dict],
    soil_rows: list[dict],
    met_rows: list[dict],
    synthetic_rows: list[dict],
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    # Figure 1: Settlement-time curves by flow condition.
    fig, ax = plt.subplots(figsize=(9.5, 6))
    grouped: dict[tuple[float, int], list[float]] = {}
    for r in physical_rows:
        key = (float(r["flow_condition_lpm"]), int(r["time_min"]))
        grouped.setdefault(key, []).append(float(r["settlement_mm"]))
    flows = sorted({k[0] for k in grouped.keys()})
    for flow in flows:
        x = []
        y = []
        for t in sorted({k[1] for k in grouped.keys() if k[0] == flow}):
            x.append(t)
            y.append(float(np.mean(grouped[(flow, t)])))
        ax.plot(x, y, linewidth=1.8, label=f"{int(flow)} L/min")
    ax.set_title("Physical Model: Settlement-Time by Leakage Flow")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Mean settlement (mm)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_physical_model_settlement_curves.png", dpi=180)
    plt.close(fig)

    # Figure 2: Field monitoring wave velocity profile and displacement trend.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))
    high_load_rows = [r for r in field_rows if float(r["load_stage_kpa"]) > 200]
    depths = sorted({float(r["depth_m"]) for r in high_load_rows})
    p_means = []
    s_means = []
    for d in depths:
        p_vals = [float(r["p_wave_velocity_mps"]) for r in high_load_rows if float(r["depth_m"]) == d]
        s_vals = [float(r["s_wave_velocity_mps"]) for r in high_load_rows if float(r["depth_m"]) == d]
        p_means.append(float(np.mean(p_vals)))
        s_means.append(float(np.mean(s_vals)))
    ax1.plot(p_means, depths, marker="o", label="P-wave")
    ax1.plot(s_means, depths, marker="s", label="S-wave")
    ax1.invert_yaxis()
    ax1.set_xlabel("Velocity (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Wave velocity profile (high load stage)")
    ax1.legend()

    sample = field_rows[::15]
    x = [float(r["load_stage_kpa"]) for r in sample]
    y = [float(r["extensometer_displacement_mm"]) for r in sample]
    c = [float(r["depth_m"]) for r in sample]
    sc = ax2.scatter(x, y, c=c, cmap="viridis", s=28, alpha=0.85)
    ax2.set_xlabel("Load stage (kPa)")
    ax2.set_ylabel("Extensometer displacement (mm)")
    ax2.set_title("Displacement response under loading")
    fig.colorbar(sc, ax=ax2, label="Depth (m)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_field_monitoring_profiles.png", dpi=180)
    plt.close(fig)

    # Figure 3: Heatmap of wall displacement by geometry.
    depths_unique = sorted({int(r["excavation_depth_m"]) for r in numerical_rows})
    widths_unique = sorted({int(r["excavation_width_m"]) for r in numerical_rows})
    matrix = np.zeros((len(depths_unique), len(widths_unique)))
    for i, d in enumerate(depths_unique):
        for j, w in enumerate(widths_unique):
            vals = [
                float(r["max_wall_displacement_mm"])
                for r in numerical_rows
                if int(r["excavation_depth_m"]) == d and int(r["excavation_width_m"]) == w
            ]
            matrix[i, j] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    im = ax.imshow(matrix, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(widths_unique)), labels=[str(v) for v in widths_unique])
    ax.set_yticks(range(len(depths_unique)), labels=[str(v) for v in depths_unique])
    ax.set_xlabel("Excavation width (m)")
    ax.set_ylabel("Excavation depth (m)")
    ax.set_title("Numerical Simulation: Mean Max Wall Displacement (mm)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Displacement (mm)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_numerical_simulation_heatmap.png", dpi=180)
    plt.close(fig)

    # Figure 4: Soil property distributions.
    sand_phi = [float(r["friction_angle_deg"]) for r in soil_rows if r["soil_type"].startswith("Sandy")]
    clay_phi = [float(r["friction_angle_deg"]) for r in soil_rows if r["soil_type"].startswith("Clay")]
    clay_cu = [
        float(r["undrained_shear_strength_cu_kpa"])
        for r in soil_rows
        if r["soil_type"].startswith("Clay")
    ]
    sand_rd = [
        float(r["relative_density_pct"]) for r in soil_rows if r["soil_type"].startswith("Sandy")
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5))
    ax1.boxplot([sand_phi, clay_phi], tick_labels=["Sand phi", "Clay phi"], patch_artist=True)
    ax1.set_ylabel("Friction angle (deg)")
    ax1.set_title("Friction angle distributions")

    ax2.boxplot([sand_rd, clay_cu], tick_labels=["Sand Dr (%)", "Clay cu (kPa)"], patch_artist=True)
    ax2.set_title("Relative density and undrained strength")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_soil_property_distributions.png", dpi=180)
    plt.close(fig)

    # Figure 5: Failure mode count in synthetic FEM dataset.
    mode_counts = Counter([str(r["failure_mode"]) for r in synthetic_rows])
    labels = sorted(mode_counts.keys())
    values = [mode_counts[k] for k in labels]

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.bar(labels, values, color="#3D7EA6")
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_title("Synthetic FEM Failure Mode Distribution")
    ax.set_ylabel("Case count")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_synthetic_failure_mode_counts.png", dpi=180)
    plt.close(fig)

    # Figure 6: Monthly rainfall and warning index.
    monthly_precip = {}
    monthly_warn = {}
    for r in met_rows:
        ym = str(r["date"])[:7]
        monthly_precip.setdefault(ym, []).append(float(r["precipitation_mm"]))
        monthly_warn.setdefault(ym, []).append(float(r["failure_warning_index"]))
    months = sorted(monthly_precip.keys())
    precip_series = [sum(monthly_precip[m]) for m in months]
    warn_series = [float(np.mean(monthly_warn[m])) for m in months]

    fig, ax1 = plt.subplots(figsize=(12.5, 5.3))
    ax2 = ax1.twinx()
    ax1.plot(months, precip_series, color="#006D77", linewidth=1.6, label="Monthly precipitation")
    ax2.plot(months, warn_series, color="#D62828", linewidth=1.4, label="Mean warning index")
    ax1.set_ylabel("Precipitation (mm)")
    ax2.set_ylabel("Failure warning index")
    ax1.set_title("Meteorological context: monthly water input vs warning index")
    ax1.tick_params(axis="x", rotation=70, labelsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_meteorological_context_timeseries.png", dpi=180)
    plt.close(fig)


def package_csvs_zip() -> Path:
    zip_path = OUTPUT_ROOT / "underground_failure_csv_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_path in sorted(CSV_DIR.glob("*.csv")):
            zf.write(csv_path, arcname=f"csv/{csv_path.name}")
    return zip_path


def main() -> None:
    ensure_dirs()

    physical_rows = generate_physical_model_dataset()
    field_rows = generate_field_monitoring_dataset()
    numerical_rows = generate_numerical_simulation_dataset()
    geohazard_rows = generate_geohazard_dataset()
    met_rows = generate_meteorological_dataset()
    soil_rows = generate_soil_properties_dataset()
    structural_rows = generate_structural_monitoring_dataset()
    synthetic_rows = generate_synthetic_fem_dataset()

    row_counts = {
        "physical_model_pipeline_sinkholes.csv": len(physical_rows),
        "field_monitoring_embankment_clay.csv": len(field_rows),
        "numerical_simulation_london_clay_excavation.csv": len(numerical_rows),
        "geohazard_susceptibility_bgs_style.csv": len(geohazard_rows),
        "meteorological_context_clay_embankment.csv": len(met_rows),
        "soil_properties_input_parameters.csv": len(soil_rows),
        "structural_monitoring_performance.csv": len(structural_rows),
        "synthetic_fem_failure_modes.csv": len(synthetic_rows),
    }

    generate_catalog(row_counts)
    write_readme(row_counts)
    make_figures(physical_rows, field_rows, numerical_rows, soil_rows, met_rows, synthetic_rows)
    zip_path = package_csvs_zip()

    summary_path = META_DIR / "generation_summary.txt"
    summary_lines = [
        "Synthetic Underground Structure Failure Dataset",
        "",
        "Generated CSV files and rows:",
    ]
    for file_name, count in sorted(row_counts.items()):
        summary_lines.append(f"- {file_name}: {count}")
    summary_lines.append("- dataset_catalog.csv: 8")
    summary_lines.append("")
    summary_lines.append(f"ZIP package: {zip_path.name}")
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Dataset package generated at: {OUTPUT_ROOT}")
    print(f"CSV archive: {zip_path}")


if __name__ == "__main__":
    main()
