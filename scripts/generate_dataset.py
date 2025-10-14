#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class MaterialCombo:
    name: str
    absorption_factor: float  # fraction of laser power effectively absorbed
    thermal_conductivity_factor: float  # higher means faster heat dissipation
    thermal_expansion_mismatch: float  # dimensionless relative mismatch
    imc_growth_coefficient: float  # relative rate for IMC growth
    base_strength_mpa: float  # effective base shear strength in MPa for sound nugget


MATERIAL_LIBRARY: Dict[str, MaterialCombo] = {
    "Cu-Al": MaterialCombo(
        name="Cu-Al",
        absorption_factor=0.28,
        thermal_conductivity_factor=1.30,
        thermal_expansion_mismatch=1.00,
        imc_growth_coefficient=1.00,
        base_strength_mpa=170.0,
    ),
    "Al-Al": MaterialCombo(
        name="Al-Al",
        absorption_factor=0.38,
        thermal_conductivity_factor=1.00,
        thermal_expansion_mismatch=0.45,
        imc_growth_coefficient=0.35,
        base_strength_mpa=160.0,
    ),
    "Al-Steel": MaterialCombo(
        name="Al-Steel",
        absorption_factor=0.34,
        thermal_conductivity_factor=1.10,
        thermal_expansion_mismatch=0.75,
        imc_growth_coefficient=0.65,
        base_strength_mpa=180.0,
    ),
}


def latin_hypercube_sample(num_samples: int, bounds: Dict[str, Tuple[float, float]], rng: np.random.Generator) -> pd.DataFrame:
    """Simple LHS for continuous variables in bounds.
    bounds: mapping of column -> (low, high)
    Returns DataFrame with shape (num_samples, len(bounds)).
    """
    columns = list(bounds.keys())
    num_dims = len(columns)
    # Create intervals
    cut = np.linspace(0, 1, num_samples + 1)
    # Uniformly sample within each interval for each dimension
    u = rng.uniform(size=(num_samples, num_dims))
    lhs = np.zeros_like(u)
    for j in range(num_dims):
        lhs[:, j] = u[:, j] * (cut[1:] - cut[:-1]) + cut[:-1]
        rng.shuffle(lhs[:, j])
    # Scale to bounds
    data = {}
    for j, col in enumerate(columns):
        low, high = bounds[col]
        data[col] = low + lhs[:, j] * (high - low)
    return pd.DataFrame(data)


def sample_categorical(num_samples: int, categories: List[str], probabilities: List[float], rng: np.random.Generator) -> List[str]:
    return rng.choice(categories, size=num_samples, p=np.array(probabilities) / np.sum(probabilities)).tolist()


def compute_effective_spot_size_mm(spot_size_um: np.ndarray, focus_position_mm: np.ndarray) -> np.ndarray:
    spot_mm = spot_size_um / 1000.0
    # Defocus widens the beam approximately linearly near focus (simplified)
    return spot_mm * (1.0 + 0.6 * np.abs(focus_position_mm))


def clamp(values: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.minimum(np.maximum(values, low), high)


def generate_parameters(num_rows: int, materials: List[str], rng: np.random.Generator) -> pd.DataFrame:
    # Continuous bounds
    bounds = {
        "Laser_Power_W": (500.0, 3000.0),
        "Welding_Speed_mm_s": (10.0, 200.0),
        "Pulse_Frequency_Hz": (0.0, 500.0),
        "Pulse_Duration_ms": (0.0, 10.0),
        "Focus_Position_mm": (-1.0, 1.0),
        "Spot_Size_um": (50.0, 500.0),
        "Clamping_Pressure_MPa": (0.2, 4.0),
        "Shield_Gas_Flow_L_min": (0.0, 20.0),
        "T_Top_mm": (0.05, 0.5),
        "T_Bottom_mm": (0.05, 0.5),
        "Overlap_Distance_mm": (0.0, 5.0),
    }
    df = latin_hypercube_sample(num_rows, bounds, rng)

    # Categorical
    df["Shield_Gas_Type"] = sample_categorical(
        num_rows, ["None", "Ar", "N2", "He"], [0.2, 0.5, 0.1, 0.2], rng
    )
    df["Joint_Type"] = sample_categorical(num_rows, ["Lap", "Butt"], [0.7, 0.3], rng)
    df["Material_Combination"] = sample_categorical(num_rows, materials, [1.0] * len(materials), rng)

    # Enforce geometry constraints
    is_butt = df["Joint_Type"] == "Butt"
    df.loc[is_butt, "Overlap_Distance_mm"] = 0.0

    return df


def model_outputs(parameters: pd.DataFrame, source: str, rng: np.random.Generator) -> pd.DataFrame:
    # Derived inputs
    power = parameters["Laser_Power_W"].to_numpy()
    speed = parameters["Welding_Speed_mm_s"].to_numpy()
    freq = parameters["Pulse_Frequency_Hz"].to_numpy()
    duration_ms = parameters["Pulse_Duration_ms"].to_numpy()
    focus_pos = parameters["Focus_Position_mm"].to_numpy()
    spot_um = parameters["Spot_Size_um"].to_numpy()
    clamp_pressure = parameters["Clamping_Pressure_MPa"].to_numpy()
    gas_flow = parameters["Shield_Gas_Flow_L_min"].to_numpy()
    t_top = parameters["T_Top_mm"].to_numpy()
    t_bottom = parameters["T_Bottom_mm"].to_numpy()

    num = len(parameters)

    # Effective duty cycle (0..1)
    duty_cycle = np.clip((freq * duration_ms) / 1000.0, 0.0, 1.0)

    # Effective spot and area
    spot_mm_eff = compute_effective_spot_size_mm(spot_um, focus_pos)
    area_mm2 = math.pi * (spot_mm_eff / 2.0) ** 2

    thickness_total = t_top + t_bottom
    thickness_avg = 0.5 * (t_top + t_bottom)

    # Material coefficients per row
    mat_props = [MATERIAL_LIBRARY[m] for m in parameters["Material_Combination"].tolist()]
    absorption = np.array([m.absorption_factor for m in mat_props])
    k_cond = np.array([m.thermal_conductivity_factor for m in mat_props])
    mismatch = np.array([m.thermal_expansion_mismatch for m in mat_props])
    imc_coeff = np.array([m.imc_growth_coefficient for m in mat_props])
    base_strength_mpa = np.array([m.base_strength_mpa for m in mat_props])

    # Line energy J/mm (approx.)
    line_energy = (power * duty_cycle) / np.maximum(speed, 1e-6)
    # Energy density (scaled) considering spot area and absorption
    energy_density = (line_energy * absorption) / np.maximum(area_mm2, 1e-9)

    # Gas effectiveness (Ar, He best; None worst; N2 mediocre esp. for Al)
    gas_type = parameters["Shield_Gas_Type"].to_numpy()
    gas_quality = np.where(gas_type == "Ar", 1.0, np.where(gas_type == "He", 0.9, np.where(gas_type == "N2", 0.6, 0.3)))
    gas_quality *= np.clip(gas_flow / 10.0, 0.0, 1.0)

    # Weld morphology models (saturating behavior vs energy density)
    # Nugget width (mm)
    k1, k2 = 3.5, 0.015  # scale and saturation rate
    nugget_width = k1 * (1.0 - np.exp(-k2 * energy_density))
    nugget_width *= 1.0 - 0.25 * np.tanh((thickness_avg - 0.3) / 0.2)  # thinner sheets respond more dramatically

    # Penetration depth (mm), limited by total thickness
    p1, p2 = 0.010, 0.005
    penetration = p1 * energy_density / (1.0 + 0.6 * (spot_mm_eff / 0.2))
    penetration *= 1.0 / (1.0 + p2 * speed)
    penetration /= (0.7 + 0.3 * k_cond)  # high conductivity (e.g., Cu) reduces penetration
    penetration = clamp(penetration, 0.0, thickness_total)

    # HAZ width (mm)
    haz = 0.15 * np.sqrt(np.maximum(line_energy, 1e-6)) / (1.0 + 0.3 * k_cond)
    haz *= 0.8 + 0.4 * np.clip(clamp_pressure / 4.0, 0.0, 1.0)

    # Surface condition
    spatter_level = np.clip(0.15 + 0.9 * (energy_density / (energy_density.mean() + 1e-6)) ** 1.2 - 0.2 * gas_quality, 0.0, 1.5)
    discoloration_level = np.clip(0.1 + 0.7 * (1.0 - gas_quality) + 0.2 * np.abs(focus_pos), 0.0, 1.5)

    # Defects probabilities
    high_ed = (energy_density > np.quantile(energy_density, 0.8)).astype(float)
    low_ed = (energy_density < np.quantile(energy_density, 0.2)).astype(float)
    porosity_pct = np.clip(
        2.0 + 8.0 * (1.0 - gas_quality) + 6.0 * low_ed + 3.0 * (spot_mm_eff > 0.3).astype(float),
        0.0,
        40.0,
    )
    # Crack propensity rises with mismatch, IMC, and low penetration ratio
    penetration_ratio = np.divide(penetration, thickness_total, out=np.zeros_like(penetration), where=thickness_total > 0)
    crack_prob = 0.05 + 0.60 * (1.0 - penetration_ratio) + 0.50 * mismatch + 0.30 * (imc_coeff)
    crack_prob += 0.10 * high_ed
    crack_prob = np.clip(crack_prob, 0.0, 0.95)
    crack_presence = rng.random(num) < crack_prob

    undercut_prob = np.clip(0.05 + 0.35 * (speed / 200.0) + 0.15 * (focus_pos > 0).astype(float), 0.0, 0.85)
    undercut_presence = rng.random(num) < undercut_prob

    expulsion_prob = np.clip(0.03 + 0.40 * high_ed + 0.10 * (spot_um < 80).astype(float), 0.0, 0.90)
    expulsion_presence = rng.random(num) < expulsion_prob

    # Mechanical properties (N)
    nugget_area_mm2 = math.pi * (nugget_width / 2.0) ** 2
    structural_factor = 0.6 + 0.6 * penetration_ratio  # more penetration -> better load sharing
    defect_penalty = (1.0 - 0.5 * crack_prob) * (1.0 - 0.01 * porosity_pct) * (1.0 - 0.25 * expulsion_prob)
    sigma_eff_mpa = base_strength_mpa * structural_factor * defect_penalty
    # Convert MPa * mm^2 -> N (1 MPa = 1 N/mm^2)
    tensile_shear_n = sigma_eff_mpa * nugget_area_mm2

    peel_strength_n = 0.35 * tensile_shear_n * (1.0 - 0.3 * undercut_prob)

    # Electrical contact resistance (micro-ohms)
    base_resist = 12.0 / np.maximum(nugget_area_mm2, 1e-6)  # inversely with area
    oxide_penalty = 8.0 * (1.0 - gas_quality)
    crack_penalty = 10.0 * crack_prob
    contact_res_uohm = np.clip(base_resist + oxide_penalty + crack_penalty, 3.0, 120.0)

    # Extreme-temperature behavior
    # Thermal cycling degradation depends on mismatch and defect state
    cycles_profile = 500  # assumed number of cycles
    degradation_strength_pct = np.clip(5.0 + 40.0 * mismatch + 20.0 * crack_prob + 0.15 * cycles_profile / 1000.0, 0.0, 85.0)
    resistance_increase_pct = np.clip(4.0 + 35.0 * mismatch + 0.4 * porosity_pct + 20.0 * crack_prob, 0.0, 150.0)

    # Cycles to failure (heuristic)
    damage_index = 0.3 * mismatch + 0.4 * crack_prob + 0.2 * (porosity_pct / 40.0) + 0.2 * (degradation_strength_pct / 100.0)
    cycles_to_failure = np.clip(1500.0 * (1.3 - damage_index), 50.0, 3000.0)

    # Creep time to failure at 100C (hours) under 0.5 UTS load
    creep_base = 100.0 + 400.0 * (sigma_eff_mpa / (base_strength_mpa + 1e-6))
    creep_time_h = np.clip(creep_base / (1.2 + 0.5 * mismatch), 5.0, 5000.0)

    # IMC growth after static aging (500h @ 120C)
    aging_time_h = 500.0
    aging_temp_k = 120.0 + 273.15
    # Parabolic growth x = k * sqrt(t), scale with material coefficient and energy density (hotter welds promote IMC)
    imc_thickness_um = np.clip(0.2 + 1.8 * np.sqrt(aging_time_h / 1000.0) * imc_coeff * (0.7 + 0.3 * (energy_density / (energy_density.mean() + 1e-6))), 0.0, 12.0)

    # Grain size change index (0..1)
    grain_change = np.clip(0.2 + 0.6 * (haz / (haz.mean() + 1e-6)) / (1.0 + 0.5 * k_cond), 0.0, 1.0)

    # Apply source-specific noise (experimental noisier than simulation)
    if source == "Experimental":
        noise_scale = 0.03
        big_noise_scale = 0.08
    else:  # Simulation
        noise_scale = 0.01
        big_noise_scale = 0.03

    def jitter(x: np.ndarray, scale: float) -> np.ndarray:
        return x * (1.0 + rng.normal(0.0, scale, size=x.shape))

    outputs = pd.DataFrame(
        {
            "Nugget_Width_mm": np.clip(jitter(nugget_width, noise_scale), 0.05, 5.0),
            "Penetration_Depth_mm": np.clip(jitter(penetration, noise_scale), 0.0, thickness_total),
            "HAZ_Width_mm": np.clip(jitter(haz, noise_scale), 0.01, 5.0),
            "Spatter_Level": np.clip(jitter(spatter_level, noise_scale), 0.0, 2.0),
            "Discoloration_Level": np.clip(jitter(discoloration_level, noise_scale), 0.0, 2.0),
            "Porosity_Area_Pct": np.clip(jitter(porosity_pct, big_noise_scale), 0.0, 60.0),
            "Crack_Presence": crack_presence.astype(int),
            "Undercut_Presence": undercut_presence.astype(int),
            "Expulsion_Presence": expulsion_presence.astype(int),
            "Tensile_Shear_Strength_N": np.clip(jitter(tensile_shear_n, big_noise_scale), 100.0, 12000.0),
            "Peel_Strength_N": np.clip(jitter(peel_strength_n, big_noise_scale), 50.0, 6000.0),
            "Contact_Resistance_uOhm": np.clip(jitter(contact_res_uohm, big_noise_scale), 2.0, 300.0),
            "ThermalCycles_Profile_Cycles": np.full(num, cycles_profile, dtype=float),
            "Strength_Degradation_Pct": np.clip(jitter(degradation_strength_pct, big_noise_scale), 0.0, 95.0),
            "Resistance_Increase_Pct": np.clip(jitter(resistance_increase_pct, big_noise_scale), 0.0, 200.0),
            "Cycles_to_Failure": np.clip(jitter(cycles_to_failure, big_noise_scale), 10.0, 5000.0),
            "Creep_TimeToFailure_h": np.clip(jitter(creep_time_h, big_noise_scale), 1.0, 10000.0),
            "IMC_Thickness_post_aging_um": np.clip(jitter(imc_thickness_um, noise_scale), 0.0, 20.0),
            "Grain_Size_Change_Index": np.clip(jitter(grain_change, noise_scale), 0.0, 1.0),
        }
    )

    # Measurement uncertainty columns for experimental subset only
    if source == "Experimental":
        outputs["Tensile_Shear_Strength_SD_N"] = 0.04 * outputs["Tensile_Shear_Strength_N"].to_numpy()
        outputs["Contact_Resistance_SD_uOhm"] = 0.05 * outputs["Contact_Resistance_uOhm"].to_numpy()
    else:
        outputs["Tensile_Shear_Strength_SD_N"] = np.nan
        outputs["Contact_Resistance_SD_uOhm"] = np.nan

    return outputs


def build_dataset(n_sim: int, n_exp: int, materials: List[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Simulation tier
    sim_params = generate_parameters(n_sim, materials, rng)
    sim_outputs = model_outputs(sim_params, source="Simulation", rng=rng)
    sim = pd.concat([sim_params, sim_outputs], axis=1)
    sim["Data_Source"] = "Simulation"

    # Experimental tier
    exp_params = generate_parameters(n_exp, materials, rng)
    exp_outputs = model_outputs(exp_params, source="Experimental", rng=rng)
    exp = pd.concat([exp_params, exp_outputs], axis=1)
    exp["Data_Source"] = "Experimental"

    # Assign Weld_IDs
    sim["Weld_ID"] = [f"S-{i:05d}" for i in range(1, len(sim) + 1)]
    exp["Weld_ID"] = [f"W-{i:05d}" for i in range(1, len(exp) + 1)]

    df = pd.concat([exp, sim], axis=0, ignore_index=True)

    # Order columns: ID, inputs, outputs, source
    input_cols = [
        "Weld_ID",
        "Material_Combination",
        "Joint_Type",
        "T_Top_mm",
        "T_Bottom_mm",
        "Overlap_Distance_mm",
        "Laser_Power_W",
        "Welding_Speed_mm_s",
        "Pulse_Frequency_Hz",
        "Pulse_Duration_ms",
        "Focus_Position_mm",
        "Spot_Size_um",
        "Clamping_Pressure_MPa",
        "Shield_Gas_Type",
        "Shield_Gas_Flow_L_min",
    ]

    output_cols = [
        "Nugget_Width_mm",
        "Penetration_Depth_mm",
        "HAZ_Width_mm",
        "Spatter_Level",
        "Discoloration_Level",
        "Porosity_Area_Pct",
        "Crack_Presence",
        "Undercut_Presence",
        "Expulsion_Presence",
        "Tensile_Shear_Strength_N",
        "Peel_Strength_N",
        "Contact_Resistance_uOhm",
        "ThermalCycles_Profile_Cycles",
        "Strength_Degradation_Pct",
        "Resistance_Increase_Pct",
        "Cycles_to_Failure",
        "Creep_TimeToFailure_h",
        "IMC_Thickness_post_aging_um",
        "Grain_Size_Change_Index",
        "Tensile_Shear_Strength_SD_N",
        "Contact_Resistance_SD_uOhm",
    ]

    other_cols = ["Data_Source"]

    ordered_cols = input_cols + output_cols + other_cols
    df = df[ordered_cols]
    return df


def write_schema(df: pd.DataFrame, path: str) -> None:
    schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"columns": schema, "num_rows": int(df.shape[0])}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a multi-fidelity welding inverse-design dataset.")
    parser.add_argument("--n-sim", type=int, default=8000, help="Number of simulated rows")
    parser.add_argument("--n-exp", type=int, default=250, help="Number of experimental-like rows")
    parser.add_argument(
        "--materials",
        type=str,
        default="Cu-Al,Al-Al,Al-Steel",
        help="Comma-separated material combinations to include",
    )
    parser.add_argument("--out-dir", type=str, default="/workspace/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--export-formats",
        type=str,
        default="csv,parquet",
        help="Comma-separated formats: csv,parquet",
    )
    parser.add_argument("--zip", dest="make_zip", action="store_true", help="Zip outputs into a bundle")
    parser.add_argument("--no-zip", dest="make_zip", action="store_false", help="Do not zip outputs")
    parser.set_defaults(make_zip=True)

    args = parser.parse_args()

    materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    for m in materials:
        if m not in MATERIAL_LIBRARY:
            raise ValueError(f"Unknown material combination: {m}")

    os.makedirs(args.out_dir, exist_ok=True)

    print("Generating dataset ...", file=sys.stderr)
    df = build_dataset(args.n_sim, args.n_exp, materials, args.seed)

    base = os.path.join(args.out_dir, "welding_inverse_design_master")
    formats = {fmt.strip().lower() for fmt in args.export_formats.split(",")}

    paths = []
    if "csv" in formats:
        csv_path = base + ".csv"
        df.to_csv(csv_path, index=False)
        paths.append(csv_path)
    if "parquet" in formats:
        pq_path = base + ".parquet"
        df.to_parquet(pq_path, index=False)
        paths.append(pq_path)

    schema_path = base + "__schema.json"
    write_schema(df, schema_path)
    paths.append(schema_path)

    if args.make_zip:
        zip_path = os.path.join(args.out_dir, "welding_inverse_design_dataset.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=os.path.basename(p))
        print(f"Wrote zip: {zip_path}")
    else:
        zip_path = None

    print("Done.")
    print("Outputs:")
    for p in paths:
        print(p)
    if zip_path:
        print(zip_path)


if __name__ == "__main__":
    main()
