import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class Material:
    name: str
    youngs_modulus_gpa: float
    poisson_ratio: float
    fracture_toughness_mpa_sqrt_m: float
    cte_k_inv: float


@dataclass
class Substrate:
    name: str
    youngs_modulus_gpa: float
    poisson_ratio: float
    cte_k_inv: float
    interface_toughness_j_m2: float


def get_material_libraries(seed: int = 42) -> Tuple[Dict[str, Material], Dict[str, Substrate]]:
    rng = np.random.default_rng(seed)

    films = {
        "Al2O3": Material("Al2O3", 380.0, 0.22, 3.5, 8.5e-6),
        "YSZ": Material("YSZ", 200.0, 0.23, 2.0, 10.5e-6),
        "SiC": Material("SiC", 450.0, 0.17, 3.0, 4.3e-6),
        "ZrO2": Material("ZrO2", 210.0, 0.28, 2.5, 10.0e-6),
        "Si3N4": Material("Si3N4", 300.0, 0.26, 4.0, 3.2e-6),
        "TiO2": Material("TiO2", 230.0, 0.27, 1.8, 8.6e-6),
    }

    substrates = {
        "Steel": Substrate("Steel", 210.0, 0.30, 12.0e-6, 12.0),
        "Si": Substrate("Si", 170.0, 0.28, 2.6e-6, 9.0),
        "Glass": Substrate("Glass", 70.0, 0.22, 8.0e-6, 5.0),
        "Ni": Substrate("Ni", 200.0, 0.31, 13.3e-6, 10.0),
        "Inconel": Substrate("Inconel", 210.0, 0.29, 13.0e-6, 14.0),
    }

    return films, substrates


def compute_effective_modulus(youngs_modulus_gpa: np.ndarray, porosity: np.ndarray) -> np.ndarray:
    # Use a Gibson-Ashby-type empirical relation for elastic modulus vs porosity
    # E_eff = E_0 * (1 - 1.9p + 0.9p^2), clipped to non-negative
    eff = youngs_modulus_gpa * (1.0 - 1.9 * porosity + 0.9 * porosity ** 2)
    return np.clip(eff, a_min=0.01 * youngs_modulus_gpa, a_max=None)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(num_samples: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    films, substrates = get_material_libraries(seed)

    film_names = np.array(list(films.keys()))
    sub_names = np.array(list(substrates.keys()))

    film_choice = rng.choice(film_names, size=num_samples)
    sub_choice = rng.choice(sub_names, size=num_samples)

    film_props = np.vectorize(lambda n: films[n].youngs_modulus_gpa)(film_choice)
    film_nu = np.vectorize(lambda n: films[n].poisson_ratio)(film_choice)
    film_cte = np.vectorize(lambda n: films[n].cte_k_inv)(film_choice)
    film_kic = np.vectorize(lambda n: films[n].fracture_toughness_mpa_sqrt_m)(film_choice)

    sub_props = np.vectorize(lambda n: substrates[n].youngs_modulus_gpa)(sub_choice)
    sub_nu = np.vectorize(lambda n: substrates[n].poisson_ratio)(sub_choice)
    sub_cte = np.vectorize(lambda n: substrates[n].cte_k_inv)(sub_choice)
    interface_gc = np.vectorize(lambda n: substrates[n].interface_toughness_j_m2)(sub_choice)

    # Processing parameters
    sinter_temp_c = rng.uniform(1200.0, 1500.0, size=num_samples)
    cooling_rate_c_per_min = rng.uniform(1.0, 10.0, size=num_samples)

    # Porosity levels (volume fraction)
    porosity = rng.beta(2.0, 8.0, size=num_samples)  # mostly low porosity, long tail

    # TEC mismatch centered around target value with small variability
    delta_alpha_k_inv = rng.normal(loc=2.3e-6, scale=0.3e-6, size=num_samples)
    delta_alpha_k_inv = np.clip(delta_alpha_k_inv, 0.2e-6, 6e-6)

    # Geometric parameters
    film_thickness_um = rng.uniform(100.0, 1000.0, size=num_samples)
    sub_thickness_mm = rng.uniform(0.5, 2.0, size=num_samples)

    # Effective properties
    film_e_eff_gpa = compute_effective_modulus(film_props, porosity)

    # Thermal history
    ambient_c = 25.0
    delta_t_c = sinter_temp_c - ambient_c

    # Thermal strain and stress in film (simplified bi-material model)
    thermal_strain = delta_alpha_k_inv * delta_t_c
    gradient_factor = 1.0 + 0.06 * (cooling_rate_c_per_min - 1.0)  # faster cooling magnifies gradients
    stress_concentration = rng.uniform(1.0, 1.5, size=num_samples)  # geometric/heterogeneity factor

    # Plane stress approximation for thin films: sigma = E_eff * eps / (1 - nu)
    thermal_stress_mpa = (
        (film_e_eff_gpa * 1000.0)
        * thermal_strain
        / (1.0 - np.clip(film_nu, 0.05, 0.49))
        * gradient_factor
        * stress_concentration
    )

    # Fracture mechanics-inspired crack initiation criterion
    # sigma_c ~ K_IC / (Y * sqrt(pi * a)) with Y ~ 1.1 and flaw size scaling with porosity
    flaw_size_m = (porosity * rng.uniform(2e-6, 50e-6, size=num_samples)) + rng.uniform(0.05e-6, 5e-6, size=num_samples)
    y_geom = 1.1
    sigma_crit_mpa = (film_kic / (y_geom * np.sqrt(np.pi * np.clip(flaw_size_m, 1e-8, None))))
    # Convert K_IC [MPa*sqrt(m)] / sqrt(m) -> MPa OK

    crack_drive_ratio = np.clip(thermal_stress_mpa / np.clip(sigma_crit_mpa, 1e-3, None), 0.0, 20.0)
    crack_initiation_risk = sigmoid(1.5 * (crack_drive_ratio - 1.0))  # >1 increases risk sharply

    # Delamination via energy release rate proxy: G ~ sigma^2 * h / E
    film_thickness_m = film_thickness_um * 1e-6
    film_e_pa = film_e_eff_gpa * 1e9
    g_release = (thermal_stress_mpa * 1e6) ** 2 * film_thickness_m / np.clip(film_e_pa, 1e6, None)
    # Interface toughness in J/m^2; mismatch and cooling gradients amplify effective driving force
    g_effective = g_release * (1.0 + 0.3 * (gradient_factor - 1.0)) * (1.0 + 0.5 * (delta_alpha_k_inv / 2.3e-6 - 1.0))
    delam_ratio = g_effective / (np.array(interface_gc) + 1e-6)
    delamination_probability = sigmoid(1.2 * (delam_ratio - 1.0))

    # Stress hotspot score combines concentration factors, porosity, and stress gradient
    hotspot_base = (thermal_stress_mpa / (np.median(thermal_stress_mpa) + 1e-6))
    hotspot_score = sigmoid(0.8 * hotspot_base) * (0.6 + 0.4 * (porosity / (porosity.max() + 1e-9))) * (0.9 + 0.2 * (gradient_factor - 1.0))
    hotspot_score = np.clip(hotspot_score, 0.0, 1.0)

    # Binary labels (optional) using calibrated thresholds
    crack_label = (crack_initiation_risk > 0.5).astype(int)
    delam_label = (delamination_probability > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "sample_id": np.arange(num_samples, dtype=int),
            "film_material": film_choice,
            "substrate": sub_choice,
            "sintering_temperature_c": sinter_temp_c,
            "cooling_rate_c_per_min": cooling_rate_c_per_min,
            "delta_alpha_k_inv": delta_alpha_k_inv,
            "porosity_fraction": porosity,
            "film_thickness_um": film_thickness_um,
            "substrate_thickness_mm": sub_thickness_mm,
            "film_youngs_modulus_gpa": film_props,
            "film_poisson_ratio": film_nu,
            "film_fracture_toughness_mpa_sqrt_m": film_kic,
            "substrate_youngs_modulus_gpa": sub_props,
            "substrate_poisson_ratio": sub_nu,
            "substrate_cte_k_inv": sub_cte,
            "interface_toughness_j_m2": interface_gc,
            "delta_t_c": delta_t_c,
            "thermal_strain": thermal_strain,
            "film_effective_modulus_gpa": film_e_eff_gpa,
            "gradient_factor": gradient_factor,
            "thermal_stress_mpa": thermal_stress_mpa,
            "stress_hotspot_score": hotspot_score,
            "crack_initiation_risk": crack_initiation_risk,
            "delamination_probability": delamination_probability,
            "crack_initiation_label": crack_label,
            "delamination_label": delam_label,
        }
    )

    return df


def split_and_save(df: pd.DataFrame, outdir: str, seed: int = 42) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n = len(df)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    df.iloc[train_idx].to_csv(os.path.join(outdir, "train.csv"), index=False)
    df.iloc[val_idx].to_csv(os.path.join(outdir, "val.csv"), index=False)
    df.iloc[test_idx].to_csv(os.path.join(outdir, "test.csv"), index=False)

    return {
        "train": os.path.join(outdir, "train.csv"),
        "val": os.path.join(outdir, "val.csv"),
        "test": os.path.join(outdir, "test.csv"),
    }


def create_validation_files(df: pd.DataFrame, outdir: str, seed: int = 42, n_dic: int = 250, n_xrd: int = 250) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    valdir = os.path.join(outdir, "validation")
    os.makedirs(valdir, exist_ok=True)

    # DIC synthetic: sampled subset with strain stats and hotspot density
    dic_idx = rng.choice(df.index.values, size=min(n_dic, len(df)), replace=False)
    dic_df = df.loc[dic_idx, [
        "sample_id",
        "thermal_strain",
        "stress_hotspot_score",
        "porosity_fraction",
        "cooling_rate_c_per_min",
    ]].copy()

    noise = rng.normal(0.0, 0.0001, size=len(dic_df))
    dic_df.rename(columns={"thermal_strain": "simulated_mean_strain"}, inplace=True)
    dic_df["measured_mean_strain"] = dic_df["simulated_mean_strain"] * (1.0 + rng.normal(0.0, 0.03, size=len(dic_df))) + 0.05 * dic_df["porosity_fraction"] + noise
    dic_df["measured_strain_std"] = np.abs(dic_df["simulated_mean_strain"]) * (0.05 + 0.02 * dic_df["cooling_rate_c_per_min"]) * (1.0 + rng.normal(0.0, 0.2, size=len(dic_df)))
    dic_df["measured_max_strain"] = dic_df["measured_mean_strain"] + 2.5 * dic_df["measured_strain_std"]
    dic_df["hotspot_density_per_cm2"] = 5.0 * dic_df["stress_hotspot_score"] * (1.0 + 0.3 * (dic_df["cooling_rate_c_per_min"] - 1.0)) + rng.normal(0.0, 0.5, size=len(dic_df))
    dic_df = dic_df[[
        "sample_id",
        "measured_mean_strain",
        "measured_strain_std",
        "measured_max_strain",
        "hotspot_density_per_cm2",
    ]]

    dic_path = os.path.join(valdir, "dic_synthetic.csv")
    dic_df.to_csv(dic_path, index=False)

    # XRD synthetic: residual stress with measurement noise and slight bias
    xrd_idx = rng.choice(df.index.values, size=min(n_xrd, len(df)), replace=False)
    xrd_df = df.loc[xrd_idx, ["sample_id", "thermal_stress_mpa", "film_material"]].copy()
    # Instrument-dependent bias per material
    mat_bias = {m: b for m, b in zip(df["film_material"].unique(), rng.normal(0.0, 5.0, size=df["film_material"].nunique()))}
    xrd_df["residual_stress_mpa"] = xrd_df["thermal_stress_mpa"] * (1.0 + rng.normal(0.0, 0.05, size=len(xrd_df))) + xrd_df["film_material"].map(mat_bias)
    xrd_df = xrd_df[["sample_id", "residual_stress_mpa"]]
    xrd_path = os.path.join(valdir, "xrd_synthetic.csv")
    xrd_df.to_csv(xrd_path, index=False)

    return {"dic": dic_path, "xrd": xrd_path}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ANN/PINN sintering dataset with physics-inspired labels.")
    parser.add_argument("--n", type=int, default=12000, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--outdir", type=str, default="./data", help="Output directory for CSV files")
    parser.add_argument("--no_validation", action="store_true", help="Skip generating DIC/XRD validation files")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = generate_dataset(args.n, seed=args.seed)
    paths = split_and_save(df, args.outdir, seed=args.seed)

    val_paths = {}
    if not args.no_validation:
        val_paths = create_validation_files(df, args.outdir, seed=args.seed)

    manifest = {
        "num_samples": int(args.n),
        "seed": int(args.seed),
        "outputs": paths,
        "validation": val_paths,
    }
    with open(os.path.join(args.outdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

