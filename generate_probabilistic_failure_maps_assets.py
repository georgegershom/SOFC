#!/usr/bin/env python3
"""
Synthetic dataset generator for:
Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs

This script fabricates reproducible CSV datasets and figures when experimental data is missing.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"

# Reproducibility seed
RNG = np.random.default_rng(20260211)

# Proxy RT statistics from uncertainty table assumptions
INTERFACE_PARAMS = {
    "YSZ|GDC": {"mu_rt": 2.15, "sigma_rt": 0.40, "weibull_rt": 7.5},
    "GDC|LSCF": {"mu_rt": 1.00, "sigma_rt": 0.20, "weibull_rt": 6.0},
}

ELASTIC_MODULUS_RATIO_800C_TO_RT = 170.0 / 200.0
STD_SCALING_800C = 1.0  # Conservative: keep RT spread at high temperature
BASELINE_RHO = 0.0
SENSITIVITY_RHO = 0.5

N_MICRO_SAMPLES_PER_INTERFACE_TEMP = 250
N_CORRELATION_SAMPLES = 500
N_MAP_MONTE_CARLO = 20_000


def truncated_normal(mean: float, std: float, size: int, lower: float = 0.02) -> np.ndarray:
    """Draw positive-valued samples from a normal distribution by rejection sampling."""
    samples = RNG.normal(loc=mean, scale=std, size=size)
    bad = samples <= lower
    while np.any(bad):
        samples[bad] = RNG.normal(loc=mean, scale=std, size=int(np.sum(bad)))
        bad = samples <= lower
    return samples


def build_material_properties_df() -> pd.DataFrame:
    rows: list[dict] = []

    for interface, params in INTERFACE_PARAMS.items():
        mu_rt = params["mu_rt"]
        sigma_rt = params["sigma_rt"]
        cv_rt = sigma_rt / mu_rt

        mu_800 = mu_rt * ELASTIC_MODULUS_RATIO_800C_TO_RT
        sigma_800 = sigma_rt * STD_SCALING_800C
        cv_800 = sigma_800 / mu_800

        rows.append(
            {
                "record_id": f"{interface.replace('|', '_')}_RT",
                "parameter_group": "interfacial_fracture_energy",
                "symbol": "Gc",
                "interface": interface,
                "temperature_C": 25,
                "distribution": "normal_truncated_positive",
                "mean_J_per_m2": round(mu_rt, 6),
                "std_dev_J_per_m2": round(sigma_rt, 6),
                "coefficient_of_variation": round(cv_rt, 6),
                "weibull_modulus_proxy": params["weibull_rt"],
                "value": np.nan,
                "unit": "J/m^2",
                "missing_data_flag": "no",
                "assumption_note": "Proxy RT statistics used as baseline input.",
                "source": "fabricated_from_user_prompt",
            }
        )

        rows.append(
            {
                "record_id": f"{interface.replace('|', '_')}_800C_estimated",
                "parameter_group": "interfacial_fracture_energy",
                "symbol": "Gc",
                "interface": interface,
                "temperature_C": 800,
                "distribution": "normal_truncated_positive",
                "mean_J_per_m2": round(mu_800, 6),
                "std_dev_J_per_m2": round(sigma_800, 6),
                "coefficient_of_variation": round(cv_800, 6),
                "weibull_modulus_proxy": params["weibull_rt"],
                "value": np.nan,
                "unit": "J/m^2",
                "missing_data_flag": "yes",
                "assumption_note": (
                    "Mean scaled by elastic modulus ratio E(800C)/E(RT); "
                    "std dev kept equal to RT by conservative assumption."
                ),
                "source": "fabricated_assumption_due_to_missing_highT_data",
            }
        )

    rows.extend(
        [
            {
                "record_id": "MISSING_DATASET_01_rho",
                "parameter_group": "interfacial_correlation",
                "symbol": "rho",
                "interface": "YSZ|GDC <-> GDC|LSCF",
                "temperature_C": np.nan,
                "distribution": "assumed_constant",
                "mean_J_per_m2": np.nan,
                "std_dev_J_per_m2": np.nan,
                "coefficient_of_variation": np.nan,
                "weibull_modulus_proxy": np.nan,
                "value": BASELINE_RHO,
                "unit": "-",
                "missing_data_flag": "yes",
                "assumption_note": "Default uncorrelated interfaces due to lack of paired failure evidence.",
                "source": "fabricated_missing_dataset_proxy",
            },
            {
                "record_id": "MISSING_DATASET_02_std_scaling",
                "parameter_group": "high_temperature_variance_scaling",
                "symbol": "eta_sigma",
                "interface": "all",
                "temperature_C": 800,
                "distribution": "assumed_constant",
                "mean_J_per_m2": np.nan,
                "std_dev_J_per_m2": np.nan,
                "coefficient_of_variation": np.nan,
                "weibull_modulus_proxy": np.nan,
                "value": STD_SCALING_800C,
                "unit": "-",
                "missing_data_flag": "yes",
                "assumption_note": "Std dev at 800C held equal to RT std dev.",
                "source": "fabricated_missing_dataset_proxy",
            },
            {
                "record_id": "E_ratio_800C_over_RT",
                "parameter_group": "modulus_scaling_reference",
                "symbol": "E800_over_ERT",
                "interface": "all",
                "temperature_C": np.nan,
                "distribution": "deterministic",
                "mean_J_per_m2": np.nan,
                "std_dev_J_per_m2": np.nan,
                "coefficient_of_variation": np.nan,
                "weibull_modulus_proxy": np.nan,
                "value": round(ELASTIC_MODULUS_RATIO_800C_TO_RT, 6),
                "unit": "-",
                "missing_data_flag": "no",
                "assumption_note": "Used to scale mean fracture energy when high-T measurements are absent.",
                "source": "prompt_context_reference",
            },
        ]
    )

    return pd.DataFrame(rows)


def build_micro_cantilever_df() -> pd.DataFrame:
    rows: list[dict] = []
    sample_id = 1

    for interface, params in INTERFACE_PARAMS.items():
        for temperature in (25, 800):
            if temperature == 25:
                mu = params["mu_rt"]
                sigma = params["sigma_rt"]
                assumption_tag = "RT_proxy_distribution"
            else:
                mu = params["mu_rt"] * ELASTIC_MODULUS_RATIO_800C_TO_RT
                sigma = params["sigma_rt"] * STD_SCALING_800C
                assumption_tag = "HT_estimated_from_RT_missing_data_assumption"

            samples = truncated_normal(mu, sigma, N_MICRO_SAMPLES_PER_INTERFACE_TEMP)

            for value in samples:
                rows.append(
                    {
                        "sample_id": f"MC_{sample_id:05d}",
                        "interface": interface,
                        "temperature_C": temperature,
                        "Gc_J_per_m2": round(float(value), 6),
                        "distribution_model": "normal_truncated_positive",
                        "source_type": "synthetic_fabricated",
                        "assumption_tag": assumption_tag,
                        "seed": 20260211,
                    }
                )
                sample_id += 1

    return pd.DataFrame(rows)


def sample_independent_pairs(n_samples: int) -> np.ndarray:
    ysz = truncated_normal(
        INTERFACE_PARAMS["YSZ|GDC"]["mu_rt"],
        INTERFACE_PARAMS["YSZ|GDC"]["sigma_rt"],
        n_samples,
    )
    lscf = truncated_normal(
        INTERFACE_PARAMS["GDC|LSCF"]["mu_rt"],
        INTERFACE_PARAMS["GDC|LSCF"]["sigma_rt"],
        n_samples,
    )
    return np.column_stack([ysz, lscf])


def sample_correlated_pairs(n_samples: int, rho: float) -> np.ndarray:
    mu1 = INTERFACE_PARAMS["YSZ|GDC"]["mu_rt"]
    sd1 = INTERFACE_PARAMS["YSZ|GDC"]["sigma_rt"]
    mu2 = INTERFACE_PARAMS["GDC|LSCF"]["mu_rt"]
    sd2 = INTERFACE_PARAMS["GDC|LSCF"]["sigma_rt"]

    cov = np.array(
        [
            [sd1**2, rho * sd1 * sd2],
            [rho * sd1 * sd2, sd2**2],
        ]
    )

    out = np.empty((n_samples, 2), dtype=float)
    filled = 0
    while filled < n_samples:
        draw = RNG.multivariate_normal(mean=[mu1, mu2], cov=cov, size=(n_samples - filled))
        draw = draw[(draw[:, 0] > 0.02) & (draw[:, 1] > 0.02)]
        take = min(len(draw), n_samples - filled)
        if take > 0:
            out[filled : filled + take, :] = draw[:take]
            filled += take

    return out


def build_correlation_df(pairs: np.ndarray, rho: float, tag: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pair_id": [f"{tag}_{i:05d}" for i in range(1, len(pairs) + 1)],
            "temperature_C": 25,
            "rho_assumed": rho,
            "Gc_YSZ_GDC_J_per_m2": np.round(pairs[:, 0], 6),
            "Gc_GDC_LSCF_J_per_m2": np.round(pairs[:, 1], 6),
        }
    )


def build_failure_map(ind_pairs: np.ndarray, cor_pairs: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_thresholds = np.linspace(0.8, 3.0, 55)
    y_thresholds = np.linspace(0.3, 1.6, 50)

    p_ind = np.zeros((len(y_thresholds), len(x_thresholds)))
    p_cor = np.zeros((len(y_thresholds), len(x_thresholds)))

    ind_x, ind_y = ind_pairs[:, 0], ind_pairs[:, 1]
    cor_x, cor_y = cor_pairs[:, 0], cor_pairs[:, 1]

    rows: list[dict] = []

    for i, x_thr in enumerate(x_thresholds):
        ind_mask_x = ind_x <= x_thr
        cor_mask_x = cor_x <= x_thr
        for j, y_thr in enumerate(y_thresholds):
            p_i = np.mean(ind_mask_x & (ind_y <= y_thr))
            p_c = np.mean(cor_mask_x & (cor_y <= y_thr))
            p_ind[j, i] = p_i
            p_cor[j, i] = p_c
            rows.append(
                {
                    "Gc_YSZ_threshold_J_per_m2": round(float(x_thr), 6),
                    "Gc_GDC_LSCF_threshold_J_per_m2": round(float(y_thr), 6),
                    "P_joint_failure_independent_rho_0": round(float(p_i), 6),
                    "P_joint_failure_correlated_rho_05": round(float(p_c), 6),
                    "delta_correlated_minus_independent": round(float(p_c - p_i), 6),
                }
            )

    return pd.DataFrame(rows), x_thresholds, y_thresholds, p_ind, p_cor


def build_missing_flags_df() -> pd.DataFrame:
    rows = [
        {
            "missing_dataset_id": "MISSING_DATASET_01",
            "title": "Interfacial Property Correlation Matrix",
            "required_variable": "rho",
            "current_assumption": "rho=0.0 (uncorrelated baseline)",
            "impact_if_wrong": "Can underpredict joint interface failure probability.",
            "required_data_to_fill_gap": "Paired post-mortem evidence across both interfaces in same stack.",
        },
        {
            "missing_dataset_id": "MISSING_DATASET_02",
            "title": "High-Temperature Fracture Statistics at 800C",
            "required_variable": "mu_800C, sigma_800C",
            "current_assumption": (
                "mu_800C scaled by E(800)/E(RT); sigma_800C set equal to RT sigma (conservative)."
            ),
            "impact_if_wrong": "Misestimated spread can distort failure probability map tails.",
            "required_data_to_fill_gap": "In-situ 800C micro-cantilever fracture experiments.",
        },
    ]
    return pd.DataFrame(rows)


def save_figures(
    micro_df: pd.DataFrame,
    ind_df: pd.DataFrame,
    cor_df: pd.DataFrame,
    x_thresholds: np.ndarray,
    y_thresholds: np.ndarray,
    p_ind: np.ndarray,
    p_cor: np.ndarray,
) -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_paths: list[Path] = []

    # Figure 1: Independent vs correlated scatter
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    axes[0].scatter(
        ind_df["Gc_YSZ_GDC_J_per_m2"],
        ind_df["Gc_GDC_LSCF_J_per_m2"],
        s=14,
        alpha=0.55,
        c="#1f77b4",
        edgecolors="none",
    )
    axes[0].set_title("Independent Sampling (rho=0.0)")
    axes[0].set_xlabel("Gc_YSZ|GDC [J/m^2]")
    axes[0].set_ylabel("Gc_GDC|LSCF [J/m^2]")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        cor_df["Gc_YSZ_GDC_J_per_m2"],
        cor_df["Gc_GDC_LSCF_J_per_m2"],
        s=14,
        alpha=0.55,
        c="#d62728",
        edgecolors="none",
    )
    axes[1].set_title("Correlated Sensitivity (rho=0.5)")
    axes[1].set_xlabel("Gc_YSZ|GDC [J/m^2]")
    axes[1].set_ylabel("Gc_GDC|LSCF [J/m^2]")
    axes[1].grid(alpha=0.25)

    fig1_path = FIG_DIR / "fig01_independent_vs_correlated_scatter.png"
    fig1.savefig(fig1_path, dpi=220)
    plt.close(fig1)
    figure_paths.append(fig1_path)

    # Figure 2: RT vs 800C distributions for each interface
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, interface in zip(axes, INTERFACE_PARAMS.keys()):
        rt = micro_df[(micro_df["interface"] == interface) & (micro_df["temperature_C"] == 25)][
            "Gc_J_per_m2"
        ]
        ht = micro_df[(micro_df["interface"] == interface) & (micro_df["temperature_C"] == 800)][
            "Gc_J_per_m2"
        ]
        ax.hist(rt, bins=28, alpha=0.65, density=True, label="RT (25C)", color="#2ca02c")
        ax.hist(ht, bins=28, alpha=0.55, density=True, label="Estimated 800C", color="#ff7f0e")
        ax.set_title(f"{interface} Fracture Energy")
        ax.set_xlabel("Gc [J/m^2]")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend()

    fig2_path = FIG_DIR / "fig02_rt_vs_800C_distributions.png"
    fig2.savefig(fig2_path, dpi=220)
    plt.close(fig2)
    figure_paths.append(fig2_path)

    # Figure 3: Probabilistic failure maps
    fig3, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    extent = [x_thresholds.min(), x_thresholds.max(), y_thresholds.min(), y_thresholds.max()]

    vmax = max(float(np.max(p_ind)), float(np.max(p_cor)))
    im0 = axes[0].imshow(
        p_ind,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    axes[0].set_title("Joint Failure Probability (rho=0.0)")
    axes[0].set_xlabel("Gc_YSZ threshold [J/m^2]")
    axes[0].set_ylabel("Gc_GDC|LSCF threshold [J/m^2]")

    im1 = axes[1].imshow(
        p_cor,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=vmax,
    )
    axes[1].set_title("Joint Failure Probability (rho=0.5)")
    axes[1].set_xlabel("Gc_YSZ threshold [J/m^2]")
    axes[1].set_ylabel("Gc_GDC|LSCF threshold [J/m^2]")

    delta = p_cor - p_ind
    lim = float(np.max(np.abs(delta)))
    im2 = axes[2].imshow(
        delta,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-lim,
        vmax=lim,
    )
    axes[2].set_title("Delta Probability (rho=0.5 - rho=0.0)")
    axes[2].set_xlabel("Gc_YSZ threshold [J/m^2]")
    axes[2].set_ylabel("Gc_GDC|LSCF threshold [J/m^2]")

    cbar0 = fig3.colorbar(im1, ax=axes[:2], shrink=0.9)
    cbar0.set_label("Joint failure probability")
    cbar1 = fig3.colorbar(im2, ax=axes[2], shrink=0.9)
    cbar1.set_label("Probability delta")

    fig3_path = FIG_DIR / "fig03_probabilistic_failure_maps.png"
    fig3.savefig(fig3_path, dpi=220)
    plt.close(fig3)
    figure_paths.append(fig3_path)

    return figure_paths


def zip_csv_files(csv_paths: list[Path], zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in csv_paths:
            zf.write(path, arcname=path.name)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    material_df = build_material_properties_df()
    micro_df = build_micro_cantilever_df()

    independent_pairs = sample_independent_pairs(N_CORRELATION_SAMPLES)
    correlated_pairs = sample_correlated_pairs(N_CORRELATION_SAMPLES, SENSITIVITY_RHO)

    ind_df = build_correlation_df(independent_pairs, BASELINE_RHO, "IND")
    cor_df = build_correlation_df(correlated_pairs, SENSITIVITY_RHO, "COR")

    map_ind_pairs = sample_independent_pairs(N_MAP_MONTE_CARLO)
    map_cor_pairs = sample_correlated_pairs(N_MAP_MONTE_CARLO, SENSITIVITY_RHO)
    failure_map_df, x_thresholds, y_thresholds, p_ind, p_cor = build_failure_map(
        map_ind_pairs, map_cor_pairs
    )

    missing_flags_df = build_missing_flags_df()

    csv_paths = [
        ROOT / "04_uncertainty_material_properties.csv",
        ROOT / "16_micro_cantilever_fracture_data.csv",
        ROOT / "17_synthetic_interfacial_correlation_samples_independent.csv",
        ROOT / "18_synthetic_interfacial_correlation_samples_rho_05.csv",
        ROOT / "20_failure_probability_map_grid.csv",
        ROOT / "21_missing_data_assumption_flags.csv",
    ]

    material_df.to_csv(csv_paths[0], index=False)
    micro_df.to_csv(csv_paths[1], index=False)
    ind_df.to_csv(csv_paths[2], index=False)
    cor_df.to_csv(csv_paths[3], index=False)
    failure_map_df.to_csv(csv_paths[4], index=False)
    missing_flags_df.to_csv(csv_paths[5], index=False)

    figure_paths = save_figures(
        micro_df=micro_df,
        ind_df=ind_df,
        cor_df=cor_df,
        x_thresholds=x_thresholds,
        y_thresholds=y_thresholds,
        p_ind=p_ind,
        p_cor=p_cor,
    )

    zip_path = ROOT / "probabilistic_failure_maps_csv_bundle.zip"
    zip_csv_files(csv_paths, zip_path)

    print("Generated CSV files:")
    for path in csv_paths:
        print(f" - {path.name}")
    print("\nGenerated figures:")
    for path in figure_paths:
        print(f" - {path.relative_to(ROOT)}")
    print(f"\nCreated ZIP bundle: {zip_path.name}")


if __name__ == "__main__":
    main()
