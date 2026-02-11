#!/usr/bin/env python3
"""
Generate synthetic uncertainty-quantification assets for:
Probabilistic Failure Maps of interfacial toughness in SOC stacks.

Outputs:
- CSV datasets in ../csv/
- Figure files in ../figures/
- A CSV zip bundle in ../downloads/
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260211
N_SAMPLES = 5000
J_REF = 3.0  # J/m^2

# Room-temperature interfacial toughness surrogates (fabricated but physically plausible)
MU_RT = np.array([1.55, 1.35])  # [YSZ|GDC, GDC|LSCF] in J/m^2
SIG_RT = np.array([0.28, 0.24])  # standard deviations in J/m^2

# High-temperature deterministic softening assumption (missing dataset 02 surrogate)
HT_MEAN_SCALE = 0.82


def make_dirs(base_dir: Path) -> dict[str, Path]:
    csv_dir = base_dir / "csv"
    fig_dir = base_dir / "figures"
    dl_dir = base_dir / "downloads"
    for d in (csv_dir, fig_dir, dl_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {"csv": csv_dir, "fig": fig_dir, "dl": dl_dir}


def sample_bivariate_normal(
    rng: np.random.Generator,
    n: int,
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: float,
    floor: float = 0.05,
) -> np.ndarray:
    cov = np.array(
        [
            [sigma[0] ** 2, rho * sigma[0] * sigma[1]],
            [rho * sigma[0] * sigma[1], sigma[1] ** 2],
        ]
    )
    samples = rng.multivariate_normal(mean=mu, cov=cov, size=n)
    return np.clip(samples, floor, None)


def build_stochastic_inputs(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    # Scenario A: independent world
    s_indep = sample_bivariate_normal(rng, N_SAMPLES, MU_RT, SIG_RT, rho=0.0)
    df_indep = pd.DataFrame(
        {
            "sample_id": np.arange(1, N_SAMPLES + 1),
            "scenario": "rho0.00_independent",
            "temperature_C": 25,
            "rho_assumed": 0.00,
            "Gc_YSZ_GDC": s_indep[:, 0],
            "Gc_GDC_LSCF": s_indep[:, 1],
            "Gc_YSZ_GDC_800C": np.nan,
            "Gc_GDC_LSCF_800C": np.nan,
        }
    )
    df_indep["R_system_total_J_m2"] = df_indep["Gc_YSZ_GDC"] + df_indep["Gc_GDC_LSCF"]

    # Scenario B: correlated world
    s_corr = sample_bivariate_normal(rng, N_SAMPLES, MU_RT, SIG_RT, rho=0.5)
    df_corr = pd.DataFrame(
        {
            "sample_id": np.arange(1, N_SAMPLES + 1),
            "scenario": "rho0.50_coupled",
            "temperature_C": 25,
            "rho_assumed": 0.50,
            "Gc_YSZ_GDC": s_corr[:, 0],
            "Gc_GDC_LSCF": s_corr[:, 1],
            "Gc_YSZ_GDC_800C": np.nan,
            "Gc_GDC_LSCF_800C": np.nan,
        }
    )
    df_corr["R_system_total_J_m2"] = df_corr["Gc_YSZ_GDC"] + df_corr["Gc_GDC_LSCF"]

    # Scenario C: high-temperature uncorrelated surrogate
    mu_ht = MU_RT * HT_MEAN_SCALE
    s_ht = sample_bivariate_normal(rng, N_SAMPLES, mu_ht, SIG_RT, rho=0.0)
    df_ht = pd.DataFrame(
        {
            "sample_id": np.arange(1, N_SAMPLES + 1),
            "scenario": "HT_uncorrelated_surrogate",
            "temperature_C": 800,
            "rho_assumed": 0.00,
            "Gc_YSZ_GDC": s_ht[:, 0],
            "Gc_GDC_LSCF": s_ht[:, 1],
            "Gc_YSZ_GDC_800C": s_ht[:, 0],
            "Gc_GDC_LSCF_800C": s_ht[:, 1],
        }
    )
    df_ht["R_system_total_J_m2"] = df_ht["Gc_YSZ_GDC"] + df_ht["Gc_GDC_LSCF"]

    return {"rho0.00": df_indep, "rho0.50": df_corr, "ht_uncorr": df_ht}


def build_uncertainty_material_properties() -> pd.DataFrame:
    rows = [
        {
            "dataset_id": "UNC_001",
            "property_group": "interfacial_toughness",
            "interface": "YSZ|GDC",
            "temperature_C": 25,
            "mean_value": MU_RT[0],
            "std_dev": SIG_RT[0],
            "units": "J/m^2",
            "distribution": "Normal",
            "source_type": "measured_independent_bilayers",
            "notes": "Room-temperature surrogate from independent micro-cantilever tests.",
        },
        {
            "dataset_id": "UNC_002",
            "property_group": "interfacial_toughness",
            "interface": "GDC|LSCF",
            "temperature_C": 25,
            "mean_value": MU_RT[1],
            "std_dev": SIG_RT[1],
            "units": "J/m^2",
            "distribution": "Normal",
            "source_type": "measured_independent_bilayers",
            "notes": "Room-temperature surrogate from independent micro-cantilever tests.",
        },
        {
            "dataset_id": "UNC_003",
            "property_group": "interfacial_toughness",
            "interface": "YSZ|GDC",
            "temperature_C": 800,
            "mean_value": MU_RT[0] * HT_MEAN_SCALE,
            "std_dev": SIG_RT[0],
            "units": "J/m^2",
            "distribution": "Normal",
            "source_type": "deterministic_scaled_surrogate",
            "notes": "High-temperature mean scaled from RT; variance held constant.",
        },
        {
            "dataset_id": "UNC_004",
            "property_group": "interfacial_toughness",
            "interface": "GDC|LSCF",
            "temperature_C": 800,
            "mean_value": MU_RT[1] * HT_MEAN_SCALE,
            "std_dev": SIG_RT[1],
            "units": "J/m^2",
            "distribution": "Normal",
            "source_type": "deterministic_scaled_surrogate",
            "notes": "High-temperature mean scaled from RT; variance held constant.",
        },
    ]
    return pd.DataFrame(rows)


def build_micro_cantilever_data(rng: np.random.Generator, n_each: int = 40) -> pd.DataFrame:
    ysz_gdc = np.clip(rng.normal(MU_RT[0], SIG_RT[0], size=n_each), 0.05, None)
    gdc_lscf = np.clip(rng.normal(MU_RT[1], SIG_RT[1], size=n_each), 0.05, None)
    rows = []
    for i, gc in enumerate(ysz_gdc, start=1):
        rows.append(
            {
                "specimen_id": f"MC-YG-{i:03d}",
                "interface": "YSZ|GDC",
                "test_temperature_C": 25,
                "Gc_J_m2": float(gc),
                "processing_batch": f"B{(i - 1) // 10 + 1}",
                "paired_full_cell_observation": "not_available",
                "comment": "Independent bilayer test.",
            }
        )
    for i, gc in enumerate(gdc_lscf, start=1):
        rows.append(
            {
                "specimen_id": f"MC-GL-{i:03d}",
                "interface": "GDC|LSCF",
                "test_temperature_C": 25,
                "Gc_J_m2": float(gc),
                "processing_batch": f"B{(i - 1) // 10 + 1}",
                "paired_full_cell_observation": "not_available",
                "comment": "Independent bilayer test.",
            }
        )
    return pd.DataFrame(rows)


def build_lscf_ferroelastic_curve() -> pd.DataFrame:
    # Synthetic hysteretic stress-strain response at 800C
    eps_up = np.linspace(0.0, 0.003, 120)
    sig_up = 26000.0 * eps_up - 2.4e6 * eps_up**2 + 2.8e8 * eps_up**3

    eps_down = np.linspace(0.003, 0.0, 120)
    # Introduce a mild offset to mimic ferroelastic switching hysteresis
    sig_down = (
        26000.0 * eps_down
        - 2.4e6 * eps_down**2
        + 2.8e8 * eps_down**3
        - 8.0 * np.sin(np.linspace(0.0, np.pi, eps_down.size))
    )

    df_up = pd.DataFrame(
        {
            "point_id": np.arange(1, eps_up.size + 1),
            "temperature_C": 800,
            "branch": "loading",
            "strain": eps_up,
            "stress_MPa": sig_up,
        }
    )
    df_down = pd.DataFrame(
        {
            "point_id": np.arange(eps_up.size + 1, eps_up.size + eps_down.size + 1),
            "temperature_C": 800,
            "branch": "unloading",
            "strain": eps_down,
            "stress_MPa": sig_down,
        }
    )
    return pd.concat([df_up, df_down], ignore_index=True)


def build_missing_flags() -> pd.DataFrame:
    rows = [
        {
            "flag_id": "MISSING_DATASET_01",
            "missing_physical_quantity": "Interfacial failure covariance between YSZ|GDC and GDC|LSCF within a single stack",
            "symbol": "cov(Gc_YSZ_GDC, Gc_GDC_LSCF), rho",
            "why_missing": "Micro-cantilever tests were done on independent bilayers, not paired full-cell interfaces.",
            "current_surrogate": "Two synthetic worlds are used: rho=0.00 baseline and rho=0.50 sensitivity.",
            "surrogate_source_file": "stochastic_inputs_rho0.00.csv; stochastic_inputs_rho0.50.csv",
            "impact_if_wrong": "Can underpredict joint interface failure probability and catastrophic delamination risk.",
            "severity": "high",
        },
        {
            "flag_id": "MISSING_DATASET_02",
            "missing_physical_quantity": "In-situ 800C PDF of interfacial fracture toughness",
            "symbol": "p(Gc | T=800C)",
            "why_missing": "No direct high-temperature micro-cantilever campaign available.",
            "current_surrogate": "Deterministic mean scaling from RT by factor 0.82 while holding std dev constant.",
            "surrogate_source_file": "stochastic_inputs_HT_uncorrelated.csv",
            "impact_if_wrong": "Misestimated spread can distort failure probability map tails and hide extreme-risk events.",
            "severity": "critical",
        },
    ]
    return pd.DataFrame(rows)


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(values)
    y = np.arange(1, x.size + 1) / x.size
    return x, y


def failure_probability_curve(resistance: np.ndarray, j_grid: np.ndarray) -> np.ndarray:
    # P_fail(J) = P(R < J)
    return (resistance[:, None] < j_grid[None, :]).mean(axis=0)


def build_analysis_tables(datasets: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    r_indep = datasets["rho0.00"]["R_system_total_J_m2"].to_numpy()
    r_corr = datasets["rho0.50"]["R_system_total_J_m2"].to_numpy()
    r_ht = datasets["ht_uncorr"]["R_system_total_J_m2"].to_numpy()

    j_grid = np.linspace(1.2, 4.8, 181)
    pf_indep = failure_probability_curve(r_indep, j_grid)
    pf_corr = failure_probability_curve(r_corr, j_grid)
    pf_ht = failure_probability_curve(r_ht, j_grid)

    curve_df = pd.DataFrame(
        {
            "J_applied_J_m2": j_grid,
            "P_fail_rho0_00": pf_indep,
            "P_fail_rho0_50": pf_corr,
            "P_fail_HT_uncorrelated": pf_ht,
            "epistemic_gap_abs_rho": np.abs(pf_indep - pf_corr),
        }
    )

    pf_indep_ref = float((r_indep < J_REF).mean())
    pf_corr_ref = float((r_corr < J_REF).mean())
    pf_ht_ref = float((r_ht < J_REF).mean())

    summary_rows = [
        {
            "metric": "J_reference_J_m2",
            "value": J_REF,
            "notes": "Driving-force level used for gap quantification.",
        },
        {"metric": "P_fail_rho0_00", "value": pf_indep_ref, "notes": "Independent assumption."},
        {"metric": "P_fail_rho0_50", "value": pf_corr_ref, "notes": "Correlated assumption."},
        {"metric": "P_fail_HT_uncorrelated", "value": pf_ht_ref, "notes": "High-temperature surrogate."},
        {
            "metric": "Epistemic_gap_percent_rho",
            "value": abs(pf_indep_ref - pf_corr_ref) * 100.0,
            "notes": "Absolute probability-point gap between rho scenarios.",
        },
        {
            "metric": "R_mean_rho0_00",
            "value": float(r_indep.mean()),
            "notes": "Mean system resistance for independent case.",
        },
        {
            "metric": "R_mean_rho0_50",
            "value": float(r_corr.mean()),
            "notes": "Mean system resistance for correlated case.",
        },
        {
            "metric": "R_mean_HT_uncorrelated",
            "value": float(r_ht.mean()),
            "notes": "Mean system resistance for high-temperature surrogate.",
        },
    ]
    summary_df = pd.DataFrame(summary_rows)

    return curve_df, summary_df


def build_correlation_manifest() -> pd.DataFrame:
    rows = [
        {"scenario": "rho0.00_independent", "corr_Gc_YSZ_GDC__Gc_GDC_LSCF": 0.00},
        {"scenario": "rho0.50_coupled", "corr_Gc_YSZ_GDC__Gc_GDC_LSCF": 0.50},
        {"scenario": "HT_uncorrelated_surrogate", "corr_Gc_YSZ_GDC__Gc_GDC_LSCF": 0.00},
    ]
    return pd.DataFrame(rows)


def plot_figures(datasets: dict[str, pd.DataFrame], curves: pd.DataFrame, fig_dir: Path) -> None:
    r_indep = datasets["rho0.00"]["R_system_total_J_m2"].to_numpy()
    r_corr = datasets["rho0.50"]["R_system_total_J_m2"].to_numpy()
    r_ht = datasets["ht_uncorr"]["R_system_total_J_m2"].to_numpy()

    # Figure 1: Fragility curves and epistemic cone
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    x = curves["J_applied_J_m2"].to_numpy()
    y1 = curves["P_fail_rho0_00"].to_numpy()
    y2 = curves["P_fail_rho0_50"].to_numpy()
    y3 = curves["P_fail_HT_uncorrelated"].to_numpy()

    ax.plot(x, y1, lw=2.2, label="Independent assumption (rho=0.00)")
    ax.plot(x, y2, lw=2.2, ls="--", label="Correlated assumption (rho=0.50)")
    ax.plot(x, y3, lw=2.2, color="crimson", label="High-T surrogate (mu scaled, sigma fixed)")
    ax.fill_between(
        x,
        np.minimum(y1, y2),
        np.maximum(y1, y2),
        color="orange",
        alpha=0.22,
        label="Epistemic uncertainty due to missing covariance data",
    )
    ax.axvline(J_REF, color="gray", ls=":", lw=1.4)
    ax.set_title("Probabilistic Failure Map: Cone of Ignorance")
    ax.set_xlabel("Applied Energy Release Rate J (J/m^2)")
    ax.set_ylabel("Probability of System Failure P_fail")
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_fragility_cone_of_ignorance.png")
    plt.close(fig)

    # Figure 2: CDF of system resistance
    fig, ax = plt.subplots(figsize=(9, 6), dpi=160)
    for arr, style, label in [
        (r_indep, "-", "rho=0.00"),
        (r_corr, "--", "rho=0.50"),
        (r_ht, "-", "High-T surrogate"),
    ]:
        ex, ey = empirical_cdf(arr)
        color = "crimson" if "High-T" in label else None
        ax.plot(ex, ey, ls=style, lw=2.1, color=color, label=label)
    ax.axvline(J_REF, color="gray", ls=":", lw=1.4, label="J_ref = 3.0 J/m^2")
    ax.set_title("CDF Comparison of Critical Energy Release Resistance")
    ax.set_xlabel("System Resistance R = Gc_YSZ_GDC + Gc_GDC_LSCF (J/m^2)")
    ax.set_ylabel("Cumulative Probability")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_cdf_critical_energy_release_rate.png")
    plt.close(fig)

    # Figure 3: Scatter to visualize correlation assumptions
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), dpi=160, sharex=True, sharey=True)
    d0 = datasets["rho0.00"].head(1200)
    d1 = datasets["rho0.50"].head(1200)

    axs[0].scatter(d0["Gc_YSZ_GDC"], d0["Gc_GDC_LSCF"], s=8, alpha=0.45)
    axs[0].set_title("Independent world (rho=0.00)")
    axs[0].set_xlabel("Gc_YSZ_GDC (J/m^2)")
    axs[0].set_ylabel("Gc_GDC_LSCF (J/m^2)")
    axs[0].grid(alpha=0.22)

    axs[1].scatter(d1["Gc_YSZ_GDC"], d1["Gc_GDC_LSCF"], s=8, alpha=0.45, color="tab:orange")
    axs[1].set_title("Coupled world (rho=0.50)")
    axs[1].set_xlabel("Gc_YSZ_GDC (J/m^2)")
    axs[1].grid(alpha=0.22)

    fig.suptitle("Synthetic Interfacial Toughness Realizations")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_interface_toughness_correlation_worlds.png")
    plt.close(fig)


def write_assumptions_markdown(base_dir: Path) -> None:
    text = """# Assumptions and Missing-Data Flags (Synthetic Package)

## Critical Data Void Analysis

### 1) MISSING_DATASET_01: Interfacial Failure Correlation Matrix

- Missing physical quantity:
  covariance (and correlation) between Gc_YSZ_GDC and Gc_GDC_LSCF inside one co-sintered cell.
- Mathematical deficit:
  off-diagonal covariance terms are unknown.
- Current surrogate:
  - stochastic_inputs_rho0.00.csv (rho = 0.00)
  - stochastic_inputs_rho0.50.csv (rho = 0.50)
- Why it matters:
  the probability of simultaneous two-interface failure depends on the unknown joint distribution.

### 2) MISSING_DATASET_02: High-Temperature Fracture Statistics (800C)

- Missing physical quantity:
  in-situ p(Gc | T=800C) for both interfaces.
- Current surrogate:
  deterministic softening of the RT mean by factor 0.82 with standard deviation unchanged.
- Surrogate file:
  stochastic_inputs_HT_uncorrelated.csv
- Why it matters:
  if real high-temperature variance is wider than assumed, tail-risk is underestimated.

## Reliability Formulation Note

Because the true interfacial correlation is unknown, P_fail should be treated as a bounded interval
between at least the rho=0 and rho=0.5 surrogate worlds, not as a single calibrated value.
"""
    (base_dir / "ASSUMPTIONS_AND_MISSING_DATA_FLAGS.md").write_text(text, encoding="utf-8")


def write_manifest(csv_dir: Path) -> None:
    manifest_rows = [
        {
            "file_name": "04_uncertainty_material_properties.csv",
            "description": "Synthetic material uncertainty parameters including RT and surrogate 800C toughness moments.",
        },
        {
            "file_name": "13_LSCF_ferroelastic_stress_strain.csv",
            "description": "Synthetic 800C stress-strain loop used to represent ferroelastic uncertainty context.",
        },
        {
            "file_name": "16_micro_cantilever_fracture_data.csv",
            "description": "Synthetic independent bilayer micro-cantilever measurements.",
        },
        {
            "file_name": "21_missing_data_assumption_flags.csv",
            "description": "Data-gap registry and impact flags.",
        },
        {
            "file_name": "stochastic_inputs_rho0.00.csv",
            "description": "Monte Carlo inputs under independence assumption.",
        },
        {
            "file_name": "stochastic_inputs_rho0.50.csv",
            "description": "Monte Carlo inputs under correlation sensitivity assumption.",
        },
        {
            "file_name": "stochastic_inputs_HT_uncorrelated.csv",
            "description": "Monte Carlo inputs under high-temperature deterministic mean-shift surrogate.",
        },
        {
            "file_name": "failure_probability_curves.csv",
            "description": "Fragility curves over applied J.",
        },
        {
            "file_name": "epistemic_gap_summary.csv",
            "description": "Summary metrics including P_fail at J=3.0 J/m^2 and epistemic gap.",
        },
        {
            "file_name": "interfacial_failure_correlation_matrix_surrogates.csv",
            "description": "Assumed correlation by scenario.",
        },
    ]
    pd.DataFrame(manifest_rows).to_csv(csv_dir / "dataset_manifest.csv", index=False)


def zip_csv_bundle(csv_dir: Path, dl_dir: Path) -> Path:
    zip_path = dl_dir / "probabilistic_failure_maps_csv_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for csv_file in sorted(csv_dir.glob("*.csv")):
            zf.write(csv_file, arcname=csv_file.name)
    return zip_path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    dirs = make_dirs(base_dir)
    csv_dir, fig_dir, dl_dir = dirs["csv"], dirs["fig"], dirs["dl"]

    rng = np.random.default_rng(SEED)

    # Build core stochastic datasets
    datasets = build_stochastic_inputs(rng)
    datasets["rho0.00"].to_csv(csv_dir / "stochastic_inputs_rho0.00.csv", index=False)
    datasets["rho0.50"].to_csv(csv_dir / "stochastic_inputs_rho0.50.csv", index=False)
    datasets["ht_uncorr"].to_csv(csv_dir / "stochastic_inputs_HT_uncorrelated.csv", index=False)

    # Build context/supporting datasets
    build_uncertainty_material_properties().to_csv(
        csv_dir / "04_uncertainty_material_properties.csv", index=False
    )
    build_micro_cantilever_data(rng).to_csv(
        csv_dir / "16_micro_cantilever_fracture_data.csv", index=False
    )
    build_lscf_ferroelastic_curve().to_csv(
        csv_dir / "13_LSCF_ferroelastic_stress_strain.csv", index=False
    )
    build_missing_flags().to_csv(csv_dir / "21_missing_data_assumption_flags.csv", index=False)
    build_correlation_manifest().to_csv(
        csv_dir / "interfacial_failure_correlation_matrix_surrogates.csv", index=False
    )

    # Analysis products
    curves_df, summary_df = build_analysis_tables(datasets)
    curves_df.to_csv(csv_dir / "failure_probability_curves.csv", index=False)
    summary_df.to_csv(csv_dir / "epistemic_gap_summary.csv", index=False)

    write_manifest(csv_dir)
    write_assumptions_markdown(base_dir)
    plot_figures(datasets, curves_df, fig_dir)
    zip_path = zip_csv_bundle(csv_dir, dl_dir)

    # Console summary
    pf_indep = float(summary_df.loc[summary_df["metric"] == "P_fail_rho0_00", "value"].iloc[0])
    pf_corr = float(summary_df.loc[summary_df["metric"] == "P_fail_rho0_50", "value"].iloc[0])
    pf_ht = float(summary_df.loc[summary_df["metric"] == "P_fail_HT_uncorrelated", "value"].iloc[0])
    gap = abs(pf_indep - pf_corr) * 100.0
    print("--- IMPACT OF MISSING DATA (SYNTHETIC) ---")
    print(f"Scenario 1 (rho=0.00): P_fail = {pf_indep:.4f}")
    print(f"Scenario 2 (rho=0.50): P_fail = {pf_corr:.4f}")
    print(f"Scenario 3 (High-T):  P_fail = {pf_ht:.4f}")
    print("-" * 40)
    print(f"Epistemic Uncertainty Gap (rho): {gap:.2f}%")
    print(f"CSV bundle written to: {zip_path}")


if __name__ == "__main__":
    main()
