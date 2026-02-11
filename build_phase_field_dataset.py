#!/usr/bin/env python3
"""Generate a calibrated/fabricated SOFC phase-field delamination dataset.

This script creates:
  - Structured CSV files aligned with user-provided calibration ranges.
  - Figure files for quick visual QA.
  - A ZIP archive that bundles all CSV files for easy download.
"""

from __future__ import annotations

import csv
import math
import zipfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "phase_field_delamination_dataset"
CSV_DIR = OUTPUT_DIR / "csv"
FIG_DIR = OUTPUT_DIR / "figures"
ZIP_PATH = OUTPUT_DIR / "phase_field_delamination_csv_bundle.zip"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_lscf_delta(temp_c: float, p_o2_atm: float) -> float:
    """Calibrated interpolation matching 0.009 to 0.047 target range."""
    temp_factor = (temp_c - 600.0) / 300.0
    oxygen_factor = max(0.0, -math.log10(p_o2_atm)) / 20.0
    delta = 0.009 + 0.020 * temp_factor + 0.018 * oxygen_factor
    return round(min(max(delta, 0.009), 0.047), 6)


def compute_gdc_delta(temp_c: float, p_o2_atm: float) -> float:
    """Calibrated interpolation peaking at 0.0178 (900 C, 1e-20 atm)."""
    temp_factor = (temp_c - 600.0) / 300.0
    oxygen_log = -math.log10(p_o2_atm)
    oxygen_factor = (min(max(oxygen_log, 5.0), 20.0) - 5.0) / 15.0
    delta = 0.001 + 0.006 * temp_factor + 0.0108 * oxygen_factor
    return round(min(max(delta, 0.001), 0.0178), 6)


def create_assumptions_inventory() -> list[dict]:
    return [
        {
            "parameter": "BK Exponent",
            "symbol": "eta",
            "value_min": 2.1,
            "value_max": 2.1,
            "nominal_value": 2.1,
            "units": "dimensionless",
            "source_rationale": "Standard for ceramic brittle fracture; recommended for GDC nanostructures.",
            "status": "calibrated",
        },
        {
            "parameter": "Cathode Anisotropy",
            "symbol": "beta33_over_beta11",
            "value_min": 1.5,
            "value_max": 2.0,
            "nominal_value": 1.7,
            "units": "ratio",
            "source_rationale": "Default expectation for LSCF diagonal anisotropic expansion tensor.",
            "status": "calibrated",
        },
        {
            "parameter": "Fracture Energy (Bulk)",
            "symbol": "Gc_bulk",
            "value_min": 1.5,
            "value_max": 10.0,
            "nominal_value": 5.0,
            "units": "J/m^2",
            "source_rationale": "Derived from bulk micro-cantilever and indentation tests for LSCF/YSZ.",
            "status": "calibrated_range",
        },
        {
            "parameter": "Interface Adhesion",
            "symbol": "Gamma_i",
            "value_min": 0.2,
            "value_max": 3.2,
            "nominal_value": 1.7,
            "units": "J/m^2",
            "source_rationale": "DFT and diffusion models; sensitive to Sr-segregation/interdiffusion chemistry.",
            "status": "calibrated_range",
        },
        {
            "parameter": "Penalty Parameter",
            "symbol": "beta_pen",
            "value_min": 1.0e2,
            "value_max": 1.0e4,
            "nominal_value": 1.0e3,
            "units": "GPa/m",
            "source_rationale": "Enforces interface constraints in phase-field/cohesive hybrid formulation.",
            "status": "assumed_to_calibrated",
        },
        {
            "parameter": "Phase-field Length",
            "symbol": "l0",
            "value_min": 5.0,
            "value_max": 20.0,
            "nominal_value": 10.0,
            "units": "nm",
            "source_rationale": "Resolves GDC nanostructure and interface width scale.",
            "status": "calibrated_range",
        },
        {
            "parameter": "Regulated Length",
            "symbol": "l",
            "value_min": 0.5,
            "value_max": 0.5,
            "nominal_value": 0.5,
            "units": "um",
            "source_rationale": "Controls diffuse crack width in bulk phase-field evolution.",
            "status": "fixed",
        },
    ]


def create_interface_fracture_rows() -> list[dict]:
    return [
        {
            "interface": "YSZ/GDC",
            "condition": "baseline",
            "Gc_int_min_Jm2": 1.8,
            "Gc_int_max_Jm2": 2.5,
            "sigma_max_min_MPa": 185.0,
            "sigma_max_max_MPa": 260.0,
            "interlayer_thickness_min_nm": 100.0,
            "interlayer_thickness_max_nm": 1000.0,
            "characteristic_length_min_um": "",
            "characteristic_length_max_um": "",
            "mechanism_note": "Interlayer initiation",
        },
        {
            "interface": "YSZ/GDC",
            "condition": "with_interdiffusion",
            "Gc_int_min_Jm2": 2.5,
            "Gc_int_max_Jm2": 3.2,
            "sigma_max_min_MPa": 185.0,
            "sigma_max_max_MPa": 260.0,
            "interlayer_thickness_min_nm": 100.0,
            "interlayer_thickness_max_nm": 1000.0,
            "characteristic_length_min_um": "",
            "characteristic_length_max_um": "",
            "mechanism_note": "(Zr,Ce)O2 solid solution formation increases adhesion",
        },
        {
            "interface": "GDC/LSCF",
            "condition": "baseline",
            "Gc_int_min_Jm2": 0.5,
            "Gc_int_max_Jm2": 1.5,
            "sigma_max_min_MPa": "",
            "sigma_max_max_MPa": "",
            "interlayer_thickness_min_nm": "",
            "interlayer_thickness_max_nm": "",
            "characteristic_length_min_um": 0.18,
            "characteristic_length_max_um": 0.35,
            "mechanism_note": "Cathode delamination window",
        },
        {
            "interface": "GDC/LSCF",
            "condition": "with_sr_segregation",
            "Gc_int_min_Jm2": 0.2,
            "Gc_int_max_Jm2": 0.8,
            "sigma_max_min_MPa": "",
            "sigma_max_max_MPa": "",
            "interlayer_thickness_min_nm": "",
            "interlayer_thickness_max_nm": "",
            "characteristic_length_min_um": 0.18,
            "characteristic_length_max_um": 0.35,
            "mechanism_note": "SrO/SrZrO3 reaction lowers interface toughness",
        },
    ]


def create_lscf_nonstoich_rows() -> list[dict]:
    temps = [600, 700, 800, 900]
    p_o2_values = [1.0, 1e-5, 1e-10, 1e-15, 1e-20]
    rows: list[dict] = []
    for temp in temps:
        for p_o2 in p_o2_values:
            rows.append(
                {
                    "material": "LSCF",
                    "temperature_C": temp,
                    "pO2_atm": f"{p_o2:.1e}",
                    "log10_pO2": round(math.log10(p_o2), 4),
                    "delta_nonstoichiometry": compute_lscf_delta(temp, p_o2),
                    "provenance": "fabricated_from_calibrated_range",
                }
            )
    return rows


def create_gdc_nonstoich_rows() -> list[dict]:
    temps = [600, 700, 800, 900]
    p_o2_values = [1e-5, 1e-10, 1e-15, 1e-20]
    rows: list[dict] = []
    for temp in temps:
        for p_o2 in p_o2_values:
            rows.append(
                {
                    "material": "GDC",
                    "temperature_C": temp,
                    "pO2_atm": f"{p_o2:.1e}",
                    "log10_pO2": round(math.log10(p_o2), 4),
                    "delta_nonstoichiometry": compute_gdc_delta(temp, p_o2),
                    "provenance": "fabricated_from_calibrated_range",
                }
            )
    return rows


def create_gdc_chemical_expansion_rows() -> list[dict]:
    temps = [600, 700, 800, 900]
    delta_points = np.linspace(0.0000, 0.0220, 22)
    rows: list[dict] = []
    for temp in temps:
        alpha_chem = 0.020 + 0.008 * ((temp - 600.0) / 300.0)
        for delta in delta_points:
            curvature = 1.0 + 0.15 * (delta / 0.0220)
            epsilon_ch = alpha_chem * float(delta) * curvature
            rows.append(
                {
                    "material": "GDC",
                    "temperature_C": temp,
                    "delta_nonstoichiometry": round(float(delta), 6),
                    "alpha_chem_per_delta": round(alpha_chem, 6),
                    "epsilon_ch": round(epsilon_ch, 8),
                    "epsilon_ch_microstrain": round(epsilon_ch * 1e6, 2),
                    "equation": "epsilon_ch = alpha_chem * delta * (1 + 0.15 * delta/0.022)",
                }
            )
    return rows


def create_qa_rows() -> list[dict]:
    return [
        {
            "check_item": "Mesh Objectivity",
            "symbol_or_setting": "h",
            "min_recommended": 1.25,
            "max_recommended": 10.0,
            "units": "nm",
            "implementation_note": "Keep h in [l0/4, l0/2] based on l0=5-20 nm.",
        },
        {
            "check_item": "Energy Balance",
            "symbol_or_setting": "g(phi)",
            "min_recommended": 1.0e-6,
            "max_recommended": 1.0e-6,
            "units": "residual stiffness",
            "implementation_note": "Use g(phi) = (1-phi)^2 + 1e-6 to avoid singular stiffness.",
        },
        {
            "check_item": "Convergence",
            "symbol_or_setting": "epsilon_tol",
            "min_recommended": 1.0e-8,
            "max_recommended": 1.0e-6,
            "units": "relative NR tolerance",
            "implementation_note": "Use Newton-Raphson tolerance between 1e-6 and 1e-8.",
        },
    ]


def create_simulation_samples(interface_rows: list[dict], n_samples: int = 240) -> list[dict]:
    rng = np.random.default_rng(20260211)
    rows: list[dict] = []
    interface_indices = np.arange(len(interface_rows))
    tolerances = [1e-6, 3e-7, 1e-7, 3e-8, 1e-8]

    for i in range(1, n_samples + 1):
        chosen = interface_rows[int(rng.choice(interface_indices))]
        temp_c = float(rng.uniform(600.0, 900.0))
        log_p_o2 = float(rng.uniform(-20.0, 0.0))
        p_o2 = 10.0 ** log_p_o2

        gc_int = float(rng.uniform(chosen["Gc_int_min_Jm2"], chosen["Gc_int_max_Jm2"]))
        gamma_i = float(np.clip(gc_int * rng.uniform(0.7, 1.3), 0.2, 3.2))

        beta_pen = 10.0 ** float(rng.uniform(2.0, 4.0))
        l0_nm = float(rng.uniform(5.0, 20.0))
        mesh_h_nm = float(rng.uniform(l0_nm / 4.0, l0_nm / 2.0))

        sigma_val = ""
        if chosen["sigma_max_min_MPa"] != "":
            sigma_val = round(
                float(
                    rng.uniform(
                        float(chosen["sigma_max_min_MPa"]),
                        float(chosen["sigma_max_max_MPa"]),
                    )
                ),
                3,
            )

        char_len = ""
        if chosen["characteristic_length_min_um"] != "":
            char_len = round(
                float(
                    rng.uniform(
                        float(chosen["characteristic_length_min_um"]),
                        float(chosen["characteristic_length_max_um"]),
                    )
                ),
                4,
            )

        rows.append(
            {
                "sample_id": i,
                "interface": chosen["interface"],
                "interface_condition": chosen["condition"],
                "temperature_C": round(temp_c, 2),
                "pO2_atm": f"{p_o2:.3e}",
                "log10_pO2": round(log_p_o2, 4),
                "eta": 2.1,
                "beta33_over_beta11": round(float(rng.triangular(1.5, 1.7, 2.0)), 4),
                "Gc_bulk_Jm2": round(float(rng.uniform(1.5, 10.0)), 4),
                "Gamma_i_Jm2": round(gamma_i, 4),
                "Gc_int_Jm2": round(gc_int, 4),
                "sigma_max_MPa": sigma_val,
                "characteristic_length_um": char_len,
                "beta_pen_GPam": round(beta_pen, 4),
                "l0_nm": round(l0_nm, 4),
                "regulated_length_um": 0.5,
                "LSCF_delta": compute_lscf_delta(temp_c, p_o2),
                "GDC_delta": compute_gdc_delta(temp_c, p_o2),
                "newton_tol": rng.choice(tolerances),
                "mesh_h_nm": round(mesh_h_nm, 4),
                "degradation_residual": 1e-6,
            }
        )

    return rows


def plot_interface_fracture_energy(interface_rows: list[dict]) -> None:
    labels = [f"{row['interface']} | {row['condition']}" for row in interface_rows]
    gmins = [float(row["Gc_int_min_Jm2"]) for row in interface_rows]
    gmaxs = [float(row["Gc_int_max_Jm2"]) for row in interface_rows]
    mids = [(lo + hi) / 2.0 for lo, hi in zip(gmins, gmaxs)]
    widths = [hi - lo for lo, hi in zip(gmins, gmaxs)]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(y=y, width=widths, left=gmins, color="#8fbcd4", edgecolor="#2b4b5f")
    ax.scatter(mids, y, color="#1d3557", zorder=3, label="mid-range value")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Interface fracture energy, Gc_int (J/m^2)")
    ax.set_title("Calibrated interface fracture-energy windows")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_01_interface_fracture_energy_ranges.png", dpi=220)
    plt.close(fig)


def _grid_from_rows(rows: list[dict]) -> tuple[np.ndarray, list[int], list[float]]:
    temps = sorted({int(r["temperature_C"]) for r in rows})
    logs = sorted({float(r["log10_pO2"]) for r in rows})
    data = np.zeros((len(logs), len(temps)))
    for i, logp in enumerate(logs):
        for j, temp in enumerate(temps):
            value = next(
                (
                    float(r["delta_nonstoichiometry"])
                    for r in rows
                    if int(r["temperature_C"]) == temp and float(r["log10_pO2"]) == logp
                ),
                np.nan,
            )
            data[i, j] = value
    return data, temps, logs


def plot_nonstoich_heatmaps(lscf_rows: list[dict], gdc_rows: list[dict]) -> None:
    lscf_grid, lscf_temps, lscf_logs = _grid_from_rows(lscf_rows)
    gdc_grid, gdc_temps, gdc_logs = _grid_from_rows(gdc_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(lscf_grid, cmap="viridis", aspect="auto", origin="lower")
    axes[0].set_title("LSCF calibrated Δδ map")
    axes[0].set_xticks(range(len(lscf_temps)))
    axes[0].set_xticklabels(lscf_temps)
    axes[0].set_yticks(range(len(lscf_logs)))
    axes[0].set_yticklabels([f"{v:.0f}" for v in lscf_logs])
    axes[0].set_xlabel("Temperature (C)")
    axes[0].set_ylabel("log10(pO2/atm)")
    fig.colorbar(im0, ax=axes[0], label="Δδ")

    im1 = axes[1].imshow(gdc_grid, cmap="magma", aspect="auto", origin="lower")
    axes[1].set_title("GDC calibrated δ map")
    axes[1].set_xticks(range(len(gdc_temps)))
    axes[1].set_xticklabels(gdc_temps)
    axes[1].set_yticks(range(len(gdc_logs)))
    axes[1].set_yticklabels([f"{v:.0f}" for v in gdc_logs])
    axes[1].set_xlabel("Temperature (C)")
    axes[1].set_ylabel("log10(pO2/atm)")
    fig.colorbar(im1, ax=axes[1], label="δ")

    fig.savefig(FIG_DIR / "figure_02_nonstoichiometry_heatmaps.png", dpi=220)
    plt.close(fig)


def plot_gdc_chemical_expansion(expansion_rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    temps = sorted({int(r["temperature_C"]) for r in expansion_rows})
    for temp in temps:
        xs = [
            float(r["delta_nonstoichiometry"])
            for r in expansion_rows
            if int(r["temperature_C"]) == temp
        ]
        ys = [
            float(r["epsilon_ch_microstrain"])
            for r in expansion_rows
            if int(r["temperature_C"]) == temp
        ]
        ax.plot(xs, ys, marker="o", markersize=3.2, linewidth=1.4, label=f"{temp} C")

    ax.set_xlabel("GDC non-stoichiometry, δ")
    ax.set_ylabel("Chemical strain, epsilon_ch (microstrain)")
    ax.set_title("GDC chemical expansion dataset (22 delta x 4 temperature)")
    ax.grid(linestyle="--", alpha=0.35)
    ax.legend(title="Temperature")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_03_gdc_chemical_expansion_curves.png", dpi=220)
    plt.close(fig)


def plot_sampling_overview(samples: list[dict]) -> None:
    temps = np.array([float(r["temperature_C"]) for r in samples])
    p_logs = np.array([float(r["log10_pO2"]) for r in samples])
    gc_int = np.array([float(r["Gc_int_Jm2"]) for r in samples])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    sc = ax.scatter(temps, p_logs, c=gc_int, cmap="plasma", s=28, alpha=0.85)
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("log10(pO2/atm)")
    ax.set_title("Fabricated simulation sample coverage (n=240)")
    ax.grid(linestyle="--", alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Sampled Gc_int (J/m^2)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure_04_simulation_sampling_coverage.png", dpi=220)
    plt.close(fig)


def write_readme(csv_files: list[Path], fig_files: list[Path]) -> None:
    readme_path = OUTPUT_DIR / "README.txt"
    lines = [
        "Phase-Field Fracture Delamination Dataset Package",
        "=================================================",
        "",
        "Topic:",
        "  Phase-Field Fracture Modeling of Delamination in Electrolyte-Electrode",
        "  Interfaces with nanoscale MIEC interlayers.",
        "",
        "Important:",
        "  Data are fabricated/synthesized from the calibrated ranges provided",
        "  by the request and intended for simulation workflow prototyping.",
        "",
        "CSV files:",
    ]
    lines.extend([f"  - {path.name}" for path in sorted(csv_files)])
    lines.extend(["", "Figure files:"])
    lines.extend([f"  - {path.name}" for path in sorted(fig_files)])
    lines.extend(
        [
            "",
            f"ZIP bundle:",
            f"  - {ZIP_PATH.name}",
            "",
            "Regeneration:",
            "  python3 build_phase_field_dataset.py",
            "",
        ]
    )
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def build_zip(csv_files: list[Path]) -> None:
    with zipfile.ZipFile(ZIP_PATH, mode="w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for csv_path in sorted(csv_files):
            bundle.write(csv_path, arcname=csv_path.name)


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    assumptions = create_assumptions_inventory()
    interface_rows = create_interface_fracture_rows()
    lscf_rows = create_lscf_nonstoich_rows()
    gdc_rows = create_gdc_nonstoich_rows()
    expansion_rows = create_gdc_chemical_expansion_rows()
    qa_rows = create_qa_rows()
    sample_rows = create_simulation_samples(interface_rows, n_samples=240)

    write_csv(
        CSV_DIR / "assumptions_calibrated_data_inventory.csv",
        [
            "parameter",
            "symbol",
            "value_min",
            "value_max",
            "nominal_value",
            "units",
            "source_rationale",
            "status",
        ],
        assumptions,
    )
    write_csv(
        CSV_DIR / "calibrated_interface_fracture_properties.csv",
        [
            "interface",
            "condition",
            "Gc_int_min_Jm2",
            "Gc_int_max_Jm2",
            "sigma_max_min_MPa",
            "sigma_max_max_MPa",
            "interlayer_thickness_min_nm",
            "interlayer_thickness_max_nm",
            "characteristic_length_min_um",
            "characteristic_length_max_um",
            "mechanism_note",
        ],
        interface_rows,
    )
    write_csv(
        CSV_DIR / "lscf_nonstoichiometry_calibrated_grid.csv",
        [
            "material",
            "temperature_C",
            "pO2_atm",
            "log10_pO2",
            "delta_nonstoichiometry",
            "provenance",
        ],
        lscf_rows,
    )
    write_csv(
        CSV_DIR / "gdc_nonstoichiometry_calibrated_grid.csv",
        [
            "material",
            "temperature_C",
            "pO2_atm",
            "log10_pO2",
            "delta_nonstoichiometry",
            "provenance",
        ],
        gdc_rows,
    )
    write_csv(
        CSV_DIR / "gdc_chemical_expansion_22x4_dataset.csv",
        [
            "material",
            "temperature_C",
            "delta_nonstoichiometry",
            "alpha_chem_per_delta",
            "epsilon_ch",
            "epsilon_ch_microstrain",
            "equation",
        ],
        expansion_rows,
    )
    write_csv(
        CSV_DIR / "phase_field_qa_verification_plan.csv",
        [
            "check_item",
            "symbol_or_setting",
            "min_recommended",
            "max_recommended",
            "units",
            "implementation_note",
        ],
        qa_rows,
    )
    write_csv(
        CSV_DIR / "phase_field_simulation_samples.csv",
        [
            "sample_id",
            "interface",
            "interface_condition",
            "temperature_C",
            "pO2_atm",
            "log10_pO2",
            "eta",
            "beta33_over_beta11",
            "Gc_bulk_Jm2",
            "Gamma_i_Jm2",
            "Gc_int_Jm2",
            "sigma_max_MPa",
            "characteristic_length_um",
            "beta_pen_GPam",
            "l0_nm",
            "regulated_length_um",
            "LSCF_delta",
            "GDC_delta",
            "newton_tol",
            "mesh_h_nm",
            "degradation_residual",
        ],
        sample_rows,
    )

    plot_interface_fracture_energy(interface_rows)
    plot_nonstoich_heatmaps(lscf_rows, gdc_rows)
    plot_gdc_chemical_expansion(expansion_rows)
    plot_sampling_overview(sample_rows)

    csv_files = sorted(CSV_DIR.glob("*.csv"))
    fig_files = sorted(FIG_DIR.glob("*.png"))
    build_zip(csv_files)
    write_readme(csv_files, fig_files)

    print("Dataset package generated:")
    print(f"  CSV directory: {CSV_DIR}")
    print(f"  Figure directory: {FIG_DIR}")
    print(f"  ZIP archive: {ZIP_PATH}")
    print(f"  CSV count: {len(csv_files)}")
    print(f"  Figure count: {len(fig_files)}")


if __name__ == "__main__":
    main()
