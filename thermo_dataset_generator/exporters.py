from __future__ import annotations

import csv
from typing import Dict, List, Tuple


def export_comsol_csv(
    path: str,
    mix_id: str,
    data_type: str,
    temperature_c: List[float],
    summary: Dict[str, Tuple[List[float], List[float]]],
) -> None:
    # CSV with columns: Mix_ID,Data_Type,T_C,property_mean,property_std...
    fieldnames = [
        "Mix_ID",
        "Data_Type",
        "T_C",
        "rho_mean_kgm3",
        "rho_std_kgm3",
        "k_mean_WmK",
        "k_std_WmK",
        "cp_mean_JkgK",
        "cp_std_JkgK",
        "E_mean_Pa",
        "E_std_Pa",
        "nu_mean",
        "nu_std",
        "alpha_th_mean_1K",
        "alpha_th_std_1K",
        "k_perm_mean_m2",
        "k_perm_std_m2",
        "phi_mean",
        "phi_std",
        "alpha_biot_mean",
        "alpha_biot_std",
        "fc_mean_Pa",
        "fc_std_Pa",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, T in enumerate(temperature_c):
            row = {
                "Mix_ID": mix_id,
                "Data_Type": data_type,
                "T_C": T,
                "rho_mean_kgm3": summary["rho_kgm3"][0][i],
                "rho_std_kgm3": summary["rho_kgm3"][1][i],
                "k_mean_WmK": summary["k_WmK"][0][i],
                "k_std_WmK": summary["k_WmK"][1][i],
                "cp_mean_JkgK": summary["cp_JkgK"][0][i],
                "cp_std_JkgK": summary["cp_JkgK"][1][i],
                "E_mean_Pa": summary["E_Pa"][0][i],
                "E_std_Pa": summary["E_Pa"][1][i],
                "nu_mean": summary["nu"][0][i],
                "nu_std": summary["nu"][1][i],
                "alpha_th_mean_1K": summary["alpha_th_1K"][0][i],
                "alpha_th_std_1K": summary["alpha_th_1K"][1][i],
                "k_perm_mean_m2": summary["k_perm_m2"][0][i],
                "k_perm_std_m2": summary["k_perm_m2"][1][i],
                "phi_mean": summary["phi"][0][i],
                "phi_std": summary["phi"][1][i],
                "alpha_biot_mean": summary["alpha_biot"][0][i],
                "alpha_biot_std": summary["alpha_biot"][1][i],
                "fc_mean_Pa": summary["fc_Pa"][0][i],
                "fc_std_Pa": summary["fc_Pa"][1][i],
            }
            writer.writerow(row)


def export_abaqus_inp(
    path: str,
    mix_id: str,
    data_type: str,
    temperature_c: List[float],
    summary: Dict[str, Tuple[List[float], List[float]]],
) -> None:
    # ABAQUS material definition with temperature-dependent properties
    # Using mean curves. Units: SI.
    lines: List[str] = []
    lines.append(f"** MATERIAL DATASET: Mix_ID={mix_id}, Data_Type={data_type}")
    lines.append(f"*MATERIAL, NAME={mix_id}_{data_type}")

    # Density
    lines.append("*DENSITY, DEPENDENCIES=1")
    for i, T in enumerate(temperature_c):
        rho = summary["rho_kgm3"][0][i]
        lines.append(f"{rho:.3f}, {T:.3f}")

    # Specific heat
    lines.append("*SPECIFIC HEAT, DEPENDENCIES=1")
    for i, T in enumerate(temperature_c):
        cp = summary["cp_JkgK"][0][i]
        lines.append(f"{cp:.3f}, {T:.3f}")

    # Conductivity
    lines.append("*CONDUCTIVITY, TYPE=ISOTROPIC, DEPENDENCIES=1")
    for i, T in enumerate(temperature_c):
        k = summary["k_WmK"][0][i]
        lines.append(f"{k:.6f}, {T:.3f}")

    # Thermal expansion
    lines.append("*EXPANSION, ZERO=20.")
    for i, T in enumerate(temperature_c):
        alpha = summary["alpha_th_1K"][0][i]
        lines.append(f"{alpha:.9e}, {T:.3f}")

    # Elasticity: E, nu, T
    lines.append("*ELASTIC, TYPE=ISOTROPIC, DEPENDENCIES=1")
    for i, T in enumerate(temperature_c):
        E = summary["E_Pa"][0][i]
        nu = summary["nu"][0][i]
        lines.append(f"{E:.3f}, {nu:.5f}, {T:.3f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def export_ansys_apdl(
    path: str,
    mix_id: str,
    data_type: str,
    temperature_c: List[float],
    summary: Dict[str, Tuple[List[float], List[float]]],
) -> None:
    # MAPDL script defining temperature-dependent material properties using MPTEMP/MPDATA
    # Property table index 1
    lines: List[str] = []
    lines.append(f"! MATERIAL DATASET: Mix_ID={mix_id}, Data_Type={data_type}")
    # Temperatures (up to 20 per MPTEMP call; we will chunk as needed)
    temps = temperature_c

    def write_mptemp(vals: List[float]) -> None:
        chunks = [vals[i : i + 20] for i in range(0, len(vals), 20)]
        start_idx = 1
        for chunk in chunks:
            lines.append("MPTEMP, %d," % (start_idx) + ", ".join(f"{t:.3f}" for t in chunk))
            start_idx += len(chunk)

    def write_mpdata(label: str, values: List[float]) -> None:
        chunks = [values[i : i + 20] for i in range(0, len(values), 20)]
        start_idx = 1
        for chunk in chunks:
            lines.append("MPDATA, %s, 1, %d, " % (label, start_idx) + ", ".join(f"{v:.6g}" for v in chunk))
            start_idx += len(chunk)

    write_mptemp(temps)

    rho_vals = [summary["rho_kgm3"][0][i] for i in range(len(temps))]
    cp_vals = [summary["cp_JkgK"][0][i] for i in range(len(temps))]
    k_vals = [summary["k_WmK"][0][i] for i in range(len(temps))]
    E_vals = [summary["E_Pa"][0][i] for i in range(len(temps))]
    nu_vals = [summary["nu"][0][i] for i in range(len(temps))]
    alpha_vals = [summary["alpha_th_1K"][0][i] for i in range(len(temps))]

    write_mpdata("DENS", rho_vals)
    write_mpdata("C", cp_vals)
    write_mpdata("KXX", k_vals)
    write_mpdata("EX", E_vals)
    write_mpdata("NUXY", nu_vals)
    write_mpdata("ALPX", alpha_vals)

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
