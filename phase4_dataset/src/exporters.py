from __future__ import annotations
from typing import Dict, List
import json
import csv
import os

from constants import UNITS


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_json(path: str, obj: Dict) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def export_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            # Filter out any unexpected keys to avoid csv errors
            filtered = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(filtered)


def export_comsol_material_table(path: str, rows: List[Dict[str, object]]) -> None:
    # COMSOL can ingest CSV tables with temperature-dependent properties
    # Ensure the first line is the header row (no comment line) for broad compatibility
    ensure_dir(os.path.dirname(path))
    headers = [
        "Temperature_C",
        "density_kgm3",
        "specific_heat_jkgk",
        "thermal_conductivity_wmk",
        "elastic_modulus_pa",
        "poisson_ratio",
        "cte_1k",
        "permeability_m2",
        "biot_coefficient",
        "moisture_diffusivity_m2s",
        "free_thermal_strain",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({h: r.get(h, "") for h in headers})


def export_ansys_engdat(path: str, rows: List[Dict[str, object]]) -> None:
    # Simplified APDL-like tabular data; users can import via TB, TBDATA blocks.
    # We output CSV sections per property for clarity.
    ensure_dir(os.path.dirname(path))
    props = [
        ("Temperature_C", "T [°C]"),
        ("density_kgm3", "DENS [kg/m^3]"),
        ("specific_heat_jkgk", "CP [J/kg/K]"),
        ("thermal_conductivity_wmk", "KXX [W/m/K]"),
        ("elastic_modulus_pa", "EX [Pa]"),
        ("poisson_ratio", "NU [-]"),
        ("cte_1k", "ALPX [1/K]"),
    ]
    for key, label in props:
        csv_path = path.replace(".csv", f"_{key}.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["#", label])
            writer.writerow(["Temperature_C", key])
            for r in rows:
                writer.writerow([r["Temperature_C"], r.get(key, "")])


def export_abaqus_material(path: str, rows: List[Dict[str, object]]) -> None:
    # ABAQUS input fragment (.inp) with *MATERIAL blocks and temperature-dependent properties
    ensure_dir(os.path.dirname(path))
    lines = []
    lines.append("** Generated temperature-dependent material data")
    lines.append("*MATERIAL, NAME=MIX")
    # Density (temperature-independent in Abaqus base; we tabulate via *DENSITY, DEPENDENCIES=1)
    lines.append("*DENSITY, DEPENDENCIES=1")
    for r in rows:
        lines.append(f"{r['density_kgm3']:.3f}, {r['Temperature_C']:.1f}")
    # Specific heat
    lines.append("*SPECIFIC HEAT, DEPENDENCIES=1")
    for r in rows:
        lines.append(f"{r['specific_heat_jkgk']:.3f}, {r['Temperature_C']:.1f}")
    # Thermal conductivity
    lines.append("*CONDUCTIVITY, TYPE=ISOTROPIC, DEPENDENCIES=1")
    for r in rows:
        lines.append(f"{r['thermal_conductivity_wmk']:.6f}, {r['Temperature_C']:.1f}")
    # Elastic
    lines.append("*ELASTIC, TYPE=ISOTROPIC, DEPENDENCIES=1")
    for r in rows:
        lines.append(f"{r['elastic_modulus_pa']:.3f}, {r['poisson_ratio']:.6f}, {r['Temperature_C']:.1f}")
    # Thermal expansion
    lines.append("*EXPANSION, ZERO=20.")
    for r in rows:
        lines.append(f"{r['cte_1k']:.9f}, {r['Temperature_C']:.1f}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
