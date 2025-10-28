#!/usr/bin/env python3
import os
import math
import csv
import random
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import yaml

RANDOM_SEED = 1337
random.seed(RANDOM_SEED)

@dataclass
class Material:
    name: str
    category: str
    include_electrochem: bool

# Category priors (rough, literature-inspired ranges)
CATEGORY_PRIORS = {
    "metal": {
        "E_GPa_20C": (60, 220),
        "UTS_MPa_20C": (150, 1200),
        "nu": (0.28, 0.36),
        "cte_uK": (9e-6, 21e-6),
        "k_WmK": (20, 400),
        "cp_JkgK": (380, 950),
        "creep_A": (1e-18, 1e-12),
        "creep_n": (3.5, 8.5),
        "creep_Q_kJmol": (180, 350),
    },
    "superalloy": {
        "E_GPa_20C": (150, 220),
        "UTS_MPa_20C": (800, 1500),
        "nu": (0.28, 0.34),
        "cte_uK": (11e-6, 16e-6),
        "k_WmK": (10, 30),
        "cp_JkgK": (380, 750),
        "creep_A": (1e-22, 1e-14),
        "creep_n": (4.0, 7.5),
        "creep_Q_kJmol": (250, 400),
    },
    "ceramic": {
        "E_GPa_20C": (200, 450),
        "UTS_MPa_20C": (100, 600),
        "nu": (0.17, 0.26),
        "cte_uK": (3e-6, 10e-6),
        "k_WmK": (2, 120),
        "cp_JkgK": (500, 900),
        "creep_A": (1e-30, 1e-20),
        "creep_n": (1.5, 3.0),
        "creep_Q_kJmol": (400, 700),
    },
    "polymer": {
        "E_GPa_20C": (1.0, 6.0),
        "UTS_MPa_20C": (30, 180),
        "nu": (0.35, 0.46),
        "cte_uK": (50e-6, 200e-6),
        "k_WmK": (0.1, 0.4),
        "cp_JkgK": (900, 2200),
        "creep_A": (1e-10, 1e-6),
        "creep_n": (1.0, 2.0),
        "creep_Q_kJmol": (50, 150),
    },
    "carbon": {
        "E_GPa_20C": (5, 30),
        "UTS_MPa_20C": (20, 120),
        "nu": (0.10, 0.25),
        "cte_uK": (-1e-6, 3e-6),
        "k_WmK": (50, 450),
        "cp_JkgK": (700, 1100),
        "creep_A": (1e-25, 1e-16),
        "creep_n": (1.5, 3.0),
        "creep_Q_kJmol": (200, 500),
    },
    # Battery-related categories (electrochem enabled)
    "battery_active": {
        "E_GPa_20C": (10, 200),
        "UTS_MPa_20C": (50, 800),
        "nu": (0.20, 0.34),
        "cte_uK": (5e-6, 15e-6),
        "k_WmK": (1, 10),
        "cp_JkgK": (500, 1000),
        "creep_A": (1e-20, 1e-12),
        "creep_n": (1.5, 4.0),
        "creep_Q_kJmol": (150, 300),
        "ionic_sigma_Sm_20C": (1e-8, 1e-5),
        "electronic_sigma_Sm_20C": (1, 1000),
        "Ea_ionic_kJmol": (15, 70),
        "Ea_elec_kJmol": (2, 20),
    },
    "solid_electrolyte": {
        "E_GPa_20C": (20, 200),
        "UTS_MPa_20C": (10, 300),
        "nu": (0.20, 0.33),
        "cte_uK": (5e-6, 15e-6),
        "k_WmK": (0.5, 4),
        "cp_JkgK": (400, 900),
        "creep_A": (1e-24, 1e-16),
        "creep_n": (1.0, 2.0),
        "creep_Q_kJmol": (250, 450),
        "ionic_sigma_Sm_20C": (1e-6, 1e-2),
        "electronic_sigma_Sm_20C": (1e-6, 1e-3),
        "Ea_ionic_kJmol": (20, 60),
        "Ea_elec_kJmol": (40, 100),
    },
    "liquid_electrolyte": {
        "E_GPa_20C": (0.001, 0.01),
        "UTS_MPa_20C": (0.1, 1.0),
        "nu": (0.45, 0.49),
        "cte_uK": (200e-6, 900e-6),
        "k_WmK": (0.1, 0.8),
        "cp_JkgK": (1500, 3000),
        "creep_A": (1e-6, 1e-3),
        "creep_n": (1.0, 1.5),
        "creep_Q_kJmol": (10, 50),
        "ionic_sigma_Sm_20C": (0.1, 2.0),
        "electronic_sigma_Sm_20C": (1e-8, 1e-6),
        "Ea_ionic_kJmol": (8, 25),
        "Ea_elec_kJmol": (15, 60),
    },
}

K_B = 8.617333262e-5  # eV/K
R_GAS = 8.314462618  # J/(mol*K)


def sample_uniform(lo: float, hi: float) -> float:
    return lo + (hi - lo) * random.random()


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_materials(materials_yaml_path: str) -> List[Material]:
    with open(materials_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    materials = []
    for m in data["materials"]:
        materials.append(Material(
            name=m["name"],
            category=m["category"],
            include_electrochem=bool(m.get("include_electrochem", False))
        ))
    return materials


def temperature_grid(category: str) -> List[float]:
    # Define plausible temperature ranges per category (Celsius)
    mapping = {
        "polymer": list(range(-40, 181, 20)),
        "liquid_electrolyte": list(range(-20, 101, 10)),
        "battery_active": list(range(-20, 201, 20)),
        "solid_electrolyte": list(range(-20, 201, 20)),
        "carbon": list(range(20, 901, 50)),
        "ceramic": list(range(20, 1201, 50)),
        "superalloy": list(range(20, 1001, 50)),
        "metal": list(range(-100, 901, 50)),
    }
    return mapping.get(category, list(range(20, 501, 25)))


def stress_levels_MPa(category: str) -> List[float]:
    # Pick representative creep stress levels
    mapping = {
        "polymer": [2, 5, 10],
        "liquid_electrolyte": [0.1, 0.2],
        "battery_active": [10, 50, 100],
        "solid_electrolyte": [5, 20, 50],
        "carbon": [5, 20, 60],
        "ceramic": [10, 30, 60],
        "superalloy": [100, 200, 400],
        "metal": [50, 150, 300],
    }
    return mapping.get(category, [50, 150, 300])


def time_grid_hours(category: str) -> List[float]:
    # Log-spaced hours suitable for creep curves
    if category in {"polymer", "liquid_electrolyte"}:
        return [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    return [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]


def draw_category_parameters(category: str) -> Dict[str, float]:
    p = CATEGORY_PRIORS[category]
    params = {
        "E0": sample_uniform(*p["E_GPa_20C"]),
        "UTS0": sample_uniform(*p["UTS_MPa_20C"]),
        "nu0": sample_uniform(*p["nu"]),
        "cte0": sample_uniform(*p["cte_uK"]),
        "k0": sample_uniform(*p["k_WmK"]),
        "cp0": sample_uniform(*p["cp_JkgK"]),
        # Slopes/curvatures to shape T dependence
        "dE_per_C": -sample_uniform(0.01, 0.25),  # GPa per C
        "dUTS_per_C": -sample_uniform(0.1, 1.5),  # MPa per C
        "dnu_per_C": sample_uniform(1e-4, 8e-4),
        "dcte_frac_per_C": sample_uniform(1e-4, 4e-4),
        "dk_per_C": -sample_uniform(0.01, 0.4) if category in {"metal", "superalloy", "carbon"} else sample_uniform(0.0, 0.15),
        "dcp_per_C": sample_uniform(0.5, 2.0),
        # Creep (Norton-Bailey): eps = A * sigma^n * t^m * exp(-Q/(R*T))
        "A": sample_uniform(*p["creep_A"]),
        "n": sample_uniform(*p["creep_n"]),
        "m": sample_uniform(0.2, 1.2),
        "Q": sample_uniform(*p["creep_Q_kJmol"]) * 1000.0,  # to J/mol
    }
    # Electrochem if present
    if "ionic_sigma_Sm_20C" in p:
        params.update({
            "sigma_ion_20C": sample_uniform(*p["ionic_sigma_Sm_20C"]),
            "sigma_elec_20C": sample_uniform(*p["electronic_sigma_Sm_20C"]),
            "Ea_ion": sample_uniform(*p["Ea_ionic_kJmol"]) * 1000.0,
            "Ea_elec": sample_uniform(*p["Ea_elec_kJmol"]) * 1000.0,
        })
    return params


def mechanical_properties_at_T(params: Dict[str, float], T_C: float) -> Tuple[float, float, float]:
    E = max(0.05, params["E0"] + params["dE_per_C"] * (T_C - 20))
    UTS = max(5.0, params["UTS0"] + params["dUTS_per_C"] * (T_C - 20))
    nu = clamp(params["nu0"] + params["dnu_per_C"] * (T_C - 20), 0.15, 0.49)
    return E, UTS, nu


def thermal_properties_at_T(params: Dict[str, float], T_C: float, category: str) -> Tuple[float, float, float]:
    cte = max(-1e-6, params["cte0"] * (1 + params["dcte_frac_per_C"] * (T_C - 20)))
    if category in {"metal", "superalloy", "carbon"}:
        k = max(0.05, params["k0"] + params["dk_per_C"] * (T_C - 20))
    else:
        k = max(0.05, params["k0"] + abs(params["dk_per_C"]) * 0.2 * (T_C - 20))
    cp = max(50.0, params["cp0"] + params["dcp_per_C"] * (T_C - 20))
    return cte, k, cp


def creep_strain(params: Dict[str, float], T_C: float, sigma_MPa: float, t_h: float) -> float:
    T_K = T_C + 273.15
    A = params["A"]
    n = params["n"]
    m = params["m"]
    Q = params["Q"]
    sig = max(0.01, sigma_MPa)
    t = max(1e-6, t_h)
    arrhenius = math.exp(-Q / (R_GAS * T_K))
    eps = A * (sig ** n) * (t ** m) * arrhenius
    # Add small lognormal noise to add realism
    noise = math.exp(random.gauss(0.0, 0.15))
    return max(1e-8, eps * noise)


def conductivity_Arrhenius(sigma_ref: float, Ea_Jmol: float, T_C: float, Tref_C: float = 20.0) -> float:
    T = T_C + 273.15
    Tref = Tref_C + 273.15
    sigma = sigma_ref * math.exp(-Ea_Jmol / R_GAS * (1.0 / T - 1.0 / Tref))
    return max(1e-10, sigma)


def write_csv(path: str, header: List[str], rows: List[List]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            writer.writerow(r)


def generate(materials_path: str, out_dir: str):
    materials = load_materials(materials_path)

    mechanical_rows: List[List] = []
    creep_rows: List[List] = []
    thermal_rows: List[List] = []
    electrochem_rows: List[List] = []

    for mat in materials:
        if mat.category not in CATEGORY_PRIORS:
            # fallback to metal priors
            category = "metal"
        else:
            category = mat.category
        params = draw_category_parameters(category)
        temps = temperature_grid(category)

        # Mechanical and thermal
        for T in temps:
            E, UTS, nu = mechanical_properties_at_T(params, T)
            cte, k, cp = thermal_properties_at_T(params, T, category)
            mechanical_rows.append([
                mat.name, category, float(T), round(E, 3), round(UTS, 2), round(nu, 4)
            ])
            thermal_rows.append([
                mat.name, category, float(T), f"{cte:.9g}", round(k, 4), round(cp, 2)
            ])

        # Creep (subset of temperatures/stresses)
        for T in temps[:: max(1, len(temps)//6)]:
            for sigma in stress_levels_MPa(category):
                for t in time_grid_hours(category):
                    eps = creep_strain(params, T, sigma, t)
                    creep_rows.append([
                        mat.name, category, float(T), float(sigma), float(t), f"{eps:.6g}"
                    ])

        # Electrochem
        if mat.include_electrochem:
            if "sigma_ion_20C" in params:
                for T in temps:
                    sig_i = conductivity_Arrhenius(params["sigma_ion_20C"], params["Ea_ion"], T)
                    sig_e = conductivity_Arrhenius(params["sigma_elec_20C"], params["Ea_elec"], T)
                    electrochem_rows.append([
                        mat.name, category, float(T), f"{sig_i:.6g}", f"{sig_e:.6g}"
                    ])

    # Write outputs
    write_csv(os.path.join(out_dir, "mechanical.csv"),
              ["material", "category", "temperature_C", "youngs_modulus_GPa", "tensile_strength_MPa", "poisson_ratio"],
              mechanical_rows)

    write_csv(os.path.join(out_dir, "creep.csv"),
              ["material", "category", "temperature_C", "stress_MPa", "time_h", "creep_strain"],
              creep_rows)

    write_csv(os.path.join(out_dir, "thermal.csv"),
              ["material", "category", "temperature_C", "cte_1_per_K", "thermal_conductivity_W_mK", "specific_heat_J_kgK"],
              thermal_rows)

    if electrochem_rows:
        write_csv(os.path.join(out_dir, "electrochemical.csv"),
                  ["material", "category", "temperature_C", "ionic_conductivity_S_m", "electronic_conductivity_S_m"],
                  electrochem_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic material properties dataset")
    parser.add_argument("--materials", type=str, default="data/material_properties/materials.yaml")
    parser.add_argument("--out", type=str, default="data/material_properties")
    args = parser.parse_args()

    generate(args.materials, args.out)
    print(f"Wrote dataset to {args.out}")
