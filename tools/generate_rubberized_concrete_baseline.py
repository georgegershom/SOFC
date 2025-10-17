#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate a baseline dataset for:
  Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

Pillar 1: Material Characterization & Mixture Design (The "Before" State)

Outputs under datasets/rubberized_concrete_baseline_v1:
- mixture_proportions.csv
- constituents.json
- aggregates_characterization.json
- rubber_characterization.csv
- rubber_psd.csv
- fresh_properties.csv
- compressive_strength.csv (replicates)
- tensile_splitting.csv (replicates)
- static_modulus.csv (replicates)
- hardened_density.csv
- upv.csv (replicates)
- mip_summary.csv
- mip_curves/mix_*.csv (pore size distributions)
- tga/rubber_tga.csv
- ftir/rubber_ftir.csv
- curing_regime.json
- metadata.json
- data_dictionary.json

Notes:
- Values are fabricated but physically plausible and internally consistent.
- Randomness is seeded for reproducibility.
"""

from __future__ import annotations

import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev

# ----------------------------
# Configuration and constants
# ----------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets" / "rubberized_concrete_baseline_v1"
MIP_DIR = DATASET_DIR / "mip_curves"
TGA_DIR = DATASET_DIR / "tga"
FTIR_DIR = DATASET_DIR / "ftir"

DATASET_VERSION = "v1.0.0"

# Mixture IDs and rubber replacement by volume of fine aggregate (percent)
MIXES = [
    {"mix_id": "CTRL", "rubber_replacement_vol_pct": 0},
    {"mix_id": "R05", "rubber_replacement_vol_pct": 5},
    {"mix_id": "R10", "rubber_replacement_vol_pct": 10},
    {"mix_id": "R15", "rubber_replacement_vol_pct": 15},
]

# Constituent types and sources (example realistic identifiers)
CONSTITUENTS = {
    "cement": {
        "type": "CEM I 52.5R (ASTM Type I/II equivalent)",
        "source": "LafargeHolcim - Plant A",
        "specific_gravity": 3.15,
    },
    "coarse_aggregate": {
        "type": "Crushed granite 10 mm nominal size",
        "grading": "ASTM C33 No. 7 equivalent",
        "source": "BlueRock Quarry",
        "ssd_specific_gravity": 2.70,
        "water_absorption_pct": 0.5,
    },
    "fine_aggregate": {
        "type": "Natural river sand, Zone II",
        "grading": "ASTM C33 fine aggregate",
        "source": "Riverside Sands Co.",
        "ssd_specific_gravity": 2.65,
        "water_absorption_pct": 0.8,
    },
    "water": {
        "type": "Potable municipal water",
        "source": "City Water Utility",
    },
    "superplasticizer": {
        "type": "Polycarboxylate ether (PCE), Glenium-type",
        "source": "Master Builders Solutions",
        "dosage_pct_bwoc": 0.9,
    },
    "rubber": {
        "type": "Crumb rubber from end-of-life truck tires",
        "source": "EcoTread Recycling Ltd.",
        "particle_size_range_mm": [1.0, 4.0],
        "ssd_specific_gravity": 1.15,
        "water_absorption_pct": 1.5,
        "hardness_shore_A": 65,
        "pretreatment": "None",
        # For reference, if pretreatment were applied, include protocol details in metadata
    },
}

# Baseline mix (per 1 m^3 target) for control (approximate HPC)
BASE_MIX = {
    "cement_kg": 450.0,
    "water_kg": 157.5,  # w/c ~ 0.35
    "coarse_aggregate_kg": 1000.0,
    "fine_aggregate_kg": 650.0,
    # SP dosage as percent by weight of cement (bwoc)
    "superplasticizer_pct_bwoc": CONSTITUENTS["superplasticizer"]["dosage_pct_bwoc"],
}

# Units and helpers
KG_PER_M3_WATER = 1000.0

FINE_SG = CONSTITUENTS["fine_aggregate"]["ssd_specific_gravity"]
COARSE_SG = CONSTITUENTS["coarse_aggregate"]["ssd_specific_gravity"]
RUBBER_SG = CONSTITUENTS["rubber"]["ssd_specific_gravity"]

FINE_DENSITY = FINE_SG * 1000.0  # kg/m^3
COARSE_DENSITY = COARSE_SG * 1000.0
RUBBER_DENSITY = RUBBER_SG * 1000.0

# Fresh concrete property baselines and sensitivities vs rubber vol%
SLUMP_FLOW_CTRL_MM = 650.0
SLUMP_FLOW_DELTA_PER_5VOL_MM = -18.0
AIR_CONTENT_CTRL_PCT = 2.0
AIR_CONTENT_DELTA_PER_5VOL_PCT = 0.7

# Mechanical baselines (Control)
F_C28_CTRL_MPA = 65.0
F_C7_TO_28_RATIO = 0.75
F_T28_CTRL_MPA = 4.5
E_STATIC_CTRL_GPA = 33.0
UPV_CTRL_KM_S = 4.60

# Sensitivities (fractional reduction per 5% vol rubber replacement)
F_C28_DELTA_PER_5VOL = 0.07  # 7% reduction each 5% vol
F_C7_DELTA_PER_5VOL = 0.06
F_T28_DELTA_PER_5VOL = 0.05
E_STATIC_DELTA_PER_5VOL = 0.08
UPV_DELTA_PER_5VOL = 0.04

# MIP porosity baseline and sensitivity
POROSITY_CTRL_PCT = 10.0
POROSITY_DELTA_PER_5VOL_PCT = 2.0

# Replicates
NUM_REPLICATES = 3

# ----------------------------
# Utility functions
# ----------------------------


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def normal_with_cv(mean_value: float, cv: float) -> float:
    """Draw from normal with coefficient of variation (std/mean)."""
    std = abs(mean_value) * cv
    return random.gauss(mean_value, std)


def logspace(start: float, stop: float, num: int) -> list[float]:
    """Return log-spaced values between 10**start and 10**stop inclusive."""
    if num == 1:
        return [10 ** start]
    step = (stop - start) / (num - 1)
    return [10 ** (start + i * step) for i in range(num)]


# ----------------------------
# Mix design calculations
# ----------------------------


def compute_mix_proportions(rubber_vol_pct: float) -> dict:
    """Compute mix proportions per m^3 for a given rubber replacement by volume of fine aggregate."""
    cement_kg = BASE_MIX["cement_kg"]
    water_kg = BASE_MIX["water_kg"]
    coarse_kg = BASE_MIX["coarse_aggregate_kg"]
    fine_kg_ctrl = BASE_MIX["fine_aggregate_kg"]

    sp_pct_bwoc = BASE_MIX["superplasticizer_pct_bwoc"]
    sp_kg = cement_kg * (sp_pct_bwoc / 100.0)

    # Fine aggregate volume (m^3) at control
    fine_vol_m3_ctrl = fine_kg_ctrl / FINE_DENSITY

    # Volume to replace with rubber (m^3)
    repl_fraction = rubber_vol_pct / 100.0
    vol_replace_m3 = fine_vol_m3_ctrl * repl_fraction

    # Adjust masses
    fine_kg_new = fine_kg_ctrl - vol_replace_m3 * FINE_DENSITY
    rubber_kg = vol_replace_m3 * RUBBER_DENSITY

    # Sanity clamp (avoid negative fine mass due to rounding)
    fine_kg_new = max(0.0, fine_kg_new)

    total_mass_kg = cement_kg + water_kg + coarse_kg + fine_kg_new + rubber_kg + sp_kg

    return {
        "cement_kg": round(cement_kg, 2),
        "water_kg": round(water_kg, 2),
        "coarse_aggregate_kg": round(coarse_kg, 2),
        "fine_aggregate_kg": round(fine_kg_new, 2),
        "crumb_rubber_kg": round(rubber_kg, 2),
        "superplasticizer_kg": round(sp_kg, 2),
        "superplasticizer_pct_bwoc": sp_pct_bwoc,
        "total_batch_mass_kg_per_m3": round(total_mass_kg, 2),
    }


# ----------------------------
# Fresh properties
# ----------------------------


def generate_fresh_properties(rubber_vol_pct: float, total_mass_kg_per_m3: float) -> dict:
    per5 = rubber_vol_pct / 5.0

    slump_flow_mm = normal_with_cv(
        SLUMP_FLOW_CTRL_MM + SLUMP_FLOW_DELTA_PER_5VOL_MM * per5, cv=0.02
    )
    slump_flow_mm = clamp(slump_flow_mm, 450.0, 750.0)

    air_content_pct = normal_with_cv(
        AIR_CONTENT_CTRL_PCT + AIR_CONTENT_DELTA_PER_5VOL_PCT * per5, cv=0.08
    )
    air_content_pct = clamp(air_content_pct, 1.0, 10.0)

    # Fresh density approximately equals total mass per m^3 minus entrained air volume effect
    air_fraction = air_content_pct / 100.0
    fresh_density_kg_m3 = total_mass_kg_per_m3 * (1.0 - 0.3 * air_fraction)

    return {
        "slump_flow_mm": round(slump_flow_mm, 1),
        "air_content_pct": round(air_content_pct, 2),
        "fresh_density_kg_m3": round(fresh_density_kg_m3, 1),
    }


# ----------------------------
# Hardened properties and NDT
# ----------------------------


def generate_compressive_strengths(rubber_vol_pct: float, day: int) -> list[float]:
    per5 = rubber_vol_pct / 5.0
    if day == 28:
        base = F_C28_CTRL_MPA * (1.0 - F_C28_DELTA_PER_5VOL * per5)
        cv = 0.05
    elif day == 7:
        base = (F_C28_CTRL_MPA * F_C7_TO_28_RATIO) * (1.0 - F_C7_DELTA_PER_5VOL * per5)
        cv = 0.06
    else:
        raise ValueError("Unsupported day; expected 7 or 28")
    return [max(5.0, normal_with_cv(base, cv)) for _ in range(NUM_REPLICATES)]


def generate_tensile_splitting(rubber_vol_pct: float) -> list[float]:
    per5 = rubber_vol_pct / 5.0
    base = F_T28_CTRL_MPA * (1.0 - F_T28_DELTA_PER_5VOL * per5)
    return [max(0.5, normal_with_cv(base, 0.07)) for _ in range(NUM_REPLICATES)]


def generate_static_modulus(rubber_vol_pct: float) -> list[float]:
    per5 = rubber_vol_pct / 5.0
    base = E_STATIC_CTRL_GPA * (1.0 - E_STATIC_DELTA_PER_5VOL * per5)
    return [max(8.0, normal_with_cv(base, 0.05)) for _ in range(NUM_REPLICATES)]


def generate_upv(rubber_vol_pct: float) -> list[float]:
    per5 = rubber_vol_pct / 5.0
    base = UPV_CTRL_KM_S * (1.0 - UPV_DELTA_PER_5VOL * per5)
    return [max(2.5, normal_with_cv(base, 0.02)) for _ in range(NUM_REPLICATES)]


def generate_hardened_densities(mix_props: dict) -> dict:
    """
    Estimate oven-dry and SSD densities (kg/m^3) from constituent masses and absorptions.
    """
    cement_kg = mix_props["cement_kg"]
    water_kg = mix_props["water_kg"]
    coarse_kg = mix_props["coarse_aggregate_kg"]
    fine_kg = mix_props["fine_aggregate_kg"]
    rubber_kg = mix_props["crumb_rubber_kg"]

    oven_dry_mass = cement_kg + coarse_kg + fine_kg + rubber_kg  # water removed

    # Absorbed water mass at SSD
    coarse_abs = CONSTITUENTS["coarse_aggregate"]["water_absorption_pct"] / 100.0
    fine_abs = CONSTITUENTS["fine_aggregate"]["water_absorption_pct"] / 100.0
    rubber_abs = CONSTITUENTS["rubber"]["water_absorption_pct"] / 100.0

    absorbed_water_kg = coarse_kg * coarse_abs + fine_kg * fine_abs + rubber_kg * rubber_abs

    oven_dry_density = oven_dry_mass  # per m^3 target
    ssd_density = oven_dry_mass + absorbed_water_kg

    return {
        "oven_dry_density_kg_m3": round(oven_dry_density, 1),
        "ssd_density_kg_m3": round(ssd_density, 1),
    }


# ----------------------------
# MIP generation (pore size distribution)
# ----------------------------


def generate_mip_curve(rubber_vol_pct: float) -> tuple[list[dict], dict]:
    """
    Generate a pore size distribution over 0.01–100 µm using a bimodal lognormal mixture.
    Returns (curve_rows, summary_dict).
    """
    # Total porosity
    per5 = rubber_vol_pct / 5.0
    total_porosity_pct = POROSITY_CTRL_PCT + POROSITY_DELTA_PER_5VOL_PCT * per5

    # Diameter grid (µm)
    d_um = logspace(-2.0, 2.0, 200)  # 0.01 to 100 µm

    # Lognormal mixture parameters
    # Micropores centered ~0.05 µm; macropores centered ~1.0 µm
    micro_mu = math.log(0.05)  # ln scale
    micro_sigma = 0.6
    macro_mu = math.log(1.0)
    macro_sigma = 0.5

    # Weight shifts toward macro with more rubber
    macro_weight = 0.25 + 0.55 * (rubber_vol_pct / 15.0)
    macro_weight = clamp(macro_weight, 0.1, 0.9)
    micro_weight = 1.0 - macro_weight

    # Lognormal pdf in terms of diameter
    def lnpdf(x, mu, sigma):
        if x <= 0:
            return 0.0
        return (1.0 / (x * sigma * math.sqrt(2 * math.pi))) * math.exp(-((math.log(x) - mu) ** 2) / (2 * sigma**2))

    # Differential intrusion per d(log D) scaled to total porosity
    pdf_vals = [micro_weight * lnpdf(x, micro_mu, micro_sigma) + macro_weight * lnpdf(x, macro_mu, macro_sigma) for x in d_um]

    # Convert to differential per log10 step: scale by x * ln(10)
    # We'll normalize such that integral over log10(D) equals total_porosity_pct
    # Approximate integral via trapezoidal rule in log10 space
    log10_d = [math.log10(x) for x in d_um]

    # Scale pdf to porosity
    # Compute raw area under curve in log10 space
    area = 0.0
    for i in range(1, len(d_um)):
        dx = log10_d[i] - log10_d[i - 1]
        area += 0.5 * (pdf_vals[i] + pdf_vals[i - 1]) * dx
    scale = (total_porosity_pct / 100.0) / area if area > 0 else 1.0
    pdf_vals = [v * scale for v in pdf_vals]

    # Build cumulative porosity and an arbitrary intrusion unit (mm^3/g) scaled to porosity
    cumulative = []
    cum = 0.0
    rows = []
    for i in range(len(d_um)):
        if i == 0:
            incr = 0.0
        else:
            dx = log10_d[i] - log10_d[i - 1]
            incr = 0.5 * (pdf_vals[i] + pdf_vals[i - 1]) * dx
        cum += incr
        cumulative.append(cum)
    # Normalize cumulative to total_porosity
    cumulative = [c for c in cumulative]

    # For reporting, convert cumulative to percent
    cum_pct = [c * 100.0 for c in cumulative]

    # Threshold and median pore diameters
    # Threshold diameter ~ where differential intrusion peaks
    peak_idx = max(range(len(pdf_vals)), key=lambda i: pdf_vals[i])
    threshold_d_um = d_um[peak_idx]

    # Median (50%) pore diameter
    def interp_x_at_y(x_list, y_list, target):
        for i in range(1, len(x_list)):
            if y_list[i - 1] <= target <= y_list[i]:
                # Linear interpolation in cumulative space
                x0, x1 = x_list[i - 1], x_list[i]
                y0, y1 = y_list[i - 1], y_list[i]
                if y1 == y0:
                    return x0
                t = (target - y0) / (y1 - y0)
                return x0 + t * (x1 - x0)
        return x_list[-1]

    median_d_um = interp_x_at_y(d_um, cum_pct, (total_porosity_pct))  # 100% of fractional porosity -> 100%?
    # Above cum_pct already in %, cumulative ends near total_porosity_pct.
    # For 50% of porosity:
    median_d_um = interp_x_at_y(d_um, cum_pct, (total_porosity_pct / 2.0))

    for i in range(len(d_um)):
        rows.append({
            "pore_diameter_um": round(d_um[i], 6),
            "d_porosity_d_log10D": round(pdf_vals[i], 6),
            "cumulative_porosity_pct": round(cum_pct[i], 4),
        })

    summary = {
        "total_porosity_pct": round(total_porosity_pct, 2),
        "threshold_pore_diameter_um": round(threshold_d_um, 3),
        "median_pore_diameter_um": round(median_d_um, 3),
    }

    return rows, summary


# ----------------------------
# Rubber TGA & FTIR (raw rubber characterization)
# ----------------------------


def generate_tga() -> list[dict]:
    """Generate TGA curve for crumb rubber from 25–800 C. Mass percent vs temperature."""
    points = []
    mass_pct = 100.0
    for T in range(25, 801, 5):
        # Piecewise mass loss: 200–350 C (plasticizers), 350–550 C (polymer), 600–800 C (char oxidation)
        if T < 200:
            dm = 0.0
        elif 200 <= T < 350:
            dm = 12.0 * (T - 200) / (350 - 200)
        elif 350 <= T < 550:
            dm = 12.0 + 45.0 * (T - 350) / (200)
        elif 550 <= T < 600:
            dm = 57.0 + 3.0 * (T - 550) / 50.0
        else:
            dm = 60.0 + 10.0 * (T - 600) / 200.0
        noisy_dm = dm + random.gauss(0.0, 0.3)
        mass = clamp(100.0 - noisy_dm, 5.0, 100.0)
        points.append({"temperature_C": T, "mass_percent": round(mass, 2)})
    return points


def generate_ftir() -> list[dict]:
    """Generate FTIR spectrum for crumb rubber from 4000–600 cm-1."""
    # Known peaks for rubber/tire material (approx): 2950, 2920, 2850, 1730, 1600, 1450, 1377, 965, 835, 700 cm-1
    peaks = [
        (2950, 0.7), (2920, 1.0), (2850, 0.8), (1730, 0.5), (1600, 0.6), (1450, 0.9), (1377, 0.7), (965, 1.0), (835, 0.6), (700, 0.5)
    ]
    # Gaussian width (std dev) in cm-1
    width = 25.0

    def gaussian(x, mu, amp, sigma):
        return amp * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

    rows = []
    for wn in range(4000, 599, -4):
        baseline = 0.05 + 0.02 * math.sin(wn / 200.0)
        intensity = baseline
        for (mu, amp) in peaks:
            intensity += gaussian(wn, mu, amp, width)
        # Add mild noise
        intensity += random.gauss(0.0, 0.01)
        intensity = clamp(intensity, 0.0, 1.5)
        rows.append({"wavenumber_cm_1": wn, "absorbance": round(intensity, 4)})
    return rows


# ----------------------------
# Aggregates PSD (basic synthetic grading curves)
# ----------------------------


def make_psd_rows(sizes_mm: list[float], target_curve: list[float]) -> list[dict]:
    return [{"sieve_mm": round(s, 3), "percent_passing": round(p, 1)} for s, p in zip(sizes_mm, target_curve)]


def generate_aggregate_psd() -> dict:
    # Fine: Zone II typical passing curve
    fine_sizes = [9.5, 4.75, 2.36, 1.18, 0.6, 0.3, 0.15]
    fine_passing = [100, 95, 80, 60, 35, 15, 3]

    # Coarse: No. 7 10 mm nominal
    coarse_sizes = [25, 19, 12.5, 9.5, 4.75]
    coarse_passing = [100, 98, 70, 35, 5]

    return {
        "fine_psd": make_psd_rows(fine_sizes, fine_passing),
        "coarse_psd": make_psd_rows(coarse_sizes, coarse_passing),
    }


def generate_rubber_psd() -> list[dict]:
    # Rubber target range 1–4 mm, centered ~2 mm
    bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0)]
    # Create a bell-shaped distribution
    weights = [1, 2, 3, 3, 2, 1]
    total_w = sum(weights)
    return [
        {
            "size_min_mm": a,
            "size_max_mm": b,
            "percent_mass": round(100.0 * w / total_w + random.gauss(0, 1.0), 1),
        }
        for (a, b), w in zip(bins, weights)
    ]


# ----------------------------
# Writers
# ----------------------------


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ----------------------------
# Main generation
# ----------------------------


def main():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MIP_DIR.mkdir(parents=True, exist_ok=True)
    TGA_DIR.mkdir(parents=True, exist_ok=True)
    FTIR_DIR.mkdir(parents=True, exist_ok=True)

    # Constituents and aggregates characterization
    ag_psd = generate_aggregate_psd()
    write_json(DATASET_DIR / "aggregates_characterization.json", {
        "coarse": {
            "type": CONSTITUENTS["coarse_aggregate"]["type"],
            "source": CONSTITUENTS["coarse_aggregate"]["source"],
            "ssd_specific_gravity": CONSTITUENTS["coarse_aggregate"]["ssd_specific_gravity"],
            "water_absorption_pct": CONSTITUENTS["coarse_aggregate"]["water_absorption_pct"],
            "psd": ag_psd["coarse_psd"],
        },
        "fine": {
            "type": CONSTITUENTS["fine_aggregate"]["type"],
            "source": CONSTITUENTS["fine_aggregate"]["source"],
            "ssd_specific_gravity": CONSTITUENTS["fine_aggregate"]["ssd_specific_gravity"],
            "water_absorption_pct": CONSTITUENTS["fine_aggregate"]["water_absorption_pct"],
            "psd": ag_psd["fine_psd"],
        }
    })

    write_json(DATASET_DIR / "constituents.json", CONSTITUENTS)

    rubber_psd_rows = generate_rubber_psd()
    write_csv(DATASET_DIR / "rubber_psd.csv", ["size_min_mm", "size_max_mm", "percent_mass"], rubber_psd_rows)

    write_csv(DATASET_DIR / "rubber_characterization.csv", [
        "source", "type", "particle_size_min_mm", "particle_size_max_mm", "ssd_specific_gravity",
        "water_absorption_pct", "hardness_shore_A", "pretreatment"
    ], [{
        "source": CONSTITUENTS["rubber"]["source"],
        "type": CONSTITUENTS["rubber"]["type"],
        "particle_size_min_mm": CONSTITUENTS["rubber"]["particle_size_range_mm"][0],
        "particle_size_max_mm": CONSTITUENTS["rubber"]["particle_size_range_mm"][1],
        "ssd_specific_gravity": CONSTITUENTS["rubber"]["ssd_specific_gravity"],
        "water_absorption_pct": CONSTITUENTS["rubber"]["water_absorption_pct"],
        "hardness_shore_A": CONSTITUENTS["rubber"]["hardness_shore_A"],
        "pretreatment": CONSTITUENTS["rubber"]["pretreatment"],
    }])

    # Mixture proportions
    mix_rows = []
    fresh_rows = []
    dens_rows = []

    comp_rows = []  # compressive
    tens_rows = []  # tensile splitting
    estatic_rows = []  # static modulus
    upv_rows = []

    mip_summary_rows = []

    for mix in MIXES:
        mix_id = mix["mix_id"]
        rvp = mix["rubber_replacement_vol_pct"]

        props = compute_mix_proportions(rvp)
        props_out = {
            "mix_id": mix_id,
            "rubber_replacement_vol_pct": rvp,
            **props,
        }
        mix_rows.append(props_out)

        fresh = generate_fresh_properties(rvp, props["total_batch_mass_kg_per_m3"])
        fresh_rows.append({"mix_id": mix_id, **fresh})

        dens = generate_hardened_densities(props)
        dens_rows.append({"mix_id": mix_id, **dens})

        # Replicates for mechanicals
        for day in (7, 28):
            strengths = generate_compressive_strengths(rvp, day)
            for i, s in enumerate(strengths, start=1):
                comp_rows.append({
                    "mix_id": mix_id,
                    "day": day,
                    "specimen_id": f"S{i}",
                    "compressive_strength_mpa": round(s, 2),
                })

        for i, val in enumerate(generate_tensile_splitting(rvp), start=1):
            tens_rows.append({
                "mix_id": mix_id,
                "day": 28,
                "specimen_id": f"S{i}",
                "tensile_splitting_strength_mpa": round(val, 2),
            })

        for i, val in enumerate(generate_static_modulus(rvp), start=1):
            estatic_rows.append({
                "mix_id": mix_id,
                "day": 28,
                "specimen_id": f"S{i}",
                "static_modulus_gpa": round(val, 2),
            })

        for i, val in enumerate(generate_upv(rvp), start=1):
            upv_rows.append({
                "mix_id": mix_id,
                "day": 28,
                "specimen_id": f"S{i}",
                "upv_km_s": round(val, 3),
            })

        # MIP curves & summary
        mip_rows, mip_summary = generate_mip_curve(rvp)
        write_csv(MIP_DIR / f"mix_{mix_id}_mip.csv", [
            "pore_diameter_um", "d_porosity_d_log10D", "cumulative_porosity_pct"
        ], mip_rows)
        mip_summary_rows.append({"mix_id": mix_id, **mip_summary})

    # Write primary CSVs
    write_csv(DATASET_DIR / "mixture_proportions.csv", [
        "mix_id", "rubber_replacement_vol_pct", "cement_kg", "water_kg",
        "coarse_aggregate_kg", "fine_aggregate_kg", "crumb_rubber_kg",
        "superplasticizer_kg", "superplasticizer_pct_bwoc", "total_batch_mass_kg_per_m3"
    ], mix_rows)

    write_csv(DATASET_DIR / "fresh_properties.csv", [
        "mix_id", "slump_flow_mm", "air_content_pct", "fresh_density_kg_m3"
    ], fresh_rows)

    write_csv(DATASET_DIR / "hardened_density.csv", [
        "mix_id", "oven_dry_density_kg_m3", "ssd_density_kg_m3"
    ], dens_rows)

    write_csv(DATASET_DIR / "compressive_strength.csv", [
        "mix_id", "day", "specimen_id", "compressive_strength_mpa"
    ], comp_rows)

    write_csv(DATASET_DIR / "tensile_splitting.csv", [
        "mix_id", "day", "specimen_id", "tensile_splitting_strength_mpa"
    ], tens_rows)

    write_csv(DATASET_DIR / "static_modulus.csv", [
        "mix_id", "day", "specimen_id", "static_modulus_gpa"
    ], estatic_rows)

    write_csv(DATASET_DIR / "upv.csv", [
        "mix_id", "day", "specimen_id", "upv_km_s"
    ], upv_rows)

    write_csv(DATASET_DIR / "mip_summary.csv", [
        "mix_id", "total_porosity_pct", "threshold_pore_diameter_um", "median_pore_diameter_um"
    ], mip_summary_rows)

    # Rubber TGA & FTIR
    tga_rows = generate_tga()
    write_csv(TGA_DIR / "rubber_tga.csv", ["temperature_C", "mass_percent"], tga_rows)

    ftir_rows = generate_ftir()
    write_csv(FTIR_DIR / "rubber_ftir.csv", ["wavenumber_cm_1", "absorbance"], ftir_rows)

    # Curing regime
    write_json(DATASET_DIR / "curing_regime.json", {
        "curing": {
            "description": "28 days in lime-saturated water at 23°C (73.4°F)",
            "duration_days": 28,
            "temperature_C": 23,
            "medium": "Lime-saturated water",
        },
        "applies_to": [m["mix_id"] for m in MIXES]
    })

    # Metadata
    write_json(DATASET_DIR / "metadata.json", {
        "dataset_name": "Baseline: High-Performance Rubberized Concrete",
        "pillar": "Pillar 1: Material Characterization & Mixture Design",
        "topic": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
        "version": DATASET_VERSION,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "random_seed": RANDOM_SEED,
        "units": {
            "mass": "kg",
            "density": "kg/m^3",
            "strength": "MPa",
            "modulus": "GPa",
            "velocity": "km/s",
            "porosity": "%",
            "diameter": "µm",
            "temperature": "°C",
        },
        "notes": [
            "Fabricated dataset with physically plausible trends vs rubber replacement.",
            "Rubber replacement is by volume of fine aggregate at 0%, 5%, 10%, 15%.",
            "Random noise applied for specimen-level replicates; seeded for reproducibility.",
        ],
    })

    # Data dictionary (JSON, not markdown)
    data_dict = {
        "mixture_proportions.csv": {
            "mix_id": "Mixture identifier (CTRL, R05, R10, R15)",
            "rubber_replacement_vol_pct": "% volume of fine aggregate replaced by crumb rubber",
            "cement_kg": "kg per m^3",
            "water_kg": "kg per m^3",
            "coarse_aggregate_kg": "kg per m^3",
            "fine_aggregate_kg": "kg per m^3",
            "crumb_rubber_kg": "kg per m^3",
            "superplasticizer_kg": "kg per m^3",
            "superplasticizer_pct_bwoc": "% by weight of cement",
            "total_batch_mass_kg_per_m3": "Sum of constituents per m^3",
        },
        "fresh_properties.csv": {
            "slump_flow_mm": "mm (slump flow)",
            "air_content_pct": "%",
            "fresh_density_kg_m3": "kg/m^3",
        },
        "hardened_density.csv": {
            "oven_dry_density_kg_m3": "kg/m^3",
            "ssd_density_kg_m3": "kg/m^3",
        },
        "compressive_strength.csv": {
            "day": "Age in days (7, 28)",
            "specimen_id": "Replicate identifier",
            "compressive_strength_mpa": "MPa",
        },
        "tensile_splitting.csv": {
            "day": "Age in days (28)",
            "specimen_id": "Replicate identifier",
            "tensile_splitting_strength_mpa": "MPa",
        },
        "static_modulus.csv": {
            "day": "Age in days (28)",
            "specimen_id": "Replicate identifier",
            "static_modulus_gpa": "GPa",
        },
        "upv.csv": {
            "day": "Age in days (28)",
            "specimen_id": "Replicate identifier",
            "upv_km_s": "km/s",
        },
        "mip_summary.csv": {
            "total_porosity_pct": "% by intruded volume",
            "threshold_pore_diameter_um": "µm",
            "median_pore_diameter_um": "µm",
        },
        "mip_curves/mix_*.csv": {
            "pore_diameter_um": "µm",
            "d_porosity_d_log10D": "Fraction per log10(D), scaled to total porosity",
            "cumulative_porosity_pct": "%",
        },
        "tga/rubber_tga.csv": {
            "temperature_C": "°C",
            "mass_percent": "% remaining mass",
        },
        "ftir/rubber_ftir.csv": {
            "wavenumber_cm_1": "cm^-1",
            "absorbance": "a.u.",
        },
        "rubber_psd.csv": {
            "size_min_mm": "mm",
            "size_max_mm": "mm",
            "percent_mass": "% in bin",
        },
        "rubber_characterization.csv": {
            "source": "Supplier/source",
            "type": "Material type",
            "particle_size_min_mm": "mm",
            "particle_size_max_mm": "mm",
            "ssd_specific_gravity": "-",
            "water_absorption_pct": "%",
            "hardness_shore_A": "Shore A",
            "pretreatment": "Procedure applied (if any)",
        },
        "aggregates_characterization.json": {
            "coarse": "Object with properties and PSD list",
            "fine": "Object with properties and PSD list",
        },
        "constituents.json": {
            "cement": "Cement details",
            "coarse_aggregate": "Coarse aggregate details",
            "fine_aggregate": "Fine aggregate details",
            "water": "Water details",
            "superplasticizer": "PCE details",
            "rubber": "Rubber details",
        },
        "curing_regime.json": {
            "curing": "Curing method details",
            "applies_to": "List of mix IDs",
        },
        "metadata.json": {
            "dataset_name": "Name",
            "pillar": "Pillar description",
            "topic": "Research topic",
            "version": "Dataset version",
            "generated_at": "UTC timestamp",
            "random_seed": "Integer seed",
            "units": "Units map",
            "notes": "List of annotations",
        },
    }
    write_json(DATASET_DIR / "data_dictionary.json", data_dict)

    print(f"Dataset generated at: {DATASET_DIR}")


if __name__ == "__main__":
    main()
