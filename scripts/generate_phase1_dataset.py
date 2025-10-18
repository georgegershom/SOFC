#!/usr/bin/env python3
"""
Generate Phase 1 baseline dataset for:
  Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

Scope (Phase 1: Material Characterization & Specimen Preparation):
- Constituent materials data
- Characterization data (TGA, FTIR, SEM-like images [PGM])
- Mix designs (Control + rubberized at 5, 10, 15, 20% vol replacement)
- Fresh concrete properties for each batch (slump/flow, air, unit weight, temperature)

This script fabricates plausible, self-consistent synthetic data for research prototyping.
It is reproducible (seeded RNG) and writes both CSV and JSON where appropriate.
No third-party dependencies required.
"""

from __future__ import annotations
import csv
import json
import math
import os
import random
import shutil
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# -----------------------------
# Configuration and parameters
# -----------------------------
SEED = 42
random.seed(SEED)

DATASET_ROOT = os.path.join("data", "phase1_baseline")
MATERIALS_DIR = os.path.join(DATASET_ROOT, "materials")
CEMENT_DIR = os.path.join(MATERIALS_DIR, "cement")
AGG_DIR = os.path.join(MATERIALS_DIR, "aggregates")
FINE_DIR = os.path.join(AGG_DIR, "fine")
COARSE_DIR = os.path.join(AGG_DIR, "coarse")
RUBBER_DIR = os.path.join(MATERIALS_DIR, "crumb_rubber")
RUBBER_A_DIR = os.path.join(RUBBER_DIR, "size_1_4mm")
RUBBER_B_DIR = os.path.join(RUBBER_DIR, "size_4_8mm")
WATER_DIR = os.path.join(MATERIALS_DIR, "water")
ADMIX_DIR = os.path.join(MATERIALS_DIR, "chemical_admixture")

CHAR_DIR = os.path.join(DATASET_ROOT, "characterization")
TGA_DIR = os.path.join(CHAR_DIR, "tga")
FTIR_DIR = os.path.join(CHAR_DIR, "ftir")
SEM_DIR = os.path.join(CHAR_DIR, "sem_images")

MIXES_DIR = os.path.join(DATASET_ROOT, "mixes")
FRESH_DIR = os.path.join(DATASET_ROOT, "fresh_properties")

README_PATH = os.path.join(DATASET_ROOT, "README.md")
ZIP_OUTPUT = os.path.abspath("dataset_phase1_baseline")  # shutil.make_archive adds .zip

# Mixture matrix
RUBBER_CONTENTS = [0, 5, 10, 15, 20]  # volume % replacement of fine aggregate
RUBBER_SIZES = [
    ("none", None),  # for control only
    ("1-4mm", RUBBER_A_DIR),
    ("4-8mm", RUBBER_B_DIR),
]

# Base material properties used for mix calculations
SPECIFIC_GRAVITY = {
    "cement": 3.15,   # OPC
    "fine": 2.65,
    "coarse": 2.70,
    "rubber": 1.13,
    "water": 1.00,
    "sp": 1.08,      # superplasticizer density ~1.05-1.10
}

# High-performance concrete baseline (example)
BASE_BINDER_KG_PER_M3 = 500.0
BASE_WB = 0.30
BASE_SP_DOSAGE_PCT_BWOC = 1.0  # % by weight of cement
BASE_AIR_VOL_PCT = 2.0
FINE_TO_COARSE_VOL_SPLIT = (0.45, 0.55)  # volume split of total aggregate

# Fresh property base values for control
BASE_SLUMP_MM = 200.0  # use as "slump/flow" proxy; HPC may be measured as flow
BASE_AIR_CONTENT_PCT = 2.0
BASE_UNIT_WEIGHT_KG_M3 = 2400.0  # typical fresh concrete unit weight
BASE_FRESH_TEMP_C = 22.0

# Effect of rubber content (per 1% vol replacement of fine aggregate)
DELTA_SLUMP_PER_1PCT = -3.0   # mm
DELTA_AIR_PER_1PCT = +0.15    # %-points
DELTA_UNITWT_PER_1PCT = -14.0 # kg/m^3

# Extra penalty for coarser rubber on slump (mm per 1%)
COARSE_SIZE_EXTRA_SLUMP_DROP_PER_1PCT = -1.0

# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def bogue_from_xrf(xrf: Dict[str, float]) -> Dict[str, float]:
    """Compute Bogue composition (C3S, C2S, C3A, C4AF) from XRF oxides.
    Assumes wt% of SiO2, Al2O3, Fe2O3, CaO available.
    """
    SiO2 = xrf.get("SiO2", 0.0)
    Al2O3 = xrf.get("Al2O3", 0.0)
    Fe2O3 = xrf.get("Fe2O3", 0.0)
    CaO = xrf.get("CaO", 0.0)

    C3S = 4.071 * CaO - 7.602 * SiO2 - 1.429 * Fe2O3 - 6.718 * Al2O3
    C2S = 2.867 * SiO2 - 0.7544 * C3S
    C3A = 2.650 * Al2O3 - 1.692 * Fe2O3
    C4AF = 3.043 * Fe2O3

    # Clamp to non-negative
    C3S = max(0.0, C3S)
    C2S = max(0.0, C2S)
    C3A = max(0.0, C3A)
    C4AF = max(0.0, C4AF)

    total = C3S + C2S + C3A + C4AF
    if total > 100.0:
        # Normalize to 100 if exceeding (rare with realistic XRF)
        scale = 100.0 / total
        C3S *= scale
        C2S *= scale
        C3A *= scale
        C4AF *= scale

    return {"C3S": C3S, "C2S": C2S, "C3A": C3A, "C4AF": C4AF}


def generate_sieve_distribution_fine() -> List[Tuple[float, float]]:
    """Return list of (sieve_mm, percent_passing) for fine aggregate."""
    # Typical sand grading (well-graded)
    sieves = [9.5, 4.75, 2.36, 1.18, 0.600, 0.300, 0.150]
    base_pp = [100, 96, 80, 63, 42, 21, 5]
    # Add slight random noise
    pp = []
    prev = 100.0
    for b in base_pp:
        val = clamp(b + random.uniform(-3, 3), 0, prev)
        pp.append(val)
        prev = val
    return list(zip(sieves, pp))


def generate_sieve_distribution_coarse() -> List[Tuple[float, float]]:
    """Return list of (sieve_mm, percent_passing) for 20 mm nominal max coarse aggregate."""
    sieves = [37.5, 19.0, 9.5, 4.75]
    base_pp = [100, 85, 30, 0]
    pp = []
    prev = 100.0
    for b in base_pp:
        val = clamp(b + random.uniform(-5, 5), 0, prev)
        pp.append(val)
        prev = val
    return list(zip(sieves, pp))


def generate_psd_rubber(size_label: str) -> List[Tuple[float, float]]:
    """Crumb rubber PSD as percent passing across key sieves in the range."""
    if size_label == "1-4mm":
        sieves = [4.75, 2.36, 1.18, 0.600]
        base_pp = [100, 80, 30, 5]
    else:  # "4-8mm"
        sieves = [9.5, 4.75, 2.36]
        base_pp = [100, 60, 10]
    pp = []
    prev = 100.0
    for b in base_pp:
        val = clamp(b + random.uniform(-4, 4), 0, prev)
        pp.append(val)
        prev = val
    return list(zip(sieves, pp))


def logistic(x: float, x0: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


def generate_tga_curve(temp_start: int = 25, temp_end: int = 800, step: int = 5, noise: float = 0.15,
                       seed_offset: int = 0) -> List[Tuple[int, float, float]]:
    """Generate synthetic TGA mass% vs temperature and derivative.
    Returns rows: (T_C, mass_pct, d_mass_pct_dT)
    """
    rng = random.Random(SEED + 1000 + seed_offset)

    temperatures = list(range(temp_start, temp_end + 1, step))
    # Three-stage mass loss typical for rubber (volatile loss, pyrolysis, char oxidation)
    initial_mass = 100.0
    residue_target = rng.uniform(24.0, 30.0)  # silica, carbon black, ash, additives

    # Stage proportions
    loss1 = rng.uniform(2.0, 4.0)    # volatiles ~80-180C
    loss2 = rng.uniform(45.0, 55.0)  # pyrolysis ~200-400C
    loss3 = initial_mass - residue_target - loss1 - loss2
    loss3 = clamp(loss3, 12.0, 30.0)

    # Logistic transitions
    def stage(temp: float, onset: float, finish: float, amp: float) -> float:
        # Cumulative fraction transitioned by temp
        mid = (onset + finish) / 2.0
        k = 0.06
        frac = logistic(temp, mid, k)
        # Scale between onset and finish more tightly
        window = logistic(temp, finish, k) - logistic(temp, onset, k)
        # Use frac for cumulative mass loss shape (smooth S)
        return amp * frac

    masses = []
    prev_mass = initial_mass
    prev_temp = temperatures[0]
    prev_mass_for_deriv = initial_mass
    rows: List[Tuple[int, float, float]] = []

    for i, T in enumerate(temperatures):
        m1 = stage(T, 80, 200, loss1)
        m2 = stage(T, 200, 400, loss2)
        m3 = stage(T, 400, 600, loss3)
        mass = initial_mass - (m1 + m2 + m3)
        mass += rng.uniform(-noise, noise)
        mass = clamp(mass, residue_target - 1.0, initial_mass + 0.2)
        masses.append(mass)

    # Derivative (simple finite difference)
    for i, T in enumerate(temperatures):
        if i == 0:
            dmdT = 0.0
        else:
            dmdT = (masses[i] - masses[i - 1]) / (temperatures[i] - temperatures[i - 1])
        rows.append((T, round(masses[i], 3), round(dmdT, 5)))
    return rows


def gaussian(x: float, mu: float, sigma: float) -> float:
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def generate_ftir_spectrum(step: int = 4, noise: float = 0.004, seed_offset: int = 0) -> List[Tuple[int, float]]:
    """Generate synthetic FTIR absorbance spectrum (4000-400 cm^-1)."""
    rng = random.Random(SEED + 2000 + seed_offset)
    wns = list(range(4000, 399, -step))  # descending wavenumber

    # Peaks typical of rubber compounds and additives (arbitrary intensities)
    peaks = [
        (2920, 0.18, 35),  # CH2 asym stretch
        (2850, 0.14, 35),  # CH2 sym stretch
        (1660, 0.08, 30),  # C=C stretch
        (1600, 0.06, 25),  # aromatic C=C
        (1450, 0.10, 30),  # CH2 scissoring
        (1375, 0.08, 25),  # CH3 symmetric bend
        (1160, 0.06, 25),  # C-O/C-C stretch (additive)
        (1100, 0.12, 40),  # silica / sulfoxide region
        (700,  0.05, 20),  # aromatic C-H out-of-plane
        (520,  0.04, 20),  # S-S / C-S vibrations
    ]

    baseline = 0.02
    spectrum: List[Tuple[int, float]] = []
    for wn in wns:
        absorb = baseline + rng.uniform(-noise, noise)
        for mu, amp, sigma in peaks:
            absorb += amp * gaussian(wn, mu, sigma)
        absorb = clamp(absorb, 0.0, 1.5)
        spectrum.append((wn, round(absorb, 5)))
    return spectrum


def write_pgm(path: str, width: int, height: int, pixels: List[int]) -> None:
    """Write ASCII PGM (P2) image."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("P2\n")
        f.write(f"{width} {height}\n")
        f.write("255\n")
        line_len = 0
        for i, val in enumerate(pixels):
            s = str(val)
            f.write(s)
            if (i + 1) % width == 0:
                f.write("\n")
            else:
                f.write(" ")


def generate_sem_like_image(width: int, height: int, seed_offset: int = 0) -> List[int]:
    """Generate a simple SEM-like grayscale texture using smoothed noise and shading.
    Pure Python, no dependencies. Returns flat list of 0..255 ints.
    """
    rng = random.Random(SEED + 3000 + seed_offset)
    # Start with random noise
    grid = [[rng.random() for _ in range(width)] for _ in range(height)]

    # Apply a few smoothing passes (box blur)
    def blur(grid: List[List[float]]) -> List[List[float]]:
        out = [[0.0] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                s = 0.0
                cnt = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        yy = min(height - 1, max(0, y + dy))
                        xx = min(width - 1, max(0, x + dx))
                        s += grid[yy][xx]
                        cnt += 1
                out[y][x] = s / cnt
        return out

    passes = 3
    for _ in range(passes):
        grid = blur(grid)

    # Add directional shading to mimic topography
    for y in range(height):
        for x in range(width):
            shade = 0.15 * math.sin(2 * math.pi * x / (width / rng.uniform(2.5, 5.0)))
            grid[y][x] = clamp(grid[y][x] + shade, 0.0, 1.0)

    # Contrast stretch and quantize
    flat = [v for row in grid for v in row]
    lo = min(flat)
    hi = max(flat)
    span = max(1e-6, hi - lo)
    pixels = [int(255 * (v - lo) / span) for v in flat]

    # Sprinkle bright specks (to mimic inclusions)
    specks = int((width * height) * 0.002)
    for _ in range(specks):
        idx = rng.randrange(0, width * height)
        pixels[idx] = 255
    return pixels


# -----------------------------
# Data generation routines
# -----------------------------

def write_cement_data() -> None:
    ensure_dir(CEMENT_DIR)
    # XRF composition (wt%) summing ~100
    xrf = {
        "CaO": 63.5,
        "SiO2": 20.1,
        "Al2O3": 5.2,
        "Fe2O3": 3.1,
        "MgO": 2.0,
        "SO3": 2.5,
        "Na2O": 0.2,
        "K2O": 0.5,
        "TiO2": 0.2,
        "P2O5": 0.1,
        "Mn2O3": 0.05,
        "LOI": 2.75,
    }
    rows = [[oxide, wt] for oxide, wt in xrf.items()]
    write_csv(os.path.join(CEMENT_DIR, "xrf_composition.csv"), ["oxide", "wt_percent"], rows)

    bogue = bogue_from_xrf(xrf)
    write_csv(
        os.path.join(CEMENT_DIR, "bogue_composition.csv"),
        ["phase", "wt_percent"],
        [[k, round(v, 2)] for k, v in bogue.items()],
    )

    write_json(
        os.path.join(CEMENT_DIR, "cement_properties.json"),
        {
            "type": "OPC (Ordinary Portland Cement)",
            "specific_gravity": SPECIFIC_GRAVITY["cement"],
            "notes": "Bogue computed from XRF; values are plausible for OPC.",
        },
    )


def write_aggregate_data() -> None:
    # Fine aggregate
    ensure_dir(FINE_DIR)
    fine_sieve = generate_sieve_distribution_fine()
    write_csv(
        os.path.join(FINE_DIR, "sieve_analysis.csv"),
        ["sieve_mm", "percent_passing"],
        [[s, round(p, 1)] for s, p in fine_sieve],
    )
    write_json(
        os.path.join(FINE_DIR, "properties.json"),
        {
            "specific_gravity": SPECIFIC_GRAVITY["fine"],
            "water_absorption_pct": 1.2,
            "bulk_density_kg_m3_loose": 1600,
            "notes": "Natural river sand, well-graded.",
        },
    )

    # Coarse aggregate
    ensure_dir(COARSE_DIR)
    coarse_sieve = generate_sieve_distribution_coarse()
    write_csv(
        os.path.join(COARSE_DIR, "sieve_analysis.csv"),
        ["sieve_mm", "percent_passing"],
        [[s, round(p, 1)] for s, p in coarse_sieve],
    )
    write_json(
        os.path.join(COARSE_DIR, "properties.json"),
        {
            "specific_gravity": SPECIFIC_GRAVITY["coarse"],
            "water_absorption_pct": 0.6,
            "bulk_density_kg_m3_loose": 1500,
            "notes": "Crushed granite, nominal max size 20 mm.",
        },
    )


def write_rubber_data() -> None:
    ensure_dir(RUBBER_A_DIR)
    ensure_dir(RUBBER_B_DIR)

    for size_label, dir_path in [("1-4mm", RUBBER_A_DIR), ("4-8mm", RUBBER_B_DIR)]:
        psd = generate_psd_rubber(size_label)
        write_csv(
            os.path.join(dir_path, "particle_size_distribution.csv"),
            ["sieve_mm", "percent_passing"],
            [[s, round(p, 1)] for s, p in psd],
        )
        write_json(
            os.path.join(dir_path, "properties.json"),
            {
                "source": "End-of-life passenger tire tread",
                "size_range_mm": size_label,
                "specific_gravity": SPECIFIC_GRAVITY["rubber"],
                "mohs_hardness_approx": 0.5,
                "notes": "Pre-heating morphology captured in SEM images.",
            },
        )


def write_water_admixture_data() -> None:
    ensure_dir(WATER_DIR)
    ensure_dir(ADMIX_DIR)

    write_json(
        os.path.join(WATER_DIR, "water_quality.json"),
        {
            "ph": 7.2,
            "impurities_mg_per_L": {
                "chloride": 50,
                "sulfate": 80,
                "total_dissolved_solids": 250,
            },
            "notes": "Potable water meeting ASTM C1602 for mixing concrete.",
        },
    )

    write_json(
        os.path.join(ADMIX_DIR, "superplasticizer.json"),
        {
            "type": "PCE-based high-range water reducer",
            "typical_dosage_pct_bwoc": [0.75, 1.50],
            "solids_content_pct": 40,
            "specific_gravity": SPECIFIC_GRAVITY["sp"],
        },
    )


def write_characterization_data() -> None:
    ensure_dir(CHAR_DIR)
    ensure_dir(TGA_DIR)
    ensure_dir(FTIR_DIR)
    ensure_dir(SEM_DIR)

    # TGA replicates
    for rep in range(1, 4):
        rows = generate_tga_curve(seed_offset=rep)
        write_csv(
            os.path.join(TGA_DIR, f"tga_rubber_rep{rep}.csv"),
            ["temperature_C", "mass_pct", "d_mass_pct_per_C"],
            rows,
        )

    write_json(
        os.path.join(TGA_DIR, "tga_summary.json"),
        {
            "instrument": "Simulated TGA",
            "atmosphere": "Air",
            "heating_rate_C_per_min": 10,
            "notes": "Three-stage mass loss: volatiles, pyrolysis, char oxidation.",
        },
    )

    # FTIR replicates
    for rep in range(1, 4):
        spectrum = generate_ftir_spectrum(seed_offset=rep)
        write_csv(
            os.path.join(FTIR_DIR, f"ftir_rubber_rep{rep}.csv"),
            ["wavenumber_cm-1", "absorbance"],
            spectrum,
        )

    write_json(
        os.path.join(FTIR_DIR, "ftir_peak_assignments.json"),
        {
            "peaks": [
                {"wavenumber_cm-1": 2920, "assignment": "CH2 asym stretch"},
                {"wavenumber_cm-1": 2850, "assignment": "CH2 sym stretch"},
                {"wavenumber_cm-1": 1660, "assignment": "C=C stretch"},
                {"wavenumber_cm-1": 1600, "assignment": "Aromatic C=C"},
                {"wavenumber_cm-1": 1450, "assignment": "CH2 scissoring"},
                {"wavenumber_cm-1": 1375, "assignment": "CH3 symmetric bend"},
                {"wavenumber_cm-1": 1160, "assignment": "C-O/C-C stretch (additive)"},
                {"wavenumber_cm-1": 1100, "assignment": "Silica / sulfoxide region"},
                {"wavenumber_cm-1": 700, "assignment": "Aromatic C-H oop bend"},
                {"wavenumber_cm-1": 520, "assignment": "S-S / C-S vibrations"},
            ],
            "mode": "ATR",
            "units": "Absorbance",
        },
    )

    # SEM-like PGM images (4 per rubber size)
    for idx in range(1, 5):
        pixels_a = generate_sem_like_image(512, 512, seed_offset=idx)
        write_pgm(os.path.join(SEM_DIR, f"sem_rubber_1_4mm_preheat_{idx}.pgm"), 512, 512, pixels_a)
        pixels_b = generate_sem_like_image(512, 512, seed_offset=100 + idx)
        write_pgm(os.path.join(SEM_DIR, f"sem_rubber_4_8mm_preheat_{idx}.pgm"), 512, 512, pixels_b)


def mix_absolute_volume(mix: dict) -> dict:
    """Compute masses for a 1 m^3 batch via absolute volume method.
    Returns a dict with masses and volumes for each constituent.
    """
    cement_mass = BASE_BINDER_KG_PER_M3
    water_mass = BASE_BINDER_KG_PER_M3 * BASE_WB
    # Adjust SP dosage by rubber content to partially recover workability (but not fully)
    sp_dosage_pct = BASE_SP_DOSAGE_PCT_BWOC + 0.02 * mix["rubber_vol_pct"]
    sp_mass = cement_mass * sp_dosage_pct / 100.0

    V_cement = cement_mass / (SPECIFIC_GRAVITY["cement"] * 1000.0)
    V_water = water_mass / (SPECIFIC_GRAVITY["water"] * 1000.0)
    V_sp = sp_mass / (SPECIFIC_GRAVITY["sp"] * 1000.0)

    # Air content target rises with rubber content
    air_vol_pct = BASE_AIR_VOL_PCT + DELTA_AIR_PER_1PCT * mix["rubber_vol_pct"]
    V_air = air_vol_pct / 100.0

    V_agg_total = 1.0 - (V_cement + V_water + V_sp + V_air)
    V_fine_base = V_agg_total * FINE_TO_COARSE_VOL_SPLIT[0]
    V_coarse = V_agg_total * FINE_TO_COARSE_VOL_SPLIT[1]

    # Rubber volume replaces fraction of fine aggregate volume
    r = mix["rubber_vol_pct"] / 100.0
    V_rubber = V_fine_base * r
    V_fine = V_fine_base * (1.0 - r)

    # Masses from volumes and densities
    mass_fine = V_fine * (SPECIFIC_GRAVITY["fine"] * 1000.0)
    mass_coarse = V_coarse * (SPECIFIC_GRAVITY["coarse"] * 1000.0)
    mass_rubber = V_rubber * (SPECIFIC_GRAVITY["rubber"] * 1000.0)

    total_mass = cement_mass + water_mass + sp_mass + mass_fine + mass_coarse + mass_rubber

    return {
        "cement_kg": round(cement_mass, 2),
        "water_kg": round(water_mass, 2),
        "sp_kg": round(sp_mass, 2),
        "fine_kg": round(mass_fine, 1),
        "coarse_kg": round(mass_coarse, 1),
        "rubber_kg": round(mass_rubber, 1),
        "V_cement_m3": round(V_cement, 4),
        "V_water_m3": round(V_water, 4),
        "V_sp_m3": round(V_sp, 4),
        "V_fine_m3": round(V_fine, 4),
        "V_coarse_m3": round(V_coarse, 4),
        "V_rubber_m3": round(V_rubber, 4),
        "V_air_m3": round(V_air, 4),
        "batch_total_mass_kg": round(total_mass, 1),
    }


def write_mixes_and_fresh() -> None:
    ensure_dir(MIXES_DIR)
    ensure_dir(FRESH_DIR)

    # Build mix matrix
    mixes: List[dict] = []
    for rc in RUBBER_CONTENTS:
        if rc == 0:
            mixes.append({
                "mix_id": "CTRL",
                "rubber_vol_pct": 0,
                "rubber_size": "none",
            })
        else:
            for size_label, _ in RUBBER_SIZES[1:]:
                mixes.append({
                    "mix_id": f"R{rc:02d}-{size_label.replace('-', '').replace('mm','')}",
                    "rubber_vol_pct": rc,
                    "rubber_size": size_label,
                })

    # Write mix designs CSV
    mix_rows = []
    for mix in mixes:
        mv = mix_absolute_volume(mix)
        mix_rows.append([
            mix["mix_id"],
            mix["rubber_vol_pct"],
            mix["rubber_size"],
            BASE_WB,
            mv["cement_kg"],
            mv["water_kg"],
            mv["fine_kg"],
            mv["coarse_kg"],
            mv["rubber_kg"],
            round((mv["sp_kg"] / mv["cement_kg"]) * 100.0, 2),  # sp dosage % bwoc
            mv["V_air_m3"] * 100.0,
            mv["batch_total_mass_kg"],
        ])

    write_csv(
        os.path.join(MIXES_DIR, "mix_designs.csv"),
        [
            "mix_id",
            "rubber_content_vol_pct",
            "rubber_size",
            "w_b_ratio",
            "cement_kg_per_m3",
            "water_kg_per_m3",
            "fine_agg_kg_per_m3",
            "coarse_agg_kg_per_m3",
            "rubber_kg_per_m3",
            "sp_dosage_pct_bwoc",
            "target_air_content_pct",
            "batch_total_mass_kg",
        ],
        mix_rows,
    )

    # Fresh properties (3 batches per mix)
    fresh_rows = []
    timestamp0 = datetime(2025, 10, 1, 9, 0, 0)
    batch_counter = 1
    for mix in mixes:
        for rep in range(1, 4):
            rc = mix["rubber_vol_pct"]
            size = mix["rubber_size"]
            extra_size_penalty = 0.0
            if size == "4-8mm":
                extra_size_penalty = COARSE_SIZE_EXTRA_SLUMP_DROP_PER_1PCT

            # Slump model
            slump = (
                BASE_SLUMP_MM
                + DELTA_SLUMP_PER_1PCT * rc
                + extra_size_penalty * rc
                + random.uniform(-8, 8)
            )
            slump = clamp(slump, 40.0, 260.0)

            # Air content model
            air = (
                BASE_AIR_CONTENT_PCT
                + DELTA_AIR_PER_1PCT * rc
                + random.uniform(-0.25, 0.25)
            )
            air = clamp(air, 1.0, 7.5)

            # Unit weight model (start from base and add deterministic drop)
            unit_wt = (
                BASE_UNIT_WEIGHT_KG_M3
                + DELTA_UNITWT_PER_1PCT * rc
                + random.uniform(-15, 15)
            )
            unit_wt = clamp(unit_wt, 2000.0, 2450.0)

            # Fresh temperature (slightly variable room temp)
            temp_c = BASE_FRESH_TEMP_C + random.uniform(-1.0, 1.0)

            batch_id = f"B{batch_counter:03d}"
            batch_counter += 1
            ts = timestamp0 + timedelta(minutes=15 * (batch_counter - 1))

            fresh_rows.append([
                batch_id,
                mix["mix_id"],
                round(slump, 1),
                round(air, 2),
                round(unit_wt, 1),
                round(temp_c, 1),
                ts.isoformat(),
            ])

    write_csv(
        os.path.join(FRESH_DIR, "fresh_concrete_batches.csv"),
        [
            "batch_id",
            "mix_id",
            "slump_mm",
            "air_content_pct",
            "unit_weight_kg_per_m3",
            "fresh_temperature_C",
            "timestamp",
        ],
        fresh_rows,
    )


def write_readme() -> None:
    ensure_dir(DATASET_ROOT)
    content = f"""
# Phase 1 Baseline Dataset: High-Performance Rubberized Concrete

Scope: Material Characterization & Specimen Preparation
Project theme: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete.

Contents
- materials/
  - cement/: XRF, Bogue composition, properties
  - aggregates/
    - fine/: sieve analysis, properties
    - coarse/: sieve analysis, properties
  - crumb_rubber/ (1-4mm, 4-8mm): PSD, properties
  - water/: pH and impurities
  - chemical_admixture/: superplasticizer data
- characterization/
  - tga/: Synthetic TGA (3 reps) + summary
  - ftir/: Synthetic FTIR (3 reps) + peak assignments
  - sem_images/: SEM-like PGM images (4 per rubber size, pre-heating)
- mixes/: Mix designs (control + 5/10/15/20% rubber by volume of fine aggregate) for both size ranges
- fresh_properties/: Fresh data per batch (slump/flow, air, unit weight, temperature)

Notes
- Data are fabricated yet plausible; values reflect typical trends: increasing rubber -> lower unit weight and slump, higher air.
- Absolute volume method used to estimate constituent masses per m^3.
- TGA shows multi-stage mass loss (volatiles, pyrolysis, char oxidation). FTIR includes typical rubber/additive peaks.
- SEM images are algorithmic textures stored as ASCII PGM (P2), 512x512 px.

How to cite/use
- Synthetic dataset for prototyping. Not from physical experiments. CC-BY 4.0.

Reproducibility
- RNG seed: {SEED}
- Generator script: scripts/generate_phase1_dataset.py
"""
    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(content)


def make_zip() -> str:
    # Create zip at project root named dataset_phase1_baseline.zip
    archive_path = shutil.make_archive(ZIP_OUTPUT, "zip", DATASET_ROOT)
    return archive_path


def main() -> None:
    # Create directories
    for d in [
        DATASET_ROOT,
        MATERIALS_DIR,
        CEMENT_DIR,
        AGG_DIR,
        FINE_DIR,
        COARSE_DIR,
        RUBBER_DIR,
        RUBBER_A_DIR,
        RUBBER_B_DIR,
        WATER_DIR,
        ADMIX_DIR,
        CHAR_DIR,
        TGA_DIR,
        FTIR_DIR,
        SEM_DIR,
        MIXES_DIR,
        FRESH_DIR,
    ]:
        ensure_dir(d)

    # Generate all artifacts
    write_cement_data()
    write_aggregate_data()
    write_rubber_data()
    write_water_admixture_data()
    write_characterization_data()
    write_mixes_and_fresh()
    write_readme()

    # Zip
    archive_path = make_zip()

    print("Dataset generated.")
    print(f"Root: {os.path.abspath(DATASET_ROOT)}")
    print(f"ZIP:  {archive_path}")


if __name__ == "__main__":
    main()
