#!/usr/bin/env python3
"""
Generate synthetic geotechnical datasets for sandy and clay soils, including
- Basic soil properties (sandy and clay)
- Mechanical properties (liquefaction potential for sand; undrained strength and sensitivity for clay)
- Failure & case study data (liquefaction-induced uplift, slip surface formation)

Outputs:
- data/synthetic/sandy/properties.csv
- data/synthetic/sandy/liquefaction_tests.csv
- data/synthetic/sandy/liquefaction_timeseries.json
- data/synthetic/clay/properties.csv
- data/synthetic/clay/undrained_shear_profile.csv
- data/synthetic/case_studies/liquefaction_uplift_events.json
- data/synthetic/case_studies/slip_surface_events.json
- data/metadata/catalog.json
- data/metadata/variables.json

No external dependencies (stdlib only).
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import statistics
import uuid
from datetime import date, timedelta

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SAND_DIR = SYNTHETIC_DIR / "sandy"
CLAY_DIR = SYNTHETIC_DIR / "clay"
CASE_DIR = SYNTHETIC_DIR / "case_studies"
META_DIR = DATA_DIR / "metadata"


def ensure_dirs() -> None:
    for d in [DATA_DIR, SYNTHETIC_DIR, SAND_DIR, CLAY_DIR, CASE_DIR, META_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def clipped_gauss(mu: float, sigma: float, lo: float, hi: float) -> float:
    """Sample from a normal distribution and clip to bounds."""
    x = random.gauss(mu, sigma)
    return max(lo, min(hi, x))


def lognormal_from_gmean_gsd(gmean: float, gsd: float) -> float:
    """Sample a lognormal given geometric mean and geometric stdev."""
    # Convert geometric parameters to underlying normal mu, sigma
    sigma = math.log(gsd)
    mu = math.log(gmean)
    return random.lognormvariate(mu, sigma)


# ---------------------------- SANDY DATASETS ----------------------------- #

def generate_sandy_properties(n: int) -> list[dict]:
    records = []
    for i in range(n):
        sample_id = f"SAND-{i+1:04d}"
        sand_content = random.uniform(70.0, 100.0)
        fines_content = 100.0 - sand_content
        # Grain size metrics (mm). D50 geometric around 0.3 mm (medium sand)
        D50 = lognormal_from_gmean_gsd(0.3, 1.7)
        # Uniformity coefficient Cu between 1.5 and 9; curvature Cc ~ 0.8-1.5
        Cu = random.uniform(1.5, 9.0)
        Cc = clipped_gauss(1.0, 0.25, 0.4, 2.0)
        D10 = max(0.01, D50 / Cu)
        D60 = D50 * (Cu / (1.0 if Cc <= 0 else Cc))
        # Relative density Dr 0.2-0.85 typical in field
        Dr = random.uniform(0.2, 0.85)
        # Void ratio estimated from e_min/e_max envelope
        e_min = clipped_gauss(0.55, 0.07, 0.4, 0.7)
        e_max = clipped_gauss(0.9, 0.08, 0.7, 1.1)
        e = e_min + (1.0 - Dr) * (e_max - e_min)
        # Friction angle as a function of density and gradation
        phi = 28.0 + 12.0 * Dr + 2.0 * (Cu > 4.0) + clipped_gauss(0.0, 1.5, -3.0, 3.0)
        phi = max(28.0, min(44.0, phi))
        # SPT N1_60 roughly correlated with Dr
        N1_60 = max(2.0, 5.0 + 35.0 * Dr - 0.1 * fines_content + random.gauss(0, 2.0))
        # Vs30 (m/s) lightly correlated with density
        Vs30 = max(120.0, 100.0 + 250.0 * Dr + random.gauss(0, 20.0))
        records.append({
            "sample_id": sample_id,
            "sand_content_percent": round(sand_content, 2),
            "fines_content_percent": round(fines_content, 2),
            "D10_mm": round(D10, 4),
            "D50_mm": round(D50, 4),
            "D60_mm": round(D60, 4),
            "Cu": round(Cu, 3),
            "Cc": round(Cc, 3),
            "relative_density_Dr": round(Dr, 3),
            "void_ratio_e": round(e, 3),
            "friction_angle_phi_deg": round(phi, 2),
            "N1_60": round(N1_60, 1),
            "Vs30_m_per_s": round(Vs30, 1),
        })
    return records


def rd_reduction_factor(depth_m: float) -> float:
    # Simplified depth reduction factor for cyclic shear stress (Youd et al.)
    if depth_m <= 9.15:
        return 1.0 - 0.00765 * depth_m
    elif depth_m <= 23.0:
        return 1.174 - 0.0267 * depth_m
    else:
        return 0.744 - 0.008 * depth_m


def estimate_CRR7p5(N1_60: float, fines_content: float) -> float:
    # Simplified empirical CRR7.5 from SPT (approximate, for synthetic data only)
    # Increase CRR with lower fines content
    fines_factor = 1.0 - min(0.5, fines_content / 200.0)
    base = 0.065 * (max(1.0, N1_60) ** 0.8)
    return max(0.03, min(0.35, base * fines_factor))


def generate_sandy_liquefaction_tests(sand_records: list[dict], n_tests: int) -> list[dict]:
    records = []
    for i in range(n_tests):
        test_id = f"LIQ-{i+1:04d}"
        sample = random.choice(sand_records)
        depth = random.uniform(3.0, 20.0)
        gamma_sat = random.uniform(18.5, 20.0)  # kN/m3
        gamma_w = 9.81  # kN/m3
        wt_depth = random.uniform(0.5, 3.0)
        sigma_v0 = gamma_sat * depth  # total stress (kPa approximated)
        u0 = gamma_w * max(0.0, depth - wt_depth)
        sigma_v0_eff = max(10.0, sigma_v0 - u0)
        Mw = clipped_gauss(7.2, 0.4, 6.0, 8.0)
        PGA = random.uniform(0.1, 0.45)  # g
        rd = rd_reduction_factor(depth)
        CSR = 0.65 * PGA * (sigma_v0 / sigma_v0_eff) * rd
        CRR7p5 = estimate_CRR7p5(sample["N1_60"], sample["fines_content_percent"])
        # Magnitude scaling factor (MSF) ~ 10^(2.24/Mw^2.56) approx; bound 0.6-1.8
        MSF = max(0.6, min(1.8, 10.0 ** (2.24 / (Mw ** 2.56))))
        CRR = CRR7p5 / MSF
        FS = CRR / CSR if CSR > 0 else 3.0
        ru_peak = max(0.1, min(1.0, 1.15 - 0.5 * FS + random.gauss(0, 0.05)))
        records.append({
            "test_id": test_id,
            "sample_id": sample["sample_id"],
            "depth_m": round(depth, 2),
            "water_table_depth_m": round(wt_depth, 2),
            "gamma_sat_kN_per_m3": round(gamma_sat, 2),
            "sigma_v0_kPa": round(sigma_v0, 1),
            "sigma_v0_eff_kPa": round(sigma_v0_eff, 1),
            "Mw": round(Mw, 2),
            "PGA_g": round(PGA, 3),
            "rd": round(rd, 3),
            "CSR": round(CSR, 3),
            "CRR7p5": round(CRR7p5, 3),
            "MSF": round(MSF, 3),
            "CRR": round(CRR, 3),
            "FS_liq": round(FS, 2),
            "ru_peak": round(ru_peak, 3),
            "fines_content_percent": sample["fines_content_percent"],
        })
    return records


def generate_liquefaction_timeseries(test_records: list[dict], n_events: int = 12) -> list[dict]:
    events = []
    chosen = random.sample(test_records, min(n_events, len(test_records)))
    for rec in chosen:
        event_id = f"EV-LIQ-{uuid.uuid4().hex[:8]}"
        duration = random.uniform(20.0, 90.0)  # seconds
        dt = 0.5
        npts = int(duration / dt) + 1
        times = [round(i * dt, 2) for i in range(npts)]
        FS = rec["FS_liq"]
        # Target ru plateau
        ru_target = min(1.0, max(0.25, 1.05 - 0.45 * FS + random.gauss(0, 0.03)))
        k = random.uniform(0.05, 0.15)  # growth rate
        ru = [round(ru_target * (1.0 - math.exp(-k * t)), 3) for t in times]
        # Add small oscillations to mimic shakes
        for j in range(len(ru)):
            ru[j] = round(
                max(
                    0.0,
                    min(1.1, ru[j] + 0.03 * math.sin(0.8 * j) + random.gauss(0, 0.005)),
                ),
                3,
            )
        uplift_mm = max(0.0, (ru_target - 0.7) * 80.0 + random.gauss(0, 5.0))
        events.append({
            "event_id": event_id,
            "test_id": rec["test_id"],
            "sample_id": rec["sample_id"],
            "time_s": times,
            "ru": [round(x, 3) for x in ru],
            "estimated_uplift_mm": round(uplift_mm, 1),
            "depth_m": rec["depth_m"],
            "PGA_g": rec["PGA_g"],
            "Mw": rec["Mw"],
        })
    return events


# ------------------------------ CLAY DATASETS ---------------------------- #

def random_clay_mineralogy(LL: float) -> dict:
    # Higher LL tends to correlate with higher smectite fraction
    smectite = clipped_gauss(0.4 if LL > 60 else 0.2, 0.1, 0.02, 0.7)
    kaolinite = clipped_gauss(0.25 if LL < 50 else 0.15, 0.08, 0.02, 0.5)
    illite = clipped_gauss(0.25, 0.08, 0.02, 0.5)
    rest = max(0.0, 1.0 - (smectite + kaolinite + illite))
    # Normalize to 1.0 if rounding drift
    total = smectite + kaolinite + illite + rest
    return {
        "smectite_percent": round(100.0 * smectite / total, 1),
        "kaolinite_percent": round(100.0 * kaolinite / total, 1),
        "illite_percent": round(100.0 * illite / total, 1),
        "mixed_layer_other_percent": round(100.0 * rest / total, 1),
    }


def generate_clay_properties(n: int) -> list[dict]:
    records = []
    for i in range(n):
        sample_id = f"CLAY-{i+1:04d}"
        LL = clipped_gauss(50.0, 15.0, 25.0, 110.0)
        PI = clipped_gauss(25.0, 10.0, 7.0, 60.0)
        PL = max(5.0, LL - PI)
        w_n = clipped_gauss(PL + 5.0, 8.0, 10.0, LL + 10.0)  # natural water content
        sigma_p = random.lognormvariate(math.log(150.0), 0.5)  # kPa
        OCR = random.uniform(1.0, 5.0)
        # Undrained shear strengths (intact/remolded) linked to PI and OCR
        su_intact = max(8.0, 10.0 + 0.6 * PI + 8.0 * (OCR - 1.0) + random.gauss(0, 4.0))
        sensitivity = random.uniform(2.0, 10.0)
        su_remolded = max(2.0, su_intact / sensitivity)
        # Pore pressure ratio under undrained shearing
        Ru = clipped_gauss(0.4 + 0.15 * (sensitivity / 10.0), 0.1, 0.1, 0.95)
        mino = random_clay_mineralogy(LL)
        records.append({
            "sample_id": sample_id,
            "liquid_limit_LL": round(LL, 1),
            "plastic_limit_PL": round(PL, 1),
            "plasticity_index_PI": round(PI, 1),
            "water_content_percent": round(w_n, 1),
            "preconsolidation_stress_kPa": round(sigma_p, 1),
            "OCR": round(OCR, 2),
            "undrained_shear_strength_su_intact_kPa": round(su_intact, 1),
            "undrained_shear_strength_su_remolded_kPa": round(su_remolded, 1),
            "sensitivity_St": round(sensitivity, 2),
            "pore_pressure_ratio_Ru": round(Ru, 2),
            **mino,
        })
    return records


def generate_undrained_shear_profiles(clay_records: list[dict], n_boreholes: int = 24, max_depth: float = 20.0, dz: float = 1.0) -> list[dict]:
    profiles = []
    chosen = random.sample(clay_records, min(n_boreholes, len(clay_records)))
    for i, sample in enumerate(chosen):
        borehole_id = f"BH-{i+1:03d}"
        base_su = sample["undrained_shear_strength_su_intact_kPa"]
        OCR = sample["OCR"]
        for d in [round(x, 2) for x in [j * dz for j in range(0, int(max_depth / dz) + 1)]]:
            gradient = 1.0 + 0.8 * (OCR - 1.0)  # kPa/m
            su = max(5.0, base_su + gradient * d + random.gauss(0, 1.5))
            profiles.append({
                "borehole_id": borehole_id,
                "sample_id": sample["sample_id"],
                "depth_m": round(d, 2),
                "su_kPa": round(su, 1),
            })
    return profiles


# ------------------------------ CASE STUDIES ----------------------------- #

def generate_liquefaction_uplift_events(test_records: list[dict], n_events: int = 10) -> list[dict]:
    events = []
    start_date = date(1990, 1, 1)
    for _ in range(n_events):
        rec = random.choice(test_records)
        days = random.randint(0, 35 * 365)
        ev_date = (start_date + timedelta(days=days)).isoformat()
        FS = rec["FS_liq"]
        uplift_mm = max(0.0, (1.0 - min(1.0, FS)) * random.uniform(20.0, 200.0))
        gwt = rec["water_table_depth_m"]
        events.append({
            "case_id": f"CASE-LIQ-{uuid.uuid4().hex[:10]}",
            "date": ev_date,
            "structure_type": random.choice(["utility_tunnel", "metro_station", "box_culvert", "underground_parking"]),
            "depth_to_structure_top_m": round(random.uniform(2.0, 10.0), 1),
            "water_table_depth_m": gwt,
            "ru_peak": rec["ru_peak"],
            "uplift_displacement_mm": round(uplift_mm, 1),
            "FS_liq": rec["FS_liq"],
            "Mw": rec["Mw"],
            "PGA_g": rec["PGA_g"],
            "notes": "Synthetic event derived from liquefaction test parameters.",
        })
    return events


def generate_slip_surface_events(clay_records: list[dict], profiles: list[dict], n_events: int = 10) -> list[dict]:
    # Group profiles by borehole
    by_bh: dict[str, list[dict]] = {}
    for p in profiles:
        by_bh.setdefault(p["borehole_id"], []).append(p)
    for bh, lst in by_bh.items():
        lst.sort(key=lambda x: x["depth_m"])  # ensure depth ordered

    events = []
    start_date = date(1990, 1, 1)
    boreholes = list(by_bh.keys())
    for _ in range(n_events):
        bh = random.choice(boreholes)
        prof = by_bh[bh]
        sample_id = prof[0]["sample_id"]
        clay = next((c for c in clay_records if c["sample_id"] == sample_id), None)
        if not clay:
            continue
        LL = clay["liquid_limit_LL"]
        St = clay["sensitivity_St"]
        # Choose groundwater and slope parameters
        slope_angle = random.uniform(10.0, 35.0)
        gwt = random.uniform(0.5, 4.0)
        rainfall = random.uniform(10.0, 70.0)  # mm/hr event intensity
        duration_h = random.uniform(2.0, 24.0)
        # Slip depth more likely shallow for high sensitivity
        slip_depth = clipped_gauss(5.0 - 0.3 * (St - 2.0), 1.5, 1.0, 12.0)
        # Factor of safety proxy: lower with higher LL, higher St, steeper slope
        FoS = max(0.6, min(1.8, 1.6 - 0.005 * (LL - 40.0) - 0.01 * (slope_angle - 15.0) - 0.02 * (rainfall - 20.0)))
        occurred = FoS < 1.0
        days = random.randint(0, 35 * 365)
        ev_date = (start_date + timedelta(days=days)).isoformat()
        events.append({
            "case_id": f"CASE-SLIP-{uuid.uuid4().hex[:10]}",
            "date": ev_date,
            "site_borehole_id": bh,
            "clay_sample_id": sample_id,
            "slope_angle_deg": round(slope_angle, 1),
            "groundwater_table_depth_m": round(gwt, 2),
            "rainfall_intensity_mm_per_hr": round(rainfall, 1),
            "rainfall_duration_hr": round(duration_h, 1),
            "slip_surface_depth_m": round(slip_depth, 2),
            "factor_of_safety": round(FoS, 2),
            "failure_occurred": occurred,
            "dominant_clay_mineral": max((k for k in ["smectite_percent", "kaolinite_percent", "illite_percent", "mixed_layer_other_percent"]), key=lambda k: clay[k]),
            "notes": "Synthetic event derived from clay properties and profile.",
        })
    return events


# ------------------------------- I/O HELPERS ----------------------------- #

def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def build_metadata_catalog(outputs: dict[str, Path]) -> tuple[dict, dict]:
    catalog = {
        "generated_by": "generate_geotech_datasets.py",
        "description": "Synthetic geotechnical datasets for sandy/clay soils and failure case studies",
        "outputs": {name: str(path.relative_to(ROOT)) for name, path in outputs.items()},
    }
    variables = {
        "sandy_properties": {
            "columns": {
                "sample_id": "Unique sand sample identifier",
                "sand_content_percent": "% sand by mass",
                "fines_content_percent": "% passing #200 (silt+clay)",
                "D10_mm": "Effective size",
                "D50_mm": "Median grain size",
                "D60_mm": "Grain size at 60% passing",
                "Cu": "Uniformity coefficient D60/D10",
                "Cc": "Coefficient of curvature",
                "relative_density_Dr": "Relative density (0-1)",
                "void_ratio_e": "Void ratio",
                "friction_angle_phi_deg": "Peak friction angle (deg)",
                "N1_60": "SPT N corrected for overburden & energy",
                "Vs30_m_per_s": "Shear-wave velocity (m/s) over 30 m",
            }
        },
        "sandy_liquefaction_tests": {
            "columns": {
                "test_id": "Test identifier",
                "sample_id": "Link to sand sample",
                "depth_m": "Depth of layer (m)",
                "water_table_depth_m": "Water table depth (m)",
                "gamma_sat_kN_per_m3": "Saturated unit weight",
                "sigma_v0_kPa": "Vertical total stress",
                "sigma_v0_eff_kPa": "Vertical effective stress",
                "Mw": "Earthquake magnitude",
                "PGA_g": "Peak ground acceleration (g)",
                "rd": "Depth reduction factor",
                "CSR": "Cyclic stress ratio",
                "CRR7p5": "Cyclic resistance ratio normalized to M=7.5",
                "MSF": "Magnitude scaling factor",
                "CRR": "Cyclic resistance ratio",
                "FS_liq": "Factor of safety against liquefaction",
                "ru_peak": "Peak pore pressure ratio",
                "fines_content_percent": "% fines in the layer",
            }
        },
        "liquefaction_timeseries": {
            "fields": {
                "event_id": "Event identifier",
                "test_id": "Related liquefaction test",
                "sample_id": "Related sample",
                "time_s": "Time vector (s)",
                "ru": "Excess pore pressure ratio time series",
                "estimated_uplift_mm": "Estimated uplift displacement (mm)",
                "depth_m": "Depth of instrument (m)",
                "PGA_g": "Excitation PGA (g)",
                "Mw": "Magnitude",
            }
        },
        "clay_properties": {
            "columns": {
                "sample_id": "Unique clay sample identifier",
                "liquid_limit_LL": "Liquid limit (%)",
                "plastic_limit_PL": "Plastic limit (%)",
                "plasticity_index_PI": "PI = LL - PL",
                "water_content_percent": "Natural water content (%)",
                "preconsolidation_stress_kPa": "Sigma'_p (kPa)",
                "OCR": "Overconsolidation ratio",
                "undrained_shear_strength_su_intact_kPa": "Intact su (kPa)",
                "undrained_shear_strength_su_remolded_kPa": "Remolded su (kPa)",
                "sensitivity_St": "Intact/remolded strength ratio",
                "pore_pressure_ratio_Ru": "Pore pressure ratio under undrained shearing",
                "smectite_percent": "% smectite by mass",
                "kaolinite_percent": "% kaolinite by mass",
                "illite_percent": "% illite by mass",
                "mixed_layer_other_percent": "% other clays/mixed-layer",
            }
        },
        "clay_undrained_profiles": {
            "columns": {
                "borehole_id": "Borehole identifier",
                "sample_id": "Linked clay sample",
                "depth_m": "Depth (m)",
                "su_kPa": "Undrained shear strength (kPa)",
            }
        },
        "case_liquefaction_uplift": {
            "fields": {
                "case_id": "Case identifier",
                "date": "Event date (ISO)",
                "structure_type": "Type of underground structure",
                "depth_to_structure_top_m": "Cover depth to structure top (m)",
                "water_table_depth_m": "Water table depth (m)",
                "ru_peak": "Peak pore pressure ratio",
                "uplift_displacement_mm": "Observed uplift (mm)",
                "FS_liq": "Factor of safety (from test)",
                "Mw": "Magnitude",
                "PGA_g": "PGA (g)",
                "notes": "Free text notes",
            }
        },
        "case_slip_surface": {
            "fields": {
                "case_id": "Case identifier",
                "date": "Event date (ISO)",
                "site_borehole_id": "Borehole used to characterize site",
                "clay_sample_id": "Linked sample",
                "slope_angle_deg": "Slope angle (deg)",
                "groundwater_table_depth_m": "GWT depth (m)",
                "rainfall_intensity_mm_per_hr": "Rainfall intensity",
                "rainfall_duration_hr": "Rainfall duration (hr)",
                "slip_surface_depth_m": "Depth of slip surface (m)",
                "factor_of_safety": "FoS for slope stability",
                "failure_occurred": "Boolean",
                "dominant_clay_mineral": "Mineral phase with highest percentage",
                "notes": "Free text notes",
            }
        },
    }
    return catalog, variables


# ---------------------------------- MAIN --------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic geotechnical datasets")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--n_sand", type=int, default=500, help="Number of sandy property samples")
    parser.add_argument("--n_clay", type=int, default=500, help="Number of clay property samples")
    parser.add_argument("--n_liq_tests", type=int, default=300, help="Number of liquefaction tests")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dirs()

    # Sandy
    sandy_props = generate_sandy_properties(args.n_sand)
    sandy_props_path = SAND_DIR / "properties.csv"
    write_csv(
        sandy_props_path,
        [
            "sample_id",
            "sand_content_percent",
            "fines_content_percent",
            "D10_mm",
            "D50_mm",
            "D60_mm",
            "Cu",
            "Cc",
            "relative_density_Dr",
            "void_ratio_e",
            "friction_angle_phi_deg",
            "N1_60",
            "Vs30_m_per_s",
        ],
        sandy_props,
    )

    sandy_liq = generate_sandy_liquefaction_tests(sandy_props, args.n_liq_tests)
    sandy_liq_path = SAND_DIR / "liquefaction_tests.csv"
    write_csv(
        sandy_liq_path,
        [
            "test_id",
            "sample_id",
            "depth_m",
            "water_table_depth_m",
            "gamma_sat_kN_per_m3",
            "sigma_v0_kPa",
            "sigma_v0_eff_kPa",
            "Mw",
            "PGA_g",
            "rd",
            "CSR",
            "CRR7p5",
            "MSF",
            "CRR",
            "FS_liq",
            "ru_peak",
            "fines_content_percent",
        ],
        sandy_liq,
    )

    liq_ts = generate_liquefaction_timeseries(sandy_liq, n_events=12)
    liq_ts_path = SAND_DIR / "liquefaction_timeseries.json"
    write_json(liq_ts_path, liq_ts)

    # Clay
    clay_props = generate_clay_properties(args.n_clay)
    clay_props_path = CLAY_DIR / "properties.csv"
    write_csv(
        clay_props_path,
        [
            "sample_id",
            "liquid_limit_LL",
            "plastic_limit_PL",
            "plasticity_index_PI",
            "water_content_percent",
            "preconsolidation_stress_kPa",
            "OCR",
            "undrained_shear_strength_su_intact_kPa",
            "undrained_shear_strength_su_remolded_kPa",
            "sensitivity_St",
            "pore_pressure_ratio_Ru",
            "smectite_percent",
            "kaolinite_percent",
            "illite_percent",
            "mixed_layer_other_percent",
        ],
        clay_props,
    )

    clay_profiles = generate_undrained_shear_profiles(clay_props, n_boreholes=24, max_depth=20.0, dz=1.0)
    clay_profiles_path = CLAY_DIR / "undrained_shear_profile.csv"
    write_csv(
        clay_profiles_path,
        ["borehole_id", "sample_id", "depth_m", "su_kPa"],
        clay_profiles,
    )

    # Case studies
    liq_cases = generate_liquefaction_uplift_events(sandy_liq, n_events=16)
    liq_cases_path = CASE_DIR / "liquefaction_uplift_events.json"
    write_json(liq_cases_path, liq_cases)

    slip_cases = generate_slip_surface_events(clay_props, clay_profiles, n_events=16)
    slip_cases_path = CASE_DIR / "slip_surface_events.json"
    write_json(slip_cases_path, slip_cases)

    # Metadata
    catalog, variables = build_metadata_catalog({
        "sandy_properties": sandy_props_path,
        "sandy_liquefaction_tests": sandy_liq_path,
        "liquefaction_timeseries": liq_ts_path,
        "clay_properties": clay_props_path,
        "clay_undrained_profiles": clay_profiles_path,
        "case_liquefaction_uplift": liq_cases_path,
        "case_slip_surface": slip_cases_path,
    })
    write_json(META_DIR / "catalog.json", catalog)
    write_json(META_DIR / "variables.json", variables)

    print("Synthetic datasets generated.")
    print(json.dumps(catalog["outputs"], indent=2))


if __name__ == "__main__":
    main()
