#!/usr/bin/env python3
import argparse
import csv
import os
import random
from typing import Dict, List

SCENARIOS = [
    {
        "case_type": "liquefaction_uplift",
        "description": "Underground structure uplift during seismic event due to excess pore pressure",
    },
    {
        "case_type": "clay_slip_surface",
        "description": "Progressive failure in sensitive clay leading to slip surface formation",
    },
]


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(x, b))


def gen_liquefaction_case(rng: random.Random, case_index: int) -> Dict[str, object]:
    magnitude_mw = clamp(rng.gauss(6.8, 0.5), 5.5, 8.2)
    peak_ground_acc_g = clamp(rng.gauss(0.32, 0.12), 0.1, 0.8)
    groundwater_depth_m = clamp(rng.gauss(2.0, 1.0), 0.0, 6.0)
    soil_relative_density = clamp(rng.gauss(55.0, 20.0), 15.0, 100.0)
    fines_content = clamp(rng.gauss(12.0, 6.0), 0.0, 30.0)
    pore_pressure_ratio_peak = clamp(0.5 + 0.4 * (peak_ground_acc_g / 0.4) * (1.0 - soil_relative_density / 100.0) + rng.gauss(0.0, 0.08), 0.05, 0.98)
    uplift_displacement_m = clamp(0.15 * pore_pressure_ratio_peak * (0.6 + 0.01 * fines_content) + rng.gauss(0.0, 0.02), 0.0, 0.6)

    structural_damage_index = clamp(0.25 + 0.9 * uplift_displacement_m + 0.7 * (pore_pressure_ratio_peak - 0.6) + rng.gauss(0.0, 0.08), 0.0, 1.0)

    return {
        "case_id": f"CASE_LQ_{case_index:04d}",
        "case_type": "liquefaction_uplift",
        "magnitude_mw": round(magnitude_mw, 2),
        "peak_ground_acc_g": round(peak_ground_acc_g, 3),
        "groundwater_depth_m": round(groundwater_depth_m, 2),
        "relative_density_percent": round(soil_relative_density, 1),
        "fines_content_percent": round(fines_content, 1),
        "pore_pressure_ratio_peak": round(pore_pressure_ratio_peak, 3),
        "uplift_displacement_m": round(uplift_displacement_m, 3),
        "structural_damage_index": round(structural_damage_index, 3),
        "notes": "Liquefaction-induced uplift scenario",
    }


def gen_clay_slip_case(rng: random.Random, case_index: int) -> Dict[str, object]:
    groundwater_level_m = clamp(rng.gauss(1.2, 0.8), -0.5, 4.0)
    rainfall_event_mm = clamp(rng.gauss(120.0, 60.0), 10.0, 350.0)
    plasticity_index = clamp(rng.gauss(28.0, 10.0), 7.0, 70.0)
    sensitivity = clamp(rng.gauss(10.0 + 0.12 * plasticity_index, 4.0), 1.5, 40.0)

    slope_angle_deg = clamp(rng.gauss(18.0, 5.0), 6.0, 35.0)
    depth_to_slip_surface_m = clamp(rng.gauss(8.0, 3.0), 2.0, 20.0)

    pore_pressure_ratio = clamp(0.15 + 0.004 * rainfall_event_mm + 0.001 * (1.0 - slope_angle_deg / 35.0) + rng.gauss(0.0, 0.05), 0.05, 0.95)
    factor_of_safety = clamp(1.6 - 0.9 * pore_pressure_ratio - 0.004 * plasticity_index + rng.gauss(0.0, 0.05), 0.5, 1.6)

    slip_surface_formed = int(factor_of_safety < 1.0)

    return {
        "case_id": f"CASE_CL_{case_index:04d}",
        "case_type": "clay_slip_surface",
        "groundwater_level_m": round(groundwater_level_m, 2),
        "rainfall_event_mm": round(rainfall_event_mm, 1),
        "plasticity_index": round(plasticity_index, 1),
        "sensitivity": round(sensitivity, 2),
        "slope_angle_deg": round(slope_angle_deg, 1),
        "depth_to_slip_surface_m": round(depth_to_slip_surface_m, 2),
        "pore_pressure_ratio": round(pore_pressure_ratio, 3),
        "factor_of_safety": round(factor_of_safety, 3),
        "slip_surface_formed": slip_surface_formed,
        "notes": "Sensitive clay progressive failure scenario",
    }


def generate_cases(n_liq: int, n_clay: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for i in range(n_liq):
        rows.append(gen_liquefaction_case(rng, i + 1))
    for j in range(n_clay):
        rows.append(gen_clay_slip_case(rng, j + 1))
    return rows


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Union of keys across heterogeneous cases
    all_fields = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                all_fields.append(k)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic case study data for liquefaction uplift and clay slip surfaces.")
    parser.add_argument("--n_liq", type=int, default=200)
    parser.add_argument("--n_clay", type=int, default=200)
    parser.add_argument("--out", type=str, default="data/synthetic/case_studies.csv")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rows = generate_cases(args.n_liq, args.n_clay, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} case study rows to {args.out}")


if __name__ == "__main__":
    main()
