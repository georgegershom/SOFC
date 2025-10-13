#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from typing import Dict, List


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def generate_sandy_sample(rng: random.Random, sample_index: int) -> Dict[str, object]:
    # Sand and fines content
    sand_content_percent = clamp(rng.gauss(90.0, 6.0), 70.0, 100.0)
    fines_content_percent = clamp(100.0 - sand_content_percent + rng.gauss(0.0, 2.0), 0.0, 25.0)

    # Grain size distribution (mm); use lognormal-ish relationships
    median_grain_size_mm = clamp(rng.lognormvariate(math.log(0.35), 0.35), 0.08, 1.2)  # d50
    d10_over_d50 = clamp(rng.uniform(0.35, 0.75), 0.25, 0.9)
    d90_over_d50 = clamp(rng.uniform(1.2, 2.2), 1.05, 3.0)
    d10_mm = median_grain_size_mm * d10_over_d50
    d60_over_d50 = clamp(rng.uniform(1.1, 1.6), 1.0, 2.0)
    d60_mm = median_grain_size_mm * d60_over_d50
    d30_over_d50 = clamp(rng.uniform(0.8, 1.1), 0.6, 1.4)
    d30_mm = median_grain_size_mm * d30_over_d50
    d90_mm = median_grain_size_mm * d90_over_d50

    uniformity_coefficient = clamp(d60_mm / max(d10_mm, 1e-6), 1.1, 8.0)
    curvature_coefficient = clamp((d30_mm ** 2) / max(d10_mm * d60_mm, 1e-6), 0.5, 3.0)

    # Relative density and index void ratios
    relative_density_percent = clamp(rng.gauss(60.0, 18.0), 20.0, 100.0)
    e_min = clamp(rng.gauss(0.60, 0.06), 0.45, 0.75)
    e_max = clamp(rng.gauss(0.95, 0.07), 0.75, 1.20)
    # Ensure e_max > e_min
    if e_max <= e_min:
        e_max = e_min + 0.15
    void_ratio = clamp(e_min + (1.0 - relative_density_percent / 100.0) * (e_max - e_min), 0.40, 1.40)
    porosity = clamp(void_ratio / (1.0 + void_ratio), 0.25, 0.60)

    # Strength parameters (phi) correlated with density and fines
    friction_angle_deg = clamp(28.0 + 0.12 * relative_density_percent - 0.06 * fines_content_percent + rng.gauss(0.0, 1.2), 28.0, 44.0)
    critical_state_phi_deg = clamp(33.0 + rng.gauss(0.0, 1.0) - 0.03 * fines_content_percent, 29.0, 36.0)
    dilation_angle_deg = clamp(friction_angle_deg - critical_state_phi_deg, 0.0, 12.0)

    # State and loading for liquefaction context
    mean_effective_stress_kpa = clamp(rng.gauss(150.0, 60.0), 30.0, 400.0)
    cyclic_stress_ratio = clamp(rng.gauss(0.22, 0.08), 0.05, 0.60)

    # Empirical in-situ indices
    n1_60 = clamp(0.38 * relative_density_percent + rng.gauss(0.0, 3.0), 2.0, 45.0)
    qc1n = clamp(4.5 * n1_60 + rng.gauss(0.0, 15.0), 50.0, 400.0)

    # Pore pressure build-up proxy
    b_pt = clamp(rng.gauss(0.55, 0.15), 0.2, 0.95)
    ru_peak = clamp(0.55 * (cyclic_stress_ratio / 0.30) + 0.4 * (1.0 - relative_density_percent / 100.0) + rng.gauss(0.0, 0.05), 0.0, 0.98)

    liquefaction_triggered = int(ru_peak > 0.8 and n1_60 < 18.0)

    lateral_earth_pressure_k0 = clamp(1.0 - math.sin(math.radians(friction_angle_deg)), 0.35, 0.65)

    return {
        "sample_id": f"SAND_{sample_index:05d}",
        "sand_content_percent": round(sand_content_percent, 2),
        "fines_content_percent": round(fines_content_percent, 2),
        "d10_mm": round(d10_mm, 4),
        "d30_mm": round(d30_mm, 4),
        "d50_mm": round(median_grain_size_mm, 4),
        "d60_mm": round(d60_mm, 4),
        "d90_mm": round(d90_mm, 4),
        "cu": round(uniformity_coefficient, 3),
        "cc": round(curvature_coefficient, 3),
        "relative_density_percent": round(relative_density_percent, 1),
        "void_ratio": round(void_ratio, 3),
        "porosity": round(porosity, 3),
        "friction_angle_deg": round(friction_angle_deg, 2),
        "dilation_angle_deg": round(dilation_angle_deg, 2),
        "critical_state_phi_deg": round(critical_state_phi_deg, 2),
        "mean_effective_stress_kpa": round(mean_effective_stress_kpa, 1),
        "cyclic_stress_ratio": round(cyclic_stress_ratio, 3),
        "n1_60": round(n1_60, 1),
        "qc1n": round(qc1n, 1),
        "b_pt": round(b_pt, 3),
        "ru_peak": round(ru_peak, 3),
        "liquefaction_triggered": liquefaction_triggered,
        "k0": round(lateral_earth_pressure_k0, 3),
    }


def generate_dataset(num_rows: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    return [generate_sandy_sample(rng, i + 1) for i in range(num_rows)]


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic sandy soils dataset with liquefaction-relevant fields.")
    parser.add_argument("--n", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--out", type=str, default="data/synthetic/sandy_soils.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rows = generate_dataset(args.n, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} sandy soil samples to {args.out}")


if __name__ == "__main__":
    main()
