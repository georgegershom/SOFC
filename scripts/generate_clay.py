#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
from typing import Dict, List


def clamp(x: float, a: float, b: float) -> float:
    return max(a, min(x, b))


def generate_clay_sample(rng: random.Random, sample_index: int) -> Dict[str, object]:
    # Atterberg limits
    liquid_limit = clamp(rng.gauss(55.0, 15.0), 25.0, 110.0)
    plastic_limit = clamp(rng.gauss(27.0, 7.0), 12.0, 55.0)
    if plastic_limit >= liquid_limit - 2.0:
        plastic_limit = max(12.0, liquid_limit - rng.uniform(2.0, 8.0))
    plasticity_index = clamp(liquid_limit - plastic_limit, 7.0, 80.0)

    # Preconsolidation stress and OCR
    vertical_effective_stress_kpa = clamp(rng.gauss(150.0, 70.0), 25.0, 500.0)
    preconsolidation_stress_kpa = clamp(vertical_effective_stress_kpa * rng.uniform(1.0, 3.5), 40.0, 1200.0)
    overconsolidation_ratio = clamp(preconsolidation_stress_kpa / max(vertical_effective_stress_kpa, 1e-3), 1.0, 6.0)

    # Water content and unit weight proxies
    natural_water_content_percent = clamp(rng.gauss(35.0 + 0.35 * plasticity_index, 8.0), 18.0, 120.0)

    # Undrained shear strength Su related to OCR and PI (e.g., SHANSEP-like)
    su_kpa = clamp(6.0 * (vertical_effective_stress_kpa ** 0.8) * (overconsolidation_ratio ** 0.8) / (100.0 + 0.5 * plasticity_index)
                   + rng.gauss(0.0, 8.0), 8.0, 400.0)

    # Sensitivity (ratio of intact to remolded Su)
    sensitivity = clamp(rng.gauss(5.0 + 0.06 * plasticity_index, 2.5), 1.2, 35.0)

    # Pore pressure ratio under undrained loading (proxy)
    pore_pressure_ratio = clamp(0.25 + 0.006 * plasticity_index - 0.03 * math.log10(max(overconsolidation_ratio, 1.0)) + rng.gauss(0.0, 0.05), 0.05, 0.95)

    # Mineralogy: simple categorical assignment skewed by PI
    if plasticity_index > 40.0:
        clay_mineral = "smectite-rich"
    elif plasticity_index > 20.0:
        clay_mineral = "illite-mixed"
    else:
        clay_mineral = "kaolinite-dominant"

    # Permeability proxy (very coarse correlation)
    log10_k_m_per_s = clamp(-10.0 + 0.02 * (30.0 - plasticity_index) + rng.gauss(0.0, 0.4), -12.0, -7.0)

    return {
        "sample_id": f"CLAY_{sample_index:05d}",
        "liquid_limit": round(liquid_limit, 1),
        "plastic_limit": round(plastic_limit, 1),
        "plasticity_index": round(plasticity_index, 1),
        "preconsolidation_stress_kpa": round(preconsolidation_stress_kpa, 1),
        "vertical_effective_stress_kpa": round(vertical_effective_stress_kpa, 1),
        "ocr": round(overconsolidation_ratio, 2),
        "natural_water_content_percent": round(natural_water_content_percent, 1),
        "su_kpa": round(su_kpa, 1),
        "sensitivity": round(sensitivity, 2),
        "pore_pressure_ratio": round(pore_pressure_ratio, 3),
        "clay_mineral": clay_mineral,
        "log10_k_m_per_s": round(log10_k_m_per_s, 3),
    }


def generate_dataset(n: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    return [generate_clay_sample(rng, i + 1) for i in range(n)]


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
    parser = argparse.ArgumentParser(description="Generate a synthetic clay soils dataset with plasticity, Su, OCR, sensitivity, and pore pressure ratio.")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, default="data/synthetic/clay_soils.csv")
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    rows = generate_dataset(args.n, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} clay soil samples to {args.out}")


if __name__ == "__main__":
    main()
