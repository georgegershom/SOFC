#!/usr/bin/env python3

"""
Synthetic Sintering Process → Microstructure Dataset Generator

This script fabricates a dataset linking sintering process parameters
(temperature profile, pressure, atmosphere, green body characteristics)
to resulting microstructure metrics (grain size stats, porosity, grain
boundary characteristics, relative density).

The relationships are intentionally simplified yet physically plausible:
- Higher peak temperature and longer hold generally increase densification
  (higher relative density → lower porosity) and promote grain growth.
- Applied pressure assists densification (e.g., hot pressing), often reducing
  final porosity and modestly limiting exaggerated grain growth.
- Atmosphere influences kinetics (e.g., reducing/vacuum may help densification
  relative to oxidizing air), reflected via small multipliers.
- Faster ramp can modestly reduce densification uniformity.
- Faster cooling tends to preserve finer grains.

Outputs include both central tendency and distribution proxies (e.g.,
lognormal sigma for pore sizes). Random noise is added to emulate
experimental variability.

No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from typing import Dict, List, Tuple


Atmosphere = str


def clamp(value: float, min_value: float, max_value: float) -> float:
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def choose_atmosphere(rng: random.Random) -> Atmosphere:
    # Weighted choice typical of lab availability and common practice
    choices: List[Tuple[Atmosphere, float]] = [
        ("air", 0.35),
        ("argon", 0.20),
        ("nitrogen", 0.20),
        ("vacuum", 0.15),
        ("hydrogen", 0.10),
    ]
    u = rng.random()
    acc = 0.0
    for label, weight in choices:
        acc += weight
        if u <= acc:
            return label
    return choices[-1][0]


def sample_inputs(sample_id: int, rng: random.Random) -> Dict[str, object]:
    # Temperature profile
    ramp_up_rate_c_per_min = rng.uniform(2.0, 20.0)
    peak_temperature_c = rng.uniform(950.0, 1450.0)
    hold_time_min = rng.triangular(30.0, 120.0, 240.0)  # favor 1–2 h
    cool_rate_c_per_min = rng.uniform(2.0, 25.0)

    # Applied pressure (MPa), often zero unless hot pressing/HIP
    if rng.random() < 0.6:
        applied_pressure_mpa = 0.0
    else:
        applied_pressure_mpa = rng.uniform(5.0, 50.0)

    atmosphere = choose_atmosphere(rng)

    # Green body characteristics
    initial_relative_density = rng.uniform(0.50, 0.65)
    # Lognormal pore diameter (µm), median around ~0.5–0.8 µm typical for many compacted powders
    # Python's lognormvariate uses underlying normal(mu, sigma); median = exp(mu)
    initial_median_pore_diameter_um = clamp(rng.lognormvariate(-0.5, 0.4), 0.10, 3.0)
    initial_pore_size_lognormal_sigma = rng.uniform(0.25, 0.55)

    return {
        "sample_id": sample_id,
        "ramp_up_rate_c_per_min": round(ramp_up_rate_c_per_min, 3),
        "peak_temperature_c": round(peak_temperature_c, 2),
        "hold_time_min": round(hold_time_min, 2),
        "cool_rate_c_per_min": round(cool_rate_c_per_min, 3),
        "applied_pressure_mpa": round(applied_pressure_mpa, 3),
        "atmosphere": atmosphere,
        "initial_relative_density": round(initial_relative_density, 4),
        "initial_median_pore_diameter_um": round(initial_median_pore_diameter_um, 4),
        "initial_pore_size_lognormal_sigma": round(initial_pore_size_lognormal_sigma, 4),
    }


def compute_microstructure(inputs: Dict[str, object], rng: random.Random) -> Dict[str, float]:
    # Unpack
    peak_temperature_c = float(inputs["peak_temperature_c"])  # type: ignore[arg-type]
    hold_time_min = float(inputs["hold_time_min"])  # type: ignore[arg-type]
    ramp_up_rate_c_per_min = float(inputs["ramp_up_rate_c_per_min"])  # type: ignore[arg-type]
    cool_rate_c_per_min = float(inputs["cool_rate_c_per_min"])  # type: ignore[arg-type]
    applied_pressure_mpa = float(inputs["applied_pressure_mpa"])  # type: ignore[arg-type]
    atmosphere = str(inputs["atmosphere"])  # type: ignore[arg-type]
    initial_relative_density = float(inputs["initial_relative_density"])  # type: ignore[arg-type]
    initial_median_pore_diameter_um = float(inputs["initial_median_pore_diameter_um"])  # type: ignore[arg-type]
    initial_pore_size_lognormal_sigma = float(inputs["initial_pore_size_lognormal_sigma"])  # type: ignore[arg-type]

    # Normalized process features
    # Peak temperature scaled roughly over common ceramic sintering range
    t_norm = clamp((peak_temperature_c - 900.0) / 600.0, 0.0, 1.2)
    hold_hours = hold_time_min / 60.0

    # Pressure assistance multiplier (modest effect)
    pressure_effect = 1.0 + 0.006 * clamp(applied_pressure_mpa, 0.0, 60.0)

    # Atmosphere multiplier: reducing/vacuum slightly assist, air slightly hinders
    atmosphere_effects = {
        "hydrogen": 1.06,
        "vacuum": 1.04,
        "argon": 1.01,
        "nitrogen": 1.00,
        "air": 0.97,
    }
    atmosphere_mult = atmosphere_effects.get(atmosphere, 1.0)

    # Faster ramp may modestly reduce densification uniformity
    ramp_penalty = 0.015 * max(0.0, (ramp_up_rate_c_per_min - 10.0) / 10.0)
    ramp_factor = clamp(1.0 - ramp_penalty, 0.9, 1.02)

    # Dimensionless sintering "index"
    sintering_index = (1.6 * t_norm + 0.5 * hold_hours) * pressure_effect * atmosphere_mult * ramp_factor

    # Relative density outcome
    base_improvement = 0.25
    densification_delta = 0.40 * math.tanh(0.9 * sintering_index)
    density_noise = rng.gauss(0.0, 0.005)
    relative_density = clamp(initial_relative_density + base_improvement + densification_delta + density_noise, 0.60, 0.999)

    porosity_percent = 100.0 * (1.0 - relative_density)

    # Grain size growth model
    base_grain_um = 0.8 + 0.7 * initial_median_pore_diameter_um
    cool_norm = clamp((cool_rate_c_per_min - 2.0) / 20.0, 0.0, 1.0)
    growth_factor = (0.6 * t_norm + 0.2 * hold_hours) * (1.0 - 0.25 * cool_norm)
    pressure_limit_factor = 1.0 - 0.10 * clamp(applied_pressure_mpa / 50.0, 0.0, 1.0)
    grain_size_mean_um = base_grain_um * (1.0 + growth_factor) * pressure_limit_factor
    # Add multiplicative lognormal noise (~10% CV)
    noise_multiplier = rng.lognormvariate(-0.5 * 0.10 * 0.10, 0.10)
    grain_size_mean_um *= noise_multiplier
    grain_size_mean_um = clamp(grain_size_mean_um, 0.15, 50.0)
    grain_size_std_um = clamp(0.25 * grain_size_mean_um * rng.uniform(0.8, 1.25), 0.02, 20.0)

    # Pore size evolution
    pore_shrink = 0.50 * math.tanh(1.2 * sintering_index)
    pore_median_diameter_um = max(0.03, initial_median_pore_diameter_um * (1.0 - pore_shrink))
    pore_median_diameter_um *= rng.lognormvariate(-0.5 * 0.08 * 0.08, 0.08)
    pore_median_diameter_um = clamp(pore_median_diameter_um, 0.03, 3.0)
    pore_size_lognormal_sigma = clamp(
        initial_pore_size_lognormal_sigma - 0.10 * math.tanh(sintering_index) + rng.gauss(0.0, 0.02),
        0.15,
        0.70,
    )

    # Grain boundary characteristics
    hagb_base = 45.0 + 25.0 * t_norm - 6.0 * (porosity_percent / 100.0)
    if atmosphere == "hydrogen":
        hagb_base += 4.0
    elif atmosphere == "vacuum":
        hagb_base += 2.0
    elif atmosphere == "air":
        hagb_base -= 2.0
    hag_boundary_fraction_percent = clamp(hagb_base + rng.gauss(0.0, 3.5), 25.0, 85.0)

    avg_misorientation_deg = clamp(20.0 + 0.35 * (hag_boundary_fraction_percent - 30.0) + rng.gauss(0.0, 2.5), 15.0, 55.0)

    return {
        "grain_size_mean_um": round(grain_size_mean_um, 4),
        "grain_size_std_um": round(grain_size_std_um, 4),
        "porosity_percent": round(porosity_percent, 3),
        "pore_median_diameter_um": round(pore_median_diameter_um, 4),
        "pore_size_lognormal_sigma": round(pore_size_lognormal_sigma, 4),
        "hag_boundary_fraction_percent": round(hag_boundary_fraction_percent, 2),
        "avg_misorientation_deg": round(avg_misorientation_deg, 2),
        "relative_density": round(relative_density, 5),
        "sintering_index_model": round(sintering_index, 5),  # for transparency/debug
    }


def generate_rows(n_samples: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []
    for i in range(n_samples):
        inputs = sample_inputs(i + 1, rng)
        outputs = compute_microstructure(inputs, rng)
        row = {**inputs, **outputs}
        rows.append(row)
    return rows


def write_csv(rows: List[Dict[str, object]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic sintering process → microstructure dataset.")
    parser.add_argument("--n-samples", type=int, default=600, help="Number of samples to generate (default: 600)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument(
        "--out",
        type=str,
        default="/workspace/data/sintering_process_microstructure.csv",
        help="Output CSV file path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_rows(args.n_samples, args.seed)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows → {args.out}")


if __name__ == "__main__":
    main()

