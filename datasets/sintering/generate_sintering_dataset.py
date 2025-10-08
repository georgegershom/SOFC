#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class ParameterRanges:
    peak_temp_c: Tuple[int, int] = (1100, 1600)
    ramp_rate_c_per_min: Tuple[float, float] = (2.0, 20.0)
    cool_rate_c_per_min: Tuple[float, float] = (2.0, 20.0)
    hold_time_min: Tuple[int, int] = (0, 180)
    pressure_mpa: Tuple[float, float] = (0.0, 50.0)
    green_rel_density: Tuple[float, float] = (0.50, 0.65)
    powder_particle_size_um: Tuple[float, float] = (0.4, 6.0)
    green_pore_size_um: Tuple[float, float] = (0.3, 10.0)


ATMOSPHERE_EFFECTS: Dict[str, Dict[str, float]] = {
    # Coefficients are heuristic and chosen to induce realistic correlations.
    # densification: positive increases final relative density
    # gb_growth: positive increases grain growth tendency
    "air": {"densification": 0.00, "gb_growth": 0.00},
    "argon": {"densification": 0.03, "gb_growth": 0.02},
    "nitrogen": {"densification": -0.02, "gb_growth": -0.01},
    "vacuum": {"densification": 0.05, "gb_growth": 0.04},
    "hydrogen": {"densification": 0.08, "gb_growth": 0.06},
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def draw_atmosphere(rng: random.Random) -> str:
    atmospheres = list(ATMOSPHERE_EFFECTS.keys())
    # Bias toward common atmospheres
    probs = [0.35, 0.20, 0.15, 0.15, 0.15]
    r = rng.random()
    c = 0.0
    for a, p in zip(atmospheres, probs):
        c += p
        if r <= c:
            return a
    return atmospheres[-1]


def sample_temperature_profile(rng: random.Random, ranges: ParameterRanges) -> Dict[str, float]:
    peak_temp = rng.uniform(*ranges.peak_temp_c)
    ramp_rate = rng.uniform(*ranges.ramp_rate_c_per_min)
    cool_rate = rng.uniform(*ranges.cool_rate_c_per_min)
    # Inclusive range; acceptable for synthetic data
    hold_time = rng.randint(ranges.hold_time_min[0], ranges.hold_time_min[1])
    return {
        "peak_temp_c": float(peak_temp),
        "ramp_rate_c_per_min": float(ramp_rate),
        "cool_rate_c_per_min": float(cool_rate),
        "hold_time_min": int(hold_time),
    }


def sample_green_body(rng: random.Random, ranges: ParameterRanges) -> Dict[str, float]:
    green_rel_density = rng.uniform(*ranges.green_rel_density)

    # Powder particle D50 (um) with lognormal spread
    powder_d50 = rng.lognormvariate(mu=math.log(1.5), sigma=0.55)
    powder_d50 = clamp(powder_d50, *ranges.powder_particle_size_um)

    # Green pore size correlated with porosity and particle size
    base_pore = 0.6 * powder_d50 + 0.4 * rng.lognormvariate(mu=math.log(1.0), sigma=0.7)
    base_pore = clamp(base_pore, *ranges.green_pore_size_um)

    return {
        "green_rel_density": float(green_rel_density),
        "green_porosity_pct": float(100.0 * (1.0 - green_rel_density)),
        "powder_particle_size_d50_um": float(powder_d50),
        "green_pore_size_d50_um": float(base_pore),
    }


def sample_pressure(rng: random.Random, ranges: ParameterRanges) -> float:
    # Many runs are pressureless; mixture distribution with point-mass at 0
    if rng.uniform(0.0, 1.0) < 0.6:
        return 0.0
    return float(rng.uniform(5.0, ranges.pressure_mpa[1]))


def compute_rel_density_final(
    peak_temp_c: float,
    hold_time_min: float,
    ramp_rate: float,
    cool_rate: float,
    pressure_mpa: float,
    green_rel_density: float,
    atmosphere: str,
    rng: random.Random,
) -> float:
    # Normalize drivers roughly to 0..1
    t_norm = (peak_temp_c - 1100.0) / (1600.0 - 1100.0)
    hold_norm = min(hold_time_min / 120.0, 1.0)
    press_norm = math.log1p(pressure_mpa) / math.log1p(50.0)
    ramp_norm = (20.0 - clamp(ramp_rate, 2.0, 20.0)) / (20.0 - 2.0)  # slower is better
    cool_norm = (20.0 - clamp(cool_rate, 2.0, 20.0)) / (20.0 - 2.0)  # slower is mildly worse for density
    atmos = ATMOSPHERE_EFFECTS[atmosphere]

    # Weighted sum forming a sintering index
    index = (
        0.35 * t_norm
        + 0.20 * hold_norm
        + 0.20 * press_norm
        + 0.20 * green_rel_density
        + 0.05 * ramp_norm
        - 0.05 * cool_norm
        + 0.05 * atmos["densification"]
    )
    index = clamp(index, 0.0, 1.2)

    noise = rng.gauss(0.0, 0.02)
    rel_density = 0.78 + 0.20 * index + noise
    # Ensure final density exceeds green density by at least a minimal increment
    rel_density = max(rel_density, green_rel_density + 0.05)
    rel_density = clamp(rel_density, 0.80, 0.99)
    return float(rel_density)


def compute_grain_stats(
    d50_powder: float,
    peak_temp_c: float,
    hold_time_min: float,
    ramp_rate: float,
    cool_rate: float,
    pressure_mpa: float,
    atmosphere: str,
    rng: random.Random,
) -> Tuple[float, float, float]:
    t_norm = (peak_temp_c - 1100.0) / (1600.0 - 1100.0)
    hold_norm = min(hold_time_min / 120.0, 1.0)
    press_norm = math.log1p(pressure_mpa) / math.log1p(50.0)
    ramp_slow = (20.0 - clamp(ramp_rate, 2.0, 20.0)) / (20.0 - 2.0)
    cool_slow = (20.0 - clamp(cool_rate, 2.0, 20.0)) / (20.0 - 2.0)
    atmos = ATMOSPHERE_EFFECTS[atmosphere]

    base = max(d50_powder, 0.2)
    growth = (
        1.0
        + 1.8 * t_norm
        + 1.2 * hold_norm
        + 0.6 * ramp_slow
        + 0.5 * cool_slow
        - 0.3 * press_norm
        + 0.4 * atmos["gb_growth"]
    )
    growth = clamp(growth, 0.9, 4.0)
    d50 = base * growth * float(1.0 + rng.gauss(0.0, 0.08))
    d50 = clamp(d50, 0.3, 80.0)

    # Lognormal spread; higher pressure tightens distribution slightly
    sigma = clamp(0.35 - 0.10 * press_norm + rng.gauss(0.0, 0.03), 0.12, 0.55)
    mu = math.log(max(d50, 1e-6))
    z = 1.2815515655446004  # 10th/90th percentiles
    d10 = math.exp(mu - z * sigma)
    d90 = math.exp(mu + z * sigma)
    return float(d10), float(d50), float(d90)


def compute_pore_stats(
    green_pore_d50: float,
    green_rel_density: float,
    final_rel_density: float,
    rng: random.Random,
) -> Tuple[float, float, float]:
    # Shrink pores with densification; retain finite size floor
    dens_gain = max(final_rel_density - green_rel_density, 0.0)
    shrink_factor = clamp(1.0 - 0.75 * (dens_gain / max(1e-6, (1.0 - green_rel_density))), 0.15, 0.95)
    d50 = green_pore_d50 * shrink_factor * (1.0 + rng.gauss(0.0, 0.08))
    d50 = clamp(d50, 0.03, 30.0)

    sigma = clamp(0.45 - 0.20 * (dens_gain / 0.4) + rng.gauss(0.0, 0.05), 0.12, 0.70)
    mu = math.log(max(d50, 1e-6))
    z = 1.2815515655446004
    d10 = math.exp(mu - z * sigma)
    d90 = math.exp(mu + z * sigma)
    return float(d10), float(d50), float(d90)


def compute_gb_characteristics(
    t_norm: float,
    hold_norm: float,
    press_norm: float,
    final_porosity_pct: float,
    atmosphere: str,
    rng: random.Random,
) -> Tuple[float, float]:
    atmos = ATMOSPHERE_EFFECTS[atmosphere]
    hagb = 0.55 + 0.12 * t_norm + 0.10 * hold_norm - 0.06 * press_norm - 0.08 * (final_porosity_pct / 25.0) + 0.06 * atmos["gb_growth"]
    hagb = clamp(hagb + rng.gauss(0.0, 0.03), 0.30, 0.90)
    mean_misori = clamp(32.0 + 20.0 * (hagb - 0.5) + rng.gauss(0.0, 2.0), 10.0, 60.0)
    return float(hagb), float(mean_misori)


def generate_dataset(n_rows: int, seed: int, outdir: str) -> Dict[str, str]:
    rng = random.Random(seed)
    ranges = ParameterRanges()

    records: List[Dict[str, float]] = []
    for i in range(n_rows):
        # Inputs
        temp = sample_temperature_profile(rng, ranges)
        pressure = sample_pressure(rng, ranges)
        green = sample_green_body(rng, ranges)
        atmosphere = draw_atmosphere(rng)

        # Outputs (microstructure)
        rel_final = compute_rel_density_final(
            peak_temp_c=temp["peak_temp_c"],
            hold_time_min=temp["hold_time_min"],
            ramp_rate=temp["ramp_rate_c_per_min"],
            cool_rate=temp["cool_rate_c_per_min"],
            pressure_mpa=pressure,
            green_rel_density=green["green_rel_density"],
            atmosphere=atmosphere,
            rng=rng,
        )
        final_porosity_pct = 100.0 * (1.0 - rel_final)

        d10_g, d50_g, d90_g = compute_grain_stats(
            d50_powder=green["powder_particle_size_d50_um"],
            peak_temp_c=temp["peak_temp_c"],
            hold_time_min=temp["hold_time_min"],
            ramp_rate=temp["ramp_rate_c_per_min"],
            cool_rate=temp["cool_rate_c_per_min"],
            pressure_mpa=pressure,
            atmosphere=atmosphere,
            rng=rng,
        )

        d10_p, d50_p, d90_p = compute_pore_stats(
            green_pore_d50=green["green_pore_size_d50_um"],
            green_rel_density=green["green_rel_density"],
            final_rel_density=rel_final,
            rng=rng,
        )

        t_norm = (temp["peak_temp_c"] - 1100.0) / (1600.0 - 1100.0)
        hold_norm = min(temp["hold_time_min"] / 120.0, 1.0)
        press_norm = math.log1p(pressure) / math.log1p(50.0)
        hagb, mean_misori = compute_gb_characteristics(
            t_norm=t_norm,
            hold_norm=hold_norm,
            press_norm=press_norm,
            final_porosity_pct=final_porosity_pct,
            atmosphere=atmosphere,
            rng=rng,
        )

        record = {
            # Identifiers
            "sample_id": f"SPM-{seed}-{i:04d}",
            # Inputs: Sintering parameters
            **temp,
            "pressure_mpa": float(pressure),
            "atmosphere": atmosphere,
            # Inputs: Green body
            **green,
            # Outputs: Microstructure
            "grain_size_d10_um": d10_g,
            "grain_size_d50_um": d50_g,
            "grain_size_d90_um": d90_g,
            "final_porosity_pct": final_porosity_pct,
            "pore_size_d10_um": d10_p,
            "pore_size_d50_um": d50_p,
            "pore_size_d90_um": d90_p,
            "gb_high_angle_fraction": hagb,
            "gb_mean_misorientation_deg": mean_misori,
            "relative_density_final": rel_final,
        }
        records.append(record)

    # Order columns logically
    col_order = [
        "sample_id",
        "peak_temp_c",
        "ramp_rate_c_per_min",
        "hold_time_min",
        "cool_rate_c_per_min",
        "pressure_mpa",
        "atmosphere",
        "green_rel_density",
        "green_porosity_pct",
        "powder_particle_size_d50_um",
        "green_pore_size_d50_um",
        "grain_size_d10_um",
        "grain_size_d50_um",
        "grain_size_d90_um",
        "pore_size_d10_um",
        "pore_size_d50_um",
        "pore_size_d90_um",
        "gb_high_angle_fraction",
        "gb_mean_misorientation_deg",
        "relative_density_final",
        "final_porosity_pct",
    ]
    # Keep column order for file outputs using `col_order`

    os.makedirs(outdir, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base = os.path.join(outdir, "sintering_process_microstructure")

    csv_path = f"{base}.csv"
    json_path = f"{base}.json"
    meta_path = f"{base}_meta.json"

    # Save CSV
    # Prepare rounded records for writing
    rounded_records: List[Dict[str, object]] = []
    for rec in records:
        new_rec: Dict[str, object] = {}
        for k in col_order:
            v = rec[k]
            if isinstance(v, float):
                new_rec[k] = round(v, 4)
            else:
                new_rec[k] = v
        rounded_records.append(new_rec)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_order)
        writer.writeheader()
        writer.writerows(rounded_records)

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rounded_records, f, indent=2)

    # Save meta
    meta = {
        "generated_at_utc": ts,
        "seed": seed,
        "rows": n_rows,
        "parameter_ranges": {
            "peak_temp_c": list(ranges.peak_temp_c),
            "ramp_rate_c_per_min": list(ranges.ramp_rate_c_per_min),
            "cool_rate_c_per_min": list(ranges.cool_rate_c_per_min),
            "hold_time_min": list(ranges.hold_time_min),
            "pressure_mpa": list(ranges.pressure_mpa),
            "green_rel_density": list(ranges.green_rel_density),
            "powder_particle_size_um": list(ranges.powder_particle_size_um),
            "green_pore_size_um": list(ranges.green_pore_size_um),
        },
        "atmospheres": list(ATMOSPHERE_EFFECTS.keys()),
        "notes": "Synthetic dataset linking sintering process inputs to microstructure outputs with plausible correlations."
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {"csv": csv_path, "json": json_path, "meta": meta_path}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic sintering Process–Microstructure dataset")
    p.add_argument("--rows", type=int, default=300, help="Number of rows to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=str, default=os.path.dirname(__file__), help="Output directory")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_dataset(n_rows=args.rows, seed=args.seed, outdir=args.outdir)
    print(json.dumps(paths))


if __name__ == "__main__":
    main()

