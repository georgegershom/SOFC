#!/usr/bin/env python3
import csv
import math
import os
import random
from typing import List, Tuple


RANDOM_SEED = 42
DATA_DIR = "/workspace/data"


def ensure_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[object]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def generate_fem_vs_experimental(num_specimens: int = 10,
                                 points_per_specimen: int = 51) -> None:
    random.seed(RANDOM_SEED)
    header = [
        "specimen_id",
        "position_mm",
        "fem_stress_MPa",
        "fem_strain",
        "exp_stress_MPa",
        "exp_strain",
        "temperature_C",
        "loading_rate_MPa_per_s"
    ]

    rows: List[List[object]] = []
    for specimen_idx in range(1, num_specimens + 1):
        specimen_id = f"S{specimen_idx:02d}"
        gauge_length_mm = random.uniform(45.0, 55.0)
        temperature_c = random.choice([25, 100, 200, 300])
        loading_rate = random.uniform(0.5, 5.0)

        modulus_gpa = random.uniform(180.0, 220.0)
        yield_strength_mpa = random.uniform(300.0, 450.0)

        for i in range(points_per_specimen):
            position_mm = (gauge_length_mm / (points_per_specimen - 1)) * i

            base_strain = 0.0025 * math.sin(2 * math.pi * position_mm / max(gauge_length_mm, 1e-6)) + 0.003
            fem_stress_mpa = min(modulus_gpa * 1000.0 * base_strain, yield_strength_mpa + 150.0)
            fem_strain = base_strain if fem_stress_mpa < yield_strength_mpa else (yield_strength_mpa / (modulus_gpa * 1000.0) + 0.0005 * (fem_stress_mpa - yield_strength_mpa))

            noise_stress = random.gauss(0.0, 0.02 * max(100.0, fem_stress_mpa))
            bias_stress = random.uniform(-0.03, 0.03) * fem_stress_mpa
            exp_stress_mpa = max(0.0, fem_stress_mpa + noise_stress + bias_stress)

            nonlin = 1.0 + 0.05 * math.sin(4 * math.pi * position_mm / max(gauge_length_mm, 1e-6) + 0.5)
            exp_strain = fem_strain * nonlin + random.gauss(0.0, 0.0002)

            rows.append([
                specimen_id,
                round(position_mm, 3),
                round(fem_stress_mpa, 3),
                round(fem_strain, 6),
                round(exp_stress_mpa, 3),
                round(exp_strain, 6),
                temperature_c,
                round(loading_rate, 3),
            ])

    write_csv(os.path.join(DATA_DIR, "fem_vs_experimental_profiles.csv"), header, rows)


def _bounded(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def generate_crack_depth_xrd_vs_model(num_samples: int = 60) -> None:
    random.seed(RANDOM_SEED + 1)
    header = [
        "sample_id",
        "region",
        "xrd_crack_depth_um",
        "model_crack_depth_um",
        "xrd_confidence_0to1",
        "beam_energy_keV",
        "exposure_ms"
    ]

    regions = ["notch_root", "midspan", "edge", "weld_zone"]
    rows: List[List[object]] = []
    for sample_idx in range(1, num_samples + 1):
        sample_id = f"X{sample_idx:03d}"
        region = random.choice(regions)
        beam_energy = random.choice([15.0, 20.0, 25.0])
        exposure_ms = random.choice([50, 75, 100, 150])

        true_depth = random.uniform(5.0, 120.0)
        xrd_noise = random.gauss(0.0, 0.07 * true_depth)
        xrd_bias = random.uniform(-0.05, 0.05) * true_depth
        xrd_depth = _bounded(true_depth + xrd_noise + xrd_bias, 1.0, 200.0)

        model_bias = random.uniform(-0.1, 0.1) * true_depth
        model_noise = random.gauss(0.0, 0.05 * true_depth)
        model_depth = _bounded(true_depth + model_bias + model_noise, 1.0, 200.0)

        confidence = _bounded(1.0 - abs(xrd_depth - true_depth) / max(true_depth, 1e-6), 0.3, 0.99)

        rows.append([
            sample_id,
            region,
            round(xrd_depth, 2),
            round(model_depth, 2),
            round(confidence, 3),
            beam_energy,
            exposure_ms,
        ])

    write_csv(os.path.join(DATA_DIR, "crack_depth_xrd_vs_model.csv"), header, rows)


def generate_sintering_optima(num_experiments: int = 120) -> None:
    random.seed(RANDOM_SEED + 2)
    header = [
        "experiment_id",
        "binder_ratio_pct",
        "peak_temp_C",
        "dwell_time_min",
        "cooling_rate_C_per_min",
        "atmosphere",
        "density_pct",
        "porosity_pct",
        "fracture_toughness_MPa_m05",
        "grain_size_um",
        "objective_score",
        "is_pso_selected"
    ]

    atmospheres = ["air", "argon", "hydrogen"]
    rows: List[List[object]] = []

    for exp_idx in range(1, num_experiments + 1):
        experiment_id = f"SIN{exp_idx:03d}"
        binder_ratio = random.uniform(1.0, 6.0)
        peak_temp = random.uniform(1100.0, 1350.0)
        dwell_time = random.uniform(30.0, 180.0)
        cooling_rate = random.uniform(1.0, 2.0)
        atmosphere = random.choice(atmospheres)

        density = _bounded(85.0 + 0.02 * (peak_temp - 1100.0) + random.gauss(0.0, 1.2), 80.0, 99.5)
        porosity = _bounded(100.0 - density + random.gauss(0.0, 0.4), 0.0, 20.0)
        fracture_toughness = _bounded(3.0 + 0.002 * (peak_temp - 1100.0) + 0.4 * (2.0 - abs(cooling_rate - 1.5)) + random.gauss(0.0, 0.15), 2.0, 7.0)
        grain_size = _bounded(2.0 + 0.01 * (peak_temp - 1100.0) + 0.02 * dwell_time + random.gauss(0.0, 0.5), 1.0, 30.0)

        objective = (
            0.6 * (density / 100.0) +
            0.2 * (fracture_toughness / 7.0) +
            0.2 * (1.0 - porosity / 20.0)
        )

        is_selected = objective > 0.78 and 1.0 <= cooling_rate <= 2.0

        rows.append([
            experiment_id,
            round(binder_ratio, 2),
            round(peak_temp, 1),
            round(dwell_time, 1),
            round(cooling_rate, 3),
            atmosphere,
            round(density, 2),
            round(porosity, 2),
            round(fracture_toughness, 3),
            round(grain_size, 2),
            round(objective, 4),
            int(is_selected),
        ])

    write_csv(os.path.join(DATA_DIR, "sintering_parameters_optima.csv"), header, rows)


def generate_geometric_design_variations(num_designs: int = 80) -> None:
    random.seed(RANDOM_SEED + 3)
    header = [
        "design_id",
        "geometry_type",
        "channel_width_mm",
        "channel_height_mm",
        "curvature_1_per_mm",
        "length_mm",
        "pred_pressure_drop_Pa",
        "pred_stress_MPa",
        "meas_pressure_drop_Pa",
        "meas_stress_MPa"
    ]

    geometry_types = ["bow_shaped", "rectangular", "serpentine"]
    rows: List[List[object]] = []

    for idx in range(1, num_designs + 1):
        design_id = f"D{idx:03d}"
        geometry = random.choice(geometry_types)
        width = random.uniform(0.5, 2.5)
        height = random.uniform(0.2, 1.2)
        length = random.uniform(20.0, 120.0)
        curvature = 0.0
        if geometry == "bow_shaped":
            curvature = random.uniform(0.01, 0.08)
        elif geometry == "serpentine":
            curvature = random.uniform(0.03, 0.12)

        hydraulic_diameter = 2.0 * width * height / (width + height)
        visc_factor = 1.0 + 8.0 * curvature
        pred_dp = _bounded(500.0 * length / max(hydraulic_diameter, 1e-3) * visc_factor + random.gauss(0.0, 50.0), 100.0, 50000.0)
        pred_stress = _bounded(50.0 + 15.0 * curvature + 2.0 * length / 10.0 + random.gauss(0.0, 3.0), 10.0, 300.0)

        meas_dp = _bounded(pred_dp * random.uniform(0.9, 1.1) + random.gauss(0.0, 60.0), 100.0, 50000.0)
        meas_stress = _bounded(pred_stress * random.uniform(0.92, 1.08) + random.gauss(0.0, 4.0), 10.0, 300.0)

        rows.append([
            design_id,
            geometry,
            round(width, 3),
            round(height, 3),
            round(curvature, 4),
            round(length, 2),
            round(pred_dp, 2),
            round(pred_stress, 3),
            round(meas_dp, 2),
            round(meas_stress, 3),
        ])

    write_csv(os.path.join(DATA_DIR, "geometric_design_variations.csv"), header, rows)


def main() -> None:
    ensure_directory(DATA_DIR)
    generate_fem_vs_experimental()
    generate_crack_depth_xrd_vs_model()
    generate_sintering_optima()
    generate_geometric_design_variations()
    print(f"Synthetic datasets written to: {DATA_DIR}")


if __name__ == "__main__":
    main()

