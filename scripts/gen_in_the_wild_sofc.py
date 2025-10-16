#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # Optional; used only for previews
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    _MATPLOTLIB_AVAILABLE = False


@dataclass
class PlateParams:
    sample_id: str
    timestamp_iso: str
    batch_index: int
    grid_size: int
    plate_length_mm: float
    plate_width_mm: float
    thickness_mm: float
    elastic_modulus_gpa: float
    poisson_ratio: float
    anisotropy_axis_deg: float
    stress_scale_mpa: float
    edge_tension_mpa: float
    defect_density: float
    measurement_noise_rms_um: float
    outlier_fraction: float
    missing_patch_fraction: float
    drift_seed: int


@dataclass
class DatasetMeta:
    dataset_name: str
    dataset_version: str
    created_at: str
    num_samples: int
    grid_size: int
    coordinate_unit: str
    warp_unit: str
    stress_unit: str
    generator_script: str
    rng_seed: int


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _make_coordinate_grid(grid_size: int, length_mm: float, width_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, length_mm, grid_size, dtype=np.float32)
    y = np.linspace(0.0, width_mm, grid_size, dtype=np.float32)
    return np.meshgrid(x, y, indexing="xy")


def _spectral_poisson_solve(rhs: np.ndarray, alpha: float) -> np.ndarray:
    n = rhs.shape[0]
    rhs_fft = np.fft.rfft2(rhs)
    ky = np.fft.fftfreq(n) * (2.0 * math.pi)
    kx = np.fft.rfftfreq(n) * (2.0 * math.pi)
    ky2 = ky[:, None] ** 2
    kx2 = kx[None, :] ** 2
    k2 = ky2 + kx2
    k2[0, 0] = 1.0
    w_fft = alpha * rhs_fft / (-k2)
    w_fft[0, 0] = 0.0
    w = np.fft.irfft2(w_fft, s=rhs.shape)
    return w.astype(np.float32)


def _gaussian_spot(grid_size: int, center: Tuple[float, float], sigma: float, angle: float) -> np.ndarray:
    yy, xx = np.meshgrid(np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size), indexing="ij")
    c, s = math.cos(angle), math.sin(angle)
    xr = c * xx + s * yy
    yr = -s * xx + c * yy
    r2 = ((xr - center[0]) ** 2 + (yr - center[1]) ** 2) / (2.0 * sigma * sigma + 1e-8)
    return np.exp(-r2).astype(np.float32)


def _generate_residual_stress_fields(rng: np.random.Generator, grid_size: int, edge_tension_mpa: float,
                                     stress_scale_mpa: float, anisotropy_axis_rad: float,
                                     defect_density: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = grid_size
    base_modes = np.zeros((n, n), dtype=np.float32)
    max_mode = max(3, n // 16)
    for kx in range(1, max_mode + 1):
        for ky in range(1, max_mode + 1):
            phase = rng.uniform(0, 2 * math.pi)
            amplitude = rng.normal(0.0, 1.0) / (kx * kx + ky * ky)
            base_modes += amplitude * np.sin((kx * math.pi * np.arange(n)[:, None]) / n + phase) * \
                          np.cos((ky * math.pi * np.arange(n)[None, :]) / n - phase)
    preferred_c = math.cos(anisotropy_axis_rad)
    preferred_s = math.sin(anisotropy_axis_rad)
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    direction = preferred_c * xx + preferred_s * yy
    anisotropy_field = np.tanh(2.0 * direction).astype(np.float32)
    rim = np.maximum(0.0, 1.0 - (np.minimum(np.minimum(xx + 1, 1 - xx), np.minimum(yy + 1, 1 - yy))))
    rim = rim.astype(np.float32)
    spots = np.zeros_like(base_modes)
    num_spots = rng.poisson(defect_density * 6.0)
    for _ in range(num_spots):
        cx = rng.uniform(-0.8, 0.8)
        cy = rng.uniform(-0.8, 0.8)
        sigma = rng.uniform(0.02, 0.15)
        angle = rng.uniform(0, 2 * math.pi)
        sign = rng.choice([-1.0, 1.0])
        amp = rng.uniform(0.5, 2.0) * sign
        spots += amp * _gaussian_spot(n, (cx, cy), sigma, angle)
    sigma_xx = (base_modes + 0.5 * anisotropy_field + 0.3 * spots)
    sigma_yy = (base_modes - 0.5 * anisotropy_field + 0.3 * spots)
    tau_xy = 0.25 * spots
    radius = np.sqrt(xx ** 2 + yy ** 2)
    edge_enhance = np.clip((radius - 0.6) / 0.4, 0, 1)
    sigma_xx += edge_enhance * (edge_tension_mpa / max(stress_scale_mpa, 1e-6))
    sigma_yy += edge_enhance * (0.8 * edge_tension_mpa / max(stress_scale_mpa, 1e-6))
    sigma_xx = stress_scale_mpa * sigma_xx / max(np.std(sigma_xx), 1e-6)
    sigma_yy = stress_scale_mpa * sigma_yy / max(np.std(sigma_yy), 1e-6)
    tau_xy = (0.35 * stress_scale_mpa) * tau_xy / max(np.std(tau_xy) + 1e-6, 1e-6)
    return sigma_xx.astype(np.float32), sigma_yy.astype(np.float32), tau_xy.astype(np.float32)


def _warp_from_stress(rng: np.random.Generator, sigma_xx: np.ndarray, sigma_yy: np.ndarray, tau_xy: np.ndarray,
                      thickness_mm: float, elastic_modulus_gpa: float, poisson_ratio: float) -> np.ndarray:
    d_plate = (elastic_modulus_gpa * 1e9) * (thickness_mm * 1e-3) ** 3 / (12.0 * (1.0 - poisson_ratio ** 2))
    stress_combined = 0.6 * sigma_xx + 0.6 * sigma_yy - 0.1 * np.abs(tau_xy)
    stress_combined = stress_combined.astype(np.float32)
    stress_combined -= np.mean(stress_combined)
    scale = (thickness_mm * 1e-3) ** 2 / max(d_plate, 1e-12)
    alpha = 1.0e-4 * scale
    w = _spectral_poisson_solve(stress_combined, alpha)
    n = sigma_xx.shape[0]
    yy, xx = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n), indexing="ij")
    bowl = 0.02 * (xx ** 2 + yy ** 2)
    tilt_x = rng.normal(0.0, 0.002) * xx
    tilt_y = rng.normal(0.0, 0.002) * yy
    w = w + bowl + tilt_x + tilt_y
    w -= np.mean(w)
    return w.astype(np.float32)


def _add_measurement_effects(rng: np.random.Generator, w_clean: np.ndarray, noise_rms_um: float,
                             outlier_fraction: float, missing_patch_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    w_um = w_clean * 1e6
    noise = rng.normal(0.0, noise_rms_um, size=w_um.shape).astype(np.float32)
    w_noisy = w_um + noise
    if outlier_fraction > 0:
        num_outliers = int(outlier_fraction * w_noisy.size)
        if num_outliers > 0:
            idx = rng.choice(w_noisy.size, size=num_outliers, replace=False)
            flat = w_noisy.reshape(-1)
            flat[idx] += rng.normal(0.0, 50.0, size=num_outliers).astype(np.float32)
            w_noisy = flat.reshape(w_noisy.shape)
    mask = np.ones_like(w_noisy, dtype=np.uint8)
    if missing_patch_fraction > 0:
        n = w_noisy.shape[0]
        area_pixels = int(missing_patch_fraction * n * n)
        attempts = rng.integers(2, 5)
        for _ in range(attempts):
            h = rng.integers(max(2, n // 16), max(3, n // 6))
            w = max(2, int(area_pixels / max(h, 1)))
            y0 = rng.integers(0, max(1, n - h))
            x0 = rng.integers(0, max(1, n - w))
            mask[y0:y0 + h, x0:x0 + w] = 0
    w_measured = np.where(mask > 0, w_noisy, np.nan)
    return w_measured.astype(np.float32), mask.astype(np.uint8)


def _edge_crack_risk(sigma_xx: np.ndarray, sigma_yy: np.ndarray) -> float:
    n = sigma_xx.shape[0]
    margin = max(2, n // 16)
    edge_band = np.zeros_like(sigma_xx, dtype=bool)
    edge_band[:margin, :] = True
    edge_band[-margin:, :] = True
    edge_band[:, :margin] = True
    edge_band[:, -margin:] = True
    tensile = np.maximum(0.0, np.maximum(sigma_xx, sigma_yy))
    if not np.any(edge_band):
        return 0.0
    edge_tensile = tensile[edge_band]
    p95 = float(np.percentile(edge_tensile, 95))
    threshold = 220.0
    return float(np.clip(p95 / threshold, 0.0, 2.0))


def _write_preview_png(out_path: str, w_measured: np.ndarray, sigma_xx: np.ndarray, sigma_yy: np.ndarray) -> None:
    if not _MATPLOTLIB_AVAILABLE:
        return
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)
    im0 = axes[0].imshow(w_measured, cmap="viridis")
    axes[0].set_title("warp_measured [um]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(sigma_xx, cmap="coolwarm")
    axes[1].set_title("sigma_xx [MPa]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(sigma_yy, cmap="coolwarm")
    axes[2].set_title("sigma_yy [MPa]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out_path)
    plt.close(fig)


def generate_dataset(output_dir: str, num_samples: int, grid_size: int, seed: int,
                     train_frac: float, val_frac: float, viz_count: int) -> None:
    rng = np.random.default_rng(seed)
    _ensure_dir(output_dir)
    samples_dir = os.path.join(output_dir, "samples")
    _ensure_dir(samples_dir)
    splits_dir = os.path.join(output_dir, "splits")
    _ensure_dir(splits_dir)

    dataset_meta = DatasetMeta(
        dataset_name="SOFC_InTheWild_WarpStress",
        dataset_version="v1.0.0",
        created_at=datetime.utcnow().isoformat() + "Z",
        num_samples=num_samples,
        grid_size=grid_size,
        coordinate_unit="mm",
        warp_unit="um",
        stress_unit="MPa",
        generator_script=os.path.basename(__file__),
        rng_seed=seed,
    )

    base_length_mm = 100.0
    base_width_mm = 100.0
    base_thickness_mm = 0.5
    base_e_gpa = 150.0
    base_nu = 0.25

    global_drift_period = rng.uniform(40, 120)
    drift_phase = rng.uniform(0, 2 * math.pi)
    run_start_time = datetime.utcnow() - timedelta(days=num_samples // 6)

    all_ids = []
    train_ids, val_ids, test_ids = [], [], []

    for i in range(num_samples):
        sample_id = f"plate_{i:05d}"
        all_ids.append(sample_id)
        t = i / max(1.0, global_drift_period)
        drift_factor_e = 1.0 + 0.08 * math.sin(2 * math.pi * t + drift_phase) + rng.normal(0.0, 0.015)
        drift_factor_thick = 1.0 + 0.06 * math.cos(2 * math.pi * t + 0.7 * drift_phase) + rng.normal(0.0, 0.02)
        drift_factor_nu = 1.0 + 0.04 * math.sin(2 * math.pi * t + 1.3 * drift_phase) + rng.normal(0.0, 0.01)

        thickness_mm = float(np.clip(base_thickness_mm * drift_factor_thick, 0.35, 0.7))
        elastic_modulus_gpa = float(np.clip(base_e_gpa * drift_factor_e, 120.0, 190.0))
        poisson_ratio = float(np.clip(base_nu * drift_factor_nu, 0.18, 0.32))

        edge_tension_mpa = float(np.clip(rng.normal(90.0, 25.0), 40.0, 160.0))
        stress_scale_mpa = float(np.clip(rng.normal(70.0, 20.0), 30.0, 130.0))
        defect_density = float(np.clip(rng.normal(0.9, 0.35), 0.2, 2.0))
        anisotropy_axis_deg = float(np.clip(rng.normal(30.0, 40.0), -80.0, 80.0))

        noise_rms_um = float(np.clip(rng.normal(2.0, 0.7), 0.5, 5.0))
        outlier_fraction = float(np.clip(rng.normal(0.002, 0.002), 0.0, 0.02))
        missing_patch_fraction = float(np.clip(rng.normal(0.015, 0.01), 0.0, 0.07))

        params = PlateParams(
            sample_id=sample_id,
            timestamp_iso=(run_start_time + timedelta(minutes=10 * i)).isoformat() + "Z",
            batch_index=int(i // max(1, int(global_drift_period)) + 1),
            grid_size=grid_size,
            plate_length_mm=base_length_mm,
            plate_width_mm=base_width_mm,
            thickness_mm=thickness_mm,
            elastic_modulus_gpa=elastic_modulus_gpa,
            poisson_ratio=poisson_ratio,
            anisotropy_axis_deg=anisotropy_axis_deg,
            stress_scale_mpa=stress_scale_mpa,
            edge_tension_mpa=edge_tension_mpa,
            defect_density=defect_density,
            measurement_noise_rms_um=noise_rms_um,
            outlier_fraction=outlier_fraction,
            missing_patch_fraction=missing_patch_fraction,
            drift_seed=int(seed),
        )

        rng_i = np.random.default_rng(seed + i * 7919)
        sigma_xx, sigma_yy, tau_xy = _generate_residual_stress_fields(
            rng_i, grid_size, edge_tension_mpa, stress_scale_mpa, math.radians(anisotropy_axis_deg), defect_density
        )
        w_clean_m = _warp_from_stress(rng_i, sigma_xx, sigma_yy, tau_xy, thickness_mm, elastic_modulus_gpa, poisson_ratio)
        w_measured_um, mask = _add_measurement_effects(rng_i, w_clean_m, noise_rms_um, outlier_fraction, missing_patch_fraction)

        sample_dir = os.path.join(samples_dir, sample_id)
        _ensure_dir(sample_dir)
        coords_x, coords_y = _make_coordinate_grid(grid_size, base_length_mm, base_width_mm)
        np.save(os.path.join(sample_dir, "coords_x_mm.npy"), coords_x)
        np.save(os.path.join(sample_dir, "coords_y_mm.npy"), coords_y)
        np.save(os.path.join(sample_dir, "warp_measured_um.npy"), w_measured_um)
        np.save(os.path.join(sample_dir, "warp_clean_um.npy"), (w_clean_m * 1e6).astype(np.float32))
        np.save(os.path.join(sample_dir, "mask_uint8.npy"), mask)
        np.savez_compressed(os.path.join(sample_dir, "stress_true_mpa.npz"), sigma_xx=sigma_xx, sigma_yy=sigma_yy, tau_xy=tau_xy)
        with open(os.path.join(sample_dir, "params.json"), "w", encoding="utf-8") as f:
            json.dump(asdict(params), f, indent=2)

        risk = _edge_crack_risk(sigma_xx, sigma_yy)
        with open(os.path.join(sample_dir, "indicators.json"), "w", encoding="utf-8") as f:
            json.dump({"edge_crack_risk": risk}, f, indent=2)

        if i < viz_count:
            preview_path = os.path.join(sample_dir, "preview.png")
            _write_preview_png(preview_path, w_measured_um, sigma_xx, sigma_yy)

        if i < int(num_samples * train_frac):
            train_ids.append(sample_id)
        elif i < int(num_samples * (train_frac + val_frac)):
            val_ids.append(sample_id)
        else:
            test_ids.append(sample_id)

    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(dataset_meta), f, indent=2)

    with open(os.path.join(output_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "SOFC In-The-Wild Warp/Stress Dataset\n"
            "Version: v1.0.0\n\n"
            "Content:\n"
            "- Synthetic but physics-inspired residual stress fields and resulting warpage for SOFC-like plates.\n"
            "- Realistic measurement effects: noise, outliers, missing patches, edge-enhanced tension, anisotropy.\n"
            "- Unknown drift across time: thickness, modulus, Poisson ratio.\n\n"
            "Files per sample:\n"
            "- coords_x_mm.npy, coords_y_mm.npy: measurement grid coordinates.\n"
            "- warp_measured_um.npy: measured warp with noise/outliers/missing in micrometers (NaN for missing).\n"
            "- warp_clean_um.npy: clean warp in micrometers.\n"
            "- mask_uint8.npy: 1 for observed, 0 for missing.\n"
            "- stress_true_mpa.npz: sigma_xx, sigma_yy, tau_xy in MPa.\n"
            "- params.json: per-sample parameters including material and generator settings.\n"
            "- indicators.json: edge_crack_risk in [0, ~2], >1 indicates elevated risk.\n\n"
            "Splits:\n"
            "- splits/train.txt, val.txt, test.txt list sample IDs.\n\n"
            "Notes:\n"
            "- Warp is computed from stress via a spectral Poisson solve proxy for plate bending.\n"
            "- This is a fabricated dataset for benchmarking ML-augmented inverse modeling under in-the-wild effects.\n"
        )

    def _write_split(path: str, items: list):
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(item + "\n")

    _write_split(os.path.join(splits_dir, "train.txt"), train_ids)
    _write_split(os.path.join(splits_dir, "val.txt"), val_ids)
    _write_split(os.path.join(splits_dir, "test.txt"), test_ids)


def main():
    parser = argparse.ArgumentParser(description="Generate in-the-wild SOFC warp/stress dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=300)
    parser.add_argument("--grid_size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--viz_count", type=int, default=24)

    args = parser.parse_args()

    if args.train_frac + args.val_frac > 0.999:
        print("train_frac + val_frac must be < 1.0", file=sys.stderr)
        sys.exit(2)

    generate_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        grid_size=args.grid_size,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        viz_count=args.viz_count,
    )

    print(f"Dataset generated at: {args.output_dir}")


if __name__ == "__main__":
    main()
