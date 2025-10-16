#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


# -------------------------------
# Physics and synthesis utilities
# -------------------------------

@dataclass
class MaterialProps:
    youngs_modulus_pa: float = 200e9  # Pa (typical advanced ceramic)
    poisson_ratio: float = 0.28
    thickness_m: float = 0.0005  # 0.5 mm


@dataclass
class GeneratorConfig:
    grid_size: int = 96
    plate_length_m: float = 0.1  # 100 mm
    plate_width_m: float = 0.1
    stress_range_mpa: Tuple[float, float] = (-600.0, 600.0)
    # Drift process parameters (AR(1)-like)
    drift_alpha: float = 0.97
    drift_sigma: float = 10.0  # MPa per step
    # Edge amplification characteristics
    edge_amp_mean_mpa: float = 120.0
    edge_decay_len_ratio: float = 0.07  # fraction of plate size
    # Noise parameters
    meas_noise_sigma: float = 5e-6  # meters (5 microns)
    pink_noise_strength: float = 2e-6
    # Mapping from stress to curvature (scaled proxy)
    gradient_factor_mean: float = 0.6
    gradient_factor_std: float = 0.15
    # Random seed
    seed: int = 42


@dataclass
class SampleParams:
    # Snapshot of slow-drifting latent parameters
    base_stress_bias_mpa: float
    anisotropy_ratio: float
    edge_amp_mpa: float
    gradient_factor: float
    # Derived/flags
    edge_crack_risk: bool
    failure_mode_label: str
    # Measurement/meta
    tilt_x_rad: float
    tilt_y_rad: float
    pink_noise_strength: float
    meas_noise_sigma: float


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def make_coord_grid(nx: int, ny: int, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-Lx / 2.0, Lx / 2.0, nx, dtype=np.float64)
    y = np.linspace(-Ly / 2.0, Ly / 2.0, ny, dtype=np.float64)
    return np.meshgrid(x, y, indexing="xy")


def distance_to_edge_mask(nx: int, ny: int, Lx: float, Ly: float) -> np.ndarray:
    X, Y = make_coord_grid(nx, ny, Lx, Ly)
    dx = Lx / 2.0 - np.abs(X)
    dy = Ly / 2.0 - np.abs(Y)
    d = np.minimum(dx, dy)
    d[d < 0] = 0.0
    return d


def generate_edge_amplification(nx: int, ny: int, Lx: float, Ly: float, decay_len: float) -> np.ndarray:
    d = distance_to_edge_mask(nx, ny, Lx, Ly)
    # Larger near edges, decaying toward interior
    return np.exp(-d / decay_len)


def gaussian_filter_fft(field: np.ndarray, sigma_px: float) -> np.ndarray:
    # Simple Gaussian blur via FFT for correlated fields
    h, w = field.shape
    ky = np.fft.fftfreq(h)[:, None]
    kx = np.fft.fftfreq(w)[None, :]
    k2 = kx * kx + ky * ky
    # Convert desired sigma in pixels to frequency domain multiplier
    # Gaussian in spatial domain corresponds to exp(-2*pi^2 * sigma^2 * k^2) in frequency domain
    G = np.exp(-2.0 * (np.pi ** 2) * (sigma_px ** 2) * k2)
    F = np.fft.fft2(field)
    return np.fft.ifft2(F * G).real


def pink_noise(shape: Tuple[int, int], strength: float) -> np.ndarray:
    h, w = shape
    Ky = np.fft.fftfreq(h)[:, None]
    Kx = np.fft.fftfreq(w)[None, :]
    k2 = Kx * Kx + Ky * Ky
    k = np.sqrt(k2)
    # Avoid singularity at DC
    k[0, 0] = 1.0
    amplitude = 1.0 / (k + 1e-6)
    phase = np.exp(1j * 2 * np.pi * np.random.rand(h, w))
    noise = np.fft.ifft2(amplitude * phase).real
    noise = noise / (np.std(noise) + 1e-12)
    return strength * noise


def map_stress_to_curvature(sigma_xx: np.ndarray, sigma_yy: np.ndarray, sigma_xy: np.ndarray,
                            material: MaterialProps, gradient_factor: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Thin plate proxy: curvature ~ -6/(E * t) * stress_gradient_through_thickness
    # We absorb the unknown through-thickness gradient into gradient_factor
    scale = -6.0 / (material.youngs_modulus_pa * material.thickness_m)
    scale *= gradient_factor
    kx = scale * sigma_xx * 1e6  # MPa to Pa
    ky = scale * sigma_yy * 1e6
    kxy = scale * sigma_xy * 1e6
    return kx, ky, kxy


def integrate_curvature_to_warp(kx: np.ndarray, ky: np.ndarray, kxy: np.ndarray,
                                Lx: float, Ly: float) -> np.ndarray:
    # Solve least-squares: minimize ||dxx w - kx||^2 + ||dyy w - ky||^2 + ||2 dxy w - kxy||^2
    # FFT solution
    h, w = kx.shape
    Ky = 2.0 * np.pi * np.fft.fftfreq(h)[:, None]  # angular spatial frequency
    Kx = 2.0 * np.pi * np.fft.fftfreq(w)[None, :]

    # Scale frequencies for physical dimensions
    Ky = Ky * (h / Ly)
    Kx = Kx * (w / Lx)

    Kx2 = Kx * Kx
    Ky2 = Ky * Ky
    KxKy = Kx * Ky

    Kx4 = Kx2 * Kx2
    Ky4 = Ky2 * Ky2

    # Fourier transforms of curvature fields
    KX = np.fft.fft2(kx)
    KY = np.fft.fft2(ky)
    KXY = np.fft.fft2(kxy)

    # Normal equation in frequency domain
    denom = Kx4 + Ky4 + 4.0 * Kx2 * Ky2
    rhs = -(Kx2 * KX + Ky2 * KY + 2.0 * KxKy * KXY)

    # Regularize near zero to avoid division blow-up
    eps = 1e-12
    denom = np.where(np.abs(denom) < eps, eps, denom)

    W = rhs / denom
    W[0, 0] = 0.0  # zero-mean warp

    w_spatial = np.fft.ifft2(W).real
    return w_spatial


def fit_and_remove_best_fit_plane(w: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    # Fit w ~ a*x + b*y + c
    A = np.stack([X.ravel(), Y.ravel(), np.ones_like(X).ravel()], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, w.ravel(), rcond=None)
    a, b, c = coeffs.tolist()
    plane = (a * X + b * Y + c)
    detrended = w - plane
    return detrended, {"tilt_x": float(a), "tilt_y": float(b), "offset": float(c)}


# -------------------------------
# Synthesis pipeline
# -------------------------------

def synthesize_stress_fields(cfg: GeneratorConfig, step_state: Dict, nx: int, ny: int, Lx: float, Ly: float,
                              rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, SampleParams]:
    # Update slow drift state
    # Base bias performs a random walk
    base_bias = step_state.get("base_bias", 0.0)
    base_bias = cfg.drift_alpha * base_bias + (1.0 - cfg.drift_alpha) * rng.normal(0.0, cfg.drift_sigma)

    # Anisotropy drifts slowly around 1.0
    anisotropy = step_state.get("anisotropy", 1.0)
    anisotropy = 0.98 * anisotropy + 0.02 * rng.normal(1.0, 0.05)

    # Edge amplification amplitude drifts
    edge_amp = step_state.get("edge_amp", cfg.edge_amp_mean_mpa)
    edge_amp = 0.98 * edge_amp + 0.02 * rng.normal(cfg.edge_amp_mean_mpa, 10.0)

    # Gradient factor per sample
    gradient_factor = float(np.clip(rng.normal(cfg.gradient_factor_mean, cfg.gradient_factor_std), 0.2, 1.2))

    # Fields
    # Base global field
    base_field = rng.normal(0.0, 1.0, size=(ny, nx)).astype(np.float64)
    base_field = gaussian_filter_fft(base_field, sigma_px=nx * 0.08)
    base_field = base_field / (np.std(base_field) + 1e-12)

    # Add a large-scale trend
    trend_x = np.linspace(-1.0, 1.0, nx)[None, :]
    trend_y = np.linspace(-1.0, 1.0, ny)[:, None]

    # Edge amplification map
    decay_len = cfg.edge_decay_len_ratio * min(Lx, Ly)
    edge_map = generate_edge_amplification(nx, ny, Lx, Ly, decay_len)

    # Defect-induced local Gaussians
    defect_field = np.zeros((ny, nx), dtype=np.float64)
    num_defects = rng.integers(0, 4)
    for _ in range(int(num_defects)):
        cx = rng.integers(0, nx)
        cy = rng.integers(0, ny)
        amp = float(rng.normal(1.0, 0.3))
        sx = float(rng.uniform(nx * 0.01, nx * 0.05))
        sy = float(rng.uniform(ny * 0.01, ny * 0.05))
        xx = (np.arange(nx) - cx)[None, :]
        yy = (np.arange(ny) - cy)[:, None]
        g = amp * np.exp(-0.5 * ((xx / sx) ** 2 + (yy / sy) ** 2))
        defect_field += g

    # Compose stress components in MPa
    sigma_xx = (base_bias
                + 160.0 * base_field
                + 40.0 * trend_x
                + 0.0 * trend_y
                + edge_amp * edge_map
                + 100.0 * defect_field)

    sigma_yy = (base_bias
                + 160.0 * (anisotropy * base_field)
                + 0.0 * trend_x
                + 40.0 * trend_y
                + edge_amp * edge_map
                + 80.0 * defect_field)

    sigma_xy = 20.0 * gaussian_filter_fft(rng.normal(0.0, 1.0, size=(ny, nx)), sigma_px=nx * 0.05)

    # Clip to plausible range
    mn, mx = cfg.stress_range_mpa
    sigma_xx = np.clip(sigma_xx, mn, mx)
    sigma_yy = np.clip(sigma_yy, mn, mx)
    sigma_xy = np.clip(sigma_xy, -200.0, 200.0)

    # Edge crack risk heuristic: tensile hoop-like stress near edges
    edge_band = edge_map > np.exp(-0.5)
    tensile_near_edge = (sigma_xx[edge_band] > 250.0).mean() + (sigma_yy[edge_band] > 250.0).mean()
    edge_crack_risk = bool(tensile_near_edge / 2.0 > 0.25)
    failure_mode_label = "edge_crack" if edge_crack_risk else ("corner_crack" if rng.random() < 0.05 else "none")

    # Measurement tilts
    tilt_x = float(rng.normal(0.0, 1e-3))  # small radians/m
    tilt_y = float(rng.normal(0.0, 1e-3))

    sample_params = SampleParams(
        base_stress_bias_mpa=float(base_bias),
        anisotropy_ratio=float(anisotropy),
        edge_amp_mpa=float(edge_amp),
        gradient_factor=gradient_factor,
        edge_crack_risk=edge_crack_risk,
        failure_mode_label=failure_mode_label,
        tilt_x_rad=tilt_x,
        tilt_y_rad=tilt_y,
        pink_noise_strength=float(cfg.pink_noise_strength * rng.uniform(0.6, 1.4)),
        meas_noise_sigma=float(cfg.meas_noise_sigma * rng.uniform(0.7, 1.3)),
    )

    # Persist drift state for next step
    step_state["base_bias"] = float(base_bias)
    step_state["anisotropy"] = float(anisotropy)
    step_state["edge_amp"] = float(edge_amp)

    return sigma_xx.astype(np.float32), sigma_yy.astype(np.float32), sigma_xy.astype(np.float32), sample_params


def generate_one_sample(cfg: GeneratorConfig, material: MaterialProps, nx: int, ny: int,
                        Lx: float, Ly: float, step_state: Dict, idx: int,
                        rng: np.random.Generator) -> Dict:
    sigma_xx, sigma_yy, sigma_xy, sparams = synthesize_stress_fields(cfg, step_state, nx, ny, Lx, Ly, rng)
    # Map to curvature and integrate to warp
    kx, ky, kxy = map_stress_to_curvature(sigma_xx, sigma_yy, sigma_xy, material, sparams.gradient_factor)
    w = integrate_curvature_to_warp(kx, ky, kxy, Lx=Lx, Ly=Ly)

    # Add measurement tilt and noise, then detrend plane to mimic fixture zeroing
    X, Y = make_coord_grid(nx, ny, Lx, Ly)
    w_meas = w + (sparams.tilt_x_rad * X + sparams.tilt_y_rad * Y)
    w_meas += pink_noise(w.shape, sparams.pink_noise_strength)
    w_meas += np.random.normal(0.0, sparams.meas_noise_sigma, size=w.shape)

    w_detrended, plane = fit_and_remove_best_fit_plane(w_meas, X, Y)

    return {
        "sigma_xx": sigma_xx,
        "sigma_yy": sigma_yy,
        "sigma_xy": sigma_xy,
        "warp": w_detrended.astype(np.float32),
        "warp_clean": w.astype(np.float32),
        "meta": {
            **asdict(sparams),
            "plane_fit": plane,
            "sample_index": idx,
        },
    }


def save_sample(out_dir: Path, sample: Dict, sample_id: str) -> None:
    p = out_dir / sample_id
    p.mkdir(parents=True, exist_ok=True)
    # Save arrays
    np.save(p / "warp.npy", sample["warp"].astype(np.float32))
    # Save stress fields in one npz
    np.savez_compressed(p / "stress.npz",
                        sigma_xx=sample["sigma_xx"].astype(np.float32),
                        sigma_yy=sample["sigma_yy"].astype(np.float32),
                        sigma_xy=sample["sigma_xy"].astype(np.float32))
    # Per-sample meta
    with open(p / "meta.json", "w") as f:
        json.dump(sample["meta"], f, indent=2)


def compute_dataset_stats(root: Path, splits: Dict[str, int]) -> Dict:
    stats = {}
    for split in ["train", "val", "test"]:
        split_dir = root / split
        warp_means = []
        warp_stds = []
        stress_means = []
        stress_stds = []
        count = 0
        for child in split_dir.iterdir():
            if not child.is_dir():
                continue
            warp_path = child / "warp.npy"
            stress_path = child / "stress.npz"
            if not (warp_path.exists() and stress_path.exists()):
                continue
            w = np.load(warp_path)
            s = np.load(stress_path)
            warp_means.append(float(w.mean()))
            warp_stds.append(float(w.std()))
            s_stack = np.stack([s["sigma_xx"], s["sigma_yy"], s["sigma_xy"]], axis=0)
            stress_means.append([float(s_stack[i].mean()) for i in range(3)])
            stress_stds.append([float(s_stack[i].std()) for i in range(3)])
            count += 1
        if count == 0:
            stats[split] = {}
            continue
        stress_means = np.array(stress_means)
        stress_stds = np.array(stress_stds)
        stats[split] = {
            "num_samples": count,
            "warp_mean_of_means": float(np.mean(warp_means)),
            "warp_mean_of_stds": float(np.mean(warp_stds)),
            "stress_mean_of_means": {
                "sigma_xx": float(np.mean(stress_means[:, 0])),
                "sigma_yy": float(np.mean(stress_means[:, 1])),
                "sigma_xy": float(np.mean(stress_means[:, 2])),
            },
            "stress_mean_of_stds": {
                "sigma_xx": float(np.mean(stress_stds[:, 0])),
                "sigma_yy": float(np.mean(stress_stds[:, 1])),
                "sigma_xy": float(np.mean(stress_stds[:, 2])),
            },
        }
    return stats


def package_zip(root: Path, version: str = "v1") -> Path:
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    zip_path = artifacts / f"dataset4_in_the_wild_{version}.zip"
    if zip_path.exists():
        zip_path.unlink()
    # Create archive (without the artifacts dir itself to avoid recursion)
    tmp_copy_root = root / f"_package_tmp_{int(time.time())}"
    if tmp_copy_root.exists():
        shutil.rmtree(tmp_copy_root)
    tmp_copy_root.mkdir(parents=True)
    # Copy splits and meta and src
    for name in ["train", "val", "test", "meta", "src"]:
        srcp = root / name
        if srcp.exists():
            destp = tmp_copy_root / name
            shutil.copytree(srcp, destp)
    # Make the zip
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(tmp_copy_root))
    # Cleanup
    shutil.rmtree(tmp_copy_root)
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Generate 'In-The-Wild' SOFC plate warp/stress dataset")
    parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1]), help="Dataset root directory")
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=60)
    parser.add_argument("--num-test", type=int, default=60)
    parser.add_argument("--grid", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zip", action="store_true")
    args = parser.parse_args()

    root = Path(args.out)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "val").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    cfg = GeneratorConfig(grid_size=args.grid, seed=args.seed)
    material = MaterialProps()

    nx = ny = cfg.grid_size
    Lx = cfg.plate_length_m
    Ly = cfg.plate_width_m

    splits = {"train": args.num_train, "val": args.num_val, "test": args.num_test}

    manifest = {
        "name": "Dataset 4: In-The-Wild SOFC Warp/Stress",
        "description": "Fabricated operational dataset with drift, noise, edge failure modes",
        "target": "Predict in-plane residual stress fields from warp measurements",
        "units": {
            "stress": "MPa",
            "warp": "meters",
            "dimensions": "meters",
        },
        "material": asdict(material),
        "config": asdict(cfg),
        "splits": splits,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "seed": args.seed,
        "license": "CC-BY-4.0",
        "citation": "ML-Augmented Inverse Modeling for Residual Stress Quantification from Warped SOFC Plates",
    }

    with open(root / "meta" / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Slow drift shared across samples
    step_state: Dict = {}

    def gen_split(split_name: str, n: int, start_idx: int) -> int:
        out_dir = root / split_name
        for i in range(n):
            idx = start_idx + i
            sample = generate_one_sample(cfg, material, nx, ny, Lx, Ly, step_state, idx, rng)
            sample_id = f"plate_{idx:06d}"
            save_sample(out_dir, sample, sample_id)
        return start_idx + n

    next_idx = 0
    next_idx = gen_split("train", splits["train"], next_idx)
    next_idx = gen_split("val", splits["val"], next_idx)
    next_idx = gen_split("test", splits["test"], next_idx)

    stats = compute_dataset_stats(root, splits)
    with open(root / "meta" / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    if args.zip:
        zip_path = package_zip(root, version="v1")
        print(f"Packaged dataset at: {zip_path}")

    print("Generation complete.")


if __name__ == "__main__":
    main()
