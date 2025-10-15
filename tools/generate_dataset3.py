#!/usr/bin/env python3
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from jsonschema import validate
from tqdm import tqdm

DATASET_ROOT = Path("/workspace/datasets/dataset3")
SAMPLES_DIR = DATASET_ROOT / "samples"
SCHEMA_DIR = DATASET_ROOT / "schema"

# ------------------------ Physics-inspired helpers -------------------------

def sample_material(name: str) -> Dict[str, float]:
    # Typical ranges for SOFC materials (illustrative)
    E_ranges = {
        "electrolyte": (140, 220),  # GPa
        "anode": (20, 80),
        "cathode": (40, 120),
    }
    alpha_ranges = {
        "electrolyte": (8.5e-6, 11.5e-6),
        "anode": (9.0e-6, 13.0e-6),
        "cathode": (11.0e-6, 15.0e-6),
    }
    nu_ranges = {
        "electrolyte": (0.26, 0.33),
        "anode": (0.20, 0.33),
        "cathode": (0.22, 0.35),
    }
    E = random.uniform(*E_ranges[name])
    alpha = random.uniform(*alpha_ranges[name])
    nu = random.uniform(*nu_ranges[name])
    return {"E_GPa": E, "nu": nu, "alpha_per_K": alpha}


def latin_hypercube(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    # Simple LHS for coverage of parameter space
    cut = np.linspace(0, 1, n + 1)
    u = rng.random((n, d))
    a = cut[:n]
    b = cut[1 : n + 1]
    rdpoints = u * (b - a)[:, None] + a[:, None]
    H = np.zeros_like(rdpoints)
    for j in range(d):
        order = rng.permutation(n)
        H[:, j] = rdpoints[order, 0]
    return H


def _gaussian_kernel1d(sigma: float) -> np.ndarray:
    # Create 1D Gaussian kernel with size ~ 6*sigma
    sigma = max(1e-6, float(sigma))
    half = int(max(1, np.ceil(3 * sigma)))
    x = np.arange(-half, half + 1)
    g = np.exp(-(x ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.astype(np.float64)


def _gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    # Separable convolution with reflect padding
    g = _gaussian_kernel1d(sigma)
    # Convolve rows
    pad_h = len(g) // 2
    padded = np.pad(image, ((0, 0), (pad_h, pad_h)), mode="reflect")
    tmp = np.empty_like(image, dtype=np.float64)
    for i in range(image.shape[0]):
        tmp[i, :] = np.convolve(padded[i, :], g, mode="valid")
    # Convolve columns
    padded2 = np.pad(tmp, ((pad_h, pad_h), (0, 0)), mode="reflect")
    out = np.empty_like(image, dtype=np.float64)
    for j in range(image.shape[1]):
        out[:, j] = np.convolve(padded2[:, j], g, mode="valid")
    return out


def generate_warp_map(length_mm: float, width_mm: float, kx: float, ky: float,
                      twist: float, waviness: float, grid_n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create a base sag surface from principal curvatures and add twist and waviness
    x = np.linspace(-length_mm / 2, length_mm / 2, grid_n)
    y = np.linspace(-width_mm / 2, width_mm / 2, grid_n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = 0.5 * (kx * (X * 1e-3) ** 2 + ky * (Y * 1e-3) ** 2)  # quadratic sag (meters)
    Z += twist * (X * 1e-3) * (Y * 1e-3)

    # add long-wavelength waviness via filtered noise
    noise = rng.normal(0, 1, size=Z.shape)
    noise = _gaussian_blur(noise, sigma=max(1.0, grid_n / 15))
    Z += waviness * noise * 1e-6  # meters

    # add short-range roughness (scanner noise)
    Z += rng.normal(0, 0.1e-6, size=Z.shape)
    return X, Y, Z


def stoney_biaxial_stress(E_sub_GPa: float, nu_sub: float, t_sub_m: float, t_film_m: float, curvature_1_per_m: float) -> float:
    # Stoney-like formula: sigma = (E_sub * t_sub^2) / (6 * (1 - nu_sub) * t_film) * curvature
    E_sub = E_sub_GPa * 1e9
    denom = 6 * (1 - nu_sub) * max(t_film_m, 1e-9)
    return (E_sub * (t_sub_m ** 2) / denom) * curvature_1_per_m  # Pa


def simulate_layer_removal_profile(depths_m: np.ndarray, base_sigma_MPa: float, gradient_MPa_per_mm: float, rng: np.random.Generator) -> np.ndarray:
    # Linear stress gradient + noise, truncated near zero-depth
    base = base_sigma_MPa + (depths_m * 1e3) * gradient_MPa_per_mm
    noise = rng.normal(0, 2.0, size=depths_m.shape)
    return base + noise


def simulate_xrd_map(npts: int, mean_sigma_MPa: float, std_sigma_MPa: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = rng.uniform(0, 1, npts)
    ys = rng.uniform(0, 1, npts)
    sigmas = rng.normal(mean_sigma_MPa, std_sigma_MPa, size=npts)
    return xs, ys, sigmas


def simulate_raman_map(npts: int, mean_sigma_MPa: float, std_sigma_MPa: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = rng.uniform(0, 1, npts)
    ys = rng.uniform(0, 1, npts)
    sigmas = rng.normal(mean_sigma_MPa, std_sigma_MPa, size=npts)
    return xs, ys, sigmas


# -------------------------- Generator main logic ---------------------------

@dataclass
class Geometry:
    length_mm: float
    width_mm: float
    electrolyte_thickness_um: float
    anode_thickness_um: float
    cathode_thickness_um: float


def generate_sample(i: int, rng: np.random.Generator, grid_n: int = 64) -> Dict:
    # Parameter space extremes and center via stratified sampling
    # Geometry
    length_mm = rng.uniform(20, 60)
    width_mm = rng.uniform(20, 60)
    electrolyte_th_um = rng.uniform(5, 30)
    anode_th_um = rng.uniform(200, 1000)
    cathode_th_um = rng.uniform(20, 100)

    # Materials
    electrolyte = sample_material("electrolyte")
    anode = sample_material("anode")
    cathode = sample_material("cathode")

    # Sintering profile
    peak_temp_C = rng.uniform(1100, 1400)
    dwell_time_min = rng.uniform(30, 180)
    ramp_up = rng.uniform(1, 5)
    ramp_down = rng.uniform(1, 5)

    # Shrinkage anisotropy (%)
    def shrink_pair(center: Tuple[float, float], span: Tuple[float, float]):
        return {
            "x_pct": center[0] + rng.uniform(-span[0] / 2, span[0] / 2),
            "y_pct": center[1] + rng.uniform(-span[1] / 2, span[1] / 2),
        }

    electrolyte_shrink = shrink_pair((14.0, 14.0), (4.0, 4.0))
    anode_shrink = shrink_pair((18.0, 18.0), (6.0, 6.0))
    cathode_shrink = shrink_pair((16.0, 16.0), (6.0, 6.0))

    # Curvatures influenced by mismatch in CTE and shrinkage
    delta_alpha_a = (electrolyte["alpha_per_K"] - anode["alpha_per_K"])  # 1/K
    delta_alpha_c = (electrolyte["alpha_per_K"] - cathode["alpha_per_K"])  # 1/K
    deltaT = peak_temp_C - 25.0

    # Heuristic curvature magnitudes
    kx = 0.5e-3 * (delta_alpha_a * deltaT) + 0.2e-3 * (electrolyte_shrink["x_pct"] - anode_shrink["x_pct"]) * 1e-2
    ky = 0.5e-3 * (delta_alpha_c * deltaT) + 0.2e-3 * (electrolyte_shrink["y_pct"] - cathode_shrink["y_pct"]) * 1e-2

    twist = rng.normal(0.0, 1e-6)
    waviness = rng.uniform(0.2, 1.5)

    X, Y, Z = generate_warp_map(length_mm, width_mm, kx, ky, twist, waviness, grid_n, rng)

    # Stoney curvature extraction (global fit of quadratic surface)
    # Fit Z ~ ax^2 + by^2 + cxy + d
    x_m = X * 1e-3
    y_m = Y * 1e-3
    A = np.column_stack([0.5 * x_m.ravel() ** 2, 0.5 * y_m.ravel() ** 2, x_m.ravel() * y_m.ravel(), np.ones_like(x_m.ravel())])
    coeffs, *_ = np.linalg.lstsq(A, Z.ravel(), rcond=None)
    kx_est, ky_est, twist_est, offset = coeffs

    # Biaxial stress estimate via Stoney (electrolyte film on anode substrate)
    t_sub = anode_th_um * 1e-6
    t_film = electrolyte_th_um * 1e-6
    sigma_x_Pa = stoney_biaxial_stress(anode["E_GPa"], anode["nu"], t_sub, t_film, kx_est)
    sigma_y_Pa = stoney_biaxial_stress(anode["E_GPa"], anode["nu"], t_sub, t_film, ky_est)
    sigma_x_MPa = float(sigma_x_Pa / 1e6)
    sigma_y_MPa = float(sigma_y_Pa / 1e6)

    # Layer removal: remove cathode then electrolyte (example)
    depths_m = np.linspace(0, (electrolyte_th_um + cathode_th_um) * 1e-6, 12)
    layer_profile_MPa = simulate_layer_removal_profile(depths_m, base_sigma_MPa=(sigma_x_MPa + sigma_y_MPa) / 2, gradient_MPa_per_mm=rng.uniform(-300, 300), rng=rng)

    # XRD/Raman local maps targeting surface stress dispersion
    mean_surf = (sigma_x_MPa + sigma_y_MPa) / 2
    xs_xrd, ys_xrd, sigmas_xrd = simulate_xrd_map(200, mean_surf, std_sigma_MPa=max(5.0, 0.1 * abs(mean_surf)), rng=rng)
    xs_ram, ys_ram, sigmas_ram = simulate_raman_map(400, mean_surf, std_sigma_MPa=max(8.0, 0.15 * abs(mean_surf)), rng=rng)

    # Save files
    sid = f"S{i:03d}"
    out_dir = SAMPLES_DIR / sid
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warp map CSV
    warp_csv = out_dir / "warp_map.csv"
    warp_arr = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    np.savetxt(warp_csv, warp_arr, delimiter=",", header="x_mm,y_mm,z_m", comments="", fmt=["%.6f", "%.6f", "%.9e"])

    # Curvature JSON (global)
    curvature_json = out_dir / "curvature.json"
    with open(curvature_json, "w") as f:
        json.dump({
            "kx_1_per_m": float(kx_est),
            "ky_1_per_m": float(ky_est),
            "twist": float(twist_est)
        }, f, indent=2)

    # Layer removal directory
    lr_dir = out_dir / "layer_removal"
    lr_dir.mkdir(exist_ok=True)
    lr_csv = lr_dir / "profile.csv"
    lr_arr = np.column_stack([depths_m, layer_profile_MPa])
    np.savetxt(lr_csv, lr_arr, delimiter=",", header="depth_m,sigma_MPa", comments="", fmt=["%.9e", "%.6f"])

    # XRD CSV
    xrd_csv = out_dir / "xrd.csv"
    xrd_arr = np.column_stack([xs_xrd, ys_xrd, sigmas_xrd])
    np.savetxt(xrd_csv, xrd_arr, delimiter=",", header="x_norm,y_norm,sigma_MPa", comments="", fmt=["%.6f", "%.6f", "%.6f"])

    # Raman CSV
    raman_csv = out_dir / "raman.csv"
    raman_arr = np.column_stack([xs_ram, ys_ram, sigmas_ram])
    np.savetxt(raman_csv, raman_arr, delimiter=",", header="x_norm,y_norm,sigma_MPa", comments="", fmt=["%.6f", "%.6f", "%.6f"])

    # Sample JSON
    sample_json = {
        "sample_id": sid,
        "parameters": {
            "geometry": {
                "length_mm": float(length_mm),
                "width_mm": float(width_mm),
                "electrolyte_thickness_um": float(electrolyte_th_um),
                "anode_thickness_um": float(anode_th_um),
                "cathode_thickness_um": float(cathode_th_um),
            },
            "materials": {
                "electrolyte": electrolyte,
                "anode": anode,
                "cathode": cathode,
            },
            "sintering": {
                "peak_temp_C": float(peak_temp_C),
                "dwell_time_min": float(dwell_time_min),
                "ramp_up_C_per_min": float(ramp_up),
                "ramp_down_C_per_min": float(ramp_down),
            },
            "shrinkage": {
                "electrolyte_pct_xy": electrolyte_shrink,
                "anode_pct_xy": anode_shrink,
                "cathode_pct_xy": cathode_shrink,
            },
        },
        "measurement_files": {
            "warp_map_csv": str(warp_csv.relative_to(DATASET_ROOT)),
            "curvature_json": str(curvature_json.relative_to(DATASET_ROOT)),
            "layer_removal_dir": str(lr_dir.relative_to(DATASET_ROOT)),
            "xrd_csv": str(xrd_csv.relative_to(DATASET_ROOT)),
            "raman_csv": str(raman_csv.relative_to(DATASET_ROOT)),
        },
        "derived": {
            "curvature_1_per_m": {"kx": float(kx_est), "ky": float(ky_est)},
            "stoney_sigma_electrolyte_MPa": {"sigma_x": sigma_x_MPa, "sigma_y": sigma_y_MPa},
        },
        "technique_metadata": {
            "warp_scanner": "synthetic-laser-confocal",
            "xrd_device": "synthetic-xrd",
            "raman_device": "synthetic-raman",
            "notes": "Synthetic data approximating realistic distributions"
        },
        "units": {"length": "mm", "thickness": "um", "curvature": "1/m", "stress": "MPa"}
    }

    # Validate against schema if present
    schema_path = SCHEMA_DIR / "sample_schema.json"
    if schema_path.exists():
        with open(schema_path) as f:
            sample_schema = json.load(f)
        validate(instance=sample_json, schema=sample_schema)

    with open(out_dir / "sample.json", "w") as f:
        json.dump(sample_json, f, indent=2)

    return sample_json


def main(n_samples: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    samples_index: List[str] = []
    for i in tqdm(range(n_samples), desc="Generating samples"):
        sample = generate_sample(i, rng)
        samples_index.append(sample["sample_id"])

    # Root dataset.json
    dataset_json = {
        "dataset_name": "Dataset 3: Experimental Validation Dataset (Reality Check)",
        "version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": str(Path(__file__).relative_to("/workspace")),
            "commit": os.environ.get("GIT_COMMIT", "unknown")
        },
        "samples": samples_index
    }
    dataset_schema_path = SCHEMA_DIR / "dataset_schema.json"
    if dataset_schema_path.exists():
        with open(dataset_schema_path) as f:
            ds_schema = json.load(f)
        validate(instance=dataset_json, schema=ds_schema)

    with open(DATASET_ROOT / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"Wrote {len(samples_index)} samples to {SAMPLES_DIR}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Dataset 3 synthetic experimental validation data")
    parser.add_argument("--n", type=int, default=30, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    main(n_samples=args.n, seed=args.seed)
