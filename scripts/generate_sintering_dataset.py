#!/usr/bin/env python3
"""
Generate physics-inspired synthetic datasets for sintering-induced residuals
for ANN and PINN training.

Outputs structure (created under --out):

out/
  README.md
  train/
    features.csv
    labels.csv
    fields/
      sample_<id>.npz  (subset of training samples, configurable fraction)
      index.csv        (maps sample_id -> npz path)
  test/
    features.csv
    labels.csv
  val/
    metadata.csv       (id, paths to DIC/XRD maps)
    ground_truth.csv   (id, true scalars for validation)
    dic/
      dic_<id>_strain_map.npy   (downsampled, noisy surface strain map)
    xrd/
      xrd_<id>_stress_map.npy   (noisy residual stress map in MPa)

Features include: temperature, cooling rate, TEC mismatch, porosity, elastic
properties, and derived stress/strain summaries. Labels include stress hotspot
fraction, crack initiation risk, and delamination probability.

This script avoids heavyweight dependencies; only numpy and pandas are used.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd


# ------------------------- Math helpers -------------------------

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def generate_correlated_field(
    grid_size: int,
    rng: np.random.Generator,
    num_modes: int = 5,
    base_amplitude: float = 1.0,
) -> np.ndarray:
    """Builds a smooth, correlated field via a few random Fourier modes.

    The field is normalized to zero mean and unit std before scaling.
    """
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    y = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    field = np.zeros((grid_size, grid_size), dtype=np.float64)
    for _ in range(num_modes):
        fx = rng.uniform(1.0, 4.0)
        fy = rng.uniform(1.0, 4.0)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        amp = rng.uniform(0.5, 1.0)
        field += amp * np.sin(2.0 * math.pi * (fx * X + fy * Y) + phase)

    # Normalize to zero mean, unit std, then scale
    mean = field.mean()
    std = field.std() if field.std() > 1e-12 else 1.0
    field = (field - mean) / std
    return base_amplitude * field


def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample by block averaging with integer factor."""
    h, w = arr.shape
    assert h % factor == 0 and w % factor == 0, "array dims must be divisible by factor"
    return (
        arr.reshape(h // factor, factor, w // factor, factor)
        .mean(axis=(1, 3))
        .astype(arr.dtype)
    )


# ------------------------- Parameter models -------------------------

@dataclass
class SampleParams:
    temperature_C: np.ndarray
    cooling_rate_C_per_min: np.ndarray
    delta_alpha_per_K: np.ndarray
    dwell_hr: np.ndarray
    porosity_initial: np.ndarray
    porosity_final: np.ndarray
    e_coating_pa: np.ndarray
    e_substrate_pa: np.ndarray
    nu_coating: np.ndarray
    nu_substrate: np.ndarray
    thickness_coating_m: np.ndarray
    thickness_substrate_m: np.ndarray
    kic_mpa_sqrtm: np.ndarray
    flaw_size_m: np.ndarray


@dataclass
class DerivedParams:
    epsilon_mismatch: np.ndarray
    e_effective_pa: np.ndarray
    sigma_interface_pa: np.ndarray
    sigma0_pa: np.ndarray


def sample_parameters(n: int, rng: np.random.Generator) -> SampleParams:
    """Sample base process/material parameters for n samples."""
    temperature_C = rng.uniform(1200.0, 1500.0, size=n)
    cooling_rate_C_per_min = rng.uniform(1.0, 10.0, size=n)
    delta_alpha_per_K = np.full(n, 2.3e-6, dtype=np.float64)
    dwell_hr = rng.uniform(0.5, 3.0, size=n)

    porosity_initial = rng.uniform(0.10, 0.30, size=n)

    # Simple densification model: higher T and time => lower final porosity
    # k depends on T; use a bounded linear mapping in [1200, 1500] C
    k = 0.6 * np.clip((temperature_C - 1200.0) / 300.0, 0.0, 1.5)
    porosity_final = porosity_initial * np.exp(-k * dwell_hr)
    porosity_final = np.clip(porosity_final, 0.01, 0.35)

    # Elastic properties
    e_coating_pa = rng.uniform(120.0, 220.0, size=n) * 1e9
    e_substrate_pa = rng.uniform(160.0, 320.0, size=n) * 1e9
    nu_coating = rng.uniform(0.22, 0.30, size=n)
    nu_substrate = rng.uniform(0.24, 0.33, size=n)

    thickness_coating_m = rng.uniform(50.0, 500.0, size=n) * 1e-6
    thickness_substrate_m = rng.uniform(500.0, 5000.0, size=n) * 1e-6

    kic_mpa_sqrtm = rng.uniform(1.0, 4.0, size=n)
    flaw_size_m = rng.uniform(1e-6, 20e-6, size=n)

    return SampleParams(
        temperature_C=temperature_C,
        cooling_rate_C_per_min=cooling_rate_C_per_min,
        delta_alpha_per_K=delta_alpha_per_K,
        dwell_hr=dwell_hr,
        porosity_initial=porosity_initial,
        porosity_final=porosity_final,
        e_coating_pa=e_coating_pa,
        e_substrate_pa=e_substrate_pa,
        nu_coating=nu_coating,
        nu_substrate=nu_substrate,
        thickness_coating_m=thickness_coating_m,
        thickness_substrate_m=thickness_substrate_m,
        kic_mpa_sqrtm=kic_mpa_sqrtm,
        flaw_size_m=flaw_size_m,
    )


def derive_parameters(p: SampleParams, rng: np.random.Generator) -> DerivedParams:
    """Compute mismatch strain and interface stress based on sampled params."""
    # Thermal drop to ambient ~25 C
    delta_T = (p.temperature_C - 25.0)
    epsilon_mismatch = p.delta_alpha_per_K * delta_T

    # Effective modulus degraded by porosity
    porosity_factor = np.clip(1.0 - 1.5 * p.porosity_final, 0.1, 1.0)
    e_effective_pa = p.e_coating_pa * porosity_factor

    # Interface stress scales with mismatch, modulus, and cooling rate
    rate_factor = 1.0 + 0.15 * (p.cooling_rate_C_per_min - 1.0) / 9.0
    sigma_interface_pa = e_effective_pa * epsilon_mismatch / (1.0 - p.nu_coating)
    sigma_interface_pa = sigma_interface_pa * rate_factor

    # Amplify with porosity to get characteristic field amplitude
    sigma0_pa = sigma_interface_pa * (1.0 + 0.6 * (p.porosity_final / 0.30))

    return DerivedParams(
        epsilon_mismatch=epsilon_mismatch,
        e_effective_pa=e_effective_pa,
        sigma_interface_pa=sigma_interface_pa,
        sigma0_pa=sigma0_pa,
    )


def approximate_hotspot_fraction(
    p: SampleParams, d: DerivedParams, rng: np.random.Generator
) -> np.ndarray:
    """A rough analytic estimate for hotspot fraction without fields.

    Baseline is ~0.07 (equivalent to > mean + 1.5 std for normal data),
    adjusted by porosity, cooling rate, and interface stress magnitude.
    """
    base = np.full_like(p.porosity_final, 0.07, dtype=np.float64)
    porosity_adj = 0.20 * (p.porosity_final - 0.15) / 0.15
    rate_adj = 0.05 * (p.cooling_rate_C_per_min - 1.0) / 9.0
    stress_norm = np.clip(d.sigma_interface_pa / 200e6, 0.0, 2.0)  # normalized to 200 MPa
    stress_adj = 0.10 * (stress_norm - 0.5)
    noise = 0.01 * rng.normal(size=p.porosity_final.shape)
    frac = base + porosity_adj + rate_adj + stress_adj + noise
    return np.clip(frac, 0.0, 1.0)


def compute_crack_risk(
    max_sigma_mpa: np.ndarray,
    kic_mpa_sqrtm: np.ndarray,
    flaw_size_m: np.ndarray,
    k_shape: float = 5.0,
) -> np.ndarray:
    """Crack initiation risk via Griffith criterion surrogate.

    sigma_c = K_IC / sqrt(pi a); risk = sigmoid(k * (sigma_max/sigma_c - 1)).
    """
    sigma_c_mpa = kic_mpa_sqrtm / np.sqrt(np.pi * flaw_size_m)
    ratio = np.clip(max_sigma_mpa / np.maximum(sigma_c_mpa, 1e-6), 0.0, 10.0)
    return sigmoid(k_shape * (ratio - 1.0))


def compute_delam_probability(
    sigma_interface_pa: np.ndarray,
    thickness_coating_m: np.ndarray,
    e_effective_pa: np.ndarray,
    rng: np.random.Generator,
    k_shape: float = 6.0,
) -> np.ndarray:
    """Delamination probability via mode-II/peel energy release surrogate.

    G ~ sigma^2 * t / E; compare vs Gc in [5, 50] J/m^2.
    """
    G = (sigma_interface_pa ** 2) * thickness_coating_m / np.maximum(e_effective_pa, 1e6)
    Gc = rng.uniform(5.0, 50.0, size=G.shape)
    ratio = np.clip(G / np.maximum(Gc, 1e-6), 0.0, 20.0)
    return sigmoid(k_shape * (ratio - 1.0))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_readme(out_dir: str) -> None:
    contents = (
        "# Sintering Residuals Synthetic Dataset\n\n"
        "This folder contains physics-inspired synthetic data for ANN/PINN training.\n\n"
        "## Splits\n"
        "- train: features.csv, labels.csv, optional fields/ subset (npz per sample)\n"
        "- test: features.csv, labels.csv\n"
        "- val: metadata.csv, ground_truth.csv, DIC/XRD-style maps (npy)\n\n"
        "## Units\n"
        "- Temperatures: Celsius\n"
        "- Stresses: MPa (fields, summaries), Pa internally\n"
        "- Strain: dimensionless\n"
        "- Thickness: meters\n"
        "- KIC: MPa*sqrt(m)\n\n"
        "## Labels\n"
        "- stress_hotspot_fraction: fraction of pixels above mean + 1.5 std (or surrogate)\n"
        "- crack_initiation_risk: sigmoid surrogate of Griffith criterion\n"
        "- delamination_probability: sigmoid surrogate of energy release vs Gc\n\n"
        "## Validation data\n"
        "- DIC: downsampled, noisy surface strain maps\n"
        "- XRD: noisy residual stress maps (quantized)\n\n"
        "Generated with scripts/generate_sintering_dataset.py\n"
    )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(contents)


def build_fields_for_sample(
    grid_size: int,
    rng: np.random.Generator,
    sigma0_pa: float,
    e_effective_pa: float,
    epsilon_mismatch: float,
    temperature_C: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthesize temperature, stress, strain fields and hotspot map for one sample."""
    x = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    y = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Temperature field with small gradient
    t_grad_amp = rng.uniform(-15.0, 15.0)
    temperature_field = temperature_C + t_grad_amp * (X - 0.5) + t_grad_amp * 0.6 * (Y - 0.5)

    # Stress field built from correlated modes + mild gradient + base level
    base_corr = generate_correlated_field(grid_size, rng, num_modes=6, base_amplitude=1.0)
    grad = (X - 0.5) * rng.uniform(-0.4, 0.4) + (Y - 0.5) * rng.uniform(-0.4, 0.4)
    field_unit = 0.6 * base_corr + 0.4 * grad
    field_unit = (field_unit - field_unit.mean()) / (field_unit.std() + 1e-12)
    stress_field_pa = (0.9 + 0.2 * rng.uniform()) * sigma0_pa * (1.0 + 0.35 * field_unit)
    stress_field_pa = np.clip(stress_field_pa, 1e5, None)

    # Strain from Hooke + mismatch baseline
    strain_field = stress_field_pa / max(e_effective_pa, 1e6) + epsilon_mismatch

    # Hotspot map relative to mean+1.5 std
    thr = stress_field_pa.mean() + 1.5 * stress_field_pa.std()
    hotspot_map = (stress_field_pa > thr).astype(np.uint8)

    return temperature_field, stress_field_pa, strain_field, hotspot_map


def save_npz_fields(
    path: str,
    temperature_C: np.ndarray,
    stress_pa: np.ndarray,
    strain: np.ndarray,
    hotspot_map: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        temperature_C=temperature_C.astype(np.float32),
        stress_MPa=(stress_pa.astype(np.float32) / 1e6),
        strain=strain.astype(np.float32),
        hotspot_map=hotspot_map.astype(np.uint8),
    )


def generate_split(
    split_name: str,
    n: int,
    out_dir: str,
    grid_size: int,
    train_field_fraction: float,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one split's features and labels, saving optional field subsets."""
    split_dir = os.path.join(out_dir, split_name)
    fields_dir = os.path.join(split_dir, "fields")
    ensure_dir(split_dir)

    # Decide on which samples to save fields (train only)
    save_fields_mask = np.zeros(n, dtype=bool)
    if split_name == "train" and train_field_fraction > 0.0:
        k = max(1, int(round(train_field_fraction * n)))
        save_indices = rng.choice(n, size=k, replace=False)
        save_fields_mask[save_indices] = True
        ensure_dir(fields_dir)

    # Sample parameters and derived
    params = sample_parameters(n, rng)
    derived = derive_parameters(params, rng)

    # Approximate summaries without generating fields for all
    sigma0_mpa = derived.sigma0_pa / 1e6
    mean_stress_mpa = (0.95 + 0.1 * rng.uniform(size=n)) * sigma0_mpa
    max_stress_mpa = (1.35 + 0.25 * rng.uniform(size=n)) * sigma0_mpa
    mean_strain = mean_stress_mpa * 1e6 / np.maximum(derived.e_effective_pa, 1e6) + derived.epsilon_mismatch

    hotspot_fraction = approximate_hotspot_fraction(params, derived, rng)
    crack_risk = compute_crack_risk(max_stress_mpa, params.kic_mpa_sqrtm, params.flaw_size_m, k_shape=5.0)
    delam_prob = compute_delam_probability(
        derived.sigma_interface_pa,
        params.thickness_coating_m,
        derived.e_effective_pa,
        rng,
        k_shape=6.0,
    )

    # If we are saving fields for some samples, recompute exact stats for those
    field_index_rows: List[Dict[str, object]] = []
    if save_fields_mask.any():
        for i in np.where(save_fields_mask)[0]:
            temp_C_f, stress_pa_f, strain_f, hotspot_map_f = build_fields_for_sample(
                grid_size=grid_size,
                rng=rng,
                sigma0_pa=float(derived.sigma0_pa[i]),
                e_effective_pa=float(derived.e_effective_pa[i]),
                epsilon_mismatch=float(derived.epsilon_mismatch[i]),
                temperature_C=float(params.temperature_C[i]),
            )
            # Update summaries with field-based values
            mean_stress_mpa[i] = stress_pa_f.mean() / 1e6
            max_stress_mpa[i] = stress_pa_f.max() / 1e6
            mean_strain[i] = strain_f.mean()
            thr = stress_pa_f.mean() + 1.5 * stress_pa_f.std()
            hotspot_fraction[i] = float((stress_pa_f > thr).mean())

            # Save npz
            sample_id = f"{split_name}_{i:06d}"
            npz_path = os.path.join(fields_dir, f"{sample_id}.npz")
            save_npz_fields(npz_path, temp_C_f, stress_pa_f, strain_f, hotspot_map_f)
            field_index_rows.append({
                "sample_id": sample_id,
                "npz_path": npz_path,
            })

        # Write index only if we saved any fields
        pd.DataFrame(field_index_rows).to_csv(
            os.path.join(fields_dir, "index.csv"), index=False
        )

    # Build features and labels tables
    features = pd.DataFrame({
        "sample_id": [f"{split_name}_{i:06d}" for i in range(n)],
        "T_sint_C": params.temperature_C,
        "cool_rate_C_per_min": params.cooling_rate_C_per_min,
        "delta_alpha_per_K": params.delta_alpha_per_K,
        "porosity_final": params.porosity_final,
        "E_coating_GPa": params.e_coating_pa / 1e9,
        "E_substrate_GPa": params.e_substrate_pa / 1e9,
        "nu_coating": params.nu_coating,
        "nu_substrate": params.nu_substrate,
        "thickness_coating_m": params.thickness_coating_m,
        "thickness_substrate_m": params.thickness_substrate_m,
        "KIC_MPa_sqrtm": params.kic_mpa_sqrtm,
        "epsilon_mismatch": derived.epsilon_mismatch,
        "sigma_interface_MPa": derived.sigma_interface_pa / 1e6,
        "mean_stress_MPa": mean_stress_mpa,
        "max_stress_MPa": max_stress_mpa,
        "mean_strain": mean_strain,
        "grid_size": np.full(n, grid_size, dtype=np.int32),
        "fields_saved": save_fields_mask.astype(np.int32),
    })

    labels = pd.DataFrame({
        "sample_id": features["sample_id"],
        "stress_hotspot_fraction": hotspot_fraction,
        "crack_initiation_risk": crack_risk,
        "delamination_probability": delam_prob,
    })

    # Persist CSVs
    features.to_csv(os.path.join(split_dir, "features.csv"), index=False)
    labels.to_csv(os.path.join(split_dir, "labels.csv"), index=False)

    return features, labels


def generate_validation(
    n_val: int,
    out_dir: str,
    grid_size: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate DIC/XRD-like validation maps with ground truth scalars."""
    val_dir = os.path.join(out_dir, "val")
    dic_dir = os.path.join(val_dir, "dic")
    xrd_dir = os.path.join(val_dir, "xrd")
    ensure_dir(dic_dir)
    ensure_dir(xrd_dir)

    params = sample_parameters(n_val, rng)
    derived = derive_parameters(params, rng)

    meta_rows: List[Dict[str, object]] = []
    truth_rows: List[Dict[str, object]] = []

    for i in range(n_val):
        temp_C_f, stress_pa_f, strain_f, hotspot_map_f = build_fields_for_sample(
            grid_size=grid_size,
            rng=rng,
            sigma0_pa=float(derived.sigma0_pa[i]),
            e_effective_pa=float(derived.e_effective_pa[i]),
            epsilon_mismatch=float(derived.epsilon_mismatch[i]),
            temperature_C=float(params.temperature_C[i]),
        )

        # DIC: block-average downsample + Gaussian-like noise
        dic_factor = 2
        dic_strain = block_reduce_mean(strain_f, dic_factor)
        dic_noise = rng.normal(loc=0.0, scale=3e-4, size=dic_strain.shape)
        dic_map = dic_strain + dic_noise

        # XRD: add measurement noise and quantize to 0.5 MPa
        xrd_stress_mpa = stress_pa_f / 1e6
        xrd_noise = rng.normal(loc=0.0, scale=5.0, size=xrd_stress_mpa.shape)  # MPa
        xrd_map = xrd_stress_mpa + xrd_noise
        xrd_map = np.round(xrd_map * 2.0) / 2.0  # 0.5 MPa quantization

        val_id = f"val_{i:05d}"
        dic_path = os.path.join(dic_dir, f"{val_id}_strain_map.npy")
        xrd_path = os.path.join(xrd_dir, f"{val_id}_stress_map.npy")
        np.save(dic_path, dic_map.astype(np.float32))
        np.save(xrd_path, xrd_map.astype(np.float32))

        # Ground truth scalars and labels
        mean_stress_mpa = float(stress_pa_f.mean() / 1e6)
        max_stress_mpa = float(stress_pa_f.max() / 1e6)
        mean_strain = float(strain_f.mean())
        thr = stress_pa_f.mean() + 1.5 * stress_pa_f.std()
        hotspot_fraction = float((stress_pa_f > thr).mean())
        crack_risk = float(
            compute_crack_risk(
                np.array([max_stress_mpa]),
                np.array([params.kic_mpa_sqrtm[i]]),
                np.array([params.flaw_size_m[i]]),
            )[0]
        )
        delam_prob = float(
            compute_delam_probability(
                np.array([derived.sigma_interface_pa[i]]),
                np.array([params.thickness_coating_m[i]]),
                np.array([derived.e_effective_pa[i]]),
                rng,
            )[0]
        )

        meta_rows.append({
            "val_id": val_id,
            "dic_strain_map_path": dic_path,
            "xrd_stress_map_path": xrd_path,
            "grid_size": grid_size,
        })
        truth_rows.append({
            "val_id": val_id,
            "T_sint_C": float(params.temperature_C[i]),
            "cool_rate_C_per_min": float(params.cooling_rate_C_per_min[i]),
            "delta_alpha_per_K": float(params.delta_alpha_per_K[i]),
            "porosity_final": float(params.porosity_final[i]),
            "E_coating_GPa": float(params.e_coating_pa[i] / 1e9),
            "E_substrate_GPa": float(params.e_substrate_pa[i] / 1e9),
            "nu_coating": float(params.nu_coating[i]),
            "nu_substrate": float(params.nu_substrate[i]),
            "thickness_coating_m": float(params.thickness_coating_m[i]),
            "thickness_substrate_m": float(params.thickness_substrate_m[i]),
            "KIC_MPa_sqrtm": float(params.kic_mpa_sqrtm[i]),
            "epsilon_mismatch": float(derived.epsilon_mismatch[i]),
            "sigma_interface_MPa": float(derived.sigma_interface_pa[i] / 1e6),
            "mean_stress_MPa": mean_stress_mpa,
            "max_stress_MPa": max_stress_mpa,
            "mean_strain": mean_strain,
            "stress_hotspot_fraction": hotspot_fraction,
            "crack_initiation_risk": crack_risk,
            "delamination_probability": delam_prob,
        })

    meta_df = pd.DataFrame(meta_rows)
    truth_df = pd.DataFrame(truth_rows)
    meta_df.to_csv(os.path.join(val_dir, "metadata.csv"), index=False)
    truth_df.to_csv(os.path.join(val_dir, "ground_truth.csv"), index=False)
    return meta_df, truth_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic sintering dataset")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=120)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--train-field-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    ensure_dir(args.out)
    write_readme(args.out)

    # Train split
    generate_split(
        split_name="train",
        n=args.n_train,
        out_dir=args.out,
        grid_size=args.grid_size,
        train_field_fraction=args.train_field_fraction,
        rng=rng,
    )

    # Test split (no fields by default)
    generate_split(
        split_name="test",
        n=args.n_test,
        out_dir=args.out,
        grid_size=args.grid_size,
        train_field_fraction=0.0,
        rng=rng,
    )

    # Validation maps
    generate_validation(
        n_val=args.n_val,
        out_dir=args.out,
        grid_size=args.grid_size,
        rng=rng,
    )

    print(
        f"Dataset written to {args.out} (train={args.n_train}, test={args.n_test}, val={args.n_val}, grid={args.grid_size})"
    )


if __name__ == "__main__":
    main()

