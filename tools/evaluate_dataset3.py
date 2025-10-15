#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict

import numpy as np

DATASET_ROOT = Path("/workspace/datasets/dataset3")


def load_sample(sample_id: str) -> Dict:
    sample_dir = DATASET_ROOT / "samples" / sample_id
    with open(sample_dir / "sample.json") as f:
        meta = json.load(f)
    warp = np.loadtxt(sample_dir / "warp_map.csv", delimiter=",", skiprows=1)
    # columns: x_mm, y_mm, z_m
    curvature = json.load(open(sample_dir / "curvature.json"))
    xrd = np.loadtxt(sample_dir / "xrd.csv", delimiter=",", skiprows=1)
    raman = np.loadtxt(sample_dir / "raman.csv", delimiter=",", skiprows=1)
    lr = np.loadtxt(sample_dir / "layer_removal" / "profile.csv", delimiter=",", skiprows=1)
    return {"meta": meta, "warp": warp, "curvature": curvature, "xrd": xrd, "raman": raman, "layer_removal": lr}


def naive_inverse_warp_to_stress(warp: np.ndarray) -> Dict[str, float]:
    # Fit quadratic sag z = 0.5 * (kx x^2 + ky y^2) + cxy + d
    x_m = warp[:, 0] * 1e-3
    y_m = warp[:, 1] * 1e-3
    z = warp[:, 2]
    A = np.column_stack([0.5 * x_m ** 2, 0.5 * y_m ** 2, x_m * y_m, np.ones_like(x_m)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    kx_est, ky_est, twist_est, _ = coeffs
    return {"kx": float(kx_est), "ky": float(ky_est), "twist": float(twist_est)}


def main():
    with open(DATASET_ROOT / "dataset.json") as f:
        ds = json.load(f)
    # Take first 5 samples as a smoke test
    for sid in ds["samples"][:5]:
        s = load_sample(sid)
        curv_pred = naive_inverse_warp_to_stress(s["warp"])
        curv_true = s["curvature"]
        err_kx = abs(curv_pred["kx"] - curv_true["kx_1_per_m"]) / max(1e-9, abs(curv_true["kx_1_per_m"]))
        err_ky = abs(curv_pred["ky"] - curv_true["ky_1_per_m"]) / max(1e-9, abs(curv_true["ky_1_per_m"]))
        print(f"{sid}: rel_err kx={err_kx:.2%}, ky={err_ky:.2%}")


if __name__ == "__main__":
    main()
