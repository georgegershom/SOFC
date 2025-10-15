from __future__ import annotations

import json
import os
from typing import Dict, Tuple

import random

try:
    import numpy as np
except Exception:
    np = None  # type: ignore


PHASES = {
    "anode": {"labels": {0: "pore", 1: "Ni", 2: "YSZ"}},
    "cathode": {"labels": {0: "pore", 1: "LSCF", 2: "GDC"}},
}


def _ensure_numpy():
    if np is None:
        raise RuntimeError("numpy not available; install numpy and scikit-image")


def synthesize_volume(seed: int, size: Tuple[int, int, int] = (64, 64, 64), phase: str = "anode"):
    _ensure_numpy()
    random.seed(seed)
    np.random.seed(seed)

    # Generate a Gaussian random field and threshold for phase separation
    grid = np.random.normal(0, 1, size)
    # Smooth-like effect via Fourier domain filtering (very rough, avoids scipy)
    # Create low-pass mask
    kx = np.fft.fftfreq(size[0])[:, None, None]
    ky = np.fft.fftfreq(size[1])[None, :, None]
    kz = np.fft.fftfreq(size[2])[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    mask = np.exp(-(k2) / (0.02))
    G = np.fft.fftn(grid)
    Gf = G * mask
    smooth = np.real(np.fft.ifftn(Gf))

    # Map to three phases by quantiles
    q1 = np.quantile(smooth, 1/3)
    q2 = np.quantile(smooth, 2/3)
    vol = np.zeros(size, dtype=np.uint8)
    vol[smooth < q1] = 0  # pore
    vol[(smooth >= q1) & (smooth < q2)] = 1  # phase 1 (metal/LSCF)
    vol[smooth >= q2] = 2  # phase 2 (YSZ/GDC)

    return vol


def volume_metrics(vol) -> Dict[str, float]:
    _ensure_numpy()
    total = vol.size
    metrics: Dict[str, float] = {}
    for label in [0, 1, 2]:
        frac = float((vol == label).sum()) / float(total)
        metrics[f"phase_{label}_fraction"] = frac
    # Approximate specific surface area via gradient magnitude counting interfaces
    gx = np.zeros_like(vol, dtype=float)
    gy = np.zeros_like(vol, dtype=float)
    gz = np.zeros_like(vol, dtype=float)
    gx[1:, :, :] = (vol[1:, :, :] != vol[:-1, :, :])
    gy[:, 1:, :] = (vol[:, 1:, :] != vol[:, :-1, :])
    gz[:, :, 1:] = (vol[:, :, 1:] != vol[:, :, :-1])
    interfaces = gx.sum() + gy.sum() + gz.sum()
    metrics["approx_specific_surface_area"] = float(interfaces) / float(total)
    return metrics


def save_volume(vol, out_path: str):
    _ensure_numpy()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Save as numpy .npy for compactness
    np.save(out_path, vol)


def fabricate_set(out_dir: str, n: int = 20, seed: int = 123):
    _ensure_numpy()
    os.makedirs(out_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    meta = {"files": []}
    for i in range(n):
        vol = synthesize_volume(seed + i)
        m = volume_metrics(vol)
        fstem = f"micro_{i:04d}"
        fpath = os.path.join(out_dir, fstem + ".npy")
        save_volume(vol, fpath)
        meta["files"].append({
            "file": os.path.basename(fpath),
            "metrics": m,
            "labels": PHASES["anode"]["labels"],
            "voxel_size_um": 0.2,
        })

    with open(os.path.join(out_dir, "microstructure_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
