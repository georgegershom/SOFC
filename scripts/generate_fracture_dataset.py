#!/usr/bin/env python3
"""
Synthetic "Ground Truth" Fracture Dataset Generator

Outputs three modalities:
1) In-situ 3D tomography time-series (phase-field + intensity volumes)
2) Ex-situ SEM-like images with crack masks
3) Macroscopic performance degradation CSV correlated with delamination growth

Dependencies: numpy, pillow

Example:
  python scripts/generate_fracture_dataset.py --out data/fracture_ground_truth \
      --nx 96 --ny 96 --nz 48 --timesteps 12 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


@dataclass
class InSituConfig:
    nx: int = 96
    ny: int = 96
    nz: int = 48
    timesteps: int = 12
    voxel_size_um: Tuple[float, float, float] = (5.0, 5.0, 5.0)  # (dz, dy, dx)
    crack_plane_z_fraction: float = 0.55  # location of interface plane (0..1)
    crack_half_thickness_vox: int = 2  # delamination "gap" half-thickness


@dataclass
class ExSituConfig:
    num_images: int = 24
    width: int = 768
    height: int = 768
    pixel_size_um: float = 0.5


@dataclass
class PerformanceConfig:
    hours_between_in_situ: float = 50.0
    samples_per_in_situ_interval: int = 6  # macroscopic sampling resolution within an interval
    initial_voltage_v: float = 0.8
    current_a: float = 2.0
    base_resistance_ohm: float = 0.08
    # effect of delamination area on resistance (ohm per mm^2)
    resistance_per_mm2: float = 0.015
    voltage_drift_per_hour: float = 1e-4
    voltage_noise_std: float = 0.002


@dataclass
class DatasetConfig:
    in_situ: InSituConfig = field(default_factory=InSituConfig)
    ex_situ: ExSituConfig = field(default_factory=ExSituConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


# ---------------------------- Utility filters/noise ----------------------------

def _pad_reflect(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (radius, radius)
    return np.pad(arr, pad_width, mode="reflect")


def uniform_filter_1d(arr: np.ndarray, radius: int, axis: int) -> np.ndarray:
    if radius <= 0:
        return arr
    padded = _pad_reflect(arr, radius, axis)
    # cumulative sum over axis with a leading zero to preserve original length
    csum = np.cumsum(padded, axis=axis)
    # prepend a zero slice along the axis
    zero_shape = list(csum.shape)
    zero_shape[axis] = 1
    zeros = np.zeros(zero_shape, dtype=csum.dtype)
    csum0 = np.concatenate([zeros, csum], axis=axis)
    # window size
    win = 2 * radius + 1
    # moving sum using cs[i+win]-cs[i], now yielding original length
    slicer_hi = [slice(None)] * csum0.ndim
    slicer_lo = [slice(None)] * csum0.ndim
    slicer_hi[axis] = slice(win, None)
    slicer_lo[axis] = slice(0, -win)
    moving_sum = csum0[tuple(slicer_hi)] - csum0[tuple(slicer_lo)]
    return moving_sum / float(win)


def uniform_filter_nd(arr: np.ndarray, radius: int) -> np.ndarray:
    out = arr
    for ax in range(arr.ndim):
        out = uniform_filter_1d(out, radius=radius, axis=ax)
    return out


def fbm_noise(shape: Tuple[int, ...], rng: np.random.Generator, scales: List[int], weights: List[float]) -> np.ndarray:
    assert len(scales) == len(weights)
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    noise = np.zeros(shape, dtype=np.float32)
    for s, w in zip(scales, weights):
        base = rng.random(shape, dtype=np.float32)
        smoothed = uniform_filter_nd(base, radius=max(1, s))
        noise += w * smoothed
    # normalize 0..1
    noise -= noise.min()
    denom = noise.max() - noise.min() + 1e-8
    noise /= denom
    return noise.astype(np.float32)


# ---------------------------- In-situ generation ----------------------------

def generate_delamination_masks_2d(nx: int, ny: int, timesteps: int, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate a series of 2D delamination masks growing over time.

    We use anisotropic elliptical growth with angular roughness controlled by a few harmonics.
    """
    cx = int(0.25 * nx + rng.integers(-nx * 0.05, nx * 0.05))
    cy = int(0.55 * ny + rng.integers(-ny * 0.05, ny * 0.05))

    # Base radii and per-step growth velocities
    rx0 = max(4, int(0.06 * nx))
    ry0 = max(4, int(0.05 * ny))
    vx = max(1.0, 0.035 * nx)
    vy = max(1.0, 0.028 * ny)

    # Angular roughness defined by few Fourier modes
    n_modes = 4
    modes = rng.integers(2, 9, size=n_modes)
    phases = rng.random(n_modes) * 2 * math.pi
    amps = rng.uniform(0.02, 0.12, size=n_modes)

    yy, xx = np.mgrid[0:ny, 0:nx]
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx)

    masks: List[np.ndarray] = []
    for t in range(timesteps):
        rx_t = rx0 + vx * t
        ry_t = ry0 + vy * t
        rough = np.ones_like(theta, dtype=np.float32)
        for k in range(n_modes):
            rough += amps[k] * np.sin(modes[k] * theta + phases[k] + 0.15 * t)
        # constrain roughness multiplier
        rough = np.clip(rough, 0.7, 1.3)
        nxn = dx / (rx_t * rough)
        nyn = dy / (ry_t * rough)
        mask = (nxn * nxn + nyn * nyn) <= 1.0
        masks.append(mask.astype(np.uint8))
    # ensure monotonic growth
    for t in range(1, timesteps):
        masks[t] = np.maximum(masks[t], masks[t - 1])
    return masks


def generate_in_situ_series(cfg: InSituConfig, out_dir: str, rng: np.random.Generator) -> Dict:
    in_situ_dir = os.path.join(out_dir, "in_situ")
    pf_dir = os.path.join(in_situ_dir, "phase_field")
    inten_dir = os.path.join(in_situ_dir, "intensity")
    os.makedirs(pf_dir, exist_ok=True)
    os.makedirs(inten_dir, exist_ok=True)

    # Base microstructure 3D noise (static over time for simplicity)
    scales = [2, 5, 12]
    weights = [0.5, 0.35, 0.15]
    micro = fbm_noise((cfg.nz, cfg.ny, cfg.nx), rng, scales, weights)
    # Normalize to intensity ~ [0.25, 0.9]
    base_intensity = 0.25 + 0.65 * micro

    # Generate 2D delamination series on a plane
    masks2d = generate_delamination_masks_2d(cfg.nx, cfg.ny, cfg.timesteps, rng)
    z0 = int(cfg.crack_plane_z_fraction * cfg.nz)
    half_th = int(cfg.crack_half_thickness_vox)
    z_lo = max(0, z0 - half_th)
    z_hi = min(cfg.nz - 1, z0 + half_th)

    voxel_size_um = cfg.voxel_size_um

    volumes_meta: List[Dict] = []

    for t in range(cfg.timesteps):
        # Phase-field in 3D: copy the 2D mask into thickness around plane and add soft edges
        phase_field = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)
        mask2d = masks2d[t].astype(bool)
        for z in range(z_lo, z_hi + 1):
            # add soft decay away from mid-plane
            dist = abs(z - z0) / max(1, half_th)
            weight = float(max(0.0, 1.0 - 0.6 * dist))
            phase_field[z, mask2d] = weight
        # small background noise
        phase_field += 0.03 * rng.standard_normal(phase_field.shape).astype(np.float32)
        phase_field = np.clip(phase_field, 0.0, 1.0)

        # Intensity volume: darken where cracked
        intensity = base_intensity.copy()
        crack_darkening = 0.18 + 0.55 * float(t + 1) / cfg.timesteps  # deeper contrast as crack grows
        cracked = phase_field > 0.2
        intensity[cracked] = np.clip(intensity[cracked] * (1.0 - crack_darkening), 0.02, 1.0)
        # add sensor noise
        intensity += 0.01 * rng.standard_normal(intensity.shape).astype(np.float32)
        intensity = np.clip(intensity, 0.0, 1.0)

        # Save arrays
        pf_path = os.path.join(pf_dir, f"phase_field_t{t:03d}.npy")
        inten_path = os.path.join(inten_dir, f"intensity_t{t:03d}.npy")
        np.save(pf_path, phase_field.astype(np.float32))
        # save intensity as uint16 for realism
        inten_u16 = (intensity * 65535.0).astype(np.uint16)
        np.save(inten_path, inten_u16)

        volumes_meta.append({
            "t_index": t,
            "phase_field_path": os.path.relpath(pf_path, out_dir),
            "intensity_path": os.path.relpath(inten_path, out_dir),
        })

    # Compute delamination area per time step in mm^2 (mask at z0 plane)
    voxel_area_mm2 = (voxel_size_um[1] * voxel_size_um[2]) * 1e-6  # um^2 to mm^2
    areas_mm2 = []
    for t in range(cfg.timesteps):
        mask2d = masks2d[t].astype(bool)
        area = mask2d.sum() * voxel_area_mm2
        areas_mm2.append(float(area))

    return {
        "volumes": volumes_meta,
        "z_interface_index": z0,
        "voxel_size_um": {
            "z": float(voxel_size_um[0]),
            "y": float(voxel_size_um[1]),
            "x": float(voxel_size_um[2]),
        },
        "areas_mm2": areas_mm2,
        "timesteps": cfg.timesteps,
        "shape": {
            "z": cfg.nz,
            "y": cfg.ny,
            "x": cfg.nx,
        },
    }


# ---------------------------- Ex-situ generation ----------------------------

def draw_polyline_mask(mask: np.ndarray, points: List[Tuple[float, float]], radius: float) -> None:
    """Stamp disks along polyline to approximate a thick anti-aliased crack path."""
    h, w = mask.shape
    rr = int(max(1, round(radius)))
    yy, xx = np.ogrid[-rr:rr + 1, -rr:rr + 1]
    disk = (xx * xx + yy * yy) <= rr * rr

    def stamp(xc: float, yc: float):
        xi = int(round(xc))
        yi = int(round(yc))
        # Intended overlay region in image coordinates
        x0_img = max(0, min(w, xi - rr))
        y0_img = max(0, min(h, yi - rr))
        x1_img = max(0, min(w, xi + rr + 1))
        y1_img = max(0, min(h, yi + rr + 1))
        if x0_img >= x1_img or y0_img >= y1_img:
            return  # no overlap

        # Corresponding region in disk coordinates before clamping
        x0_disk = x0_img - (xi - rr)
        y0_disk = y0_img - (yi - rr)
        x1_disk = x1_img - (xi - rr)
        y1_disk = y1_img - (yi - rr)

        # Clamp disk region to valid range and adjust image region to match
        disk_h, disk_w = disk.shape
        x0_disk_c = max(0, min(disk_w, x0_disk))
        y0_disk_c = max(0, min(disk_h, y0_disk))
        x1_disk_c = max(0, min(disk_w, x1_disk))
        y1_disk_c = max(0, min(disk_h, y1_disk))

        # Compute how much we clipped on each side
        x_lo_clip = x0_disk_c - x0_disk
        y_lo_clip = y0_disk_c - y0_disk
        x_hi_clip = x1_disk - x1_disk_c
        y_hi_clip = y1_disk - y1_disk_c

        # Adjust image coordinates accordingly
        x0_img_c = int(x0_img + x_lo_clip)
        y0_img_c = int(y0_img + y_lo_clip)
        x1_img_c = int(x1_img - x_hi_clip)
        y1_img_c = int(y1_img - y_hi_clip)

        if x0_img_c >= x1_img_c or y0_img_c >= y1_img_c:
            return

        sub = disk[int(y0_disk_c):int(y1_disk_c), int(x0_disk_c):int(x1_disk_c)]
        # Combine with max to thicken path
        mask[y0_img_c:y1_img_c, x0_img_c:x1_img_c] = np.maximum(
            mask[y0_img_c:y1_img_c, x0_img_c:x1_img_c], sub.astype(mask.dtype)
        )

    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        length = max(1.0, math.hypot(x1 - x0, y1 - y0))
        steps = int(length / max(0.5, radius * 0.6))
        for s in range(steps + 1):
            t = s / max(1, steps)
            x = x0 + t * (x1 - x0)
            y = y0 + t * (y1 - y0)
            stamp(x, y)


def generate_sem_like_image(w: int, h: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    # Base microstructure texture via 2D fBm noise
    scales = [2, 5, 12, 24]
    weights = [0.45, 0.3, 0.2, 0.05]
    base = fbm_noise((h, w), rng, scales, weights)

    # Introduce pores by thresholding a shifted noise field
    pores = fbm_noise((h, w), rng, [3, 9, 18], [0.5, 0.35, 0.15])
    pore_thresh = 0.8 - 0.2 * rng.random()
    pore_mask = pores > pore_thresh

    # Crack path: main polyline across image with random turns
    num_pts = rng.integers(6, 12)
    pts: List[Tuple[float, float]] = []
    margin = w * 0.05
    x = -margin
    y = rng.uniform(h * 0.2, h * 0.8)
    pts.append((x, y))
    for _ in range(num_pts - 2):
        x += rng.uniform(w * 0.05, w * 0.2)
        y += rng.uniform(-h * 0.15, h * 0.15)
        x = np.clip(x, -margin, w + margin)
        y = np.clip(y, 0, h - 1)
        pts.append((x, y))
    pts.append((w + margin, rng.uniform(h * 0.2, h * 0.8)))

    crack_radius = rng.uniform(1.5, 3.5)
    crack_mask = np.zeros((h, w), dtype=np.uint8)
    draw_polyline_mask(crack_mask, pts, radius=crack_radius)

    # Add side branches
    for _ in range(rng.integers(2, 6)):
        # pick a random point on main path
        idx = int(rng.integers(1, len(pts) - 1))
        x0, y0 = pts[idx]
        blen = rng.uniform(w * 0.05, w * 0.18)
        angle = rng.uniform(-math.pi / 1.5, math.pi / 1.5)
        x1 = x0 + blen * math.cos(angle)
        y1 = y0 + blen * math.sin(angle)
        branch_pts = [(x0, y0), (x1, y1)]
        draw_polyline_mask(crack_mask, branch_pts, radius=max(1.2, crack_radius * 0.7))

    # Build grayscale image: pores bright, crack dark, base mid-tone
    img = 0.15 + 0.75 * base
    img[pore_mask] = np.clip(img[pore_mask] + 0.15, 0.0, 1.0)
    img[crack_mask.astype(bool)] = np.clip(img[crack_mask.astype(bool)] - 0.5, 0.0, 1.0)

    # Small shading gradient to mimic SEM illumination
    gx = np.linspace(0.9, 1.05, w, dtype=np.float32)
    gy = np.linspace(1.03, 0.97, h, dtype=np.float32)
    shading = gy[:, None] * gx[None, :]
    img *= shading

    # add noise
    img += 0.02 * rng.standard_normal(img.shape).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    return img.astype(np.float32), crack_mask


def generate_ex_situ(cfg: ExSituConfig, out_dir: str, rng: np.random.Generator) -> Dict:
    ex_dir = os.path.join(out_dir, "ex_situ")
    img_dir = os.path.join(ex_dir, "images")
    msk_dir = os.path.join(ex_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    items: List[Dict] = []
    for i in range(cfg.num_images):
        img, crack_mask = generate_sem_like_image(cfg.width, cfg.height, rng)
        # 8-bit PNG
        img8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
        mask8 = (crack_mask > 0).astype(np.uint8) * 255
        img_p = Image.fromarray(img8, mode="L")
        mask_p = Image.fromarray(mask8, mode="L")
        img_path = os.path.join(img_dir, f"sem_{i:04d}.png")
        msk_path = os.path.join(msk_dir, f"sem_{i:04d}_mask.png")
        img_p.save(img_path)
        mask_p.save(msk_path)
        items.append({
            "image_path": os.path.relpath(img_path, out_dir),
            "mask_path": os.path.relpath(msk_path, out_dir),
            "pixel_size_um": cfg.pixel_size_um,
        })

    return {"samples": items, "pixel_size_um": cfg.pixel_size_um}


# ---------------------------- Performance generation ----------------------------

def generate_performance_csv(perf_cfg: PerformanceConfig, out_dir: str, areas_mm2: List[float], in_situ_timesteps: int) -> Dict:
    perf_dir = os.path.join(out_dir, "performance")
    os.makedirs(perf_dir, exist_ok=True)
    csv_path = os.path.join(perf_dir, "performance.csv")

    hours_between = perf_cfg.hours_between_in_situ
    samples = perf_cfg.samples_per_in_situ_interval

    # Build a time vector with samples per interval
    times_h: List[float] = []
    areas_interp: List[float] = []
    for t_idx in range(in_situ_timesteps):
        t0 = t_idx * hours_between
        t1 = (t_idx + 1) * hours_between
        a0 = areas_mm2[min(t_idx, len(areas_mm2) - 1)]
        a1 = areas_mm2[min(t_idx + 1, len(areas_mm2) - 1)] if t_idx + 1 < in_situ_timesteps else a0
        for s in range(samples):
            tau = s / samples
            times_h.append(t0 + tau * (t1 - t0))
            areas_interp.append((1 - tau) * a0 + tau * a1)
    # include final point at last imaging time
    times_h.append(in_situ_timesteps * hours_between)
    areas_interp.append(areas_mm2[-1])

    times_h = np.array(times_h, dtype=np.float32)
    areas_interp = np.array(areas_interp, dtype=np.float32)

    I = perf_cfg.current_a
    R0 = perf_cfg.base_resistance_ohm
    k = perf_cfg.resistance_per_mm2
    drift = perf_cfg.voltage_drift_per_hour

    # Assume nominal open-circuit voltage E such that initial V0 = E - I*R0
    V0 = perf_cfg.initial_voltage_v
    E = V0 + I * R0

    # Resistance increases with delamination area
    R_t = R0 + k * areas_interp

    # Voltage under constant current with aging drift and noise
    rng = np.random.default_rng(12345)
    V_t = E - I * R_t - drift * times_h + rng.normal(0.0, perf_cfg.voltage_noise_std, size=times_h.shape)
    V_t = np.clip(V_t, 0.2, E)  # avoid negative or >E

    # Write CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("time_hours,normalized_time,current_A,voltage_V,internal_resistance_Ohm,delamination_area_mm2\n")
        for th, a, v, r in zip(times_h, areas_interp, V_t, R_t):
            f.write(f"{th:.3f},{th/(in_situ_timesteps*hours_between):.6f},{I:.6f},{v:.6f},{r:.6f},{a:.6f}\n")

    return {
        "csv_path": os.path.relpath(csv_path, out_dir),
        "initial_voltage_v": V0,
        "current_a": I,
        "base_resistance_ohm": R0,
        "resistance_per_mm2": k,
        "voltage_drift_per_hour": drift,
        "num_rows": int(len(times_h)),
    }


# ---------------------------- Orchestration ----------------------------

def generate_dataset(cfg: DatasetConfig, out_dir: str, seed: int) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    in_situ_meta = generate_in_situ_series(cfg.in_situ, out_dir, rng)
    ex_situ_meta = generate_ex_situ(cfg.ex_situ, out_dir, rng)
    perf_meta = generate_performance_csv(cfg.performance, out_dir, in_situ_meta["areas_mm2"], in_situ_meta["timesteps"])

    manifest = {
        "modality": {
            "in_situ": in_situ_meta,
            "ex_situ": ex_situ_meta,
            "performance": perf_meta,
        },
        "config": {
            "in_situ": asdict(cfg.in_situ),
            "ex_situ": asdict(cfg.ex_situ),
            "performance": asdict(cfg.performance),
            "seed": seed,
        },
        "description": "Synthetic dataset approximating SOFC delamination crack evolution (phase-field proxy), SEM-like ex-situ images, and correlated macroscopic performance decay.",
        "notes": [
            "Phase-field here is a proxy [0..1] indicating delamination, centered on a single interface plane.",
            "Intensity volumes are normalized [0..1] and saved as uint16 .npy for realism.",
            "Ex-situ images include crack masks; intensities are 8-bit PNGs.",
            "Performance CSV models voltage decay via increasing internal resistance proportional to delamination area.",
        ],
        "paths_relative_to": os.path.abspath(out_dir),
    }

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic fracture ground-truth dataset")
    p.add_argument("--out", dest="out", type=str, default="data/fracture_ground_truth")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=96)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--timesteps", type=int, default=12)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--num-ex", dest="num_ex", type=int, default=24, help="number of ex-situ images")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(
        in_situ=InSituConfig(nx=args.nx, ny=args.ny, nz=args.nz, timesteps=args.timesteps),
        ex_situ=ExSituConfig(num_images=args.num_ex),
        performance=PerformanceConfig(),
    )
    manifest = generate_dataset(cfg, out_dir=args.out, seed=args.seed)
    print(json.dumps({
        "ok": True,
        "out": os.path.abspath(args.out),
        "in_situ_timesteps": cfg.in_situ.timesteps,
        "ex_situ_images": cfg.ex_situ.num_images,
        "performance_rows": manifest["modality"]["performance"]["num_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
