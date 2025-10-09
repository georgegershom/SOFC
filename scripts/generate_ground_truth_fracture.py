#!/usr/bin/env python3
import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageFilter
import tifffile as tiff
from scipy.ndimage import gaussian_filter, distance_transform_edt, rotate


@dataclass
class InSitu3DConfig:
    volume_shape: Tuple[int, int, int] = (96, 96, 64)  # (Z, Y, X)
    time_steps: int = 12
    crack_init_center: Tuple[float, float, float] = (48.0, 20.0, 10.0)
    crack_growth_rate: float = 4.0  # voxels per time step along X
    crack_thickness: float = 2.0
    crack_curvature: float = 0.02  # mild bending
    noise_sigma: float = 0.03
    blur_sigma: float = 0.6
    seed: int = 42


@dataclass
class ExSituSEMConfig:
    image_size: Tuple[int, int] = (768, 1024)
    num_images: int = 16
    crack_length_px: Tuple[int, int] = (200, 700)
    crack_width_px: Tuple[float, float] = (2.0, 8.0)
    pore_density: float = 0.002  # fraction of pixels with pores
    pore_size_px: Tuple[int, int] = (3, 11)
    blur_radius: float = 0.5
    noise_sigma: float = 8.0
    seed: int = 123


@dataclass
class MacroPerfConfig:
    total_hours: int = 1500
    sample_every_hours: int = 10
    baseline_voltage: float = 1.0
    ohmic_drop_rate: float = 0.00002
    crack_coupling_gain: float = 0.25
    measurement_noise: float = 0.002
    seed: int = 7


@dataclass
class DatasetConfig:
    out_root: str
    in_situ: InSitu3DConfig = InSitu3DConfig()
    ex_situ: ExSituSEMConfig = ExSituSEMConfig()
    macro: MacroPerfConfig = MacroPerfConfig()


# ---------- Helper geometry utilities ----------

def sigmoid(x: np.ndarray, s: float = 1.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x / s))


def generate_phase_field_crack(z: int, y: int, x: int, center, t_idx, cfg: InSitu3DConfig) -> np.ndarray:
    cz, cy, cx0 = center
    # crack front advances along +X with slight curvature in Z and Y
    advance = cfg.crack_growth_rate * t_idx
    cx = cx0 + advance
    # curvature: quadratic bowing in z and y
    grid_z, grid_y, grid_x = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing='ij')
    curve_offset = cfg.crack_curvature * ((grid_z - cz) ** 2 - (grid_y - cy) ** 2) / max(z, y)
    signed_dist = (grid_x - (cx + curve_offset))
    # use distance to a plane-like surface to define phase-field
    phase = sigmoid(-signed_dist, s=cfg.crack_thickness)
    return phase.astype(np.float32)


def simulate_tomography_from_phase(phase: np.ndarray, blur_sigma: float, noise_sigma: float) -> np.ndarray:
    # Map phase to attenuation: crack (phase~1) -> low density, solid (phase~0) -> high density
    base = 0.7 * (1.0 - phase) + 0.15
    vol = gaussian_filter(base, sigma=blur_sigma)
    noise = np.random.normal(0.0, noise_sigma, size=vol.shape).astype(np.float32)
    vol = np.clip(vol + noise, 0.0, 1.0)
    return vol.astype(np.float32)


# ---------- Ex-situ SEM-like image synthesis ----------

def draw_crack_line(img: np.ndarray, length: int, width: float, angle_deg: float, start: Tuple[int, int]):
    h, w = img.shape
    theta = math.radians(angle_deg)
    dx = math.cos(theta)
    dy = math.sin(theta)
    x0, y0 = start[1], start[0]

    for i in range(length):
        x = int(round(x0 + i * dx))
        y = int(round(y0 + i * dy))
        if 0 <= x < w and 0 <= y < h:
            rr = int(max(1, round(width)))
            y_min = max(0, y - rr)
            y_max = min(h, y + rr + 1)
            x_min = max(0, x - rr)
            x_max = min(w, x + rr + 1)
            img[y_min:y_max, x_min:x_max] = np.minimum(img[y_min:y_max, x_min:x_max], 50)


def add_pores(img: np.ndarray, density: float, size_range: Tuple[int, int], rng: np.random.RandomState):
    h, w = img.shape
    num = int(h * w * density)
    for _ in range(num):
        r = rng.randint(size_range[0], size_range[1] + 1)
        y = rng.randint(0, h)
        x = rng.randint(0, w)
        y0 = max(0, y - r)
        y1 = min(h, y + r)
        x0 = max(0, x - r)
        x1 = min(w, x + r)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (yy - y) ** 2 + (xx - x) ** 2 <= r ** 2
        img[y0:y1, x0:x1][mask] = np.minimum(img[y0:y1, x0:x1][mask], 80)


# ---------- Macroscopic coupling ----------

def compute_delamination_area_from_phase(phase: np.ndarray, threshold: float = 0.5) -> float:
    crack_voxels = (phase >= threshold).sum()
    return float(crack_voxels)


def generate_macro_performance(delam_area_ts: List[float], cfg: MacroPerfConfig):
    rng = np.random.RandomState(cfg.seed)
    times = np.arange(0, cfg.total_hours + 1, cfg.sample_every_hours)
    # interpolate delam area to match length of times
    area_idx = np.linspace(0, len(delam_area_ts) - 1, len(times))
    areas = np.interp(area_idx, np.arange(len(delam_area_ts)), delam_area_ts)

    # model: voltage = baseline - ohmic* t - gain * normalized_area + noise
    areas_norm = (areas - areas.min()) / max(1e-6, (areas.max() - areas.min()))
    voltage = (
        cfg.baseline_voltage
        - cfg.ohmic_drop_rate * times
        - cfg.crack_coupling_gain * areas_norm
    )
    voltage += rng.normal(0.0, cfg.measurement_noise, size=voltage.shape)
    return times, voltage, areas


# ---------- Main generation ----------

def generate_in_situ(cfg: InSitu3DConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)
    z, y, x = cfg.volume_shape
    delam_area_ts = []

    for t_idx in range(cfg.time_steps):
        phase = generate_phase_field_crack(z, y, x, cfg.crack_init_center, t_idx, cfg)
        # add mild spatial noise to phase to look more natural
        phase = np.clip(phase + rng.normal(0.0, cfg.noise_sigma, size=phase.shape).astype(np.float32), 0.0, 1.0)
        vol = simulate_tomography_from_phase(phase, cfg.blur_sigma, cfg.noise_sigma)
        # save as tiff stack and metadata
        tiff.imwrite(os.path.join(out_dir, f"phase_t{t_idx:03d}.tiff"), phase, photometric='minisblack')
        tiff.imwrite(os.path.join(out_dir, f"volume_t{t_idx:03d}.tiff"), vol, photometric='minisblack')
        delam_area = compute_delamination_area_from_phase(phase)
        delam_area_ts.append(delam_area)

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return delam_area_ts


def generate_ex_situ(cfg: ExSituSEMConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(cfg.seed)
    h, w = cfg.image_size
    annotations = []

    for i in range(cfg.num_images):
        img = np.full((h, w), 200, dtype=np.uint8)
        # background texture
        bg_noise = rng.normal(0, cfg.noise_sigma, size=(h, w)).astype(np.float32)
        img = np.clip(img.astype(np.float32) + bg_noise, 0, 255)

        # crack
        length = int(rng.randint(cfg.crack_length_px[0], cfg.crack_length_px[1] + 1))
        width = float(rng.uniform(cfg.crack_width_px[0], cfg.crack_width_px[1]))
        angle = float(rng.uniform(-50, 50))
        start_y = rng.randint(h // 6, 5 * h // 6)
        start_x = rng.randint(w // 6, w // 3)
        draw_crack_line(img, length=length, width=width, angle_deg=angle, start=(start_y, start_x))

        # pores
        add_pores(img, cfg.pore_density, cfg.pore_size_px, rng)

        # blur and contrast
        pil = Image.fromarray(img.astype(np.uint8))
        pil = pil.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))
        img = np.array(pil)

        out_path = os.path.join(out_dir, f"sem_{i:03d}.png")
        Image.fromarray(img).save(out_path)
        annotations.append({
            "filename": os.path.basename(out_path),
            "crack_length_px": length,
            "crack_width_px": width,
            "crack_angle_deg": angle,
        })

    with open(os.path.join(out_dir, "annotations.json"), "w") as f:
        json.dump({"config": asdict(cfg), "images": annotations}, f, indent=2)



def generate_macro(delam_area_ts: List[float], cfg: MacroPerfConfig, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    times, voltage, areas_interp = generate_macro_performance(delam_area_ts, cfg)
    # save csv
    csv_path = os.path.join(out_dir, "performance.csv")
    with open(csv_path, "w") as f:
        f.write("hours,voltage,delamination_area_interp\n")
        for t, v, a in zip(times, voltage, areas_interp):
            f.write(f"{int(t)},{v:.6f},{a:.3f}\n")
    # save metadata
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)



def main():
    out_root = os.path.join("data", "ground_truth_fracture")
    in_situ_dir = os.path.join(out_root, "in_situ_3d")
    ex_situ_dir = os.path.join(out_root, "ex_situ_sem")
    macro_dir = os.path.join(out_root, "macroscopic")

    cfg = DatasetConfig(out_root=out_root)

    os.makedirs(out_root, exist_ok=True)

    delam_area_ts = generate_in_situ(cfg.in_situ, in_situ_dir)
    generate_ex_situ(cfg.ex_situ, ex_situ_dir)
    generate_macro(delam_area_ts, cfg.macro, macro_dir)

    # write top-level manifest
    manifest = {
        "in_situ": asdict(cfg.in_situ),
        "ex_situ": asdict(cfg.ex_situ),
        "macro": asdict(cfg.macro),
        "paths": {
            "root": out_root,
            "in_situ_3d": in_situ_dir,
            "ex_situ_sem": ex_situ_dir,
            "macroscopic": macro_dir,
        }
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
