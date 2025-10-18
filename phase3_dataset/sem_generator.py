import os
from typing import Dict, List, Tuple

import numpy as np

from .utils import ensure_dir, normalize01, save_png


def _make_base_sem_canvas(image_size: int = 768) -> np.ndarray:
    x = np.linspace(0, 1, image_size)
    xv, yv = np.meshgrid(x, x)
    # Base paste texture: low-frequency variations + fine noise
    low_freq = (
        0.6
        + 0.15 * np.sin(2 * np.pi * (xv * 1.2 + yv * 0.8))
        + 0.15 * np.cos(2 * np.pi * (xv * 0.9 - yv * 1.1))
    )
    fine = 0.1 * np.random.normal(0.0, 1.0, size=(image_size, image_size))
    paste = normalize01(low_freq + fine)
    # Convert to SEM-like grayscale: darker = pores/cracks, brighter = dense paste
    img = (paste * 200 + 30).astype(np.float32)
    return img


def _draw_disk(img: np.ndarray, center: Tuple[int, int], radius: int, intensity: float) -> None:
    h, w = img.shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    img[mask] = intensity


def _draw_ring(img: np.ndarray, center: Tuple[int, int], r_inner: int, r_outer: int, intensity: float) -> None:
    h, w = img.shape
    cy, cx = center
    y, x = np.ogrid[:h, :w]
    d2 = (y - cy) ** 2 + (x - cx) ** 2
    mask = (d2 >= r_inner ** 2) & (d2 <= r_outer ** 2)
    img[mask] = intensity


def _bresenham_line(y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
    points = []
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((y, x))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points


def _draw_crack(img: np.ndarray, yx0: Tuple[int, int], angle_rad: float, length: int, width: int, darkness: float) -> List[Tuple[int, int]]:
    h, w = img.shape
    y0, x0 = yx0
    y1 = int(y0 + length * np.sin(angle_rad))
    x1 = int(x0 + length * np.cos(angle_rad))
    line_pts = _bresenham_line(y0, x0, y1, x1)
    crack_pixels = []
    for (y, x) in line_pts:
        if 0 <= y < h and 0 <= x < w:
            rr0 = max(0, y - width)
            rr1 = min(h, y + width + 1)
            cc0 = max(0, x - width)
            cc1 = min(w, x + width + 1)
            img[rr0:rr1, cc0:cc1] = np.minimum(img[rr0:rr1, cc0:cc1], darkness)
            crack_pixels.append((y, x))
    return crack_pixels


def _generate_aggregates_and_itz(img: np.ndarray, num_agg: int, itz_thickness: int, itz_intensity: float) -> List[Tuple[int, int, int]]:
    h, w = img.shape
    aggregates: List[Tuple[int, int, int]] = []  # (cy, cx, r)
    for _ in range(num_agg):
        r = np.random.randint(max(12, h // 40), max(18, h // 20))
        cy = np.random.randint(r + 5, h - r - 5)
        cx = np.random.randint(r + 5, w - r - 5)
        # Aggregate core brighter (dense)
        _draw_disk(img, (cy, cx), r, intensity=220 + np.random.uniform(-10, 10))
        # ITZ ring slightly darker
        _draw_ring(img, (cy, cx), r, r + itz_thickness, intensity=150 + np.random.uniform(-10, 10))
        aggregates.append((cy, cx, r))
    return aggregates


def _apply_rubber_effects(img: np.ndarray, aggregates: List[Tuple[int, int, int]], specimen: str, temp_c: int) -> Tuple[float, float]:
    rubber_fraction = 0.0
    itz_degradation = 0.0
    if specimen != "rubber":
        return rubber_fraction, itz_degradation

    h, w = img.shape
    num_rubber = max(6, (h * w) // (128 * 128))
    for _ in range(num_rubber):
        r = np.random.randint(max(8, h // 64), max(14, h // 40))
        cy = np.random.randint(r + 5, h - r - 5)
        cx = np.random.randint(r + 5, w - r - 5)
        # Rubber particles: darker than paste
        _draw_disk(img, (cy, cx), r, intensity=90 + np.random.uniform(-10, 10))
        # ITZ around rubber: even darker, thickness depends on temp
        itz_t = max(2, int(2 + (temp_c / 200)))
        _draw_ring(img, (cy, cx), r, r + itz_t, intensity=80 + np.random.uniform(-10, 10))
        rubber_fraction += np.pi * r * r / (h * w)

        # At elevated temps, rubber melts -> voids; simulate by dark core for T>=300C
        if temp_c >= 300:
            void_r = int(r * (0.5 + 0.5 * min(1.0, (temp_c - 300) / 300)))
            _draw_disk(img, (cy, cx), max(2, void_r), intensity=40 + np.random.uniform(-5, 5))
            itz_degradation += 0.5
        else:
            itz_degradation += 0.2

    return rubber_fraction, itz_degradation


def _add_microcracks(img: np.ndarray, specimen: str, temp_c: int) -> Dict[str, float]:
    h, w = img.shape
    num_cracks_base = 50
    temp_factor = np.interp(temp_c, [20, 800], [0.2, 3.0])
    rubber_factor = 1.4 if specimen == "rubber" else 1.0
    num_cracks = int(num_cracks_base * temp_factor * rubber_factor)

    width_base = 1 + int(np.interp(temp_c, [20, 800], [0, 3]))
    cracks_info: List[Tuple[float, int]] = []  # (angle_rad, length)

    for _ in range(num_cracks):
        y0 = np.random.randint(0, h)
        x0 = np.random.randint(0, w)
        angle = np.random.uniform(0, np.pi)
        length = np.random.randint(h // 20, h // 6)
        width = max(1, int(np.random.normal(width_base, 1)))
        darkness = 30 + np.random.uniform(-10, 10)
        _draw_crack(img, (y0, x0), angle, length, width, darkness)
        cracks_info.append((angle, length))

    # Compute simple crack density and anisotropy metrics from generation parameters
    if len(cracks_info) == 0:
        return {"crack_density_mm_per_mm2": 0.0, "crack_orientation_anisotropy": 0.0}

    lengths = np.array([l for _, l in cracks_info], dtype=np.float32)
    angles = np.array([a for a, _ in cracks_info], dtype=np.float32)

    crack_density = float(np.sum(lengths) / (h * w))
    # Anisotropy: resultant vector magnitude of angles (2*theta for 180-periodicity)
    vx = np.mean(np.cos(2 * angles))
    vy = np.mean(np.sin(2 * angles))
    anisotropy = float(np.sqrt(vx * vx + vy * vy))

    return {
        "crack_density_mm_per_mm2": crack_density,
        "crack_orientation_anisotropy": anisotropy,
    }


def generate_sem_batch(output_dir: str, specimen: str, temp_c: int, replicates: int) -> List[Dict]:
    ensure_dir(output_dir)

    items: List[Dict] = []
    for i in range(replicates):
        img = _make_base_sem_canvas()
        aggregates = _generate_aggregates_and_itz(img, num_agg=12, itz_thickness=4, itz_intensity=160)
        rubber_fraction, itz_deg = _apply_rubber_effects(img, aggregates, specimen, temp_c)
        crack_metrics = _add_microcracks(img, specimen, temp_c)

        # Normalize and add subtle contrast scaling with temperature
        img = img + np.random.uniform(-5, 5)
        img = np.clip(img, 0, 255).astype(np.uint8)

        fname = f"sem_{specimen}_{temp_c}C_rep{i+1}.png"
        fpath = os.path.join(output_dir, fname)
        save_png(img, fpath)

        item = {
            "modality": "SEM",
            "specimen": specimen,
            "temperature_c": temp_c,
            "replicate": i + 1,
            "image_path": fpath,
            "metrics": {
                "rubber_projected_area_fraction": float(rubber_fraction),
                "itz_degradation_index": float(itz_deg),
                **crack_metrics,
            },
            "notes": "Synthetic SEM of ITZ focus; darker regions = pores/voids/cracks; aggregates bright; rubber darker and melts to voids at elevated T.",
        }
        items.append(item)

    return items
