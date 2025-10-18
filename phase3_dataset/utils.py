import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_rng(seed_key: str) -> np.random.Generator:
    # Create deterministic seed from string
    h = hashlib.sha256(seed_key.encode("utf-8")).hexdigest()
    seed_int = int(h[:16], 16) % (2**32 - 1)
    return np.random.default_rng(seed_int)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_json(path: Path, data: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def sample_id(mix_id: str, temp_c: int, rep: int) -> str:
    return f"C28-{mix_id}-{temp_c}C-Furnace-Rep{rep}"


def fov_id(fov_idx: int) -> str:
    return f"FOV{fov_idx:02d}"


# 3D utilities for connected components (6-connectivity)
Neighbor3D = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
]


def connected_components_3d(binary: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    assert binary.ndim == 3
    zdim, ydim, xdim = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    sizes: List[int] = []

    # Use Python list as stack for DFS to avoid recursion limits
    for z in range(zdim):
        for y in range(ydim):
            for x in range(xdim):
                if binary[z, y, x] and labels[z, y, x] == 0:
                    current_label += 1
                    size = 0
                    stack = [(z, y, x)]
                    labels[z, y, x] = current_label
                    while stack:
                        cz, cy, cx = stack.pop()
                        size += 1
                        for dz, dy, dx in Neighbor3D:
                            nz, ny, nx = cz + dz, cy + dy, cx + dx
                            if 0 <= nz < zdim and 0 <= ny < ydim and 0 <= nx < xdim:
                                if binary[nz, ny, nx] and labels[nz, ny, nx] == 0:
                                    labels[nz, ny, nx] = current_label
                                    stack.append((nz, ny, nx))
                    sizes.append(size)
    return labels, sizes


def component_spans_axes(labels: np.ndarray, label_id: int) -> Tuple[bool, bool, bool]:
    # Check whether component touches both min and max boundary along each axis
    zdim, ydim, xdim = labels.shape
    spans = []
    for axis, dim in enumerate((zdim, ydim, xdim)):
        # positions where labels == label_id
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            spans.append(False)
            continue
        mins = coords[:, axis].min()
        maxs = coords[:, axis].max()
        spans.append(mins == 0 and maxs == dim - 1)
    return tuple(spans)  # type: ignore


# 2D connected components (4-connectivity) for SEM slices
Neighbor2D = [
    (1, 0), (-1, 0), (0, 1), (0, -1)
]


def connected_components_2d(binary: np.ndarray) -> Tuple[np.ndarray, List[int], List[Tuple[int, int, int, int]]]:
    assert binary.ndim == 2
    h, w = binary.shape
    labels = np.zeros_like(binary, dtype=np.int32)
    current_label = 0
    sizes: List[int] = []
    bboxes: List[Tuple[int, int, int, int]] = []  # (min_y, min_x, max_y, max_x)

    for y in range(h):
        for x in range(w):
            if binary[y, x] and labels[y, x] == 0:
                current_label += 1
                size = 0
                min_y = max_y = y
                min_x = max_x = x
                stack = [(y, x)]
                labels[y, x] = current_label
                while stack:
                    cy, cx = stack.pop()
                    size += 1
                    if cy < min_y: min_y = cy
                    if cy > max_y: max_y = cy
                    if cx < min_x: min_x = cx
                    if cx > max_x: max_x = cx
                    for dy, dx in Neighbor2D:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if binary[ny, nx] and labels[ny, nx] == 0:
                                labels[ny, nx] = current_label
                                stack.append((ny, nx))
                sizes.append(size)
                bboxes.append((min_y, min_x, max_y, max_x))
    return labels, sizes, bboxes


def equivalent_diameter_from_voxel_count_3d(voxel_count: int, voxel_size_um: float) -> float:
    # diameter of sphere with same volume (3D)
    # V = voxel_count * (voxel_size)^3
    # d = 2 * (3V/(4pi))^(1/3)
    v_um3 = voxel_count * (voxel_size_um ** 3)
    d_um = 2.0 * ((3.0 * v_um3) / (4.0 * math.pi)) ** (1.0 / 3.0)
    return d_um


def equivalent_diameter_from_pixel_count_2d(pixel_count: int, pixel_size_um: float) -> float:
    # diameter of circle with same area (2D)
    # A = pixel_count * (pixel_size)^2
    # d = 2 * sqrt(A/pi)
    a_um2 = pixel_count * (pixel_size_um ** 2)
    d_um = 2.0 * math.sqrt(a_um2 / math.pi)
    return d_um
