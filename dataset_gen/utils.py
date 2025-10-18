from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def ensure_dir(path: os.PathLike | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_pgm_grayscale(image: np.ndarray, out_path: os.PathLike | str) -> None:
    """
    Save a 2D uint8 NumPy array as a binary PGM (P5) file without external deps.
    """
    if image.dtype != np.uint8:
        raise ValueError("write_pgm_grayscale expects uint8 array")
    if image.ndim != 2:
        raise ValueError("write_pgm_grayscale expects 2D array")

    h, w = image.shape
    header = f"P5\n{w} {h}\n255\n".encode("ascii")
    data = image.tobytes()
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(data)


def upsample_repeat_2d(coarse: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """
    Upsample a 2D array to target_size by repeating values (nearest-neighbor).
    """
    h, w = coarse.shape
    th, tw = target_shape
    ry = math.ceil(th / h)
    rx = math.ceil(tw / w)
    up = np.kron(coarse, np.ones((ry, rx), dtype=coarse.dtype))
    return up[:th, :tw]


def upsample_repeat_3d(coarse: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    z, y, x = coarse.shape
    tz, ty, tx = target_shape
    rz = math.ceil(tz / z)
    ry = math.ceil(ty / y)
    rx = math.ceil(tx / x)
    up = np.kron(coarse, np.ones((rz, ry, rx), dtype=coarse.dtype))
    return up[:tz, :ty, :tx]


def fractal_noise_2d(shape: Tuple[int, int], octaves: int = 4, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = np.zeros((h, w), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for o in range(octaves):
        scale = 2 ** o
        ch = max(2, h // (8 * scale))
        cw = max(2, w // (8 * scale))
        coarse = rng.random((ch, cw), dtype=np.float32)
        up = upsample_repeat_2d(coarse, (h, w))
        # Simple separable smoothing with [1,2,1] kernel applied a few times
        for _ in range(2):
            up = smooth2d_weighted(up)
        noise += amplitude * up
        total_amp += amplitude
        amplitude *= 0.5
    noise /= max(total_amp, 1e-6)
    return noise


def fractal_noise_3d(shape: Tuple[int, int, int], octaves: int = 4, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z, y, x = shape
    noise = np.zeros((z, y, x), dtype=np.float32)
    amplitude = 1.0
    total_amp = 0.0
    for o in range(octaves):
        scale = 2 ** o
        cz = max(2, z // (8 * scale))
        cy = max(2, y // (8 * scale))
        cx = max(2, x // (8 * scale))
        coarse = rng.random((cz, cy, cx), dtype=np.float32)
        up = upsample_repeat_3d(coarse, (z, y, x))
        # Light smoothing
        for _ in range(2):
            up = smooth3d_weighted(up)
        noise += amplitude * up
        total_amp += amplitude
        amplitude *= 0.5
    noise /= max(total_amp, 1e-6)
    return noise


def smooth2d_weighted(arr: np.ndarray) -> np.ndarray:
    """Apply a simple separable [1,2,1]/4 smoothing in 2D without external deps."""
    arr = (arr.astype(np.float32, copy=False))
    # Horizontal
    padded = np.pad(arr, ((0, 0), (1, 1)), mode="edge")
    left = padded[:, :-2]
    center = padded[:, 1:-1]
    right = padded[:, 2:]
    arr = (left + 2 * center + right) * 0.25
    # Vertical
    padded = np.pad(arr, ((1, 1), (0, 0)), mode="edge")
    up = padded[:-2, :]
    center = padded[1:-1, :]
    down = padded[2:, :]
    arr = (up + 2 * center + down) * 0.25
    return arr


def smooth3d_weighted(arr: np.ndarray) -> np.ndarray:
    # Z axis
    arr = arr.astype(np.float32, copy=False)
    padded = np.pad(arr, ((1, 1), (0, 0), (0, 0)), mode="edge")
    a = padded[:-2, :, :]
    b = padded[1:-1, :, :]
    c = padded[2:, :, :]
    arr = (a + 2 * b + c) * 0.25
    # Y axis
    padded = np.pad(arr, ((0, 0), (1, 1), (0, 0)), mode="edge")
    a = padded[:, :-2, :]
    b = padded[:, 1:-1, :]
    c = padded[:, 2:, :]
    arr = (a + 2 * b + c) * 0.25
    # X axis
    padded = np.pad(arr, ((0, 0), (0, 0), (1, 1)), mode="edge")
    a = padded[:, :, :-2]
    b = padded[:, :, 1:-1]
    c = padded[:, :, 2:]
    arr = (a + 2 * b + c) * 0.25
    return arr


def draw_disks_mask(h: int, w: int, centers_radii: List[Tuple[float, float, float]]) -> np.ndarray:
    """Return mask with True inside any of the disks."""
    yy, xx = np.mgrid[0:h, 0:w]
    mask = np.zeros((h, w), dtype=bool)
    for cx, cy, r in centers_radii:
        mask |= ((xx - cx) ** 2 + (yy - cy) ** 2) <= (r ** 2)
    return mask


def draw_spheres_mask(z: int, y: int, x: int, centers_radii: List[Tuple[float, float, float]]) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    mask = np.zeros((z, y, x), dtype=bool)
    for cx, cy, cz, r in centers_radii:
        mask |= ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= (r ** 2)
    return mask


def distance_to_disks(h: int, w: int, centers_radii: List[Tuple[float, float, float]]) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.full((h, w), np.inf, dtype=np.float32)
    for cx, cy, r in centers_radii:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r
        d = np.minimum(d, dist)
    return d


def distance_to_spheres(z: int, y: int, x: int, centers_radii: List[Tuple[float, float, float]]) -> np.ndarray:
    zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
    d = np.full((z, y, x), np.inf, dtype=np.float32)
    for cx, cy, cz, r in centers_radii:
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) - r
        d = np.minimum(d, dist)
    return d


def draw_random_cracks_2d(h: int, w: int, density: float, seed: int | None = None) -> np.ndarray:
    """
    Generate a boolean mask of "cracks" as thin lines. Density in [0, ~0.3].
    """
    rng = np.random.default_rng(seed)
    mask = np.zeros((h, w), dtype=bool)
    num_lines = int(5 + density * 60)
    for _ in range(num_lines):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        angle = rng.random() * math.tau
        length = rng.integers(int(min(h, w) * 0.3), int(min(h, w) * (0.5 + density)))
        thickness = 1 + int(2 * density * rng.random())
        dx = math.cos(angle)
        dy = math.sin(angle)
        for t in range(length):
            xx = int(x0 + dx * t)
            yy = int(y0 + dy * t)
            if 0 <= xx < w and 0 <= yy < h:
                mask[max(0, yy - thickness):min(h, yy + thickness + 1), max(0, xx - 1):min(w, xx + 2)] = True
    return mask


def draw_random_cracks_3d(shape: Tuple[int, int, int], density: float, seed: int | None = None) -> np.ndarray:
    """
    Generate a boolean 3D mask of crack-like planar segments.
    """
    z, y, x = shape
    rng = np.random.default_rng(seed)
    mask = np.zeros(shape, dtype=bool)
    num_planes = int(2 + density * 25)
    for _ in range(num_planes):
        # Random plane: n·(X - p0) = 0; implement as a slab with small thickness
        nx, ny, nz = rng.normal(size=3)
        norm = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-6
        nx, ny, nz = nx / norm, ny / norm, nz / norm
        p0 = np.array([rng.uniform(0, x), rng.uniform(0, y), rng.uniform(0, z)], dtype=np.float32)
        thickness = 1.0 + 2.0 * density * rng.random()
        # compute signed distance for grid
        zz, yy, xx = np.mgrid[0:z, 0:y, 0:x]
        pts = np.stack([xx, yy, zz], axis=-1).astype(np.float32)
        # distance from plane
        d = np.abs((pts - p0) @ np.array([nx, ny, nz], dtype=np.float32))
        mask |= d < thickness
    return mask


def downsample_maxpool_3d(vol: np.ndarray, factor: int) -> np.ndarray:
    z, y, x = vol.shape
    tz = z // factor
    ty = y // factor
    tx = x // factor
    vol = vol[: tz * factor, : ty * factor, : tx * factor]
    vol = vol.reshape(tz, factor, ty, factor, tx, factor)
    vol = vol.max(axis=(1, 3, 5))
    return vol


def bfs_connected_fraction(binary: np.ndarray) -> float:
    """Compute fraction of voxels in the largest 6-connected component."""
    if binary.ndim != 3:
        raise ValueError("binary must be 3D")
    z, y, x = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    max_count = 0
    # Iterate over candidates sparsely to avoid worst-case time
    indices = np.argwhere(binary)
    if indices.size == 0:
        return 0.0
    # Limit BFS invocations by shuffling
    rng = np.random.default_rng(0)
    rng.shuffle(indices)
    for idx in indices:
        iz, iy, ix = idx
        if visited[iz, iy, ix]:
            continue
        # BFS
        stack = [(iz, iy, ix)]
        visited[iz, iy, ix] = True
        count = 0
        while stack:
            cz, cy, cx = stack.pop()
            count += 1
            for dz, dy, dx in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nz, ny, nx = cz + dz, cy + dy, cx + dx
                if 0 <= nz < z and 0 <= ny < y and 0 <= nx < x:
                    if binary[nz, ny, nx] and not visited[nz, ny, nx]:
                        visited[nz, ny, nx] = True
                        stack.append((nz, ny, nx))
        if count > max_count:
            max_count = count
        # Early exit if we already covered >50%
        if max_count / indices.shape[0] > 0.5:
            break
    return max_count / max(1, indices.shape[0])
