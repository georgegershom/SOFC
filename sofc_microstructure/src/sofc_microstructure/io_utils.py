import os
import json
from typing import Dict, Any
import numpy as np
import imageio.v3 as iio


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_manifest(manifest: Dict[str, Any], out_path: str) -> None:
    with open(out_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def save_npz(array: np.ndarray, out_path: str, key: str = 'volume') -> None:
    np.savez_compressed(out_path, **{key: array})


def write_slices_uint8(volume_uint8: np.ndarray, out_dir: str, plane: str = 'z') -> None:
    ensure_dir(out_dir)
    if plane not in {'z', 'y', 'x'}:
        raise ValueError("plane must be one of {'z','y','x'}")
    vol = volume_uint8
    if plane == 'z':
        num = vol.shape[0]
        for k in range(num):
            iio.imwrite(os.path.join(out_dir, f"slice_{k:04d}.png"), vol[k])
    elif plane == 'y':
        num = vol.shape[1]
        for k in range(num):
            iio.imwrite(os.path.join(out_dir, f"slice_{k:04d}.png"), vol[:, k, :])
    else:  # x
        num = vol.shape[2]
        for k in range(num):
            iio.imwrite(os.path.join(out_dir, f"slice_{k:04d}.png"), vol[:, :, k])


def make_preview_grid(volume_uint8: np.ndarray, out_path: str, num_slices: int = 16) -> None:
    z_count = volume_uint8.shape[0]
    indices = np.linspace(0, z_count - 1, num=min(num_slices, z_count), dtype=int)
    tiles = [volume_uint8[i] for i in indices]
    # make square-ish grid
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape
    grid = np.zeros((rows * h, cols * w), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    iio.imwrite(out_path, grid)
