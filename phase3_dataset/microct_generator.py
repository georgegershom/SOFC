import os
from typing import Dict, List, Tuple

import numpy as np
import tifffile

from .config import CONFIG
from .utils import ensure_dir


def _generate_volume(shape: Tuple[int, int, int], specimen: str, temp_c: int) -> Tuple[np.ndarray, Dict[str, float]]:
    dz, dy, dx = shape
    vol = np.ones(shape, dtype=np.uint8)  # 1 = solid, 0 = void

    # Base porosity increases with temperature; rubber adds additional voids from melted particles
    base_pore_prob = float(np.interp(temp_c, [20, 800], [0.005, 0.08]))
    if specimen == "rubber":
        base_pore_prob *= 1.6

    noise = np.random.rand(*shape)
    vol[noise < base_pore_prob] = 0

    # Add cracks as planar faults with finite thickness
    num_faults = int(np.interp(temp_c, [20, 800], [2, 14]))
    if specimen == "rubber":
        num_faults = int(num_faults * 1.4)

    crack_voxels = 0
    for _ in range(num_faults):
        # Random planar faults; index using axis-specific slices to avoid boolean broadcast on 3D volume
        axis = np.random.choice([0, 1, 2])
        thickness = np.random.randint(1, 3 + int(np.interp(temp_c, [20, 800], [0, 3])))
        offset = np.random.randint(0, shape[axis])
        if axis == 0:
            z_mask = np.abs(np.arange(dz) - offset) <= thickness
            voxels = int(np.sum(z_mask)) * dy * dx
            vol[z_mask, :, :] = 0
        elif axis == 1:
            y_mask = np.abs(np.arange(dy) - offset) <= thickness
            voxels = dz * int(np.sum(y_mask)) * dx
            vol[:, y_mask, :] = 0
        else:
            x_mask = np.abs(np.arange(dx) - offset) <= thickness
            voxels = dz * dy * int(np.sum(x_mask))
            vol[:, :, x_mask] = 0
        crack_voxels += voxels

    # Rubber-derived spherical voids
    rubber_voids = 0
    if specimen == "rubber":
        num_voids = int(np.interp(temp_c, [20, 300, 600, 800], [2, 6, 18, 22]))
        for _ in range(num_voids):
            rz = np.random.randint(3, max(4, dz // 20))
            ry = np.random.randint(3, max(4, dy // 20))
            rx = np.random.randint(3, max(4, dx // 20))
            cz = np.random.randint(rz + 2, dz - rz - 2)
            cy = np.random.randint(ry + 2, dy - ry - 2)
            cx = np.random.randint(rx + 2, dx - rx - 2)
            z = np.arange(dz)[:, None, None]
            y = np.arange(dy)[None, :, None]
            x = np.arange(dx)[None, None, :]
            mask = (((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0
            rubber_voids += int(np.sum(mask))
            vol[mask] = 0

    total_voxels = vol.size
    void_voxels = int(total_voxels - int(np.sum(vol)))
    porosity = float(void_voxels) / float(total_voxels)
    crack_fraction = float(crack_voxels) / float(total_voxels)

    # Simple connectivity proxy: fraction of slices containing at least one void path across
    def spanning_fraction_along_axis(axis: int) -> float:
        if axis == 0:
            slices = [vol[z, :, :] for z in range(dz)]
        elif axis == 1:
            slices = [vol[:, y, :] for y in range(dy)]
        else:
            slices = [vol[:, :, x] for x in range(dx)]
        spans = 0
        for sl in slices:
            has_void = np.any(sl == 0)
            if has_void:
                spans += 1
        return spans / len(slices)

    connectivity_proxy = float(np.mean([spanning_fraction_along_axis(0), spanning_fraction_along_axis(1), spanning_fraction_along_axis(2)]))

    metrics = {
        "porosity": porosity,
        "crack_volume_fraction": crack_fraction,
        "connectivity_proxy": connectivity_proxy,
        "rubber_void_voxels": float(rubber_voids),
    }

    return vol, metrics


def generate_microct_batch(output_dir: str, specimen: str, temp_c: int, replicates: int) -> List[Dict]:
    ensure_dir(output_dir)
    items: List[Dict] = []

    shape = CONFIG.microct_volume_shape
    for i in range(replicates):
        vol, metrics = _generate_volume(shape, specimen, temp_c)
        base = f"microct_{specimen}_{temp_c}C_rep{i+1}"
        tiff_path = os.path.join(output_dir, base + ".tiff")
        tifffile.imwrite(tiff_path, vol.astype(np.uint8), photometric="minisblack")

        # Also save central orthogonal slices as quicklook PNGs
        zc, yc, xc = shape[0] // 2, shape[1] // 2, shape[2] // 2
        # Convert to 0-255 grayscale where void=0, solid=255
        z_slice = (vol[zc, :, :] * 255).astype(np.uint8)
        y_slice = (vol[:, yc, :] * 255).astype(np.uint8)
        x_slice = (vol[:, :, xc] * 255).astype(np.uint8)
        try:
            from PIL import Image

            Image.fromarray(z_slice).save(os.path.join(output_dir, base + "_sliceZ.png"))
            Image.fromarray(y_slice).save(os.path.join(output_dir, base + "_sliceY.png"))
            Image.fromarray(x_slice).save(os.path.join(output_dir, base + "_sliceX.png"))
        except Exception:
            pass

        item = {
            "modality": "MicroCT",
            "specimen": specimen,
            "temperature_c": temp_c,
            "replicate": i + 1,
            "tiff_path": tiff_path,
            "metrics": metrics,
            "notes": "Synthetic micro-CT volume: voids (0) and solid (1). Porosity and connectivity rise with temperature, more so for rubber mixes due to melted rubber voids and crack amplification.",
        }
        items.append(item)

    return items
