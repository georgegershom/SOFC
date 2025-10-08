from __future__ import annotations
import numpy as np
from skimage import measure
from typing import Dict


def compute_volume_fractions(labels_material: np.ndarray, labels_region: np.ndarray) -> Dict[str, float]:
    nz, ny, nx = labels_material.shape
    total = float(nz * ny * nx)
    vf = {}
    for name, value in [
        ("pore", 0),
        ("ni", 1),
        ("ysz", 2),
        ("electrolyte", 3),
    ]:
        vf[f"vf_{name}_total"] = float((labels_material == value).sum()) / total

    # regional porosity and solids
    region_names = {1: 'anode_bulk', 2: 'interlayer', 3: 'electrolyte'}
    for r_label, r_name in region_names.items():
        region_mask = labels_region == r_label
        region_total = float(region_mask.sum())
        if region_total <= 0:
            vf[f"vf_pore_{r_name}"] = 0.0
            vf[f"vf_solids_{r_name}"] = 0.0
            continue
        vf[f"vf_pore_{r_name}"] = float(((labels_material == 0) & region_mask).sum()) / region_total
        vf[f"vf_solids_{r_name}"] = 1.0 - vf[f"vf_pore_{r_name}"]
    return vf


def triangulate_interface_from_height(height_map: np.ndarray, voxel_um: float):
    # Build a scalar field F(z,y,x) = z - h(y,x); zero-level is interface
    ny, nx = height_map.shape
    nz = int(np.ceil(height_map.max())) + 2
    yy, xx = np.meshgrid(np.arange(ny, dtype=np.float32), np.arange(nx, dtype=np.float32), indexing='ij')
    # we sample z from 0..nz-1 uniformly
    z = np.arange(nz, dtype=np.float32)[:, None, None]
    h = height_map[None, :, :]
    field = z - h
    verts, faces, normals, values = measure.marching_cubes(field, level=0.0, spacing=(voxel_um, voxel_um, voxel_um))
    return verts, faces


def triangulate_binary_interface(binary_electrolyte: np.ndarray, voxel_um: float):
    # This will include outer domain surfaces; caller may crop post-hoc if desired
    verts, faces, normals, values = measure.marching_cubes(binary_electrolyte.astype(np.float32), level=0.5, spacing=(voxel_um, voxel_um, voxel_um))
    return verts, faces


def compute_interface_roughness(height_map: np.ndarray, voxel_um: float) -> Dict[str, float]:
    h = height_map.astype(np.float64) * voxel_um
    h_centered = h - h.mean()
    rms = float(np.sqrt(np.mean(h_centered ** 2)))
    p2v = float(h.max() - h.min())
    # gradient-based slope
    gy, gx = np.gradient(h)
    slope = np.sqrt(gx**2 + gy**2)
    mean_slope = float(np.mean(slope))
    # area via triangles approximated from height map
    # approximate by summing local patch areas
    # using simple finite difference
    area = 0.0
    ny, nx = h.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            # two triangles per quad
            p00 = np.array([i * voxel_um, j * voxel_um, h[j, i]])
            p10 = np.array([(i+1) * voxel_um, j * voxel_um, h[j, i+1]])
            p01 = np.array([i * voxel_um, (j+1) * voxel_um, h[j+1, i]])
            p11 = np.array([(i+1) * voxel_um, (j+1) * voxel_um, h[j+1, i+1]])
            area += 0.5 * np.linalg.norm(np.cross(p10 - p00, p01 - p00))
            area += 0.5 * np.linalg.norm(np.cross(p11 - p10, p01 - p10))
    projected_area = float((ny - 1) * (nx - 1) * (voxel_um ** 2))
    area_ratio = float(area / max(projected_area, 1e-12))
    return {
        'interface_rms_um': rms,
        'interface_peak_to_valley_um': p2v,
        'interface_mean_slope': mean_slope,
        'interface_area_um2': float(area),
        'interface_area_ratio': area_ratio,
    }
