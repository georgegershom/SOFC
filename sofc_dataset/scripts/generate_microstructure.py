#!/usr/bin/env python3
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from skimage import measure
from tifffile import imwrite


@dataclass
class Params:
    nx: int = 160
    ny: int = 160
    nz: int = 160
    voxel_size_um: float = 0.1
    electrolyte_thickness_vox: int = 20
    interlayer_thickness_vox: int = 3
    interface_roughness_vox: float = 3.0
    porosity_fraction: float = 0.32
    anode_solid_fraction: float = 1.0  # kept for clarity; complement of porosity
    ni_fraction_of_anode_solid: float = 0.45
    seed: int = 42
    delamination_prob: float = 0.02
    delamination_max_radius_vox: int = 8
    delamination_min_radius_vox: int = 3


LABELS = {
    "pore": 0,
    "anode": 1,  # Ni-YSZ composite as a single label for the main stack
    "electrolyte": 2,  # dense YSZ
    "interlayer": 3,
}

# Additional derived arrays (not part of main labels)
DERIVED = {
    "ni": 10,  # stored as boolean mask in npy, constant just for mapping
    "ysz_anode": 11,
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def fractal_noise_2d(shape: Tuple[int, int], alpha: float = 2.0, rng: np.random.Generator = None) -> NDArray[np.float32]:
    if rng is None:
        rng = np.random.default_rng()
    ny, nx = shape
    ky = np.fft.fftfreq(ny)[:, None]
    kx = np.fft.fftfreq(nx)[None, :]
    k = np.sqrt(kx * kx + ky * ky)
    k[0, 0] = 1.0
    amplitude = 1.0 / (k ** (alpha / 2.0))  # power ~ 1/k^alpha
    amplitude[0, 0] = 0.0
    phase = rng.uniform(0, 2 * np.pi, size=(ny, nx))
    spectrum = amplitude * np.exp(1j * phase)
    field = np.fft.ifft2(spectrum).real
    field = (field - field.mean()) / (field.std() + 1e-8)
    return field.astype(np.float32)


def multi_scale_noise_3d(shape: Tuple[int, int, int], rng: np.random.Generator) -> NDArray[np.float32]:
    # Create a multi-scale random field by smoothing white noise at different sigmas
    nz, ny, nx = shape
    field = np.zeros(shape, dtype=np.float32)
    sigmas = [0.8, 1.6, 3.2, 6.4]
    weights = [1.0, 0.7, 0.4, 0.25]
    for sigma, w in zip(sigmas, weights):
        noise = rng.standard_normal(shape).astype(np.float32)
        sm = ndi.gaussian_filter(noise, sigma=sigma, mode="reflect")
        field += w * sm
    field -= field.mean()
    field /= (field.std() + 1e-8)
    return field


def threshold_for_volume_fraction(field: NDArray[np.float32], target_fraction: float, higher_is_one: bool = True) -> float:
    # Find threshold so that fraction of ones matches target_fraction
    # Using percentile is equivalent for monotonic mapping
    q = 100.0 * (1.0 - target_fraction if higher_is_one else target_fraction)
    return float(np.percentile(field, q))


def generate_microstructure(p: Params) -> Tuple[NDArray[np.uint8], Dict[str, float], Dict[str, NDArray[np.bool_]], Dict[str, float]]:
    rng = np.random.default_rng(p.seed)

    # Allocate volume: z, y, x
    vol = np.full((p.nz, p.ny, p.nx), fill_value=LABELS["pore"], dtype=np.uint8)

    # Build undulating electrolyte/anode interface
    base = p.electrolyte_thickness_vox
    h = fractal_noise_2d((p.ny, p.nx), alpha=2.2, rng=rng)
    h = h * p.interface_roughness_vox
    interface_height = np.clip(np.rint(base + h), 1, p.nz - p.interlayer_thickness_vox - 2).astype(np.int32)

    # Fill electrolyte up to interface height
    for y in range(p.ny):
        for x in range(p.nx):
            z_top = int(interface_height[y, x])
            vol[:z_top, y, x] = LABELS["electrolyte"]

    # Optional interlayer as a thin dense layer following the interface
    if p.interlayer_thickness_vox > 0:
        for y in range(p.ny):
            for x in range(p.nx):
                z0 = int(interface_height[y, x])
                z1 = min(z0 + p.interlayer_thickness_vox, p.nz - 1)
                vol[z0:z1, y, x] = LABELS["interlayer"]

    # Introduce delamination cavities near interface (convert some interlayer/electrolyte voxels near interface to pore)
    num_candidates = int(p.nx * p.ny * p.delamination_prob)
    for _ in range(num_candidates):
        x = rng.integers(0, p.nx)
        y = rng.integers(0, p.ny)
        r = int(rng.integers(p.delamination_min_radius_vox, p.delamination_max_radius_vox + 1))
        zc = int(interface_height[y, x] + max(1, p.interlayer_thickness_vox // 2))
        zc = np.clip(zc, 1, p.nz - 2)
        z0, z1 = max(0, zc - r), min(p.nz, zc + r)
        y0, y1 = max(0, y - r), min(p.ny, y + r)
        x0, x1 = max(0, x - r), min(p.nx, x + r)
        zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
        mask = (zz - zc) ** 2 + (yy - y) ** 2 + (xx - x) ** 2 <= r * r
        sub = vol[z0:z1, y0:y1, x0:x1]
        sub[mask] = LABELS["pore"]

    # Generate anode region above interlayer: composite of Ni+YSZ+pore
    anode_start = np.min(interface_height) + p.interlayer_thickness_vox
    anode_start = int(np.clip(anode_start, 1, p.nz - 2))
    # Create a full 3D mask for the anode region (shape must match volume)
    anode_region_1d = (np.arange(p.nz) >= anode_start)[:, None, None]
    anode_region = np.broadcast_to(anode_region_1d, (p.nz, p.ny, p.nx))

    # Random field for porosity
    rf = multi_scale_noise_3d(vol.shape, rng)
    thr = threshold_for_volume_fraction(rf[anode_region], p.porosity_fraction, higher_is_one=False)
    pore_mask = np.zeros_like(vol, dtype=bool)
    pore_mask[anode_region] = rf[anode_region] < thr

    # Smooth pore morphology a bit
    pore_mask = ndi.binary_opening(pore_mask, iterations=1)
    pore_mask = ndi.binary_closing(pore_mask, iterations=1)

    # Assign anode label where not pore and in anode region
    anode_mask = anode_region & (~pore_mask)
    vol[anode_mask] = LABELS["anode"]

    # Partition anode solids into Ni vs YSZ (derived masks)
    rf_solid = multi_scale_noise_3d(vol.shape, rng)
    f_ni = p.ni_fraction_of_anode_solid
    thr_ni = threshold_for_volume_fraction(rf_solid[anode_mask], f_ni, higher_is_one=True)
    ni_mask = np.zeros_like(vol, dtype=bool)
    ysz_anode_mask = np.zeros_like(vol, dtype=bool)
    ni_mask[anode_mask] = rf_solid[anode_mask] >= thr_ni
    ysz_anode_mask[anode_mask] = ~ni_mask[anode_mask]

    # Ensure electrolyte and interlayer stay dense (except delamination pores already carved)
    # no-op as we already overwrote earlier

    # Compute interface mask: electrolyte voxels that touch anode voxels
    is_electrolyte = vol == LABELS["electrolyte"]
    is_anode = vol == LABELS["anode"]
    anode_dil = ndi.binary_dilation(is_anode, structure=ndi.generate_binary_structure(3, 2))
    electrolyte_eroded = ndi.binary_erosion(is_electrolyte, structure=ndi.generate_binary_structure(3, 1))
    electrolyte_surface = is_electrolyte & (~electrolyte_eroded)
    interface_mask = electrolyte_surface & anode_dil

    # Compute interface height map (first interface voxel along z for each x,y)
    interface_height_map = np.full((p.ny, p.nx), fill_value=np.nan, dtype=np.float32)
    for y in range(p.ny):
        for x in range(p.nx):
            col = interface_mask[:, y, x]
            idx = np.where(col)[0]
            if idx.size > 0:
                interface_height_map[y, x] = float(idx.max())  # top-most electrolyte touching anode

    # Mesh the interface via marching cubes on the interface mask
    spacing = (p.voxel_size_um, p.voxel_size_um, p.voxel_size_um)
    try:
        verts, faces, _, _ = measure.marching_cubes(interface_mask.astype(np.float32), level=0.5, spacing=spacing)
        faces = faces.astype(np.int32)
        interface_area_um2 = float(_mesh_area(verts, faces))
    except Exception:
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)
        interface_area_um2 = 0.0

    # Volume fractions
    total_vox = vol.size
    vf = {
        "pore": float(np.count_nonzero(vol == LABELS["pore"]) / total_vox),
        "anode": float(np.count_nonzero(vol == LABELS["anode"]) / total_vox),
        "electrolyte": float(np.count_nonzero(vol == LABELS["electrolyte"]) / total_vox),
        "interlayer": float(np.count_nonzero(vol == LABELS["interlayer"]) / total_vox),
        "ni_in_anode_solid": float(np.count_nonzero(ni_mask) / max(np.count_nonzero(anode_mask), 1)),
    }

    # Interface metrics
    # Contact coverage: fraction of (x,y) columns where electrolyte touches anode
    contact_cols = np.count_nonzero(~np.isnan(interface_height_map))
    possible_cols = p.nx * p.ny
    contact_fraction = float(contact_cols / possible_cols)

    # Roughness metrics based on interface_height_map
    heights = interface_height_map[~np.isnan(interface_height_map)]
    if heights.size > 0:
        z_mean = float(np.mean(heights))
        sa = float(np.mean(np.abs(heights - z_mean)) * p.voxel_size_um)
        sq = float(np.sqrt(np.mean((heights - z_mean) ** 2)) * p.voxel_size_um)
    else:
        sa = 0.0
        sq = 0.0

    interface_metrics = {
        "area_um2": interface_area_um2,
        "contact_fraction": contact_fraction,
        "Sa_um": sa,
        "Sq_um": sq,
    }

    derived_masks = {
        "ni_mask": ni_mask,
        "ysz_anode_mask": ysz_anode_mask,
        "interface_mask": interface_mask,
    }

    return vol, vf, derived_masks, interface_metrics


def _mesh_area(verts: NDArray[np.float32], faces: NDArray[np.int32]) -> float:
    if faces.size == 0:
        return 0.0
    tri = verts[faces]
    v0 = tri[:, 1] - tri[:, 0]
    v1 = tri[:, 2] - tri[:, 0]
    cross = np.cross(v0, v1)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return float(np.sum(areas))


def save_stack(vol: NDArray[np.uint8], out_dir: str) -> None:
    ensure_dir(out_dir)
    nz = vol.shape[0]
    for z in range(nz):
        path = os.path.join(out_dir, f"slice_{z:04d}.tif")
        imwrite(path, vol[z, :, :].astype(np.uint8))


def save_numpy_arrays(vol: NDArray[np.uint8], derived: Dict[str, NDArray[np.bool_]], arrays_dir: str) -> None:
    ensure_dir(arrays_dir)
    np.save(os.path.join(arrays_dir, "labels_main.npy"), vol)
    for name, arr in derived.items():
        np.save(os.path.join(arrays_dir, f"{name}.npy"), arr.astype(np.uint8))


def save_interface_mesh_ply(vertices: NDArray[np.float32], faces: NDArray[np.int32], out_path: str) -> None:
    # Write ASCII PLY triangle mesh
    ensure_dir(os.path.dirname(out_path))
    num_v = vertices.shape[0]
    num_f = faces.shape[0]
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {num_v}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {num_f}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def save_vtk_image(vol: NDArray[np.uint8], voxel_size_um: float, out_path: str) -> Optional[str]:
    # Prefer writing VTK image using pyvista; fallback to npy if not available
    try:
        import pyvista as pv
        ensure_dir(os.path.dirname(out_path))
        nz, ny, nx = vol.shape
        grid = pv.UniformGrid()
        # UniformGrid expects number of POINTS in each direction; for cell data sized (nx,ny,nz), set dims to (nx+1, ny+1, nz+1)
        grid.dimensions = (nx + 1, ny + 1, nz + 1)
        grid.spacing = (voxel_size_um, voxel_size_um, voxel_size_um)
        # Arrange data with x-fastest, y-next, z-slowest ordering
        cell_values = np.ascontiguousarray(vol.transpose(2, 1, 0)).ravel(order="F")
        grid.cell_data["phase"] = cell_values
        grid.save(out_path)
        return out_path
    except Exception as e:
        return None


def main():
    parser = argparse.ArgumentParser(description="Synthetic SOFC microstructure generator")
    parser.add_argument("--nx", type=int, default=160)
    parser.add_argument("--ny", type=int, default=160)
    parser.add_argument("--nz", type=int, default=160)
    parser.add_argument("--voxel-size-um", type=float, default=0.1)
    parser.add_argument("--electrolyte-thickness", type=int, default=20)
    parser.add_argument("--interlayer-thickness", type=int, default=3)
    parser.add_argument("--interface-roughness", type=float, default=3.0)
    parser.add_argument("--porosity", type=float, default=0.32)
    parser.add_argument("--ni-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=str, default="/workspace/sofc_dataset")

    args = parser.parse_args()

    out_root = args.out_root
    data_dir = os.path.join(out_root, "data", "images")
    arrays_dir = os.path.join(out_root, "arrays")
    interface_dir = os.path.join(out_root, "interface")
    mesh_dir = os.path.join(out_root, "mesh")

    p = Params(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        voxel_size_um=args.voxel_size_um,
        electrolyte_thickness_vox=args.electrolyte_thickness,
        interlayer_thickness_vox=args.interlayer_thickness,
        interface_roughness_vox=args.interface_roughness,
        porosity_fraction=args.porosity,
        ni_fraction_of_anode_solid=args.ni_fraction,
        seed=args.seed,
    )

    vol, vf, derived, iface_metrics = generate_microstructure(p)

    # Save stacks and arrays
    save_stack(vol, data_dir)
    save_numpy_arrays(vol, derived, arrays_dir)

    # Save metadata
    meta = {
        "dimensions": {"nz": p.nz, "ny": p.ny, "nx": p.nx},
        "voxel_size_um": p.voxel_size_um,
        "labels": LABELS,
        "volume_fractions": vf,
        "generator_params": p.__dict__,
    }
    ensure_dir(out_root)
    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Interface mesh and metrics
    # Regenerate verts/faces from derived interface mask for saving
    try:
        verts, faces, _, _ = measure.marching_cubes(derived["interface_mask"].astype(np.float32), level=0.5, spacing=(p.voxel_size_um, p.voxel_size_um, p.voxel_size_um))
        faces = faces.astype(np.int32)
    except Exception:
        verts = np.zeros((0, 3), dtype=np.float32)
        faces = np.zeros((0, 3), dtype=np.int32)

    save_interface_mesh_ply(verts, faces, os.path.join(interface_dir, "interface_surface.ply"))
    with open(os.path.join(interface_dir, "metrics.json"), "w") as f:
        json.dump(iface_metrics, f, indent=2)

    # VTK image mesh with phase tags
    vtk_path = os.path.join(mesh_dir, "voxel_grid.vti")
    vtk_written = save_vtk_image(vol, p.voxel_size_um, vtk_path)

    # Also save a simple npy mesh fallback
    np.save(os.path.join(mesh_dir, "voxel_labels.npy"), vol)

    # Report
    print("Generation complete.")
    print("Volume fractions:", json.dumps(vf, indent=2))
    if vtk_written is None:
        print("VTK image not written (pyvista/vtk missing). Fallback .npy saved.")
    else:
        print(f"VTK image written to: {vtk_written}")


if __name__ == "__main__":
    main()
