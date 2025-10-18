from typing import Dict, Tuple
import math
import numpy as np

from .config import VOLUME_SHAPE, VOXEL_SIZE_UM


def _draw_sphere(volume: np.ndarray, center: Tuple[int, int, int], radius: float, value: int) -> None:
    zc, yc, xc = center
    zdim, ydim, xdim = volume.shape
    r2 = radius * radius
    zmin = max(0, int(zc - radius) )
    zmax = min(zdim - 1, int(zc + radius) )
    ymin = max(0, int(yc - radius) )
    ymax = min(ydim - 1, int(yc + radius) )
    xmin = max(0, int(xc - radius) )
    xmax = min(xdim - 1, int(xc + radius) )
    for z in range(zmin, zmax + 1):
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                if (z - zc) ** 2 + (y - yc) ** 2 + (x - xc) ** 2 <= r2:
                    volume[z, y, x] = value


def _draw_crack_plane(volume: np.ndarray, normal: Tuple[float, float, float], offset: float, half_thickness: float, value: int) -> int:
    # Plane: n dot r = offset; include voxels with |n dot r - offset| <= half_thickness
    # Returns number of voxels written
    zdim, ydim, xdim = volume.shape
    nz, ny, nx = normal
    count = 0
    # Precompute grid coordinates
    zz, yy, xx = np.meshgrid(np.arange(zdim), np.arange(ydim), np.arange(xdim), indexing='ij')
    dist = nz * zz + ny * yy + nx * xx - offset
    mask = np.abs(dist) <= half_thickness
    volume[mask] = value
    count = int(mask.sum())
    return count


def generate_microstructure(latent: Dict[str, object], rng: np.random.Generator) -> Dict[str, object]:
    """Generate a 3D binary microstructure volume and auxiliary metrics.
    Returns dict with keys: volume (np.uint8), crack_fraction_est, rubber_void_fraction_est
    """
    zdim, ydim, xdim = VOLUME_SHAPE
    vol = np.zeros(VOLUME_SHAPE, dtype=np.uint8)  # 0 = solid matrix, 1 = void (pores, cracks)

    rubber_vf = float(latent["rubber_vol_frac"])  # nominal rubber volume fraction (fresh)
    predicted_porosity = float(latent["predicted_porosity"])  # target porosity fraction
    crack_factor = float(latent["crack_factor"])  # 0..1
    melt_phase_fraction = float(latent["melt_phase_fraction"])  # 0..1

    # Place rubber particles (spherical) as initial sites. Number based on volume and desired vf.
    num_voxels = zdim * ydim * xdim
    approx_particle_voxels = int(num_voxels * rubber_vf)
    # Choose particle radius distribution (2..6 voxels)
    radii = rng.integers(2, 6, size=max(1, approx_particle_voxels // 1500))
    centers = []
    for r in radii:
        zc = int(r + rng.integers(0, zdim - 2 * r))
        yc = int(r + rng.integers(0, ydim - 2 * r))
        xc = int(r + rng.integers(0, xdim - 2 * r))
        centers.append((zc, yc, xc, int(r)))

    # Thermal evolution: convert rubber to voids progressively; coalescence at higher T
    # We create void spheres with radius scale depending on melt/pyrolysis
    rubber_void_fraction_est = 0.0
    for (zc, yc, xc, r) in centers:
        r_void = r * (1.0 + 0.6 * melt_phase_fraction + 0.8 * crack_factor)
        _draw_sphere(vol, (zc, yc, xc), r_void, 1)
        rubber_void_fraction_est += (4.0 / 3.0) * math.pi * (r_void ** 3) / num_voxels

    # Coalescence bridges between nearby voids (simplified): add random thin plates
    num_bridges = max(1, int(len(centers) * 0.5 * (melt_phase_fraction + crack_factor)))
    for _ in range(num_bridges):
        normal = rng.normal(size=3)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        offset = rng.uniform(0, zdim + ydim + xdim) * 0.33
        half_thickness = rng.uniform(0.4, 1.2) * (0.5 + 2.0 * crack_factor)
        _draw_crack_plane(vol, (float(normal[0]), float(normal[1]), float(normal[2])), float(offset), float(half_thickness), 1)

    # Thermal cracking: add several crack planes with thickness depending on crack_factor
    n_cracks = int(2 + 8 * crack_factor)
    crack_voxels = 0
    for _ in range(n_cracks):
        normal = rng.normal(size=3)
        normal = normal / (np.linalg.norm(normal) + 1e-9)
        offset = rng.uniform(0, zdim + ydim + xdim) * 0.33
        half_thickness = rng.uniform(0.6, 1.8) * (0.5 + 2.5 * crack_factor)
        crack_voxels += _draw_crack_plane(vol, (float(normal[0]), float(normal[1]), float(normal[2])), float(offset), float(half_thickness), 1)

    # Adjust porosity to target predicted_porosity by random sprinkling/removal
    current_porosity = vol.mean()
    target = predicted_porosity
    if target > current_porosity:
        # Open random pores in solid regions to reach target
        need = int((target - current_porosity) * num_voxels)
        solid_coords = np.argwhere(vol == 0)
        if solid_coords.size > 0 and need > 0:
            idxs = rng.choice(solid_coords.shape[0], size=min(need, solid_coords.shape[0]), replace=False)
            sel = solid_coords[idxs]
            vol[sel[:, 0], sel[:, 1], sel[:, 2]] = 1
    elif target < current_porosity:
        # Close some pores
        need = int((current_porosity - target) * num_voxels)
        pore_coords = np.argwhere(vol == 1)
        if pore_coords.size > 0:
            idxs = rng.choice(pore_coords.shape[0], size=min(need, pore_coords.shape[0]), replace=False)
            sel = pore_coords[idxs]
            vol[sel[:, 0], sel[:, 1], sel[:, 2]] = 0

    crack_fraction_est = crack_voxels / float(num_voxels)

    return {
        "volume": vol.astype(np.uint8),
        "crack_fraction_est": float(crack_fraction_est),
        "rubber_void_fraction_est": float(rubber_void_fraction_est),
    }
