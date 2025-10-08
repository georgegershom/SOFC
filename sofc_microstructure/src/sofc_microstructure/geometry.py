from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict


def generate_height_map(nx: int, ny: int, base_z_vox: int, amplitude_vox: float,
                        corr_len_vox: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(ny, nx))
    smoothed = gaussian_filter(noise, sigma=corr_len_vox, mode='reflect')
    smoothed -= smoothed.min()
    smoothed /= (smoothed.max() + 1e-12)
    smoothed = (smoothed - 0.5) * 2.0  # [-1, 1]
    height = base_z_vox + amplitude_vox * smoothed
    return height.astype(np.float32)


def threshold_for_target_fraction(field: np.ndarray, target_fraction: float) -> float:
    # field assumed roughly Gaussian; threshold by percentile
    target_fraction = np.clip(target_fraction, 0.0, 1.0)
    return np.percentile(field, 100.0 * (1.0 - target_fraction))


def generate_anode_skeleton(nz: int, ny: int, nx: int, porosity: float,
                            corr_len_vox: float, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(nz, ny, nx)).astype(np.float32)
    smoothed = gaussian_filter(noise, sigma=corr_len_vox, mode='reflect')
    # target solids fraction = 1 - porosity
    solids_fraction = max(0.0, min(1.0, 1.0 - porosity))
    thr = threshold_for_target_fraction(smoothed, target_fraction=solids_fraction)
    skeleton = smoothed >= thr
    return skeleton


def split_ni_ysz(solids_mask: np.ndarray, ni_fraction_in_solid: float,
                 corr_len_vox: float, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    nz, ny, nx = solids_mask.shape
    noise = rng.normal(size=(nz, ny, nx)).astype(np.float32)
    smoothed = gaussian_filter(noise, sigma=corr_len_vox, mode='reflect')
    thr = threshold_for_target_fraction(smoothed, target_fraction=ni_fraction_in_solid)
    ni = (smoothed >= thr) & solids_mask
    ysz = solids_mask & (~ni)
    return ni, ysz


def build_phase_volumes(nx: int, ny: int, nz: int,
                        voxel_um: float,
                        base_interface_z_vox: int,
                        interface_amp_um: float,
                        interface_corr_len_um: float,
                        anode_porosity: float,
                        ni_fraction_in_solid: float,
                        interlayer_thickness_um: float,
                        interlayer_porosity: float,
                        seed: int | None = None) -> Dict[str, np.ndarray]:
    amp_vox = float(interface_amp_um / voxel_um)
    corr_xy_vox = float(interface_corr_len_um / voxel_um)
    height = generate_height_map(nx, ny, base_interface_z_vox, amp_vox, corr_xy_vox, seed=seed)

    # region masks by z relative to height map
    z_axis = np.arange(nz, dtype=np.float32)[:, None, None]
    height_broadcast = height[None, :, :]

    is_electrolyte = z_axis >= height_broadcast

    # Anode region below interface; interlayer is a band immediately below
    interlayer_thickness_vox = max(1, int(round(interlayer_thickness_um / voxel_um)))
    is_interlayer = (z_axis >= (height_broadcast - interlayer_thickness_vox)) & (~is_electrolyte)
    is_anode_bulk = (~is_electrolyte) & (~is_interlayer)

    # Generate porous skeleton for anode bulk and interlayer separately
    corr_len_3d_vox = max(1.0, (2.0 / voxel_um))  # ~2 µm correlation in 3D by default

    skeleton_bulk = generate_anode_skeleton(nz, ny, nx, porosity=anode_porosity,
                                            corr_len_vox=corr_len_3d_vox, seed=None if seed is None else seed + 1)
    skeleton_inter = generate_anode_skeleton(nz, ny, nx, porosity=interlayer_porosity,
                                             corr_len_vox=corr_len_3d_vox, seed=None if seed is None else seed + 2)

    solids = (skeleton_bulk & is_anode_bulk) | (skeleton_inter & is_interlayer)

    # split solids into Ni and YSZ in anode+interlayer
    ni_mask, ysz_anode_mask = split_ni_ysz(solids, ni_fraction_in_solid=ni_fraction_in_solid,
                                           corr_len_vox=max(1.0, corr_len_3d_vox * 0.6),
                                           seed=None if seed is None else seed + 3)

    # electrolyte is dense YSZ
    electrolyte_mask = is_electrolyte

    # pores are whatever is not solid or electrolyte
    pore_mask = ~(solids | electrolyte_mask)

    return {
        'height_map': height.astype(np.float32),
        'region_anode_bulk': is_anode_bulk.astype(np.uint8),
        'region_interlayer': is_interlayer.astype(np.uint8),
        'region_electrolyte': electrolyte_mask.astype(np.uint8),
        'phase_pore': pore_mask.astype(np.uint8),
        'phase_ni': ni_mask.astype(np.uint8),
        'phase_ysz_anode': ysz_anode_mask.astype(np.uint8),
        'phase_ysz_electrolyte': electrolyte_mask.astype(np.uint8),
    }
