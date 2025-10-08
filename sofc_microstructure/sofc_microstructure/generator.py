from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology


@dataclass
class GeneratorConfig:
    # Volume dimensions
    nx: int = 256
    ny: int = 256
    nz: int = 192

    # Phase labels
    label_pore: int = 0
    label_anode: int = 1  # Ni-YSZ
    label_electrolyte: int = 2  # YSZ dense
    label_interlayer: int = 3  # optional thin layer

    # Target fractions (approximate)
    porosity: float = 0.35
    interlayer_thickness_vox: int = 2

    # Morphology parameters
    anode_feature_scale: float = 8.0
    pore_feature_scale: float = 6.0
    pore_anisotropy: float = 1.6  # elongation along z for gas channels

    # Interface waviness
    interface_mean_z: int = 96
    interface_rms_amplitude: float = 4.0
    interface_corr_xy: float = 24.0

    # Random seed
    seed: int = 42


def _gaussian_random_field(shape: Tuple[int, int, int], length_scale: Tuple[float, float, float], rng: np.random.Generator) -> np.ndarray:
    kx = np.fft.fftfreq(shape[0])
    ky = np.fft.fftfreq(shape[1])
    kz = np.fft.fftfreq(shape[2])
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    spectrum = np.exp(-0.5 * ((kx * length_scale[0]) ** 2 + (ky * length_scale[1]) ** 2 + (kz * length_scale[2]) ** 2))
    noise = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    field = np.fft.ifftn(np.fft.fftn(noise) * spectrum).real
    field = (field - field.mean()) / (field.std() + 1e-8)
    return field


def generate_volume(cfg: GeneratorConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)

    # Initialize all voxels as electrolyte (dense YSZ)
    vol = np.full((cfg.nx, cfg.ny, cfg.nz), cfg.label_electrolyte, dtype=np.uint8)

    # Build a wavy interface height map between anode and electrolyte
    hx, hy = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")
    base = np.full((cfg.nx, cfg.ny), cfg.interface_mean_z, dtype=float)

    # Correlated noise for waviness
    corr_field = _gaussian_random_field((cfg.nx, cfg.ny, 1), (1.0 / cfg.interface_corr_xy, 1.0 / cfg.interface_corr_xy, 1.0), rng)[..., 0]
    waviness = cfg.interface_rms_amplitude * corr_field
    interface_z = base + waviness
    # Keep interface inside the volume with a small margin for interlayer
    margin = max(2, cfg.interlayer_thickness_vox + 1)
    interface_z = np.clip(interface_z, margin, cfg.nz - 1 - margin)

    # Create an anode region above the interface (lower z values)
    z_idx = np.arange(cfg.nz)[None, None, :]
    anode_region = z_idx < interface_z[..., None]
    vol[anode_region] = cfg.label_anode

    # Optionally insert a thin interlayer at the interface on the electrolyte side
    interlayer_thickness = max(cfg.interlayer_thickness_vox, 0)
    if interlayer_thickness > 0:
        interlayer_region = (z_idx >= interface_z[..., None]) & (z_idx < (interface_z[..., None] + interlayer_thickness))
        vol[interlayer_region] = cfg.label_interlayer

    # Sculpt anode porosity using an anisotropic Gaussian random field and thresholding
    aniso_length = (
        1.0 / cfg.pore_feature_scale,
        1.0 / cfg.pore_feature_scale,
        1.0 / (cfg.pore_feature_scale * cfg.pore_anisotropy),
    )
    pore_field = _gaussian_random_field((cfg.nx, cfg.ny, cfg.nz), aniso_length, rng)

    # Threshold to hit target porosity within anode; compute threshold by quantile over anode voxels
    anode_mask = vol == cfg.label_anode
    # Only use field within anode region for threshold
    field_vals = pore_field[anode_mask]
    pore_thresh = np.quantile(field_vals, cfg.porosity)
    pore_mask = np.zeros_like(vol, dtype=bool)
    pore_mask[anode_mask] = pore_field[anode_mask] < pore_thresh

    # Carve pores in anode
    vol[pore_mask] = cfg.label_pore

    # Mild morphological cleanup in 3D: remove tiny pore speckles and fill tiny cavities
    pore_mask_full = vol == cfg.label_pore
    pore_mask_full = morphology.remove_small_objects(pore_mask_full, min_size=27, connectivity=1)
    pore_mask_full = morphology.remove_small_holes(pore_mask_full, area_threshold=27)

    # Restore removed pore voxels and revert tiny pores to anode within anode domain
    vol[pore_mask_full] = cfg.label_pore
    vol[(~pore_mask_full) & (anode_region)] = cfg.label_anode

    # Optional: light morphological closing on the binary anode/electrolyte boundary to reduce staircasing
    # This avoids introducing invalid labels via median filtering with sentinels.
    # Create binary mask of anode vs not-anode and smooth slightly
    boundary_mask = morphology.binary_dilation(vol == cfg.label_anode, morphology.ball(1))
    boundary_mask &= ~morphology.binary_erosion(vol == cfg.label_anode, morphology.ball(1))

    return vol, interface_z


def compute_volume_fractions(vol: np.ndarray, cfg: GeneratorConfig) -> Dict[str, float]:
    total = vol.size
    counts = {label: int((vol == label).sum()) for label in [cfg.label_pore, cfg.label_anode, cfg.label_electrolyte, cfg.label_interlayer]}
    return {
        "pore": counts[cfg.label_pore] / total,
        "anode": counts[cfg.label_anode] / total,
        "electrolyte": counts[cfg.label_electrolyte] / total,
        "interlayer": counts[cfg.label_interlayer] / total,
    }


def extract_interface_mesh(vol: np.ndarray, interface_z: np.ndarray, cfg: GeneratorConfig):
    """Extract the anode/electrolyte interface as an isosurface of the height field.

    We build a scalar field phi(x,y,z) = z - interface_z(x,y). The interface is phi == 0.
    Using the height field ensures a single, watertight interface surface.
    """
    z_idx = np.arange(cfg.nz)[None, None, :].astype(float)
    phi = z_idx - interface_z[..., None]
    verts, faces, normals, values = measure.marching_cubes(phi, level=0.0, spacing=(1.0, 1.0, 1.0))
    return verts, faces


def _radial_profile(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute isotropic radial average profile of a 2D image about its center.

    Returns radii (pixels) and radial mean values.
    """
    ny, nx = image.shape
    cy = (ny - 1) / 2.0
    cx = (nx - 1) / 2.0
    y, x = np.indices((ny, nx))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_int = np.floor(r).astype(np.int32)
    # Bin by integer radius
    max_r = r_int.max()
    bin_counts = np.bincount(r_int.ravel(), minlength=max_r + 1)
    bin_sums = np.bincount(r_int.ravel(), weights=image.ravel(), minlength=max_r + 1)
    with np.errstate(invalid="ignore"):
        radial_mean = bin_sums / np.maximum(bin_counts, 1)
    radii = np.arange(len(radial_mean))
    return radii, radial_mean


def compute_interface_metrics(interface_z: np.ndarray) -> Dict[str, float]:
    """Compute RMS height and isotropic autocorrelation length of the interface height map.

    Correlation length is taken as the first radius where the normalized autocorrelation
    decays to 1/e.
    """
    h = interface_z - float(interface_z.mean())
    rms = float(np.sqrt(np.mean(h ** 2)))

    # 2D autocorrelation via FFT
    H = np.fft.fft2(h)
    ac = np.fft.ifft2(np.abs(H) ** 2).real
    ac = np.fft.fftshift(ac)
    # Normalize so that ac(0) = 1
    ac /= ac.max() if ac.max() != 0 else 1.0

    radii, radial_ac = _radial_profile(ac)
    # Find first radius where correlation falls below 1/e
    target = 1.0 / math.e
    idxs = np.where(radial_ac <= target)[0]
    if len(idxs) == 0:
        corr_len = float(radii[-1])
    else:
        corr_len = float(radii[int(idxs[0])])

    return {
        "mean_z": float(interface_z.mean()),
        "rms_height": rms,
        "corr_length_px": corr_len,
    }


def save_slices(output_dir: str, vol: np.ndarray, cfg: GeneratorConfig) -> Dict[str, str]:
    import imageio
    from tifffile import imwrite

    os.makedirs(output_dir, exist_ok=True)

    # Save labeled stack as single TIFF
    labeled_path = os.path.join(output_dir, "labels.tiff")
    imwrite(labeled_path, vol.astype(np.uint8))

    # Create a pseudo-grayscale stack resembling tomography attenuation
    # Assign intensities: pore (low), anode (mid), interlayer (high-mid), electrolyte (high)
    grayscale = np.zeros_like(vol, dtype=np.uint8)
    grayscale[vol == cfg.label_pore] = 15
    grayscale[vol == cfg.label_anode] = 110
    grayscale[vol == cfg.label_interlayer] = 160
    grayscale[vol == cfg.label_electrolyte] = 200

    grayscale_path = os.path.join(output_dir, "grayscale.tiff")
    imwrite(grayscale_path, grayscale)

    # Also export per-slice PNGs for quick view (limited count to avoid huge dirs)
    preview_dir = os.path.join(output_dir, "preview_png")
    os.makedirs(preview_dir, exist_ok=True)
    step = max(1, vol.shape[2] // 48)
    for z in range(0, vol.shape[2], step):
        imageio.imwrite(os.path.join(preview_dir, f"slice_{z:04d}.png"), grayscale[:, :, z])

    return {"labels_tiff": labeled_path, "grayscale_tiff": grayscale_path, "preview_dir": preview_dir}


def save_mesh(output_dir: str, verts: np.ndarray, faces: np.ndarray) -> str:
    import trimesh

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh_path = os.path.join(output_dir, "interface_mesh.ply")
    mesh.export(mesh_path)
    return mesh_path


def write_manifest(output_dir: str, cfg: GeneratorConfig, fractions: Dict[str, float], interface_metrics: Dict[str, float], paths: Dict[str, str]):
    manifest = {
        "description": "Synthetic SOFC microstructure dataset (labeled 3D volume, grayscale, interface mesh)",
        "dimensions": {"nx": cfg.nx, "ny": cfg.ny, "nz": cfg.nz},
        "labels": {
            "pore": cfg.label_pore,
            "anode": cfg.label_anode,
            "electrolyte": cfg.label_electrolyte,
            "interlayer": cfg.label_interlayer,
        },
        "target_porosity": cfg.porosity,
        "fractions": fractions,
        "interface_metrics": interface_metrics,
        "files": paths,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def generate_dataset(output_root: str, cfg: GeneratorConfig) -> str:
    os.makedirs(output_root, exist_ok=True)
    out_dir = os.path.join(output_root, "dataset")
    os.makedirs(out_dir, exist_ok=True)

    vol, interface_z = generate_volume(cfg)
    fractions = compute_volume_fractions(vol, cfg)
    interface_metrics = compute_interface_metrics(interface_z)
    verts, faces = extract_interface_mesh(vol, interface_z, cfg)

    paths = save_slices(out_dir, vol, cfg)
    mesh_path = save_mesh(out_dir, verts, faces)
    paths["mesh_ply"] = mesh_path

    write_manifest(out_dir, cfg, fractions, interface_metrics, paths)

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic SOFC microstructure dataset")
    parser.add_argument("--out", type=str, default="/workspace/sofc_microstructure/output", help="Output root directory")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--nz", type=int, default=192)
    parser.add_argument("--porosity", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    cfg = GeneratorConfig(nx=args.nx, ny=args.ny, nz=args.nz, porosity=args.porosity, seed=args.seed)
    out_dir = generate_dataset(args.out, cfg)
    print(f"Dataset written to: {out_dir}")
