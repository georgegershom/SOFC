#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters, measure
import imageio.v3 as iio
import meshio
from tqdm import tqdm

# Phase labels
# 0: Pore
# 1: Ni-YSZ (anode)
# 2: YSZ (electrolyte)
# 3: Interlayer (optional)

@dataclass
class MicrostructureConfig:
    size: Tuple[int, int, int] = (256, 256, 256)
    voxel_size_um: float = 0.05
    porosity: float = 0.35
    anode_fraction: float = 0.55  # of solids
    interlayer_thickness_um: float = 2.0
    electrolyte_thickness_um: float = 5.0
    interface_roughness_um: float = 3.0
    seed: int = 42

@dataclass
class Metrics:
    voxel_size_um: float
    size: Tuple[int, int, int]
    volume_um3: float
    volume_fractions: dict


def generate_perlin_noise(shape, scale, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 1, shape)
    # Smooth to introduce spatial correlation
    sigma = scale
    noise = ndi.gaussian_filter(noise, sigma=sigma, mode='reflect')
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-12)
    return noise


def synthesize_microstructure(cfg: MicrostructureConfig) -> np.ndarray:
    np.random.seed(cfg.seed)

    nx, ny, nz = cfg.size
    voxel = cfg.voxel_size_um

    # Build layered base: anode (bottom), electrolyte (top), optional interlayer
    total_thickness_um = nz * voxel
    elec_vox = max(1, int(cfg.electrolyte_thickness_um / voxel))
    inter_vox = max(0, int(cfg.interlayer_thickness_um / voxel))

    # Clamp layer thicknesses to fit within the domain ensuring anode exists
    if elec_vox >= nz:
        elec_vox = max(1, nz // 4)
    if inter_vox >= nz - elec_vox:
        inter_vox = max(0, (nz - elec_vox) // 8)
    if elec_vox + inter_vox >= nz:
        # enforce at least one voxel for an anode region
        extra = elec_vox + inter_vox - (nz - 1)
        inter_vox = max(0, inter_vox - extra)
        if elec_vox + inter_vox >= nz:
            elec_vox = max(1, elec_vox - (elec_vox + inter_vox - (nz - 1)))

    volume = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Electrolyte slab at top
    volume[:, :, -elec_vox:] = 2

    # Interlayer below electrolyte if requested
    if inter_vox > 0 and nz - elec_vox - inter_vox > 0:
        volume[:, :, nz - elec_vox - inter_vox: nz - elec_vox] = 3

    # Anode below that
    anode_end = max(0, nz - elec_vox - inter_vox)
    volume[:, :, :anode_end] = 1

    # Roughen the anode/electrolyte interface
    rough_scale = max(1, int(cfg.interface_roughness_um / voxel))
    rough_field = generate_perlin_noise((nx, ny), rough_scale, cfg.seed + 1)
    rough_amplitude_vox = max(1, int(cfg.interface_roughness_um / voxel))
    interface_base = anode_end
    offset = (rough_field - 0.5) * 2 * rough_amplitude_vox
    offset = offset.astype(int)
    for x in range(nx):
        for y in range(ny):
            z_shift = offset[x, y]
            z0 = np.clip(interface_base + z_shift, 1, nz - 2)
            # Carve into anode or electrolyte depending on shift
            if z_shift > 0:
                volume[x, y, interface_base:z0] = volume[x, y, z0]
            elif z_shift < 0:
                volume[x, y, z0:interface_base] = volume[x, y, z0 - 1]

    # Generate porosity and anode/electrolyte micro-porosity using correlated noise
    corr_scale = max(1, int(1.5 / voxel))
    noise3d = generate_perlin_noise(cfg.size, corr_scale, cfg.seed + 2)

    # Thresholds to reach target porosity primarily in anode
    pore_mask = np.zeros_like(volume, dtype=bool)

    # Target pore only in anode/interlayer; keep electrolyte dense
    anode_mask = volume == 1
    inter_mask = volume == 3

    # Compute threshold to achieve desired porosity in anode+interlayer region
    noise_vals = noise3d[anode_mask | inter_mask]
    if noise_vals.size > 0:
        thresh = np.quantile(noise_vals, cfg.porosity)
        pore_mask[anode_mask | inter_mask] = noise3d[anode_mask | inter_mask] <= thresh

    # Clean pores: remove tiny specks and enforce connectivity
    pore_mask = morphology.remove_small_objects(pore_mask, min_size=8)
    pore_mask = morphology.binary_opening(pore_mask, morphology.ball(1))

    # Apply pores
    volume[pore_mask] = 0

    # Split anode solids into Ni and YSZ subphases not requested now; keep as 1
    return volume


def compute_volume_fractions(volume: np.ndarray, voxel_size_um: float) -> Metrics:
    nx, ny, nz = volume.shape
    total_voxels = nx * ny * nz
    fractions = {}
    for label, name in [(0, 'pore'), (1, 'anode'), (2, 'electrolyte'), (3, 'interlayer')]:
        count = int(np.sum(volume == label))
        fractions[name] = count / total_voxels
    vol_um3 = (voxel_size_um ** 3) * total_voxels
    return Metrics(
        voxel_size_um=voxel_size_um,
        size=(nx, ny, nz),
        volume_um3=vol_um3,
        volume_fractions=fractions,
    )


def export_slice_stack(volume: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    nz = volume.shape[2]
    for z in tqdm(range(nz), desc='Writing slices'):
        img = volume[:, :, z].astype(np.uint8)
        iio.imwrite(os.path.join(out_dir, f'slice_{z:04d}.png'), img)


def export_numpy(volume: np.ndarray, out_path: str):
    np.save(out_path, volume)


def export_meshes(volume: np.ndarray, voxel_size_um: float, mesh_dir: str):
    os.makedirs(mesh_dir, exist_ok=True)
    # Export marching cubes surfaces for each solid/phase interface
    labels = {
        1: 'anode',
        2: 'electrolyte',
        3: 'interlayer',
    }
    for label, name in labels.items():
        mask = (volume == label).astype(np.uint8)
        if mask.sum() == 0:
            continue
        # Marching cubes on binary mask; skimage returns verts in voxel coords
        try:
            verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)
        except ValueError:
            continue
        # Scale to micrometers
        verts = verts * voxel_size_um
        mesh = meshio.Mesh(points=verts, cells={'triangle': faces})
        meshio.write(os.path.join(mesh_dir, f'{name}_surface.ply'), mesh)

    # Optionally write voxel grid as VTI-like mesh (using meshio XDMF)
    # We write an XDMF + raw .bin for labeled voxels
    xdmf_path = os.path.join(mesh_dir, 'labels.xdmf')
    raw_path = os.path.join(mesh_dir, 'labels.raw')
    data = volume.astype(np.uint8)
    data.tofile(raw_path)
    nx, ny, nz = volume.shape
    # XDMF for uniform grid
    xdmf = f"""
<?xml version="1.0" ?>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="voxels" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">0 0 0</DataItem>
        <DataItem Dimensions="3" NumberType="Float" Precision="8" Format="XML">{voxel_size_um} {voxel_size_um} {voxel_size_um}</DataItem>
      </Geometry>
      <Attribute Name="labels" AttributeType="Scalar" Center="Cell">
        <DataItem Dimensions="{nz} {ny} {nx}" NumberType="UInt" Precision="1" Format="Binary">{os.path.basename(raw_path)}</DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    with open(xdmf_path, 'w') as f:
        f.write(xdmf)


def save_metadata(cfg: MicrostructureConfig, metrics: Metrics, out_path: str):
    payload = {
        'config': asdict(cfg),
        'metrics': asdict(metrics),
        'phase_labels': {"0":"pore","1":"anode","2":"electrolyte","3":"interlayer"}
    }
    with open(out_path, 'w') as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic 3D SOFC microstructure dataset')
    parser.add_argument('--nx', type=int, default=256)
    parser.add_argument('--ny', type=int, default=256)
    parser.add_argument('--nz', type=int, default=256)
    parser.add_argument('--voxel_um', type=float, default=0.05)
    parser.add_argument('--porosity', type=float, default=0.35)
    parser.add_argument('--electrolyte_um', type=float, default=5.0)
    parser.add_argument('--interlayer_um', type=float, default=2.0)
    parser.add_argument('--interface_roughness_um', type=float, default=3.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='/workspace/data/synthetic_microstructure')

    args = parser.parse_args()
    cfg = MicrostructureConfig(
        size=(args.nx, args.ny, args.nz),
        voxel_size_um=args.voxel_um,
        porosity=args.porosity,
        electrolyte_thickness_um=args.electrolyte_um,
        interlayer_thickness_um=args.interlayer_um,
        interface_roughness_um=args.interface_roughness_um,
        seed=args.seed,
    )

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    volume = synthesize_microstructure(cfg)
    metrics = compute_volume_fractions(volume, cfg.voxel_size_um)

    # Exports
    export_numpy(volume, os.path.join(out_dir, 'labels.npy'))
    export_slice_stack(volume, os.path.join(out_dir, 'slices'))
    export_meshes(volume, cfg.voxel_size_um, os.path.join(out_dir, 'mesh'))
    save_metadata(cfg, metrics, os.path.join(out_dir, 'metadata.json'))

    print(json.dumps(asdict(metrics), indent=2))


if __name__ == '__main__':
    main()
