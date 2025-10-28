import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.draw import line_nd
from skimage.filters import threshold_otsu
from tqdm import tqdm


@dataclass
class OperationalParameters:
    temperature_celsius: float
    applied_stress_mpa: float
    scan_times_s: np.ndarray  # shape (T,)


@dataclass
class MaterialSpecifications:
    alloy_composition: Dict[str, float]  # element -> wt%
    heat_treatment: str
    initial_grain_size_um: float


@dataclass
class SampleGeometry:
    dimensions_mm: Tuple[float, float, float]  # (x, y, z)


@dataclass
class SimulationConfig:
    volume_size: Tuple[int, int, int]  # (Z, Y, X)
    time_steps: int
    voxel_size_um: float
    seed: int = 42


def set_seed(seed: int):
    np.random.seed(seed)


def perlin_like_noise(shape, scale=16.0, octaves=3, seed=0):
    rng = np.random.default_rng(seed)
    noise = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    frequency = 1.0 / scale
    for _ in range(octaves):
        grid = rng.normal(size=shape).astype(np.float32)
        sigma = max(1.0, scale / (2.0 * frequency))
        smooth = gaussian_filter(grid, sigma=sigma, mode="reflect")
        noise += amplitude * smooth
        amplitude *= 0.5
        frequency *= 2.0
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    return noise


def generate_initial_microstructure(cfg: SimulationConfig):
    Z, Y, X = cfg.volume_size
    base = perlin_like_noise((Z, Y, X), scale=24.0, octaves=4, seed=cfg.seed)
    grains = (base > threshold_otsu(base)).astype(np.uint8)
    # Simulate porosity as low-intensity pockets
    pores = perlin_like_noise((Z, Y, X), scale=12.0, octaves=3, seed=cfg.seed + 1)
    pores = pores < 0.08
    # Combine: 0=void, 1=solid
    solid = np.ones((Z, Y, X), dtype=np.uint8)
    solid[pores] = 0
    # Boundaries: where grains change locally
    gx = np.gradient(base, axis=2)
    gy = np.gradient(base, axis=1)
    gz = np.gradient(base, axis=0)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    boundaries = grad_mag > np.percentile(grad_mag, 85)
    return solid, grains, boundaries


def add_crack(volume: np.ndarray, start, end, radius=1):
    rr, cc, dd = line_nd(start, end, endpoint=True)
    for r, c, d in zip(rr, cc, dd):
        r0, r1 = max(0, r - radius), min(volume.shape[0], r + radius + 1)
        c0, c1 = max(0, c - radius), min(volume.shape[1], c + radius + 1)
        d0, d1 = max(0, d - radius), min(volume.shape[2], d + radius + 1)
        volume[r0:r1, c0:c1, d0:d1] = 0


def evolve_creep(solid0: np.ndarray, boundaries: np.ndarray, cfg: SimulationConfig):
    Z, Y, X = cfg.volume_size
    volumes = np.zeros((cfg.time_steps, Z, Y, X), dtype=np.uint8)
    volumes[0] = solid0.copy()

    rng = np.random.default_rng(cfg.seed + 99)

    # Seed some initial micro-cracks along boundaries
    num_seeds = max(1, (Z * Y * X) // 50000)
    boundary_indices = np.array(np.where(boundaries)).T
    if boundary_indices.size > 0:
        chosen = boundary_indices[rng.choice(len(boundary_indices), size=min(num_seeds, len(boundary_indices)), replace=False)]
        for (z, y, x) in chosen:
            dz, dy, dx = rng.integers(-5, 6, size=3)
            z2 = int(np.clip(z + dz, 0, Z - 1))
            y2 = int(np.clip(y + dy, 0, Y - 1))
            x2 = int(np.clip(x + dx, 0, X - 1))
            add_crack(volumes[0], (z, y, x), (z2, y2, x2), radius=int(rng.integers(1, 2)))

    # Time evolution: creep cavitation and crack growth
    for t in range(1, cfg.time_steps):
        prev = volumes[t - 1].astype(np.float32)
        # Cavitation grows near boundaries and existing voids
        cavitation_bias = gaussian_filter(boundaries.astype(np.float32), sigma=1.5)
        void_influence = gaussian_filter((1 - prev), sigma=1.0)
        growth_prob = 0.02 * cavitation_bias + 0.03 * void_influence
        noise = np.random.rand(Z, Y, X).astype(np.float32)
        void_new = (noise < growth_prob).astype(np.uint8)
        next_vol = prev.copy()
        next_vol[void_new == 1] = 0

        # Crack propagation along high gradients
        if t % 3 == 0:
            # Introduce a biased line growth
            for _ in range(rng.integers(1, 3)):
                z = int(rng.integers(0, Z))
                y = int(rng.integers(0, Y))
                x = int(rng.integers(0, X))
                dz, dy, dx = rng.integers(-8, 9, size=3)
                add_crack(next_vol, (z, y, x), (int(np.clip(z + dz, 0, Z - 1)), int(np.clip(y + dy, 0, Y - 1)), int(np.clip(x + dx, 0, X - 1))), radius=1)

        # Slight densification smoothing
        next_vol = gaussian_filter(next_vol, sigma=0.5)
        next_vol = (next_vol > 0.5).astype(np.uint8)
        volumes[t] = next_vol

    return volumes


def generate_residual_strain_map(cfg: SimulationConfig, operational: OperationalParameters):
    Z, Y, X = cfg.volume_size
    # Base elastic strain gradient under uniaxial stress along Z
    z_axis = np.linspace(-1, 1, Z).reshape(Z, 1, 1)
    base_strain = 0.002 * z_axis  # 0.2% gradient
    thermal_component = (operational.temperature_celsius - 700.0) * 1e-6  # ppm per C
    noise = perlin_like_noise((Z, Y, X), scale=18.0, octaves=3, seed=cfg.seed + 7) * 1e-4
    residual_strain = base_strain + thermal_component + noise
    return residual_strain.astype(np.float32)


def simulate_xrd_pattern(material: MaterialSpecifications, two_theta_range=(20.0, 120.0), points=4000):
    # Simple kinematic peaks from pseudo phases based on composition keys
    elements = sorted(material.alloy_composition.keys())
    phases = [f"{el}-phase" for el in elements[:3]] or ["Fe-phase"]
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], points)

    intensity = np.zeros_like(two_theta)
    rng = np.random.default_rng(123)
    for i, phase in enumerate(phases):
        center = two_theta_range[0] + (i + 1) * (two_theta_range[1] - two_theta_range[0]) / (len(phases) + 1)
        width = 0.8 + 0.3 * i
        height = 1000 * (1.0 - 0.1 * i)
        peak = height * np.exp(-0.5 * ((two_theta - center) / width) ** 2)
        intensity += peak
    # Add background and noise
    background = 50.0 + 30.0 * np.sin(two_theta * np.pi / 180.0)
    intensity += background
    intensity += rng.normal(0, 5.0, size=two_theta.shape)
    intensity = np.clip(intensity, 0, None)
    return two_theta.astype(np.float32), intensity.astype(np.float32)


def save_hdf5(output_path: str,
              volumes: np.ndarray,
              residual_strain: np.ndarray,
              two_theta: np.ndarray,
              intensity: np.ndarray,
              operational: OperationalParameters,
              material: MaterialSpecifications,
              geometry: SampleGeometry,
              cfg: SimulationConfig):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("tomography/volume", data=volumes, compression="gzip")
        # Binary mask for solid
        f.create_dataset("tomography/solid_mask", data=(volumes > 0).astype(np.uint8), compression="gzip")
        # Simple pore volume fraction per time
        pore_fraction = 1.0 - volumes.reshape(volumes.shape[0], -1).mean(axis=1)
        f.create_dataset("tomography/pore_fraction", data=pore_fraction.astype(np.float32))

        f.create_dataset("strain/residual_strain", data=residual_strain, compression="gzip")

        grp = f.create_group("xrd")
        grp.create_dataset("two_theta_deg", data=two_theta)
        grp.create_dataset("intensity", data=intensity)

        meta = {
            "operational_parameters": {
                "temperature_celsius": operational.temperature_celsius,
                "applied_stress_mpa": operational.applied_stress_mpa,
                "scan_times_s": operational.scan_times_s.tolist(),
            },
            "material_specifications": {
                "alloy_composition_wt_pct": material.alloy_composition,
                "heat_treatment": material.heat_treatment,
                "initial_grain_size_um": material.initial_grain_size_um,
            },
            "sample_geometry": {
                "dimensions_mm": geometry.dimensions_mm,
            },
            "simulation_config": {
                "volume_size": cfg.volume_size,
                "time_steps": cfg.time_steps,
                "voxel_size_um": cfg.voxel_size_um,
                "seed": cfg.seed,
            },
        }
        # Store metadata as a UTF-8 encoded JSON bytes blob (NumPy >=2 compatible)
        f.create_dataset("metadata/json", data=json.dumps(meta, indent=2).encode("utf-8"))


def main():
    cfg = SimulationConfig(volume_size=(64, 96, 96), time_steps=8, voxel_size_um=1.0, seed=1337)

    # Example metadata
    operational = OperationalParameters(
        temperature_celsius=750.0,
        applied_stress_mpa=25.0,
        scan_times_s=np.arange(cfg.time_steps) * 60.0,
    )
    material = MaterialSpecifications(
        alloy_composition={"Fe": 72.0, "Cr": 22.0, "Mn": 2.0, "Ni": 4.0},
        heat_treatment="Solutionized 1050C, water quench",
        initial_grain_size_um=12.0,
    )
    geometry = SampleGeometry(dimensions_mm=(2.0, 2.0, 10.0))

    set_seed(cfg.seed)
    solid0, grains, boundaries = generate_initial_microstructure(cfg)
    volumes = evolve_creep(solid0, boundaries, cfg)

    residual_strain = generate_residual_strain_map(cfg, operational)
    two_theta, intensity = simulate_xrd_pattern(material)

    output_path = os.path.join("synthetic_xray_data", "output", "synchrotron_core_dataset.h5")
    save_hdf5(output_path, volumes, residual_strain, two_theta, intensity,
              operational, material, geometry, cfg)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
