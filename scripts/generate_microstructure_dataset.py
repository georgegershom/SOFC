#!/usr/bin/env python3
"""Generate a synthetic microstructure-informed dataset for SOFC sintering studies.

This script fabricates multi-modal data products mimicking SEM, \u03bcCT, EDS,
and FEM-ready assets for the research topic:

    "A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the
    Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells
    through Targeted Creep Activation"

The generated dataset is *microstructure-informed*: the same stochastic
seeded 3D microstructure feeds SEM slices, \u03bcCT volumes, constitutive
metrics, and FEM boundary/initial condition fields to ensure consistency
between modalities.

Outputs are written beneath ``datasets/sofc_microstructure_informed``.

Usage::

    python scripts/generate_microstructure_dataset.py [--seed 42]

Dependencies:
    numpy, pandas, scipy, matplotlib, pillow (through matplotlib), h5py(optional)

"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Data classes for structured metadata
# ---------------------------------------------------------------------------


@dataclass
class VolumeMetadata:
    name: str
    description: str
    units: str
    voxel_spacing_m: Tuple[float, float, float]
    array_shape: Tuple[int, int, int]
    dtype: str


@dataclass
class ImageMetadata:
    name: str
    description: str
    pixel_size_m: Tuple[float, float]
    array_shape: Tuple[int, int]
    dtype: str


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "sofc_microstructure_informed"


def ensure_clean_root() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_png(path: Path, array: np.ndarray, cmap_name: str = "gray") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5), dpi=300)
    plt.imshow(array, cmap=cmap_name, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr_ptp = np.ptp(arr)
    if arr_ptp < 1e-12:
        return np.zeros_like(arr)
    return arr / arr_ptp


# ---------------------------------------------------------------------------
# Synthetic microstructure generation
# ---------------------------------------------------------------------------


def generate_microstructure(seed: int = 42) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    volume_shape = (96, 96, 96)
    raw_noise = rng.normal(0.0, 1.0, size=volume_shape)

    # Introduce long-range correlations to mimic agglomerated particles
    low_freq = ndi.gaussian_filter(raw_noise, sigma=3)

    # Embed elongated defects by anisotropic filtering
    elongated = ndi.gaussian_filter(raw_noise, sigma=(1, 3, 6))
    composite = 0.6 * low_freq + 0.4 * elongated

    norm = normalize(composite)

    # Threshold for porosity and grains
    pore_threshold = 0.42
    solid_phase = (norm > pore_threshold).astype(np.uint8)

    # Remove isolated points to mimic realistic pore distribution
    cleaned = ndi.binary_opening(solid_phase, structure=np.ones((3, 3, 3)))
    cleaned = ndi.binary_closing(cleaned, structure=np.ones((3, 3, 3)))

    # Generate a finer grain structure via watershed-like distance transform
    distance = ndi.distance_transform_edt(cleaned)
    seeds = distance > 3
    grains, _ = ndi.label(seeds)

    porosity = 1 - cleaned

    return {
        "microstructure_scalar": norm,
        "solid_phase": cleaned.astype(np.uint8),
        "porosity_field": porosity.astype(np.uint8),
        "grain_ids": grains.astype(np.int32),
    }


def simulate_sintered_state(initial: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    solid = initial["solid_phase"].astype(bool)
    porosity = initial["porosity_field"].astype(float)

    # Sintering densification: shrink pores and activate targeted creep
    densified = ndi.binary_closing(solid, structure=np.ones((5, 5, 5)))
    densified = ndi.binary_opening(densified, structure=np.ones((3, 3, 3)))

    densified_porosity = 1 - densified.astype(np.uint8)

    # Compute volumetric strain surrogate from porosity change
    volumetric_strain = (porosity - densified_porosity) * 0.015

    warp_field = np.stack(
        [
            ndi.gaussian_filter(volumetric_strain, sigma=4) * scale
            for scale in (1.2e-3, -9.5e-4, 8.0e-4)
        ],
        axis=0,
    )

    return {
        "solid_phase": densified.astype(np.uint8),
        "porosity_field": densified_porosity.astype(np.uint8),
        "volumetric_strain": volumetric_strain.astype(np.float32),
        "warp_field": warp_field.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Material property tables & constitutive models
# ---------------------------------------------------------------------------


def build_material_tables(seed: int = 42) -> Tuple[pd.DataFrame, Dict]:
    temperatures = np.array([25, 400, 800, 1000])

    records = []
    property_definitions = [
        {
            "component": "8YSZ Electrolyte",
            "elastic_modulus_GPa": lambda T: 205 - 0.04 * (T - 25),
            "poisson_ratio": 0.23,
            "cte_ppm": lambda T: 9.8 + 0.0009 * (T - 25),
            "thermal_conductivity_WmK": lambda T: 2.0 + 0.0003 * (T - 25),
            "density_kgm3": 5900,
        },
        {
            "component": "Ni-YSZ Anode",
            "elastic_modulus_GPa": lambda T: 65 * math.exp(-0.0018 * (T - 25)),
            "poisson_ratio": 0.29,
            "cte_ppm": lambda T: 12.2 + 0.0011 * (T - 25),
            "thermal_conductivity_WmK": lambda T: 6.2 - 0.0025 * (T - 25),
            "density_kgm3": 6500,
        },
        {
            "component": "LSM Cathode",
            "elastic_modulus_GPa": lambda T: 48 - 0.008 * (T - 25),
            "poisson_ratio": 0.25,
            "cte_ppm": lambda T: 11.3 + 0.0007 * (T - 25),
            "thermal_conductivity_WmK": lambda T: 3.4 - 0.0012 * (T - 25),
            "density_kgm3": 6200,
        },
        {
            "component": "Crofer 22 APU",
            "elastic_modulus_GPa": lambda T: 165 - 0.025 * (T - 25),
            "poisson_ratio": 0.3,
            "cte_ppm": lambda T: 11.0 + 0.0009 * (T - 25),
            "thermal_conductivity_WmK": lambda T: 11.5 - 0.0031 * (T - 25),
            "density_kgm3": 7700,
        },
    ]

    for definition in property_definitions:
        for T in temperatures:
            records.append(
                {
                    "component": definition["component"],
                    "temperature_C": T,
                    "elastic_modulus_GPa": round(definition["elastic_modulus_GPa"](T), 3),
                    "poisson_ratio": definition["poisson_ratio"],
                    "cte_ppm": round(definition["cte_ppm"](T), 4),
                    "thermal_conductivity_WmK": round(
                        definition["thermal_conductivity_WmK"](T), 4
                    ),
                    "density_kgm3": definition["density_kgm3"],
                }
            )

    table = pd.DataFrame.from_records(records)

    constitutive = {
        "electrolyte": {
            "linear_elastic": {
                "E_ref_GPa": 205.0,
                "dEdT_GPa_per_C": -0.04,
                "poisson_ratio": 0.23,
            },
            "norton_bailey_creep": {
                "pre_exponential_B": 8.5e-12,
                "stress_exponent_n": 1.85,
                "activation_energy_Jmol": 3.85e5,
                "reference_temperature_C": 1000,
            },
            "targeted_creep_activation": {
                "strategy": "Temperature-gradient guided relaxation",
                "activation_temperature_C": 860,
                "control_variable": "volumetric strain energy density",
            },
        },
        "anode": {
            "viscoplastic": {
                "yield_strength_MPa": 48,
                "hardening_modulus_MPa": 120,
                "creep_B": 2.4e-10,
                "creep_n": 2.1,
            }
        },
        "cathode": {
            "kelvin_voigt": {
                "instantaneous_modulus_GPa": 50.0,
                "viscosity_GPa_s": 3.5e4,
            }
        },
        "interconnect": {
            "linear_elastic": {
                "E_ref_GPa": 165.0,
                "poisson_ratio": 0.30,
            }
        },
    }

    return table, constitutive


# ---------------------------------------------------------------------------
# Derived analytics
# ---------------------------------------------------------------------------


def compute_porosity_statistics(porosity_field: np.ndarray) -> pd.DataFrame:
    total_voxels = porosity_field.size
    porosity_fraction = porosity_field.sum() / total_voxels

    labeled, num = ndi.label(porosity_field)
    pore_volumes = ndi.sum(porosity_field, labeled, index=np.arange(1, num + 1))

    if pore_volumes.size == 0:
        return pd.DataFrame(
            {
                "metric": ["porosity_fraction"],
                "value": [porosity_fraction],
            }
        )

    pore_vol_sorted = np.sort(pore_volumes)[::-1]
    quantiles = np.quantile(pore_volumes, [0.25, 0.5, 0.75])

    stats = pd.DataFrame(
        {
            "metric": [
                "porosity_fraction",
                "num_pores",
                "largest_pore_voxels",
                "median_pore_voxels",
                "pore_volume_p25",
                "pore_volume_p75",
            ],
            "value": [
                porosity_fraction,
                int(num),
                float(pore_vol_sorted[0]),
                float(quantiles[1]),
                float(quantiles[0]),
                float(quantiles[2]),
            ],
        }
    )

    return stats


def compute_layer_thickness(solid_phase: np.ndarray, voxel_size_um: float) -> pd.DataFrame:
    # Approximate thickness along z by counting solid voxels per column
    layer_ids = ["anode", "electrolyte", "cathode"]
    layer_bounds = [
        (0, int(0.4 * solid_phase.shape[0])),
        (int(0.4 * solid_phase.shape[0]), int(0.55 * solid_phase.shape[0])),
        (int(0.55 * solid_phase.shape[0]), solid_phase.shape[0]),
    ]

    rows = []
    grid_to_sample = 50
    xs = np.linspace(0, solid_phase.shape[1] - 1, grid_to_sample, dtype=int)
    ys = np.linspace(0, solid_phase.shape[2] - 1, grid_to_sample, dtype=int)

    for (layer_name, (z0, z1)) in zip(layer_ids, layer_bounds):
        for x in xs:
            for y in ys:
                column = solid_phase[z0:z1, x, y]
                thickness = column.sum() * voxel_size_um
                rows.append(
                    {
                        "layer": layer_name,
                        "x_index": int(x),
                        "y_index": int(y),
                        "thickness_um": float(thickness),
                    }
                )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FEM-ready assets
# ---------------------------------------------------------------------------


def write_rectilinear_vtk(path: Path, scalar_field: np.ndarray, spacing: Tuple[float, float, float]):
    nz, ny, nx = scalar_field.shape
    x = np.linspace(0, spacing[0] * (nx - 1), nx)
    y = np.linspace(0, spacing[1] * (ny - 1), ny)
    z = np.linspace(0, spacing[2] * (nz - 1), nz)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="ascii") as fh:
        fh.write("# vtk DataFile Version 4.2\n")
        fh.write("SOFC microstructure-informed field\n")
        fh.write("ASCII\n")
        fh.write("DATASET RECTILINEAR_GRID\n")
        fh.write(f"DIMENSIONS {nx} {ny} {nz}\n")

        fh.write(f"X_COORDINATES {nx} float\n")
        fh.write(" ".join(f"{val:.6e}" for val in x) + "\n")
        fh.write(f"Y_COORDINATES {ny} float\n")
        fh.write(" ".join(f"{val:.6e}" for val in y) + "\n")
        fh.write(f"Z_COORDINATES {nz} float\n")
        fh.write(" ".join(f"{val:.6e}" for val in z) + "\n")

        fh.write("CELL_DATA {}\n".format((nx - 1) * (ny - 1) * (nz - 1)))
        fh.write("SCALARS porosity float 1\n")
        fh.write("LOOKUP_TABLE default\n")
        flattened = scalar_field[:-1, :-1, :-1].reshape(-1)
        fh.write(" \n".join(
            " ".join(f"{val:.6e}" for val in flattened[i : i + 6])
            for i in range(0, flattened.size, 6)
        ))
        fh.write("\n")


def write_initial_conditions(path: Path, temperature: np.ndarray, stress: np.ndarray, creep: np.ndarray):
    save_npz(path, temperature=temperature, residual_stress=stress, creep_strain=creep)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def build_dataset(seed: int) -> None:
    ensure_clean_root()

    voxel_spacing = (2.5e-6, 2.5e-6, 2.5e-6)  # 2.5 micrometers
    pixel_size_sem = (0.2e-6, 0.2e-6)

    initial_micro = generate_microstructure(seed)
    sintered_micro = simulate_sintered_state(initial_micro)

    # Save \u03bcCT volumes
    save_npz(
        DATASET_ROOT / "initial_state" / "microCT_volume_unfired.npz",
        **initial_micro,
        voxel_spacing_m=np.array(voxel_spacing),
    )
    save_npz(
        DATASET_ROOT / "post_mortem" / "microCT_volume_sintered.npz",
        **sintered_micro,
        voxel_spacing_m=np.array(voxel_spacing),
    )

    # Metadata
    volume_metadata = [
        VolumeMetadata(
            name="microCT_volume_unfired",
            description="3D porosity and grain identifiers for the unfired laminate",
            units="binary (solid=1, pore=0) and arbitrary",
            voxel_spacing_m=voxel_spacing,
            array_shape=initial_micro["solid_phase"].shape,
            dtype=str(initial_micro["solid_phase"].dtype),
        ),
        VolumeMetadata(
            name="microCT_volume_sintered",
            description="3D porosity and volumetric strain fields after sintering",
            units="binary (solid=1, pore=0) and strain",
            voxel_spacing_m=voxel_spacing,
            array_shape=sintered_micro["solid_phase"].shape,
            dtype=str(sintered_micro["solid_phase"].dtype),
        ),
    ]

    metadata_payload = {
        "dataset_name": "SOFC Microstructure-Informed Sintering Dataset",
        "version": "v1.0.0",
        "seed": seed,
        "topic": "A Closed-Loop, Microstructure-Informed DIC-FEM Framework for the Real-Time Mitigation of Sintering Stresses in Solid Oxide Fuel Cells through Targeted Creep Activation",
        "volumes": [asdict(meta) for meta in volume_metadata],
    }
    save_json(DATASET_ROOT / "dataset_description.json", metadata_payload)

    # SEM slices (initial & post)
    z_indices = [16, 32, 64]
    for idx in z_indices:
        slice_img = initial_micro["microstructure_scalar"][idx]
        save_png(
            DATASET_ROOT
            / "initial_state"
            / "SEM_cross_sections"
            / f"sem_cross_section_z{idx:03}.png",
            slice_img,
        )

    for idx in z_indices:
        slice_img = sintered_micro["volumetric_strain"][idx]
        save_png(
            DATASET_ROOT
            / "post_mortem"
            / "SEM_EDS"
            / f"strain_mapped_sem_z{idx:03}.png",
            slice_img,
            cmap_name="viridis",
        )

    # Synthetic EDS composition maps (3 channels: Zr, Ni, O)
    zr = normalize(initial_micro["microstructure_scalar"]) * 0.8
    ni = 0.6 * normalize(sintered_micro["volumetric_strain"])
    o = normalize(1 - initial_micro["microstructure_scalar"])
    eds_cube = np.stack((zr[48], ni[48], o[48]), axis=-1)
    save_npz(
        DATASET_ROOT / "post_mortem" / "SEM_EDS" / "eds_composition_map.npz",
        elemental_map=eds_cube,
        elements=np.array(["Zr", "Ni", "O"]),
    )
    save_png(
        DATASET_ROOT / "post_mortem" / "SEM_EDS" / "eds_composition_map.png",
        eds_cube,
        cmap_name="viridis",
    )

    # Material properties & constitutive models
    mat_table, constitutive = build_material_tables(seed)
    save_csv(DATASET_ROOT / "material_properties" / "thermo_mechanical_properties.csv", mat_table)
    save_json(DATASET_ROOT / "material_properties" / "constitutive_model_params.json", constitutive)

    # Porosity statistics, layer thickness
    porosity_stats_initial = compute_porosity_statistics(initial_micro["porosity_field"])
    porosity_stats_initial.insert(0, "state", "unfired")

    porosity_stats_sintered = compute_porosity_statistics(sintered_micro["porosity_field"])
    porosity_stats_sintered.insert(0, "state", "sintered")

    porosity_stats = pd.concat([porosity_stats_initial, porosity_stats_sintered], ignore_index=True)
    save_csv(DATASET_ROOT / "analysis" / "porosity_statistics.csv", porosity_stats)

    thickness_df = compute_layer_thickness(initial_micro["solid_phase"], voxel_size_um=voxel_spacing[0] * 1e6)
    save_csv(DATASET_ROOT / "analysis" / "layer_thickness_map.csv", thickness_df)

    # FEM ready fields
    write_rectilinear_vtk(
        DATASET_ROOT / "fem_ready" / "microstructure_porosity.vtr",
        initial_micro["porosity_field"].astype(np.float32),
        spacing=voxel_spacing,
    )

    temperature_field = 1073.15 * np.ones_like(initial_micro["microstructure_scalar"], dtype=np.float32)
    residual_stress = 30e6 * normalize(initial_micro["microstructure_scalar"])  # Pa
    creep_strain = sintered_micro["volumetric_strain"].astype(np.float32)
    write_initial_conditions(
        DATASET_ROOT / "fem_ready" / "initial_conditions_fields.npz",
        temperature_field,
        residual_stress,
        creep_strain,
    )

    # Synthetic DIC displacement field (2D surface)
    x = np.linspace(0, 1, 256)
    y = np.linspace(0, 1, 256)
    X, Y = np.meshgrid(x, y)
    displacement_u = 3.5e-4 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    displacement_v = -2.1e-4 * np.cos(2 * np.pi * X) * np.sin(2 * np.pi * Y)
    save_npz(
        DATASET_ROOT / "post_mortem" / "dic_surface_displacement.npz",
        u=displacement_u.astype(np.float32),
        v=displacement_v.astype(np.float32),
        x=x.astype(np.float32),
        y=y.astype(np.float32),
    )

    save_png(
        DATASET_ROOT / "post_mortem" / "dic_displacement_magnitude.png",
        np.sqrt(displacement_u**2 + displacement_v**2),
        cmap_name="magma",
    )

    # Boundary & process schedule metadata
    schedule = pd.DataFrame(
        {
            "step": ["sintering_cooldown", "steady_operation", "thermal_cycle_1", "thermal_cycle_2", "thermal_cycle_3"],
            "start_temperature_C": [1350, 800, 25, 25, 25],
            "end_temperature_C": [25, 800, 800, 800, 800],
            "rate_C_per_min": [-2.0, 0.0, 5.0, 5.0, 5.0],
            "dwell_minutes": [0, 0, 120, 120, 120],
            "creep_activation": ["inactive", "targeted", "targeted", "targeted", "targeted"],
        }
    )
    save_csv(DATASET_ROOT / "fem_ready" / "thermal_creep_schedule.csv", schedule)

    boundary_conditions = {
        "symmetry_planes": ["x=0", "y=0"],
        "assembly_pressure_MPa": 0.2,
        "support_condition": "Simply supported at bottom face",
        "dic_surface_reference": "post_mortem/dic_surface_displacement.npz",
    }
    save_json(DATASET_ROOT / "fem_ready" / "boundary_conditions.json", boundary_conditions)

    # Aggregate metadata for images
    image_metadata = []
    for idx in z_indices:
        image_metadata.append(
            ImageMetadata(
                name=f"sem_cross_section_z{idx:03}",
                description=f"SEM-like grayscale slice at z-index {idx}",
                pixel_size_m=pixel_size_sem,
                array_shape=initial_micro["microstructure_scalar"][idx].shape,
                dtype="float32",
            )
        )
    for idx in z_indices:
        image_metadata.append(
            ImageMetadata(
                name=f"strain_mapped_sem_z{idx:03}",
                description=f"Post-sintering strain mapped pseudo-SEM at z-index {idx}",
                pixel_size_m=pixel_size_sem,
                array_shape=sintered_micro["volumetric_strain"][idx].shape,
                dtype="float32",
            )
        )

    save_json(
        DATASET_ROOT / "metadata" / "image_assets.json",
        {"images": [asdict(meta) for meta in image_metadata]},
    )

    # Validation metrics referencing initial vs sintered differences
    validation_metrics = {
        "warpage_curvature_1mm": float(np.mean(sintered_micro["volumetric_strain"]) * 1e3),
        "porosity_reduction_percent": float(
            100
            * (
                initial_micro["porosity_field"].mean()
                - sintered_micro["porosity_field"].mean()
            )
            / initial_micro["porosity_field"].mean()
        ),
        "max_creep_strain": float(np.max(sintered_micro["volumetric_strain"])),
        "dic_peak_displacement_mm": float(np.max(np.sqrt(displacement_u**2 + displacement_v**2)) * 1e3),
    }
    save_json(DATASET_ROOT / "analysis" / "validation_metrics.json", validation_metrics)

    # Package zip for download convenience
    zip_path = DATASET_ROOT.parent / "sofc_microstructure_informed_v1.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in DATASET_ROOT.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, file_path.relative_to(DATASET_ROOT.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate microstructure-informed SOFC dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(seed=args.seed)
    print(f"Dataset generated at: {DATASET_ROOT}")


if __name__ == "__main__":
    main()
