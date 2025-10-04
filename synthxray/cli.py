from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

from .microstructure import generate_grain_labels, compute_boundary_mask, generate_initial_porosity
from .stress import generate_residual_stress_field
from .evolution import EvolutionConfig, simulate_creep
from .xrd import XRDConfig, simulate_xrd_image
from .io import save_tiff_stack, save_array_npy, save_json, ensure_dir


def _parse_shape(text: str) -> Tuple[int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("shape must be Z,Y,X")
    return (parts[0], parts[1], parts[2])


def _parse_vector(text: str) -> Tuple[float, float, float]:
    parts = [float(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise ValueError("vector must be x,y,z components")
    return (parts[0], parts[1], parts[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 4D CT and XRD dataset for SOFC creep studies.")
    parser.add_argument("--outdir", type=str, default="/workspace/synthxray_output")
    parser.add_argument("--shape", type=str, default="96,128,128", help="Z,Y,X")
    parser.add_argument("--voxel-size-um", type=float, default=3.0)
    parser.add_argument("--time-steps", type=int, default=8)
    parser.add_argument("--dt-seconds", type=float, default=300.0)
    parser.add_argument("--temperature-c", type=float, default=700.0)
    parser.add_argument("--stress-mpa", type=float, default=50.0)
    parser.add_argument("--heterogeneity", type=float, default=0.35)
    parser.add_argument("--load-dir", type=str, default="0,0,1")
    parser.add_argument("--porosity-frac", type=float, default=0.02)
    parser.add_argument("--approx-grains", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--phases", type=str, default="Ni:1.0,FeCr:0.6,YSZ:0.5")

    args = parser.parse_args()

    outdir = args.outdir
    shape = _parse_shape(args.shape)
    voxel_um = float(args.voxel_size_um)
    load_vec = _parse_vector(args.load_dir)

    ensure_dir(outdir)
    ensure_dir(os.path.join(outdir, "tomo"))
    ensure_dir(os.path.join(outdir, "xrd"))

    # Microstructure
    labels = generate_grain_labels(shape, approx_num_grains=int(args.approx_grains), seed=args.seed)
    boundaries = compute_boundary_mask(labels)
    pores0 = generate_initial_porosity(shape, target_porosity_fraction=float(args.porosity_frac), boundary_mask=boundaries, voxel_size_um=voxel_um, seed=args.seed + 1)

    # Stress/strain
    stress = generate_residual_stress_field(shape, base_stress_mpa=float(args.stress_mpa), heterogeneity=float(args.heterogeneity), load_direction=load_vec, seed=args.seed + 2)

    # Time evolution
    evo_cfg = EvolutionConfig(
        num_time_steps=int(args.time_steps),
        dt_seconds=float(args.dt_seconds),
        nucleation_rate=0.003,
        growth_rate=0.9,
        crack_bias_strength=0.7,
        load_direction=load_vec,
        random_seed=args.seed + 3,
    )
    series = simulate_creep(labels, boundaries, pores0, stress.equivalent_stress_mpa, evo_cfg)

    # Save CT series and per-step XRD
    phases: Dict[str, float] = {}
    for token in args.phases.split(","):
        if ":" in token:
            name, sval = token.split(":", 1)
            phases[name.strip()] = float(sval)
    xrd_cfg = XRDConfig(energy_keV=60.0, detector_distance_mm=800.0, pixel_size_um=100.0, image_size=(1024, 1024), texture_anisotropy=0.25, microstrain=float(stress.elastic_strain.mean()))

    time_points_s = []
    porosity_fracs = []
    for t, step in enumerate(series):
        tiff_path = os.path.join(outdir, "tomo", f"tomo_t{t:03d}.tif")
        save_tiff_stack(step.ct_volume_uint16, tiff_path)

        xrd_img = simulate_xrd_image(phases, xrd_cfg)
        xrd_path = os.path.join(outdir, "xrd", f"xrd_t{t:03d}.tif")
        save_tiff_stack(xrd_img, xrd_path)

        time_points_s.append(t * evo_cfg.dt_seconds)
        porosity_fracs.append(step.porosity_fraction)

    # Save fields and metadata
    save_array_npy(stress.equivalent_stress_mpa.astype(np.float32), os.path.join(outdir, "residual_stress_mpa.npy"))
    save_array_npy(stress.elastic_strain.astype(np.float32), os.path.join(outdir, "elastic_strain.npy"))
    save_array_npy(labels.astype(np.int32), os.path.join(outdir, "grain_labels.npy"))
    save_array_npy(boundaries.astype(np.uint8), os.path.join(outdir, "grain_boundaries.npy"))
    save_array_npy(pores0.astype(np.uint8), os.path.join(outdir, "initial_porosity.npy"))

    # Metadata
    sample_dims_mm = [d * voxel_um / 1000.0 for d in shape[::-1]]  # X,Y,Z in mm

    metadata = {
        "dataset": "Synthetic SOFC creep CT+XRD",
        "version": "0.1.0",
        "operational_parameters": {
            "temperature_c": float(args.temperature_c),
            "applied_stress_mpa": float(args.stress_mpa),
            "time_points_s": time_points_s,
        },
        "material_specifications": {
            "alloy": "Fe-22Cr (Crofer-like) + Ni-YSZ anode",
            "alloy_composition_wt%": {"Fe": 70.0, "Cr": 22.0, "Ni": 4.0, "Others": 4.0},
            "heat_treatment": "solution-treated, air-cooled (synthetic)",
            "initial_grain_count": int(labels.max()),
        },
        "sample_geometry": {
            "voxel_size_um": voxel_um,
            "volume_shape_zyx": list(shape),
            "dimensions_mm_xyz": sample_dims_mm,
        },
        "files": {
            "ct_series_dir": os.path.join(outdir, "tomo"),
            "xrd_series_dir": os.path.join(outdir, "xrd"),
            "residual_stress_field": os.path.join(outdir, "residual_stress_mpa.npy"),
            "elastic_strain_field": os.path.join(outdir, "elastic_strain.npy"),
            "grain_labels": os.path.join(outdir, "grain_labels.npy"),
            "grain_boundaries": os.path.join(outdir, "grain_boundaries.npy"),
            "initial_porosity": os.path.join(outdir, "initial_porosity.npy"),
        },
        "xrd_config": {
            "phases": phases,
            "energy_keV": 60.0,
            "detector_distance_mm": 800.0,
            "pixel_size_um": 100.0,
            "image_size": [1024, 1024],
        },
        "notes": "Synthetic data for method development and validation."
    }

    save_json(metadata, os.path.join(outdir, "metadata.json"))


if __name__ == "__main__":
    main()
