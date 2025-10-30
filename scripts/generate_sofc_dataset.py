#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, List

import numpy as np

# Local package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from sofc_dataset import generate_fields, get_parameter_space, sample_parameter_sets  # noqa: E402


def write_schema(schema_path: Path) -> None:
    schema = {
        "description": "Synthetic high-fidelity-like multi-physics SOFC dataset (3D fields)",
        "field_shapes": "(nz, ny, nx)",
        "fields": {
            # Electrochemical
            "current_density_A_per_cm2": "float32",
            "overpotential_V": "float32",
            # Thermal
            "temperature_K": "float32",
            # Species
            "H2_molfrac": "float32",
            "H2O_molfrac": "float32",
            # Mechanical
            "von_mises_stress_Pa": "float32",
            "strain_exx": "float32",
            "strain_eyy": "float32",
            "strain_ezz": "float32",
            "strain_exy": "float32",
            "strain_exz": "float32",
            "strain_eyz": "float32",
            "displacement_ux_m": "float32",
            "displacement_uy_m": "float32",
            "displacement_uz_m": "float32",
            # Coordinates/aux
            "coord_x_m": "float32",
            "coord_y_m": "float32",
            "coord_z_m": "float32",
            "layer_index": "int16",
        },
        "notes": {
            "units": {
                "current_density_A_per_cm2": "A/cm^2",
                "overpotential_V": "V",
                "temperature_K": "K",
                "H2_molfrac": "-",
                "H2O_molfrac": "-",
                "von_mises_stress_Pa": "Pa",
                "displacement_ux_m": "m",
                "displacement_uy_m": "m",
                "displacement_uz_m": "m",
            },
            "layer_index_map": {"0": "anode", "1": "electrolyte", "2": "cathode", "3": "interconnect", "4": "sealant"},
        },
    }
    schema_path.write_text(json.dumps(schema, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic SOFC multi-physics dataset (3D fields)")
    ap.add_argument("--n", type=int, default=10, help="number of simulation runs")
    ap.add_argument("--nx", type=int, default=32, help="grid points in x")
    ap.add_argument("--ny", type=int, default=32, help="grid points in y")
    ap.add_argument("--nz", type=int, default=16, help="grid points in z")
    ap.add_argument("--out", type=str, default="/workspace/datasets/sofc_hifi", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--zip", action="store_true", help="zip the dataset directory after generation")
    args = ap.parse_args()

    out_dir = Path(args.out)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.jsonl"
    schema_path = out_dir / "schema.json"
    meta_f = meta_path.open("w", encoding="utf-8")
    try:
        write_schema(schema_path)

        # Sample inputs
        param_sets: List[Dict[str, float | str]] = sample_parameter_sets(args.n, seed=args.seed)
        grid = dict(nx=args.nx, ny=args.ny, nz=args.nz)

        for i, params in enumerate(param_sets):
            run_id = f"run_{i:06d}"
            t0 = time.time()
            # Deterministic per-run seed
            run_seed = (args.seed or 0) + i * 9973

            fields = generate_fields(params, grid=type("Grid", (), grid), seed=run_seed)  # type: ignore[arg-type]

            # Save fields
            run_file = runs_dir / f"{run_id}.npz"
            np.savez_compressed(run_file, **fields)

            # Metadata line
            record = {
                "run_id": run_id,
                "file": str(run_file.relative_to(out_dir)),
                "grid": grid,
                "inputs": params,
                "arrays": list(fields.keys()),
            }
            meta_f.write(json.dumps(record) + "\n")
            dt = time.time() - t0
            print(f"[OK] {run_id} -> {run_file.name} in {dt:.2f}s")
    finally:
        meta_f.close()

    if args.zip:
        zip_path = out_dir.with_suffix(".zip")
        print(f"Zipping dataset to {zip_path} ...")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    fpath = Path(root) / fn
                    arcname = fpath.relative_to(out_dir.parent)
                    zf.write(fpath, arcname=str(arcname))
        print(f"[ZIP] {zip_path}")


if __name__ == "__main__":
    main()
