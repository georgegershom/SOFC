from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from sofc_sim.lhs import LatinHypercubeSampler, build_param_specs
from sofc_sim.fabricator import GridSpec, FieldFabricator, params_to_jsonable


RAW_PARAM_SPECS: Dict[str, Dict[str, Any]] = {
    # Operating conditions
    "current_density_A_per_cm2": {"min": 0.2, "max": 1.5, "scale": "linear"},
    "cell_voltage_V": {"min": 0.6, "max": 0.9, "scale": "linear"},
    "air_flow_sccm": {"min": 200, "max": 1500, "scale": "log", "dtype": "int"},
    "fuel_flow_sccm": {"min": 100, "max": 800, "scale": "log", "dtype": "int"},
    "inlet_temp_air_K": {"min": 650, "max": 1023, "scale": "linear"},
    "inlet_temp_fuel_K": {"min": 650, "max": 1023, "scale": "linear"},

    # Geometric parameters
    "thickness_anode_um": {"min": 200, "max": 1000, "scale": "linear"},
    "thickness_electrolyte_um": {"min": 5, "max": 20, "scale": "linear"},
    "thickness_cathode_um": {"min": 30, "max": 100, "scale": "linear"},
    "thickness_interconnect_um": {"min": 200, "max": 1000, "scale": "linear"},
    "thickness_sealant_um": {"min": 50, "max": 300, "scale": "linear"},
    "active_area_cm2": {"min": 4.0, "max": 100.0, "scale": "log"},
    "channel_pitch_mm": {"min": 0.5, "max": 3.0, "scale": "linear"},
    "channel_land_fraction": {"min": 0.3, "max": 0.7, "scale": "linear"},

    # Material properties - anode
    "anode_porosity": {"min": 0.25, "max": 0.45, "scale": "linear"},
    "anode_permeability_m2": {"min": 1e-15, "max": 1e-12, "scale": "log"},
    "anode_ionic_conductivity_Spm": {"min": 0.1, "max": 1.0, "scale": "log"},
    "anode_electronic_conductivity_Spm": {"min": 1e4, "max": 1e6, "scale": "log"},
    "anode_youngs_modulus_Pa": {"min": 10e9, "max": 50e9, "scale": "log"},
    "anode_cte_per_K": {"min": 10e-6, "max": 15e-6, "scale": "linear"},

    # Material properties - electrolyte
    "electrolyte_porosity": {"min": 0.01, "max": 0.1, "scale": "linear"},
    "electrolyte_permeability_m2": {"min": 1e-18, "max": 1e-15, "scale": "log"},
    "electrolyte_ionic_conductivity_Spm": {"min": 1.0, "max": 10.0, "scale": "log"},
    "electrolyte_electronic_conductivity_Spm": {"min": 1e-3, "max": 1e-1, "scale": "log"},
    "electrolyte_youngs_modulus_Pa": {"min": 100e9, "max": 220e9, "scale": "log"},
    "electrolyte_cte_per_K": {"min": 8e-6, "max": 12e-6, "scale": "linear"},

    # Material properties - cathode
    "cathode_porosity": {"min": 0.25, "max": 0.45, "scale": "linear"},
    "cathode_permeability_m2": {"min": 1e-15, "max": 1e-12, "scale": "log"},
    "cathode_ionic_conductivity_Spm": {"min": 0.1, "max": 2.0, "scale": "log"},
    "cathode_electronic_conductivity_Spm": {"min": 1e3, "max": 1e5, "scale": "log"},
    "cathode_youngs_modulus_Pa": {"min": 80e9, "max": 150e9, "scale": "log"},
    "cathode_cte_per_K": {"min": 11e-6, "max": 15e-6, "scale": "linear"},

    # Material properties - interconnect
    "interconnect_porosity": {"min": 0.0, "max": 0.05, "scale": "linear"},
    "interconnect_permeability_m2": {"min": 1e-20, "max": 1e-16, "scale": "log"},
    "interconnect_ionic_conductivity_Spm": {"min": 1e-6, "max": 1e-4, "scale": "log"},
    "interconnect_electronic_conductivity_Spm": {"min": 1e6, "max": 2e6, "scale": "log"},
    "interconnect_youngs_modulus_Pa": {"min": 150e9, "max": 220e9, "scale": "log"},
    "interconnect_cte_per_K": {"min": 10e-6, "max": 13e-6, "scale": "linear"},

    # Material properties - sealant
    "sealant_porosity": {"min": 0.05, "max": 0.2, "scale": "linear"},
    "sealant_permeability_m2": {"min": 1e-18, "max": 1e-15, "scale": "log"},
    "sealant_ionic_conductivity_Spm": {"min": 1e-8, "max": 1e-6, "scale": "log"},
    "sealant_electronic_conductivity_Spm": {"min": 1e-4, "max": 1.0, "scale": "log"},
    "sealant_youngs_modulus_Pa": {"min": 10e9, "max": 40e9, "scale": "log"},
    "sealant_cte_per_K": {"min": 7e-6, "max": 10e-6, "scale": "linear"},
}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_npz(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    np.savez_compressed(str(path), **arrays)


def build_index_row(run_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
    row = {"run_id": run_id}
    for k, v in params.items():
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
        else:
            row[k] = v
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic high-fidelity SOFC multi-physics dataset (fabricated).")
    parser.add_argument("--out", type=str, default="/workspace/sofc_hifi_dataset", help="输出数据集目录")
    parser.add_argument("--num-runs", type=int, default=5, help="仿真样本数量")
    parser.add_argument("--nx", type=int, default=48, help="网格 X 方向尺寸")
    parser.add_argument("--ny", type=int, default=32, help="网格 Y 方向尺寸")
    parser.add_argument("--nz", type=int, default=24, help="网格 Z 方向尺寸")
    parser.add_argument("--seed", type=int, default=1234, help="随机种子")
    args = parser.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)

    grid = GridSpec(nx=args.nx, ny=args.ny, nz=args.nz)

    # Write grid meta
    with open(out_root / "grid.json", "w", encoding="utf-8") as f:
        json.dump({"nx": grid.nx, "ny": grid.ny, "nz": grid.nz}, f, indent=2)

    # Build sampler
    specs = build_param_specs(RAW_PARAM_SPECS)
    sampler = LatinHypercubeSampler(specs, seed=args.seed)
    samples = sampler.sample(args.num_runs)

    # Fabricator
    fabricator = FieldFabricator(grid=grid, seed=args.seed)

    # Prepare index.csv
    index_path = out_root / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as fidx:
        writer = csv.DictWriter(fidx, fieldnames=["run_id"] + list(RAW_PARAM_SPECS.keys()))
        writer.writeheader()

        for i, params in enumerate(samples, start=1):
            run_dir = out_root / f"run_{i:06d}"
            ensure_dir(run_dir)

            # Save inputs
            with open(run_dir / "inputs.json", "w", encoding="utf-8") as fin:
                json.dump(params_to_jsonable(params), fin, indent=2)

            # Generate fields
            fields = fabricator.generate_fields(params)

            # Save fields
            save_npz(run_dir / "fields.npz", fields)

            # Update index
            writer.writerow(build_index_row(i, params))

    print(f"Dataset generated at: {out_root}")


if __name__ == "__main__":
    main()
