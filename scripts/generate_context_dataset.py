#!/usr/bin/env python3
"""
Generate Dataset 2: Process and Material Parameters (The "Context" Dataset)

This script synthesizes a design-of-experiments (DOE) over geometry, material, and process
parameters relevant to residual stress in multilayer ceramic (e.g., SOFC-like) laminates.

Outputs:
- data/context/context_dataset.csv: tabular dataset with one row per DOE sample
- data/context/context_schema.json: JSON schema describing each field

Notes:
- Values are sampled using Latin Hypercube Sampling to cover the parameter space uniformly
- Temperature-dependent Young's modulus is represented via a reference E0 and linear temp coeff
- Sintering profile is parameterized by ramp rates, hold temperatures/durations, and cooling rate
- Atmosphere captured categorically (air, inert, reducing) and as oxygen partial pressure proxy

Reproducibility:
- Seed can be set via CLI option --seed

"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception as exc:
    raise SystemExit(
        "numpy is required. Please install with: pip install numpy"
    ) from exc


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a Latin Hypercube sample in [0,1]^n_dims.

    Simple implementation: divide [0,1] into n_samples strata per dim, shuffle per-dim.
    """
    # Base points: (i + u) / n_samples per dimension
    H = np.zeros((n_samples, n_dims), dtype=float)
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        u = rng.random(n_samples)
        H[:, j] = (perm + u) / n_samples
    return H


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


class Atmosphere(str, Enum):
    AIR = "air"
    INERT = "inert"
    REDUCING = "reducing"


# ----------------------------
# Parameter spaces (engineering ranges)
# ----------------------------

# Geometry (meters)
GEOM_RANGES = {
    # Plate planform dimensions
    "plate_length_m": (0.02, 0.20),  # 20 mm to 200 mm
    "plate_width_m": (0.02, 0.20),
    # Individual layer thicknesses (meters)
    "anode_thickness_m": (50e-6, 800e-6),
    "electrolyte_thickness_m": (5e-6, 100e-6),
    "cathode_thickness_m": (10e-6, 200e-6),
    # Initial green state prior to sintering
    "green_density_rel": (0.4, 0.65),  # relative density fraction
    "green_length_m": (0.0205, 0.202),  # slightly larger pre-sinter
    "green_width_m": (0.0205, 0.202),
}

# Material properties per layer
MATERIAL_RANGES = {
    # Young's modulus at 25C (Pa)
    "anode_E0_Pa": (20e9, 120e9),
    "electrolyte_E0_Pa": (100e9, 300e9),
    "cathode_E0_Pa": (30e9, 180e9),
    # Linear temperature coefficient for E (1/K)
    "anode_E_temp_coeff": (-4e-4, -2e-5),
    "electrolyte_E_temp_coeff": (-2e-4, -1e-5),
    "cathode_E_temp_coeff": (-3e-4, -1e-5),
    # Poisson's ratio (dimensionless)
    "anode_nu": (0.18, 0.33),
    "electrolyte_nu": (0.18, 0.30),
    "cathode_nu": (0.18, 0.33),
    # Coefficient of Thermal Expansion (1/K)
    "anode_cte": (8e-6, 14e-6),
    "electrolyte_cte": (9e-6, 12e-6),
    "cathode_cte": (9e-6, 14e-6),
    # Sintering shrinkage model parameters
    # We use: linear_shrinkage = A * sigmoid((T - T_onset) / width), where width fixed here
    "anode_shrink_A": (0.06, 0.18),   # max linear shrink fraction
    "anode_shrink_Ton_C": (700.0, 1100.0),
    "electrolyte_shrink_A": (0.04, 0.12),
    "electrolyte_shrink_Ton_C": (900.0, 1250.0),
    "cathode_shrink_A": (0.03, 0.12),
    "cathode_shrink_Ton_C": (700.0, 1100.0),
    # Creep parameters (Norton): strain_rate = A * sigma^n * exp(-Q/(R*T))
    "anode_creep_A": (1e-25, 1e-18),
    "anode_creep_n": (1.0, 3.5),
    "anode_creep_Q_kJmol": (150.0, 500.0),
    "electrolyte_creep_A": (1e-30, 1e-20),
    "electrolyte_creep_n": (1.0, 3.0),
    "electrolyte_creep_Q_kJmol": (250.0, 600.0),
    "cathode_creep_A": (1e-28, 1e-19),
    "cathode_creep_n": (1.0, 3.5),
    "cathode_creep_Q_kJmol": (150.0, 500.0),
}

# Process parameters: sintering profile and atmosphere
PROCESS_RANGES = {
    # Ramp rates (C/min)
    "ramp1_rate_C_per_min": (1.0, 10.0),
    "ramp2_rate_C_per_min": (1.0, 10.0),
    # Hold temperatures and durations
    "hold1_temp_C": (900.0, 1350.0),
    "hold1_minutes": (30.0, 240.0),
    "hold2_temp_C": (900.0, 1400.0),  # optional second hold
    "hold2_minutes": (0.0, 180.0),    # 0 means no second hold
    # Cooling rate
    "cool_rate_C_per_min": (1.0, 15.0),
    # Atmosphere selection; we will sample a categorical plus oxygen partial pressure proxy
    # pO2 proxy in atm. For air ~0.21, inert ~ ~1e-5-1e-3 (trace), reducing 1e-20-1e-10
    "pO2_atm": (1e-20, 0.21),
}

ATMOSPHERES = [Atmosphere.AIR.value, Atmosphere.INERT.value, Atmosphere.REDUCING.value]


# ----------------------------
# Sampling helpers
# ----------------------------

def scale_unit(value01: float, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return lo + (hi - lo) * float(value01)


def sample_categorical(u: float, categories: List[str]) -> str:
    idx = min(int(u * len(categories)), len(categories) - 1)
    return categories[idx]


# ----------------------------
# Schema definition
# ----------------------------

def build_schema(fields: List[Tuple[str, str, str]]) -> Dict:
    """Create a JSON schema-like dict for documentation and validation hints."""
    properties = {}
    required = []
    for name, dtype, desc in fields:
        properties[name] = {"type": dtype, "description": desc}
        required.append(name)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Context Dataset Schema",
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ----------------------------
# Main generator
# ----------------------------

def generate_context_dataset(n_samples: int, seed: int) -> Tuple[List[str], List[List[float]]]:
    rng = np.random.default_rng(seed)

    # Define field order and descriptions
    fields: List[Tuple[str, str, str]] = []

    # Geometry
    for key, (lo, hi) in GEOM_RANGES.items():
        fields.append((key, "number", f"Range: [{lo}, {hi}]") )

    # Material props
    for key, (lo, hi) in MATERIAL_RANGES.items():
        fields.append((key, "number", f"Range: [{lo}, {hi}]") )

    # Process
    for key, (lo, hi) in PROCESS_RANGES.items():
        fields.append((key, "number", f"Range: [{lo}, {hi}]") )

    # Atmosphere category explicitly
    fields.append(("atmosphere", "string", f"One of: {', '.join(ATMOSPHERES)}"))

    # Build schema once
    schema = build_schema(fields)

    # Latin hypercube over all continuous numeric dims EXCEPT the categorical atmosphere.
    # Determine how many numeric dims: all but atmosphere
    numeric_fields = [name for name, dtype, _ in fields if dtype == "number"]
    n_dims = len(numeric_fields)
    H = latin_hypercube(n_samples=n_samples, n_dims=n_dims, rng=rng)

    # For atmosphere, sample separately per row
    u_atm = rng.random(n_samples)

    # Map to actual values
    header = [name for name, _, _ in fields]
    rows: List[List[float]] = []

    # Build bound map for scaling
    bounds: Dict[str, Tuple[float, float]] = {}
    bounds.update(GEOM_RANGES)
    bounds.update(MATERIAL_RANGES)
    bounds.update(PROCESS_RANGES)

    for i in range(n_samples):
        row_values: Dict[str, float | str] = {}
        # scale numeric dims
        for j, fname in enumerate(numeric_fields):
            row_values[fname] = scale_unit(H[i, j], bounds[fname])
        # atmosphere categorical
        row_values["atmosphere"] = sample_categorical(u_atm[i], ATMOSPHERES)
        # Assemble per header order
        rows.append([row_values[name] for name in header])

    return header, rows, schema


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Context dataset for residual stress modeling")
    p.add_argument("--out-dir", default="/workspace/data/context", help="Output directory")
    p.add_argument("--samples", type=int, default=2000, help="Number of DOE samples to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    header, rows, schema = generate_context_dataset(n_samples=args.samples, seed=args.seed)

    csv_path = os.path.join(args.out_dir, "context_dataset.csv")
    schema_path = os.path.join(args.out_dir, "context_schema.json")

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    # Write schema
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Wrote {len(rows)} samples to {csv_path}")
    print(f"Wrote schema to {schema_path}")


if __name__ == "__main__":
    main()
