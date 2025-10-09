#!/usr/bin/env python3
"""
Generator for Mechanical Boundary Conditions dataset.

Source fields represent experimental setup details such as fixture type and applied stack pressure.
Outputs a CSV file with one row per experiment/condition.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


RANDOM_SEED = 42

# Controlled vocabularies / enumerations for fixtures and constraints
FIXTURE_TYPES = [
    "uniaxial_compression_fixture",
    "biaxial_fixture",
    "triaxial_cell",
    "four_point_bending",
    "three_point_bending",
    "shear_fixture_single_lap",
    "shear_fixture_double_lap",
    "custom_fixture",
]

BOUNDARY_CONSTRAINTS = [
    "fixed_fixed",
    "fixed_free",
    "pinned_pinned",
    "pinned_fixed",
    "clamped",
    "simply_supported",
]

LOAD_TYPES = [
    "compressive",
    "tensile",
    "shear",
    "bending",
    "torsional",
]

PRESSURE_UNITS = "MPa"  # stack pressure units
FORCE_UNITS = "N"
MOMENT_UNITS = "N*m"


@dataclass
class MechanicalBoundaryCondition:
    condition_id: str
    fixture_type: str
    boundary_constraint: str
    stack_pressure_value: float  # in MPa
    load_type: str
    applied_force_value: Optional[float]  # in Newtons (for tension/compression/shear)
    applied_moment_value: Optional[float]  # in N*m (for bending/torsion)
    normal_force_direction: Optional[str]  # e.g., +z, -z
    shear_plane: Optional[str]  # e.g., xy, yz, zx
    temperature_celsius: float
    notes: str

    def to_row(self) -> Dict[str, Any]:
        row = asdict(self)
        row.update(
            {
                "stack_pressure_units": PRESSURE_UNITS,
                "applied_force_units": FORCE_UNITS if self.applied_force_value is not None else "",
                "applied_moment_units": MOMENT_UNITS if self.applied_moment_value is not None else "",
            }
        )
        return row


def generate_conditions(
    num_rows: int,
    seed: int = RANDOM_SEED,
    min_pressure_mpa: float = 0.1,
    max_pressure_mpa: float = 10.0,
    temp_range: tuple[float, float] = (20.0, 120.0),
) -> List[MechanicalBoundaryCondition]:
    random.seed(seed)

    conditions: List[MechanicalBoundaryCondition] = []

    for i in range(1, num_rows + 1):
        fixture_type = random.choice(FIXTURE_TYPES)
        boundary_constraint = random.choice(BOUNDARY_CONSTRAINTS)
        load_type = random.choice(LOAD_TYPES)
        stack_pressure_value = round(random.uniform(min_pressure_mpa, max_pressure_mpa), 3)
        temperature_celsius = round(random.uniform(*temp_range), 1)

        applied_force_value: Optional[float] = None
        applied_moment_value: Optional[float] = None
        normal_force_direction: Optional[str] = None
        shear_plane: Optional[str] = None

        if load_type in {"compressive", "tensile", "shear"}:
            applied_force_value = round(random.uniform(10.0, 5000.0), 2)
        if load_type in {"bending", "torsional"}:
            applied_moment_value = round(random.uniform(1.0, 500.0), 2)

        if load_type in {"compressive", "tensile"}:
            normal_force_direction = random.choice(["+z", "-z", "+y", "-y", "+x", "-x"])
        if load_type == "shear":
            shear_plane = random.choice(["xy", "yz", "zx"]) 

        condition = MechanicalBoundaryCondition(
            condition_id=f"MBC-{i:05d}",
            fixture_type=fixture_type,
            boundary_constraint=boundary_constraint,
            stack_pressure_value=stack_pressure_value,
            load_type=load_type,
            applied_force_value=applied_force_value,
            applied_moment_value=applied_moment_value,
            normal_force_direction=normal_force_direction,
            shear_plane=shear_plane,
            temperature_celsius=temperature_celsius,
            notes="synthetic sample generated",
        )
        conditions.append(condition)

    return conditions


def write_csv(conditions: List[MechanicalBoundaryCondition], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV header
    fieldnames = [
        "condition_id",
        "fixture_type",
        "boundary_constraint",
        "stack_pressure_value",
        "stack_pressure_units",
        "load_type",
        "applied_force_value",
        "applied_force_units",
        "applied_moment_value",
        "applied_moment_units",
        "normal_force_direction",
        "shear_plane",
        "temperature_celsius",
        "notes",
        "source_metadata",
        "created_utc",
    ]

    # Basic source metadata example
    source_metadata = {
        "source": "experimental_setup_details",
        "schema_version": "1.0",
        "fixture_vocab": FIXTURE_TYPES,
        "boundary_constraints_vocab": BOUNDARY_CONSTRAINTS,
        "load_types_vocab": LOAD_TYPES,
        "units": {
            "stack_pressure": PRESSURE_UNITS,
            "force": FORCE_UNITS,
            "moment": MOMENT_UNITS,
        },
    }

    created_utc = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition in conditions:
            row = condition.to_row()
            row["source_metadata"] = json.dumps(source_metadata, separators=(",", ":"))
            row["created_utc"] = created_utc
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Mechanical Boundary Conditions dataset CSV")
    parser.add_argument("--rows", type=int, default=250, help="Number of rows to generate")
    parser.add_argument("--out", type=Path, default=Path("data/mechanical_boundary_conditions.csv"), help="Output CSV path")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--min-pressure", type=float, default=0.1, help="Minimum stack pressure in MPa")
    parser.add_argument("--max-pressure", type=float, default=10.0, help="Maximum stack pressure in MPa")
    parser.add_argument("--min-temp", type=float, default=20.0, help="Minimum temperature in Celsius")
    parser.add_argument("--max-temp", type=float, default=120.0, help="Maximum temperature in Celsius")

    args = parser.parse_args()
    temp_range = (args.min_temp, args.max_temp)

    conditions = generate_conditions(
        num_rows=args.rows,
        seed=args.seed,
        min_pressure_mpa=args.min_pressure,
        max_pressure_mpa=args.max_pressure,
        temp_range=temp_range,
    )

    write_csv(conditions, args.out)
    print(f"Wrote {len(conditions)} rows to {args.out}")


if __name__ == "__main__":
    main()
