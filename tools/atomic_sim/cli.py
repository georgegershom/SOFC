from __future__ import annotations

import argparse
import os
from typing import List

from .common import MATERIAL_DATABASE, set_global_seed, temperature_grid
from .dft_generator import (
    generate_activation_barriers,
    generate_defect_formation_energies,
    generate_grain_boundary_energies,
    generate_surface_energies,
)
from .md_generator import (
    generate_dislocation_mobility,
    generate_gb_sliding_curves,
)


DEFAULT_MATERIALS: List[str] = list(MATERIAL_DATABASE.keys())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fabricate DFT-like and MD-like atomic-scale datasets")
    p.add_argument("--output-dir", default="./data/atomic_sim", help="Directory for CSV outputs")
    p.add_argument("--materials", default=",".join(DEFAULT_MATERIALS), help="Comma-separated materials (keys of MATERIAL_DATABASE)")
    p.add_argument("--t-min", type=int, default=300, help="Minimum temperature [K]")
    p.add_argument("--t-max", type=int, default=1000, help="Maximum temperature [K]")
    p.add_argument("--t-n", type=int, default=5, help="Number of temperature points")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for reproducibility")

    # MD detail controls
    p.add_argument("--n-sliding-points", type=int, default=60, help="Points per GB sliding curve")
    p.add_argument("--n-mobility-stress", type=int, default=12, help="Stress samples per material/temperature/dislocation-type")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    materials = [m.strip() for m in args.materials.split(",") if m.strip()]
    temps = temperature_grid(args.t_min, args.t_max, args.t_n)

    # DFT-like outputs
    generate_defect_formation_energies(materials, temps, os.path.join(output_dir, "defect_formation_energies.csv"))
    generate_activation_barriers(materials, temps, os.path.join(output_dir, "activation_barriers.csv"))
    generate_surface_energies(materials, temps, os.path.join(output_dir, "surface_energies.csv"))
    generate_grain_boundary_energies(materials, temps, os.path.join(output_dir, "grain_boundary_energies.csv"))

    # MD-like outputs
    generate_gb_sliding_curves(
        materials,
        temps,
        os.path.join(output_dir, "gb_sliding_curves.csv"),
        n_points_per_curve=args.n_sliding_points,
    )
    generate_dislocation_mobility(
        materials,
        temps,
        os.path.join(output_dir, "dislocation_mobility.csv"),
        n_stress_points=args.n_mobility_stress,
    )

    print(f"Synthetic datasets written to: {output_dir}")


if __name__ == "__main__":
    main()
