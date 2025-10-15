from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import click
import numpy as np
from tqdm import tqdm

from .materials import make_default_materials, MaterialModel
from .laminate import Layer, LaminateSpec, solve_laminate, warp_surface_from_kappa
from .fields import voxelize_layer_stresses, warp_surfaces_to_heightmap
from .doe import DOESpace, sample_doe
from .writer import write_sample_hdf5


@click.group()
def cli() -> None:
    pass


def layer_from_material(mat: MaterialModel, t: float, T_peak: float, T_ambient: float, eigen_relax: float) -> Layer:
    # Effective properties between peak and ambient
    E = 0.5 * (mat.modulus_vs_T(T_peak) + mat.modulus_vs_T(T_ambient))
    alpha = 0.5 * (mat.cte_vs_T(T_peak) + mat.cte_vs_T(T_ambient))
    # Eigenstrain fraction of free shrinkage retained
    eigen = mat.sintering_linear_shrinkage * (1.0 - eigen_relax)
    return Layer(name=mat.name, thickness=t, E=E, nu=mat.poisson_ratio, alpha=alpha, eigenstrain=eigen)


@cli.command()
@click.option("--out-dir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True)
@click.option("--n", type=int, default=10, help="Number of DOE samples")
@click.option("--seed", type=int, default=42)
@click.option("--grid", type=str, default="64,64,6", help="nx,ny,nz grid for fields")
@click.option("--surf", type=str, default="128,128", help="nx,ny for heightmap")
def generate(out_dir: Path, n: int, seed: int, grid: str, surf: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = map(int, grid.split(","))
    sx, sy = map(int, surf.split(","))

    mats = make_default_materials()
    space = DOESpace(
        Lx_range=(0.03, 0.12),
        Ly_range=(0.03, 0.12),
        t_ranges={"anode": (50e-6, 400e-6), "electrolyte": (5e-6, 50e-6), "cathode": (20e-6, 200e-6)},
        T_peak_range=(1373.15, 1673.15),
        eigen_relax_range=(0.2, 0.9),
    )

    rng = np.random.default_rng(seed)
    samples = sample_doe(space, n, rng)

    for idx, s in enumerate(tqdm(samples, desc="Generating")):
        layers = [
            layer_from_material(mats["anode"], s.t_anode, s.T_peak, s.T_ambient, s.eigen_relax_anode),
            layer_from_material(mats["electrolyte"], s.t_electrolyte, s.T_peak, s.T_ambient, s.eigen_relax_electrolyte),
            layer_from_material(mats["cathode"], s.t_cathode, s.T_peak, s.T_ambient, s.eigen_relax_cathode),
        ]
        deltaT = s.T_ambient - s.T_peak
        result = solve_laminate(layers, deltaT)

        X, Y, W = warp_surface_from_kappa(result.kappa, (s.plate_Lx, s.plate_Ly), (sx, sy))
        heightmap = warp_surfaces_to_heightmap(X, Y, W)

        stress_field = voxelize_layer_stresses({k: v.astype(np.float32) for k, v in result.layer_stress_xy.items()},
                                               ("anode", "electrolyte", "cathode"), (nx, ny, nz))

        meta = {
            "idx": idx,
            "plate_Lx": s.plate_Lx,
            "plate_Ly": s.plate_Ly,
            "thickness_anode": s.t_anode,
            "thickness_electrolyte": s.t_electrolyte,
            "thickness_cathode": s.t_cathode,
            "T_peak": s.T_peak,
            "T_ambient": s.T_ambient,
        }

        out_path = out_dir / f"sample_{idx:05d}.h5"
        write_sample_hdf5(str(out_path), meta, heightmap, stress_field.tensors)


if __name__ == "__main__":
    cli()
