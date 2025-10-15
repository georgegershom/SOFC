from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np

from .doe import sample_params
from .physics import LayerProps, PlateDims, solve_laminate, plate_warp_height_map, voxelize_stress_field
from .io import write_dataset
from . import __version__


def _build_layers(s: Dict) -> List[LayerProps]:
    return [
        LayerProps(
            name="anode",
            thickness=s["t_anode"],
            youngs_modulus=s["E_anode"],
            poissons_ratio=s["nu_anode"],
            cte=s["alpha_anode"],
            sinter_eigenstrain=s["es_anode"],
            relaxation=s["relax_anode"],
        ),
        LayerProps(
            name="electrolyte",
            thickness=s["t_electrolyte"],
            youngs_modulus=s["E_electrolyte"],
            poissons_ratio=s["nu_electrolyte"],
            cte=s["alpha_electrolyte"],
            sinter_eigenstrain=s["es_electrolyte"],
            relaxation=s["relax_electrolyte"],
        ),
        LayerProps(
            name="cathode",
            thickness=s["t_cathode"],
            youngs_modulus=s["E_cathode"],
            poissons_ratio=s["nu_cathode"],
            cte=s["alpha_cathode"],
            sinter_eigenstrain=s["es_cathode"],
            relaxation=s["relax_cathode"],
        ),
    ]


def generate_dataset(
    output_path: Path,
    num_samples: int,
    seed: int,
    grid_shape: Tuple[int, int],
    num_z: int,
    zip_after: bool = False,
) -> Path:
    rng = np.random.default_rng(seed)
    scenarios = sample_params(num_samples, seed)

    grids: List[Dict] = []
    stresses: List[np.ndarray] = []
    z_coords: List[np.ndarray] = []

    for s in scenarios:
        layers = _build_layers(s)
        dims = PlateDims(length_x=s["Lx"], length_y=s["Ly"]) 
        h_total = sum(l.thickness for l in layers)

        # Solve laminate for thermal + eigenstrains
        res = solve_laminate(layers, delta_T=s["delta_T"]) 

        # Height maps
        X, Y, w, z_top, z_bot = plate_warp_height_map(
            res.curvature, grid_shape=grid_shape, dims=dims, thickness_total=h_total
        )
        z, stress = voxelize_stress_field(
            layers, res, grid_shape=grid_shape, num_z=num_z, delta_T=s["delta_T"]
        )

        grids.append({
            "x": X.astype(np.float32)[0, :],  # store 1D axes
            "y": Y.astype(np.float32)[:, 0],
            "w": w.astype(np.float32),
            "z_top": z_top.astype(np.float32),
            "z_bot": z_bot.astype(np.float32),
        })
        stresses.append(stress)
        z_coords.append(z)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset(str(path), scenarios, grids, stresses, z_coords, version=__version__)

    if zip_after:
        import zipfile
        zip_path = path.with_suffix(path.suffix + ".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as zf:
            zf.write(path, arcname=path.name)
        return zip_path

    return path
