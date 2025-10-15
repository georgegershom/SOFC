from __future__ import annotations

from typing import Dict, List, Tuple
from datetime import datetime, timezone
import json
import h5py
import numpy as np


def write_dataset(
    path: str,
    samples: List[Dict],
    grids: List[Dict],
    stresses: List[np.ndarray],
    z_coords: List[np.ndarray],
    version: str,
    compression: str = "gzip",
    compression_level: int = 4,
) -> None:
    assert len(samples) == len(grids) == len(stresses) == len(z_coords)
    with h5py.File(path, "w") as f:
        f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()
        f.attrs["version"] = version
        f.attrs["description"] = (
            "Synthetic SOFC Dataset 1: paired warp surfaces and residual stress fields"
        )
        for idx in range(len(samples)):
            g = f.create_group(f"samples/{idx:05d}")
            meta = samples[idx]
            grid = grids[idx]
            stress = stresses[idx]
            z = z_coords[idx]

            # Metadata
            g.attrs["scenario_json"] = json.dumps(meta)

            # Grid and surfaces
            gg = g.create_group("grid")
            gg.create_dataset("x", data=grid["x"], compression=compression, compression_opts=compression_level)
            gg.create_dataset("y", data=grid["y"], compression=compression, compression_opts=compression_level)
            gg.create_dataset("z", data=z, compression=compression, compression_opts=compression_level)

            sg = g.create_group("surfaces")
            sg.create_dataset(
                "top",
                data=grid["z_top"],
                compression=compression,
                compression_opts=compression_level,
            )
            sg.create_dataset(
                "bottom",
                data=grid["z_bot"],
                compression=compression,
                compression_opts=compression_level,
            )
            sg.create_dataset(
                "w",
                data=grid["w"],
                compression=compression,
                compression_opts=compression_level,
            )

            # Stress field
            g.create_dataset(
                "stress/voxel",
                data=stress,
                compression=compression,
                compression_opts=compression_level,
            )
