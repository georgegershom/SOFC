from __future__ import annotations
from typing import Callable, Dict, Iterable, Tuple
from pathlib import Path
import json
import h5py
import numpy as np


FieldFunc = Callable[[np.ndarray, np.ndarray, np.ndarray, Dict[str, float], int | None], Dict[str, object]]


def write_dataset1_hdf5(
    out_path: Path,
    grid: Tuple[np.ndarray, np.ndarray, np.ndarray],
    samples: Iterable[Dict[str, float]],
    field_fn: FieldFunc,
    param_ranges: Dict[str, object],
    compression: str = "gzip",
) -> None:
    x, y, z = grid
    with h5py.File(out_path, "w") as f:
        f.attrs["description"] = "SOFC high-fidelity synthetic multiphysics dataset"
        f.attrs["coordinate_system"] = "fractional [0,1] along x,y,z"
        f.attrs["units_json"] = json.dumps({
            "temperature_K": "K",
            "current_density_A_per_cm2": "A/cm^2",
            "species_molfrac": "-",
            "displacement_m": "m",
            "strain": "-",
            "stress_Pa": "Pa",
            "von_mises_Pa": "Pa",
            "creep_strain": "-",
            "cell_voltage_V": "V",
            "fracture_metrics": "SI units",
        })
        f.create_dataset("grid/x", data=x, compression=compression)
        f.create_dataset("grid/y", data=y, compression=compression)
        f.create_dataset("grid/z", data=z, compression=compression)
        # NumPy 2.0+: use np.bytes_ instead of deprecated/removed np.string_
        f.create_dataset(
            "param_ranges_json",
            data=np.bytes_(json.dumps({k: v.__dict__ if hasattr(v, "__dict__") else v for k, v in param_ranges.items()})),
        )

        samples_grp = f.create_group("samples")

        for idx, sample in enumerate(samples):
            g = samples_grp.create_group(f"sample_{idx:05d}")
            # store inputs
            g.create_dataset("inputs_json", data=np.bytes_(json.dumps(sample)))
            # compute fields
            fields = field_fn(x, y, z, sample, seed=idx)
            # store outputs
            for key, value in fields.items():
                if isinstance(value, dict):
                    g.create_dataset(f"outputs/{key}_json", data=np.bytes_(json.dumps(value)))
                elif isinstance(value, np.ndarray):
                    g.create_dataset(f"outputs/{key}", data=value, compression=compression)
                else:
                    g.attrs[key] = value

        # Multi-resolution pyramids (simple downsampling by stride for a few fields)
        # Not all fields need pyramids; provide for temperature and von_mises
        pyr_group = f.create_group("pyramids")
        base_temp = f["samples/sample_00000/outputs/temperature_K"][()]
        base_vm = f["samples/sample_00000/outputs/von_mises_Pa"][()]
        for level, stride in enumerate([2, 4]):
            pyr_group.create_dataset(f"temperature/level_{level}", data=base_temp[::stride, ::stride, ::stride], compression=compression)
            pyr_group.create_dataset(f"von_mises/level_{level}", data=base_vm[::stride, ::stride, ::stride], compression=compression)
