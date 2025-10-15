from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Any
import h5py
import numpy as np


def write_sample_hdf5(path: str, metadata: Dict[str, Any], heightmap: np.ndarray, stress_tensors: Dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as f:
        g_meta = f.create_group("metadata")
        for k, v in metadata.items():
            if isinstance(v, (int, float, str, np.floating, np.integer)):
                g_meta.attrs[k] = v
            else:
                try:
                    g_meta.create_dataset(k, data=np.array(v))
                except Exception:
                    g_meta.attrs[k] = str(v)
        f.create_dataset("heightmap", data=heightmap, compression="gzip")
        g_stress = f.create_group("stress")
        for name, arr in stress_tensors.items():
            g_stress.create_dataset(name, data=arr, compression="gzip")
