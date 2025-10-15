from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_sample_npz(out_dir: Path, sample_id: int, arrays: Dict[str, np.ndarray], metadata: Dict[str, Any]) -> Path:
    ensure_dir(out_dir)
    fname = out_dir / f"sample_{sample_id:06d}.npz"
    # Convert metadata to arrays
    meta_keys = np.array(list(metadata.keys()), dtype=object)
    meta_vals = np.array([str(v) for v in metadata.values()], dtype=object)
    np.savez_compressed(fname, **arrays, meta_keys=meta_keys, meta_vals=meta_vals)
    return fname
