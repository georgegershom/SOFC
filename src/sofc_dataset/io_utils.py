from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any
import numpy as np
import os


@dataclass
class Manifest:
    dataset_name: str
    version: str
    description: str
    num_items: int
    schema: Dict[str, Any]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_npz(path: str, arrays: Dict[str, Any]) -> None:
    # Flatten nested dicts using keys like 'stress.s_xx'
    flat: Dict[str, Any] = {}
    def _flatten(prefix: str, obj: Any):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(f"{prefix}.{k}" if prefix else k, v)
        else:
            flat[prefix] = obj
    _flatten("", arrays)
    np.savez_compressed(path, **flat)


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def write_manifest(path: str, manifest: Manifest) -> None:
    save_json(path, asdict(manifest))
