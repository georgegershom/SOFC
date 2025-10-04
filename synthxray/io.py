from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict

import numpy as np
from tifffile import imwrite


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tiff_stack(volume: np.ndarray, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    imwrite(filepath, volume, imagej=False)


def save_array_npy(arr: np.ndarray, filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    np.save(filepath, arr)


def save_json(data: Dict[str, Any], filepath: str) -> None:
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
