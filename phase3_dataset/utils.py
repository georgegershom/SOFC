import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_png(array: np.ndarray, path: str) -> None:
    arr = array
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_json(obj: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random

        random.seed(seed)
    except Exception:
        pass


def gaussian(x: np.ndarray, mu: float, sigma: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def logistic_step(x: np.ndarray, center: float, width: float, magnitude: float) -> np.ndarray:
    return magnitude / (1.0 + np.exp(-(x - center) / width))


def normalize01(x: np.ndarray) -> np.ndarray:
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-12:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)
