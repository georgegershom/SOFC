from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Union

import numpy as np

from .parameters import CategoricalParam, ContinuousParam, Param, get_parameter_space


def _latin_hypercube_unit(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """Generate an n_samples x n_dims Latin Hypercube in [0,1]."""
    # For each dimension, divide [0,1] into n_samples bins and permute
    H = np.zeros((n_samples, n_dims), dtype=float)
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    for j in range(n_dims):
        # Random point within each interval
        u = rng.random(n_samples)
        pts = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(pts)
        H[:, j] = pts
    return H


def _map_unit_to_range(u: float, param: ContinuousParam) -> float:
    if param.log_scale:
        # Map to log space
        lo, hi = math.log(param.low), math.log(param.high)
        val = math.exp(lo + u * (hi - lo))
    else:
        val = param.low + u * (param.high - param.low)
    return float(val)


def sample_parameter_sets(
    n_samples: int,
    seed: int | None = None,
) -> List[Dict[str, Union[float, str]]]:
    """Sample parameter sets using LHS for continuous and uniform for categorical."""
    space: Dict[str, Param] = get_parameter_space()
    cont_keys: List[str] = [k for k, v in space.items() if isinstance(v, ContinuousParam)]
    cat_keys: List[str] = [k for k, v in space.items() if isinstance(v, CategoricalParam)]

    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    H = _latin_hypercube_unit(n_samples, len(cont_keys), rng) if cont_keys else np.zeros((n_samples, 0))

    samples: List[Dict[str, Union[float, str]]] = []
    for i in range(n_samples):
        record: Dict[str, Union[float, str]] = {}
        # continuous
        for j, key in enumerate(cont_keys):
            param = space[key]  # type: ignore[assignment]
            assert isinstance(param, ContinuousParam)
            record[key] = _map_unit_to_range(H[i, j], param)
        # categorical
        for key in cat_keys:
            p = space[key]  # type: ignore[assignment]
            assert isinstance(p, CategoricalParam)
            record[key] = py_rng.choice(p.choices)
        samples.append(record)

    return samples
