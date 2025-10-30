from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import numpy as np


@dataclass
class ParamSpec:
    name: str
    min: float
    max: float
    scale: str = "linear"  # "linear" or "log"
    dtype: str = "float"   # "float" or "int"


class LatinHypercubeSampler:
    """
    Simple Latin Hypercube Sampling (LHS) for continuous parameters.
    - Supports linear and logarithmic scaling
    - Supports integer rounding for discrete integer parameters
    """

    def __init__(self, param_specs: Dict[str, ParamSpec], seed: int | None = None) -> None:
        self.param_specs = param_specs
        self.rng = np.random.default_rng(seed)

    def _lhs_unit(self, num_samples: int, dim: int) -> np.ndarray:
        """Generate a Latin hypercube in [0,1]^dim with num_samples points."""
        # Divide [0,1] into num_samples strata per dimension and permute
        result = np.empty((num_samples, dim), dtype=np.float64)
        cut = np.linspace(0, 1, num_samples + 1)
        for j in range(dim):
            # Random point within each interval
            u = self.rng.uniform(low=cut[:-1], high=cut[1:])
            # Permute intervals across samples
            self.rng.shuffle(u)
            result[:, j] = u
        return result

    def _transform(self, u: np.ndarray, spec: ParamSpec) -> np.ndarray:
        if spec.scale == "linear":
            values = spec.min + u * (spec.max - spec.min)
        elif spec.scale == "log":
            # interpret min/max as positive bounds
            if spec.min <= 0 or spec.max <= 0:
                raise ValueError(f"Log scale requires positive bounds for {spec.name}")
            log_min = math.log(spec.min)
            log_max = math.log(spec.max)
            values = np.exp(log_min + u * (log_max - log_min))
        else:
            raise ValueError(f"Unknown scale: {spec.scale}")

        if spec.dtype == "int":
            return np.round(values).astype(np.int64)
        return values.astype(np.float64)

    def sample(self, num_samples: int) -> List[Dict[str, Any]]:
        names = list(self.param_specs.keys())
        dim = len(names)
        U = self._lhs_unit(num_samples, dim)
        samples: List[Dict[str, Any]] = []
        for i in range(num_samples):
            s: Dict[str, Any] = {}
            for j, name in enumerate(names):
                spec = self.param_specs[name]
                s[name] = self._transform(U[i, j], spec).item() if spec.dtype == "int" else float(self._transform(U[i, j], spec))
            samples.append(s)
        return samples


def build_param_specs(raw_specs: Dict[str, Dict[str, Any]]) -> Dict[str, ParamSpec]:
    specs: Dict[str, ParamSpec] = {}
    for name, rs in raw_specs.items():
        specs[name] = ParamSpec(
            name=name,
            min=float(rs["min"]),
            max=float(rs["max"]),
            scale=str(rs.get("scale", "linear")),
            dtype=str(rs.get("dtype", "float")),
        )
    return specs
