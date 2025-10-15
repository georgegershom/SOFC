from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RangeSpec:
    low: float
    high: float
    scale: str = "linear"  # 'linear' | 'log'

    def sample_uniform(self) -> float:
        if self.scale == "log":
            lo = math.log10(self.low)
            hi = math.log10(self.high)
            return 10 ** (random.random() * (hi - lo) + lo)
        return random.random() * (self.high - self.low) + self.low


def latin_hypercube(n_samples: int, ranges: List[RangeSpec], seed: int | None = None) -> List[List[float]]:
    """Basic maximin-free Latin Hypercube in unit space mapped to given ranges.

    - Each dimension is stratified into n_samples bins
    - A random permutation selects bin order per dimension
    - Within each assigned bin, a uniform random value is drawn
    - Values are mapped to physical ranges; log ranges are mapped in log10 space
    """
    if seed is not None:
        random.seed(seed)

    d = len(ranges)
    # Generate strata midpoints per dimension
    unit_samples = [[0.0 for _ in range(d)] for _ in range(n_samples)]

    for j in range(d):
        perm = list(range(n_samples))
        random.shuffle(perm)
        for i in range(n_samples):
            a = perm[i]
            # Uniform inside the stratum [a/n, (a+1)/n)
            u = (a + random.random()) / n_samples
            unit_samples[i][j] = u

    # Map to physical ranges
    mapped: List[List[float]] = []
    for i in range(n_samples):
        row: List[float] = []
        for j, r in enumerate(ranges):
            if r.scale == "log":
                lo = math.log10(r.low)
                hi = math.log10(r.high)
                v = 10 ** (lo + unit_samples[i][j] * (hi - lo))
            else:
                v = r.low + unit_samples[i][j] * (r.high - r.low)
            row.append(v)
        mapped.append(row)

    return mapped


def full_factorial(levels_per_dim: List[List[float]]) -> List[List[float]]:
    """Cartesian product of given levels per dimension."""
    # Iterative product to avoid recursion limits
    out: List[List[float]] = [[]]
    for lvls in levels_per_dim:
        new_out: List[List[float]] = []
        for prefix in out:
            for v in lvls:
                new_out.append(prefix + [v])
        out = new_out
    return out


def numeric_levels(spec: RangeSpec, n_levels: int) -> List[float]:
    if n_levels <= 1:
        return [(spec.low + spec.high) / 2.0]
    if spec.scale == "log":
        lo = math.log10(spec.low)
        hi = math.log10(spec.high)
        return [10 ** (lo + k * (hi - lo) / (n_levels - 1)) for k in range(n_levels)]
    return [spec.low + k * (spec.high - spec.low) / (n_levels - 1) for k in range(n_levels)]
