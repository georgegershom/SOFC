from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import math
import random

@dataclass
class StatValue:
    mean: float
    std: float


def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(vmax, value))


def normal_with_bounds(mean: float, std: float, vmin: float, vmax: float, rng: random.Random) -> float:
    # simple truncated normal via rejection sampling
    for _ in range(64):
        v = rng.gauss(mean, std)
        if vmin <= v <= vmax:
            return v
    return clamp(v, vmin, vmax)


def compute_stochastic_bounds(value: float, rel_cv: float, abs_min: float, abs_max: float) -> StatValue:
    std = abs(value) * rel_cv
    return StatValue(mean=clamp(value, abs_min, abs_max), std=max(1e-12, min(std, abs_max * 0.5)))


def correlated_perturbation(mean: float, std: float, corr: float, base_z: float) -> float:
    # Return a value such that two calls with same base_z produce correlated samples
    # value = mean + std * (corr * base_z + sqrt(1-corr^2) * eps)
    eps = random.gauss(0.0, 1.0)
    mixed = corr * base_z + math.sqrt(max(0.0, 1.0 - corr * corr)) * eps
    return mean + std * mixed
