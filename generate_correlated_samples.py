#!/usr/bin/env python3
"""Synthetic correlation generator for interfacial fracture energies."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_correlated_fracture_energies(
    n_samples: int,
    mu1: float,
    sigma1: float,
    mu2: float,
    sigma2: float,
    rho: float,
    seed: int = 20260211,
) -> pd.DataFrame:
    """
    Generate correlated Gc values for YSZ|GDC (1) and GDC|LSCF (2).
    """
    rng = np.random.default_rng(seed)
    mean = [mu1, mu2]
    cov = [[sigma1**2, rho * sigma1 * sigma2], [rho * sigma1 * sigma2, sigma2**2]]

    out = np.empty((n_samples, 2), dtype=float)
    filled = 0
    while filled < n_samples:
        draw = rng.multivariate_normal(mean, cov, size=(n_samples - filled))
        draw = draw[(draw[:, 0] > 0.02) & (draw[:, 1] > 0.02)]
        n_take = min(len(draw), n_samples - filled)
        if n_take > 0:
            out[filled : filled + n_take, :] = draw[:n_take]
            filled += n_take

    return pd.DataFrame(out, columns=["Gc_YSZ_GDC", "Gc_GDC_LSCF"])


if __name__ == "__main__":
    df_correlated = generate_correlated_fracture_energies(
        n_samples=500,
        mu1=2.15,
        sigma1=0.4,
        mu2=1.00,
        sigma2=0.2,
        rho=0.5,
    )
    print("Generated synthetic dataset to test impact of missing correlations.")
    print(df_correlated.head())
