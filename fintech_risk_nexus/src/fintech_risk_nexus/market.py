from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import COUNTRY_BRANDS
from .utils import make_rng_for_key


def compute_hhi_from_shares(shares: np.ndarray) -> float:
    shares = np.asarray(shares, dtype=float)
    if shares.sum() <= 0:
        return 0.0
    norm = shares / shares.sum()
    return float(np.sum((norm * 100) ** 2))


def fabricate_market_structure(
    countries: List[str],
    years: List[int],
    seed: int = 42,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for country in countries:
        brands = COUNTRY_BRANDS.get(country, ["GenericPay", "AltPay", "NeoWallet"]) or [
            "GenericPay",
            "AltPay",
            "NeoWallet",
        ]
        num_brands = max(4, len(brands))
        rng = make_rng_for_key(f"hhi-{country}", seed)

        # Start with a Dirichlet-distributed market share vector
        conc_base = rng.uniform(0.8, 2.2)
        alpha = np.full(num_brands, conc_base)
        current = rng.dirichlet(alpha)

        for year in years:
            # Random walk with mild mean reversion
            noise = rng.normal(0, 0.03, size=current.size)
            proposed = current + noise
            proposed = np.clip(proposed, 0.001, None)
            proposed /= proposed.sum()

            # Entry/exit dynamics
            if rng.random() < 0.25:
                # New entrant with small share
                entrant_share = rng.uniform(0.01, 0.04)
                proposed = np.append(proposed * (1 - entrant_share), entrant_share)
            if proposed.size > 6 and rng.random() < 0.2:
                # Weakest exits
                weakest_idx = np.argmin(proposed)
                lost = proposed[weakest_idx]
                proposed = np.delete(proposed, weakest_idx)
                proposed *= (1 + lost)
                proposed /= proposed.sum()

            current = proposed / proposed.sum()
            hhi = compute_hhi_from_shares(current)

            frames.append(
                pd.DataFrame(
                    {
                        "year": [year] * current.size,
                        "country": [country] * current.size,
                        "firm_index": list(range(1, current.size + 1)),
                        "market_share": current,
                        "hhi": [hhi] * current.size,
                    }
                )
            )

    return pd.concat(frames, ignore_index=True)


def fabricate_license_counts(
    countries: List[str],
    years: List[int],
    seed: int = 42,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for country in countries:
        rng = make_rng_for_key(f"licenses-{country}", seed)
        baseline = rng.integers(2, 20)
        for year in years:
            # Economic cycles
            cycle = rng.normal(0, 3)
            growth = 0.3 * (year - years[0])
            count = int(max(0, baseline + cycle + growth))
            frames.append(
                pd.DataFrame(
                    {
                        "year": [year],
                        "country": [country],
                        "licenses_new": [count],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)
