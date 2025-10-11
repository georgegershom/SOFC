from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .utils import make_rng_for_key


def fabricate_cyber_incidents(
    countries: List[str],
    months: Iterable[pd.Timestamp],
    seed: int = 42,
    exogenous_trend: Optional[pd.DataFrame] = None,
    exogenous_sentiment: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    months = list(months)

    trend_lookup: Dict[str, np.ndarray] = {}
    if exogenous_trend is not None and not exogenous_trend.empty:
        for c, sub in exogenous_trend.groupby("country"):
            arr = sub.sort_values("date")["google_trend_mobile_money_fraud"].to_numpy()
            if arr.size:
                norm = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
                trend_lookup[c] = norm

    sentiment_lookup: Dict[str, np.ndarray] = {}
    if exogenous_sentiment is not None and not exogenous_sentiment.empty:
        for c, sub in exogenous_sentiment.groupby("country"):
            arr = sub.sort_values("date")["social_sentiment_avg"].to_numpy()
            if arr.size:
                sentiment_lookup[c] = arr

    frames: List[pd.DataFrame] = []

    for country in countries:
        rng = make_rng_for_key(f"cyber-{country}", seed)
        t = np.arange(len(months))

        # Country baseline for financial-sector incidents per month
        base = rng.uniform(3, 30)  # baseline intensity

        # Construct a latent rate with seasonality and slow drift
        seasonal = 0.25 * np.sin(2 * np.pi * t / 6.0 + rng.uniform(0, 2 * np.pi))
        slow_drift = np.linspace(0, rng.uniform(-0.1, 0.25), num=len(months))

        latent = base * (1 + seasonal + slow_drift)

        # Trend impact: higher fraud trend -> higher incident rate
        if country in trend_lookup:
            latent *= 1.0 + 0.6 * (trend_lookup[country] - 0.3)

        # Sentiment impact: worse sentiment -> more incidents reported
        if country in sentiment_lookup:
            latent *= 1.0 + 0.5 * np.clip(-sentiment_lookup[country], 0, 1)

        # Stochasticity with overdispersion (Negative Binomial via Gamma-Poisson)
        dispersion = rng.uniform(0.4, 1.2)
        gamma_shape = 1.0 / max(dispersion, 1e-3)
        gamma_scale = latent / max(gamma_shape, 1e-3)
        shocks = rng.gamma(shape=gamma_shape, scale=gamma_scale)
        incidents = rng.poisson(lam=np.clip(shocks, 0.5, None))

        frames.append(
            pd.DataFrame(
                {
                    "date": months,
                    "country": country,
                    "cyber_incidents_finsec": incidents.astype(int),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)
