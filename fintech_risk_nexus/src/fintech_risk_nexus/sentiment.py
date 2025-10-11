from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from .utils import make_rng_for_key


def fabricate_social_sentiment(
    countries: List[str],
    months: Iterable[pd.Timestamp],
    brands_by_country: Dict[str, List[str]],
    seed: int = 42,
    exogenous_trend: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    months = list(months)
    frames: List[pd.DataFrame] = []

    trend_lookup: Dict[str, np.ndarray] = {}
    if exogenous_trend is not None and not exogenous_trend.empty:
        for c, sub in exogenous_trend.groupby("country"):
            arr = sub.sort_values("date")["google_trend_mobile_money_fraud"].to_numpy()
            if arr.size:
                norm = (arr - arr.min()) / (np.ptp(arr) + 1e-6)
                trend_lookup[c] = norm

    for country in countries:
        rng = make_rng_for_key(f"sentiment-{country}", seed)
        brands = brands_by_country.get(country, ["GenericPay"]) or ["GenericPay"]

        t = np.arange(len(months))
        # Baseline sentiment by country (slightly positive overall), scale [-1, 1]
        baseline = rng.uniform(0.05, 0.25)
        seasonal = 0.08 * np.sin(2 * np.pi * t / 12.0 + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0, 0.08, size=len(months))

        sentiment = baseline + seasonal + noise

        # If fraud trend is high, sentiment dips
        if country in trend_lookup:
            fraud = trend_lookup[country]
            sentiment -= 0.35 * (fraud - 0.4)

        sentiment = np.clip(sentiment, -0.8, 0.9)

        # Mention volume scales with population + fintech penetration proxy
        base_mentions = rng.integers(400, 4000)
        mentions = base_mentions + (rng.normal(0, 120, size=len(months))).astype(int)
        mentions = np.clip(mentions, 50, None)

        # Brand mix: simulate small dispersion of brand-level scores (not returned here)
        brand_dispersion = rng.uniform(0.03, 0.12)
        brand_sentiments = [
            np.clip(sentiment + rng.normal(0, brand_dispersion, size=len(months)), -1.0, 1.0)
            for _ in brands
        ]
        # Aggregate: mean sentiment across brands
        mean_sentiment = np.mean(brand_sentiments, axis=0)

        frames.append(
            pd.DataFrame(
                {
                    "date": months,
                    "country": country,
                    "social_sentiment_avg": mean_sentiment,
                    "social_mention_volume": mentions.astype(int),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)
