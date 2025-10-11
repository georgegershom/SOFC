from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import GEO_MAP
from .utils import make_rng_for_key


@dataclass
class TrendsConfig:
    keyword: str = "mobile money fraud"
    fetch_timeout_s: int = 30


class TrendsFetcher:
    def __init__(self, config: Optional[TrendsConfig] = None):
        self.config = config or TrendsConfig()
        self._pytrends = None

    def _ensure_client(self):
        if self._pytrends is not None:
            return self._pytrends
        try:
            from pytrends.request import TrendReq  # type: ignore

            self._pytrends = TrendReq(hl="en-US", tz=0)
        except Exception:
            self._pytrends = None
        return self._pytrends

    def fetch_or_fabricate(
        self,
        countries: List[str],
        start: date,
        end: date,
        seed: int = 42,
    ) -> pd.DataFrame:
        months = pd.period_range(
            pd.Timestamp(start.year, start.month, 1),
            pd.Timestamp(end.year, end.month, 1),
            freq="M",
        ).to_timestamp()

        frames: List[pd.DataFrame] = []
        client = self._ensure_client()
        for country in countries:
            if client is not None:
                try:
                    frame = self._fetch_country(client, country, months)
                except Exception:
                    frame = self._fabricate_country(country, months, seed)
            else:
                frame = self._fabricate_country(country, months, seed)
            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    def _fetch_country(self, client, country: str, months: pd.DatetimeIndex) -> pd.DataFrame:
        geo = GEO_MAP.get(country, country)
        start = months.min().date()
        end = months.max().date()
        timeframe = f"{start:%Y-%m-%d} {end:%Y-%m-%d}"

        client.build_payload(
            kw_list=[self.config.keyword], geo=geo, timeframe=timeframe, gprop=""
        )
        df = client.interest_over_time()
        if df is None or df.empty:
            raise RuntimeError("Empty trends response")

        df = df.reset_index().rename(columns={"date": "date"})
        df = df[["date", self.config.keyword]]
        df = df.rename(columns={self.config.keyword: "google_trend_mobile_money_fraud"})
        df["country"] = country

        # Align to target months (pytrends may return weekly for narrow ranges)
        df = (
            df.assign(date=pd.to_datetime(df["date"]))
            .groupby([pd.Grouper(key="date", freq="M")])
            .mean(numeric_only=True)
            .reset_index()
        )

        # Reindex to ensure full month coverage
        full = pd.DataFrame({"date": months})
        df = full.merge(df, on="date", how="left")
        df["country"] = df["country"].ffill().bfill().fillna(country)
        df["google_trend_mobile_money_fraud"] = (
            df["google_trend_mobile_money_fraud"].round().clip(lower=0, upper=100).fillna(0)
        )
        return df

    def _fabricate_country(
        self, country: str, months: pd.DatetimeIndex, seed: int
    ) -> pd.DataFrame:
        rng = make_rng_for_key(f"trends-{country}", seed)
        t = np.arange(len(months))

        # Seasonality + slow trend + random spikes
        seasonal = 15 * np.sin(2 * np.pi * t / 12.0)
        slow_trend = np.clip(0.15 * t, 0, 45)
        noise = rng.normal(0, 6, size=len(months))

        base = 30 + seasonal + slow_trend + noise

        # Random shock months (fraud news cycles)
        num_shocks = int(rng.integers(2, 6))
        shock_idx = rng.choice(len(months), size=num_shocks, replace=False)
        shock_magnitudes = rng.uniform(15, 35, size=num_shocks)
        base[shock_idx] += shock_magnitudes

        series = np.clip(base, 0, 100)
        return pd.DataFrame(
            {
                "date": months,
                "country": country,
                "google_trend_mobile_money_fraud": series.round().astype(int),
            }
        )
