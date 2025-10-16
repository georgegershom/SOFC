from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class TimeConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str = "15min"
    tz: str = "UTC"

    def make_index(self) -> pd.DatetimeIndex:
        start = pd.Timestamp(self.start).tz_convert(self.tz) if pd.Timestamp(self.start).tzinfo else pd.Timestamp(self.start).tz_localize(self.tz)
        end = pd.Timestamp(self.end).tz_convert(self.tz) if pd.Timestamp(self.end).tzinfo else pd.Timestamp(self.end).tz_localize(self.tz)
        # Inclusive of end; pandas >= 2 supports inclusive
        try:
            idx = pd.date_range(start=start, end=end, freq=self.freq, inclusive="both")
        except TypeError:
            # Fallback for older pandas
            idx = pd.date_range(start=start, end=end, freq=self.freq, closed=None)
        return idx

    @property
    def step_hours(self) -> float:
        delta = pd.Timedelta(self.freq)
        return delta / pd.Timedelta(hours=1)


# Randomness utilities

def get_rng(seed: Optional[int]) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


def ar1_process(n: int, phi: float, sigma: float, rng: np.random.Generator, mu: float = 0.0) -> np.ndarray:
    x = np.empty(n)
    x[0] = mu + sigma * rng.standard_normal()
    for i in range(1, n):
        x[i] = mu + phi * (x[i - 1] - mu) + sigma * rng.standard_normal()
    return x


def smoothstep(x: np.ndarray, edge0: float = 0.0, edge1: float = 1.0) -> np.ndarray:
    # Scale, clamp, and apply smoothstep 3t^2 - 2t^3
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def workday_profile(index: pd.DatetimeIndex, start_hour: float = 8.0, peak_hour: float = 11.0, end_hour: float = 18.0) -> np.ndarray:
    # 0 on weekends; ramp up in morning, plateau midday, ramp down evening
    hour = index.hour + index.minute / 60.0
    weekday = index.weekday
    is_workday = (weekday < 5).astype(float)
    rise = smoothstep(hour, start_hour - 1.0, peak_hour)
    fall = 1.0 - smoothstep(hour, end_hour - 1.0, end_hour + 1.0)
    plate = np.minimum(1.0, rise) * np.minimum(1.0, fall)
    return is_workday * plate


def seasonality_day_of_year(index: pd.DatetimeIndex, amplitude: float = 1.0, phase_shift_days: float = 0.0) -> np.ndarray:
    doy = index.dayofyear.values.astype(float)
    return amplitude * np.sin(2.0 * np.pi * (doy - phase_shift_days) / 365.0)


def clamp_series(s: pd.Series, low: Optional[float] = None, high: Optional[float] = None) -> pd.Series:
    if low is not None:
        s = s.clip(lower=low)
    if high is not None:
        s = s.clip(upper=high)
    return s


def resample_like(src: pd.Series | pd.DataFrame, target_index: pd.DatetimeIndex, method: str = "linear") -> pd.Series | pd.DataFrame:
    # Reindex with interpolation to match target resolution
    out = src.reindex(target_index.union(src.index)).interpolate(method=method).reindex(target_index)
    if isinstance(src, pd.Series):
        out.name = src.name
    return out


def ensure_dir(path: str) -> None:
    import os
    os.makedirs(path, exist_ok=True)


def with_columns_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}{c}" for c in df.columns]
    return df
