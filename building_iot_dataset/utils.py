import numpy as np
import pandas as pd
from typing import Tuple


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def clamp(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return np.clip(values, min_value, max_value)


def to_series(index: pd.DatetimeIndex, values) -> pd.Series:
    return pd.Series(values, index=index)


def to_frame(index: pd.DatetimeIndex, data: dict) -> pd.DataFrame:
    return pd.DataFrame(data, index=index)


def smooth(series: pd.Series, window: int = 4) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def seasonal_profile(index: pd.DatetimeIndex, base: float, amplitude: float, phase_shift_days: int = 0) -> pd.Series:
    # 1-year sinusoid, peak in mid-summer by default
    day_of_year = index.dayofyear.values
    radians = 2 * np.pi * (day_of_year - phase_shift_days) / 365.25
    return pd.Series(base + amplitude * np.sin(radians), index=index)


def diurnal_profile(index: pd.DatetimeIndex, base: float, amplitude: float, phase_shift_hours: float = 0.0) -> pd.Series:
    # 24-hour sinusoid
    seconds = index.hour * 3600 + index.minute * 60
    radians = 2 * np.pi * ((seconds / 86400.0) - (phase_shift_hours / 24.0))
    return pd.Series(base + amplitude * np.sin(radians), index=index)


def piecewise_linear(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


def ensure_tz(index: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
    if index.tz is None:
        return index.tz_localize(tz)
    return index.tz_convert(tz)
