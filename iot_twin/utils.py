from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple

SECONDS_PER_HOUR = 3600

@dataclass
class TimeConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.end, freq=self.freq, inclusive="left", tz="UTC")


def seeded_random_state(seed: int | None) -> np.random.Generator:
    if seed is None:
        seed = np.random.SeedSequence().entropy
    return np.random.default_rng(int(seed) % (2**32 - 1))


def smooth_series(series: pd.Series, window: int = 5) -> pd.Series:
    if window <= 1:
        return series
    return series.rolling(window=window, min_periods=1, center=True).mean()


def clamp(series: pd.Series, min_value: float | None = None, max_value: float | None = None) -> pd.Series:
    if min_value is not None:
        series = series.clip(lower=min_value)
    if max_value is not None:
        series = series.clip(upper=max_value)
    return series


def daily_profile(index: pd.DatetimeIndex, peak_hour: int, spread_hours: float = 3.0, amplitude: float = 1.0) -> pd.Series:
    hours = index.hour + index.minute / 60.0
    peak = np.exp(-0.5 * ((hours - peak_hour) / spread_hours) ** 2)
    return pd.Series(amplitude * peak, index=index)


def weekly_scaler(index: pd.DatetimeIndex, weekend_scale: float = 0.2) -> pd.Series:
    weekday = pd.Series(index.weekday, index=index)
    return pd.Series(np.where(weekday >= 5, weekend_scale, 1.0), index=index)


def add_measurement_noise(series: pd.Series, rng: np.random.Generator, sigma: float) -> pd.Series:
    return series + rng.normal(0.0, sigma, size=len(series))


def first_order_response(input_series: pd.Series, tau_hours: float, dt_minutes: float) -> pd.Series:
    alpha = float(dt_minutes) / (tau_hours * 60.0 + float(dt_minutes))
    out = np.empty(len(input_series))
    out[0] = input_series.iloc[0]
    for i in range(1, len(input_series)):
        out[i] = out[i - 1] + alpha * (input_series.iloc[i] - out[i - 1])
    return pd.Series(out, index=input_series.index)


def co2_mass_balance(co2_outdoor_ppm: pd.Series, occupancy: pd.Series, ventilation_ach: pd.Series, room_volume_m3: float, emission_rate_lps_per_person: float, dt_minutes: float) -> pd.Series:
    # Simple discrete-time mass-balance CO2 model
    # dC/dt = (G/V) - ACH*(C - C_out)
    dt_hours = dt_minutes / 60.0
    C = np.empty(len(co2_outdoor_ppm))
    C[0] = max(400.0, co2_outdoor_ppm.iloc[0])
    for i in range(1, len(co2_outdoor_ppm)):
        Cout = co2_outdoor_ppm.iloc[i]
        G_lps = occupancy.iloc[i] * emission_rate_lps_per_person
        ach = max(0.1, ventilation_ach.iloc[i])
        # Convert generation l/s to ppm/h: ppm = (G * 3600 / V) * 1e6 / (R*T) approx -> we simplify with factor k
        # Use simplified proportional factor k to keep units reasonable
        k = 12.0  # tuned factor for realism
        dC = dt_hours * (k * G_lps / max(room_volume_m3, 50.0) - ach * (C[i - 1] - Cout))
        C[i] = C[i - 1] + dC
    return pd.Series(C, index=co2_outdoor_ppm.index)


def dew_point_temperature(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    # Magnus formula
    a, b = 17.62, 243.12
    gamma = (a * temp_c / (b + temp_c)) + np.log(rh.clip(lower=1e-6) / 100.0)
    return (b * gamma) / (a - gamma)
