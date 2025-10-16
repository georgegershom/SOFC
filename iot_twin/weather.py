from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np
import requests

from .utils import TimeConfig, smooth_series, clamp


@dataclass
class WeatherData:
    frame: pd.DataFrame  # indexed by time, columns: outdoor_temp_c, outdoor_rh_pct, wind_speed_ms, wind_dir_deg, rain_mm, irradiance_wm2

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index


OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/era5"


def _download_open_meteo(time_cfg: TimeConfig, latitude: float, longitude: float) -> Optional[pd.DataFrame]:
    start_str = time_cfg.start.strftime("%Y-%m-%d")
    # end must be inclusive date range for API; subtract one minute to ensure coverage
    end_adj = (time_cfg.end - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_str,
        "end_date": end_adj,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "wind_direction_10m",
            "rain",
            "shortwave_radiation",
        ],
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = pd.to_datetime(hourly.get("time", []))
        if len(times) == 0:
            return None
        df = pd.DataFrame({
            "outdoor_temp_c": hourly.get("temperature_2m", []),
            "outdoor_rh_pct": hourly.get("relative_humidity_2m", []),
            "wind_speed_ms": hourly.get("wind_speed_10m", []),
            "wind_dir_deg": hourly.get("wind_direction_10m", []),
            "rain_mm": hourly.get("rain", []),
            "irradiance_wm2": hourly.get("shortwave_radiation", []),
        }, index=times)
        df.index = df.index.tz_localize("UTC")
        return df
    except Exception:
        return None


def _synthetic_weather(time_cfg: TimeConfig, rng: np.random.Generator) -> pd.DataFrame:
    idx_hourly = pd.date_range(time_cfg.start, time_cfg.end, freq="H", inclusive="left", tz="UTC")
    n = len(idx_hourly)

    # Seasonal temperature cycle (NYC-like)
    day_of_year = idx_hourly.dayofyear.values
    temp_seasonal = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 200) / 365.0)  # avg 10C, amp 15C
    temp_daily = 5 * np.sin(2 * np.pi * (idx_hourly.hour.values - 15) / 24.0)  # daily swing
    temp_noise = rng.normal(0, 1.5, size=n)
    outdoor_temp_c = temp_seasonal + temp_daily + temp_noise

    # Humidity inversely related to temperature with noise
    outdoor_rh_pct = clamp(pd.Series(70 - 0.8 * (outdoor_temp_c - np.mean(outdoor_temp_c)) + rng.normal(0, 5, size=n), index=idx_hourly), 15, 98)

    # Wind speed and direction
    wind_speed_ms = np.abs(rng.normal(4.0, 2.0, size=n))
    wind_dir_deg = (rng.uniform(0, 360, size=n) + smooth_series(pd.Series(wind_speed_ms, index=idx_hourly), 24)) % 360

    # Rain events
    rain_events = rng.random(n) < 0.03  # 3% hourly chance
    rain_mm = np.where(rain_events, np.abs(rng.normal(1.5, 2.0, size=n)), 0.0)

    # Solar irradiance: diurnal + seasonal
    hour = idx_hourly.hour.values
    diurnal = np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0))
    seasonal = np.maximum(0.0, np.sin(2 * np.pi * (day_of_year - 80) / 365.0))
    clouds = 0.6 + 0.4 * rng.random(n)
    irradiance_wm2 = 800 * diurnal * seasonal * clouds

    df = pd.DataFrame({
        "outdoor_temp_c": outdoor_temp_c,
        "outdoor_rh_pct": outdoor_rh_pct.values,
        "wind_speed_ms": wind_speed_ms,
        "wind_dir_deg": wind_dir_deg,
        "rain_mm": rain_mm,
        "irradiance_wm2": irradiance_wm2,
    }, index=idx_hourly)
    return df


def get_weather(time_cfg: TimeConfig, latitude: float, longitude: float, allow_download: bool, rng: np.random.Generator) -> WeatherData:
    df_hourly = None
    if allow_download:
        df_hourly = _download_open_meteo(time_cfg, latitude, longitude)
    if df_hourly is None:
        df_hourly = _synthetic_weather(time_cfg, rng)

    # Resample to target frequency with interpolation
    df = df_hourly.resample(time_cfg.freq).interpolate(method="time")

    # Ensure index covers exactly desired time range
    df = df.reindex(time_cfg.index).interpolate(limit_direction="both")

    return WeatherData(frame=df)
