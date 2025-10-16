from __future__ import annotations
import datetime as dt
import pandas as pd
import numpy as np
import requests
from typing import Optional

BASE_ARCHIVE = "https://archive-api.open-meteo.com/v1/era5"


def _request_open_meteo(lat: float, lon: float, tz: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "wind_direction_10m",
                "shortwave_radiation",
                "precipitation",
                "cloud_cover",
            ]
        ),
        "timezone": tz,
        "timeformat": "iso8601",
    }
    try:
        r = requests.get(BASE_ARCHIVE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        hourly = data.get("hourly", {})
        if not hourly or "time" not in hourly:
            return None
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"], utc=False).dt.tz_localize(tz)
        df = df.set_index("time").sort_index()
        # rename to standardized columns
        rename = {
            "temperature_2m": "ambient_temp_c",
            "relative_humidity_2m": "ambient_rh_pct",
            "wind_speed_10m": "wind_speed_mps",
            "wind_direction_10m": "wind_dir_deg",
            "shortwave_radiation": "solar_irradiance_wm2",
            "precipitation": "rain_mm",
            "cloud_cover": "cloud_cover_pct",
        }
        df = df.rename(columns=rename)
        return df
    except Exception:
        return None


def _synthetic_weather(index: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day_of_year = index.dayofyear.values

    # Ambient temperature seasonal + diurnal
    t_season = 12 + 10 * np.sin(2 * np.pi * (day_of_year - 172) / 365.25)  # peak around day 172 ~ Jun 21
    seconds = index.hour * 3600 + index.minute * 60
    t_diurnal = 3 * np.sin(2 * np.pi * (seconds - 15 * 3600) / 86400.0)  # warmest ~15:00
    ambient_temp_c = t_season + t_diurnal + rng.normal(0, 1.0, size=len(index))

    # Relative humidity inverse-ish to temperature
    ambient_rh_pct = np.clip(65 - 0.8 * (ambient_temp_c - np.mean(ambient_temp_c)) + rng.normal(0, 5, len(index)), 20, 100)

    # Solar irradiance positive during day with seasonal amplitude
    solar_base = np.maximum(0, np.sin(2 * np.pi * (seconds - 6 * 3600) / 86400.0))
    solar_season = np.clip(0.6 + 0.4 * np.sin(2 * np.pi * (day_of_year - 172) / 365.25), 0.2, 1.0)
    solar_irradiance_wm2 = 800 * solar_base * solar_season * (1 - 0.2 * rng.random(len(index)))

    # Wind
    wind_speed_mps = np.clip(rng.normal(4.5, 2.0, len(index)), 0.0, None)
    wind_dir_deg = rng.uniform(0, 360, len(index))

    # Rain events
    rain_events = rng.random(len(index)) < 0.02
    rain_mm = np.where(rain_events, rng.gamma(2.0, 1.5, len(index)), 0.0)

    # Cloud cover correlated with solar
    cloud_cover_pct = np.clip(100 * (1 - solar_base) + rng.normal(0, 10, len(index)), 0, 100)

    df = pd.DataFrame(
        {
            "ambient_temp_c": ambient_temp_c,
            "ambient_rh_pct": ambient_rh_pct,
            "wind_speed_mps": wind_speed_mps,
            "wind_dir_deg": wind_dir_deg,
            "solar_irradiance_wm2": solar_irradiance_wm2,
            "rain_mm": rain_mm,
            "cloud_cover_pct": cloud_cover_pct,
        },
        index=index,
    )
    return df


def get_weather(index: pd.DatetimeIndex, lat: float, lon: float, tz: str, seed: int = 42) -> pd.DataFrame:
    start_date = index[0].date().isoformat()
    end_date = index[-1].date().isoformat()

    # Try Open-Meteo hourly and then upsample
    df = _request_open_meteo(lat, lon, tz, start_date, end_date)
    if df is None:
        return _synthetic_weather(index, seed=seed)

    # Reindex to target index with interpolation
    df = df.resample("15min").interpolate(limit_direction="both")
    df = df.reindex(index).interpolate(limit_direction="both")
    return df
