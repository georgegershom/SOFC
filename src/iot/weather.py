from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, seasonality_day_of_year


try:
    from meteostat import Point, Hourly
    _HAS_METEOSTAT = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_METEOSTAT = False


@dataclass
class WeatherConfig:
    latitude: float = 40.71
    longitude: float = -74.01
    use_download: bool = True


def _approx_solar_irradiance(index: pd.DatetimeIndex, lat_deg: float, cloud: np.ndarray) -> pd.Series:
    # Compute approximate GHI using solar geometry; not accounting for EoT or atmos effects
    lat = np.deg2rad(lat_deg)
    doy = index.dayofyear.values.astype(float)
    hour = index.hour + index.minute / 60.0
    # Declination (Cooper)
    delta = np.deg2rad(23.45) * np.sin(np.deg2rad(360.0 * (284.0 + doy) / 365.0))
    # Hour angle (approx): 15 deg per hour from solar noon ~ 12:00 local clock
    h_angle = np.deg2rad(15.0) * (hour - 12.0)
    cos_z = np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(h_angle)
    cos_z = np.clip(cos_z, 0.0, 1.0)
    s0 = 1000.0  # peak clear-sky GHI W/m2
    ghi = s0 * cos_z * cloud
    return pd.Series(ghi, index=index, name="solar_irradiance_Wm2")


def _download_hourly_weather(tc: TimeConfig, lat: float, lon: float) -> Optional[pd.DataFrame]:
    if not _HAS_METEOSTAT:
        return None
    try:
        point = Point(lat, lon)
        # meteostat Hourly returns tz-aware index if timezone specified
        data = Hourly(point, tc.start.tz_convert("UTC").tz_localize(None).to_pydatetime(),
                      tc.end.tz_convert("UTC").tz_localize(None).to_pydatetime(), timezone=tc.tz)
        df = data.fetch()
        if df is None or df.empty:
            return None
        # Normalize columns
        out = pd.DataFrame(index=pd.DatetimeIndex(df.index).tz_convert(tc.tz))
        # Ambient temp C
        out["ambient_temp_C"] = df.get("temp")
        # Relative humidity %
        if "rhum" in df:
            out["relative_humidity_pct"] = df["rhum"].clip(0, 100)
        else:
            out["relative_humidity_pct"] = 50.0
        # Wind speed/direction
        out["wind_speed_mps"] = (df.get("wspd") or 0.0) / 3.6  # km/h -> m/s if present
        out["wind_dir_deg"] = df.get("wdir") or 0.0
        # Rainfall mm
        out["rain_mm"] = df.get("prcp") or 0.0
        # Cloud factor: derive from precip and relative humidity crudely
        cloud = 1.0 - 0.6 * (out["relative_humidity_pct"].fillna(50.0) / 100.0) - 0.4 * (out["rain_mm"].fillna(0.0) > 0).astype(float)
        cloud = np.clip(cloud.values, 0.2, 1.0)
        out["solar_irradiance_Wm2"] = _approx_solar_irradiance(out.index, lat, cloud)
        return out
    except Exception:
        return None


def generate_weather(tc: TimeConfig, wc: WeatherConfig, seed: Optional[int] = None) -> pd.DataFrame:
    """Return weather dataframe at the desired resolution.

    Columns:
      - ambient_temp_C, relative_humidity_pct, wind_speed_mps, wind_dir_deg,
        rain_mm, solar_irradiance_Wm2
    """
    rng = get_rng(seed)

    # Try download at hourly resolution; resample to target
    df_hourly: Optional[pd.DataFrame] = None
    if wc.use_download:
        df_hourly = _download_hourly_weather(tc, wc.latitude, wc.longitude)

    index = tc.make_index()

    if df_hourly is not None and not df_hourly.empty:
        # Interpolate to target resolution
        df = df_hourly.reindex(index.union(df_hourly.index)).interpolate().reindex(index)
        return df

    # Synthetic weather
    n = len(index)
    # Ambient temp seasonal baseline
    temp_season = 10.0 + 12.0 * seasonality_day_of_year(index, amplitude=1.0)  # -2..+22C around 10C mean
    diurnal = 4.0 * np.sin(2.0 * np.pi * (index.hour.values + index.minute.values / 60.0 - 8.0) / 24.0)
    ambient_temp = temp_season + diurnal + rng.normal(0.0, 1.5, size=n)

    # Relative humidity inversely correlated with temp
    rh = 65.0 - 0.5 * (ambient_temp - ambient_temp.mean()) + rng.normal(0.0, 5.0, size=n)
    rh = np.clip(rh, 15.0, 100.0)

    # Wind speed and direction
    wind_speed = np.clip(rng.gamma(shape=2.0, scale=1.0, size=n), 0.0, None)  # m/s
    wind_dir = rng.uniform(0.0, 360.0, size=n)

    # Rainfall as a two-state Markov process
    rain_state = np.zeros(n)
    p_start = 0.03  # chance to start raining
    p_stop = 0.30   # chance to stop raining
    raining = False
    for i in range(n):
        if not raining and rng.random() < p_start:
            raining = True
        elif raining and rng.random() < p_stop:
            raining = False
        rain_state[i] = 1.0 if raining else 0.0
    rain_intensity = rain_state * np.maximum(0.0, rng.normal(1.2, 0.6, size=n))  # mm per interval

    # Cloud factor inversely related to rain and humidity
    cloud = 1.0 - 0.5 * (rh / 100.0) - 0.4 * rain_state
    cloud = np.clip(cloud, 0.2, 1.0)

    solar = _approx_solar_irradiance(index, wc.latitude, cloud)

    df = pd.DataFrame(
        index=index,
        data={
            "ambient_temp_C": ambient_temp,
            "relative_humidity_pct": rh,
            "wind_speed_mps": wind_speed,
            "wind_dir_deg": wind_dir,
            "rain_mm": rain_intensity,
            "solar_irradiance_Wm2": solar.values,
        },
    )
    return df
