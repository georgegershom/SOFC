from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pvlib
from meteostat import Hourly, Point


@dataclass
class WeatherConfig:
    latitude: float
    longitude: float
    altitude_m: float
    timezone: str
    year: int


def _get_hourly_meteostat(lat: float, lon: float, alt_m: float, start: datetime, end: datetime, tz: str) -> pd.DataFrame:
    location = Point(lat, lon, alt_m)
    # Newer Meteostat versions do not accept a 'tz' argument on Hourly
    data = Hourly(location, start, end).fetch()
    # Meteostat returns columns in SI-ish units
    # temp (t) in C, relative humidity (rhum) %, pressure (pres) Pa, wind speed (wspd) km/h, wind direction (wdir) deg
    df = data.rename(
        columns={
            "temp": "temp_C",
            "rhum": "rel_humidity_pct",
            "pres": "pressure_pa",
            "wspd": "wind_speed_kmh",
            "wdir": "wind_dir_deg",
            "prcp": "precip_mm",
            "snow": "snow_cm",
            "tsun": "sunshine_min",
        }
    )

    # Meteostat sometimes uses 'dwpt' for dewpoint, ensure consistent naming
    if "dwpt" in df.columns:
        df = df.rename(columns={"dwpt": "dewpoint_C"})

    # Convert wind speed km/h to m/s
    if "wind_speed_kmh" in df.columns:
        df["wind_speed_ms"] = df["wind_speed_kmh"].astype(float) * (1000.0 / 3600.0)
    else:
        df["wind_speed_ms"] = np.nan

    # Ensure pressure in Pa (meteostat often gives hPa)
    # If values look like ~1013, treat as hPa and convert to Pa
    if "pressure_pa" in df.columns:
        median_pres = df["pressure_pa"].dropna().median()
        if median_pres is not None and median_pres < 2000:  # likely hPa
            df["pressure_pa"] = df["pressure_pa"] * 100.0

    # Index as timezone-aware hourly
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        # Meteostat typically provides UTC when tz is not set
        idx = idx.tz_localize("UTC").tz_convert(tz)
    else:
        idx = idx.tz_convert(tz)
    df.index = idx
    return df


def _clearsky_irradiance(lat: float, lon: float, alt_m: float, tz: str, times: pd.DatetimeIndex) -> pd.DataFrame:
    location = pvlib.location.Location(latitude=lat, longitude=lon, altitude=alt_m, tz=tz)
    solpos = location.get_solarposition(times)
    cs = location.get_clearsky(times, model="ineichen")  # GHI, DNI, DHI

    # Air mass and POA on a default surface for completeness
    airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"]).rename("airmass_rel")
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30.0,
        surface_azimuth=180.0,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=cs["dni"],
        ghi=cs["ghi"],
        dhi=cs["dhi"],
        albedo=0.2,
    )

    out = pd.concat([
        solpos[["apparent_zenith", "azimuth"]].rename(columns={"apparent_zenith": "solar_zenith_deg", "azimuth": "solar_azimuth_deg"}),
        cs.rename(columns={"ghi": "ghi_clearsky_Wm2", "dni": "dni_clearsky_Wm2", "dhi": "dhi_clearsky_Wm2"}),
        airmass,
        poa[["poa_global", "poa_direct", "poa_diffuse"]].rename(columns={
            "poa_global": "poa_global_Wm2",
            "poa_direct": "poa_direct_Wm2",
            "poa_diffuse": "poa_diffuse_Wm2",
        }),
    ], axis=1)

    return out


def _synthesize_tmy(df_hourly: pd.DataFrame, tz: str, year: int) -> pd.DataFrame:
    # Simple TMY-like: select provided year, fill gaps, and ensure complete hourly coverage
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    end = pd.Timestamp(f"{year}-12-31 23:00:00", tz=tz)
    idx = pd.date_range(start, end, freq="H", tz=tz)

    df = df_hourly.reindex(idx)
    # Forward/back fill minor gaps
    df = df.ffill().bfill()

    return df


def build_weather_dataset(cfg: WeatherConfig) -> pd.DataFrame:
    start = datetime(cfg.year, 1, 1)
    end = datetime(cfg.year, 12, 31, 23)

    met = _get_hourly_meteostat(
        lat=cfg.latitude,
        lon=cfg.longitude,
        alt_m=cfg.altitude_m,
        start=start,
        end=end,
        tz=cfg.timezone,
    )

    # Create uniform hourly index in timezone
    hourly_index = pd.date_range(
        pd.Timestamp(f"{cfg.year}-01-01 00:00:00", tz=cfg.timezone),
        pd.Timestamp(f"{cfg.year}-12-31 23:00:00", tz=cfg.timezone),
        freq="H",
    )

    # Clearsky series on the same index
    cs = _clearsky_irradiance(
        lat=cfg.latitude,
        lon=cfg.longitude,
        alt_m=cfg.altitude_m,
        tz=cfg.timezone,
        times=hourly_index,
    )

    # Merge
    df = met.reindex(hourly_index)
    df = pd.concat([df, cs], axis=1)

    # Basic QA: non-negative irradiance
    irr_cols = [c for c in df.columns if c.endswith("_Wm2")]
    for c in irr_cols:
        df[c] = df[c].clip(lower=0)

    # Attach metadata
    df["latitude"] = cfg.latitude
    df["longitude"] = cfg.longitude
    df["altitude_m"] = cfg.altitude_m
    df["timezone"] = cfg.timezone
    df["year"] = cfg.year

    # Reorder useful columns
    preferred = [
        "temp_C",
        "dewpoint_C",
        "rel_humidity_pct",
        "pressure_pa",
        "wind_speed_ms",
        "wind_dir_deg",
        "precip_mm",
        "solar_zenith_deg",
        "solar_azimuth_deg",
        "ghi_clearsky_Wm2",
        "dni_clearsky_Wm2",
        "dhi_clearsky_Wm2",
        "poa_global_Wm2",
        "poa_direct_Wm2",
        "poa_diffuse_Wm2",
        "latitude",
        "longitude",
        "altitude_m",
        "timezone",
        "year",
    ]

    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    df = df[cols]
    return df
