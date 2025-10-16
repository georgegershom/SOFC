from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, workday_profile, clamp_series


@dataclass
class IEQConfig:
    num_zones: int = 6
    co2_outdoor_ppm: float = 420.0
    co2_per_person_lps: float = 0.005  # per-person CO2 generation (m3/s) approx
    ventilation_lps_per_person: float = 0.01


def _zone_temp_rh(index: pd.DatetimeIndex, ambient_C: pd.Series, occ_factor: np.ndarray, rng: np.random.Generator, zone_bias: float) -> pd.DataFrame:
    work = workday_profile(index)
    # Target temps shift by season; 22C in winter, 24C in summer typical
    season = np.sin(2.0 * np.pi * (index.dayofyear.values / 365.0))
    target = 23.0 + 1.0 * season
    temp = target + 0.5 * work + 0.2 * occ_factor + zone_bias + rng.normal(0.0, 0.3, size=len(index))

    # RH derived from ambient with dehumidification during occupied times
    rh = 50.0 + 0.1 * (ambient_C - ambient_C.mean()) - 5.0 * work + rng.normal(0.0, 2.0, size=len(index))
    rh = np.clip(rh, 25.0, 70.0)

    return pd.DataFrame({"air_temp_C": temp, "rh_pct": rh}, index=index)


def _co2_series(index: pd.DatetimeIndex, occ: np.ndarray, cfg: IEQConfig, rng: np.random.Generator) -> pd.Series:
    # Simple mass-balance: dC/dt = (G/V) - (Q/V)(C - C_out)
    dt_s = pd.Timedelta(index.freq).total_seconds() if index.freq is not None else 900.0
    # Use equivalent V/Q time constant ~ 30 minutes during occupied, slower when unoccupied
    tau_occ = 30.0 * 60.0
    tau_unocc = 180.0 * 60.0
    co2 = np.empty(len(index))
    co2[0] = cfg.co2_outdoor_ppm + 20.0
    for i in range(1, len(index)):
        tau = tau_occ if occ[i] > 0.1 else tau_unocc
        decay = np.exp(-dt_s / tau)
        gen = 800.0 * (occ[i] / (occ.max() + 1e-6))  # scale generation
        co2[i] = cfg.co2_outdoor_ppm + (co2[i-1] - cfg.co2_outdoor_ppm) * decay + gen * (1.0 - decay)
    co2 = co2 + rng.normal(0.0, 15.0, size=len(index))
    return pd.Series(co2, index=index, name="co2_ppm")


def _pm_tvoc(index: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    pm25 = np.clip(rng.lognormal(mean=1.5, sigma=0.3, size=len(index)), 2.0, 150.0)
    pm10 = pm25 * rng.uniform(1.2, 1.8, size=len(index))
    tvoc = np.clip(rng.lognormal(mean=0.5, sigma=0.4, size=len(index)) * 200.0, 50.0, 1000.0)
    return pd.DataFrame({"pm25_ugm3": pm25, "pm10_ugm3": pm10, "tvoc_ppb": tvoc}, index=index)


def _light_noise(index: pd.DatetimeIndex, occ_factor: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    hour = index.hour + index.minute / 60.0
    daylight = np.clip(np.sin(2.0 * np.pi * (hour - 6.0) / 24.0), 0.0, None)
    illuminance = 150.0 + 400.0 * daylight + 300.0 * occ_factor + rng.normal(0.0, 30.0, size=len(index))
    noise_db = 40.0 + 10.0 * occ_factor + rng.normal(0.0, 2.0, size=len(index))
    return pd.DataFrame({"illuminance_lux": illuminance, "noise_db": noise_db}, index=index)


def generate_ieq(tc: TimeConfig, cfg: IEQConfig, weather: pd.DataFrame, occupancy_df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    rng = get_rng(seed)
    idx = tc.make_index()
    occ = occupancy_df["occupant_count"].values
    occ_factor = np.clip(occ / (occ.max() + 1e-9), 0.0, 1.0)
    ambient_C = weather["ambient_temp_C"]

    # Build per-zone readings then aggregate columns
    cols = {}
    for z in range(cfg.num_zones):
        zone_bias = rng.normal(0.0, 0.6)
        zr = _zone_temp_rh(idx, ambient_C, occ_factor, rng, zone_bias)
        zr = zr.add_suffix(f"_z{z+1}")
        cols.update({c: zr[c] for c in zr.columns})

    co2 = _co2_series(idx, occ, cfg, rng)
    pm_tvoc = _pm_tvoc(idx, rng)
    light_noise = _light_noise(idx, occ_factor, rng)

    df = pd.DataFrame(cols, index=idx)
    df = pd.concat([df, co2, pm_tvoc, light_noise], axis=1)

    return df
