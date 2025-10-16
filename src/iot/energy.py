from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, workday_profile, seasonality_day_of_year


@dataclass
class EnergyConfig:
    floor_area_m2: float = 25000.0
    elec_base_kw: float = 400.0
    elec_peak_kw: float = 1500.0
    gas_base_kw: float = 0.0
    water_base_m3ph: float = 1.5
    district_cooling_kw_per_C: float = 50.0
    district_heating_kw_per_C: float = 45.0


def _cooling_load(index: pd.DatetimeIndex, ambient_C: np.ndarray, occ_factor: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    # Cooling proportional to (T - 22C)+ and occupancy and irradiance proxy via time-of-day
    t_excess = np.clip(ambient_C - 22.0, 0.0, None)
    solar_proxy = np.sin(2.0 * np.pi * (index.hour + index.minute / 60.0 - 6.0) / 24.0)
    solar_proxy = np.clip(solar_proxy, 0.0, None)
    return (cfg.district_cooling_kw_per_C * (t_excess + 0.3 * solar_proxy) * (0.4 + 0.6 * occ_factor))


def _heating_load(ambient_C: np.ndarray, occ_factor: np.ndarray, cfg: EnergyConfig) -> np.ndarray:
    t_deficit = np.clip(20.0 - ambient_C, 0.0, None)
    return cfg.district_heating_kw_per_C * t_deficit * (0.3 + 0.7 * occ_factor)


def generate_energy(tc: TimeConfig, cfg: EnergyConfig, weather: pd.DataFrame, occ_factor: np.ndarray, seed: Optional[int] = None) -> pd.DataFrame:
    rng = get_rng(seed)
    idx = tc.make_index()

    # Electric: base + workday peaks tied to occupancy
    work_profile = workday_profile(idx)
    elec_kw = cfg.elec_base_kw + cfg.elec_peak_kw * np.maximum(work_profile, occ_factor)
    # Add noise and small seasonality (more fan power in summer)
    elec_kw += 50.0 * seasonality_day_of_year(idx, amplitude=1.0) + rng.normal(0.0, 30.0, size=len(idx))
    elec_kw = np.clip(elec_kw, 0.0, None)

    # Gas: primarily heating in cold weather
    ambient_C = weather["ambient_temp_C"].values
    heat_kw = _heating_load(ambient_C, np.maximum(work_profile, occ_factor), cfg)
    gas_kw = cfg.gas_base_kw + heat_kw + rng.normal(0.0, 20.0, size=len(idx))
    gas_kw = np.clip(gas_kw, 0.0, None)

    # District cooling: proportional to cooling load
    cool_kw = _cooling_load(idx, ambient_C, np.maximum(work_profile, occ_factor), cfg)
    dist_cool_kw = np.clip(cool_kw + rng.normal(0.0, 30.0, size=len(idx)), 0.0, None)

    # Water: base + occupancy usage
    water_m3ph = cfg.water_base_m3ph + 0.001 * np.maximum(work_profile, occ_factor) * cfg.floor_area_m2 + rng.normal(0.0, 0.2, size=len(idx))
    water_m3ph = np.clip(water_m3ph, 0.0, None)

    # Submetering: split electricity into HVAC, lighting, plug loads
    hvac_kw = 0.4 * elec_kw + 0.5 * dist_cool_kw  # fans + chillers on electricity if onsite; we model district separately
    lighting_kw = 0.25 * elec_kw * (0.2 + 0.8 * work_profile)
    plugs_kw = np.clip(elec_kw - hvac_kw - lighting_kw, 0.0, None)

    df = pd.DataFrame({
        "electric_kw_whole": elec_kw,
        "gas_kw_whole": gas_kw,
        "water_m3ph": water_m3ph,
        "district_cooling_kw": dist_cool_kw,
        "hvac_kw": hvac_kw,
        "lighting_kw": lighting_kw,
        "plugs_kw": plugs_kw,
    }, index=idx)

    return df
