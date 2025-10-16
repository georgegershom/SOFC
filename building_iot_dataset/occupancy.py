from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OccupancyConfig:
    num_zones: int = 5
    building_capacity: int = 800
    weekday_peak_fraction: float = 0.65  # fraction of capacity
    weekend_peak_fraction: float = 0.1
    arrival_spread_hours: float = 2.0
    departure_spread_hours: float = 2.0
    lunch_dip_fraction: float = 0.15
    wfh_fraction: float = 0.2
    seed: int = 42


def _daily_profile(index: pd.DatetimeIndex, peak: float, arrival_spread: float, departure_spread: float, lunch_dip: float) -> pd.Series:
    seconds = index.hour * 3600 + index.minute * 60
    morning = np.exp(-((seconds - 9 * 3600) ** 2) / (2 * (arrival_spread * 3600) ** 2))
    evening = np.exp(-((seconds - 17 * 3600) ** 2) / (2 * (departure_spread * 3600) ** 2))
    workday = morning + evening
    workday = workday / np.max(workday)
    # Flatten to a plateau between 10-16h with lunch dip
    plateau = ((seconds >= 10 * 3600) & (seconds <= 16 * 3600)).astype(float)
    lunch = np.exp(-((seconds - 12.5 * 3600) ** 2) / (2 * (0.7 * 3600) ** 2))
    profile = 0.25 * workday + 0.75 * plateau - lunch_dip * lunch
    profile = np.clip(profile, 0, None)
    return pd.Series(peak * profile, index=index)


def generate_occupancy(index: pd.DatetimeIndex, cfg: OccupancyConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    is_weekend = index.weekday >= 5
    peak_fraction = np.where(is_weekend, cfg.weekend_peak_fraction, cfg.weekday_peak_fraction * (1 - cfg.wfh_fraction))
    daily_peak = cfg.building_capacity * peak_fraction

    base = _daily_profile(index, 1.0, cfg.arrival_spread_hours, cfg.departure_spread_hours, cfg.lunch_dip_fraction)
    occ_building = daily_peak * base.values

    # Add random day-to-day variability
    days = pd.Series(index.normalize()).astype("int64") // 10**9 // (24*3600)
    day_noise = pd.Series(rng.normal(1.0, 0.10, days.nunique()), index=np.unique(days.values))
    occ_building = occ_building * day_noise.loc[days.values].values

    occ_building = np.clip(occ_building, 0, cfg.building_capacity).astype(float)

    # People counters (in/out)
    occ_shifted = np.r_[0, np.diff(occ_building)]
    inflow = np.clip(occ_shifted, 0, None)
    outflow = np.clip(-occ_shifted, 0, None)

    # Distribute to zones with slowly varying proportions
    zone_weights = rng.dirichlet(np.ones(cfg.num_zones))
    zone_weights = np.tile(zone_weights, (len(index), 1))
    # Slowly drift weights
    drift = rng.normal(0, 0.02, size=zone_weights.shape)
    zone_weights = np.clip(zone_weights + np.cumsum(drift, axis=0), 0.05, None)
    zone_weights = zone_weights / zone_weights.sum(axis=1, keepdims=True)

    zone_occ = (occ_building.reshape(-1, 1) * zone_weights).astype(float)

    # Wi-Fi counts approx equal to occupancy with device factor and noise
    devices_per_person = rng.normal(1.6, 0.2, len(index))
    wifi_clients = np.clip(occ_building * devices_per_person, 0, None)

    data = {
        "occupancy_building": occ_building,
        "entrance_in_count": inflow,
        "entrance_out_count": outflow,
        "wifi_client_count": wifi_clients,
    }
    for z in range(cfg.num_zones):
        data[f"zone_{z+1}_occupancy"] = zone_occ[:, z]

    return pd.DataFrame(data, index=index)
