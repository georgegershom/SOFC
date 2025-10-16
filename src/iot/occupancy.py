from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, workday_profile


@dataclass
class OccupancyConfig:
    peak_occupancy: int = 1200
    arrival_spread_hours: float = 2.0
    departure_spread_hours: float = 2.0
    wfh_fraction: float = 0.2  # fraction working from home on weekdays


def generate_occupancy(tc: TimeConfig, oc: OccupancyConfig, seed: Optional[int] = None) -> pd.DataFrame:
    rng = get_rng(seed)
    idx = tc.make_index()

    base_profile = workday_profile(idx, start_hour=8.0, peak_hour=11.0, end_hour=18.0)

    # Arrival/departure smoothness via convolution with Gaussian kernel (in time)
    minutes = (idx - idx[0]).total_seconds().values / 60.0
    sigma_arr = oc.arrival_spread_hours * 60.0
    sigma_dep = oc.departure_spread_hours * 60.0
    # Use single sigma as average
    sigma = 0.5 * (sigma_arr + sigma_dep)
    kernel_size = int(max(3, sigma / (tc.step_hours * 60.0)))
    kernel_t = np.linspace(-3.0, 3.0, 2 * kernel_size + 1)
    kernel = np.exp(-0.5 * kernel_t ** 2)
    kernel /= kernel.sum()
    smoothed = np.convolve(base_profile, kernel, mode="same")

    # Weekday WFH reduction and random daily variation
    weekday = pd.Series(idx.weekday, index=idx)
    is_workday = (weekday < 5).astype(float)
    daily_mult = pd.Series(1.0, index=idx)
    for date, group in pd.Series(1.0, index=idx.date).groupby(level=0):
        day_mask = (idx.date == date)
        reduction = oc.wfh_fraction * is_workday[day_mask].max()  # 0 on weekends
        # Random daily scaling 0.9..1.1
        rand_scale = rng.uniform(0.9, 1.1)
        daily_mult.loc[day_mask] = (1.0 - reduction) * rand_scale

    occ = oc.peak_occupancy * smoothed * daily_mult.values
    # Add some short-term variability
    occ += rng.normal(0.0, 0.03 * oc.peak_occupancy, size=len(idx))
    occ = np.clip(occ, 0.0, None)

    # Wi-Fi clients as fraction of occupants + noise; aggregated and anonymized
    wifi = 0.75 * occ + rng.normal(0.0, 10.0, size=len(idx))
    wifi = np.clip(wifi, 0.0, None)

    df = pd.DataFrame({
        "occupant_count": occ,
        "wifi_clients_agg": wifi,
    }, index=idx)

    return df
