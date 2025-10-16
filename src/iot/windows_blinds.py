from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, workday_profile


@dataclass
class WindowsBlindsConfig:
    window_open_bias: float = 0.05
    blind_close_bias: float = 0.1


def generate_windows_blinds(tc: TimeConfig, cfg: WindowsBlindsConfig, weather: pd.DataFrame, occupancy_df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    rng = get_rng(seed)
    idx = tc.make_index()

    occ = occupancy_df["occupant_count"].values
    occ_factor = np.clip(occ / (occ.max() + 1e-9), 0.0, 1.0)
    work = workday_profile(idx)

    temp = weather["ambient_temp_C"].values
    solar = weather["solar_irradiance_Wm2"].values

    # Window open probability increases with mild temps and occupancy
    comfort = np.exp(-((temp - 22.0) ** 2) / (2 * 4.0 ** 2))  # peak near 22C
    p_open = cfg.window_open_bias + 0.3 * comfort * np.maximum(work, occ_factor)
    p_open = np.clip(p_open, 0.0, 0.5)

    # Blind close probability increases with solar and occupancy
    solar_norm = np.clip(solar / 800.0, 0.0, 1.0)
    p_blind = cfg.blind_close_bias + 0.5 * solar_norm * np.maximum(work, occ_factor)
    p_blind = np.clip(p_blind, 0.0, 0.9)

    windows_open = (rng.random(len(idx)) < p_open).astype(int)
    blinds_closed = (rng.random(len(idx)) < p_blind).astype(int)

    df = pd.DataFrame({
        "window_open": windows_open,
        "blinds_closed": blinds_closed,
    }, index=idx)

    return df
