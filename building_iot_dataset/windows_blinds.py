from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FacadeConfig:
    num_zones: int = 5
    seed: int = 42


def simulate_facade(index: pd.DatetimeIndex, weather: pd.DataFrame, occupancy: pd.DataFrame, cfg: FacadeConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    solar = weather["solar_irradiance_wm2"].values
    t_out = weather["ambient_temp_c"].values

    num_zones = cfg.num_zones
    windows_open_frac = np.zeros((len(index), num_zones))
    blinds_pos = np.zeros((len(index), num_zones))

    for t in range(len(index)):
        occ = occupancy[[c for c in occupancy.columns if c.startswith("zone_")]].iloc[t].values
        comfort = (t_out[t] > 18) & (t_out[t] < 26)
        co2_proxy = occ > np.percentile(occ, 60)

        # Windows open when mild, occupied, and crowded
        windows_open_frac[t, :] = np.clip(0.3 * comfort + 0.4 * (occ > 5) + 0.3 * co2_proxy, 0, 1)
        # Blinds close with high solar
        blinds_pos[t, :] = np.clip(0.2 + 0.8 * (solar[t] / 800.0), 0, 1)

    data = {}
    for z in range(num_zones):
        data[f"zone_{z+1}_window_open_frac"] = windows_open_frac[:, z]
        data[f"zone_{z+1}_blinds_pos"] = blinds_pos[:, z]

    return pd.DataFrame(data, index=index)
