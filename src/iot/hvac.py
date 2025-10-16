from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .utils import TimeConfig, get_rng, workday_profile


@dataclass
class HVACConfig:
    supply_temp_setpoint_C: float = 13.0
    return_temp_offset_C: float = 8.0


def generate_hvac(tc: TimeConfig, cfg: HVACConfig, weather: pd.DataFrame, occupancy_df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
    rng = get_rng(seed)
    idx = tc.make_index()

    occ = occupancy_df["occupant_count"].values
    occ_factor = np.clip(occ / (occ.max() + 1e-9), 0.0, 1.0)
    work = workday_profile(idx)

    # Supply/return temps
    supply = cfg.supply_temp_setpoint_C + rng.normal(0.0, 0.4, size=len(idx))
    return_air = supply + cfg.return_temp_offset_C + 2.0 * occ_factor + rng.normal(0.0, 0.5, size=len(idx))

    # Fan speed and damper/valve positions 0..1
    fan_speed = np.clip(0.2 + 0.8 * np.maximum(work, occ_factor) + rng.normal(0.0, 0.05, size=len(idx)), 0.0, 1.0)
    oa_damper = np.clip(0.1 + 0.7 * np.maximum(work, occ_factor) + rng.normal(0.0, 0.05, size=len(idx)), 0.0, 1.0)
    chiller_on = ((weather["ambient_temp_C"].values > 20.0) & (fan_speed > 0.4)).astype(float)
    boiler_on = ((weather["ambient_temp_C"].values < 10.0) & (fan_speed > 0.3)).astype(float)
    valve_pos = np.clip(0.2 * chiller_on + 0.3 * boiler_on + rng.normal(0.0, 0.05, size=len(idx)), 0.0, 1.0)

    # Setpoints time series (heating/cooling)
    heat_sp = 21.0 - 0.5 * work
    cool_sp = 24.0 + 0.5 * work

    df = pd.DataFrame({
        "supply_air_temp_C": supply,
        "return_air_temp_C": return_air,
        "fan_speed_frac": fan_speed,
        "oa_damper_pos_frac": oa_damper,
        "valve_pos_frac": valve_pos,
        "chiller_status": chiller_on,
        "boiler_status": boiler_on,
        "heat_setpoint_C": heat_sp,
        "cool_setpoint_C": cool_sp,
    }, index=idx)

    return df
