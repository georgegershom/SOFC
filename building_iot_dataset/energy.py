from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class EnergyConfig:
    lighting_w_per_m2: float = 7.0
    plug_w_per_person: float = 50.0
    base_plug_kw: float = 50.0
    pumps_kw: float = 40.0
    server_kw: float = 30.0
    water_m3_per_person_per_day: float = 0.05
    gas_boiler_kw_per_kw_heat: float = 1.11  # inverse efficiency


def compute_energy(index: pd.DatetimeIndex, occupancy: pd.DataFrame, hvac_df: pd.DataFrame, floor_area_m2: float) -> pd.DataFrame:
    # Lighting ~ occupied and lux setpoint driving electric light (approx from IEQ not available here). Use occupancy proxy.
    occ = occupancy["occupancy_building"].values
    hours = (index[1] - index[0]).total_seconds() / 3600.0

    lighting_kw = (floor_area_m2 * 7.0 / 1000.0) * (occ > 50).astype(float) * (0.3 + 0.7 * (occ / (occ.max() + 1e-6)))
    plug_kw = 0.5 * (occupancy["wifi_client_count"].values / (occupancy["wifi_client_count"].max() + 1e-6)) * 200 + 50
    pumps_kw = np.full(len(index), 40.0)
    server_kw = np.full(len(index), 30.0)

    # HVAC from simulated powers
    cooling_kw = hvac_df["cooling_power_kw"].values
    heating_kw = hvac_df["heating_power_kw"].values
    fan_kw = hvac_df["ahu_fan_power_kw"].values

    elec_kw = lighting_kw + plug_kw + pumps_kw + server_kw + cooling_kw + fan_kw
    gas_kw = heating_kw * 1.11

    # Water
    water_m3 = (occupancy["occupancy_building"].rolling(96, min_periods=1).mean() * (0.05 / 96.0)).values

    df = pd.DataFrame({
        "electricity_kw": elec_kw,
        "gas_kw": gas_kw,
        "water_m3ph": water_m3,
        "cooling_kw": cooling_kw,
        "heating_kw": heating_kw,
        "fan_kw": fan_kw,
        "lighting_kw": lighting_kw,
        "plug_kw": plug_kw,
        "pumps_kw": pumps_kw,
        "server_kw": server_kw,
    }, index=index)

    return df
