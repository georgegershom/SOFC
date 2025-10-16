from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HVACConfig:
    num_zones: int = 5
    floor_area_m2: float = 15000.0
    ach_infiltration: float = 0.3
    cooling_setpoint_occupied_c: float = 24.0
    heating_setpoint_occupied_c: float = 21.0
    setback_cooling_c: float = 28.0
    setback_heating_c: float = 16.0
    supply_air_temp_cooling_c: float = 12.0
    supply_air_temp_heating_c: float = 40.0
    air_heat_capacity_kj_per_kgk: float = 1.006
    air_density_kg_per_m3: float = 1.2
    zone_volume_m3_per_person: float = 20.0
    cop_cooling: float = 3.2
    boiler_efficiency: float = 0.9
    fan_efficiency: float = 0.6
    fan_max_power_kw: float = 150.0
    seed: int = 42


@dataclass
class ScheduleConfig:
    occupied_start_hour: int = 8
    occupied_end_hour: int = 18


def _setpoints(index: pd.DatetimeIndex, sched: ScheduleConfig, hvac: HVACConfig) -> pd.DataFrame:
    hour = index.hour
    is_weekend = (index.weekday >= 5)
    occupied = (hour >= sched.occupied_start_hour) & (hour < sched.occupied_end_hour) & (~is_weekend)

    cooling_sp = np.where(occupied, hvac.cooling_setpoint_occupied_c, hvac.setback_cooling_c)
    heating_sp = np.where(occupied, hvac.heating_setpoint_occupied_c, hvac.setback_heating_c)

    return pd.DataFrame({
        "cooling_setpoint_c": cooling_sp,
        "heating_setpoint_c": heating_sp,
        "occupied": occupied.astype(int),
    }, index=index)


def simulate_hvac(index: pd.DatetimeIndex, weather: pd.DataFrame, occupancy: pd.DataFrame, hvac: HVACConfig, sched: ScheduleConfig) -> pd.DataFrame:
    rng = np.random.default_rng(hvac.seed)
    sp = _setpoints(index, sched, hvac)

    num_zones = hvac.num_zones
    zone_temp = np.zeros((len(index), num_zones))
    zone_rh = np.zeros_like(zone_temp)

    # Initialize with ambient
    zone_temp[0, :] = weather["ambient_temp_c"].iloc[0]
    zone_rh[0, :] = np.clip(weather["ambient_rh_pct"].iloc[0] + rng.normal(0, 5, num_zones), 20, 70)

    # Thermal dynamics parameters
    tau_hours = 6.0
    dt_hours = (index[1] - index[0]).total_seconds() / 3600.0
    alpha = dt_hours / tau_hours

    # Internal gains per person (W)
    sensible_w_per_person = 75.0

    # Airflow per person (m3/h) when occupied
    supply_air_per_person_m3ph = 50.0

    # Data collectors
    cooling_valve = np.zeros_like(zone_temp)
    heating_valve = np.zeros_like(zone_temp)
    damper_pos = np.zeros_like(zone_temp)

    fan_power_kw = np.zeros(len(index))
    cooling_power_kw = np.zeros(len(index))
    heating_power_kw = np.zeros(len(index))

    # Additional AHU/system-level telemetry
    ahu_supply_air_temp_c = np.zeros(len(index))
    ahu_return_air_temp_c = np.zeros(len(index))
    ahu_fan_speed_pct = np.zeros(len(index))
    chiller_status = np.zeros(len(index), dtype=int)
    boiler_status = np.zeros(len(index), dtype=int)

    for t in range(1, len(index)):
        t_out = weather["ambient_temp_c"].iloc[t]
        solar = weather["solar_irradiance_wm2"].iloc[t]
        occ_total = occupancy[[c for c in occupancy.columns if c.startswith("zone_")]].iloc[t].values
        occupied_flag = sp["occupied"].iloc[t]

        # Simple control: target midpoint between setpoints
        target = 0.5 * (sp["cooling_setpoint_c"].iloc[t] + sp["heating_setpoint_c"].iloc[t])

        # HVAC action
        error = target - zone_temp[t-1, :]

        # Proportional valves and dampers
        cooling_valve[t, :] = np.clip(-error / 3.0, 0, 1)  # cool when too hot
        heating_valve[t, :] = np.clip(error / 3.0, 0, 1)   # heat when too cold
        damper_pos[t, :] = np.clip(0.2 + (occ_total / max(1, occ_total.max())) * 0.6, 0.2, 1.0)

        # Airflow from occupancy (m3/h)
        airflow_m3ph = np.clip(occ_total * supply_air_per_person_m3ph, 0.0, None)
        airflow_m3ps = airflow_m3ph / 3600.0

        # Temperature change: leakage to ambient + internal + HVAC
        leakage = alpha * (t_out - zone_temp[t-1, :])
        internal = (occ_total * sensible_w_per_person) / (1000.0 * 1.2 * 1.006) * dt_hours  # degC approx

        # HVAC cooling/heating effect per airflow
        mixed_air_temp = 0.7 * t_out + 0.3 * zone_temp[t-1, :]  # crude mix
        supply_cool = hvac.supply_air_temp_cooling_c
        supply_heat = hvac.supply_air_temp_heating_c

        cool_effect = (mixed_air_temp - supply_cool) * cooling_valve[t, :] * (airflow_m3ps * hvac.air_density_kg_per_m3 * hvac.air_heat_capacity_kj_per_kgk / 3600.0) * dt_hours
        heat_effect = (supply_heat - mixed_air_temp) * heating_valve[t, :] * (airflow_m3ps * hvac.air_density_kg_per_m3 * hvac.air_heat_capacity_kj_per_kgk / 3600.0) * dt_hours

        zone_temp[t, :] = zone_temp[t-1, :] + leakage + internal - cool_effect + heat_effect

        # Humidity: relax to ambient with occupancy moisture add
        rh_ambient = np.clip(weather["ambient_rh_pct"].iloc[t] + np.random.normal(0, 2), 15, 100)
        zone_rh[t, :] = np.clip(zone_rh[t-1, :] + 0.1 * (rh_ambient - zone_rh[t-1, :]) + 0.02 * np.sqrt(occ_total), 15, 75)

        # Energy
        # Fan power ~ cube of damper/airflow proxy
        airflow_fraction = np.clip(damper_pos[t, :].mean(), 0, 1)
        fan_power_kw[t] = hvac.fan_max_power_kw * (airflow_fraction ** 3) * (0.6 + 0.4 * occupied_flag)
        ahu_fan_speed_pct[t] = 100.0 * airflow_fraction

        # Cooling/heating loads (kW)
        cool_delta_k = np.clip(zone_temp[t-1, :]-sp["cooling_setpoint_c"].iloc[t], 0, None)
        heat_delta_k = np.clip(sp["heating_setpoint_c"].iloc[t]-zone_temp[t-1, :], 0, None)
        cooling_load_kw = (cool_delta_k * airflow_m3ps * hvac.air_density_kg_per_m3 * hvac.air_heat_capacity_kj_per_kgk) / 3.6
        heating_load_kw = (heat_delta_k * airflow_m3ps * hvac.air_density_kg_per_m3 * hvac.air_heat_capacity_kj_per_kgk) / 3.6
        cooling_power_kw[t] = cooling_load_kw.sum() / max(hvac.cop_cooling, 0.5)
        heating_power_kw[t] = heating_load_kw.sum() / max(hvac.boiler_efficiency, 0.5)

        # AHU supply/return temps and plant statuses
        ahu_return_air_temp_c[t] = float(zone_temp[t-1, :].mean())
        avg_cool = float(np.clip(cooling_valve[t, :].mean(), 0.0, 1.0))
        avg_heat = float(np.clip(heating_valve[t, :].mean(), 0.0, 1.0))
        idle_frac = max(0.0, 1.0 - (avg_cool + avg_heat))
        supply_mix = (
            avg_cool * hvac.supply_air_temp_cooling_c +
            avg_heat * hvac.supply_air_temp_heating_c +
            idle_frac * (ahu_return_air_temp_c[t] - 1.0)
        )
        # Keep within min/max practical bounds
        ahu_supply_air_temp_c[t] = float(np.clip(supply_mix, hvac.supply_air_temp_cooling_c, max(hvac.supply_air_temp_heating_c, ahu_return_air_temp_c[t])))
        chiller_status[t] = int(cooling_power_kw[t] > 5.0)
        boiler_status[t] = int(heating_power_kw[t] > 5.0)

    columns = {}
    for z in range(num_zones):
        columns[f"zone_{z+1}_air_temp_c"] = zone_temp[:, z]
        columns[f"zone_{z+1}_rh_pct"] = zone_rh[:, z]
        columns[f"zone_{z+1}_cool_valve_pos"] = np.clip(cooling_valve[:, z], 0, 1)
        columns[f"zone_{z+1}_heat_valve_pos"] = np.clip(heating_valve[:, z], 0, 1)
        columns[f"zone_{z+1}_vav_damper_pos"] = np.clip(damper_pos[:, z], 0, 1)

    # Add system-level & setpoint telemetry
    columns.update({
        "ahu_fan_power_kw": np.clip(fan_power_kw, 0, None),
        "ahu_fan_speed_pct": np.clip(ahu_fan_speed_pct, 0, 100),
        "ahu_supply_air_temp_c": ahu_supply_air_temp_c,
        "ahu_return_air_temp_c": ahu_return_air_temp_c,
        "cooling_power_kw": np.clip(cooling_power_kw, 0, None),
        "heating_power_kw": np.clip(heating_power_kw, 0, None),
        "chiller_status": chiller_status,
        "boiler_status": boiler_status,
        "cooling_setpoint_c": sp["cooling_setpoint_c"].values,
        "heating_setpoint_c": sp["heating_setpoint_c"].values,
        "occupied": sp["occupied"].values,
    })

    return pd.DataFrame(columns, index=index)
