from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import os
import json
import numpy as np
import pandas as pd

from .utils import TimeConfig, seeded_random_state, daily_profile, weekly_scaler, add_measurement_noise, first_order_response, clamp, co2_mass_balance, dew_point_temperature
from .weather import WeatherData


@dataclass
class Dataset:
    occupancy: pd.DataFrame
    utilization: pd.DataFrame
    setpoints: pd.DataFrame
    hvac: pd.DataFrame
    ieq: pd.DataFrame
    energy_whole: pd.DataFrame
    submeter: pd.DataFrame
    windows_blinds: pd.DataFrame


def _simulate_occupancy(time_cfg: TimeConfig, num_zones: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = time_cfg.index
    base = daily_profile(idx, peak_hour=11, spread_hours=3.0, amplitude=1.0) * weekly_scaler(idx, weekend_scale=0.15)
    seasonal = 0.8 + 0.2 * np.sin(2 * np.pi * (idx.dayofyear - 180) / 365.0)
    building_capacity = 1000
    occupants_total = clamp((base * seasonal * building_capacity).rename("occupants"), 0, building_capacity)
    occupants_total = add_measurement_noise(occupants_total, rng, sigma=20.0).clip(lower=0)

    # Zone splits (e.g., 5 zones with different mixes)
    weights = rng.dirichlet(alpha=np.ones(num_zones))
    zone_cols = [f"zone_{i+1}" for i in range(num_zones)]
    zone_occupancy = pd.DataFrame({c: occupants_total.values * w for c, w in zip(zone_cols, weights)}, index=idx)

    # Utilization via PIR motion rates and desk/room booking busy fraction per zone
    # Motion counts per 15 min proportional to occupants with noise
    motion = zone_occupancy.apply(lambda s: (s / 5.0) + rng.normal(0, 2.0, size=len(s)))
    motion = motion.clip(lower=0).add_prefix("motion_")
    # Booking busy fraction
    busy = zone_occupancy.apply(lambda s: clamp((s / max(float(s.max()), 1.0)) * 0.85 + rng.normal(0, 0.05, size=len(s)), 0.0, 1.0))
    busy = busy.add_prefix("busy_")

    occ_df = pd.concat([
        occupants_total.rename("occupants_total"),
        zone_occupancy
    ], axis=1)

    util_df = pd.concat([motion, busy], axis=1)

    return occ_df, util_df


def _simulate_setpoints_and_hvac(time_cfg: TimeConfig, weather: WeatherData, num_zones: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = time_cfg.index
    # Setpoints vary by schedule (occupied vs unoccupied) and season
    heating_sp = 20.0 + 1.0 * np.sin(2 * np.pi * (idx.dayofyear - 30) / 365.0)
    cooling_sp = 24.0 - 1.0 * np.sin(2 * np.pi * (idx.dayofyear - 30) / 365.0)

    occupied = (daily_profile(idx, peak_hour=11, spread_hours=3.0, amplitude=1.0) > 0.1) & (weekly_scaler(idx, weekend_scale=0.2) > 0.2)
    heating_sp = np.where(occupied, heating_sp, heating_sp - 3.0)
    cooling_sp = np.where(occupied, cooling_sp, cooling_sp + 3.0)

    sp_df = pd.DataFrame({"heating_sp_c": heating_sp, "cooling_sp_c": cooling_sp}, index=idx)

    # HVAC: Compute supply air temperature targets and statuses based on outdoor conditions
    oat = weather.frame["outdoor_temp_c"]
    cooling_demand = clamp(pd.Series(cooling_sp - oat, index=idx), -10, 15)
    heating_demand = clamp(pd.Series(oat - heating_sp, index=idx), -15, 10)

    sat_target = 14.0 + 6.0 * (cooling_demand < 0) + rng.normal(0, 0.3, size=len(idx))  # cooler SAT during cooling
    rat = 22.0 + 0.2 * np.sin(2 * np.pi * (idx.hour + idx.minute/60.0) / 24.0) + rng.normal(0, 0.3, size=len(idx))

    fan_speed = clamp(pd.Series(0.4 + 0.6 * (cooling_demand > 0) + rng.normal(0, 0.05, size=len(idx)), index=idx), 0.1, 1.0)
    oa_damper = clamp(pd.Series(0.1 + 0.4 * (oat > 15) + rng.normal(0, 0.05, size=len(idx)), index=idx), 0.05, 1.0)

    chiller_on = ((oat > 20) & (cooling_demand > 2)).astype(int)
    boiler_on = ((oat < 10) & (heating_demand > 2)).astype(int)

    hvac_df = pd.DataFrame({
        "supply_air_temp_c": sat_target,
        "return_air_temp_c": rat,
        "fan_vfd_pct": fan_speed * 100.0,
        "oa_damper_pct": oa_damper * 100.0,
        "chiller_status": chiller_on,
        "boiler_status": boiler_on,
    }, index=idx)

    return sp_df, hvac_df


def _simulate_ieq(time_cfg: TimeConfig, weather: WeatherData, occupancy: pd.DataFrame, hvac: pd.DataFrame, num_zones: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = time_cfg.index
    oat = weather.frame["outdoor_temp_c"]

    # Zone air temperature follows blend of SAT, setpoints, OAT with first-order response
    base_zone_temp = 0.5 * hvac["supply_air_temp_c"] + 0.2 * oat + 0.3 * 22.0
    zone_temp = first_order_response(base_zone_temp, tau_hours=1.5, dt_minutes=pd.Timedelta(time_cfg.freq).total_seconds()/60.0)

    # Relative humidity derived with noise around outdoor RH brought indoors
    rh = clamp(weather.frame["outdoor_rh_pct"] * 0.8 + rng.normal(0, 3.0, size=len(idx)), 20, 70)

    # CO2 model per zone using occupancy shares
    outdoor_co2 = pd.Series(420.0 + 40.0 * np.sin(2*np.pi*(idx.hour)/24.0), index=idx)
    ventilation_ach = clamp(pd.Series(0.8 + 1.2 * (hvac["fan_vfd_pct"] / 100.0), index=idx), 0.5, 5.0)

    zone_cols = [c for c in occupancy.columns if c.startswith("zone_")]
    co2_cols: dict[str, pd.Series] = {}
    for c in zone_cols:
        occ = occupancy[c]
        co2 = co2_mass_balance(
            co2_outdoor_ppm=outdoor_co2,
            occupancy=occ,
            ventilation_ach=ventilation_ach,
            room_volume_m3=1500.0,
            emission_rate_lps_per_person=0.005,
            dt_minutes=pd.Timedelta(time_cfg.freq).total_seconds()/60.0,
        )
        co2_cols[f"co2_{c}_ppm"] = clamp(co2, 380, 2000)

    # PM and TVOC correlated with occupancy and ventilation
    occ_total = occupancy["occupants_total"].clip(lower=0)
    pm25 = clamp(pd.Series(5.0 + 0.01 * occ_total - 0.5 * (ventilation_ach - 1.0) + rng.normal(0, 1.0, size=len(idx)), index=idx), 1, 75)
    pm10 = pm25 * (1.5 + rng.normal(0, 0.1, size=len(idx)))
    tvoc = clamp(pd.Series(150.0 + 0.2 * occ_total - 20.0 * (ventilation_ach - 1.0) + rng.normal(0, 20.0, size=len(idx)), index=idx), 50, 1200)

    # Illuminance: mix of solar and artificial lighting tied to occupancy
    lux_daylight = clamp(pd.Series(weather.frame["irradiance_wm2"] * 1.2 + rng.normal(0, 50, size=len(idx)), index=idx), 0, 1200)
    lights_on = (occ_total > occ_total.quantile(0.3)).astype(int)
    lux_lighting = lights_on * (300 + rng.normal(0, 30, size=len(idx)))
    lux = clamp(lux_daylight + lux_lighting, 0, 2000)

    # Noise: correlated with occupancy
    noise_db = clamp(pd.Series(40 + 10 * np.log10(1 + occ_total / 100.0) + rng.normal(0, 2.0, size=len(idx)), index=idx), 35, 85)

    ieq_df = pd.DataFrame({
        "zone_air_temp_c": zone_temp,
        "zone_rh_pct": rh,
        "illuminance_lux": lux,
        "noise_db": noise_db,
        "pm25_ugm3": pm25,
        "pm10_ugm3": pm10,
        "tvoc_ppb": tvoc,
    }, index=idx)
    for k, v in co2_cols.items():
        ieq_df[k] = v

    return ieq_df


def _simulate_energy(time_cfg: TimeConfig, weather: WeatherData, occupancy: pd.DataFrame, hvac: pd.DataFrame, num_zones: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = time_cfg.index
    oat = weather.frame["outdoor_temp_c"]
    occ_total = occupancy["occupants_total"].clip(lower=0)

    # Base loads
    base_plug_kw = 50 + 0.1 * occ_total + rng.normal(0, 5, size=len(idx))
    lighting_kw = 30 + 0.05 * occ_total + rng.normal(0, 3, size=len(idx))

    # HVAC electric loads tied to fan VFD and chiller status
    fans_kw = 20 + 80 * (hvac["fan_vfd_pct"] / 100.0) ** 3
    pumps_kw = 10 + 20 * (hvac["chiller_status"] + hvac["boiler_status"]) / 2.0
    chiller_kw = 0 + 250 * hvac["chiller_status"] * np.maximum(0, (oat - 18) / 10.0)
    boiler_gas_kw = 0 + 300 * hvac["boiler_status"] * np.maximum(0, (18 - oat) / 10.0)

    # Water use tied to occupancy with diurnal profile
    water_m3ph = clamp(pd.Series(0.5 + 0.003 * occ_total + 0.2 * (np.sin(2*np.pi*(idx.hour)/24.0) > 0).astype(int), index=idx), 0.1, 10.0)

    # District heating/cooling optional (derive from boiler/chiller)
    district_cooling_kwh = chiller_kw * 0.1
    district_heating_kwh = boiler_gas_kw * 0.1

    whole = pd.DataFrame({
        "electricity_kwh": clamp(pd.Series(base_plug_kw + lighting_kw + fans_kw + pumps_kw + chiller_kw, index=idx), 5, None),
        "gas_kwh": clamp(pd.Series(boiler_gas_kw, index=idx), 0, None),
        "water_m3": water_m3ph * (pd.Timedelta(time_cfg.freq).total_seconds() / 3600.0),
        "district_cooling_kwh": district_cooling_kwh,
        "district_heating_kwh": district_heating_kwh,
    }, index=idx)

    # Submeters: HVAC, lighting, plug, fans, pumps
    sub = pd.DataFrame({
        "hvac_chiller_kwh": chiller_kw,
        "hvac_fans_kwh": fans_kw,
        "hvac_pumps_kwh": pumps_kw,
        "lighting_kwh": lighting_kw,
        "plug_kwh": base_plug_kw,
    }, index=idx)

    return whole, sub


def _simulate_windows_blinds(time_cfg: TimeConfig, weather: WeatherData, occupancy: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    idx = time_cfg.index
    occ_total = occupancy["occupants_total"].clip(lower=0)
    irradiance = weather.frame["irradiance_wm2"]
    oat = weather.frame["outdoor_temp_c"]

    # Window opening probability influenced by temp and occupancy
    prob_window = clamp(pd.Series(0.05 + 0.002 * (oat - 18) + 0.0002 * occ_total + 0.02 * (irradiance < 50).astype(int), index=idx), 0.0, 0.6)
    windows_open = (rng.random(len(idx)) < prob_window).astype(int)

    # Blinds position 0-100% based on irradiance
    blinds_pct = clamp(pd.Series(100.0 * (irradiance / (irradiance.max() + 1e-6)), index=idx), 0.0, 100.0)
    blinds_pct = (0.7 * blinds_pct + 30 * (irradiance > irradiance.quantile(0.6)).astype(int) + rng.normal(0, 5.0, size=len(idx))).clip(0, 100)

    return pd.DataFrame({
        "windows_open": windows_open,
        "blinds_position_pct": blinds_pct,
    }, index=idx)


def generate_all_streams(time_cfg: TimeConfig, weather: WeatherData, num_zones: int, rng: np.random.Generator) -> Dataset:
    occ, util = _simulate_occupancy(time_cfg, num_zones, rng)
    sp, hvac = _simulate_setpoints_and_hvac(time_cfg, weather, num_zones, rng)
    ieq = _simulate_ieq(time_cfg, weather, occ, hvac, num_zones, rng)
    whole, sub = _simulate_energy(time_cfg, weather, occ, hvac, num_zones, rng)
    wb = _simulate_windows_blinds(time_cfg, weather, occ, rng)

    return Dataset(
        occupancy=occ,
        utilization=util,
        setpoints=sp,
        hvac=hvac,
        ieq=ieq,
        energy_whole=whole,
        submeter=sub,
        windows_blinds=wb,
    )


def build_catalog(building_name: str, time_cfg: TimeConfig, num_zones: int) -> dict:
    sensors = []
    # Weather sensors
    sensors += [
        {"stream": "weather", "field": "outdoor_temp_c", "unit": "C", "desc": "Outdoor air temperature"},
        {"stream": "weather", "field": "outdoor_rh_pct", "unit": "%", "desc": "Outdoor relative humidity"},
        {"stream": "weather", "field": "wind_speed_ms", "unit": "m/s", "desc": "Wind speed"},
        {"stream": "weather", "field": "wind_dir_deg", "unit": "deg", "desc": "Wind direction"},
        {"stream": "weather", "field": "rain_mm", "unit": "mm", "desc": "Rainfall depth"},
        {"stream": "weather", "field": "irradiance_wm2", "unit": "W/m2", "desc": "Global horizontal irradiance"},
    ]
    # Occupancy
    sensors += [
        {"stream": "occupancy", "field": "occupants_total", "unit": "people", "desc": "Total occupant count"},
    ]
    for i in range(num_zones):
        sensors.append({"stream": "occupancy", "field": f"zone_{i+1}", "unit": "people", "desc": f"Zone {i+1} occupants"})
        sensors.append({"stream": "utilization", "field": f"motion_zone_{i+1}", "unit": "counts/15min", "desc": f"Zone {i+1} motion counts"})
        sensors.append({"stream": "utilization", "field": f"busy_zone_{i+1}", "unit": "fraction", "desc": f"Zone {i+1} busy fraction"})
        sensors.append({"stream": "ieq", "field": f"co2_zone_{i+1}_ppm", "unit": "ppm", "desc": f"Zone {i+1} CO2"})

    # Setpoints and HVAC
    sensors += [
        {"stream": "setpoints", "field": "heating_sp_c", "unit": "C", "desc": "Heating setpoint"},
        {"stream": "setpoints", "field": "cooling_sp_c", "unit": "C", "desc": "Cooling setpoint"},
        {"stream": "hvac", "field": "supply_air_temp_c", "unit": "C", "desc": "Supply air temperature"},
        {"stream": "hvac", "field": "return_air_temp_c", "unit": "C", "desc": "Return air temperature"},
        {"stream": "hvac", "field": "fan_vfd_pct", "unit": "%", "desc": "Fan speed"},
        {"stream": "hvac", "field": "oa_damper_pct", "unit": "%", "desc": "Outside air damper"},
        {"stream": "hvac", "field": "chiller_status", "unit": "0/1", "desc": "Chiller status"},
        {"stream": "hvac", "field": "boiler_status", "unit": "0/1", "desc": "Boiler status"},
    ]

    # IEQ common
    sensors += [
        {"stream": "ieq", "field": "zone_air_temp_c", "unit": "C", "desc": "Zone air temperature"},
        {"stream": "ieq", "field": "zone_rh_pct", "unit": "%", "desc": "Zone relative humidity"},
        {"stream": "ieq", "field": "illuminance_lux", "unit": "lux", "desc": "Illuminance"},
        {"stream": "ieq", "field": "noise_db", "unit": "dB", "desc": "Noise level"},
        {"stream": "ieq", "field": "pm25_ugm3", "unit": "ug/m3", "desc": "PM2.5"},
        {"stream": "ieq", "field": "pm10_ugm3", "unit": "ug/m3", "desc": "PM10"},
        {"stream": "ieq", "field": "tvoc_ppb", "unit": "ppb", "desc": "TVOC"},
    ]

    # Energy
    sensors += [
        {"stream": "energy_whole", "field": "electricity_kwh", "unit": "kWh/interval", "desc": "Whole-building electricity"},
        {"stream": "energy_whole", "field": "gas_kwh", "unit": "kWh/interval", "desc": "Gas consumption"},
        {"stream": "energy_whole", "field": "water_m3", "unit": "m3/interval", "desc": "Water consumption"},
        {"stream": "energy_whole", "field": "district_cooling_kwh", "unit": "kWh/interval", "desc": "District cooling"},
        {"stream": "energy_whole", "field": "district_heating_kwh", "unit": "kWh/interval", "desc": "District heating"},
        {"stream": "submeter", "field": "hvac_chiller_kwh", "unit": "kWh/interval", "desc": "Chiller submeter"},
        {"stream": "submeter", "field": "hvac_fans_kwh", "unit": "kWh/interval", "desc": "Fans submeter"},
        {"stream": "submeter", "field": "hvac_pumps_kwh", "unit": "kWh/interval", "desc": "Pumps submeter"},
        {"stream": "submeter", "field": "lighting_kwh", "unit": "kWh/interval", "desc": "Lighting submeter"},
        {"stream": "submeter", "field": "plug_kwh", "unit": "kWh/interval", "desc": "Plug loads submeter"},
    ]

    # Windows/blinds
    sensors += [
        {"stream": "windows_blinds", "field": "windows_open", "unit": "0/1", "desc": "Window open state"},
        {"stream": "windows_blinds", "field": "blinds_position_pct", "unit": "%", "desc": "Blinds position"},
    ]

    return {
        "building": building_name,
        "sampling": time_cfg.freq,
        "timezone": "UTC",
        "start": time_cfg.start.isoformat(),
        "end": time_cfg.end.isoformat(),
        "zones": num_zones,
        "sensors": sensors,
    }


def write_outputs(output_dir: str, weather: WeatherData, data: Dataset, catalog: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    # Write dataframes
    weather.frame.to_csv(os.path.join(output_dir, "weather.csv"), index_label="timestamp")
    data.occupancy.to_csv(os.path.join(output_dir, "occupancy.csv"), index_label="timestamp")
    data.utilization.to_csv(os.path.join(output_dir, "utilization.csv"), index_label="timestamp")
    data.setpoints.to_csv(os.path.join(output_dir, "setpoints.csv"), index_label="timestamp")
    data.hvac.to_csv(os.path.join(output_dir, "hvac.csv"), index_label="timestamp")
    data.ieq.to_csv(os.path.join(output_dir, "ieq.csv"), index_label="timestamp")
    data.energy_whole.to_csv(os.path.join(output_dir, "energy_whole.csv"), index_label="timestamp")
    data.submeter.to_csv(os.path.join(output_dir, "submeter.csv"), index_label="timestamp")
    data.windows_blinds.to_csv(os.path.join(output_dir, "windows_blinds.csv"), index_label="timestamp")

    with open(os.path.join(output_dir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)
