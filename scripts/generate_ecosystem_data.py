#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import requests
except Exception:
    requests = None

# -------------------------
# Config and Utilities
# -------------------------
DEFAULT_LOCATION = {
    "city": "New York",
    "country": "USA",
    "latitude": 40.7128,
    "longitude": -74.0060,
    "altitude_m": 10.0,
    "timezone": "America/New_York",
    "urban_context": {
        "building_density": 0.65,  # 0-1 proxy for shading
        "tree_canopy": 0.2,
        "street_canyon_aspect_ratio": 1.5
    }
}

OUTPUT_ROOT = os.environ.get("ECOSYSTEM_DATA_DIR", "/workspace/data/ecosystem")

RNG = np.random.default_rng(42)


def ensure_dirs(root: str):
    for sub in ["weather", "climate", "economics", "geospatial", "carbon", "regulatory"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)


# -------------------------
# Weather: TMY-like fabrication
# -------------------------

def synthesize_tmy_hourly(location: Dict, year: int = 2022) -> pd.DataFrame:
    tz = pd.Timestamp.now(tz=location.get("timezone", None)).tz if location.get("timezone") else None
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz=tz)
    periods = 8760 if not pd.Timestamp(year=year, month=12, day=31).is_leap_year else 8784
    idx = pd.date_range(start=start, periods=periods, freq="H")

    day_of_year = idx.day_of_year.to_numpy()
    hour = idx.hour.to_numpy()

    # Air temperature seasonal + diurnal + noise
    t_season = 10 * np.sin(2 * np.pi * (day_of_year - 81) / 365)
    t_diurnal = 6 * np.sin(2 * np.pi * (hour - 8) / 24)
    base = 12  # mean annual temperature
    temp_c = base + t_season + t_diurnal + RNG.normal(0, 2, size=len(idx))

    # Global horizontal irradiance (simple clear-sky proxy with clouds)
    ghi_clear = np.maximum(0, 900 * np.sin(np.pi * (hour - 6) / 12))
    cloud_factor = RNG.uniform(0.5, 1.0, size=len(idx))
    ghi = ghi_clear * cloud_factor

    # Relative humidity
    rh = np.clip(60 + 20 * np.sin(2 * np.pi * (hour - 4) / 24) + RNG.normal(0, 10, len(idx)), 15, 100)

    # Wind speed (m/s)
    wind = np.clip(RNG.normal(4.5, 1.8, len(idx)), 0.1, None)

    # Precipitation (mm/h) using a simple stochastic model
    rain_prob = 0.15
    rain = RNG.binomial(1, rain_prob, len(idx)) * np.clip(RNG.gamma(2, 0.8, len(idx)), 0, 15)

    df = pd.DataFrame({
        "timestamp": idx,
        "drybulb_C": temp_c,
        "ghi_Wm2": ghi,
        "rh_pct": rh,
        "wind_speed_ms": wind,
        "precip_mm": rain,
    })
    return df


# -------------------------
# Future climate projections (fabricated from scenario deltas)
# -------------------------

def synthesize_future_climate_baseline(tmy_df: pd.DataFrame, scenario: str = "SSP2-4.5", horizon_year: int = 2050) -> pd.DataFrame:
    # Scenario deltas (very simplified)
    scenario_temp_delta = {
        "SSP1-2.6": 1.2,
        "SSP2-4.5": 2.0,
        "SSP3-7.0": 3.5,
        "SSP5-8.5": 4.5,
    }.get(scenario, 2.0)

    delta_temp = scenario_temp_delta

    # Humidity increases slightly with warming
    rh_multiplier = 1.03 if delta_temp >= 2 else 1.02

    # Wind changes small
    wind_multiplier = 0.98

    # Precipitation intensity increases modestly
    precip_multiplier = 1.1 if delta_temp >= 2 else 1.05

    # Solar: minor change assumed
    ghi_multiplier = 1.00

    proj = tmy_df.copy()
    proj["drybulb_C"] = proj["drybulb_C"] + delta_temp
    proj["rh_pct"] = np.clip(proj["rh_pct"] * rh_multiplier, 0, 100)
    proj["wind_speed_ms"] = np.clip(proj["wind_speed_ms"] * wind_multiplier, 0.05, None)
    proj["precip_mm"] = proj["precip_mm"] * precip_multiplier
    proj["ghi_Wm2"] = proj["ghi_Wm2"] * ghi_multiplier
    proj["scenario"] = scenario
    proj["horizon_year"] = horizon_year
    return proj


# -------------------------
# Economic & Market Data (fabricated)
# -------------------------

def synthesize_energy_prices(start_year: int = 2018, years: int = 10, timezone: Optional[str] = None) -> pd.DataFrame:
    hours = years * 365 * 24
    idx = pd.date_range(f"{start_year}-01-01", periods=hours, freq="H", tz=timezone)

    # Base tariffs ($/kWh) with TOU tiers
    base_price = 0.14
    tou_peak_add = 0.10  # added during 16:00-21:00 weekdays

    weekday = idx.weekday
    hour = idx.hour
    is_peak = ((weekday < 5) & (hour >= 16) & (hour < 21)).astype(int)

    stochastic = RNG.normal(0, 0.01, len(idx))
    price = np.clip(base_price + is_peak * tou_peak_add + stochastic, 0.05, 0.60)

    # Demand charge proxy ($/kW-month) sampled monthly
    months = pd.period_range(start=idx[0], end=idx[-1], freq='M')
    demand_charge = pd.Series(RNG.normal(18, 3, len(months)), index=months)

    df = pd.DataFrame({
        "timestamp": idx,
        "price_per_kwh_usd": price,
        "is_peak": is_peak,
    })

    # Map demand charge to each hour
    df["month"] = df["timestamp"].dt.to_period('M')
    df = df.merge(demand_charge.rename("demand_charge_usd_per_kw_month").to_frame(), left_on="month", right_index=True, how="left")
    df.drop(columns=["month"], inplace=True)

    return df


def synthesize_material_costs() -> pd.DataFrame:
    items = [
        ("insulation", "per_m2", 18, 45),
        ("window_double_lowE", "per_m2", 150, 320),
        ("heat_pump_air_source", "per_ton", 2500, 4500),
        ("boiler_condensing", "per_kW", 120, 220),
        ("pv_module", "per_Wdc", 0.6, 1.1),
        ("inverter", "per_Wac", 0.2, 0.4),
        ("battery", "per_kWh", 250, 500),
        ("lighting_LED", "per_fixture", 20, 80),
        ("controls_BMS", "per_m2", 10, 30),
    ]
    rows = []
    for name, unit, low, high in items:
        cost = RNG.uniform(low, high)
        rows.append({"item": name, "unit": unit, "cost_usd": round(float(cost), 2), "p10": low, "p90": high})
    return pd.DataFrame(rows)


def synthesize_labor_costs() -> pd.DataFrame:
    trades = [
        ("HVAC_installer", 65, 110),
        ("electrician", 70, 120),
        ("glazier", 60, 100),
        ("insulation_tech", 45, 85),
        ("roofer", 50, 95),
        ("general_contractor", 80, 150),
    ]
    rows = []
    for trade, low, high in trades:
        rate = RNG.uniform(low, high)
        rows.append({"trade": trade, "hourly_rate_usd": round(float(rate), 2), "p10": low, "p90": high})
    return pd.DataFrame(rows)


def synthesize_financial_params() -> pd.DataFrame:
    params = [
        ("discount_rate_real", 0.03, 0.07),
        ("inflation_rate", 0.02, 0.04),
        ("loan_interest_rate", 0.04, 0.08),
        ("analysis_period_years", 20, 30),
    ]
    rows = []
    for key, low, high in params:
        if "years" in key:
            value = int(RNG.uniform(low, high))
        else:
            value = round(float(RNG.uniform(low, high)), 4)
        rows.append({"parameter": key, "value": value, "low": low, "high": high})
    # Incentives catalog (fabricated)
    incentives = [
        {"program": "Green Retrofit Rebate", "type": "rebate", "measure": "heat_pump_air_source", "value_usd": 1500},
        {"program": "Efficient Windows Credit", "type": "tax_credit", "measure": "window_double_lowE", "value_usd": 500},
        {"program": "Demand Response Enrollment", "type": "annual_payment", "measure": "controls_BMS", "value_usd": 200},
    ]
    df = pd.DataFrame(rows)
    df2 = pd.DataFrame(incentives)
    return df, df2


# -------------------------
# Geospatial & Regulatory (fabricated)
# -------------------------

def synthesize_geospatial(location: Dict) -> Dict:
    return {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "altitude_m": location.get("altitude_m", 0.0),
        "urban_context": location.get("urban_context", {}),
        "terrain_roughness_length_m": 0.8,  # urban
        "sky_view_factor": max(0.2, 1 - location.get("urban_context", {}).get("building_density", 0.5) - 0.15),
    }


def synthesize_carbon_intensity(timezone: Optional[str] = None, start_year: int = 2022, years: int = 2) -> pd.DataFrame:
    hours = years * 365 * 24
    idx = pd.date_range(f"{start_year}-01-01", periods=hours, freq="H", tz=timezone)

    # Diurnal and seasonal patterns + randomness (kgCO2e/kWh)
    diurnal = 0.10 + 0.04 * np.sin(2 * np.pi * (idx.hour - 6) / 24)
    seasonal = 0.02 * np.sin(2 * np.pi * (idx.dayofyear - 81) / 365)
    random_component = RNG.normal(0, 0.01, len(idx))

    intensity = np.clip(diurnal + seasonal + random_component, 0.02, 0.40)

    return pd.DataFrame({
        "timestamp": idx,
        "kgco2e_per_kwh": intensity,
    })


def synthesize_regulatory(location: Dict) -> Dict:
    city = location.get("city", "Unknown")
    requirements = {
        "energy_code": "IECC 2021 baseline with local amendments",
        "envelope_U_values_Wm2K": {
            "wall": 0.35,
            "roof": 0.18,
            "window": 1.6,
        },
        "lighting_power_density_Wm2": 8.5,
        "ventilation_standards": "ASHRAE 62.1-2019",
        "emissions_policy": {
            "name": "Local Law 97-like",
            "threshold_kgco2e_m2_yr": 70,
            "phasing": [
                {"start_year": 2024, "threshold": 80},
                {"start_year": 2030, "threshold": 70},
                {"start_year": 2035, "threshold": 55}
            ]
        }
    }
    return {"jurisdiction": f"{city}", "requirements": requirements}


# -------------------------
# Orchestration
# -------------------------

def write_dataframe(df: pd.DataFrame, path_csv: str, path_parquet: Optional[str] = None):
    df.to_csv(path_csv, index=False)
    if path_parquet:
        try:
            df.to_parquet(path_parquet, index=False)
        except Exception:
            pass


# -------------------------
# Optional downloads (best-effort, no API key)
# -------------------------

def try_download_open_meteo_era5_hourly(latitude: float, longitude: float, timezone: Optional[str], year: int, out_csv: str) -> bool:
    if requests is None:
        return False
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    base_url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join([
            "temperature_2m",
            "relative_humidity_2m",
            "shortwave_radiation",
            "precipitation",
            "wind_speed_10m",
            "cloudcover"
        ]),
        "timezone": timezone or "UTC",
    }
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "hourly" not in data or "time" not in data["hourly"]:
            return False
        hourly = data["hourly"]
        df = pd.DataFrame(hourly)
        # Rename and align to our schema-ish
        rename_map = {
            "time": "timestamp",
            "temperature_2m": "drybulb_C",
            "shortwave_radiation": "ghi_Wm2",
            "relative_humidity_2m": "rh_pct",
            "wind_speed_10m": "wind_speed_ms",
        }
        df.rename(columns=rename_map, inplace=True)
        df.to_csv(out_csv, index=False)
        return True
    except Exception:
        return False


def try_download_open_meteo_cmip6_monthly(latitude: float, longitude: float, scenario: str, start_year: int, end_year: int, out_csv: str) -> bool:
    if requests is None:
        return False
    base_url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_year": start_year,
        "end_year": end_year,
        "scenario": scenario.lower(),  # e.g., ssp245, ssp370, ssp585
        "monthly": ",".join([
            "temperature_2m_mean",
            "precipitation_sum",
            "shortwave_radiation_sum",
            "wind_speed_10m_mean"
        ]),
        # Optionally, you can specify models, but default ensemble is fine
    }
    try:
        resp = requests.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        monthly = data.get("monthly") or data.get("data")
        if not monthly or "time" not in monthly:
            return False
        df = pd.DataFrame(monthly)
        df.to_csv(out_csv, index=False)
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Generate ecosystem datasets (fabricated)")
    parser.add_argument("--output", type=str, default=OUTPUT_ROOT)
    parser.add_argument("--city", type=str, default=DEFAULT_LOCATION["city"]) 
    parser.add_argument("--latitude", type=float, default=DEFAULT_LOCATION["latitude"]) 
    parser.add_argument("--longitude", type=float, default=DEFAULT_LOCATION["longitude"]) 
    parser.add_argument("--altitude", type=float, default=DEFAULT_LOCATION["altitude_m"]) 
    parser.add_argument("--timezone", type=str, default=DEFAULT_LOCATION["timezone"]) 
    parser.add_argument("--download", action="store_true", help="Attempt to download open datasets (ERA5, CMIP6)")
    args = parser.parse_args()

    location = {
        "city": args.city,
        "latitude": args.latitude,
        "longitude": args.longitude,
        "altitude_m": args.altitude,
        "timezone": args.timezone,
        "urban_context": DEFAULT_LOCATION["urban_context"],
    }

    output_root = args.output
    ensure_dirs(output_root)

    # Weather TMY-like
    tmy = synthesize_tmy_hourly(location, year=2022)
    write_dataframe(tmy, os.path.join(output_root, "weather", "tmy_hourly.csv"))

    # Optional: Try downloading a reanalysis weather year (ERA5 via Open-Meteo)
    if args.download:
        era5_out = os.path.join(output_root, "weather", "era5_hourly_2022.csv")
        try_download_open_meteo_era5_hourly(location["latitude"], location["longitude"], location.get("timezone"), 2022, era5_out)

    # Future projections for several SSPs
    projections = []
    for scenario in ["SSP1-2.6", "SSP2-4.5", "SSP3-7.0", "SSP5-8.5"]:
        proj = synthesize_future_climate_baseline(tmy, scenario=scenario, horizon_year=2050)
        projections.append(proj)
    proj_df = pd.concat(projections, ignore_index=True)
    write_dataframe(proj_df, os.path.join(output_root, "climate", "future_climate_projections.csv"))

    # Optional: Download CMIP6 monthly projections (Open-Meteo climate API)
    if args.download:
        cmip6_out = os.path.join(output_root, "climate", "cmip6_monthly_ssp245_2031_2060.csv")
        try_download_open_meteo_cmip6_monthly(location["latitude"], location["longitude"], "ssp245", 2031, 2060, cmip6_out)

    # Economic data
    prices = synthesize_energy_prices(start_year=2018, years=10, timezone=args.timezone)
    write_dataframe(prices, os.path.join(output_root, "economics", "energy_prices_hourly.csv"))

    materials = synthesize_material_costs()
    materials.to_csv(os.path.join(output_root, "economics", "material_costs_catalog.csv"), index=False)

    labor = synthesize_labor_costs()
    labor.to_csv(os.path.join(output_root, "economics", "labor_costs_catalog.csv"), index=False)

    financial_params, incentives = synthesize_financial_params()
    financial_params.to_csv(os.path.join(output_root, "economics", "financial_parameters.csv"), index=False)
    incentives.to_csv(os.path.join(output_root, "economics", "incentives_catalog.csv"), index=False)

    # Geospatial
    geospatial = synthesize_geospatial(location)
    with open(os.path.join(output_root, "geospatial", "geospatial.json"), "w") as f:
        json.dump(geospatial, f, indent=2)

    # Carbon intensity (hourly)
    carbon = synthesize_carbon_intensity(timezone=args.timezone, start_year=2022, years=2)
    write_dataframe(carbon, os.path.join(output_root, "carbon", "grid_carbon_intensity_hourly.csv"))

    # Regulatory
    regulatory = synthesize_regulatory(location)
    with open(os.path.join(output_root, "regulatory", "regulatory_requirements.json"), "w") as f:
        json.dump(regulatory, f, indent=2)

    print(f"Data generated under: {output_root}")


if __name__ == "__main__":
    main()
