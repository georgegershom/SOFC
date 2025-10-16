from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from iot.utils import TimeConfig, ensure_dir
from iot.weather import WeatherConfig, generate_weather
from iot.occupancy import OccupancyConfig, generate_occupancy
from iot.energy import EnergyConfig, generate_energy
from iot.ieq import IEQConfig, generate_ieq
from iot.hvac import HVACConfig, generate_hvac
from iot.windows_blinds import WindowsBlindsConfig, generate_windows_blinds


def write_outputs(out_dir: str, site_name: str, dfs: dict[str, pd.DataFrame], write_parquet: bool) -> None:
    ensure_dir(out_dir)
    for name, df in dfs.items():
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index_label="timestamp")
        if write_parquet:
            pq_path = os.path.join(out_dir, f"{name}.parquet")
            df.to_parquet(pq_path)


def write_manifest(out_dir: str, site_name: str, tc: TimeConfig, seed: Optional[int], configs: dict) -> None:
    manifest = {
        "site_name": site_name,
        "time": {
            "start": str(tc.start),
            "end": str(tc.end),
            "freq": tc.freq,
            "tz": tc.tz,
        },
        "seed": seed,
        "streams": list(configs.keys()),
        "configs": {k: asdict(v) for k, v in configs.items()},
        "schema": {
            "weather": [
                "ambient_temp_C", "relative_humidity_pct", "wind_speed_mps", "wind_dir_deg", "rain_mm", "solar_irradiance_Wm2"
            ],
            "occupancy": ["occupant_count", "wifi_clients_agg"],
            "energy": [
                "electric_kw_whole", "gas_kw_whole", "water_m3ph", "district_cooling_kw", "hvac_kw", "lighting_kw", "plugs_kw"
            ],
            "ieq": "various per-zone air_temp_C_zN, rh_pct_zN, plus co2_ppm, pm25_ugm3, pm10_ugm3, tvoc_ppb, illuminance_lux, noise_db",
            "hvac": [
                "supply_air_temp_C", "return_air_temp_C", "fan_speed_frac", "oa_damper_pos_frac", "valve_pos_frac", "chiller_status", "boiler_status", "heat_setpoint_C", "cool_setpoint_C"
            ],
            "windows_blinds": ["window_open", "blinds_closed"],
        },
        "notes": "Synthetic dataset integrating IoT streams for digital twin and retrofit optimization research."
    }
    with open(os.path.join(out_dir, "manifest.yaml"), "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)


def main():
    p = argparse.ArgumentParser(description="Generate synthetic IoT & real-time monitoring dataset")
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True)
    p.add_argument("--tz", type=str, default="UTC")
    p.add_argument("--freq", type=str, default="15min")
    p.add_argument("--site-name", type=str, default="Building-A")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--parquet", action="store_true", help="Write Parquet alongside CSV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lat", type=float, default=40.71)
    p.add_argument("--lon", type=float, default=-74.01)
    p.add_argument("--no-download-weather", action="store_true")
    args = p.parse_args()

    tc = TimeConfig(start=pd.Timestamp(args.start), end=pd.Timestamp(args.end), freq=args.freq, tz=args.tz)

    weather_cfg = WeatherConfig(latitude=args.lat, longitude=args.lon, use_download=not args.no_download_weather)
    occ_cfg = OccupancyConfig()
    energy_cfg = EnergyConfig()
    ieq_cfg = IEQConfig()
    hvac_cfg = HVACConfig()
    win_cfg = WindowsBlindsConfig()

    # Generate streams
    weather = generate_weather(tc, weather_cfg, seed=args.seed)
    occupancy = generate_occupancy(tc, occ_cfg, seed=args.seed + 1 if args.seed is not None else None)
    # Occupancy factor for other models
    occ_factor = np.clip(occupancy["occupant_count"].values / (occupancy["occupant_count"].max() + 1e-9), 0.0, 1.0)

    energy = generate_energy(tc, energy_cfg, weather, occ_factor, seed=args.seed + 2 if args.seed is not None else None)
    ieq = generate_ieq(tc, ieq_cfg, weather, occupancy, seed=args.seed + 3 if args.seed is not None else None)
    hvac = generate_hvac(tc, hvac_cfg, weather, occupancy, seed=args.seed + 4 if args.seed is not None else None)
    windows_blinds = generate_windows_blinds(tc, win_cfg, weather, occupancy, seed=args.seed + 5 if args.seed is not None else None)

    dfs = {
        "weather": weather,
        "occupancy": occupancy,
        "energy": energy,
        "ieq": ieq,
        "hvac": hvac,
        "windows_blinds": windows_blinds,
    }

    ensure_dir(args.out_dir)
    write_outputs(args.out_dir, args.site_name, dfs, args.parquet)

    configs = {
        "weather": weather_cfg,
        "occupancy": occ_cfg,
        "energy": energy_cfg,
        "ieq": ieq_cfg,
        "hvac": hvac_cfg,
        "windows_blinds": win_cfg,
    }
    write_manifest(args.out_dir, args.site_name, tc, args.seed, configs)

    # Zip folder for convenience
    try:
        import shutil
        zip_path = os.path.abspath(args.out_dir.rstrip('/'))
        shutil.make_archive(zip_path, 'zip', args.out_dir)
    except Exception:
        pass

    print(f"Generated dataset at {args.out_dir}")


if __name__ == "__main__":
    main()
