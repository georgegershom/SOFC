from __future__ import annotations
import argparse
from dataclasses import dataclass
import pandas as pd
import numpy as np
from .timeindex import build_time_index
from .weather import get_weather
from .occupancy import generate_occupancy, OccupancyConfig
from .hvac import simulate_hvac, HVACConfig, ScheduleConfig
from .ieq import simulate_ieq, IEQConfig
from .windows_blinds import simulate_facade, FacadeConfig
from .energy import compute_energy
from .metadata import build_metadata


@dataclass
class GeneratorConfig:
    year: int = 2024
    tz: str = "UTC"
    lat: float = 40.7128
    lon: float = -74.0060
    interval: str = "15min"
    num_zones: int = 5
    seed: int = 42
    floor_area_m2: float = 15000.0


def generate_dataset(cfg: GeneratorConfig) -> dict:
    index = build_time_index(cfg.year, cfg.tz, cfg.interval)

    weather = get_weather(index, cfg.lat, cfg.lon, cfg.tz, seed=cfg.seed)

    occ_cfg = OccupancyConfig(num_zones=cfg.num_zones, seed=cfg.seed)
    occupancy = generate_occupancy(index, occ_cfg)

    hvac_cfg = HVACConfig(num_zones=cfg.num_zones, seed=cfg.seed)
    sched_cfg = ScheduleConfig()
    hvac_df = simulate_hvac(index, weather, occupancy, hvac_cfg, sched_cfg)

    ieq_cfg = IEQConfig(num_zones=cfg.num_zones, seed=cfg.seed)
    ieq_df = simulate_ieq(index, weather, occupancy, hvac_df, ieq_cfg)

    fac_cfg = FacadeConfig(num_zones=cfg.num_zones, seed=cfg.seed)
    facade_df = simulate_facade(index, weather, occupancy, fac_cfg)

    energy_df = compute_energy(index, occupancy, hvac_df, cfg.floor_area_m2)

    # Assemble
    df = pd.concat([weather, occupancy, hvac_df, ieq_df, facade_df, energy_df], axis=1)

    meta = build_metadata(df.columns.tolist())

    return {"data": df, "metadata": meta}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic building IoT time-series dataset")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--tz", type=str, default="UTC")
    parser.add_argument("--lat", type=float, default=40.7128)
    parser.add_argument("--lon", type=float, default=-74.0060)
    parser.add_argument("--interval", type=str, choices=["15min", "1h"], default="15min")
    parser.add_argument("--zones", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data")
    parser.add_argument("--prefix", type=str, default="building_iot")

    args = parser.parse_args()
    cfg = GeneratorConfig(year=args.year, tz=args.tz, lat=args.lat, lon=args.lon, interval=args.interval, num_zones=args.zones, seed=args.seed)

    result = generate_dataset(cfg)
    df: pd.DataFrame = result["data"]
    meta = result["metadata"]

    outdir = args.outdir
    import os
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, f"{args.prefix}_{cfg.year}_{cfg.interval}.csv")
    parquet_path = os.path.join(outdir, f"{args.prefix}_{cfg.year}_{cfg.interval}.parquet")
    meta_path = os.path.join(outdir, f"{args.prefix}_{cfg.year}_metadata.json")

    # Save
    df.to_csv(csv_path, index_label="timestamp")
    try:
        df.to_parquet(parquet_path, index=True)
    except Exception:
        pass

    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {parquet_path}")
    print(f"Wrote: {meta_path}")


if __name__ == "__main__":
    main()
