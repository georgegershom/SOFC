from __future__ import annotations

import argparse
import json
import os
from typing import Optional
import pandas as pd

from .utils import TimeConfig, seeded_random_state
from .weather import get_weather
from .pipeline import generate_all_streams, build_catalog, write_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IoT building dataset for digital twin retrofit studies")
    parser.add_argument("--start", type=str, default="2024-01-01T00:00:00Z", help="Start timestamp (UTC, ISO8601)")
    parser.add_argument("--end", type=str, default="2025-01-01T00:00:00Z", help="End timestamp (UTC, ISO8601, exclusive)")
    parser.add_argument("--freq", type=str, default="15min", help="Sampling frequency, e.g., 15min or 1H")
    parser.add_argument("--lat", type=float, default=40.7128, help="Latitude for weather (default NYC)")
    parser.add_argument("--lon", type=float, default=-74.0060, help="Longitude for weather (default NYC)")
    parser.add_argument("--output", type=str, default="generated_dataset", help="Output directory")
    parser.add_argument("--zones", type=int, default=5, help="Number of thermal zones")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-download-weather", action="store_true", help="Disable weather download; use synthetic weather")
    parser.add_argument("--building-name", type=str, default="Demo Building", help="Building name for catalog")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    time_cfg = TimeConfig(
        start=pd.to_datetime(args.start),
        end=pd.to_datetime(args.end),
        freq=args.freq,
    )

    rng = seeded_random_state(args.seed)

    weather = get_weather(
        time_cfg=time_cfg,
        latitude=args.lat,
        longitude=args.lon,
        allow_download=not args.no_download_weather,
        rng=rng,
    )

    data = generate_all_streams(
        time_cfg=time_cfg,
        weather=weather,
        num_zones=args.zones,
        rng=rng,
    )

    catalog = build_catalog(
        building_name=args.building_name,
        time_cfg=time_cfg,
        num_zones=args.zones,
    )

    write_outputs(output_dir=args.output, weather=weather, data=data, catalog=catalog)

    print(json.dumps({
        "output_dir": os.path.abspath(args.output),
        "rows": len(time_cfg.index),
        "zones": args.zones,
    }, indent=2))


if __name__ == "__main__":
    main()
