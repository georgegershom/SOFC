from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from ecosystem.weather import WeatherConfig, build_weather_dataset


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


essential_dirs = [
    "weather",
    "climate_projections",
    "energy_prices",
    "materials",
    "labor",
    "financial",
    "geospatial",
    "carbon",
    "regulatory",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Ecosystem dataset")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--alt", type=float, default=10.0)
    parser.add_argument("--timezone", type=str, required=True)
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--out", type=str, default="data/ecosystem")

    args = parser.parse_args()

    out_root = Path(args.out)
    for d in essential_dirs:
        ensure_dir(out_root / d)

    # 1) Weather baseline (TMY-like)
    weather_cfg = WeatherConfig(
        latitude=args.lat,
        longitude=args.lon,
        altitude_m=args.alt,
        timezone=args.timezone,
        year=args.year,
    )

    weather_df = build_weather_dataset(weather_cfg)

    weather_csv = out_root / "weather" / f"weather_{args.year}_{args.lat:.4f}_{args.lon:.4f}.csv"
    weather_parquet = out_root / "weather" / f"weather_{args.year}_{args.lat:.4f}_{args.lon:.4f}.parquet"

    weather_df.to_csv(weather_csv, index_label="timestamp")
    weather_df.to_parquet(weather_parquet, index=True)

    # Save a small metadata JSON
    (out_root / "_meta.json").write_text(
        json.dumps(
            {
                "coords": {"lat": args.lat, "lon": args.lon, "alt": args.alt},
                "timezone": args.timezone,
                "year": args.year,
                "versions": {
                    "pandas": pd.__version__,
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
