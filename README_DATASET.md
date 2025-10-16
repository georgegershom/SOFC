# Dynamic Digital Twin IoT Dataset Generator

Generates a one-year, 15-minute high-frequency dataset for a commercial building, including energy, IEQ, occupancy, utilization, HVAC operations, windows/blinds, and on-site weather, aligned to the paper topic: "A Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization".

## Features
- Weather ingestion via `meteostat` with fallback to synthetic local weather
- Whole-building + submeter energy (electric, gas, water, district thermal)
- IEQ (temperature, RH, CO2, PM2.5/PM10, TVOC, illuminance, noise)
- Occupancy via schedule + stochastic variation + CO2-derived estimate
- Space utilization (PIR/motion) and Wi-Fi aggregated counts
- HVAC operational points (supply/return temps, dampers, valves, fan speeds, chiller/boiler)
- Setpoints (heating/cooling, ventilation)
- Windows & blinds events with weather- and occupancy-driven behavior
- Outputs CSV + optional Parquet; includes YAML manifest

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/generate_dataset.py --start 2024-01-01 --end 2024-12-31 --tz America/New_York --site-name "HQ-East" --out-dir data/hq_east --freq 15min --parquet
```

## Directory Layout
- `src/generate_dataset.py` — CLI entry point
- `src/iot/` — generation modules by domain
- `data/<site>/` — generated CSV/Parquet and manifest

## Notes
- Weather download attempts use Meteostat with nearest stations; if none, synthetic weather is generated using seasonal and diurnal patterns.
- Randomness uses a seed for reproducibility; override with `--seed`.
