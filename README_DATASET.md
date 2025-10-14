# Integrated Building Retrofit Dataset (Synthetic)

This repository includes a synthetic, research-ready dataset integrating IoT sensor streams, static building attributes, energy performance summaries, and lifecycle assessment (LCA) elements to support AI- and IoT-driven retrofit optimization research. Data reflect plausible magnitudes and relationships but are not real measurements.

## Contents
- `data/synthetic/`
  - `buildings/buildings.csv`: Building stock with attributes and envelope performance (U/R-values)
  - `iot/weather_timeseries.csv`: Hourly weather by climate zone
  - `iot/iot_timeseries.csv`: Hourly electricity/gas end-uses and IEQ (CO₂, TVOC, PM2.5, indoor T/RH)
  - `iot/occupancy_timeseries.csv`: Hourly synthetic occupancy counts
  - `energy/energy_timeseries.csv`: Hourly total energy subset by building
  - `energy/energy_monthly.csv`: Monthly aggregates per building
  - `energy/energy_annual.csv`: Annual totals and computed EUI with EU-style ratings
  - `lca/lca_materials.csv`: Envelope material masses, EPD-like IDs, embodied carbon per component
  - `lca/lca_building_totals.csv`: Per-building areas and embodied carbon totals
  - `lca/retrofit_scenarios.csv`: Simple retrofit scenario with predicted savings, EC, capex, payback
  - `dataset_index.json`: Index with paths, counts, and generation metadata
- `scripts/generate_dataset.py`: Generator script (configurable via CLI)
- `scripts/download_external.py`: Helper for fetching external datasets into `data/external`
- `scripts/external_sources.example.json`: Example config listing external sources (replace URLs)

## Quick start
Generate a month of data for 25 buildings:

```bash
python3 scripts/generate_dataset.py \
  --output-root data/synthetic \
  --num-buildings 25 \
  --start-date 2024-01-01T00:00:00 \
  --end-date   2024-01-31T23:00:00 \
  --seed 7
```

Result archive:

```bash
# already created for you
ls -lh data/building-retrofit-dataset.tar.gz
```

## Schema highlights
- **IoT**: electricity/gas end-uses (kWh), IEQ (CO₂ ppm, TVOC ppb, PM2.5 µg/m³, indoor T/RH)
- **Buildings**: climate zone, type/function, style, quality, year, floors, areas/volumes, U/R-values, heating fuel, EU rating approximation
- **Energy performance**: hourly, monthly, annual with computed EUI
- **LCA**: component-level masses and embodied carbon using a small synthetic material library and EPD-like identifiers; retrofit scenario includes added insulation, window U-value, HVAC efficiency improvement, predicted savings, capex, and payback

## External datasets (optional)
To fetch external datasets into `data/external`:

```bash
python3 scripts/download_external.py \
  --config scripts/external_sources.example.json \
  --output-root data/external
```

Replace placeholder URLs with real links and review licenses before use.

## Notes
- Values are generated from simplified physics- and behavior-inspired models for prototyping. Validate against real data for production research.
- Time zone is naive ISO 8601; align to your study region as needed.
