# Ecosystem Dataset

This directory contains contextual and external datasets used by the Dynamic Digital Twin Framework for multi-objective building retrofit optimization.

Subfolders:
- `weather`: Historical baseline weather (TMY-like) with irradiance and meteorological fields
- `climate_projections`: Future climate scenario adjustments (e.g., SSP2-4.5, SSP5-8.5)
- `energy_prices`: Electricity and gas tariffs, TOU schedules, demand charges, hourly price series
- `materials`: Costs for retrofit materials and technologies (low/typical/high)
- `labor`: Labor cost library by trade and skill level
- `financial`: Discount/inflation/WACC parameters and incentives
- `geospatial`: Location metadata and hourly urban shading factors
- `carbon`: Hourly grid carbon intensity and decarbonization scenarios
- `regulatory`: Building codes and standards references (e.g., NYC LL97 style limits)

Run the generator to populate all folders:

```bash
python scripts/generate_ecosystem_dataset.py --lat 40.7128 --lon -74.0060 --alt 10 \
  --timezone America/New_York --year 2022 --out data/ecosystem
```
