### Integrated Building Retrofit Dataset (Synthetic)

This repository generates a synthetic, research-ready dataset integrating:
- **IoT sensor data** (energy, IEQ, occupancy proxies, outdoor weather)
- **Building attributes** (geometry, construction, materials, envelope U-values)
- **Energy performance** (4-year monthly series with retrofit events and savings)
- **Lifecycle Assessment (LCA)** (material inventory and embodied carbon)

### Quickstart
```bash
python3 scripts/generate_dataset.py \
  --num-buildings 200 \
  --iot-fraction 0.35 \
  --iot-days 90 \
  --iot-interval-minutes 15 \
  --seed 42 \
  --start-date 2024-01-01
```
Output artifacts appear in `data/processed/`:
- `buildings.csv`
- `epd_factors.csv`
- `lca_material_inventory.csv`
- `lca_results.csv`
- `energy_performance_monthly.csv`
- `iot_timeseries.csv.gz`

### Methodological Notes
- The generator aims to reflect realistic relationships (seasonality, schedules, load splits, envelope quality effects) without using any proprietary data.
- LCA values use simplified EPD-like factors and coarse bill-of-quantities.
- Energy ratings (A–G) are mapped from synthetic EUI bands.

### Extending
- Replace materials and EPD factors in `scripts/generate_dataset.py` for a specific region.
- Expand `RETROFIT_SCOPES` and logic to reflect your intervention set.
- Replace synthetic outdoor weather with reanalysis data (e.g., ERA5) for a location.

### License
Synthetic data and code are provided "as is" for research and educational use.
