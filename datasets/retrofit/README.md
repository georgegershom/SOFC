# Integrated Building Retrofit Dataset

This project fabricates a multi-faceted synthetic dataset for AI- and IoT-driven building retrofit research. It produces CSVs for:

- Buildings (attributes & fabric)
- Outdoor weather (hourly)
- IoT time series (15-min end-use, IEQ, occupancy)
- Energy performance (monthly, pre/post)
- Retrofit measures (dates, CAPEX, cumulative savings)
- LCA (EPD catalog and per-building embodied carbon summary)

## Quick start

```bash
python -m datasets.retrofit.src.build --num-buildings 80 --start 2022-01-01 --end 2022-12-31
```

Outputs will be stored under `datasets/retrofit/data`.

## Notes
- Values are synthetic and intended for methodology development and benchmarking.
- You can adjust parameters in `datasets/retrofit/src` to tailor distributions and correlations.
