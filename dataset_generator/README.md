# HPRC Fire Dataset Generator (Synthetic)

This generator fabricates a comprehensive synthetic experimental dataset for High-Performance Rubberized Concrete (HPRC) under ambient and high-temperature exposure.

It includes:
- Ambient tests at 7, 28, 56 days: compressive strength, splitting tensile or flexural, static modulus (E), density, UPV
- High-temperature exposure: 23, 200, 400, 600, 800 °C; furnace vs quench cooling; standardized heating rate; soak time
- Residual property tests: mass loss, UPV, residual strengths, residual stress–strain curves
- In-situ high-temp tests: transient thermal strain, in-situ strength & E, thermal expansion (CTE)
- Spalling behaviour: depth/pattern metrics; optional pore pressure traces

Outputs (under `generated/`):
- CSVs/Parquet for tabular data
- PNGs for curves, photos, and visualizations
- YAML metadata describing protocol and parameters

This is synthetic data generated from physically-informed stochastic models; not a real experiment.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r dataset_generator/requirements.txt
python dataset_generator/generate_dataset.py --out generated --seed 42
```

Artifacts will be written to `generated/` and a `hprc_dataset.zip` will be created.
