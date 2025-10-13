SOFC Adaptive-Scale Digital Twin Datasets (Synthetic)

Usage:

```bash
python generate_sofc_dataset.py generate --dataset all --n-samples 8 --grid 32x32x8 --duration-hours 2 --sample-rate-hz 1
```

Outputs are written to `data/sofc_datasets/`:
- `dataset1_hf/dataset1.h5` – high-fidelity multiphysics 3D fields per sample.
- `dataset2_exp/` – experimental-like timeseries, EIS CSVs, thermal images, strain gauges, AE events.
- `dataset3_rt/` – realtime stream CSV.

Notes:
- The physics is synthetic but structured to emulate electro-thermal-structural coupling and fracture metrics.
- HDF5 contains inputs and outputs with units in attributes.
