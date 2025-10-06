# Sintering Residuals Synthetic Dataset

This folder contains physics-inspired synthetic data for ANN/PINN training.

## Splits
- train: features.csv, labels.csv, optional fields/ subset (npz per sample)
- test: features.csv, labels.csv
- val: metadata.csv, ground_truth.csv, DIC/XRD-style maps (npy)

## Units
- Temperatures: Celsius
- Stresses: MPa (fields, summaries), Pa internally
- Strain: dimensionless
- Thickness: meters
- KIC: MPa*sqrt(m)

## Labels
- stress_hotspot_fraction: fraction of pixels above mean + 1.5 std (or surrogate)
- crack_initiation_risk: sigmoid surrogate of Griffith criterion
- delamination_probability: sigmoid surrogate of energy release vs Gc

## Validation data
- DIC: downsampled, noisy surface strain maps
- XRD: noisy residual stress maps (quantized)

Generated with scripts/generate_sintering_dataset.py
