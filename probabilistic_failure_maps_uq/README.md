# Probabilistic Failure Maps (Synthetic UQ Package)

This package contains fabricated, reproducible uncertainty-quantification assets for:
**"Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs."**

## Important note

All datasets in this folder are **synthetic surrogates** for sensitivity analysis and method development.
They are not calibrated to proprietary or missing experimental campaigns.

## Folder structure

- `csv/`:
  - `stochastic_inputs_rho0.00.csv`
  - `stochastic_inputs_rho0.50.csv`
  - `stochastic_inputs_HT_uncorrelated.csv`
  - `21_missing_data_assumption_flags.csv`
  - `04_uncertainty_material_properties.csv`
  - `16_micro_cantilever_fracture_data.csv`
  - `13_LSCF_ferroelastic_stress_strain.csv`
  - `failure_probability_curves.csv`
  - `epistemic_gap_summary.csv`
  - `interfacial_failure_correlation_matrix_surrogates.csv`
  - `dataset_manifest.csv`
- `figures/`:
  - `fig01_fragility_cone_of_ignorance.png`
  - `fig02_cdf_critical_energy_release_rate.png`
  - `fig03_interface_toughness_correlation_worlds.png`
- `downloads/`:
  - `probabilistic_failure_maps_csv_bundle.zip` (all CSVs packaged for download)
- `scripts/`:
  - `generate_probabilistic_failure_assets.py`
  - `quantify_epistemic_gap.py`

## How to regenerate

From repository root:

```bash
python3 probabilistic_failure_maps_uq/scripts/generate_probabilistic_failure_assets.py
python3 probabilistic_failure_maps_uq/scripts/quantify_epistemic_gap.py
```

## Reference output metric at J = 3.0 J/m^2

- `P_fail` (rho=0.00): ~0.5892
- `P_fail` (rho=0.50): ~0.5908
- `P_fail` (High-T surrogate): ~0.9562
- Epistemic gap (rho scenarios): ~0.16 percentage points
