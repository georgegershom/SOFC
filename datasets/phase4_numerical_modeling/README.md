# Phase 4: Numerical Modeling Dataset (HPRC)

This dataset contains fabricated, physics-informed data for High-Performance Rubberized Concrete (HPRC) to support development and validation of thermo-mechanical models under fire.

Contents:
- `properties/` temperature-dependent material and poro-mechanical properties
- `validation/` fire test-like time histories for temperature, deformation/strain, and spalling
- `generator/` scripts used to generate the dataset

Provenance: Synthetic data generated via `generator/generate_dataset.py` using seeded randomness and literature-informed trends.
