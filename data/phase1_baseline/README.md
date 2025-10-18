
# Phase 1 Baseline Dataset: High-Performance Rubberized Concrete

Scope: Material Characterization & Specimen Preparation
Project theme: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete.

Contents
- materials/
  - cement/: XRF, Bogue composition, properties
  - aggregates/
    - fine/: sieve analysis, properties
    - coarse/: sieve analysis, properties
  - crumb_rubber/ (1-4mm, 4-8mm): PSD, properties
  - water/: pH and impurities
  - chemical_admixture/: superplasticizer data
- characterization/
  - tga/: Synthetic TGA (3 reps) + summary
  - ftir/: Synthetic FTIR (3 reps) + peak assignments
  - sem_images/: SEM-like PGM images (4 per rubber size, pre-heating)
- mixes/: Mix designs (control + 5/10/15/20% rubber by volume of fine aggregate) for both size ranges
- fresh_properties/: Fresh data per batch (slump/flow, air, unit weight, temperature)

Notes
- Data are fabricated yet plausible; values reflect typical trends: increasing rubber -> lower unit weight and slump, higher air.
- Absolute volume method used to estimate constituent masses per m^3.
- TGA shows multi-stage mass loss (volatiles, pyrolysis, char oxidation). FTIR includes typical rubber/additive peaks.
- SEM images are algorithmic textures stored as ASCII PGM (P2), 512x512 px.

How to cite/use
- Synthetic dataset for prototyping. Not from physical experiments. CC-BY 4.0.

Reproducibility
- RNG seed: 42
- Generator script: scripts/generate_phase1_dataset.py
