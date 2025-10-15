
SOFC Multi-Fidelity Input Dataset
=================================

This bundle contains synthetic, design-of-experiments datasets for a Multi-Fidelity Digital Twin of SOFCs. It covers:
- LF (Low Fidelity): System-level operating conditions and transients
- MF (Medium Fidelity): Adds cell/stack geometry and bulk material properties
- HF (High Fidelity): Adds microstructural summaries and synthetic 3D voxel microstructures for anode and cathode

Generation method
-----------------
- Latin Hypercube Sampling (LHS) is used across continuous parameters.
- Fuel and air compositions are sampled from Dirichlet distributions and normalized.
- Microstructures are synthesized by thresholding smoothed Gaussian noise to match target phase fractions, then summarized.

Defaults used (can be changed via CLI):
- LF samples: 2000
- MF samples: 1000
- HF samples: 200
- Microstructure size: 64^3 voxels (voxel_size_um saved in each .npz)

Caution
-------
- This data is synthetic and for research/prototyping only. Ranges are plausible but not tied to any specific system. Validate and tailor before use.

File layout
-----------
- lf.csv, lf.parquet
- mf.csv, mf.parquet
- hf.csv, hf.parquet
- data_dictionary.csv, data_dictionary.json
- microstructures/
  - anode/anode_XXXXX.npz
  - cathode/cathode_XXXXX.npz

