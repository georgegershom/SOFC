Phase-Field Fracture Delamination Dataset Package
=================================================

Topic:
  Phase-Field Fracture Modeling of Delamination in Electrolyte-Electrode
  Interfaces with nanoscale MIEC interlayers.

Important:
  Data are fabricated/synthesized from the calibrated ranges provided
  by the request and intended for simulation workflow prototyping.

CSV files:
  - assumptions_calibrated_data_inventory.csv
  - calibrated_interface_fracture_properties.csv
  - gdc_chemical_expansion_22x4_dataset.csv
  - gdc_nonstoichiometry_calibrated_grid.csv
  - lscf_nonstoichiometry_calibrated_grid.csv
  - phase_field_qa_verification_plan.csv
  - phase_field_simulation_samples.csv

Figure files:
  - figure_01_interface_fracture_energy_ranges.png
  - figure_02_nonstoichiometry_heatmaps.png
  - figure_03_gdc_chemical_expansion_curves.png
  - figure_04_simulation_sampling_coverage.png

ZIP bundle:
  - phase_field_delamination_csv_bundle.zip

Regeneration:
  python3 build_phase_field_dataset.py
