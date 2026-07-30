# README for generated dataset
This folder (dataset_generated/) contains a synthetic, literature-compiled example dataset for an SOFC anode-electrolyte half-cell delamination study.

Files included:
- sample_reference.csv : fabrication and reference parameters
- env_logs_testA.csv : example environmental log (Test A thermal cycle)
- dic_metadata.csv : DIC acquisition parameters
- dic_frame00001.csv : example DIC output for frame 1 (grid values)
- ae_hits.csv : example acoustic emission hits table
- fibsem_metadata.csv : FIB-SEM tomography metadata
- microstructure_stats.csv : microstructural summary measures
- material_properties_thermal.csv : thermal properties vs temperature (synthetic)
- material_properties_elastic.csv : Young's modulus vs T (synthetic)
- fracture_params.csv : fracture and creep parameters
- postmortem_crack_measurements.csv : measured crack metrics
- attention_papers_since2020.csv : selected attention mechanism papers since 2020

Notes:
- All numerical data in these CSVs are synthetic or literature-compiled example values intended for method development, simulation input, and demonstration. They are NOT raw experimental measurements from a specific lab unless explicitly stated.
- Sources and suggested primary literature: Heenan et al. (2018, RSC), Materials (2024) articles (see repository or search results). For exact experimental datasets (FIB-SEM stacks, TIFF DIC images, AE raw waveforms), contact corresponding authors or check Zenodo/Figshare links in the cited papers.

If you want me to (pick one):
1) create a ZIP containing all CSVs for direct download in this repo, or
2) also generate SVG figure files (temperature vs time, example strain map, AE timeline) and commit them now.

I will now add a set of simple SVG figures (temperature_vs_time.svg, strain_map_example.svg, ae_timeline.svg) to the same folder.
