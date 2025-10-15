# Multi-Fidelity SOFC Dataset Summary Report

**Generated on:** 2025-10-15 19:12:28

## Dataset Overview

This dataset contains comprehensive multi-scale parameters for SOFC digital twin modeling.

### Fidelity Levels

- **LF Fidelity**: 1000 samples, 42 parameters
- **MF Fidelity**: 500 samples, 156 parameters
- **HF Fidelity**: 200 samples, 156 parameters

### Additional Data

- **Microstructural Samples**: 200
- **Transient Profiles**: 300

### Parameter Categories

1. **System Level**: Operating conditions (fuel utilization, temperature, pressure, etc.)
2. **Cell/Stack Level**: Geometric parameters (thicknesses, areas, channel dimensions)
3. **Material Level**: Properties for all SOFC components (Ni-YSZ, LSCF, YSZ, Crofer 22APU)
4. **Microstructural Level**: High-fidelity experimental data (phase fractions, connectivity)

### Files Generated

- `sofc_parameters_combined.csv`
- `anode_parameters_hf.png`
- `sofc_transient_profiles.json`
- `sofc_parameters_mf_fidelity.csv`
- `sofc_parameters_hf_fidelity.csv`
- `electrolyte_parameters_hf.png`
- `system_parameters_hf.png`
- `system_parameters_lf.png`
- `interconnect_parameters_hf.png`
- `correlation_matrix_hf.png`
- `cathode_parameters_mf.png`
- `interconnect_parameters_mf.png`
- `system_parameters_mf.png`
- `anode_parameters_mf.png`
- `cathode_parameters_hf.png`
- `dataset_summary_report.md`
- `electrolyte_parameters_mf.png`
- `dataset_metadata.json`
- `sofc_microstructural_data.json`
- `correlation_matrix_mf.png`
- `sofc_parameters_lf_fidelity.csv`
- `correlation_matrix_lf.png`

### Usage Instructions

1. Load the appropriate fidelity level dataset based on your modeling needs
2. Use the combined dataset for multi-fidelity model training
3. Incorporate transient profiles for degradation studies
4. Utilize microstructural data for high-fidelity physics-based modeling
