# Multi-Fidelity SOFC Dataset Summary Report
==================================================

## Dataset Overview
- **Total Samples**: 3,000
- **Fidelity Levels**: LF, MF, HF
- **Total Parameters**: 68
- **Sampling Method**: Latin Hypercube Sampling
- **Generation Date**: 2025-10-15T19:18:06.638571

## Parameter Categories
### System Parameters
- **Count**: 8 parameters
- **Parameters**: fuel_utilization, oxidant_utilization, current_density, voltage, temperature, pressure, fuel_flow_rate, air_flow_rate

### Transient Parameters
- **Count**: 4 parameters
- **Parameters**: startup_ramp_rate, shutdown_ramp_rate, load_following_ramp_rate, thermal_cycling_frequency

### Geometry Parameters
- **Count**: 7 parameters
- **Parameters**: cell_active_area, anode_thickness, cathode_thickness, electrolyte_thickness, interconnect_thickness, channel_width, channel_height

### Anode Parameters
- **Count**: 10 parameters
- **Parameters**: anode_porosity, anode_tortuosity, anode_ni_particle_size, anode_ysz_particle_size, anode_tpb_density, anode_ionic_conductivity, anode_electronic_conductivity, anode_youngs_modulus, anode_poisson_ratio, anode_thermal_expansion_coefficient

### Cathode Parameters
- **Count**: 11 parameters
- **Parameters**: cathode_porosity, cathode_tortuosity, cathode_lscf_particle_size, cathode_gdc_particle_size, cathode_tpb_density, cathode_ionic_conductivity, cathode_electronic_conductivity, cathode_youngs_modulus, cathode_poisson_ratio, cathode_thermal_expansion_coefficient, cathode_chemical_expansion_coefficient

### Electrolyte Parameters
- **Count**: 5 parameters
- **Parameters**: electrolyte_ionic_conductivity, electrolyte_youngs_modulus, electrolyte_poisson_ratio, electrolyte_thermal_expansion_coefficient, electrolyte_fracture_toughness

### Interconnect Parameters
- **Count**: 5 parameters
- **Parameters**: interconnect_thermal_expansion_coefficient, interconnect_youngs_modulus, interconnect_poisson_ratio, interconnect_oxide_scale_growth_rate, interconnect_electrical_resistivity

### Microstructural Parameters
- **Count**: 8 parameters
- **Parameters**: phase_fraction_ni, phase_fraction_ysz, phase_fraction_pore, specific_surface_area, connectivity_ni, connectivity_ysz, pore_size_distribution_mean, particle_size_distribution_mean

### Degradation Parameters
- **Count**: 6 parameters
- **Parameters**: ni_agglomeration_rate, ni_oxidation_rate, cathode_poisoning_rate, electrolyte_cracking_rate, thermal_stress_accumulation, redox_cycling_damage

## Statistical Summary by Fidelity Level
### LF Fidelity
- **Samples**: 1,000
- **Parameters**: 68
- **temperature**: 1022.998 ± 28.883 (range: 973.069 - 1072.954)
- **current_density**: 0.550 ± 0.260 (range: 0.100 - 0.999)
- **voltage**: 0.800 ± 0.116 (range: 0.600 - 1.000)
- **fuel_utilization**: 0.750 ± 0.087 (range: 0.600 - 0.900)

### MF Fidelity
- **Samples**: 1,000
- **Parameters**: 68
- **temperature**: 1022.998 ± 28.883 (range: 973.069 - 1072.954)
- **current_density**: 0.550 ± 0.260 (range: 0.100 - 0.999)
- **voltage**: 0.800 ± 0.116 (range: 0.600 - 1.000)
- **fuel_utilization**: 0.750 ± 0.087 (range: 0.600 - 0.900)

### HF Fidelity
- **Samples**: 1,000
- **Parameters**: 68
- **temperature**: 1022.998 ± 28.883 (range: 973.069 - 1072.954)
- **current_density**: 0.550 ± 0.260 (range: 0.100 - 0.999)
- **voltage**: 0.800 ± 0.116 (range: 0.600 - 1.000)
- **fuel_utilization**: 0.750 ± 0.087 (range: 0.600 - 0.900)

## Data Quality Assessment
### LF Fidelity Quality Metrics
- **Missing Values**: 0
- **Duplicate Rows**: 0
- **Memory Usage**: 0.56 MB

### MF Fidelity Quality Metrics
- **Missing Values**: 0
- **Duplicate Rows**: 0
- **Memory Usage**: 0.56 MB

### HF Fidelity Quality Metrics
- **Missing Values**: 0
- **Duplicate Rows**: 0
- **Memory Usage**: 0.56 MB
