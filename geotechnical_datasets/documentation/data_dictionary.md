# Data Dictionary - Geotechnical Datasets

## Sandy Soils Basic Properties

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| sample_id | - | Unique identifier | - |
| location | - | Geographic location | - |
| depth_m | m | Sampling depth | 0-50 |
| sand_content_% | % | Percentage of sand particles | 50-100 |
| silt_content_% | % | Percentage of silt particles | 0-50 |
| clay_content_% | % | Percentage of clay particles | 0-20 |
| d10_mm | mm | Effective grain size | 0.01-1.0 |
| d30_mm | mm | Grain size at 30% passing | 0.05-2.0 |
| d50_mm | mm | Median grain size | 0.1-3.0 |
| d60_mm | mm | Grain size at 60% passing | 0.2-5.0 |
| cu_coefficient | - | Uniformity coefficient (d60/d10) | 1-15 |
| cc_coefficient | - | Curvature coefficient | 0.5-3.0 |
| relative_density_% | % | Relative density | 15-100 |
| void_ratio | - | Volume of voids/volume of solids | 0.3-1.0 |
| specific_gravity | - | Specific gravity of solids | 2.6-2.75 |
| moisture_content_% | % | Water content | 0-30 |
| bulk_density_kg/m3 | kg/m³ | Total unit weight | 1600-2200 |
| dry_density_kg/m3 | kg/m³ | Dry unit weight | 1400-2000 |

## Sandy Soils Mechanical Properties

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| friction_angle_deg | degrees | Internal friction angle | 25-45 |
| cohesion_kPa | kPa | Cohesion intercept | 0-5 |
| peak_friction_angle_deg | degrees | Peak friction angle | 30-50 |
| residual_friction_angle_deg | degrees | Residual friction angle | 20-40 |
| dilatancy_angle_deg | degrees | Dilatancy angle | 0-15 |
| elastic_modulus_MPa | MPa | Young's modulus | 10-100 |
| poisson_ratio | - | Poisson's ratio | 0.2-0.45 |
| shear_modulus_MPa | MPa | Shear modulus | 5-50 |
| bulk_modulus_MPa | MPa | Bulk modulus | 10-100 |
| compression_index | - | Compression index | 0.01-0.1 |
| recompression_index | - | Recompression index | 0.001-0.02 |
| hydraulic_conductivity_m/s | m/s | Hydraulic conductivity | 1E-8 to 1E-3 |
| permeability_coefficient_m2 | m² | Intrinsic permeability | 1E-15 to 1E-10 |

## Liquefaction Data

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| test_type | - | Type of liquefaction test | - |
| CSR | - | Cyclic stress ratio | 0.1-0.5 |
| CRR | - | Cyclic resistance ratio | 0.1-0.5 |
| safety_factor | - | Factor of safety against liquefaction | 0.5-3.0 |
| N1_60 | - | Normalized SPT blow count | 5-50 |
| initial_void_ratio | - | Initial void ratio | 0.4-1.0 |
| relative_density_% | % | Relative density | 15-100 |
| fines_content_% | % | Percentage of fines | 0-35 |
| cyclic_stress_ratio | - | Applied cyclic stress ratio | 0.1-0.5 |
| number_of_cycles | - | Number of loading cycles | 1-100 |
| pore_pressure_ratio | - | Excess pore pressure ratio | 0-1.0 |
| excess_pore_pressure_kPa | kPa | Excess pore water pressure | 0-500 |
| liquefaction_potential | - | Qualitative assessment | Very_Low to Very_High |
| peak_ground_acceleration_g | g | PGA in gravity units | 0.1-1.0 |
| earthquake_magnitude | - | Moment magnitude | 5.0-9.0 |
| depth_to_water_table_m | m | Groundwater depth | 0-10 |
| initial_effective_stress_kPa | kPa | Initial effective stress | 50-500 |
| final_effective_stress_kPa | kPa | Final effective stress | 0-500 |

## Clay Basic Properties

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| clay_content_% | % | Percentage of clay particles | 30-100 |
| silt_content_% | % | Percentage of silt particles | 0-50 |
| sand_content_% | % | Percentage of sand particles | 0-20 |
| liquid_limit_% | % | Liquid limit | 20-500 |
| plastic_limit_% | % | Plastic limit | 10-100 |
| plasticity_index | - | Plasticity index (LL-PL) | 10-400 |
| liquidity_index | - | Liquidity index | -0.5-2.0 |
| activity | - | Activity (PI/clay%) | 0.3-3.0 |
| preconsolidation_stress_kPa | kPa | Preconsolidation pressure | 50-1000 |
| OCR | - | Overconsolidation ratio | 1-10 |
| water_content_% | % | Natural water content | 10-300 |
| bulk_density_kg/m3 | kg/m³ | Total unit weight | 1100-2000 |
| dry_density_kg/m3 | kg/m³ | Dry unit weight | 300-1600 |
| specific_gravity | - | Specific gravity | 2.6-2.8 |
| void_ratio | - | Void ratio | 0.5-8.0 |
| porosity_% | % | Porosity | 30-90 |
| degree_saturation_% | % | Degree of saturation | 50-100 |

## Clay Mechanical Properties

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| undrained_shear_strength_kPa | kPa | Undrained shear strength | 10-200 |
| drained_cohesion_kPa | kPa | Drained cohesion | 0-50 |
| drained_friction_angle_deg | degrees | Drained friction angle | 5-35 |
| effective_cohesion_kPa | kPa | Effective cohesion | 0-30 |
| effective_friction_angle_deg | degrees | Effective friction angle | 15-35 |
| sensitivity | - | Sensitivity ratio | 1-50 |
| compression_index_Cc | - | Virgin compression index | 0.1-2.0 |
| swelling_index_Cs | - | Swelling/recompression index | 0.01-0.3 |
| coefficient_consolidation_cv_m2/yr | m²/yr | Coefficient of consolidation | 0.1-10 |
| coefficient_volume_change_mv_m2/kN | m²/kN | Coefficient of volume compressibility | 1E-5 to 1E-2 |
| time_factor_T90 | - | Time factor for 90% consolidation | 0.848 |
| elastic_modulus_MPa | MPa | Young's modulus | 0.5-50 |
| poisson_ratio | - | Poisson's ratio | 0.3-0.5 |
| shear_modulus_MPa | MPa | Shear modulus | 0.2-20 |
| bulk_modulus_MPa | MPa | Bulk modulus | 1-50 |
| permeability_m/s | m/s | Permeability | 1E-12 to 1E-7 |

## Clay Mineralogy

| Column | Unit | Description | Typical Range |
|--------|------|-------------|---------------|
| smectite_% | % | Smectite content | 0-80 |
| illite_% | % | Illite content | 0-50 |
| kaolinite_% | % | Kaolinite content | 0-50 |
| chlorite_% | % | Chlorite content | 0-30 |
| montmorillonite_% | % | Montmorillonite content | 0-30 |
| vermiculite_% | % | Vermiculite content | 0-10 |
| mixed_layer_% | % | Mixed layer clays | 0-20 |
| cation_exchange_capacity_meq/100g | meq/100g | CEC | 5-150 |
| specific_surface_area_m2/g | m²/g | Specific surface area | 10-800 |
| swelling_potential | - | Qualitative assessment | Low to Very_High |
| clay_fraction_activity | - | Activity classification | Inactive to Very_Active |
| XRD_peak_intensity | counts | X-ray diffraction intensity | 100-10000 |
| SEM_particle_size_um | μm | Particle size from SEM | 0.1-10 |
| organic_content_% | % | Organic matter content | 0-15 |
| carbonate_content_% | % | Carbonate content | 0-40 |
| iron_oxide_% | % | Iron oxide content | 0-10 |
| pH_value | - | pH value | 4-9 |

## Case Study Parameters

| Column | Unit | Description |
|--------|------|-------------|
| case_id | - | Unique case identifier |
| location | - | Geographic location |
| date | - | Date of failure event |
| structure_type | - | Type of underground structure |
| soil_type | - | Predominant soil type |
| depth_m | m | Structure depth |
| span_m | m | Structure span/diameter |
| failure_mode | - | Primary failure mechanism |
| triggering_factor | - | Main triggering event |
| max_settlement_mm | mm | Maximum vertical settlement |
| max_horizontal_displacement_mm | mm | Maximum horizontal movement |
| ground_loss_% | % | Volume loss percentage |
| damage_severity | - | Severity classification |
| casualties | - | Number of casualties |
| economic_loss_million_USD | Million USD | Economic impact |
| recovery_time_days | days | Time to restore operation |
| water_table_depth_m | m | Groundwater level |
| overburden_pressure_kPa | kPa | Overburden stress |
| support_system_type | - | Type of support system |
| lining_thickness_mm | mm | Lining thickness |
| grouting_performed | - | Whether grouting was done |