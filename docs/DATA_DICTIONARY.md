## Data Dictionary — Sintering Process → Microstructure

All units are SI unless noted. Angles in degrees.

### Identifiers
- **sample_id**: Integer identifier for each synthetic sample.

### Sintering parameters (inputs)
- **ramp_up_rate_c_per_min**: Temperature ramp rate during heating (°C/min).
- **peak_temperature_c**: Peak sintering temperature reached (°C).
- **hold_time_min**: Soak time at peak temperature (min).
- **cool_rate_c_per_min**: Cooling rate after hold (°C/min).
- **applied_pressure_mpa**: External pressure applied during sintering (MPa). Zero for conventional pressureless sintering.
- **atmosphere**: Furnace atmosphere category: `air`, `argon`, `nitrogen`, `vacuum`, or `hydrogen`.
- **initial_relative_density**: Green body relative density before sintering (fraction 0–1).
- **initial_median_pore_diameter_um**: Median pore diameter in the green body (µm), lognormal median.
- **initial_pore_size_lognormal_sigma**: Lognormal shape parameter (σ) for initial pore size distribution (dimensionless).

### Microstructure (outputs)
- **grain_size_mean_um**: Mean grain diameter (µm) estimated from synthetic SEM analysis proxy.
- **grain_size_std_um**: Standard deviation of grain diameter (µm).
- **porosity_percent**: Final total porosity (%), equals `100*(1 - relative_density)`.
- **pore_median_diameter_um**: Median pore diameter after sintering (µm), lognormal median.
- **pore_size_lognormal_sigma**: Lognormal σ after sintering (dimensionless).
- **hag_boundary_fraction_percent**: High-angle grain boundary fraction (%) from synthetic EBSD proxy.
- **avg_misorientation_deg**: Average grain boundary misorientation (°), synthetic EBSD proxy.
- **relative_density**: Final relative density (fraction 0–1).
- **sintering_index_model**: Dimensionless internal index used by the generator; useful for debugging and benchmarking optimization.

### Notes
- Values are clipped to plausible ranges to avoid nonphysical outcomes.
- The dataset is for algorithm development only; it does not represent any specific material system.

