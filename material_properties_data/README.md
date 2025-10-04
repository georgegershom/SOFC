# Material Properties Dataset for SOFC Multi-Physics Models

Generated: 2025-10-04 11:25:21

## Materials Included

- **8YSZ**: 8% Yttria-Stabilized Zirconia (Electrolyte)
- **LSM**: La0.8Sr0.2MnO3 (Cathode)
- **Ni-YSZ**: Nickel-YSZ Cermet (Anode)
- **FSS**: Ferritic Stainless Steel (Interconnect)

## Dataset Files

### 1. mechanical_properties.csv
Contains temperature-dependent mechanical properties:
- Temperature range: 20°C to 1000°C (20°C intervals)
- Young's Modulus (GPa)
- Tensile Strength (MPa)
- Poisson's Ratio (dimensionless)

**Use for**: Structural mechanics simulations, stress analysis, thermal-mechanical coupling

### 2. creep_properties.csv
Contains creep strain vs. time data:
- Temperatures: 700, 750, 800, 850, 900°C
- Stress levels: 50, 75, 100, 125, 150 MPa
- Time range: 0 to 1000 hours
- Creep Strain (%)

**Use for**: Long-term degradation modeling, creep-fatigue analysis, lifetime prediction

### 3. thermophysical_properties.csv
Contains temperature-dependent thermal properties:
- Temperature range: 20°C to 1000°C (20°C intervals)
- Coefficient of Thermal Expansion (CTE, 1/K)
- Thermal Conductivity (W/(m·K))
- Specific Heat Capacity (J/(kg·K))
- Density (kg/m³)

**Use for**: Heat transfer simulations, thermal stress analysis, thermal-fluid coupling

### 4. electrochemical_properties.csv
Contains temperature-dependent electrochemical properties:
- Temperature range: 400°C to 1000°C (20°C intervals)
- Ionic Conductivity (S/m)
- Electronic Conductivity (S/m)
- Total Conductivity (S/m)
- Ionic Transport Number (dimensionless)

**Use for**: Electrochemical simulations, Butler-Volmer kinetics, charge transport modeling

## Data Generation Methodology

All data is **synthetically generated** based on:
1. Published literature values for SOFC materials
2. Physics-based models (Arrhenius, Norton-Bailey creep law)
3. Realistic experimental noise (2-5% depending on property)

### Key Equations Used:

**Mechanical Properties**: Linear temperature dependence with material-specific coefficients

**Creep**: Norton-Bailey Law
```
ε = A × σⁿ × tᵐ × exp(-Q/RT)
```

**Thermal Properties**: Empirical temperature-dependent correlations

**Electrochemical**: Arrhenius Equation
```
σ = σ₀ × exp(-Eₐ/RT)
```

## Material Property Highlights

### 8YSZ (Electrolyte)
- High ionic conductivity (~0.1 S/m at 800°C)
- Low electronic conductivity (insulator)
- Moderate mechanical strength
- Low thermal conductivity

### LSM (Cathode)
- Mixed ionic-electronic conductor (MIEC)
- Good electronic conductivity at operating temperatures
- Compatible CTE with YSZ
- Moderate mechanical strength

### Ni-YSZ (Anode)
- Very high electronic conductivity (due to Ni network)
- Good ionic conductivity (due to YSZ phase)
- Porous composite structure (reflected in lower mechanical properties)
- Risk of Ni coarsening at high temperatures

### FSS (Interconnect)
- Pure electronic conductor
- High thermal conductivity
- Good mechanical strength
- CTE matching is critical

## Usage Notes

1. **Temperature Units**: All temperatures are in Celsius (°C)
2. **Consistency**: Convert to Kelvin for thermodynamic calculations
3. **Interpolation**: Linear interpolation is recommended for intermediate temperatures
4. **Extrapolation**: DO NOT extrapolate beyond the given temperature ranges
5. **Validation**: This is synthetic data - validate with experimental data when available

## Multi-Physics Coupling Considerations

### Thermo-Mechanical
- Use CTE mismatch for thermal stress calculations
- Temperature-dependent Young's modulus for accurate stress prediction
- Include creep for long-term simulations

### Electro-Thermal
- Joule heating: Q = σ × E²
- Temperature affects conductivity (Arrhenius)
- Include contact resistance at interfaces

### Electro-Chemo-Mechanical
- Volume changes during redox cycling (especially Ni-YSZ)
- Stress affects ionic conductivity
- Chemical expansion coupling

## References & Validation

For production use, validate against:
- NIST materials database
- Manufacturer datasheets
- Direct experimental measurements
- Peer-reviewed literature (key papers in SOFC field)

## Data Format

All CSV files use the following conventions:
- Comma-separated values
- Header row with descriptive column names
- Consistent material naming across all files
- SI-derived units (specified in column headers)

## License & Usage

This synthetic dataset is provided for research and development purposes.
For commercial applications, validate with experimental data.

---

**Contact**: For questions about data generation methodology or specific material properties.
