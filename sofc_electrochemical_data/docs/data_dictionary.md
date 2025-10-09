# Data Dictionary - SOFC Electrochemical Loading Dataset

## IV Curves Data Fields

| Field Name | Unit | Description | Range | Data Type |
|------------|------|-------------|-------|-----------|
| Temperature_C | °C | Operating temperature | 700-800 | Float |
| Current_Density_A_cm2 | A/cm² | Applied current density | 0-1.0 | Float |
| Voltage_V | V | Cell voltage | 0.6-1.15 | Float |
| Power_Density_W_cm2 | W/cm² | Power output per unit area | 0-0.65 | Float |
| Fuel_Utilization_% | % | Percentage of fuel consumed | 0-100 | Float |
| Air_Utilization_% | % | Percentage of oxygen consumed | 0-72 | Float |
| Time_h | hours | Time point of measurement | 0-20 | Float |

### Derived Metrics
- **Open Circuit Voltage (OCV)**: Voltage at zero current
- **Area Specific Resistance (ASR)**: Slope of V-I curve (Ω·cm²)
- **Maximum Power Point**: Peak of power density curve

## EIS Data Fields

| Field Name | Unit | Description | Range | Data Type |
|------------|------|-------------|-------|-----------|
| Frequency_Hz | Hz | AC frequency | 0.01-100000 | Float |
| Real_Impedance_Ohm_cm2 | Ω·cm² | Real part of impedance | 0.14-0.45 | Float |
| Imaginary_Impedance_Ohm_cm2 | Ω·cm² | Imaginary part of impedance | -0.065-0 | Float |
| Magnitude_Ohm_cm2 | Ω·cm² | |Z| = √(Re²+Im²) | 0.14-0.45 | Float |
| Phase_Angle_deg | degrees | θ = arctan(Im/Re) | -15-0 | Float |
| Temperature_C | °C | Operating temperature | 750-800 | Float |
| Current_Density_A_cm2 | A/cm² | DC bias current | 0.5 | Float |

### Key Impedance Components
- **High Frequency Intercept**: Ohmic resistance (electrolyte + contacts)
- **Low Frequency Intercept**: Total resistance (ohmic + polarization)
- **Arc Width**: Polarization resistance

## Overpotentials Data Fields

| Field Name | Unit | Description | Range | Data Type |
|------------|------|-------------|-------|-----------|
| Current_Density_A_cm2 | A/cm² | Applied current density | 0-1.0 | Float |
| Anode_Overpotential_mV | mV | Voltage loss at anode | 0-185 | Float |
| Cathode_Overpotential_mV | mV | Voltage loss at cathode | 0-306 | Float |
| Ohmic_Overpotential_mV | mV | Ohmic losses | 0-240 | Float |
| Total_Overpotential_mV | mV | Sum of all losses | 0-731 | Float |
| Ni_Oxidation_Risk | - | Risk categorization | Low/Medium/High/Very High | String |
| Local_pO2_atm | atm | O₂ partial pressure at anode | 1e-22 - 1e-16 | Float |
| Volume_Change_% | % | Volume expansion from oxidation | 0-3.1 | Float |
| Stress_MPa | MPa | Mechanical stress induced | 0-114 | Float |
| Temperature_C | °C | Operating temperature | 700-800 | Float |

### Risk Level Definitions

| Risk Level | pO₂ Range (atm) | Stress (MPa) | Description |
|------------|-----------------|--------------|-------------|
| Low | < 1e-19 | < 25 | Minimal oxidation risk |
| Medium | 1e-19 - 1e-18 | 25-50 | Some oxidation possible |
| High | 1e-18 - 1e-17 | 50-75 | Significant oxidation expected |
| Very High | > 1e-17 | > 75 | Severe oxidation and stress |

## Relationships Between Parameters

### Overpotential Contributions
```
V_cell = V_OCV - η_anode - η_cathode - η_ohmic
```

### Butler-Volmer Kinetics (Simplified)
```
η_anode ≈ (RT/αnF) × ln(i/i₀)
```

### Ohmic Loss
```
η_ohmic = i × R_ohmic
```
where R_ohmic includes:
- Electrolyte resistance
- Electrode resistance
- Contact resistance

### Stress Calculation
```
σ = E_eff × ε_mismatch × X_oxidized
```
where:
- E_eff = Effective modulus (~200 GPa)
- ε_mismatch = Strain from 70% volume expansion
- X_oxidized = Fraction of Ni oxidized

## Data Quality Indicators

### Measurement Precision
- Voltage: ±2 mV
- Current: ±1 mA/cm²
- Frequency: ±0.1%
- Temperature: ±2°C

### Data Validation Checks
1. **Conservation**: Power = Voltage × Current
2. **Monotonicity**: Voltage decreases with current
3. **Kramers-Kronig**: EIS data consistency
4. **Physical Limits**: 0 < η < V_OCV

## Units Conversion Reference

| Parameter | SI Unit | Common Unit | Conversion |
|-----------|---------|-------------|------------|
| Current density | A/m² | A/cm² | 1 A/cm² = 10⁴ A/m² |
| Resistance | Ω·m² | Ω·cm² | 1 Ω·cm² = 10⁻⁴ Ω·m² |
| Pressure | Pa | atm | 1 atm = 101325 Pa |
| Stress | Pa | MPa | 1 MPa = 10⁶ Pa |
| Power density | W/m² | W/cm² | 1 W/cm² = 10⁴ W/m² |

## Time Evolution Indicators

While this dataset represents steady-state conditions, time-dependent phenomena include:
- **Activation**: Initial performance improvement (0-10 h)
- **Steady State**: Stable operation (10-1000 h)
- **Degradation**: Long-term decline (>1000 h)

## Missing Data Handling

- Empty cells: No measurement at that condition
- NaN values: Measurement failed or out of range
- Zero values: Actual zero (except where physically impossible)