# SOFC Electrochemical Loading Dataset

## Overview
This dataset contains comprehensive electrochemical loading data for Solid Oxide Fuel Cells (SOFCs), including IV curves, Electrochemical Impedance Spectroscopy (EIS) measurements, and detailed overpotential analysis. The data is designed to support research on SOFC performance, degradation mechanisms, and particularly the stress-induced effects from Ni oxidation at the anode.

## Dataset Structure

```
sofc_electrochemical_data/
├── iv_curves/
│   ├── iv_curves_700C.csv
│   ├── iv_curves_750C.csv
│   └── iv_curves_800C.csv
├── eis_data/
│   ├── eis_800C_0.5A_cm2.csv
│   └── eis_750C_0.5A_cm2.csv
├── overpotentials/
│   ├── overpotentials_700C.csv
│   ├── overpotentials_750C.csv
│   └── overpotentials_800C.csv
├── scripts/
│   ├── visualize_iv_curves.py
│   ├── visualize_eis.py
│   ├── visualize_overpotentials.py
│   └── requirements.txt
├── docs/
│   └── data_dictionary.md
└── README.md
```

## Data Description

### 1. IV Curves (`iv_curves/`)
Current-Voltage characteristics at different operating temperatures:
- **Temperature Range**: 700°C, 750°C, 800°C
- **Current Density**: 0 - 1.0 A/cm²
- **Parameters**:
  - Operating voltage (V)
  - Current density (A/cm²)
  - Power density (W/cm²)
  - Fuel utilization (%)
  - Air utilization (%)
  - Time stamps (h)

### 2. EIS Data (`eis_data/`)
Electrochemical Impedance Spectroscopy measurements:
- **Frequency Range**: 10 mHz - 100 kHz (logarithmic spacing)
- **Operating Conditions**: Various temperatures and current densities
- **Parameters**:
  - Frequency (Hz)
  - Real impedance (Ω·cm²)
  - Imaginary impedance (Ω·cm²)
  - Magnitude (Ω·cm²)
  - Phase angle (degrees)

### 3. Overpotentials (`overpotentials/`)
Detailed breakdown of voltage losses and their effects:
- **Components**:
  - Anode overpotential (mV)
  - Cathode overpotential (mV)
  - Ohmic overpotential (mV)
- **Critical Parameters**:
  - Ni oxidation risk level (Low/Medium/High/Very High)
  - Local oxygen partial pressure at anode (atm)
  - Volume change from Ni→NiO conversion (%)
  - Induced mechanical stress (MPa)

## Key Features

### Oxygen Chemical Potential Gradient
The dataset captures the relationship between:
- Operating voltage and oxygen chemical potential (μO₂)
- Current density and oxygen flux across the electrolyte
- Local pO₂ variations at electrode interfaces

### Ni Oxidation Mechanism
Critical data for understanding anode degradation:
- **Threshold pO₂**: ~10⁻¹⁸ atm at 800°C for Ni/NiO transition
- **Volume expansion**: ~70% increase from Ni→NiO
- **Stress generation**: Up to 100+ MPa at high current densities
- **Risk zones**: Mapped across temperature-current operating space

## Physical Relationships

### Nernst Equation
```
V = V₀ - (RT/4F) × ln(pO₂,cathode/pO₂,anode)
```

### Local pO₂ at Anode
```
pO₂,local = pO₂,fuel × exp(η_anode × 4F/RT)
```

### Stress from Volume Change
```
σ = E × ε × (ΔV/V) × f_oxidation
```
where:
- E = Young's modulus (~200 GPa for NiO)
- ε = strain from volume mismatch
- ΔV/V = 0.7 (70% volume increase)
- f_oxidation = fraction of Ni oxidized

## Usage Instructions

### Installation
```bash
cd scripts/
pip install -r requirements.txt
```

### Visualization Scripts

1. **IV Curve Analysis**:
```bash
python visualize_iv_curves.py
```
Generates:
- IV curves comparison
- Power density curves
- Fuel/air utilization plots
- Performance metrics calculation

2. **EIS Analysis**:
```bash
python visualize_eis.py
```
Generates:
- Nyquist plots
- Bode plots (magnitude & phase)
- DRT analysis
- Extracted impedance parameters

3. **Overpotential Analysis**:
```bash
python visualize_overpotentials.py
```
Generates:
- Overpotential breakdown charts
- Ni oxidation risk maps
- Stress evolution plots
- Operating safety boundaries

## Data Quality Notes

- **Temperature Dependence**: All measurements include temperature compensation
- **Steady-State**: Data represents stabilized operating conditions
- **Uncertainty**: Typical measurement uncertainty ±2% for voltage, ±1% for current
- **Sampling Rate**: 41-50 data points per IV curve for high resolution

## Applications

This dataset is suitable for:
1. **Performance Modeling**: Developing and validating SOFC models
2. **Degradation Studies**: Understanding anode oxidation mechanisms
3. **Control Strategy Development**: Identifying safe operating boundaries
4. **Machine Learning**: Training predictive models for SOFC behavior
5. **Stress Analysis**: Coupling electrochemical and mechanical models

## Critical Operating Boundaries

### Safe Operating Zones (Ni Oxidation Risk = Low)
- 700°C: < 0.225 A/cm²
- 750°C: < 0.250 A/cm²  
- 800°C: < 0.300 A/cm²

### High Risk Zones (Significant Ni Oxidation)
- Current density > 0.5 A/cm²
- Local pO₂ > 10⁻¹⁸ atm
- Stress > 50 MPa

## References

Key physical constants used:
- Faraday constant (F): 96485 C/mol
- Gas constant (R): 8.314 J/mol·K
- Ni/NiO volume ratio: 1.7
- YSZ electrolyte conductivity: Temperature-dependent

## Data Format

All CSV files use:
- Delimiter: Comma (,)
- Decimal: Period (.)
- Encoding: UTF-8
- Headers: First row contains parameter names with units

## License and Citation

This is a synthetic dataset generated for research purposes. When using this data, please acknowledge:

"SOFC Electrochemical Loading Dataset - Synthetic data for SOFC performance and degradation analysis, focusing on stress-induced effects from Ni oxidation"

## Contact

For questions about the dataset or to report issues, please refer to the documentation in the `docs/` folder.