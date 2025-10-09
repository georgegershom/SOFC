# SOFC Electrochemical Loading Dataset (Fabricated)

This dataset contains fabricated IV and overpotential data across SOFC operating conditions.

## Files
- `electrochemical_loading.csv`: main data
- `schema.json`: column schema and units

## Key Columns
- `current_density_A_per_cm2`: Applied current density.
- `voltage_V`: Operating voltage under load.
- `ocv_V`: Open-circuit voltage (approximate).
- `overpotential_*_V`: Components (ohmic, activation anode/cathode, concentration).
- `R_*_ohm_cm2`: EIS-derived resistances (ohmic, charge-transfer).
- `delta_mu_O2_J_per_mol`: Oxygen chemical potential difference across electrolyte (≈ 4 F V).
- `anode_oxidation_risk_score` / `anode_ni_oxidation_flag`: Heuristic risk indicators.

## Assumptions
- Simplified Nernst, Butler–Volmer, and mass-transport models tuned for plausibility.
- Parameters chosen to yield realistic magnitudes at 700–850 °C.
- EIS peak frequencies assume constant double-layer capacitances.
