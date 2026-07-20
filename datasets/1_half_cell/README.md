# Half-Cell Thermo-Mechanical Characterisation

## ⚠️ SYNTHETIC DATA DISCLAIMER
All data in this directory are **synthetically generated** for research-scaffolding
purposes only. They are NOT real experimental measurements. Physical trends follow
published literature relationships with added Gaussian noise.

## Files
| File | Description | Units |
|------|-------------|-------|
| cte_vs_temperature.csv | CTE vs T for 6 materials, 10 specimens each | ppm/K |
| chemical_expansion_vs_pO2.csv | Chemical strain vs oxygen partial pressure | dimensionless |
| elastic_moduli_vs_temperature.csv | E, G, Poisson vs T | GPa, - |
| creep_parameters.csv | Norton creep strain rates; A, n, Q coefficients | s⁻¹, MPa, J/mol |
| weibull_fracture_strength.csv | Fracture strength distribution; Weibull m, σ₀ | MPa |
| interfacial_fracture_energy.csv | Gc vs T for anode/electrolyte interfaces | J/m² |
| redox_expansion.csv | Irreversible strain after re-oxidation cycles | dimensionless |

## Test Methods (simulated)
- CTE: dilatometry at 2 K/min, 25–900 °C
- Chemical expansion: isothermal optical dilatometry in H₂–H₂O atmospheres
- Elastic moduli: impulse excitation technique
- Creep: 4-point bending at 650/750/850 °C, constant stress 100 h
- Fracture strength: ring-on-ring biaxial flexure
- Interfacial fracture energy: double-cantilever beam
- Redox: isothermal re-oxidation cycles
