# Fabricated Material Property Dataset

This dataset provides plausible, literature-inspired but fabricated material properties for SOFC-relevant phases and interfaces. It is intended for simulation, sensitivity analysis, and surrogate modeling. Do not treat any value as measured.

## Contents
- `elastic_properties_rt.csv`: Room-temperature elastic properties for YSZ, Ni, NiO, and Ni–YSZ composites.
- `elastic_properties_T.json`: Temperature-dependent Young's modulus and Poisson's ratio from 25–1000 C (25 C step).
- `cte_T.json`: Temperature-dependent coefficient of thermal expansion (CTE) for pure phases and Ni–YSZ composites.
- `fracture_properties.csv`: Bulk and interface fracture properties, including derived G_c and fabricated K_IC.
- `chemical_expansion.csv`: Chemical expansion coefficients for Ni→NiO and representative perovskites (LSCF, LSM).
- `manifest.json`: Index of files and the temperature grid.

## Schema

### elastic_properties_rt.csv
- `material` (string): YSZ, Ni, NiO, Ni-YSZ
- `phase` (string): ceramic, metal, composite
- `composition_note` (string): purity or Ni vol.% for composites
- `porosity_vol_frac` (float): Assumed 0 for pure, 0.10 for composites
- `temperature_C` (int): 25
- `E_GPa` (float): Young's modulus
- `poisson_ratio` (float)

### elastic_properties_T.json
Array of entries with fields:
- `material`, `phase`, `composition_note`, `porosity_vol_frac`
- `temperature_C` (array[int])
- `E_GPa` (array[float])
- `poisson_ratio` (array[float])

### cte_T.json
Array of entries with fields:
- `material`, `phase`, `composition_note`
- `temperature_C` (array[int])
- `alpha_1_per_K` (array[float])

### fracture_properties.csv
- `type` (string): bulk | interface
- `material_or_interface` (string)
- `description` (string)
- `temperature_C` (int)
- `environment` (string)
- `E_ref_GPa` (float): Reference plane-strain modulus for interfaces; Young's modulus for bulk
- `poisson_ref` (float|null)
- `K_IC_MPa_sqrt_m` (float)
- `G_c_J_per_m2` (float): Derived via plane-strain relation G_c = K_IC^2 / E'
- `uncertainty_note` (string)

### chemical_expansion.csv
- `material` (string)
- `family` (string): phase-change | perovskite
- `parameter` (string)
- `value` (float)
- `unit` (string)
- `reference_temperature_C` (int)
- `std_dev` (float)
- `notes` (string)

## Modeling Assumptions
- Elastic properties degrade linearly with temperature; Poisson's ratio increases slightly with temperature.
- Composites use Voigt-Reuss-Hill mixing for E and linear mixing for ν and CTE.
- Interface equivalent plane-strain modulus uses harmonic average of E' for adjoining media.
- Bulk K_IC values are sampled from normal distributions within plausible ranges; G_c computed from K_IC and E'.
- Interface G_c values are fabricated around condition-specific means (as-sintered, reduced, oxidized) and back-converted to K_IC using E'.
- Ni→NiO chemical expansion uses PB ratio ≈ 1.69 (volumetric) with linear-strain proxy ≈ 0.19.
- LSCF/LSM chemical expansion coefficients are representative fabricated magnitudes.

## Units
- Temperatures in C; convert to K by adding 273.15.
- E in GPa; ν unitless; CTE in 1/K.
- K_IC in MPa√m; G_c in J/m².
- Chemical expansion as strain per unit composition change unless otherwise stated.

## Reproducibility
Run:
```bash
python3 /workspace/scripts/generate_material_properties.py
```

## Disclaimer
All values are fabricated for testing and modeling. Not suitable for design or validation.
