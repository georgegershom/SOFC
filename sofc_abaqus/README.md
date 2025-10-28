## SOFC Abaqus model (2D, coupled temperature–displacement, CPS4T)

This project builds and runs a 2D coupled temperature–displacement Abaqus/Standard simulation for a single SOFC repeat unit cross-section, matching the methods you described.

### Features
- Geometry: 10 mm × 1 mm rectangle, partitioned at y = 0.40/0.50/0.90 mm
- Layers: Anode, Electrolyte, Cathode, Interconnect with temperature-dependent elastic, CTE, k, cp
- Steps: Single coupled temperature–displacement step (CPS4T) with NLGEOM
- Thermal BCs: Bottom temperature ramp (HR1/HR4/HR10); Top film h = 25 W/m²K, T∞ = 298.15 K
- Mechanical BCs: Left edge Ux = 0; Bottom edge Uy = 0
- Mesh: Structured seeding; ~80 elements along x; refined around interfaces and in electrolyte
- Outputs: Field S, Mises, E/LE, TEMP, HFL; History S12 along interfaces

Note: The script currently implements elastic + thermal expansion. Plasticity (Johnson–Cook) and creep (Norton–Arrhenius) hooks are indicated but disabled to keep the noGUI script broadly compatible. You can enable them in Abaqus/CAE interactively or extend the script if your Abaqus version supports those API calls.

### Requirements
- Abaqus/Standard with CAE noGUI available on your machine (`abaqus` in PATH)

### Usage
```bash
# Default HR4 (4 °C/min), coupled CPS4T
bash run.sh

# Choose heating rate: hr1 | hr4 | hr10
bash run.sh hr1

# Or invoke directly
abaqus cae noGUI=build_model.py -- --hr hr4 --job sofc_hr4
```

Artifacts:
- Input/ODB and report files are written under `./jobs/<jobname>/`

### Heating profiles (Kelvin)
- HR1: ramp 25→1173.15 K in 875 min, hold 10 min, cool 875 min
- HR4: ramp in 218.75 min, hold 10 min, cool 218.75 min
- HR10: ramp in 87.5 min, hold 10 min, cool 87.5 min

### Layer properties (short form)
Temperatures are absolute (K). Elastic E in Pa; α in 1/K; k in W/m·K; cp in J/kg·K.
- Anode (Ni-YSZ): E 140e9→91e9; ν 0.30; α 12.5e-6→13.5e-6; k 6.0→4.0; cp 450→570
- Electrolyte (8YSZ): E 210e9→170e9; ν 0.28; α 10.5e-6→11.2e-6; k 2.6→2.0; cp 400→600
- Cathode (LSM): E 120e9→84e9; ν 0.30; α 11.5e-6→12.4e-6; k 2.0→1.8; cp 480→610
- Interconnect (Steel): E 205e9→150e9; ν 0.30; α 12.5e-6→13.2e-6; k 20→15; cp 500→700

### Notes
- Units are SI: m–kg–s–K–N–Pa. Geometry uses meters (10 mm = 0.01 m, 1 mm = 0.001 m).
- Out-of-plane thickness for 2D elements uses Abaqus default (1.0).
- If you require sequential heat→mechanical steps, Johnson–Cook plasticity, and Norton–Arrhenius creep in-script, say the word and I’ll extend this to generate both DC2D4 and CPS4 models with temperature import.

