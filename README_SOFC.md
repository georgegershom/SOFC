### SOFC Abaqus/Standard deck generator (CPS4T coupled)

This workspace includes a Python script that generates a runnable Abaqus/Standard input deck for a 2D SOFC cross-section using coupled temperature–displacement elements (CPS4T). The model matches the described methods: geometry (10 mm × 1 mm), layer partitions at 0.40/0.50/0.90 mm, temperature-dependent materials, bottom temperature schedule (HR1/HR4/HR10), and top-edge film convection.

#### Files
- `scripts/generate_sofc_inp.py`: generator script.

#### Requirements
- Python 3.8+
- Abaqus/Standard (command: `abaqus`) installed and licensed on your machine.

#### Generate input deck
```bash
python /workspace/scripts/generate_sofc_inp.py --schedule HR4 --output /workspace/sofc_cps4t_HR4.inp
```

Options:
- `--schedule {HR1,HR4,HR10}`: thermal schedule (default HR4)
- `--nelx N`: elements along x (default 80)
- `--ny a,b,c,d`: elements across thickness for [anode, electrolyte, cathode, interconnect] (default `48,12,48,12`)

#### Run in Abaqus/Standard
```bash
cd /workspace
abaqus job=sofc_cps4t_HR4 input=/workspace/sofc_cps4t_HR4.inp standard parallel=domain cpus=4 interactive
```

Notes:
- The step is `*Coupled Temperature-Displacement, nlgeom=YES`. Bottom edge temperature is prescribed by amplitude, top edge has film h=25 W/m^2-K to 298 K ambient.
- Field outputs: S, E, LE, NT, HFL; Node outputs: U, NT. History outputs for S11/S22/S12 along interface-adjacent element rows.
- Materials include temperature-dependent elastic, expansion, conductivity, and specific heat with densities to enable transient capacity. Plasticity and creep can be added later if desired.

#### Switching schedules
```bash
python /workspace/scripts/generate_sofc_inp.py --schedule HR1 --output /workspace/sofc_cps4t_HR1.inp
python /workspace/scripts/generate_sofc_inp.py --schedule HR10 --output /workspace/sofc_cps4t_HR10.inp
```

#### Post-processing
Open the ODB in Abaqus/Viewer. Plot contours of von Mises stress, temperature, and create XY plots of S12 along the interface element-row sets to evaluate interfacial shear proxies.

