SOFC PI-DT Synthetic Dataset Generator

This package generates a cohesive, multi-modal dataset for an adaptive-scale, physics-informed digital twin of a Solid Oxide Fuel Cell (SOFC). Modalities include:
- Microstructure (3D volumes)
- Geometry / CAD parameters
- Operational profiles (inputs and boundary conditions)
- Electrochemical response (polarization and EIS)
- Thermo-structural fields (thermocouples, IR, strain gauges, DIC)
- Degradation timeseries and events
- Post-mortem synthetic SEM-like images

Use the CLI:

python -m scripts.sofc_pi_dt.cli --out data/sofc_pi_dt_dataset
