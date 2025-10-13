# SOFC Synthetic Datasets

Synthetic, multi-fidelity, multi-physics dataset generator for Solid Oxide Fuel Cells (SOFC).

- Dataset 1: Physics-based surrogate fields (temperature, current density, species, stress/strain/displacement).
- Dataset 2: Experimental-like validation data (IV, EIS, thermal images, strain gauges, AE).
- Dataset 3: Real-time operational stream (1 Hz core signals, periodic EIS/images, AE events).

Install: `pip install -e .`

Usage: `sofc-gen --help`
