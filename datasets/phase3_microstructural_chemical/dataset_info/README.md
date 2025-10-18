
Phase 3: Microstructural & Chemical Analysis Dataset (Synthetic)
===============================================================

Theme: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete (HPRC)

Contents
- SEM (ITZ-focused) grayscale images (.pgm) with crack/porosity metrics
- XRD patterns (.csv) with semi-quantitative phase areas (CH, CaCO3, CaO, C-S-H hump)
- TGA/DTG curves (.csv) with mass loss metrics vs temperature
- Micro-CT 3D labeled volumes (.npy) with porosity/crack fractions and connectivity
- Summary metrics linking microstructure/chemistry to macro degradation index

Interpretation (why macro-behavior occurs)
- Heating drives dehydroxylation of Portlandite (CH) and dehydration of C-S-H, reducing cohesive strength.
- Above ~700–900°C, decarbonation of CaCO3 and formation of CaO further embrittle the matrix.
- Rubber particles pyrolyze ~300–500°C, leaving voids; the ITZ around rubber weakens and fosters microcrack initiation.
- Thermal incompatibility between aggregates and paste amplifies ITZ microcracking.
- The emerging interconnected pore/crack network (captured by Micro-CT and SEM metrics) increases permeability and reduces load-bearing cross-section, explaining mass/strength loss.

Usage
- See summary_metrics.csv at the dataset root to correlate temperature, rubber content, phase changes, and microstructural metrics with a degradation index.

Note: All data are synthetic but constrained by temperature-dependent phenomena reported for cementitious materials and rubberized concrete.
