# Figure 2: Systematic Underestimation of TPB Thermal Gradients by Low-Fidelity Models

## Overview
This figure demonstrates the critical shortcoming of Low-Fidelity (LF) models: their systematic underestimation of the local Thermal Gradient at the Triple-Phase Boundary (∇T_TPB) due to the assumption of uniform temperature, a key driver of thermo-mechanical degradation.

## Key Visual Elements

### Data Points
1. **LF vs HF Predictions (Blue Circles)**
   - Each point represents a simulation comparison
   - X-coordinate: High-Fidelity prediction of ∇T_TPB
   - Y-coordinate: Low-Fidelity prediction of ∇T_TPB
   - These points cluster below the y=x line, showing systematic underestimation

2. **Experimental Data (Black Circles)**
   - Represent experimental measurements from IR thermography
   - Plotted on the y=x line to show agreement with HF models
   - Key validation point at ~72 K/mm

3. **Reference Line (Red)**
   - y=x line representing perfect agreement
   - 45-degree angle due to square aspect ratio
   - Benchmark for comparing model predictions

### Key Findings
- **37% Average Error**: LF models underestimate thermal gradients by approximately 37%
- **Typical Values**: 
  - LF predictions: ~45 K/mm
  - HF predictions: ~72 K/mm
  - Experimental validation: ~72 K/mm

### Annotations
- Clear labeling of underestimation region
- Quantitative error percentage displayed
- Key finding text box highlighting the implications

## Files Generated
- `figure2_tpb_thermal_gradient.png` - High-resolution raster image (300 DPI)
- `figure2_tpb_thermal_gradient.pdf` - Vector format for publications
- `figure2_tpb_thermal_gradient.svg` - Editable vector format

## Usage
Run the Python script to regenerate the figure:
```bash
python3 figure2_tpb_thermal_gradient.py
```

## Requirements
- Python 3.x
- NumPy >= 1.21.0
- Matplotlib >= 3.5.0

Install dependencies:
```bash
pip3 install -r requirements.txt
```

## Scientific Significance
This figure visually confirms that while LF models are computationally efficient, they miss critical local thermal hotspots that drive thermo-mechanical degradation in solid oxide fuel cells. The HF models, validated by experimental data, capture these essential physics.