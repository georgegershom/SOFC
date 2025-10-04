# Synthetic Synchrotron X-ray Data Generator for SOFC Creep Studies

This repository provides a comprehensive tool for generating realistic synthetic 4D (3D + time) synchrotron X-ray tomography and diffraction data for Solid Oxide Fuel Cell (SOFC) creep deformation studies. The generated data serves as "ground truth" for validating computational models and developing analysis algorithms.

## Features

### Core Dataset Generation
- **4D Synchrotron X-ray Tomography**: High-resolution 3D microstructure evolution over time
- **X-ray Diffraction (XRD)**: Phase identification and residual stress/strain mapping
- **Realistic Physics**: Implements creep deformation mechanisms including:
  - Cavity nucleation and growth
  - Crack propagation along grain boundaries
  - Grain boundary sliding
  - Stress concentration effects

### Key Measurements
- **Tomography Data**:
  - Initial microstructure (grains, grain boundaries, porosity)
  - Time-lapse microstructural evolution
  - Creep cavitation and crack propagation
  - Grain rotation and boundary migration

- **XRD Data**:
  - Multi-phase diffraction patterns
  - Spatial strain and stress field mapping
  - Peak broadening and shifting due to deformation
  - Realistic crystallographic phases for SOFC materials

### Comprehensive Metadata
- Operational parameters (temperature, stress, time)
- Material specifications (composition, grain size, properties)
- Sample geometry and imaging parameters
- Quantitative damage evolution metrics

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Required Packages
- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.5.0
- h5py >= 3.7.0
- scikit-image >= 0.19.0
- pandas >= 1.3.0
- tqdm >= 4.62.0
- pyvista >= 0.32.0 (optional, for advanced 3D visualization)

## Quick Start

### Basic Usage

```python
from synchrotron_data_generator import (
    SynchrotronDataGenerator, 
    MaterialProperties, 
    OperationalParameters, 
    SampleGeometry
)

# Define material properties (e.g., Crofer 22 APU)
material_props = MaterialProperties(
    alloy_composition={'Fe': 76.8, 'Cr': 22.0, 'Mn': 0.5, 'Ti': 0.08},
    grain_size_mean=15.0,  # μm
    initial_porosity=0.002,
    elastic_modulus=200.0,  # GPa
    poisson_ratio=0.3,
    thermal_expansion_coeff=12.0e-6,
    creep_exponent=5.0,
    activation_energy=300.0  # kJ/mol
)

# Define operating conditions
op_params = OperationalParameters(
    temperature=700.0,  # °C
    mechanical_stress=50.0,  # MPa
    time_points=[0, 10, 25, 50, 100, 200, 500, 1000],  # hours
    atmosphere="Air"
)

# Define sample geometry
sample_geom = SampleGeometry(
    length=5.0, width=2.0, thickness=0.5,  # mm
    shape="rectangular",
    volume=5.0  # mm³
)

# Generate dataset
generator = SynchrotronDataGenerator(
    voxel_size=0.5,  # μm
    image_dimensions=(512, 512, 256),
    seed=42
)

dataset = generator.generate_complete_dataset(
    material_props, op_params, sample_geom,
    output_dir="sofc_creep_data"
)
```

### Generate Multiple Scenarios

```bash
# Run example scenarios
python example_usage.py
```

This generates three realistic scenarios:
1. **High Stress Accelerated Testing**: 750°C, 100 MPa, 320 hours
2. **Long-term Service Conditions**: 650°C, 25 MPa, 8000 hours  
3. **Thermal Cycling**: 700°C, 40 MPa with thermal cycling effects

## Data Structure

### Generated Files
```
output_directory/
├── metadata.json                 # Experimental parameters and conditions
├── tomography_4d.h5            # 4D tomography data (3D + time)
├── xrd_data.h5                  # XRD patterns and strain/stress maps
├── analysis_metrics.json        # Quantitative damage evolution
└── dataset_summary.txt          # Human-readable summary
```

### Data Formats

#### Tomography Data (HDF5)
- **Structure**: `/time_{hours}h` datasets
- **Content**: 3D arrays with phase IDs (0=pore, 1=grain_boundary, 2+=grain_interior)
- **Dimensions**: User-defined (default: 512×512×256)
- **Voxel Size**: User-defined (default: 0.5 μm)

#### XRD Data (HDF5)
- **Strain Maps**: 6-component strain tensors (εxx, εyy, εzz, εxy, εxz, εyz)
- **Stress Maps**: 6-component stress tensors (σxx, σyy, σzz, τxy, τxz, τyz)
- **Diffraction Patterns**: 2θ, intensity, peak width, strain shifts for each phase

#### Analysis Metrics (JSON)
- Porosity evolution over time
- Crack density evolution
- Pore connectivity (Euler characteristic)
- Overall damage parameter progression

## Visualization

### Quick Visualization
```bash
# Generate comprehensive report
python visualize_data.py sofc_creep_data --report --output visualization_report

# Interactive plotting
python visualize_data.py sofc_creep_data
```

### Available Visualizations
1. **Damage Evolution**: Quantitative metrics over time
2. **Microstructure Slices**: 2D cross-sections showing evolution
3. **3D Damage Visualization**: Spatial distribution of pores/cracks
4. **XRD Pattern Evolution**: Diffraction peak changes over time
5. **Strain/Stress Field Maps**: Spatial stress concentration visualization

### Custom Visualization
```python
from visualize_data import SynchrotronDataVisualizer

visualizer = SynchrotronDataVisualizer('sofc_creep_data')

# Plot specific aspects
visualizer.plot_damage_evolution()
visualizer.plot_microstructure_slices(time_points=[0, 100, 500])
visualizer.plot_3d_damage_visualization(time_point=1000)
visualizer.plot_strain_stress_maps(time_point=500)
```

## Applications

### Model Validation
- **Finite Element Models**: Compare predicted vs. synthetic damage evolution
- **Phase Field Models**: Validate microstructure evolution mechanisms
- **Creep Laws**: Calibrate Norton-Bailey creep parameters

### Algorithm Development
- **Image Segmentation**: Train ML algorithms for phase identification
- **Crack Detection**: Develop automated crack tracking algorithms
- **Damage Quantification**: Validate porosity and connectivity measurements

### Research Applications
- **Failure Prediction**: Study damage accumulation patterns
- **Material Design**: Optimize microstructure for creep resistance
- **Operating Conditions**: Assess impact of temperature and stress variations

## Customization

### Material Properties
Modify `MaterialProperties` for different SOFC materials:
- **Interconnects**: Crofer 22 APU, 441, Haynes 230
- **Anodes**: Ni-YSZ cermets with varying compositions
- **Custom Alloys**: User-defined compositions and properties

### Physics Models
Extend creep mechanisms in `_apply_creep_mechanisms()`:
- Diffusion-controlled cavity growth
- Grain boundary migration
- Phase transformations
- Environmental effects (oxidation, carburization)

### Output Formats
Add custom export formats:
- VTK files for ParaView visualization
- TIFF stacks for ImageJ analysis
- Custom binary formats for specific analysis tools

## Validation

The synthetic data incorporates realistic physics based on:
- **Literature Values**: Material properties from peer-reviewed sources
- **Experimental Observations**: Typical damage patterns in SOFC materials
- **Physical Constraints**: Conservation laws and thermodynamic consistency
- **Statistical Validation**: Realistic grain size distributions and defect densities

### Key Validation Metrics
- Grain size distribution matches Weibull/log-normal statistics
- Creep rates follow Norton power law behavior
- Stress concentrations around defects match analytical solutions
- XRD peak positions and widths consistent with crystal structure

## Contributing

Contributions are welcome! Areas for improvement:
- Additional creep mechanisms (e.g., dislocation creep)
- More sophisticated XRD simulation (texture effects)
- Environmental degradation models
- GPU acceleration for large datasets
- Integration with experimental data formats

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{synchrotron_data_generator,
  title={Synthetic Synchrotron X-ray Data Generator for SOFC Creep Studies},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[username]/synchrotron-data-generator}
}
```

## Contact

For questions, suggestions, or collaborations:
- Email: [your.email@domain.com]
- Issues: [GitHub Issues](https://github.com/[username]/synchrotron-data-generator/issues)

## Acknowledgments

This work builds upon research in:
- SOFC materials science and degradation mechanisms
- Synchrotron X-ray imaging techniques
- Computational materials modeling
- High-temperature creep deformation physics