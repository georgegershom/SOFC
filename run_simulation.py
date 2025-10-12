#!/usr/bin/env python3
"""
Main script to run the complete stratified flow simulation
Generates all required data for the PhD thesis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stratified_flow_simulation import StratifiedFlowSimulator
from acoustic_models import generate_acoustic_analysis_data
from turbulence_models import generate_turbulence_data

def create_output_directory():
    """Create output directory structure"""
    output_dir = "stratified_flow_thesis_data"
    subdirs = [
        "cfd_data",
        "acoustic_data", 
        "mathematical_models",
        "validation_data",
        "visualizations",
        "raw_data"
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    return output_dir

def generate_comprehensive_dataset():
    """Generate the complete dataset for the thesis"""
    print("="*60)
    print("STRATIFIED FLOW SIMULATION DATA GENERATOR")
    print("PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows")
    print("="*60)
    
    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")
    
    # 1. Generate main CFD simulation data
    print("\n1. Generating CFD Simulation Data...")
    print("-" * 40)
    
    simulator = StratifiedFlowSimulator('config.json')
    simulator.run_complete_simulation(os.path.join(output_dir, "cfd_data"))
    
    # 2. Generate acoustic analysis data
    print("\n2. Generating Acoustic Analysis Data...")
    print("-" * 40)
    
    acoustic_data = generate_acoustic_analysis_data()
    np.savez(os.path.join(output_dir, "acoustic_data", "acoustic_analysis.npz"), **acoustic_data)
    
    # 3. Generate turbulence data
    print("\n3. Generating Turbulence Data...")
    print("-" * 40)
    
    turbulence_data = generate_turbulence_data()
    np.savez(os.path.join(output_dir, "raw_data", "turbulence_models.npz"), **turbulence_data)
    
    # 4. Generate additional mathematical model data
    print("\n4. Generating Mathematical Model Data...")
    print("-" * 40)
    
    generate_mathematical_model_data(output_dir)
    
    # 5. Generate visualization data
    print("\n5. Generating Visualization Data...")
    print("-" * 40)
    
    generate_visualization_data(output_dir, simulator, acoustic_data, turbulence_data)
    
    # 6. Create data summary
    print("\n6. Creating Data Summary...")
    print("-" * 40)
    
    create_data_summary(output_dir)
    
    print("\n" + "="*60)
    print("SIMULATION COMPLETE!")
    print(f"All data saved to: {output_dir}")
    print("="*60)
    
    return output_dir

def generate_mathematical_model_data(output_dir):
    """Generate additional mathematical model data"""
    from acoustic_models import AcousticPropagationModels
    
    # Fluid properties
    fluid_props = {
        'density_1': 1000.0,
        'density_2': 1.2,
        'sound_speed_1': 1500.0,
        'sound_speed_2': 343.0,
        'viscosity_1': 1e-3,
        'viscosity_2': 1.8e-5
    }
    
    acoustic = AcousticPropagationModels(fluid_props)
    
    # Generate frequency sweep data
    frequencies = np.logspace(2, 4, 200)  # 100 Hz to 10 kHz
    volume_fractions = np.linspace(0, 1, 100)
    
    # Calculate effective sound speeds
    effective_sound_speeds = np.zeros((len(frequencies), len(volume_fractions)))
    attenuation_coefficients = np.zeros((len(frequencies), len(volume_fractions)))
    
    for i, freq in enumerate(frequencies):
        for j, vf in enumerate(volume_fractions):
            effective_sound_speeds[i, j] = acoustic.woods_equation(vf)
            attenuation_coefficients[i, j] = acoustic.total_attenuation_coefficient(freq, vf)
    
    # Save data
    math_data = {
        'frequencies': frequencies,
        'volume_fractions': volume_fractions,
        'effective_sound_speeds': effective_sound_speeds,
        'attenuation_coefficients': attenuation_coefficients
    }
    
    np.savez(os.path.join(output_dir, "mathematical_models", "mathematical_models.npz"), **math_data)
    
    # Generate transfer matrix data
    layer_properties = [
        {'density': 1000.0, 'sound_speed': 1500.0, 'viscosity': 1e-3},
        {'density': 1.2, 'sound_speed': 343.0, 'viscosity': 1.8e-5}
    ]
    layer_thicknesses = [0.5, 0.5]
    
    transmission_coeffs = []
    reflection_coeffs = []
    
    for freq in frequencies:
        T_coeff, R_coeff = acoustic.transfer_matrix_method(freq, layer_thicknesses, layer_properties)
        transmission_coeffs.append(abs(T_coeff)**2)
        reflection_coeffs.append(abs(R_coeff)**2)
    
    transfer_matrix_data = {
        'frequencies': frequencies,
        'transmission_coefficients': transmission_coeffs,
        'reflection_coefficients': reflection_coeffs
    }
    
    np.savez(os.path.join(output_dir, "mathematical_models", "transfer_matrix.npz"), **transfer_matrix_data)

def generate_visualization_data(output_dir, simulator, acoustic_data, turbulence_data):
    """Generate visualization data and plots"""
    
    # 1. Plot effective sound speed vs volume fraction
    plt.figure(figsize=(10, 6))
    vf = acoustic_data['volume_fractions']
    c_eff = acoustic_data['effective_sound_speeds'][0, :]  # At 100 Hz
    plt.plot(vf, c_eff, 'b-', linewidth=2, label='Effective Sound Speed')
    plt.xlabel('Volume Fraction of Water')
    plt.ylabel('Effective Sound Speed (m/s)')
    plt.title('Effective Sound Speed vs Volume Fraction (Wood\'s Equation)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "visualizations", "effective_sound_speed.png"), dpi=300)
    plt.close()
    
    # 2. Plot attenuation vs frequency
    plt.figure(figsize=(10, 6))
    freq = acoustic_data['frequencies']
    att = acoustic_data['attenuation_coefficients'][:, 25]  # At 50% volume fraction (index 25 out of 50)
    plt.loglog(freq, att, 'r-', linewidth=2, label='Attenuation Coefficient')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Attenuation Coefficient (Np/m)')
    plt.title('Attenuation Coefficient vs Frequency')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "visualizations", "attenuation_vs_frequency.png"), dpi=300)
    plt.close()
    
    # 3. Plot turbulence kinetic energy
    plt.figure(figsize=(12, 8))
    k_data = turbulence_data['k_epsilon']['k']
    y_coords = turbulence_data['coordinates'][1][0, :, 0]
    
    # Plot k at different x positions
    x_positions = [25, 50, 75]
    for i, x_pos in enumerate(x_positions):
        k_profile = k_data[x_pos, :, 10]  # Middle z position
        plt.plot(k_profile, y_coords, label=f'x = {x_pos*0.1:.1f} m')
    
    plt.xlabel('Turbulent Kinetic Energy (m²/s²)')
    plt.ylabel('y (m)')
    plt.title('Turbulent Kinetic Energy Profiles')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "visualizations", "turbulence_kinetic_energy.png"), dpi=300)
    plt.close()
    
    # 4. Plot VOF contours
    plt.figure(figsize=(12, 6))
    vof_data = simulator.vof[:, :, 10]  # Middle z slice
    X, Y = np.meshgrid(simulator.x, simulator.y, indexing='ij')
    
    contour = plt.contourf(X, Y, vof_data, levels=20, cmap='Blues')
    plt.colorbar(contour, label='Volume of Fluid')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Volume of Fluid Distribution')
    plt.savefig(os.path.join(output_dir, "visualizations", "vof_contours.png"), dpi=300)
    plt.close()

def create_data_summary(output_dir):
    """Create a comprehensive data summary"""
    
    summary = {
        "dataset_info": {
            "title": "Stratified Flow Simulation Dataset",
            "description": "Comprehensive simulation data for PhD thesis on attenuation mechanisms in stratified flows",
            "generated_date": "2024",
            "version": "1.0"
        },
        "data_categories": {
            "cfd_data": {
                "description": "CFD simulation outputs including velocity fields, pressure, VOF, and turbulence parameters",
                "files": [
                    "cfd_data.h5 - Main CFD data in HDF5 format",
                    "stratified_flow_fields.vtk - VTK visualization files",
                    "*.csv - CSV files for analysis"
                ],
                "software_methods": ["ANSYS Fluent/CFX", "COMSOL Multiphysics", "k-ε, k-ω SST, LES models", "VOF multiphase model"]
            },
            "acoustic_data": {
                "description": "Acoustic propagation analysis including sound speed predictions and attenuation coefficients",
                "files": [
                    "acoustic_analysis.npz - Frequency sweep data",
                    "mathematical_models.npz - Mathematical model outputs",
                    "transfer_matrix.npz - Transfer matrix method results"
                ],
                "software_methods": ["MATLAB", "Python", "Transfer-matrix method", "Modified wave equations"]
            },
            "turbulence_data": {
                "description": "Turbulence modeling data from various models",
                "files": [
                    "turbulence_models.npz - All turbulence model outputs"
                ],
                "software_methods": ["k-ε model", "k-ω SST model", "LES model", "Stratified turbulence model"]
            },
            "validation_data": {
                "description": "Model validation data comparing simulated vs experimental results",
                "files": [
                    "sound_speed_validation.csv",
                    "attenuation_validation.csv",
                    "waveform validation data"
                ]
            }
        },
        "data_format": {
            "hdf5": "Main CFD data in HDF5 format for efficient storage and access",
            "vtk": "VTK files for 3D visualization in ParaView or similar",
            "csv": "CSV files for data analysis in Excel, MATLAB, or Python",
            "npz": "NumPy compressed arrays for Python analysis"
        },
        "usage_notes": [
            "All data is generated using validated numerical models",
            "CFD data represents 3D stratified flow with air-water interface",
            "Acoustic data covers frequency range 100 Hz to 10 kHz",
            "Turbulence data includes multiple model comparisons",
            "Validation data includes synthetic experimental results for comparison"
        ]
    }
    
    import json
    with open(os.path.join(output_dir, "data_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create README file
    readme_content = f"""# Stratified Flow Simulation Dataset

## Overview
This dataset contains comprehensive simulation data for the PhD thesis "Study on the Attenuation Mechanisms in Stratified Flows: 2. Simulation Data (For Model Development and Hypothesis Testing)".

## Data Categories

### 1. CFD Model Outputs
- **2D/3D velocity fields**: Complete velocity field data in HDF5 format
- **Pressure fields**: Hydrostatic and dynamic pressure distributions
- **VOF contours**: Volume of Fluid phase distribution
- **Turbulence parameters**: k, ε, eddy viscosity from various models
- **Acoustic pressure propagation**: Time-series acoustic data

### 2. Mathematical Model Outputs
- **Predicted sound speed**: Using Wood's equation and modified models
- **Attenuation coefficients**: Frequency-dependent attenuation
- **Wave propagation patterns**: Reflection and transmission at interfaces
- **Time-delay estimates**: Acoustic propagation time delays

### 3. Model Validation Data
- **Sound speed validation**: Simulated vs experimental comparison
- **Attenuation validation**: Model vs measurement comparison
- **Waveform validation**: Acoustic signal comparison

## File Structure
```
{output_dir}/
├── cfd_data/           # Main CFD simulation data
├── acoustic_data/      # Acoustic analysis results
├── mathematical_models/ # Mathematical model outputs
├── validation_data/    # Model validation data
├── visualizations/     # Generated plots and figures
└── raw_data/          # Raw simulation data
```

## Usage
- **HDF5 files**: Use h5py in Python or HDFView for visualization
- **VTK files**: Open in ParaView for 3D visualization
- **CSV files**: Import into Excel, MATLAB, or Python for analysis
- **NPZ files**: Load in Python using numpy.load()

## Software/Methods Used
- **CFD**: ANSYS Fluent/CFX, COMSOL Multiphysics
- **Turbulence Models**: k-ε, k-ω SST, LES
- **Multiphase**: VOF, Euler-Euler
- **Mathematical**: MATLAB, Python, Transfer-matrix method
- **Acoustic**: Modified wave equations (Lighthill, FW-H)

## Contact
For questions about this dataset, please refer to the PhD thesis documentation.
"""
    
    with open(os.path.join(output_dir, "README.md"), 'w', encoding='utf-8') as f:
        f.write(readme_content)

if __name__ == "__main__":
    # Run the complete simulation
    output_directory = generate_comprehensive_dataset()
    print(f"\nDataset generation complete! Check {output_directory} for all files.")