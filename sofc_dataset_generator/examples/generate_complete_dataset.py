"""
Complete SOFC dataset generation example.
Demonstrates how to generate all three types of datasets for digital twin training.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import logging

# Import the dataset generator
from sofc_dataset_generator import SOFCDatasetGenerator
from sofc_dataset_generator.data_formats.hdf5_utils import HDF5DatasetManager

def setup_logging():
    """Setup logging for the example."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('sofc_dataset_generation.log'),
            logging.StreamHandler()
        ]
    )

def generate_high_fidelity_dataset():
    """Generate high-fidelity physics-based simulation dataset."""
    print("=" * 60)
    print("GENERATING HIGH-FIDELITY SIMULATION DATASET")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = SOFCDatasetGenerator()
    
    # Define parameter ranges for Latin Hypercube Sampling
    operating_ranges = {
        'current_density': (0.1, 1.0),  # A/cm²
        'fuel_utilization': (0.6, 0.9),  # %
        'air_utilization': (0.1, 0.3),  # %
        'inlet_fuel_temp': (973.15, 1173.15),  # K
        'inlet_air_temp': (973.15, 1173.15),  # K
    }
    
    material_ranges = {
        'anode_porosity': (0.2, 0.4),
        'cathode_porosity': (0.2, 0.4),
        'electrolyte_thickness': (5e-6, 20e-6),  # m
        'anode_thickness': (200e-6, 800e-6),  # m
        'cathode_thickness': (20e-6, 100e-6),  # m
    }
    
    # Generate parameter combinations
    n_samples = 100  # Reduced for example
    param_combinations = generator.high_fidelity_gen.generate_parameter_sweep(
        n_samples=n_samples,
        operating_ranges=operating_ranges,
        material_ranges=material_ranges
    )
    
    # Define degradation states
    degradation_states = [
        {'crack_length': 1e-6, 'porosity_change': 0.05},  # 5% porosity increase
        {'crack_length': 5e-6, 'porosity_change': 0.10},  # 10% porosity increase
        {'crack_length': 10e-6, 'porosity_change': 0.15}, # 15% porosity increase
    ]
    
    # Generate high-fidelity dataset
    start_time = time.time()
    hf_dataset = generator.generate_high_fidelity_data(
        n_samples=n_samples,
        operating_conditions_range=operating_ranges,
        material_properties_range=material_ranges,
        degradation_states=degradation_states,
        output_file='datasets/high_fidelity_simulations.h5'
    )
    generation_time = time.time() - start_time
    
    print(f"High-fidelity dataset generated in {generation_time:.2f} seconds")
    print(f"Number of samples: {len(hf_dataset['parameters'])}")
    print(f"Spatial resolution: {hf_dataset['spatial_resolution']}")
    print(f"Dataset size: {len(hf_dataset)} variables")
    
    return hf_dataset

def generate_experimental_dataset():
    """Generate experimental validation dataset."""
    print("\n" + "=" * 60)
    print("GENERATING EXPERIMENTAL VALIDATION DATASET")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = SOFCDatasetGenerator()
    
    # Define operating profile
    operating_profile = {
        'current_density': 0.5,  # A/cm²
        'inlet_fuel_temp': 1073.15,  # K
        'inlet_air_temp': 1073.15,  # K
        'fuel_flow_rate': 0.1,  # mol/s
        'air_flow_rate': 0.5,  # mol/s
    }
    
    # Generate experimental dataset
    start_time = time.time()
    exp_dataset = generator.generate_experimental_data(
        test_duration_hours=24.0,  # 24 hours for example
        operating_profile=operating_profile,
        output_file='datasets/experimental_validation.h5'
    )
    generation_time = time.time() - start_time
    
    print(f"Experimental dataset generated in {generation_time:.2f} seconds")
    print(f"Test duration: {exp_dataset['metadata']['test_duration_hours']} hours")
    print(f"Sampling frequency: {exp_dataset['metadata']['sampling_frequency']} Hz")
    print(f"Number of time points: {len(exp_dataset['time_points'])}")
    
    return exp_dataset

def generate_monitoring_dataset():
    """Generate real-time monitoring dataset."""
    print("\n" + "=" * 60)
    print("GENERATING REAL-TIME MONITORING DATASET")
    print("=" * 60)
    
    # Initialize dataset generator
    generator = SOFCDatasetGenerator()
    
    # Generate monitoring dataset
    start_time = time.time()
    mon_dataset = generator.generate_monitoring_data(
        duration_hours=6.0,  # 6 hours for example
        high_freq_sampling=1.0,  # 1 Hz
        low_freq_sampling=1/3600.0,  # Hourly
        output_file='datasets/real_time_monitoring.h5'
    )
    generation_time = time.time() - start_time
    
    print(f"Monitoring dataset generated in {generation_time:.2f} seconds")
    print(f"Duration: {mon_dataset['metadata']['duration_hours']} hours")
    print(f"High-frequency sampling: {mon_dataset['metadata']['high_freq_sampling']} Hz")
    print(f"Low-frequency sampling: {mon_dataset['metadata']['low_freq_sampling']} Hz")
    
    return mon_dataset

def visualize_high_fidelity_data(dataset):
    """Visualize high-fidelity simulation data."""
    print("\n" + "=" * 60)
    print("VISUALIZING HIGH-FIDELITY DATA")
    print("=" * 60)
    
    # Create visualization directory
    viz_dir = Path('visualizations/high_fidelity')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Operating conditions distribution
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('High-Fidelity Dataset: Operating Conditions Distribution', fontsize=16)
    
    param_names = ['current_density', 'fuel_utilization', 'air_utilization', 
                   'inlet_fuel_temp', 'anode_porosity', 'cathode_porosity']
    
    for i, param in enumerate(param_names):
        if param in dataset['parameters'][0]:
            values = [p[param] for p in dataset['parameters']]
            row, col = i // 3, i % 3
            axes[row, col].hist(values, bins=20, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{param.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'operating_conditions_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Global performance metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('High-Fidelity Dataset: Global Performance Metrics', fontsize=16)
    
    # Cell voltage distribution
    axes[0, 0].hist(dataset['cell_voltage'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Cell Voltage Distribution')
    axes[0, 0].set_xlabel('Voltage (V)')
    axes[0, 0].set_ylabel('Frequency')
    
    # Efficiency distribution
    axes[0, 1].hist(dataset['efficiency'], bins=20, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Efficiency Distribution')
    axes[0, 1].set_xlabel('Efficiency')
    axes[0, 1].set_ylabel('Frequency')
    
    # Power density distribution
    axes[1, 0].hist(dataset['power_density'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Power Density Distribution')
    axes[1, 0].set_xlabel('Power Density (W/cm²)')
    axes[1, 0].set_ylabel('Frequency')
    
    # Max temperature distribution
    axes[1, 1].hist(dataset['max_temperature'], bins=20, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Maximum Temperature Distribution')
    axes[1, 1].set_xlabel('Temperature (K)')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'global_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 3D field visualization (sample)
    if len(dataset['temperature_field']) > 0:
        sample_idx = 0
        temp_field = dataset['temperature_field'][sample_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'3D Field Visualization (Sample {sample_idx})', fontsize=16)
        
        # Temperature field at different z-slices
        z_slices = [temp_field.shape[2]//4, temp_field.shape[2]//2, 3*temp_field.shape[2]//4]
        
        for i, z in enumerate(z_slices):
            im = axes[i].imshow(temp_field[:, :, z], cmap='hot', origin='lower')
            axes[i].set_title(f'Temperature at z-slice {z}')
            axes[i].set_xlabel('X')
            axes[i].set_ylabel('Y')
            plt.colorbar(im, ax=axes[i], label='Temperature (K)')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'temperature_field_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"High-fidelity visualizations saved to {viz_dir}")

def visualize_experimental_data(dataset):
    """Visualize experimental validation data."""
    print("\n" + "=" * 60)
    print("VISUALIZING EXPERIMENTAL DATA")
    print("=" * 60)
    
    # Create visualization directory
    viz_dir = Path('visualizations/experimental')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. Time series plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Experimental Data: Time Series', fontsize=16)
    
    time_hours = dataset['time_points'] / 3600  # Convert to hours
    
    # Voltage time series
    axes[0, 0].plot(time_hours, dataset['global_operational']['voltage'], 'b-', linewidth=1)
    axes[0, 0].set_title('Cell Voltage vs Time')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Current time series
    axes[0, 1].plot(time_hours, dataset['global_operational']['current'], 'r-', linewidth=1)
    axes[0, 1].set_title('Current vs Time')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature time series
    axes[1, 0].plot(time_hours, dataset['global_operational']['inlet_fuel_temp'], 'g-', linewidth=1, label='Inlet Fuel')
    axes[1, 0].plot(time_hours, dataset['global_operational']['outlet_fuel_temp'], 'g--', linewidth=1, label='Outlet Fuel')
    axes[1, 0].set_title('Temperature vs Time')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power time series
    axes[1, 1].plot(time_hours, dataset['global_operational']['power'], 'm-', linewidth=1)
    axes[1, 1].set_title('Power vs Time')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'time_series.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. EIS data visualization
    if 'eis_data' in dataset and len(dataset['eis_data']['impedance_real']) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Electrochemical Impedance Spectroscopy', fontsize=16)
        
        # Select a few EIS measurements
        n_measurements = min(5, len(dataset['eis_data']['impedance_real']))
        colors = plt.cm.viridis(np.linspace(0, 1, n_measurements))
        
        for i in range(n_measurements):
            Z_real = dataset['eis_data']['impedance_real'][i]
            Z_imag = dataset['eis_data']['impedance_imag'][i]
            
            # Nyquist plot
            axes[0].plot(Z_real, -Z_imag, 'o-', color=colors[i], 
                        label=f'Measurement {i+1}', markersize=3)
            
            # Bode plot
            freq = dataset['eis_data']['frequencies']
            magnitude = np.sqrt(Z_real**2 + Z_imag**2)
            phase = np.angle(Z_real + 1j * Z_imag) * 180 / np.pi
            
            axes[1].semilogx(freq, magnitude, 'o-', color=colors[i], 
                           label=f'Measurement {i+1}', markersize=3)
        
        axes[0].set_xlabel('Real Impedance (Ω)')
        axes[0].set_ylabel('-Imaginary Impedance (Ω)')
        axes[0].set_title('Nyquist Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('|Z| (Ω)')
        axes[1].set_title('Bode Plot - Magnitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'eis_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Experimental visualizations saved to {viz_dir}")

def visualize_monitoring_data(dataset):
    """Visualize real-time monitoring data."""
    print("\n" + "=" * 60)
    print("VISUALIZING MONITORING DATA")
    print("=" * 60)
    
    # Create visualization directory
    viz_dir = Path('visualizations/monitoring')
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    
    # 1. High-frequency data
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Real-Time Monitoring: High-Frequency Data', fontsize=16)
    
    time_hours = dataset['high_freq_time'] / 3600  # Convert to hours
    
    # Voltage
    axes[0, 0].plot(time_hours, dataset['high_frequency_data']['voltage'], 'b-', linewidth=0.5)
    axes[0, 0].set_title('Cell Voltage')
    axes[0, 0].set_xlabel('Time (hours)')
    axes[0, 0].set_ylabel('Voltage (V)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Current
    axes[0, 1].plot(time_hours, dataset['high_frequency_data']['current'], 'r-', linewidth=0.5)
    axes[0, 1].set_title('Current')
    axes[0, 1].set_xlabel('Time (hours)')
    axes[0, 1].set_ylabel('Current (A)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Temperature
    axes[1, 0].plot(time_hours, dataset['high_frequency_data']['inlet_fuel_temp'], 'g-', linewidth=0.5, label='Inlet')
    axes[1, 0].plot(time_hours, dataset['high_frequency_data']['outlet_fuel_temp'], 'g--', linewidth=0.5, label='Outlet')
    axes[1, 0].set_title('Temperature')
    axes[1, 0].set_xlabel('Time (hours)')
    axes[1, 0].set_ylabel('Temperature (K)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power
    axes[1, 1].plot(time_hours, dataset['high_frequency_data']['power'], 'm-', linewidth=0.5)
    axes[1, 1].set_title('Power')
    axes[1, 1].set_xlabel('Time (hours)')
    axes[1, 1].set_ylabel('Power (W)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(viz_dir / 'high_frequency_data.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Adaptive triggers
    if 'adaptive_triggers' in dataset:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Adaptive Monitoring Triggers', fontsize=16)
        
        trigger_types = ['voltage_drop', 'temperature_spike', 'power_anomaly', 'strain_threshold']
        
        for i, trigger_type in enumerate(trigger_types):
            if trigger_type in dataset['adaptive_triggers']:
                triggers = dataset['adaptive_triggers'][trigger_type]
                if triggers:
                    times = [t['time'] for t in triggers]
                    severities = [t['severity'] for t in triggers]
                    
                    axes[i//2, i%2].scatter(times, severities, alpha=0.7, s=50)
                    axes[i//2, i%2].set_title(f'{trigger_type.replace("_", " ").title()} Triggers')
                    axes[i//2, i%2].set_xlabel('Time (s)')
                    axes[i//2, i%2].set_ylabel('Severity')
                    axes[i//2, i%2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'adaptive_triggers.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Monitoring visualizations saved to {viz_dir}")

def main():
    """Main function to generate complete SOFC dataset."""
    print("SOFC Digital Twin Dataset Generation")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    # Create output directories
    Path('datasets').mkdir(exist_ok=True)
    Path('visualizations').mkdir(exist_ok=True)
    
    # Generate datasets
    try:
        # 1. High-fidelity simulation data
        hf_dataset = generate_high_fidelity_dataset()
        
        # 2. Experimental validation data
        exp_dataset = generate_experimental_dataset()
        
        # 3. Real-time monitoring data
        mon_dataset = generate_monitoring_dataset()
        
        # Visualize datasets
        visualize_high_fidelity_data(hf_dataset)
        visualize_experimental_data(exp_dataset)
        visualize_monitoring_data(mon_dataset)
        
        print("\n" + "=" * 60)
        print("DATASET GENERATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"High-fidelity dataset: {len(hf_dataset['parameters'])} samples")
        print(f"Experimental dataset: {len(exp_dataset['time_points'])} time points")
        print(f"Monitoring dataset: {len(mon_dataset['high_freq_time'])} high-freq points")
        print("\nFiles generated:")
        print("- datasets/high_fidelity_simulations.h5")
        print("- datasets/experimental_validation.h5")
        print("- datasets/real_time_monitoring.h5")
        print("- visualizations/ (various plots)")
        
    except Exception as e:
        print(f"Error during dataset generation: {str(e)}")
        logging.error(f"Dataset generation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()