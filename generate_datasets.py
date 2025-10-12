#!/usr/bin/env python3
"""
Memory-efficient Stratified Flow Dataset Generator
"""

import numpy as np
import pandas as pd
import os

def create_output_directory():
    """Create output directory for datasets"""
    output_dir = 'stratified_flow_datasets'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def generate_frequency_domain_data():
    """Generate frequency-domain attenuation data"""
    print("Generating frequency-domain data...")
    
    n_samples = 2000
    frequencies = np.logspace(1, 5, n_samples)
    flow_configs = ['horizontal_stratified', 'inclined_stratified', 'wavy_interface', 
                   'slug_flow', 'annular_flow', 'dispersed_flow']
    
    data = []
    
    for i, freq in enumerate(frequencies):
        config = flow_configs[i % len(flow_configs)]
        
        # Generate realistic parameters based on flow configuration
        if config == 'horizontal_stratified':
            gas_fraction = np.random.uniform(0.3, 0.8)
            interface_roughness = np.random.uniform(0.1, 2.0)
            attenuation = 10 * np.log10(1 + (freq / 1000)**0.5) + np.random.normal(0, 2)
            
        elif config == 'wavy_interface':
            gas_fraction = np.random.uniform(0.2, 0.9)
            interface_roughness = np.random.uniform(0.5, 5.0)
            attenuation = 15 * np.log10(1 + (freq / 800)**0.6) + np.random.normal(0, 3)
            
        elif config == 'slug_flow':
            gas_fraction = np.random.uniform(0.6, 0.95)
            interface_roughness = np.random.uniform(2.0, 10.0)
            attenuation = 25 * np.log10(1 + (freq / 500)**0.4) + np.random.normal(0, 5)
            
        elif config == 'annular_flow':
            gas_fraction = np.random.uniform(0.8, 0.98)
            interface_roughness = np.random.uniform(0.1, 1.0)
            attenuation = 20 * np.log10(1 + (freq / 1500)**0.3) + np.random.normal(0, 3)
            
        elif config == 'dispersed_flow':
            gas_fraction = np.random.uniform(0.1, 0.6)
            interface_roughness = np.random.uniform(0.01, 0.5)
            attenuation = 12 * np.log10(1 + (freq / 2000)**0.7) + np.random.normal(0, 2)
            
        else:  # inclined_stratified
            gas_fraction = np.random.uniform(0.4, 0.9)
            interface_roughness = np.random.uniform(0.2, 3.0)
            attenuation = 8 * np.log10(1 + (freq / 1200)**0.5) + np.random.normal(0, 2)
        
        attenuation = np.clip(attenuation, 0.1, 150)
        
        data.append({
            'frequency_hz': freq,
            'flow_configuration': config,
            'attenuation_db_per_m': attenuation,
            'gas_fraction': gas_fraction,
            'liquid_fraction': 1 - gas_fraction,
            'interface_roughness_mm': interface_roughness,
            'reynolds_number_gas': np.random.uniform(1000, 50000),
            'reynolds_number_liquid': np.random.uniform(100, 10000),
            'weber_number': np.random.uniform(0.1, 100),
            'froude_number': np.random.uniform(0.1, 10),
            'temperature_c': np.random.uniform(10, 80),
            'pressure_bar': np.random.uniform(1, 50)
        })
    
    return pd.DataFrame(data)

def generate_experimental_conditions():
    """Generate experimental conditions dataset"""
    print("Generating experimental conditions...")
    
    n_experiments = 500
    data = []
    
    for exp_id in range(n_experiments):
        # Pipe geometry
        pipe_diameter = np.random.choice([0.025, 0.05, 0.1, 0.15, 0.2, 0.3])
        pipe_length = np.random.uniform(1.0, 10.0)
        inclination_angle = np.random.uniform(-10, 45)
        
        # Flow conditions
        gas_velocity = np.random.uniform(0.1, 20.0)
        liquid_velocity = np.random.uniform(0.01, 5.0)
        
        # Environmental conditions
        temperature = np.random.uniform(15, 80)
        pressure = np.random.uniform(1, 30)
        
        # Flow pattern prediction
        if liquid_velocity < 0.1 and gas_velocity < 3:
            flow_pattern = 'stratified'
        elif liquid_velocity < 0.3 and gas_velocity > 3:
            flow_pattern = 'wavy'
        elif liquid_velocity > 0.3 and gas_velocity < 5:
            flow_pattern = 'slug'
        elif liquid_velocity > 1.0:
            flow_pattern = 'dispersed'
        else:
            flow_pattern = 'annular'
        
        data.append({
            'experiment_id': f"EXP_{exp_id:04d}",
            'pipe_diameter_m': pipe_diameter,
            'pipe_length_m': pipe_length,
            'inclination_angle_deg': inclination_angle,
            'gas_superficial_velocity_ms': gas_velocity,
            'liquid_superficial_velocity_ms': liquid_velocity,
            'temperature_c': temperature,
            'pressure_bar': pressure,
            'predicted_flow_pattern': flow_pattern,
            'transducer_frequency_hz': np.random.choice([0.5, 1.0, 2.25, 5.0, 10.0]) * 1e6,
            'measurement_distance_m': np.random.uniform(0.1, 2.0),
            'measurement_angle_deg': np.random.uniform(0, 90),
            'ambient_temperature_c': np.random.uniform(18, 35),
            'humidity_percent': np.random.uniform(30, 80)
        })
    
    return pd.DataFrame(data)

def generate_attenuation_models():
    """Generate theoretical attenuation models data"""
    print("Generating attenuation models...")
    
    n_samples = 1500
    models = ['rayleigh_scattering', 'mie_scattering', 'viscous_losses', 
             'thermal_losses', 'interface_scattering', 'mode_conversion']
    
    data = []
    
    for i in range(n_samples):
        model = models[i % len(models)]
        frequency = np.random.uniform(100, 100000)
        temperature = np.random.uniform(10, 80)
        pressure = np.random.uniform(1, 20)
        
        if model == 'rayleigh_scattering':
            particle_radius = np.random.uniform(1e-6, 1e-4)
            attenuation = 5 * (frequency / 1000)**2 * (particle_radius * 1e6)**4
            
        elif model == 'mie_scattering':
            particle_radius = np.random.uniform(1e-4, 1e-2)
            attenuation = 15 * np.log10(1 + frequency / 1000) * (particle_radius * 1000)
            
        elif model == 'viscous_losses':
            viscosity = np.random.uniform(1e-5, 1e-1)
            attenuation = 0.1 * frequency**2 * viscosity * 1e6
            
        elif model == 'thermal_losses':
            thermal_conductivity = np.random.uniform(0.1, 2.0)
            attenuation = 2 * np.sqrt(frequency) * thermal_conductivity
            
        elif model == 'interface_scattering':
            roughness = np.random.uniform(1e-6, 1e-3)
            attenuation = 20 * (frequency / 1000) * (roughness * 1e6)**2
            
        else:  # mode_conversion
            impedance_contrast = np.random.uniform(0.1, 10.0)
            attenuation = 10 * np.log10(impedance_contrast) * (frequency / 10000)**0.5
        
        attenuation = np.clip(attenuation * np.random.uniform(0.8, 1.2), 0.001, 500)
        
        data.append({
            'model_type': model,
            'frequency_hz': frequency,
            'temperature_c': temperature,
            'pressure_bar': pressure,
            'predicted_attenuation_db_per_m': attenuation,
            'particle_radius_m': particle_radius if 'particle_radius' in locals() else np.nan,
            'viscosity_pas': viscosity if 'viscosity' in locals() else np.nan,
            'roughness_m': roughness if 'roughness' in locals() else np.nan
        })
    
    return pd.DataFrame(data)

def generate_multiphase_flow_data():
    """Generate multiphase flow characterization data"""
    print("Generating multiphase flow data...")
    
    n_samples = 1000
    data = []
    
    flow_regimes = ['bubble', 'slug', 'churn', 'annular', 'stratified', 'wavy']
    
    for i in range(n_samples):
        flow_regime = flow_regimes[i % len(flow_regimes)]
        
        if flow_regime == 'bubble':
            gas_fraction = np.random.uniform(0.01, 0.3)
            bubble_diameter = np.random.uniform(0.001, 0.01)
            
        elif flow_regime == 'slug':
            gas_fraction = np.random.uniform(0.2, 0.8)
            bubble_diameter = np.random.uniform(0.005, 0.05)
            
        elif flow_regime == 'churn':
            gas_fraction = np.random.uniform(0.6, 0.9)
            bubble_diameter = np.random.uniform(0.01, 0.1)
            
        elif flow_regime == 'annular':
            gas_fraction = np.random.uniform(0.8, 0.99)
            bubble_diameter = np.nan
            
        else:  # stratified or wavy
            gas_fraction = np.random.uniform(0.3, 0.9)
            bubble_diameter = np.nan
        
        # Calculate mixture properties
        rho_water, rho_air = 1000, 1.225
        c_water, c_air = 1480, 343
        
        mixture_density = gas_fraction * rho_air + (1 - gas_fraction) * rho_water
        
        # Simplified mixture sound speed
        if gas_fraction < 0.1:
            mixture_sound_speed = c_water * (1 - 0.5 * gas_fraction)
        else:
            mixture_sound_speed = c_air * (1 + 2 * (1 - gas_fraction))
        
        # Calculate attenuation at 1 kHz
        frequency_test = 1000
        
        if not np.isnan(bubble_diameter):
            scattering_attenuation = 10 * (bubble_diameter * 1000)**2 * gas_fraction
        else:
            scattering_attenuation = 2 * gas_fraction
        
        viscous_attenuation = 0.5 * frequency_test / 1000
        interface_attenuation = 3 * gas_fraction * (1 - gas_fraction)
        
        total_attenuation = scattering_attenuation + viscous_attenuation + interface_attenuation
        
        data.append({
            'sample_id': f"MF_{i:04d}",
            'flow_regime': flow_regime,
            'gas_fraction': gas_fraction,
            'liquid_fraction': 1 - gas_fraction,
            'bubble_diameter_m': bubble_diameter,
            'mixture_density_kgm3': mixture_density,
            'mixture_sound_speed_ms': mixture_sound_speed,
            'total_attenuation_db_per_m_at_1khz': total_attenuation,
            'scattering_component_db_per_m': scattering_attenuation,
            'viscous_component_db_per_m': viscous_attenuation,
            'interface_component_db_per_m': interface_attenuation,
            'interface_area_density_m2m3': np.random.uniform(10, 5000),
            'turbulent_kinetic_energy_m2s2': np.random.uniform(0.001, 1.0)
        })
    
    return pd.DataFrame(data)

def generate_time_series_features():
    """Generate time-series acoustic features"""
    print("Generating time-series features...")
    
    n_samples = 800
    data = []
    
    flow_patterns = ['smooth_stratified', 'wavy_stratified', 'slug_intermittent', 
                    'churn_turbulent', 'annular_dispersed']
    
    for i in range(n_samples):
        pattern = flow_patterns[i % len(flow_patterns)]
        
        # Generate characteristic features for each pattern
        if pattern == 'smooth_stratified':
            rms_amplitude = np.random.uniform(0.1, 0.3)
            dominant_frequency = np.random.uniform(50, 200)
            zero_crossings_rate = np.random.uniform(100, 400)
            spectral_centroid = np.random.uniform(150, 500)
            
        elif pattern == 'wavy_stratified':
            rms_amplitude = np.random.uniform(0.2, 0.6)
            dominant_frequency = np.random.uniform(100, 500)
            zero_crossings_rate = np.random.uniform(200, 800)
            spectral_centroid = np.random.uniform(300, 1000)
            
        elif pattern == 'slug_intermittent':
            rms_amplitude = np.random.uniform(0.5, 1.5)
            dominant_frequency = np.random.uniform(200, 1000)
            zero_crossings_rate = np.random.uniform(400, 1200)
            spectral_centroid = np.random.uniform(500, 2000)
            
        elif pattern == 'churn_turbulent':
            rms_amplitude = np.random.uniform(0.8, 2.0)
            dominant_frequency = np.random.uniform(300, 2000)
            zero_crossings_rate = np.random.uniform(600, 2000)
            spectral_centroid = np.random.uniform(800, 3000)
            
        else:  # annular_dispersed
            rms_amplitude = np.random.uniform(0.3, 1.0)
            dominant_frequency = np.random.uniform(500, 3000)
            zero_crossings_rate = np.random.uniform(800, 2500)
            spectral_centroid = np.random.uniform(1000, 4000)
        
        data.append({
            'sample_id': f"TS_{i:04d}",
            'flow_pattern': pattern,
            'rms_amplitude': rms_amplitude,
            'peak_amplitude': rms_amplitude * np.random.uniform(1.5, 3.0),
            'dominant_frequency_hz': dominant_frequency,
            'spectral_centroid_hz': spectral_centroid,
            'zero_crossings_per_sec': zero_crossings_rate,
            'spectral_rolloff_hz': spectral_centroid * np.random.uniform(1.2, 2.0),
            'spectral_bandwidth_hz': np.random.uniform(100, 1000),
            'mfcc_1': np.random.normal(0, 1),
            'mfcc_2': np.random.normal(0, 1),
            'mfcc_3': np.random.normal(0, 1)
        })
    
    return pd.DataFrame(data)

def main():
    """Main function to generate all datasets"""
    print("Stratified Flow Attenuation Dataset Generator")
    print("=" * 50)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Generate datasets
    datasets = {
        'frequency_domain': generate_frequency_domain_data(),
        'experimental_conditions': generate_experimental_conditions(),
        'attenuation_models': generate_attenuation_models(),
        'multiphase_flow': generate_multiphase_flow_data(),
        'time_series_features': generate_time_series_features()
    }
    
    # Save datasets
    print(f"\nSaving datasets to {output_dir}/...")
    
    total_samples = 0
    for dataset_name, dataset in datasets.items():
        # Save as CSV
        csv_path = os.path.join(output_dir, f"{dataset_name}.csv")
        dataset.to_csv(csv_path, index=False)
        print(f"✅ {dataset_name}: {len(dataset)} samples -> {csv_path}")
        total_samples += len(dataset)
    
    # Create dataset summary
    summary_path = os.path.join(output_dir, "dataset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Stratified Flow Attenuation Dataset Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write("PhD Research Topic:\n")
        f.write("'Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics'\n\n")
        f.write("Generated Datasets:\n")
        
        for dataset_name, dataset in datasets.items():
            f.write(f"\n{dataset_name.upper()}:\n")
            f.write(f"  - Samples: {len(dataset)}\n")
            f.write(f"  - Features: {len(dataset.columns)}\n")
            f.write(f"  - Description: {get_dataset_description(dataset_name)}\n")
    
    print(f"\n🎉 Dataset generation completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Total samples: {total_samples}")
    print(f"📋 Summary: {summary_path}")

def get_dataset_description(dataset_name):
    """Get description for each dataset"""
    descriptions = {
        'frequency_domain': 'Frequency-dependent attenuation measurements across different flow configurations',
        'experimental_conditions': 'Experimental setup parameters and flow conditions',
        'attenuation_models': 'Theoretical attenuation predictions from various physical models',
        'multiphase_flow': 'Multiphase flow characterization with acoustic properties',
        'time_series_features': 'Time-domain acoustic signal features for flow pattern recognition'
    }
    return descriptions.get(dataset_name, 'Dataset description')

if __name__ == "__main__":
    main()