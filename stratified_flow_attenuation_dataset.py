#!/usr/bin/env python3
"""
Stratified Flow Attenuation Mechanisms Dataset Generator
PhD Thesis: "Study on the Attenuation Mechanisms in Stratified Flows: Beyond Single Phase Leakage Acoustics"

This script generates a comprehensive dataset for studying acoustic attenuation
in stratified multiphase flows, including theoretical models, empirical data,
and synthetic experimental results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import minimize
import json
from datetime import datetime
import os

class StratifiedFlowAttenuationDataset:
    """
    Comprehensive dataset generator for stratified flow attenuation mechanisms.
    
    This class generates data covering:
    1. Single-phase acoustic properties
    2. Multiphase flow characteristics
    3. Stratified flow geometry parameters
    4. Attenuation mechanisms (viscous, thermal, scattering)
    5. Frequency-dependent behavior
    6. Temperature and pressure effects
    """
    
    def __init__(self, seed=42):
        """Initialize the dataset generator with random seed for reproducibility."""
        np.random.seed(seed)
        self.data = {}
        self.metadata = {
            'generation_date': datetime.now().isoformat(),
            'description': 'Stratified Flow Attenuation Mechanisms Dataset',
            'version': '1.0',
            'author': 'PhD Thesis Dataset Generator',
            'parameters': {}
        }
    
    def generate_fluid_properties(self, n_samples=1000):
        """
        Generate fluid properties for different phases in stratified flows.
        
        Parameters:
        - n_samples: Number of data points to generate
        
        Returns:
        - DataFrame with fluid properties
        """
        print("Generating fluid properties data...")
        
        # Define fluid types and their properties
        fluids = {
            'water': {
                'density_range': (950, 1050),  # kg/m³
                'viscosity_range': (0.0008, 0.0015),  # Pa·s
                'sound_speed_range': (1400, 1600),  # m/s
                'thermal_conductivity_range': (0.6, 0.7),  # W/(m·K)
                'specific_heat_range': (4180, 4200),  # J/(kg·K)
                'bulk_modulus_range': (2.0e9, 2.3e9)  # Pa
            },
            'oil': {
                'density_range': (800, 950),
                'viscosity_range': (0.001, 0.01),
                'sound_speed_range': (1200, 1500),
                'thermal_conductivity_range': (0.1, 0.2),
                'specific_heat_range': (2000, 2500),
                'bulk_modulus_range': (1.5e9, 2.0e9)
            },
            'gas': {
                'density_range': (0.5, 50),
                'viscosity_range': (1e-5, 2e-5),
                'sound_speed_range': (300, 500),
                'thermal_conductivity_range': (0.01, 0.1),
                'specific_heat_range': (1000, 1500),
                'bulk_modulus_range': (1e5, 2e5)
            }
        }
        
        data = []
        for i in range(n_samples):
            for fluid_type, properties in fluids.items():
                sample = {
                    'sample_id': i,
                    'fluid_type': fluid_type,
                    'temperature': np.random.uniform(273, 373),  # K
                    'pressure': np.random.uniform(1e5, 10e5),  # Pa
                    'density': np.random.uniform(*properties['density_range']),
                    'viscosity': np.random.uniform(*properties['viscosity_range']),
                    'sound_speed': np.random.uniform(*properties['sound_speed_range']),
                    'thermal_conductivity': np.random.uniform(*properties['thermal_conductivity_range']),
                    'specific_heat': np.random.uniform(*properties['specific_heat_range']),
                    'bulk_modulus': np.random.uniform(*properties['bulk_modulus_range'])
                }
                data.append(sample)
        
        df = pd.DataFrame(data)
        self.data['fluid_properties'] = df
        return df
    
    def generate_flow_geometry(self, n_samples=1000):
        """
        Generate stratified flow geometry parameters.
        
        Parameters:
        - n_samples: Number of data points to generate
        
        Returns:
        - DataFrame with flow geometry data
        """
        print("Generating flow geometry data...")
        
        data = []
        for i in range(n_samples):
            # Pipe/duct dimensions
            diameter = np.random.uniform(0.01, 0.5)  # m
            length = np.random.uniform(1, 100)  # m
            
            # Stratified flow parameters
            layer_thickness_ratio = np.random.uniform(0.1, 0.9)  # ratio of heavy phase
            interface_roughness = np.random.uniform(1e-6, 1e-4)  # m
            interface_angle = np.random.uniform(0, 5)  # degrees from horizontal
            
            # Flow velocities
            heavy_phase_velocity = np.random.uniform(0.1, 5.0)  # m/s
            light_phase_velocity = np.random.uniform(0.1, 10.0)  # m/s
            
            # Reynolds numbers
            Re_heavy = np.random.uniform(1000, 100000)
            Re_light = np.random.uniform(1000, 100000)
            
            sample = {
                'sample_id': i,
                'diameter': diameter,
                'length': length,
                'layer_thickness_ratio': layer_thickness_ratio,
                'interface_roughness': interface_roughness,
                'interface_angle': interface_angle,
                'heavy_phase_velocity': heavy_phase_velocity,
                'light_phase_velocity': light_phase_velocity,
                'Re_heavy': Re_heavy,
                'Re_light': Re_light,
                'flow_regime': self._classify_flow_regime(Re_heavy, Re_light, layer_thickness_ratio)
            }
            data.append(sample)
        
        df = pd.DataFrame(data)
        self.data['flow_geometry'] = df
        return df
    
    def _classify_flow_regime(self, Re_heavy, Re_light, thickness_ratio):
        """Classify flow regime based on Reynolds numbers and geometry."""
        if Re_heavy < 2300 and Re_light < 2300:
            return 'laminar_laminar'
        elif Re_heavy > 4000 and Re_light > 4000:
            return 'turbulent_turbulent'
        else:
            return 'mixed_regime'
    
    def generate_acoustic_properties(self, n_samples=1000):
        """
        Generate acoustic properties for different frequencies and conditions.
        
        Parameters:
        - n_samples: Number of data points to generate
        
        Returns:
        - DataFrame with acoustic properties
        """
        print("Generating acoustic properties data...")
        
        data = []
        frequencies = np.logspace(1, 5, 20)  # 10 Hz to 100 kHz
        
        for i in range(n_samples):
            for freq in frequencies:
                # Attenuation mechanisms
                viscous_attenuation = self._calculate_viscous_attenuation(freq)
                thermal_attenuation = self._calculate_thermal_attenuation(freq)
                scattering_attenuation = self._calculate_scattering_attenuation(freq)
                interface_attenuation = self._calculate_interface_attenuation(freq)
                
                total_attenuation = (viscous_attenuation + thermal_attenuation + 
                                   scattering_attenuation + interface_attenuation)
                
                # Acoustic impedance
                impedance_real = np.random.uniform(1e5, 1e7)  # Pa·s/m
                impedance_imag = np.random.uniform(1e4, 1e6)  # Pa·s/m
                
                # Reflection and transmission coefficients
                reflection_coeff = np.random.uniform(0.1, 0.9)
                transmission_coeff = 1 - reflection_coeff
                
                sample = {
                    'sample_id': i,
                    'frequency': freq,
                    'viscous_attenuation': viscous_attenuation,
                    'thermal_attenuation': thermal_attenuation,
                    'scattering_attenuation': scattering_attenuation,
                    'interface_attenuation': interface_attenuation,
                    'total_attenuation': total_attenuation,
                    'impedance_real': impedance_real,
                    'impedance_imag': impedance_imag,
                    'reflection_coefficient': reflection_coeff,
                    'transmission_coefficient': transmission_coeff,
                    'wavelength': 1500 / freq,  # assuming sound speed ~1500 m/s
                    'acoustic_power_level': np.random.uniform(60, 120)  # dB
                }
                data.append(sample)
        
        df = pd.DataFrame(data)
        self.data['acoustic_properties'] = df
        return df
    
    def _calculate_viscous_attenuation(self, frequency):
        """Calculate viscous attenuation coefficient."""
        # Stokes-Kirchhoff formula
        alpha_v = 2 * np.pi**2 * frequency**2 / (3 * 1500**3) * 1e-6  # simplified
        return alpha_v * np.random.uniform(0.5, 2.0)  # add variability
    
    def _calculate_thermal_attenuation(self, frequency):
        """Calculate thermal attenuation coefficient."""
        # Thermal relaxation effects
        alpha_t = 2 * np.pi**2 * frequency**2 / (1500**3) * 1e-7  # simplified
        return alpha_t * np.random.uniform(0.3, 1.5)
    
    def _calculate_scattering_attenuation(self, frequency):
        """Calculate scattering attenuation coefficient."""
        # Rayleigh scattering approximation
        alpha_s = frequency**4 * 1e-12  # simplified
        return alpha_s * np.random.uniform(0.1, 1.0)
    
    def _calculate_interface_attenuation(self, frequency):
        """Calculate interface-related attenuation coefficient."""
        # Interface scattering and mode conversion
        alpha_i = frequency**1.5 * 1e-8  # simplified
        return alpha_i * np.random.uniform(0.2, 2.0)
    
    def generate_experimental_conditions(self, n_samples=1000):
        """
        Generate experimental conditions and measurement parameters.
        
        Parameters:
        - n_samples: Number of data points to generate
        
        Returns:
        - DataFrame with experimental conditions
        """
        print("Generating experimental conditions data...")
        
        data = []
        for i in range(n_samples):
            # Environmental conditions
            temperature = np.random.uniform(273, 373)  # K
            pressure = np.random.uniform(1e5, 10e5)  # Pa
            humidity = np.random.uniform(0, 100)  # %
            
            # Measurement setup
            transducer_distance = np.random.uniform(0.1, 2.0)  # m
            transducer_frequency = np.random.uniform(1e3, 100e3)  # Hz
            measurement_duration = np.random.uniform(1, 3600)  # s
            
            # Signal processing parameters
            sampling_rate = np.random.uniform(44.1e3, 192e3)  # Hz
            window_size = np.random.choice([1024, 2048, 4096, 8192])
            overlap_ratio = np.random.uniform(0.5, 0.95)
            
            # Noise characteristics
            signal_to_noise_ratio = np.random.uniform(10, 60)  # dB
            background_noise_level = np.random.uniform(30, 80)  # dB
            
            sample = {
                'sample_id': i,
                'temperature': temperature,
                'pressure': pressure,
                'humidity': humidity,
                'transducer_distance': transducer_distance,
                'transducer_frequency': transducer_frequency,
                'measurement_duration': measurement_duration,
                'sampling_rate': sampling_rate,
                'window_size': window_size,
                'overlap_ratio': overlap_ratio,
                'signal_to_noise_ratio': signal_to_noise_ratio,
                'background_noise_level': background_noise_level,
                'measurement_uncertainty': np.random.uniform(0.01, 0.1)  # relative uncertainty
            }
            data.append(sample)
        
        df = pd.DataFrame(data)
        self.data['experimental_conditions'] = df
        return df
    
    def generate_theoretical_models(self):
        """
        Generate theoretical model predictions for comparison.
        
        Returns:
        - DataFrame with theoretical model results
        """
        print("Generating theoretical model data...")
        
        # Frequency range
        frequencies = np.logspace(1, 5, 50)
        
        data = []
        for freq in frequencies:
            # Classical attenuation models
            stokes_kirchhoff = self._stokes_kirchhoff_model(freq)
            navier_stokes = self._navier_stokes_model(freq)
            thermoacoustic = self._thermoacoustic_model(freq)
            multiphase = self._multiphase_model(freq)
            
            # Advanced models
            interface_scattering = self._interface_scattering_model(freq)
            mode_conversion = self._mode_conversion_model(freq)
            
            sample = {
                'frequency': freq,
                'stokes_kirchhoff_attenuation': stokes_kirchhoff,
                'navier_stokes_attenuation': navier_stokes,
                'thermoacoustic_attenuation': thermoacoustic,
                'multiphase_attenuation': multiphase,
                'interface_scattering_attenuation': interface_scattering,
                'mode_conversion_attenuation': mode_conversion,
                'wavelength': 1500 / freq
            }
            data.append(sample)
        
        df = pd.DataFrame(data)
        self.data['theoretical_models'] = df
        return df
    
    def _stokes_kirchhoff_model(self, frequency):
        """Stokes-Kirchhoff classical attenuation model."""
        return 2 * np.pi**2 * frequency**2 / (3 * 1500**3) * 1e-6
    
    def _navier_stokes_model(self, frequency):
        """Navier-Stokes based attenuation model."""
        return 2 * np.pi**2 * frequency**2 / (1500**3) * 1e-6 * 1.5
    
    def _thermoacoustic_model(self, frequency):
        """Thermoacoustic attenuation model."""
        return 2 * np.pi**2 * frequency**2 / (1500**3) * 1e-7 * 2.0
    
    def _multiphase_model(self, frequency):
        """Multiphase flow attenuation model."""
        return frequency**1.5 * 1e-8 * 3.0
    
    def _interface_scattering_model(self, frequency):
        """Interface scattering model."""
        return frequency**2 * 1e-9 * 2.5
    
    def _mode_conversion_model(self, frequency):
        """Mode conversion attenuation model."""
        return frequency**1.8 * 1e-9 * 1.8
    
    def generate_correlation_data(self):
        """
        Generate correlation analysis data between different parameters.
        
        Returns:
        - DataFrame with correlation data
        """
        print("Generating correlation analysis data...")
        
        # Generate correlated parameters
        n_samples = 1000
        base_frequency = np.random.uniform(1e3, 100e3, n_samples)
        
        # Correlate attenuation with various parameters
        attenuation = (base_frequency**1.5 * 1e-8 * 
                      np.random.uniform(0.5, 2.0, n_samples))
        
        # Temperature effect
        temperature = np.random.uniform(273, 373, n_samples)
        temp_effect = 1 + 0.001 * (temperature - 298)
        
        # Pressure effect
        pressure = np.random.uniform(1e5, 10e5, n_samples)
        pressure_effect = 1 + 0.0001 * (pressure - 1e5) / 1e5
        
        # Viscosity effect
        viscosity = np.random.uniform(1e-6, 1e-3, n_samples)
        viscosity_effect = viscosity**0.5 * 1e3
        
        # Combined effects
        total_attenuation = (attenuation * temp_effect * pressure_effect * 
                           viscosity_effect * np.random.uniform(0.8, 1.2, n_samples))
        
        data = {
            'sample_id': range(n_samples),
            'frequency': base_frequency,
            'temperature': temperature,
            'pressure': pressure,
            'viscosity': viscosity,
            'attenuation': total_attenuation,
            'temp_effect': temp_effect,
            'pressure_effect': pressure_effect,
            'viscosity_effect': viscosity_effect
        }
        
        df = pd.DataFrame(data)
        self.data['correlation_data'] = df
        return df
    
    def save_dataset(self, output_dir='stratified_flow_dataset'):
        """
        Save the complete dataset to files.
        
        Parameters:
        - output_dir: Directory to save the dataset
        """
        print(f"Saving dataset to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save individual datasets
        for name, df in self.data.items():
            filename = os.path.join(output_dir, f'{name}.csv')
            df.to_csv(filename, index=False)
            print(f"Saved {name}: {len(df)} samples")
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save combined dataset
        combined_file = os.path.join(output_dir, 'combined_dataset.csv')
        if len(self.data) > 0:
            # Merge all datasets on sample_id where applicable
            combined_df = list(self.data.values())[0]
            for name, df in list(self.data.items())[1:]:
                if 'sample_id' in df.columns:
                    combined_df = combined_df.merge(df, on='sample_id', how='outer')
                else:
                    # For datasets without sample_id, add as additional columns
                    for col in df.columns:
                        if col not in combined_df.columns:
                            combined_df[col] = df[col].iloc[0] if len(df) == 1 else np.nan
            
            combined_df.to_csv(combined_file, index=False)
            print(f"Saved combined dataset: {len(combined_df)} samples")
        
        print(f"Dataset saved successfully to {output_dir}/")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for the dataset."""
        print("\n=== DATASET SUMMARY STATISTICS ===")
        
        for name, df in self.data.items():
            print(f"\n{name.upper()}:")
            print(f"  Samples: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show basic statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  Numeric columns: {list(numeric_cols)}")
                print(f"  Basic statistics:")
                print(df[numeric_cols].describe().round(4))
    
    def create_visualizations(self, output_dir='stratified_flow_dataset/plots'):
        """Create visualization plots for the dataset."""
        print(f"Creating visualizations in {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Attenuation vs Frequency
        if 'acoustic_properties' in self.data:
            df = self.data['acoustic_properties']
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.loglog(df['frequency'], df['total_attenuation'], 'b.', alpha=0.6)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Total Attenuation (Np/m)')
            plt.title('Total Attenuation vs Frequency')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.semilogx(df['frequency'], df['viscous_attenuation'], 'r-', label='Viscous')
            plt.semilogx(df['frequency'], df['thermal_attenuation'], 'g-', label='Thermal')
            plt.semilogx(df['frequency'], df['scattering_attenuation'], 'b-', label='Scattering')
            plt.semilogx(df['frequency'], df['interface_attenuation'], 'm-', label='Interface')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Attenuation (Np/m)')
            plt.title('Attenuation Mechanisms')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.hist(df['total_attenuation'], bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Total Attenuation (Np/m)')
            plt.ylabel('Frequency')
            plt.title('Attenuation Distribution')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.scatter(df['frequency'], df['acoustic_power_level'], 
                       c=df['total_attenuation'], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Attenuation (Np/m)')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Acoustic Power Level (dB)')
            plt.title('Power Level vs Frequency')
            plt.xscale('log')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'attenuation_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Flow Geometry Analysis
        if 'flow_geometry' in self.data:
            df = self.data['flow_geometry']
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.scatter(df['diameter'], df['layer_thickness_ratio'], 
                       c=df['Re_heavy'], cmap='plasma', alpha=0.6)
            plt.colorbar(label='Reynolds Number (Heavy)')
            plt.xlabel('Diameter (m)')
            plt.ylabel('Layer Thickness Ratio')
            plt.title('Flow Geometry')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.hist(df['flow_regime'], bins=20, alpha=0.7, edgecolor='black')
            plt.xlabel('Flow Regime')
            plt.ylabel('Count')
            plt.title('Flow Regime Distribution')
            plt.xticks(rotation=45)
            plt.grid(True)
            
            plt.subplot(2, 2, 3)
            plt.scatter(df['heavy_phase_velocity'], df['light_phase_velocity'], 
                       c=df['interface_roughness'], cmap='viridis', alpha=0.6)
            plt.colorbar(label='Interface Roughness (m)')
            plt.xlabel('Heavy Phase Velocity (m/s)')
            plt.ylabel('Light Phase Velocity (m/s)')
            plt.title('Velocity Relationship')
            plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.loglog(df['Re_heavy'], df['Re_light'], 'b.', alpha=0.6)
            plt.xlabel('Reynolds Number (Heavy Phase)')
            plt.ylabel('Reynolds Number (Light Phase)')
            plt.title('Reynolds Number Correlation')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'flow_geometry_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Theoretical Model Comparison
        if 'theoretical_models' in self.data:
            df = self.data['theoretical_models']
            plt.figure(figsize=(12, 8))
            
            plt.loglog(df['frequency'], df['stokes_kirchhoff_attenuation'], 
                      'r-', label='Stokes-Kirchhoff', linewidth=2)
            plt.loglog(df['frequency'], df['navier_stokes_attenuation'], 
                      'g-', label='Navier-Stokes', linewidth=2)
            plt.loglog(df['frequency'], df['thermoacoustic_attenuation'], 
                      'b-', label='Thermoacoustic', linewidth=2)
            plt.loglog(df['frequency'], df['multiphase_attenuation'], 
                      'm-', label='Multiphase', linewidth=2)
            plt.loglog(df['frequency'], df['interface_scattering_attenuation'], 
                      'c-', label='Interface Scattering', linewidth=2)
            plt.loglog(df['frequency'], df['mode_conversion_attenuation'], 
                      'y-', label='Mode Conversion', linewidth=2)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Attenuation (Np/m)')
            plt.title('Theoretical Model Comparison')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'theoretical_models.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}/")

def main():
    """Main function to generate the complete dataset."""
    print("=== Stratified Flow Attenuation Mechanisms Dataset Generator ===")
    print("PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows")
    print("Beyond Single Phase Leakage Acoustics\n")
    
    # Initialize dataset generator
    dataset = StratifiedFlowAttenuationDataset(seed=42)
    
    # Generate all components of the dataset
    print("Generating comprehensive dataset...")
    dataset.generate_fluid_properties(n_samples=1000)
    dataset.generate_flow_geometry(n_samples=1000)
    dataset.generate_acoustic_properties(n_samples=1000)
    dataset.generate_experimental_conditions(n_samples=1000)
    dataset.generate_theoretical_models()
    dataset.generate_correlation_data()
    
    # Generate summary statistics
    dataset.generate_summary_statistics()
    
    # Create visualizations
    dataset.create_visualizations()
    
    # Save the complete dataset
    dataset.save_dataset()
    
    print("\n=== Dataset Generation Complete ===")
    print("The dataset includes:")
    print("- Fluid properties for water, oil, and gas phases")
    print("- Stratified flow geometry parameters")
    print("- Acoustic properties and attenuation mechanisms")
    print("- Experimental conditions and measurement parameters")
    print("- Theoretical model predictions")
    print("- Correlation analysis data")
    print("\nFiles saved in 'stratified_flow_dataset/' directory")
    print("Use the generated CSV files and visualizations for your PhD research!")

if __name__ == "__main__":
    main()