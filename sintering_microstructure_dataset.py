#!/usr/bin/env python3
"""
Process & Microstructure Dataset Generator for Sintering Optimization
=====================================================================

This script generates a comprehensive dataset linking sintering process parameters
to resulting microstructure characteristics. The data is designed to support
machine learning models for sintering process optimization.

Dataset Structure:
- Sintering Parameters (Inputs): Temperature profiles, pressure, atmosphere, green body characteristics
- Microstructure Outputs: Grain size, porosity, density, pore distribution, grain boundaries

Author: AI Assistant
Date: October 8, 2025
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

class SinteringDatasetGenerator:
    """
    Generates realistic sintering process and microstructure data based on
    physical relationships and empirical correlations from ceramic processing literature.
    """
    
    def __init__(self, n_samples=500):
        self.n_samples = n_samples
        self.data = {}
        
    def generate_sintering_parameters(self):
        """Generate realistic sintering parameter combinations."""
        
        # Temperature Profile Parameters
        # Ramp-up rate: 1-10°C/min (typical for ceramics)
        ramp_up_rate = np.random.uniform(1, 10, self.n_samples)
        
        # Peak sintering temperature: 1200-1600°C (typical for advanced ceramics)
        peak_temperature = np.random.uniform(1200, 1600, self.n_samples)
        
        # Hold time at peak temperature: 0.5-8 hours
        hold_time = np.random.uniform(0.5, 8, self.n_samples)
        
        # Cool-down rate: 2-15°C/min
        cool_down_rate = np.random.uniform(2, 15, self.n_samples)
        
        # Applied Pressure (for pressure-assisted sintering)
        # 0 = pressureless, 10-100 MPa for pressure-assisted
        pressure_type = np.random.choice(['pressureless', 'pressure_assisted'], 
                                       self.n_samples, p=[0.6, 0.4])
        applied_pressure = np.where(pressure_type == 'pressureless', 
                                  0, 
                                  np.random.uniform(10, 100, self.n_samples))
        
        # Green Body Characteristics
        # Initial relative density: 50-70% (typical for pressed powders)
        initial_density = np.random.uniform(50, 70, self.n_samples)
        
        # Initial pore size distribution (log-normal)
        # Mean pore size in green body: 0.1-2.0 μm
        initial_pore_size_mean = np.random.lognormal(np.log(0.5), 0.5, self.n_samples)
        initial_pore_size_std = initial_pore_size_mean * np.random.uniform(0.3, 0.8, self.n_samples)
        
        # Particle size of starting powder: 0.1-5.0 μm
        particle_size = np.random.lognormal(np.log(1.0), 0.6, self.n_samples)
        
        # Atmosphere conditions
        atmosphere = np.random.choice(['air', 'nitrogen', 'argon', 'vacuum'], 
                                    self.n_samples, p=[0.4, 0.3, 0.2, 0.1])
        
        # Oxygen partial pressure (relevant for atmosphere)
        oxygen_pressure = np.where(atmosphere == 'air', 0.21,
                          np.where(atmosphere == 'nitrogen', np.random.uniform(1e-6, 1e-3, self.n_samples),
                          np.where(atmosphere == 'argon', np.random.uniform(1e-8, 1e-5, self.n_samples),
                                 np.random.uniform(1e-10, 1e-7, self.n_samples))))  # vacuum
        
        return {
            'ramp_up_rate_C_per_min': ramp_up_rate,
            'peak_temperature_C': peak_temperature,
            'hold_time_hours': hold_time,
            'cool_down_rate_C_per_min': cool_down_rate,
            'applied_pressure_MPa': applied_pressure,
            'pressure_type': pressure_type,
            'initial_relative_density_percent': initial_density,
            'initial_pore_size_mean_um': initial_pore_size_mean,
            'initial_pore_size_std_um': initial_pore_size_std,
            'particle_size_um': particle_size,
            'atmosphere': atmosphere,
            'oxygen_partial_pressure_atm': oxygen_pressure
        }
    
    def calculate_microstructure_properties(self, sintering_params):
        """
        Calculate resulting microstructure properties based on sintering parameters.
        Uses empirical relationships and physical models from ceramic processing literature.
        """
        
        n = self.n_samples
        
        # Extract key parameters for calculations
        T_peak = sintering_params['peak_temperature_C']
        t_hold = sintering_params['hold_time_hours']
        P_applied = sintering_params['applied_pressure_MPa']
        rho_initial = sintering_params['initial_relative_density_percent']
        d_particle = sintering_params['particle_size_um']
        
        # 1. FINAL RELATIVE DENSITY
        # Based on sintering kinetics and pressure effects
        # Higher temperature, longer time, and pressure increase density
        
        # Temperature effect (Arrhenius-type)
        T_norm = (T_peak - 1200) / 400  # Normalize temperature
        temp_factor = 1 + 0.4 * T_norm
        
        # Time effect (logarithmic)
        time_factor = 1 + 0.15 * np.log(t_hold + 0.1)
        
        # Pressure effect
        pressure_factor = 1 + 0.002 * P_applied
        
        # Particle size effect (smaller particles sinter better)
        particle_factor = 1 / (1 + 0.1 * d_particle)
        
        # Calculate final density with some noise
        density_base = rho_initial + (100 - rho_initial) * 0.7 * temp_factor * time_factor * pressure_factor * particle_factor
        final_density = np.clip(density_base + np.random.normal(0, 2, n), 
                               rho_initial, 99.5)
        
        # 2. POROSITY
        porosity = 100 - final_density
        
        # 3. GRAIN SIZE
        # Grain growth follows power law with temperature and time
        # Direct temperature and time dependence
        
        # Base grain size related to particle size
        grain_size_base = d_particle * (1.2 + 0.3 * np.random.random(n))
        
        # Strong temperature effect on grain growth
        T_norm = (T_peak - 1200) / 400  # Normalize temperature range (0 to 1)
        time_norm = np.log(t_hold + 0.1) / np.log(8.1)  # Normalize time (0 to 1)
        
        # Grain growth with strong temperature dependence
        growth_multiplier = 1 + 3 * T_norm + 1.5 * time_norm + 0.5 * T_norm * time_norm
        grain_size = grain_size_base * growth_multiplier
        
        # Add some scatter
        grain_size *= (1 + np.random.normal(0, 0.15, n))
        grain_size = np.clip(grain_size, d_particle, 50)  # Physical limits
        
        # 4. GRAIN SIZE DISTRIBUTION
        # Standard deviation as fraction of mean grain size
        grain_size_std = grain_size * np.random.uniform(0.2, 0.6, n)
        
        # 5. PORE SIZE DISTRIBUTION
        # Pores shrink during sintering but distribution changes
        pore_size_mean = sintering_params['initial_pore_size_mean_um'] * (porosity / 30)**0.5
        pore_size_std = pore_size_mean * np.random.uniform(0.4, 1.2, n)
        
        # 6. GRAIN BOUNDARY CHARACTERISTICS
        # Grain boundary area per unit volume (related to grain size)
        gb_area_per_volume = 3 / grain_size  # Simplified geometric relationship
        
        # Grain boundary thickness (typically 0.5-2 nm)
        gb_thickness_nm = np.random.uniform(0.5, 2.0, n)
        
        # Grain boundary energy (affected by atmosphere and impurities)
        gb_energy_base = np.random.uniform(0.5, 1.5, n)  # J/m²
        
        # Atmosphere effects on grain boundary energy
        atmosphere_effect = np.where(sintering_params['atmosphere'] == 'air', 1.0,
                            np.where(sintering_params['atmosphere'] == 'nitrogen', 0.95,
                            np.where(sintering_params['atmosphere'] == 'argon', 0.90, 0.85)))
        
        gb_energy = gb_energy_base * atmosphere_effect
        
        # 7. ADDITIONAL MICROSTRUCTURAL FEATURES
        
        # Coordination number (number of grain contacts per grain)
        coordination_number = 8 + 4 * (final_density - 70) / 30  # Increases with density
        coordination_number = np.clip(coordination_number + np.random.normal(0, 0.5, n), 6, 14)
        
        # Pore connectivity (fraction of connected porosity)
        # Higher porosity tends to be more connected
        pore_connectivity = np.clip(0.1 + 0.8 * (porosity / 20)**0.5 + np.random.normal(0, 0.1, n), 0, 1)
        
        return {
            'final_relative_density_percent': final_density,
            'porosity_percent': porosity,
            'grain_size_mean_um': grain_size,
            'grain_size_std_um': grain_size_std,
            'pore_size_mean_um': pore_size_mean,
            'pore_size_std_um': pore_size_std,
            'grain_boundary_area_per_volume_um2_per_um3': gb_area_per_volume,
            'grain_boundary_thickness_nm': gb_thickness_nm,
            'grain_boundary_energy_J_per_m2': gb_energy,
            'coordination_number': coordination_number,
            'pore_connectivity_fraction': pore_connectivity
        }
    
    def add_experimental_metadata(self):
        """Add realistic experimental metadata and measurement uncertainties."""
        
        n = self.n_samples
        
        # Sample IDs
        sample_ids = [f"SINT_{i+1:04d}" for i in range(n)]
        
        # Experimental dates (spread over 2 years)
        base_date = datetime(2023, 1, 1)
        dates = [base_date + pd.Timedelta(days=np.random.randint(0, 730)) for _ in range(n)]
        
        # Measurement techniques and their typical uncertainties
        sem_magnification = np.random.choice([1000, 2000, 5000, 10000, 20000], n)
        sem_resolution_nm = 50000 / sem_magnification  # Typical SEM resolution
        
        # CT scan parameters
        ct_voxel_size_um = np.random.uniform(0.1, 2.0, n)
        ct_scan_time_hours = np.random.uniform(2, 12, n)
        
        # Archimedes density measurement uncertainty
        density_measurement_error_percent = np.random.uniform(0.1, 0.5, n)
        
        # Operator/batch effects
        operators = np.random.choice(['Operator_A', 'Operator_B', 'Operator_C'], n, p=[0.4, 0.35, 0.25])
        furnace_id = np.random.choice(['Furnace_1', 'Furnace_2', 'Furnace_3'], n, p=[0.5, 0.3, 0.2])
        
        return {
            'sample_id': sample_ids,
            'experiment_date': dates,
            'operator': operators,
            'furnace_id': furnace_id,
            'sem_magnification': sem_magnification,
            'sem_resolution_nm': sem_resolution_nm,
            'ct_voxel_size_um': ct_voxel_size_um,
            'ct_scan_time_hours': ct_scan_time_hours,
            'density_measurement_error_percent': density_measurement_error_percent
        }
    
    def generate_complete_dataset(self):
        """Generate the complete dataset with all parameters and properties."""
        
        print("Generating sintering parameters...")
        sintering_params = self.generate_sintering_parameters()
        
        print("Calculating microstructure properties...")
        microstructure_props = self.calculate_microstructure_properties(sintering_params)
        
        print("Adding experimental metadata...")
        metadata = self.add_experimental_metadata()
        
        # Combine all data
        self.data = {**metadata, **sintering_params, **microstructure_props}
        
        # Create DataFrame
        self.df = pd.DataFrame(self.data)
        
        print(f"Dataset generated with {len(self.df)} samples and {len(self.df.columns)} features.")
        
        return self.df
    
    def save_dataset(self, base_filename="sintering_microstructure_dataset"):
        """Save dataset in multiple formats."""
        
        if not hasattr(self, 'df'):
            raise ValueError("Dataset not generated yet. Call generate_complete_dataset() first.")
        
        # Create output directory
        os.makedirs('/workspace/datasets', exist_ok=True)
        
        # Save as CSV
        csv_path = f'/workspace/datasets/{base_filename}.csv'
        self.df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Save as Excel with multiple sheets
        excel_path = f'/workspace/datasets/{base_filename}.xlsx'
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main dataset
            self.df.to_excel(writer, sheet_name='Complete_Dataset', index=False)
            
            # Separate sheets for different categories
            input_cols = [col for col in self.df.columns if any(x in col.lower() for x in 
                         ['ramp', 'peak', 'hold', 'cool', 'pressure', 'initial', 'particle', 'atmosphere', 'oxygen'])]
            output_cols = [col for col in self.df.columns if any(x in col.lower() for x in 
                          ['final', 'porosity', 'grain', 'pore', 'boundary', 'coordination', 'connectivity'])]
            metadata_cols = [col for col in self.df.columns if col not in input_cols + output_cols]
            
            self.df[input_cols].to_excel(writer, sheet_name='Sintering_Parameters', index=False)
            self.df[output_cols].to_excel(writer, sheet_name='Microstructure_Properties', index=False)
            self.df[metadata_cols].to_excel(writer, sheet_name='Experimental_Metadata', index=False)
        
        print(f"Saved Excel: {excel_path}")
        
        # Save as JSON
        json_path = f'/workspace/datasets/{base_filename}.json'
        # Convert datetime objects to strings for JSON serialization
        df_json = self.df.copy()
        df_json['experiment_date'] = df_json['experiment_date'].dt.strftime('%Y-%m-%d')
        
        with open(json_path, 'w') as f:
            json.dump(df_json.to_dict('records'), f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        return csv_path, excel_path, json_path
    
    def generate_summary_statistics(self):
        """Generate summary statistics and correlations."""
        
        if not hasattr(self, 'df'):
            raise ValueError("Dataset not generated yet. Call generate_complete_dataset() first.")
        
        # Basic statistics
        summary_stats = self.df.describe()
        
        # Correlation matrix for numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        
        # Save summary
        summary_path = '/workspace/datasets/dataset_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("SINTERING MICROSTRUCTURE DATASET SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Total features: {len(self.df.columns)}\n\n")
            
            f.write("DATASET STRUCTURE:\n")
            f.write("-" * 20 + "\n")
            for col in self.df.columns:
                f.write(f"{col}: {self.df[col].dtype}\n")
            
            f.write(f"\n\nSUMMARY STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(summary_stats.to_string())
        
        print(f"Summary statistics saved: {summary_path}")
        
        return summary_stats, correlation_matrix

def main():
    """Main function to generate the complete dataset."""
    
    print("=" * 60)
    print("SINTERING MICROSTRUCTURE DATASET GENERATOR")
    print("=" * 60)
    print()
    
    # Generate dataset
    generator = SinteringDatasetGenerator(n_samples=500)
    df = generator.generate_complete_dataset()
    
    print("\nDataset Preview:")
    print("-" * 40)
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    
    # Save in multiple formats
    print("\nSaving dataset...")
    paths = generator.save_dataset()
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats, correlation_matrix = generator.generate_summary_statistics()
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print("\nFiles created:")
    for path in paths:
        print(f"  - {path}")
    print(f"  - /workspace/datasets/dataset_summary.txt")
    
    print(f"\nKey dataset characteristics:")
    print(f"  - {len(df)} samples")
    print(f"  - {len(df.columns)} features")
    print(f"  - Sintering temperature range: {df['peak_temperature_C'].min():.0f}-{df['peak_temperature_C'].max():.0f}°C")
    print(f"  - Final density range: {df['final_relative_density_percent'].min():.1f}-{df['final_relative_density_percent'].max():.1f}%")
    print(f"  - Grain size range: {df['grain_size_mean_um'].min():.2f}-{df['grain_size_mean_um'].max():.2f} μm")
    
    return df, generator

if __name__ == "__main__":
    df, generator = main()