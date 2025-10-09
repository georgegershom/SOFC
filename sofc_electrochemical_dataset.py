#!/usr/bin/env python3
"""
SOFC Electrochemical Loading Dataset Generator

This script generates comprehensive electrochemical loading data for Solid Oxide Fuel Cells (SOFCs),
including operating voltage, current density, and overpotentials with focus on anode oxidation effects.

Author: AI Assistant
Date: 2024
Purpose: Generate synthetic but realistic SOFC performance data for research and analysis
"""

import numpy as np
import pandas as pd
import json
import h5py
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class SOFCElectrochemicalDataGenerator:
    """
    Generator for SOFC electrochemical loading data including IV curves, 
    impedance spectroscopy, and overpotential analysis.
    """
    
    def __init__(self, seed=42):
        """Initialize the data generator with random seed for reproducibility."""
        np.random.seed(seed)
        self.data = {}
        self.metadata = {
            'generator_version': '1.0',
            'creation_date': datetime.now().isoformat(),
            'description': 'SOFC Electrochemical Loading Dataset - Synthetic but realistic data',
            'cell_configuration': 'Planar SOFC with 8YSZ electrolyte',
            'operating_conditions': {
                'temperature_range': [600, 1000],  # Celsius
                'pressure': 1.0,  # atm
                'fuel_composition': 'H2/H2O/N2',
                'oxidant_composition': 'Air'
            }
        }
    
    def generate_iv_curves(self, n_curves=50, temperature_range=(700, 850)):
        """
        Generate I-V curves for different operating conditions.
        
        Parameters:
        - n_curves: Number of I-V curves to generate
        - temperature_range: Temperature range in Celsius
        """
        print("Generating I-V curves...")
        
        # Temperature points
        temperatures = np.linspace(temperature_range[0], temperature_range[1], n_curves)
        
        iv_data = []
        
        for i, temp in enumerate(temperatures):
            # Current density range (A/cm²)
            current_density = np.linspace(0, 1.2, 100)
            
            # Open circuit voltage (temperature dependent)
            ocv = 1.1 - 0.0003 * (temp - 800)  # V
            
            # Area specific resistance (temperature dependent)
            asr = 0.15 * np.exp(1000 * (1/temp - 1/800))  # Ohm·cm²
            
            # Ohmic overpotential
            eta_ohmic = current_density * asr
            
            # Activation overpotential (Butler-Volmer equation)
            i0 = 0.1 * np.exp(-8000 * (1/temp - 1/800))  # Exchange current density
            eta_act = (2 * 8.314 * (temp + 273.15) / (4 * 96485)) * np.arcsinh(current_density / (2 * i0))
            
            # Concentration overpotential (mass transport limitation)
            i_lim = 2.0 * np.exp(-5000 * (1/temp - 1/800))  # Limiting current density
            eta_conc = (8.314 * (temp + 273.15) / (4 * 96485)) * np.log(1 - current_density / i_lim)
            
            # Total voltage
            voltage = ocv - eta_ohmic - eta_act - eta_conc
            
            # Add realistic noise
            voltage += np.random.normal(0, 0.005, len(voltage))
            
            # Store data
            for j in range(len(current_density)):
                iv_data.append({
                    'curve_id': i,
                    'temperature_c': temp,
                    'current_density_acm2': current_density[j],
                    'voltage_v': voltage[j],
                    'ocv_v': ocv,
                    'asr_ohmic_cm2': asr,
                    'eta_ohmic_v': eta_ohmic[j],
                    'eta_activation_v': eta_act[j],
                    'eta_concentration_v': eta_conc[j],
                    'power_density_wcm2': current_density[j] * voltage[j]
                })
        
        self.data['iv_curves'] = pd.DataFrame(iv_data)
        print(f"Generated {n_curves} I-V curves with {len(iv_data)} data points")
        
    def generate_impedance_spectroscopy(self, n_spectra=30, frequency_range=(0.1, 100000)):
        """
        Generate Electrochemical Impedance Spectroscopy (EIS) data.
        
        Parameters:
        - n_spectra: Number of impedance spectra to generate
        - frequency_range: Frequency range in Hz
        """
        print("Generating EIS spectra...")
        
        # Frequency points (logarithmic spacing)
        frequencies = np.logspace(np.log10(frequency_range[0]), np.log10(frequency_range[1]), 50)
        
        eis_data = []
        
        for i in range(n_spectra):
            # Operating conditions
            temp = np.random.uniform(700, 850)
            current_density = np.random.uniform(0.1, 0.8)
            
            # Equivalent circuit parameters (Randles circuit with Warburg element)
            R_ohmic = 0.1 + 0.05 * np.random.normal(0, 1)  # Ohmic resistance
            R_ct = 0.2 * np.exp(1000 * (1/temp - 1/800))  # Charge transfer resistance
            C_dl = 1e-4 * np.exp(-2000 * (1/temp - 1/800))  # Double layer capacitance
            W_R = 0.1 * np.exp(500 * (1/temp - 1/800))  # Warburg resistance
            W_T = 1e-3 * np.exp(-1000 * (1/temp - 1/800))  # Warburg time constant
            
            # Calculate impedance
            omega = 2 * np.pi * frequencies
            Z_real = R_ohmic + R_ct / (1 + (omega * R_ct * C_dl)**2) + W_R / np.sqrt(omega * W_T)
            Z_imag = -(omega * R_ct**2 * C_dl) / (1 + (omega * R_ct * C_dl)**2) - W_R / np.sqrt(omega * W_T)
            
            # Add noise
            Z_real += np.random.normal(0, 0.001, len(Z_real))
            Z_imag += np.random.normal(0, 0.001, len(Z_imag))
            
            # Store data
            for j, freq in enumerate(frequencies):
                eis_data.append({
                    'spectrum_id': i,
                    'temperature_c': temp,
                    'current_density_acm2': current_density,
                    'frequency_hz': freq,
                    'z_real_ohm_cm2': Z_real[j],
                    'z_imag_ohm_cm2': Z_imag[j],
                    'z_magnitude_ohm_cm2': np.sqrt(Z_real[j]**2 + Z_imag[j]**2),
                    'phase_angle_deg': np.degrees(np.arctan2(Z_imag[j], Z_real[j])),
                    'r_ohmic_ohm_cm2': R_ohmic,
                    'r_ct_ohm_cm2': R_ct
                })
        
        self.data['eis_spectra'] = pd.DataFrame(eis_data)
        print(f"Generated {n_spectra} EIS spectra with {len(eis_data)} data points")
    
    def generate_overpotential_analysis(self, n_measurements=200):
        """
        Generate detailed overpotential analysis with focus on anode oxidation effects.
        
        Parameters:
        - n_measurements: Number of overpotential measurements
        """
        print("Generating overpotential analysis...")
        
        overpotential_data = []
        
        for i in range(n_measurements):
            # Operating conditions
            temp = np.random.uniform(700, 900)
            current_density = np.random.uniform(0.05, 1.0)
            time_hours = np.random.uniform(0, 1000)  # Operating time
            
            # Anode overpotential (Ni to NiO oxidation effects)
            # Higher temperatures and longer operation increase Ni oxidation
            ni_oxidation_factor = 1 + 0.1 * (temp - 700) / 200 + 0.05 * np.log(1 + time_hours / 100)
            
            # Anode activation overpotential (increases with Ni oxidation)
            eta_anode_act = 0.05 * ni_oxidation_factor * (1 + 0.2 * np.random.normal(0, 1))
            
            # Anode concentration overpotential (affected by NiO formation)
            eta_anode_conc = 0.02 * ni_oxidation_factor * current_density
            
            # Cathode overpotential (less affected by time)
            eta_cathode_act = 0.03 * (1 + 0.1 * np.random.normal(0, 1))
            eta_cathode_conc = 0.01 * current_density
            
            # Ohmic overpotential (increases with Ni oxidation due to reduced conductivity)
            asr_ohmic = 0.1 * (1 + 0.3 * ni_oxidation_factor)
            eta_ohmic = current_density * asr_ohmic
            
            # Total overpotential
            eta_total = eta_anode_act + eta_anode_conc + eta_cathode_act + eta_cathode_conc + eta_ohmic
            
            # Volume change due to Ni to NiO oxidation (affects mechanical stress)
            ni_volume_change = 0.69 * ni_oxidation_factor  # Ni to NiO volume expansion factor
            stress_induced = 50 * ni_volume_change  # MPa (simplified stress calculation)
            
            # Store data
            overpotential_data.append({
                'measurement_id': i,
                'temperature_c': temp,
                'current_density_acm2': current_density,
                'time_hours': time_hours,
                'ni_oxidation_factor': ni_oxidation_factor,
                'eta_anode_activation_v': eta_anode_act,
                'eta_anode_concentration_v': eta_anode_conc,
                'eta_cathode_activation_v': eta_cathode_act,
                'eta_cathode_concentration_v': eta_cathode_conc,
                'eta_ohmic_v': eta_ohmic,
                'eta_total_v': eta_total,
                'asr_ohmic_ohm_cm2': asr_ohmic,
                'ni_volume_change_percent': (ni_volume_change - 1) * 100,
                'induced_stress_mpa': stress_induced,
                'oxygen_chemical_potential_gradient': 0.5 * eta_total  # Simplified relationship
            })
        
        self.data['overpotential_analysis'] = pd.DataFrame(overpotential_data)
        print(f"Generated {n_measurements} overpotential measurements")
    
    def generate_operating_conditions(self, n_conditions=100):
        """
        Generate various operating conditions and their effects on electrochemical performance.
        """
        print("Generating operating conditions data...")
        
        conditions_data = []
        
        for i in range(n_conditions):
            # Operating parameters
            temp = np.random.uniform(650, 900)
            pressure = np.random.uniform(0.8, 1.2)
            fuel_utilization = np.random.uniform(0.6, 0.9)
            air_utilization = np.random.uniform(0.15, 0.35)
            current_density = np.random.uniform(0.1, 1.0)
            
            # Calculate performance metrics
            # Nernst potential
            e_nernst = 1.253 - 0.000245 * temp
            
            # Area specific resistance (temperature and pressure dependent)
            asr = 0.15 * np.exp(1000 * (1/temp - 1/800)) * (1/pressure)**0.5
            
            # Voltage losses
            eta_ohmic = current_density * asr
            eta_activation = 0.1 * np.log(current_density / 0.1)
            eta_concentration = 0.05 * np.log(1 / (1 - fuel_utilization))
            
            # Cell voltage
            voltage = e_nernst - eta_ohmic - eta_activation - eta_concentration
            
            # Power density
            power_density = current_density * voltage
            
            # Efficiency
            efficiency = voltage / e_nernst
            
            # Oxygen chemical potential gradient (related to overpotentials)
            oxygen_gradient = (eta_activation + eta_concentration) * 96485 / (4 * 8.314 * (temp + 273.15))
            
            conditions_data.append({
                'condition_id': i,
                'temperature_c': temp,
                'pressure_atm': pressure,
                'fuel_utilization': fuel_utilization,
                'air_utilization': air_utilization,
                'current_density_acm2': current_density,
                'voltage_v': voltage,
                'power_density_wcm2': power_density,
                'efficiency': efficiency,
                'asr_ohm_cm2': asr,
                'eta_ohmic_v': eta_ohmic,
                'eta_activation_v': eta_activation,
                'eta_concentration_v': eta_concentration,
                'oxygen_chemical_potential_gradient': oxygen_gradient,
                'e_nernst_v': e_nernst
            })
        
        self.data['operating_conditions'] = pd.DataFrame(conditions_data)
        print(f"Generated {n_conditions} operating conditions")
    
    def save_data(self, output_dir='/workspace/sofc_data'):
        """Save all generated data in multiple formats."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving data to {output_dir}...")
        
        # Save as CSV files
        for dataset_name, df in self.data.items():
            csv_path = os.path.join(output_dir, f'{dataset_name}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Saved {dataset_name}.csv")
        
        # Save as JSON
        json_data = {}
        for dataset_name, df in self.data.items():
            json_data[dataset_name] = df.to_dict('records')
        
        json_data['metadata'] = self.metadata
        
        with open(os.path.join(output_dir, 'sofc_electrochemical_data.json'), 'w') as f:
            json.dump(json_data, f, indent=2)
        print("Saved sofc_electrochemical_data.json")
        
        # Save as HDF5
        h5_path = os.path.join(output_dir, 'sofc_electrochemical_data.h5')
        with h5py.File(h5_path, 'w') as f:
            for dataset_name, df in self.data.items():
                f.create_dataset(dataset_name, data=df.to_records(index=False))
            
            # Save metadata as attributes
            for key, value in self.metadata.items():
                f.attrs[key] = str(value)
        
        print("Saved sofc_electrochemical_data.h5")
        
        # Save metadata separately
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print("Saved metadata.json")
    
    def generate_summary_statistics(self):
        """Generate summary statistics for all datasets."""
        print("\n" + "="*50)
        print("DATASET SUMMARY STATISTICS")
        print("="*50)
        
        for dataset_name, df in self.data.items():
            print(f"\n{dataset_name.upper()}:")
            print(f"  Records: {len(df)}")
            print(f"  Columns: {len(df.columns)}")
            print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show key statistics for numerical columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                print(f"  Key statistics:")
                for col in numeric_cols[:3]:  # Show first 3 numerical columns
                    print(f"    {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

def main():
    """Main function to generate the complete SOFC electrochemical dataset."""
    print("SOFC Electrochemical Loading Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = SOFCElectrochemicalDataGenerator(seed=42)
    
    # Generate all datasets
    generator.generate_iv_curves(n_curves=50, temperature_range=(700, 850))
    generator.generate_impedance_spectroscopy(n_spectra=30, frequency_range=(0.1, 100000))
    generator.generate_overpotential_analysis(n_measurements=200)
    generator.generate_operating_conditions(n_conditions=100)
    
    # Save data
    generator.save_data()
    
    # Generate summary
    generator.generate_summary_statistics()
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE!")
    print("="*50)
    print("Files saved in /workspace/sofc_data/")
    print("- CSV files for each dataset")
    print("- JSON file with all data")
    print("- HDF5 file for efficient storage")
    print("- Metadata file with dataset information")

if __name__ == "__main__":
    main()