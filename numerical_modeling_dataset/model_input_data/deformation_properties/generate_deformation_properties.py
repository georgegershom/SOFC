#!/usr/bin/env python3
"""
Generate Deformation Properties for Rubberized Concrete
Includes Coefficient of Thermal Expansion (CTE) and Transient Thermal Strain
Based on dilatometry and high-temperature deformation measurements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter
import json
from datetime import datetime

class DeformationPropertiesGenerator:
    def __init__(self, rubber_content_percent=0, aggregate_type='siliceous'):
        """
        Initialize deformation properties generator
        rubber_content_percent: 0-30% typical range
        aggregate_type: 'siliceous', 'calcareous', or 'lightweight'
        """
        self.rubber_content = rubber_content_percent
        self.aggregate_type = aggregate_type
        self.temperature_range = np.arange(20, 1001, 5)  # Higher resolution for CTE
        
    def coefficient_thermal_expansion(self, T):
        """
        Generate CTE α(T) [10^-6/°C]
        Based on aggregate type and rubber content
        """
        # Base CTE values depend on aggregate type
        if self.aggregate_type == 'siliceous':
            if T <= 100:
                cte_base = 12.0
            elif T <= 300:
                cte_base = 12.0 + 2.0 * (T - 100) / 200
            elif T <= 500:
                cte_base = 14.0 + 4.0 * (T - 300) / 200
            elif T <= 700:
                cte_base = 18.0 - 2.0 * (T - 500) / 200
            else:
                cte_base = 16.0 - 4.0 * (T - 700) / 300
        elif self.aggregate_type == 'calcareous':
            if T <= 100:
                cte_base = 6.0
            elif T <= 400:
                cte_base = 6.0 + 2.0 * (T - 100) / 300
            elif T <= 700:
                cte_base = 8.0 + 3.0 * (T - 400) / 300
            else:
                cte_base = 11.0
        else:  # lightweight
            if T <= 200:
                cte_base = 8.0
            elif T <= 600:
                cte_base = 8.0 + 2.0 * (T - 200) / 400
            else:
                cte_base = 10.0
        
        # Rubber modification (rubber has higher CTE ~200-400 x10^-6/°C)
        rubber_factor = 1 + (self.rubber_content / 100) * 0.5
        
        # Phase transformation effects (quartz at 573°C for siliceous)
        if self.aggregate_type == 'siliceous' and 550 < T < 600:
            phase_spike = 10.0 * np.exp(-((T - 573) / 10)**2)
            cte_base += phase_spike
        
        # Add realistic scatter
        noise = np.random.normal(0, 0.5)
        
        return max(2.0, cte_base * rubber_factor + noise)
    
    def free_thermal_strain(self, T):
        """
        Calculate free thermal strain εth(T) [-]
        Integration of CTE over temperature
        """
        if T <= 20:
            return 0.0
        
        # Numerical integration of CTE
        temp_points = np.linspace(20, T, 100)
        cte_values = [self.coefficient_thermal_expansion(t) for t in temp_points]
        
        # Trapezoidal integration
        strain = np.trapz(cte_values, temp_points) * 1e-6
        
        return strain
    
    def transient_thermal_strain(self, T, stress_level=0.3):
        """
        Generate transient thermal strain εtr(T,σ) [-]
        Includes LITS (Load Induced Thermal Strain) and creep
        stress_level: ratio of applied stress to strength (0-0.6 typical)
        """
        # Base transient strain coefficient
        if T <= 100:
            k_tr = 0.0
        elif T <= 200:
            k_tr = 1.8 * (T - 100) / 100
        elif T <= 400:
            k_tr = 1.8 + 3.2 * (T - 200) / 200
        elif T <= 600:
            k_tr = 5.0 + 3.0 * (T - 400) / 200
        elif T <= 800:
            k_tr = 8.0 + 2.0 * (T - 600) / 200
        else:
            k_tr = 10.0
        
        # Stress dependency
        stress_factor = stress_level * (1 + 0.5 * stress_level)  # Nonlinear stress effect
        
        # Rubber increases transient strain (more deformable)
        rubber_factor = 1 + (self.rubber_content / 100) * 0.4
        
        # Calculate transient strain
        transient_strain = k_tr * stress_factor * rubber_factor * 1e-4
        
        # Add time-dependent creep component
        creep_strain = self.basic_creep(T, stress_level, time_hours=2)
        
        # Add scatter
        noise = np.random.normal(0, transient_strain * 0.05)
        
        return transient_strain + creep_strain + noise
    
    def basic_creep(self, T, stress_level, time_hours):
        """
        Calculate basic creep strain εcr(T,σ,t) [-]
        """
        # Temperature factor
        if T <= 100:
            temp_factor = 1.0
        elif T <= 400:
            temp_factor = 1.0 + 2.0 * (T - 100) / 300
        elif T <= 700:
            temp_factor = 3.0 + 4.0 * (T - 400) / 300
        else:
            temp_factor = 7.0
        
        # Time function (power law)
        time_factor = time_hours ** 0.3
        
        # Creep coefficient
        creep_coeff = 25e-6 * temp_factor * stress_level
        
        return creep_coeff * time_factor
    
    def total_strain(self, T, stress_level=0.3, mechanical_strain=0.002):
        """
        Calculate total strain εtot = εmech + εth + εtr + εcr
        """
        thermal_strain = self.free_thermal_strain(T)
        transient_strain = self.transient_thermal_strain(T, stress_level)
        
        total = mechanical_strain + thermal_strain + transient_strain
        
        return {
            'total_strain': total,
            'mechanical_strain': mechanical_strain,
            'thermal_strain': thermal_strain,
            'transient_strain': transient_strain
        }
    
    def generate_dilatometry_data(self):
        """
        Generate simulated dilatometry test data
        """
        data = {
            'temperature_C': [],
            'thermal_strain': [],
            'cte_instantaneous_1e6_perC': [],
            'cte_secant_1e6_perC': [],
            'length_change_mm': [],
            'heating_rate_C_min': []
        }
        
        # Initial specimen length
        L0 = 100.0  # mm
        heating_rate = 5.0  # °C/min
        
        for T in self.temperature_range:
            thermal_strain = self.free_thermal_strain(T)
            cte_inst = self.coefficient_thermal_expansion(T)
            
            # Secant CTE (average from 20°C to T)
            if T > 20:
                cte_secant = thermal_strain / (T - 20) * 1e6
            else:
                cte_secant = cte_inst
            
            # Length change
            delta_L = L0 * thermal_strain
            
            data['temperature_C'].append(T)
            data['thermal_strain'].append(thermal_strain)
            data['cte_instantaneous_1e6_perC'].append(cte_inst)
            data['cte_secant_1e6_perC'].append(cte_secant)
            data['length_change_mm'].append(delta_L)
            data['heating_rate_C_min'].append(heating_rate)
        
        return pd.DataFrame(data)
    
    def generate_lits_test_data(self, stress_levels=[0.1, 0.2, 0.3, 0.4]):
        """
        Generate Load-Induced Thermal Strain test data
        """
        lits_data = []
        
        for stress_level in stress_levels:
            for T in np.arange(20, 801, 50):
                strain_components = self.total_strain(T, stress_level)
                
                record = {
                    'temperature_C': T,
                    'stress_level': stress_level,
                    'applied_stress_MPa': stress_level * 40,  # Assuming fc=40 MPa at 20°C
                    'total_strain': strain_components['total_strain'],
                    'mechanical_strain': strain_components['mechanical_strain'],
                    'free_thermal_strain': strain_components['thermal_strain'],
                    'transient_thermal_strain': strain_components['transient_strain'],
                    'rubber_content_%': self.rubber_content,
                    'test_duration_hours': np.random.uniform(2, 4)
                }
                
                lits_data.append(record)
        
        return pd.DataFrame(lits_data)
    
    def generate_restraint_test_data(self, restraint_levels=[0.0, 0.5, 1.0]):
        """
        Generate data for restrained thermal expansion tests
        restraint_level: 0 = free, 1 = fully restrained
        """
        restraint_data = []
        
        for restraint in restraint_levels:
            for T in np.arange(20, 601, 40):
                free_strain = self.free_thermal_strain(T)
                
                # Calculate stress induced by restraint
                E = 30 - 0.02 * T  # Simplified E(T) in GPa
                induced_stress = restraint * free_strain * E * 1000  # MPa
                
                # Account for stress relaxation and cracking
                if induced_stress > 0.5 * (40 - 0.03 * T):  # Exceeds tensile strength
                    crack_factor = 0.5
                    effective_stress = induced_stress * crack_factor
                else:
                    effective_stress = induced_stress
                
                record = {
                    'temperature_C': T,
                    'restraint_level': restraint,
                    'free_thermal_strain': free_strain,
                    'restrained_strain': free_strain * (1 - restraint),
                    'induced_stress_MPa': effective_stress,
                    'crack_width_mm': max(0, (induced_stress - 20) * 0.001) if induced_stress > 20 else 0,
                    'rubber_content_%': self.rubber_content
                }
                
                restraint_data.append(record)
        
        return pd.DataFrame(restraint_data)

def main():
    # Generate data for different configurations
    rubber_contents = [0, 5, 10, 15, 20, 25, 30]
    aggregate_types = ['siliceous', 'calcareous', 'lightweight']
    
    all_data = {}
    
    for aggregate in aggregate_types:
        for rubber_pct in rubber_contents:
            print(f"Generating deformation properties for {aggregate} aggregate, {rubber_pct}% rubber...")
            
            generator = DeformationPropertiesGenerator(
                rubber_content_percent=rubber_pct,
                aggregate_type=aggregate
            )
            
            # Generate dilatometry data
            dilato_data = generator.generate_dilatometry_data()
            
            # Generate LITS test data
            lits_data = generator.generate_lits_test_data()
            
            # Generate restraint test data
            restraint_data = generator.generate_restraint_test_data()
            
            # Store data
            key = f'{aggregate}_{rubber_pct}pct'
            all_data[key] = {
                'dilatometry': dilato_data,
                'lits': lits_data,
                'restraint': restraint_data
            }
            
            # Save CSV files
            dilato_filename = f'dilatometry_{aggregate}_{rubber_pct}pct.csv'
            dilato_data.to_csv(dilato_filename, index=False)
            
            lits_filename = f'lits_test_{aggregate}_{rubber_pct}pct.csv'
            lits_data.to_csv(lits_filename, index=False)
            
            restraint_filename = f'restraint_test_{aggregate}_{rubber_pct}pct.csv'
            restraint_data.to_csv(restraint_filename, index=False)
    
    print(f"\nSaved {len(rubber_contents) * len(aggregate_types) * 3} deformation property files")
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot CTE for different rubber contents (siliceous aggregate)
    for rubber_pct in rubber_contents:
        key = f'siliceous_{rubber_pct}pct'
        if key in all_data:
            dilato = all_data[key]['dilatometry']
            
            # CTE vs Temperature
            axes[0, 0].plot(dilato['temperature_C'],
                          dilato['cte_instantaneous_1e6_perC'],
                          label=f'{rubber_pct}% rubber', alpha=0.7)
            
            # Thermal strain vs Temperature
            axes[0, 1].plot(dilato['temperature_C'],
                          dilato['thermal_strain'] * 1000,  # Convert to millistrain
                          label=f'{rubber_pct}% rubber', alpha=0.7)
    
    # Plot LITS data for 15% rubber
    key = 'siliceous_15pct'
    if key in all_data:
        lits = all_data[key]['lits']
        
        for stress_level in lits['stress_level'].unique():
            data_subset = lits[lits['stress_level'] == stress_level]
            axes[0, 2].plot(data_subset['temperature_C'],
                          data_subset['transient_thermal_strain'] * 1000,
                          label=f'σ/fc = {stress_level}', marker='o', markersize=4)
    
    # Compare aggregate types (10% rubber)
    for aggregate in aggregate_types:
        key = f'{aggregate}_10pct'
        if key in all_data:
            dilato = all_data[key]['dilatometry']
            axes[1, 0].plot(dilato['temperature_C'],
                          dilato['cte_instantaneous_1e6_perC'],
                          label=f'{aggregate}', linewidth=2)
    
    # Total strain components (siliceous, 20% rubber, 30% stress)
    key = 'siliceous_20pct'
    if key in all_data:
        lits = all_data[key]['lits']
        data_30 = lits[lits['stress_level'] == 0.3]
        
        axes[1, 1].plot(data_30['temperature_C'],
                       data_30['mechanical_strain'] * 1000,
                       label='Mechanical', marker='s')
        axes[1, 1].plot(data_30['temperature_C'],
                       data_30['free_thermal_strain'] * 1000,
                       label='Thermal', marker='o')
        axes[1, 1].plot(data_30['temperature_C'],
                       data_30['transient_thermal_strain'] * 1000,
                       label='Transient', marker='^')
        axes[1, 1].plot(data_30['temperature_C'],
                       data_30['total_strain'] * 1000,
                       label='Total', linewidth=3, color='black')
    
    # Restraint effects
    key = 'siliceous_15pct'
    if key in all_data:
        restraint = all_data[key]['restraint']
        
        for level in restraint['restraint_level'].unique():
            data_subset = restraint[restraint['restraint_level'] == level]
            axes[1, 2].plot(data_subset['temperature_C'],
                          data_subset['induced_stress_MPa'],
                          label=f'Restraint = {level}', linewidth=2)
    
    # Format all plots
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('CTE (×10⁻⁶/°C)')
    axes[0, 0].set_title('Coefficient of Thermal Expansion (Siliceous)')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Thermal Strain (millistrain)')
    axes[0, 1].set_title('Free Thermal Strain (Siliceous)')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[0, 2].set_xlabel('Temperature (°C)')
    axes[0, 2].set_ylabel('Transient Thermal Strain (millistrain)')
    axes[0, 2].set_title('Load-Induced Thermal Strain (15% Rubber)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Temperature (°C)')
    axes[1, 0].set_ylabel('CTE (×10⁻⁶/°C)')
    axes[1, 0].set_title('CTE - Aggregate Type Comparison (10% Rubber)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Strain (millistrain)')
    axes[1, 1].set_title('Strain Components (20% Rubber, 30% Stress)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[1, 2].set_xlabel('Temperature (°C)')
    axes[1, 2].set_ylabel('Induced Stress (MPa)')
    axes[1, 2].set_title('Restraint-Induced Stresses (15% Rubber)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Deformation Properties of Rubberized Concrete at Elevated Temperatures', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('deformation_properties_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate metadata
    metadata = {
        'generated_date': datetime.now().isoformat(),
        'temperature_range_C': [20, 1000],
        'rubber_content_range_percent': [0, 30],
        'aggregate_types': aggregate_types,
        'test_methods': {
            'dilatometry': {
                'standard': 'ASTM E228',
                'heating_rate_C_min': 5,
                'specimen_length_mm': 100,
                'measurement_accuracy': '±1 μm'
            },
            'lits': {
                'standard': 'RILEM TC 129-MHT',
                'stress_levels': [0.1, 0.2, 0.3, 0.4],
                'loading_type': 'constant compression',
                'measurement_interval_min': 10
            },
            'restraint': {
                'method': 'Restrained ring test',
                'restraint_levels': [0.0, 0.5, 1.0],
                'measurement': 'Strain gauges and load cells'
            }
        },
        'data_source': 'Synthetic data based on literature correlations and models'
    }
    
    with open('deformation_properties_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Deformation properties dataset generation complete!")

if __name__ == "__main__":
    main()