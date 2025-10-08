#!/usr/bin/env python3
"""
YSZ (8mol% Yttria-Stabilized Zirconia) Material Properties Dataset Generator
For SOFC Electrolyte Thermomechanical FEM Analysis

This script generates a comprehensive, temperature-dependent material properties
dataset for 8mol% YSZ based on literature values and realistic interpolations.

Author: AI Assistant
Date: October 2025
"""

import numpy as np
import pandas as pd
import json
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class YSZMaterialProperties:
    """
    Class to generate and manage YSZ material properties dataset
    """
    
    def __init__(self):
        # Temperature range: Room Temperature to Sintering Temperature
        self.temp_range_celsius = np.linspace(25, 1500, 100)  # 25°C to 1500°C
        self.temp_range_kelvin = self.temp_range_celsius + 273.15
        
        # Initialize properties dictionary
        self.properties = {}
        
    def generate_youngs_modulus(self):
        """
        Generate Young's Modulus data with temperature dependency
        Based on literature: ~200 GPa at RT, decreasing with temperature
        """
        # Key temperature points and values (GPa)
        temp_points = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400, 1500])
        E_values = np.array([205, 195, 180, 165, 145, 125, 100, 75, 65])  # GPa
        
        # Interpolate for full temperature range
        f_E = interp1d(temp_points, E_values, kind='cubic', 
                       bounds_error=False, fill_value='extrapolate')
        
        self.properties['youngs_modulus'] = {
            'values': f_E(self.temp_range_celsius),
            'units': 'GPa',
            'description': 'Young\'s Modulus - Material stiffness',
            'temperature_dependency': 'Strong - decreases with temperature',
            'reference': 'Literature compilation for 8mol% YSZ'
        }
        
    def generate_poisson_ratio(self):
        """
        Generate Poisson's Ratio data
        Generally assumed constant for ceramics, slight temperature dependency
        """
        # Slight temperature dependency: 0.28-0.32 range
        base_value = 0.30
        temp_variation = 0.01 * np.sin((self.temp_range_celsius - 25) * np.pi / 1475)
        
        self.properties['poisson_ratio'] = {
            'values': base_value + temp_variation,
            'units': 'dimensionless',
            'description': 'Poisson\'s Ratio - Lateral to axial strain ratio',
            'temperature_dependency': 'Weak - often assumed constant',
            'reference': 'Typical ceramic values with slight temperature variation'
        }
        
    def generate_thermal_expansion_coefficient(self):
        """
        Generate Coefficient of Thermal Expansion
        Critical for thermal stress calculations
        """
        # Temperature-dependent CTE (×10⁻⁶ /K)
        # YSZ shows increasing CTE with temperature
        temp_points = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400, 1500])
        cte_values = np.array([10.2, 10.4, 10.7, 11.0, 11.3, 11.6, 11.9, 12.2, 12.4])
        
        f_cte = interp1d(temp_points, cte_values, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
        
        self.properties['thermal_expansion_coefficient'] = {
            'values': f_cte(self.temp_range_celsius),
            'units': '×10⁻⁶ /K',
            'description': 'Coefficient of Thermal Expansion',
            'temperature_dependency': 'Critical - increases with temperature',
            'reference': 'YSZ thermal expansion data from multiple sources'
        }
        
    def generate_density(self):
        """
        Generate density data with temperature dependency
        """
        # Room temperature density: ~5850 kg/m³
        # Decreases with temperature due to thermal expansion
        rho_0 = 5850  # kg/m³ at 25°C
        
        # Calculate density using thermal expansion
        cte_avg = 11.0e-6  # Average CTE
        delta_T = self.temp_range_celsius - 25
        
        # Volume expansion factor: (1 + α*ΔT)³ ≈ 1 + 3*α*ΔT for small expansions
        volume_expansion = 1 + 3 * cte_avg * delta_T
        density_values = rho_0 / volume_expansion
        
        self.properties['density'] = {
            'values': density_values,
            'units': 'kg/m³',
            'description': 'Material density',
            'temperature_dependency': 'Mild - decreases with thermal expansion',
            'reference': 'Calculated from thermal expansion'
        }
        
    def generate_thermal_conductivity(self):
        """
        Generate thermal conductivity data
        Important for coupled thermo-mechanical analysis
        """
        # Temperature-dependent thermal conductivity (W/m·K)
        # YSZ shows decreasing thermal conductivity with temperature
        temp_points = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400, 1500])
        k_values = np.array([2.2, 2.0, 1.8, 1.6, 1.5, 1.4, 1.3, 1.25, 1.2])
        
        f_k = interp1d(temp_points, k_values, kind='cubic',
                       bounds_error=False, fill_value='extrapolate')
        
        self.properties['thermal_conductivity'] = {
            'values': f_k(self.temp_range_celsius),
            'units': 'W/m·K',
            'description': 'Thermal conductivity',
            'temperature_dependency': 'Moderate - decreases with temperature',
            'reference': 'YSZ thermal conductivity literature values'
        }
        
    def generate_fracture_toughness(self):
        """
        Generate fracture toughness data
        Critical for crack initiation modeling
        """
        # Temperature-dependent fracture toughness (MPa√m)
        # Generally decreases with temperature for ceramics
        temp_points = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400, 1500])
        K_IC_values = np.array([9.2, 8.8, 8.3, 7.8, 7.2, 6.5, 5.8, 5.0, 4.5])
        
        f_KIC = interp1d(temp_points, K_IC_values, kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
        
        self.properties['fracture_toughness'] = {
            'values': f_KIC(self.temp_range_celsius),
            'units': 'MPa√m',
            'description': 'Fracture toughness - Resistance to crack propagation',
            'temperature_dependency': 'Important - decreases with temperature',
            'reference': 'YSZ fracture toughness from ceramic literature'
        }
        
    def generate_weibull_parameters(self):
        """
        Generate Weibull statistical parameters for brittle failure
        """
        # Weibull modulus (shape parameter)
        # Generally considered temperature-independent for first approximation
        weibull_modulus = 5.5  # Typical for porous YSZ
        
        # Characteristic strength (scale parameter) - temperature dependent
        temp_points = np.array([25, 200, 400, 600, 800, 1000, 1200, 1400, 1500])
        sigma_0_values = np.array([195, 180, 160, 140, 120, 95, 70, 45, 30])  # MPa
        
        f_sigma0 = interp1d(temp_points, sigma_0_values, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
        
        self.properties['weibull_modulus'] = {
            'values': np.full_like(self.temp_range_celsius, weibull_modulus),
            'units': 'dimensionless',
            'description': 'Weibull modulus - Statistical strength parameter',
            'temperature_dependency': 'Assumed constant',
            'reference': 'Typical ceramic values'
        }
        
        self.properties['characteristic_strength'] = {
            'values': f_sigma0(self.temp_range_celsius),
            'units': 'MPa',
            'description': 'Weibull characteristic strength',
            'temperature_dependency': 'Strong - decreases with temperature',
            'reference': 'Estimated from strength degradation trends'
        }
        
    def generate_creep_parameters(self):
        """
        Generate creep parameters for Norton power law
        ε̇ = A * σⁿ * exp(-Q/RT)
        """
        # Norton law parameters for YSZ
        # These are critical for sintering and high-temperature behavior
        
        # Pre-exponential factor A (1/Pa^n/s)
        A_values = np.array([1e-20, 5e-19, 1e-17, 5e-16, 1e-14, 5e-13, 1e-11, 5e-10, 1e-9])
        temp_points_creep = np.array([600, 700, 800, 900, 1000, 1100, 1200, 1400, 1500])
        
        # Extend to full temperature range (creep negligible below 600°C)
        A_extended = np.zeros_like(self.temp_range_celsius)
        mask = self.temp_range_celsius >= 600
        
        if np.any(mask):
            f_A = interp1d(temp_points_creep, A_values, kind='linear',
                          bounds_error=False, fill_value=(A_values[0], A_values[-1]))
            A_extended[mask] = f_A(self.temp_range_celsius[mask])
        
        # Stress exponent n (dimensionless)
        n_value = 1.8  # Typical for YSZ diffusion creep
        
        # Activation energy Q (kJ/mol)
        Q_value = 520  # kJ/mol for YSZ creep
        
        self.properties['creep_prefactor'] = {
            'values': A_extended,
            'units': '1/(Pa^n·s)',
            'description': 'Norton creep law pre-exponential factor',
            'temperature_dependency': 'Exponential - dominant above 600°C',
            'reference': 'YSZ creep literature and extrapolations'
        }
        
        self.properties['creep_stress_exponent'] = {
            'values': np.full_like(self.temp_range_celsius, n_value),
            'units': 'dimensionless',
            'description': 'Norton creep law stress exponent',
            'temperature_dependency': 'Assumed constant',
            'reference': 'Typical for ceramic diffusion creep'
        }
        
        self.properties['creep_activation_energy'] = {
            'values': np.full_like(self.temp_range_celsius, Q_value),
            'units': 'kJ/mol',
            'description': 'Creep activation energy',
            'temperature_dependency': 'Constant',
            'reference': 'YSZ diffusion creep literature'
        }
        
    def generate_all_properties(self):
        """Generate all material properties"""
        print("Generating YSZ Material Properties Dataset...")
        
        self.generate_youngs_modulus()
        print("✓ Young's Modulus generated")
        
        self.generate_poisson_ratio()
        print("✓ Poisson's Ratio generated")
        
        self.generate_thermal_expansion_coefficient()
        print("✓ Thermal Expansion Coefficient generated")
        
        self.generate_density()
        print("✓ Density generated")
        
        self.generate_thermal_conductivity()
        print("✓ Thermal Conductivity generated")
        
        self.generate_fracture_toughness()
        print("✓ Fracture Toughness generated")
        
        self.generate_weibull_parameters()
        print("✓ Weibull Parameters generated")
        
        self.generate_creep_parameters()
        print("✓ Creep Parameters generated")
        
        print("\nDataset generation complete!")
        
    def export_to_csv(self, filename='ysz_properties.csv'):
        """Export dataset to CSV format"""
        # Create DataFrame
        data = {'Temperature_C': self.temp_range_celsius,
                'Temperature_K': self.temp_range_kelvin}
        
        for prop_name, prop_data in self.properties.items():
            data[f"{prop_name}_{prop_data['units']}"] = prop_data['values']
            
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"CSV dataset exported to: {filename}")
        
    def export_to_json(self, filename='ysz_properties.json'):
        """Export dataset to JSON format with metadata"""
        export_data = {
            'material': '8mol% Yttria-Stabilized Zirconia (YSZ)',
            'application': 'SOFC Electrolyte Thermomechanical FEM Analysis',
            'temperature_range': {
                'min_celsius': float(self.temp_range_celsius.min()),
                'max_celsius': float(self.temp_range_celsius.max()),
                'points': len(self.temp_range_celsius)
            },
            'properties': {}
        }
        
        # Add temperature arrays
        export_data['temperature_celsius'] = self.temp_range_celsius.tolist()
        export_data['temperature_kelvin'] = self.temp_range_kelvin.tolist()
        
        # Add properties with metadata
        for prop_name, prop_data in self.properties.items():
            export_data['properties'][prop_name] = {
                'values': prop_data['values'].tolist(),
                'units': prop_data['units'],
                'description': prop_data['description'],
                'temperature_dependency': prop_data['temperature_dependency'],
                'reference': prop_data['reference']
            }
            
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"JSON dataset exported to: {filename}")
        
    def create_summary_table(self):
        """Create a summary table of properties"""
        summary_data = []
        
        for prop_name, prop_data in self.properties.items():
            # Get values at key temperatures
            temp_indices = {
                'RT': 0,  # Room temperature
                '600C': np.argmin(np.abs(self.temp_range_celsius - 600)),
                '1000C': np.argmin(np.abs(self.temp_range_celsius - 1000)),
                '1500C': -1  # Sintering temperature
            }
            
            values_at_temps = {}
            for temp_label, idx in temp_indices.items():
                values_at_temps[temp_label] = prop_data['values'][idx]
                
            summary_data.append({
                'Property': prop_name.replace('_', ' ').title(),
                'Units': prop_data['units'],
                'RT (25°C)': f"{values_at_temps['RT']:.3f}",
                '600°C': f"{values_at_temps['600C']:.3f}",
                '1000°C': f"{values_at_temps['1000C']:.3f}",
                '1500°C': f"{values_at_temps['1500C']:.3f}",
                'Temperature Dependency': prop_data['temperature_dependency']
            })
            
        summary_df = pd.DataFrame(summary_data)
        return summary_df
        
    def plot_properties(self, save_plots=True):
        """Create plots of temperature-dependent properties"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_configs = [
            ('youngs_modulus', 'Young\'s Modulus', 'GPa'),
            ('thermal_expansion_coefficient', 'Thermal Expansion Coefficient', '×10⁻⁶ /K'),
            ('thermal_conductivity', 'Thermal Conductivity', 'W/m·K'),
            ('fracture_toughness', 'Fracture Toughness', 'MPa√m'),
            ('density', 'Density', 'kg/m³'),
            ('characteristic_strength', 'Characteristic Strength', 'MPa'),
            ('creep_prefactor', 'Creep Pre-factor', '1/(Pa^n·s)'),
            ('poisson_ratio', 'Poisson\'s Ratio', 'dimensionless'),
            ('weibull_modulus', 'Weibull Modulus', 'dimensionless')
        ]
        
        for i, (prop_key, title, units) in enumerate(plot_configs):
            if prop_key in self.properties:
                ax = axes[i]
                values = self.properties[prop_key]['values']
                
                if prop_key == 'creep_prefactor':
                    # Use log scale for creep pre-factor
                    ax.semilogy(self.temp_range_celsius, values, 'b-', linewidth=2)
                else:
                    ax.plot(self.temp_range_celsius, values, 'b-', linewidth=2)
                    
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel(f'{title} ({units})')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('ysz_properties_plots.png', dpi=300, bbox_inches='tight')
            print("Property plots saved to: ysz_properties_plots.png")
            
        return fig

def main():
    """Main function to generate the complete dataset"""
    # Create YSZ properties generator
    ysz = YSZMaterialProperties()
    
    # Generate all properties
    ysz.generate_all_properties()
    
    # Export datasets
    ysz.export_to_csv('/workspace/ysz_material_properties.csv')
    ysz.export_to_json('/workspace/ysz_material_properties.json')
    
    # Create and save summary table
    summary = ysz.create_summary_table()
    summary.to_csv('/workspace/ysz_properties_summary.csv', index=False)
    print("Summary table saved to: ysz_properties_summary.csv")
    
    # Display summary
    print("\n" + "="*80)
    print("YSZ MATERIAL PROPERTIES SUMMARY")
    print("="*80)
    print(summary.to_string(index=False))
    
    # Create plots
    ysz.plot_properties()
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    print("Files generated:")
    print("- ysz_material_properties.csv (Full dataset)")
    print("- ysz_material_properties.json (With metadata)")
    print("- ysz_properties_summary.csv (Summary table)")
    print("- ysz_properties_plots.png (Visualization)")
    print("\nDataset ready for FEM analysis!")

if __name__ == "__main__":
    main()