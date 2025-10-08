#!/usr/bin/env python3
"""
YSZ Material Properties Analysis and Visualization
For SOFC Thermomechanical FEM Modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, PchipInterpolator
import json
from pathlib import Path

class YSZMaterialProperties:
    """Class to handle YSZ material property data and interpolation"""
    
    def __init__(self, csv_file='ysz_material_properties.csv'):
        """Load material properties from CSV file"""
        self.df = pd.read_csv(csv_file)
        self.properties = self.df.columns[2:].tolist()  # Exclude temperature columns
        self.interpolators = {}
        self._create_interpolators()
    
    def _create_interpolators(self):
        """Create interpolation functions for each property"""
        temp_k = self.df['Temperature_K'].values
        
        for prop in self.properties:
            # Use PCHIP interpolation for smooth, monotonic interpolation
            self.interpolators[prop] = PchipInterpolator(
                temp_k, 
                self.df[prop].values,
                extrapolate=False
            )
    
    def get_property(self, property_name, temperature_k):
        """
        Get interpolated property value at given temperature
        
        Args:
            property_name: Name of the property
            temperature_k: Temperature in Kelvin
        
        Returns:
            Interpolated property value
        """
        if property_name not in self.interpolators:
            raise ValueError(f"Property {property_name} not found")
        
        return self.interpolators[property_name](temperature_k)
    
    def plot_all_properties(self, save_dir='plots'):
        """Generate plots for all material properties"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Create a 3x3 subplot for the main properties
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('YSZ Material Properties vs Temperature', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('Youngs_Modulus_GPa', 'Young\'s Modulus (GPa)', axes[0, 0]),
            ('Poissons_Ratio', 'Poisson\'s Ratio', axes[0, 1]),
            ('CTE_1e-6_per_K', 'CTE (10⁻⁶/K)', axes[0, 2]),
            ('Density_kg_m3', 'Density (kg/m³)', axes[1, 0]),
            ('Thermal_Conductivity_W_mK', 'Thermal Conductivity (W/m·K)', axes[1, 1]),
            ('Fracture_Toughness_MPa_sqrt_m', 'Fracture Toughness (MPa√m)', axes[1, 2]),
            ('Weibull_Modulus', 'Weibull Modulus', axes[2, 0]),
            ('Characteristic_Strength_MPa', 'Characteristic Strength (MPa)', axes[2, 1]),
            ('Creep_Rate_Coefficient_A', 'Creep Rate Coefficient A', axes[2, 2])
        ]
        
        temp_c = self.df['Temperature_C'].values
        
        for prop, label, ax in plot_configs:
            values = self.df[prop].values
            
            # Plot data points
            ax.plot(temp_c, values, 'bo-', markersize=4, linewidth=1.5)
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.set_title(label, fontsize=10)
            
            # Special handling for log scale properties
            if prop == 'Creep_Rate_Coefficient_A':
                ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/ysz_properties_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual high-resolution plots for key properties
        self._plot_key_properties(save_dir)
        
        print(f"Plots saved to {save_dir}/")
    
    def _plot_key_properties(self, save_dir):
        """Create detailed individual plots for key properties"""
        key_properties = [
            ('Youngs_Modulus_GPa', 'Young\'s Modulus', 'E (GPa)'),
            ('CTE_1e-6_per_K', 'Coefficient of Thermal Expansion', 'α (10⁻⁶/K)'),
            ('Thermal_Conductivity_W_mK', 'Thermal Conductivity', 'k (W/m·K)'),
            ('Fracture_Toughness_MPa_sqrt_m', 'Fracture Toughness', 'K_IC (MPa√m)')
        ]
        
        temp_c = self.df['Temperature_C'].values
        temp_c_fine = np.linspace(25, 1500, 500)
        temp_k_fine = temp_c_fine + 273.15
        
        for prop, title, ylabel in key_properties:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Original data points
            ax.plot(temp_c, self.df[prop].values, 'ro', markersize=8, label='Data Points')
            
            # Interpolated curve
            interp_values = self.interpolators[prop](temp_k_fine)
            ax.plot(temp_c_fine, interp_values, 'b-', linewidth=2, label='Interpolated')
            
            ax.set_xlabel('Temperature (°C)', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f'YSZ {title} vs Temperature', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add annotations for key temperatures
            key_temps = [25, 800, 1500]  # RT, Operating, Sintering
            for kt in key_temps:
                idx = np.argmin(np.abs(temp_c - kt))
                value = self.df[prop].values[idx]
                ax.annotate(f'{value:.2f}',
                           xy=(kt, value),
                           xytext=(kt, value + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05),
                           ha='center',
                           fontsize=9,
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{prop}_detailed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def export_to_json(self, filename='ysz_properties.json'):
        """Export data to JSON format"""
        data = {
            'material': '8YSZ (8 mol% Yttria-Stabilized Zirconia)',
            'description': 'Temperature-dependent material properties for SOFC electrolyte FEM modeling',
            'units': {
                'Temperature': 'Celsius and Kelvin',
                'Youngs_Modulus': 'GPa',
                'Poissons_Ratio': 'dimensionless',
                'CTE': '10^-6 per Kelvin',
                'Density': 'kg/m^3',
                'Thermal_Conductivity': 'W/(m·K)',
                'Fracture_Toughness': 'MPa·sqrt(m)',
                'Weibull_Modulus': 'dimensionless',
                'Characteristic_Strength': 'MPa',
                'Creep_Rate_Coefficient': 'varies with units',
                'Creep_Stress_Exponent': 'dimensionless',
                'Creep_Activation_Energy': 'kJ/mol'
            },
            'data': self.df.to_dict('records')
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Data exported to {filename}")
    
    def export_to_excel(self, filename='ysz_properties.xlsx'):
        """Export data to Excel with multiple sheets"""
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # Main data sheet
            self.df.to_excel(writer, sheet_name='Material Properties', index=False)
            
            # Create a metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Material', 'Application', 'Temperature Range', 'Data Source'],
                'Value': [
                    '8YSZ (8 mol% Yttria-Stabilized Zirconia)',
                    'SOFC Electrolyte',
                    '25-1500°C',
                    'Literature compilation and validated estimates'
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
            # Create a units sheet
            units_data = pd.DataFrame({
                'Property': self.df.columns.tolist(),
                'Units': ['°C', 'K', 'GPa', '-', '10⁻⁶/K', 'kg/m³', 'W/(m·K)', 
                         'MPa√m', '-', 'MPa', 'varies', '-', 'kJ/mol']
            })
            units_data.to_excel(writer, sheet_name='Units', index=False)
        
        print(f"Data exported to {filename}")
    
    def generate_fem_input_file(self, temperature_c, filename='fem_input.txt'):
        """
        Generate FEM input file for specific temperature
        
        Args:
            temperature_c: Temperature in Celsius
            filename: Output filename
        """
        temp_k = temperature_c + 273.15
        
        with open(filename, 'w') as f:
            f.write(f"! YSZ Material Properties at {temperature_c}°C\n")
            f.write(f"! Generated for FEM Thermomechanical Analysis\n")
            f.write("!-------------------------------------------------\n\n")
            
            f.write("! Elastic Properties\n")
            f.write(f"MP,EX,1,{self.get_property('Youngs_Modulus_GPa', temp_k)*1e9:.3e}  ! Young's Modulus (Pa)\n")
            f.write(f"MP,NUXY,1,{self.get_property('Poissons_Ratio', temp_k):.3f}      ! Poisson's Ratio\n\n")
            
            f.write("! Thermal Properties\n")
            f.write(f"MP,ALPX,1,{self.get_property('CTE_1e-6_per_K', temp_k)*1e-6:.3e}  ! CTE (1/K)\n")
            f.write(f"MP,KXX,1,{self.get_property('Thermal_Conductivity_W_mK', temp_k):.3f}  ! Thermal Conductivity (W/m·K)\n")
            f.write(f"MP,DENS,1,{self.get_property('Density_kg_m3', temp_k):.1f}  ! Density (kg/m³)\n\n")
            
            f.write("! Failure Properties\n")
            f.write(f"! Fracture Toughness: {self.get_property('Fracture_Toughness_MPa_sqrt_m', temp_k):.2f} MPa√m\n")
            f.write(f"! Weibull Modulus: {self.get_property('Weibull_Modulus', temp_k):.1f}\n")
            f.write(f"! Characteristic Strength: {self.get_property('Characteristic_Strength_MPa', temp_k):.1f} MPa\n\n")
            
            f.write("! Creep Properties (Norton Power Law)\n")
            f.write(f"! A = {self.get_property('Creep_Rate_Coefficient_A', temp_k):.3e}\n")
            f.write(f"! n = {self.get_property('Creep_Stress_Exponent_n', temp_k):.2f}\n")
            f.write(f"! Q = {self.get_property('Creep_Activation_Energy_kJ_mol', temp_k):.1f} kJ/mol\n")
        
        print(f"FEM input file generated: {filename}")

def main():
    """Main function to demonstrate usage"""
    print("YSZ Material Properties Dataset Generator")
    print("==========================================\n")
    
    # Initialize the material properties handler
    ysz = YSZMaterialProperties()
    
    # Display basic statistics
    print("Temperature Range: 25°C to 1500°C")
    print(f"Number of data points: {len(ysz.df)}")
    print(f"Properties included: {len(ysz.properties)}\n")
    
    # Example: Get properties at operating temperature (800°C)
    operating_temp_k = 800 + 273.15
    print("Properties at 800°C (typical SOFC operating temperature):")
    print("-" * 50)
    for prop in ['Youngs_Modulus_GPa', 'CTE_1e-6_per_K', 'Thermal_Conductivity_W_mK']:
        value = ysz.get_property(prop, operating_temp_k)
        print(f"{prop}: {value:.3f}")
    print()
    
    # Generate all outputs
    print("Generating outputs...")
    ysz.export_to_json()
    ysz.export_to_excel()
    ysz.generate_fem_input_file(800, 'fem_input_800C.txt')
    ysz.generate_fem_input_file(1500, 'fem_input_1500C.txt')
    ysz.plot_all_properties()
    
    print("\nDataset generation complete!")

if __name__ == "__main__":
    main()