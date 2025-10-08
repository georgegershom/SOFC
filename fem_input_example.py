#!/usr/bin/env python3
"""
SOFC Material Properties Database - FEM Input Generator
Example script showing how to load and interpolate material properties for FEM analysis
"""

import json
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt

class SOFCMaterialDatabase:
    """Class to manage SOFC material properties for FEM analysis"""
    
    def __init__(self, json_file='material_properties_database.json'):
        """Initialize database from JSON file"""
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.materials = self.data['materials']
        
    def get_material_list(self):
        """Return list of available materials"""
        return list(self.materials.keys())
    
    def interpolate_property(self, material, property_name, temperature):
        """
        Interpolate material property at given temperature
        
        Args:
            material: Material key (e.g., 'YSZ_8mol')
            property_name: Property to interpolate (e.g., 'youngs_modulus_GPa')
            temperature: Temperature in Celsius
            
        Returns:
            Interpolated property value
        """
        mat_data = self.materials[material]['temperature_dependent_properties']
        temps = np.array(mat_data['temperature_C'])
        values = np.array(mat_data[property_name])
        
        # Create interpolation function
        f_interp = interpolate.interp1d(temps, values, kind='linear', 
                                       fill_value='extrapolate')
        
        return float(f_interp(temperature))
    
    def get_properties_at_temperature(self, material, temperature):
        """
        Get all properties at a specific temperature
        
        Args:
            material: Material key
            temperature: Temperature in Celsius
            
        Returns:
            Dictionary of properties at given temperature
        """
        mat_data = self.materials[material]['temperature_dependent_properties']
        properties = {}
        
        for prop_name in mat_data.keys():
            if prop_name != 'temperature_C':
                properties[prop_name] = self.interpolate_property(material, 
                                                                 prop_name, 
                                                                 temperature)
        
        # Add temperature-independent properties
        if 'weibull_parameters' in self.materials[material]:
            properties['weibull_modulus'] = self.materials[material]['weibull_parameters']['modulus']
            
        if 'creep_parameters' in self.materials[material]:
            creep = self.materials[material]['creep_parameters']
            properties['creep_A'] = creep.get('A_prefactor_MPa-n_per_s', None)
            properties['creep_n'] = creep.get('n_stress_exponent', None)
            properties['creep_Q'] = creep.get('Q_activation_energy_kJ_per_mol', None)
            
        return properties
    
    def export_for_ansys(self, material, temp_points=None, output_file=None):
        """
        Export material properties in ANSYS APDL format
        
        Args:
            material: Material key
            temp_points: List of temperatures for export (default: use all)
            output_file: Output filename (default: material_ansys.txt)
        """
        if output_file is None:
            output_file = f"{material}_ansys.txt"
            
        mat_data = self.materials[material]['temperature_dependent_properties']
        
        if temp_points is None:
            temp_points = mat_data['temperature_C']
        
        with open(output_file, 'w') as f:
            f.write(f"! ANSYS APDL Material Definition for {material}\n")
            f.write(f"! {self.materials[material]['name']}\n\n")
            
            # Young's Modulus
            f.write("! Young's Modulus vs Temperature\n")
            for i, temp in enumerate(temp_points):
                E = self.interpolate_property(material, 'youngs_modulus_GPa', temp)
                f.write(f"MP,EX,1,{E*1e9},{temp+273.15}  ! {temp}°C\n")
            
            f.write("\n! Poisson's Ratio vs Temperature\n")
            for i, temp in enumerate(temp_points):
                nu = self.interpolate_property(material, 'poissons_ratio', temp)
                f.write(f"MP,NUXY,1,{nu},{temp+273.15}  ! {temp}°C\n")
            
            f.write("\n! Thermal Expansion Coefficient vs Temperature\n")
            for i, temp in enumerate(temp_points):
                cte = self.interpolate_property(material, 
                                               'thermal_expansion_coefficient_1e-6_per_K', temp)
                f.write(f"MP,ALPX,1,{cte*1e-6},{temp+273.15}  ! {temp}°C\n")
            
            f.write("\n! Density vs Temperature\n")
            for i, temp in enumerate(temp_points):
                rho = self.interpolate_property(material, 'density_kg_per_m3', temp)
                f.write(f"MP,DENS,1,{rho},{temp+273.15}  ! {temp}°C\n")
            
            f.write("\n! Thermal Conductivity vs Temperature\n")
            for i, temp in enumerate(temp_points):
                k = self.interpolate_property(material, 'thermal_conductivity_W_per_mK', temp)
                f.write(f"MP,KXX,1,{k},{temp+273.15}  ! {temp}°C\n")
        
        print(f"ANSYS material file exported: {output_file}")
    
    def plot_property_comparison(self, materials, property_name, temp_range=None):
        """
        Plot property comparison for multiple materials
        
        Args:
            materials: List of material keys
            property_name: Property to plot
            temp_range: Temperature range [min, max] (default: full range)
        """
        plt.figure(figsize=(10, 6))
        
        for material in materials:
            mat_data = self.materials[material]['temperature_dependent_properties']
            temps = np.array(mat_data['temperature_C'])
            values = np.array(mat_data[property_name])
            
            if temp_range:
                mask = (temps >= temp_range[0]) & (temps <= temp_range[1])
                temps = temps[mask]
                values = values[mask]
            
            plt.plot(temps, values, marker='o', label=self.materials[material]['name'])
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel(property_name.replace('_', ' ').title())
        plt.title(f'{property_name.replace("_", " ").title()} vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{property_name}_comparison.png', dpi=150)
        plt.show()
        
    def calculate_thermal_stress(self, material, delta_T, constrained=True):
        """
        Calculate thermal stress for given temperature change
        
        Args:
            material: Material key
            delta_T: Temperature change in K
            constrained: If True, assume fully constrained (no strain allowed)
            
        Returns:
            Thermal stress in MPa
        """
        # Get average properties over temperature range
        T_ref = 25  # Reference temperature
        T_final = T_ref + delta_T
        
        # Average properties
        E_avg = (self.interpolate_property(material, 'youngs_modulus_GPa', T_ref) + 
                self.interpolate_property(material, 'youngs_modulus_GPa', T_final)) / 2
        
        nu_avg = (self.interpolate_property(material, 'poissons_ratio', T_ref) + 
                 self.interpolate_property(material, 'poissons_ratio', T_final)) / 2
        
        alpha_avg = (self.interpolate_property(material, 
                    'thermal_expansion_coefficient_1e-6_per_K', T_ref) + 
                    self.interpolate_property(material, 
                    'thermal_expansion_coefficient_1e-6_per_K', T_final)) / 2
        
        if constrained:
            # Fully constrained: σ = E * α * ΔT / (1 - ν)
            stress = E_avg * 1e3 * alpha_avg * 1e-6 * delta_T / (1 - nu_avg)
        else:
            # Partially constrained: σ = E * α * ΔT
            stress = E_avg * 1e3 * alpha_avg * 1e-6 * delta_T
        
        return stress

def example_usage():
    """Example usage of the material database"""
    
    # Initialize database
    db = SOFCMaterialDatabase('material_properties_database.json')
    
    # Get list of materials
    print("Available materials:")
    for mat in db.get_material_list():
        print(f"  - {mat}: {db.materials[mat]['name']}")
    
    # Get properties at specific temperature
    print("\n8YSZ Properties at 800°C:")
    props_800 = db.get_properties_at_temperature('YSZ_8mol', 800)
    for prop, value in props_800.items():
        print(f"  {prop}: {value:.3f}")
    
    # Calculate thermal stress
    print("\nThermal Stress Calculations:")
    materials = ['YSZ_8mol', 'GDC_20', 'LSCF_6428']
    delta_T = 500  # 500K temperature change
    
    for mat in materials:
        stress = db.calculate_thermal_stress(mat, delta_T, constrained=True)
        print(f"  {mat}: {stress:.1f} MPa (for ΔT = {delta_T}K, fully constrained)")
    
    # Export for ANSYS
    print("\nExporting YSZ properties for ANSYS...")
    db.export_for_ansys('YSZ_8mol', temp_points=[25, 500, 800, 1000, 1200, 1400])
    
    # Plot property comparison
    print("\nGenerating property comparison plots...")
    db.plot_property_comparison(['YSZ_8mol', 'GDC_20', 'LSCF_6428'], 
                                'youngs_modulus_GPa', 
                                temp_range=[25, 1000])
    
    # Load CSV data
    print("\nLoading CSV data...")
    df = pd.read_csv('material_properties_all_SOFC.csv', comment='#')
    print(f"Loaded {len(df)} data points for {df['Material'].nunique()} materials")
    
    # Filter for specific material
    ysz_data = df[df['Material'] == 'YSZ-8mol']
    print(f"\nYSZ-8mol data points: {len(ysz_data)}")
    print(ysz_data[['Temperature(°C)', 'E(GPa)', 'CTE(10^-6/K)']].head())

if __name__ == "__main__":
    example_usage()