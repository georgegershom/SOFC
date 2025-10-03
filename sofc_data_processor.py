#!/usr/bin/env python3
"""
SOFC Material Properties Data Processor

This script provides utilities to work with the SOFC material properties dataset.
It can load, process, and interpolate material properties for different temperatures.
"""

import json
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class SOFCMaterialDatabase:
    """Class to handle SOFC material properties data."""
    
    def __init__(self, json_file: str = "sofc_material_properties.json"):
        """Initialize the database with material properties."""
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.materials = self.data['materials']
        self.temperatures = [25, 500, 800, 1000]  # °C
    
    def get_material_property(self, material: str, property_name: str, 
                            temperature: float) -> float:
        """
        Get a material property at a specific temperature using linear interpolation.
        
        Args:
            material: Material name (e.g., 'anode_ni_ysz')
            property_name: Property name (e.g., 'thermal_conductivity')
            temperature: Temperature in °C
        
        Returns:
            Interpolated property value
        """
        if material not in self.materials:
            raise ValueError(f"Material {material} not found")
        
        material_data = self.materials[material]
        
        # Navigate to the property
        if property_name in material_data['thermo_physical']:
            prop_data = material_data['thermo_physical'][property_name]
        elif property_name in material_data['mechanical']:
            prop_data = material_data['mechanical'][property_name]
        elif property_name in material_data['electrochemical']:
            prop_data = material_data['electrochemical'][property_name]
        else:
            raise ValueError(f"Property {property_name} not found for {material}")
        
        # Handle temperature-dependent properties
        if isinstance(prop_data, dict) and any('°C' in str(k) for k in prop_data.keys()):
            temps = []
            values = []
            for temp_str, value in prop_data.items():
                temp = float(temp_str.replace('°C', ''))
                temps.append(temp)
                values.append(value)
            
            # Sort by temperature
            sorted_data = sorted(zip(temps, values))
            temps, values = zip(*sorted_data)
            
            # Interpolate
            if temperature < min(temps):
                return values[0]
            elif temperature > max(temps):
                return values[-1]
            else:
                f = interp1d(temps, values, kind='linear')
                return float(f(temperature))
        else:
            # Constant property
            return prop_data
    
    def get_all_properties(self, material: str, temperature: float) -> Dict:
        """Get all properties for a material at a specific temperature."""
        properties = {}
        material_data = self.materials[material]
        
        # Thermo-physical properties
        for prop in material_data['thermo_physical']:
            properties[prop] = self.get_material_property(material, prop, temperature)
        
        # Mechanical properties
        for prop in material_data['mechanical']:
            if prop != 'creep_parameters' and prop != 'plasticity_parameters':
                properties[prop] = self.get_material_property(material, prop, temperature)
        
        # Electrochemical properties
        for prop in material_data['electrochemical']:
            if prop != 'activation_overpotential':
                properties[prop] = self.get_material_property(material, prop, temperature)
        
        return properties
    
    def plot_property_vs_temperature(self, material: str, property_name: str, 
                                   temperature_range: Tuple[float, float] = (25, 1000),
                                   num_points: int = 100):
        """Plot a property vs temperature for a material."""
        temps = np.linspace(temperature_range[0], temperature_range[1], num_points)
        values = [self.get_material_property(material, property_name, t) for t in temps]
        
        plt.figure(figsize=(10, 6))
        plt.plot(temps, values, 'b-', linewidth=2)
        plt.xlabel('Temperature (°C)')
        plt.ylabel(f'{property_name.replace("_", " ").title()}')
        plt.title(f'{self.materials[material]["name"]} - {property_name.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def compare_materials(self, property_name: str, temperature: float = 800):
        """Compare a property across all materials at a specific temperature."""
        comparison = {}
        for material_name, material_data in self.materials.items():
            try:
                value = self.get_material_property(material_name, property_name, temperature)
                comparison[material_data['name']] = value
            except (ValueError, KeyError):
                continue
        
        return comparison
    
    def export_to_dataframe(self, temperature: float = 800) -> pd.DataFrame:
        """Export all material properties to a pandas DataFrame."""
        data = []
        for material_name, material_data in self.materials.items():
            row = {'Material': material_data['name'], 'Temperature': temperature}
            properties = self.get_all_properties(material_name, temperature)
            row.update(properties)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def calculate_thermal_stress(self, material: str, temperature: float, 
                               reference_temp: float = 25) -> float:
        """Calculate thermal stress assuming constrained thermal expansion."""
        alpha = self.get_material_property(material, 'thermal_expansion_coefficient', temperature)
        E = self.get_material_property(material, 'youngs_modulus', temperature)
        nu = self.get_material_property(material, 'poisson_ratio', temperature)
        
        delta_T = temperature - reference_temp
        thermal_strain = alpha * delta_T
        thermal_stress = E * thermal_strain / (1 - nu)
        
        return thermal_stress
    
    def get_creep_parameters(self, material: str) -> Dict:
        """Get creep parameters for a material."""
        if material not in self.materials:
            raise ValueError(f"Material {material} not found")
        
        creep_data = self.materials[material]['mechanical'].get('creep_parameters', {})
        return creep_data
    
    def calculate_creep_strain_rate(self, material: str, stress: float, 
                                  temperature: float) -> float:
        """Calculate creep strain rate using Norton-Bailey law."""
        creep_params = self.get_creep_parameters(material)
        if 'norton_bailey' not in creep_params:
            raise ValueError(f"No Norton-Bailey parameters for {material}")
        
        B = creep_params['norton_bailey']['B']
        n = creep_params['norton_bailey']['n']
        Q = creep_params['norton_bailey']['Q'] * 1000  # Convert to J/mol
        
        R = 8.314  # Gas constant J/mol·K
        T_K = temperature + 273.15  # Convert to Kelvin
        
        strain_rate = B * (stress ** n) * np.exp(-Q / (R * T_K))
        return strain_rate

def main():
    """Example usage of the SOFC Material Database."""
    # Initialize database
    db = SOFCMaterialDatabase()
    
    print("SOFC Material Properties Database")
    print("=" * 40)
    
    # Example 1: Get thermal conductivity of Ni-YSZ at 600°C
    thermal_cond = db.get_material_property('anode_ni_ysz', 'thermal_conductivity', 600)
    print(f"Ni-YSZ thermal conductivity at 600°C: {thermal_cond:.2f} W/m·K")
    
    # Example 2: Get all properties for 8YSZ at 800°C
    print("\n8YSZ properties at 800°C:")
    ysz_props = db.get_all_properties('electrolyte_8ysz', 800)
    for prop, value in ysz_props.items():
        print(f"  {prop}: {value}")
    
    # Example 3: Compare thermal expansion coefficients
    print("\nThermal expansion coefficients at 800°C:")
    expansion_comparison = db.compare_materials('thermal_expansion_coefficient', 800)
    for material, value in expansion_comparison.items():
        print(f"  {material}: {value:.2e} 1/K")
    
    # Example 4: Calculate thermal stress
    thermal_stress = db.calculate_thermal_stress('anode_ni_ysz', 800)
    print(f"\nNi-YSZ thermal stress (25°C to 800°C): {thermal_stress/1e6:.1f} MPa")
    
    # Example 5: Calculate creep strain rate
    creep_rate = db.calculate_creep_strain_rate('anode_ni_ysz', 10, 800)  # 10 MPa, 800°C
    print(f"Ni-YSZ creep strain rate (10 MPa, 800°C): {creep_rate:.2e} 1/s")
    
    # Example 6: Export to DataFrame
    df = db.export_to_dataframe(800)
    print(f"\nDataFrame shape: {df.shape}")
    print("Available materials:", df['Material'].tolist())

if __name__ == "__main__":
    main()