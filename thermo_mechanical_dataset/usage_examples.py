#!/usr/bin/env python3
"""
Usage Examples for Thermo-Mechanical Modeling Dataset
Fire-Resistant Structural Elements with High-Performance Rubberized Concrete

This script demonstrates various ways to use the generated dataset for
finite element analysis and research applications.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class ThermoMechanicalDataHandler:
    """
    Comprehensive data handler for thermo-mechanical modeling dataset.
    Provides methods for data access, interpolation, and analysis.
    """
    
    def __init__(self, data_dir='.'):
        """Initialize the data handler with dataset location"""
        self.data_dir = data_dir
        self.thermal_df = None
        self.mechanical_df = None
        self.transport_df = None
        self.deformation_df = None
        self.stress_strain_curves = None
        self._load_data()
    
    def _load_data(self):
        """Load all dataset files"""
        try:
            self.thermal_df = pd.read_csv(f'{self.data_dir}/thermal_properties.csv')
            self.mechanical_df = pd.read_csv(f'{self.data_dir}/mechanical_properties.csv')
            self.transport_df = pd.read_csv(f'{self.data_dir}/transport_properties.csv')
            self.deformation_df = pd.read_csv(f'{self.data_dir}/deformation_properties.csv')
            
            with open(f'{self.data_dir}/stress_strain_curves.json', 'r') as f:
                self.stress_strain_curves = json.load(f)
            
            print("Dataset loaded successfully!")
            print(f"Thermal properties: {len(self.thermal_df)} data points")
            print(f"Mechanical properties: {len(self.mechanical_df)} data points")
            print(f"Transport properties: {len(self.transport_df)} data points")
            print(f"Deformation properties: {len(self.deformation_df)} data points")
            
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            print("Please ensure all dataset files are in the specified directory.")
    
    def get_property_at_temperature(self, mix_id, property_name, temperature, 
                                  data_type='all', return_std=False):
        """
        Get material property at specific temperature with optional uncertainty bounds.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier (C, R5S, R10S, R15S, R20S, R10L)
        property_name : str
            Name of the property to retrieve
        temperature : float
            Temperature in Celsius
        data_type : str
            'all', 'calibration', or 'validation'
        return_std : bool
            Whether to return standard deviation
        
        Returns:
        --------
        float or tuple
            Property value (and standard deviation if requested)
        """
        # Select appropriate dataframe based on property
        if property_name in self.thermal_df.columns:
            df = self.thermal_df
        elif property_name in self.mechanical_df.columns:
            df = self.mechanical_df
        elif property_name in self.transport_df.columns:
            df = self.transport_df
        elif property_name in self.deformation_df.columns:
            df = self.deformation_df
        else:
            raise ValueError(f"Property '{property_name}' not found in dataset")
        
        # Filter by mix and data type
        mask = df['Mix_ID'] == mix_id
        if data_type != 'all':
            mask &= df['Data_Type'] == data_type.capitalize()
        
        mix_data = df[mask]
        
        if len(mix_data) == 0:
            raise ValueError(f"No data found for mix '{mix_id}' and data_type '{data_type}'")
        
        # Create interpolation function
        f = interp1d(mix_data['Temperature_C'], mix_data[property_name], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        
        property_value = f(temperature)
        
        if return_std:
            std_name = property_name + '_Std'
            if std_name in mix_data.columns:
                f_std = interp1d(mix_data['Temperature_C'], mix_data[std_name],
                               kind='linear', bounds_error=False, fill_value='extrapolate')
                std_value = f_std(temperature)
                return property_value, std_value
            else:
                return property_value, 0.0
        else:
            return property_value
    
    def get_property_range(self, mix_id, property_name, temp_range, 
                          data_type='all', include_uncertainty=False):
        """
        Get material property over temperature range.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        property_name : str
            Name of the property to retrieve
        temp_range : array-like
            Temperature range in Celsius
        data_type : str
            'all', 'calibration', or 'validation'
        include_uncertainty : bool
            Whether to include uncertainty bounds
        
        Returns:
        --------
        dict
            Dictionary with temperature, property values, and optionally uncertainty
        """
        temperatures = np.array(temp_range)
        properties = np.zeros_like(temperatures)
        uncertainties = np.zeros_like(temperatures) if include_uncertainty else None
        
        for i, temp in enumerate(temperatures):
            if include_uncertainty:
                prop, std = self.get_property_at_temperature(mix_id, property_name, 
                                                           temp, data_type, return_std=True)
                properties[i] = prop
                uncertainties[i] = std
            else:
                properties[i] = self.get_property_at_temperature(mix_id, property_name, 
                                                               temp, data_type)
        
        result = {
            'temperature': temperatures,
            'property': properties
        }
        
        if include_uncertainty:
            result['uncertainty'] = uncertainties
        
        return result
    
    def get_stress_strain_curve(self, mix_id, temperature):
        """
        Get complete stress-strain curve for specific mix and temperature.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        temperature : float
            Temperature in Celsius (rounded to nearest 50°C increment)
        
        Returns:
        --------
        dict
            Dictionary with strain and stress arrays
        """
        # Round temperature to nearest 50°C increment
        temp_rounded = round(temperature / 50) * 50
        temp_rounded = max(20, min(770, temp_rounded))  # Clamp to valid range
        temp_rounded = f"{temp_rounded:.1f}"  # Convert to string with one decimal place
        
        if mix_id not in self.stress_strain_curves:
            raise ValueError(f"Mix '{mix_id}' not found in stress-strain curves")
        
        if temp_rounded not in self.stress_strain_curves[mix_id]:
            raise ValueError(f"Temperature {temp_rounded}°C not found for mix '{mix_id}'")
        
        curve_data = self.stress_strain_curves[mix_id][str(temp_rounded)]
        
        return {
            'strain': np.array(curve_data['strain']),
            'stress': np.array(curve_data['stress']),
            'fc': curve_data['fc'],
            'E': curve_data['E'],
            'temperature': temp_rounded
        }
    
    def get_uncertainty_bounds(self, mix_id, property_name, temperature, 
                              confidence=0.95):
        """
        Get statistical uncertainty bounds for a property.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        property_name : str
            Name of the property
        temperature : float
            Temperature in Celsius
        confidence : float
            Confidence level (0.95 for 95%, 0.99 for 99%)
        
        Returns:
        --------
        tuple
            (mean, lower_bound, upper_bound)
        """
        mean_val, std_val = self.get_property_at_temperature(mix_id, property_name, 
                                                           temperature, return_std=True)
        
        # Calculate confidence bounds
        z_score = 1.96 if confidence == 0.95 else 2.58
        lower_bound = mean_val - z_score * std_val
        upper_bound = mean_val + z_score * std_val
        
        return mean_val, lower_bound, upper_bound
    
    def compare_mixes(self, property_name, temperature, mixes=None):
        """
        Compare a property across different concrete mixes.
        
        Parameters:
        -----------
        property_name : str
            Name of the property to compare
        temperature : float
            Temperature in Celsius
        mixes : list, optional
            List of mix IDs to compare (default: all mixes)
        
        Returns:
        --------
        pandas.DataFrame
            Comparison table with property values and uncertainties
        """
        if mixes is None:
            mixes = ['C', 'R5S', 'R10S', 'R15S', 'R20S', 'R10L']
        
        comparison_data = []
        
        for mix_id in mixes:
            try:
                mean_val, std_val = self.get_property_at_temperature(
                    mix_id, property_name, temperature, return_std=True)
                
                comparison_data.append({
                    'Mix_ID': mix_id,
                    'Property_Value': mean_val,
                    'Standard_Deviation': std_val,
                    'Coefficient_of_Variation': std_val / mean_val * 100,
                    'Temperature': temperature
                })
            except ValueError:
                print(f"Warning: Could not retrieve data for mix {mix_id}")
        
        return pd.DataFrame(comparison_data)
    
    def plot_property_evolution(self, mix_id, property_name, temp_range=None, 
                              data_type='all', include_uncertainty=True, 
                              save_plot=False, filename=None):
        """
        Plot property evolution with temperature.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        property_name : str
            Name of the property to plot
        temp_range : array-like, optional
            Temperature range (default: 20-800°C)
        data_type : str
            'all', 'calibration', or 'validation'
        include_uncertainty : bool
            Whether to include uncertainty bounds
        save_plot : bool
            Whether to save the plot
        filename : str, optional
            Filename for saved plot
        """
        if temp_range is None:
            temp_range = np.linspace(20, 800, 100)
        
        data = self.get_property_range(mix_id, property_name, temp_range, 
                                     data_type, include_uncertainty)
        
        plt.figure(figsize=(10, 6))
        
        if include_uncertainty and 'uncertainty' in data:
            plt.fill_between(data['temperature'], 
                           data['property'] - data['uncertainty'],
                           data['property'] + data['uncertainty'],
                           alpha=0.3, label='±1σ Uncertainty')
        
        plt.plot(data['temperature'], data['property'], 'b-', linewidth=2, 
                label=f'{mix_id} - {property_name}')
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel(property_name)
        plt.title(f'Property Evolution: {property_name} for Mix {mix_id}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_plot:
            if filename is None:
                filename = f'{mix_id}_{property_name}_evolution.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_stress_strain_curve(self, mix_id, temperature, save_plot=False, 
                               filename=None):
        """
        Plot stress-strain curve for specific mix and temperature.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        temperature : float
            Temperature in Celsius
        save_plot : bool
            Whether to save the plot
        filename : str, optional
            Filename for saved plot
        """
        curve_data = self.get_stress_strain_curve(mix_id, temperature)
        
        plt.figure(figsize=(10, 6))
        plt.plot(curve_data['strain'], curve_data['stress'], 'b-', linewidth=2)
        plt.xlabel('Strain')
        plt.ylabel('Stress (MPa)')
        plt.title(f'Stress-Strain Curve: Mix {mix_id} at {curve_data["temperature"]}°C')
        plt.grid(True, alpha=0.3)
        
        # Add peak stress annotation
        peak_idx = np.argmax(curve_data['stress'])
        plt.annotate(f'Peak: {curve_data["stress"][peak_idx]:.1f} MPa',
                    xy=(curve_data['strain'][peak_idx], curve_data['stress'][peak_idx]),
                    xytext=(0.3, 0.7), textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        if save_plot:
            if filename is None:
                filename = f'{mix_id}_stress_strain_{curve_data["temperature"]}C.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_abaqus_material_file(self, mix_id, output_file=None):
        """
        Generate ABAQUS material input file for specific mix.
        
        Parameters:
        -----------
        mix_id : str
            Concrete mix identifier
        output_file : str, optional
            Output filename (default: material_{mix_id}.inp)
        """
        if output_file is None:
            output_file = f'material_{mix_id}.inp'
        
        # Get properties at 20°C for initial values
        E = self.get_property_at_temperature(mix_id, 'Elastic_Modulus_GPa', 20) * 1e9
        nu = self.get_property_at_temperature(mix_id, 'Poissons_Ratio', 20)
        rho = self.get_property_at_temperature(mix_id, 'Density_kg_m3', 20)
        k = self.get_property_at_temperature(mix_id, 'Thermal_Conductivity_W_mK', 20)
        cp = self.get_property_at_temperature(mix_id, 'Specific_Heat_J_kgK', 20)
        alpha = self.get_property_at_temperature(mix_id, 'Thermal_Expansion_1_K', 20)
        
        with open(output_file, 'w') as f:
            f.write(f"*MATERIAL, NAME=CONCRETE_{mix_id}\n")
            f.write("*ELASTIC\n")
            f.write(f"{E:.2e}, {nu:.3f}\n")
            f.write("*THERMAL CONDUCTIVITY\n")
            f.write(f"{k:.3f}\n")
            f.write("*SPECIFIC HEAT\n")
            f.write(f"{cp:.1f}\n")
            f.write("*DENSITY\n")
            f.write(f"{rho:.1f}\n")
            f.write("*EXPANSION\n")
            f.write(f"{alpha:.2e}\n")
            f.write("*PERMEABILITY\n")
            f.write("*CREEP\n")
            f.write("*DAMAGE INITIATION, CRITERION=MAXIMUM PRINCIPAL STRESS\n")
            f.write("*DAMAGE EVOLUTION, TYPE=ENERGY\n")
        
        print(f"ABAQUS material file generated: {output_file}")

def example_usage():
    """Demonstrate various usage examples"""
    print("Thermo-Mechanical Dataset Usage Examples")
    print("=" * 50)
    
    # Initialize data handler
    handler = ThermoMechanicalDataHandler()
    
    # Example 1: Get property at specific temperature
    print("\n1. Getting property at specific temperature:")
    fc = handler.get_property_at_temperature('R10S', 'Compressive_Strength_MPa', 400)
    print(f"Compressive strength of R10S at 400°C: {fc:.2f} MPa")
    
    # Example 2: Get property with uncertainty bounds
    print("\n2. Getting property with uncertainty bounds:")
    fc_mean, fc_lower, fc_upper = handler.get_uncertainty_bounds('R10S', 'Compressive_Strength_MPa', 400)
    print(f"Compressive strength of R10S at 400°C: {fc_mean:.2f} ± {fc_upper-fc_mean:.2f} MPa")
    
    # Example 3: Compare mixes
    print("\n3. Comparing mixes at 400°C:")
    comparison = handler.compare_mixes('Compressive_Strength_MPa', 400)
    print(comparison[['Mix_ID', 'Property_Value', 'Coefficient_of_Variation']])
    
    # Example 4: Get stress-strain curve
    print("\n4. Getting stress-strain curve:")
    curve = handler.get_stress_strain_curve('R10S', 20)  # Use room temperature
    print(f"Peak stress at {curve['temperature']}°C: {curve['fc']:.2f} MPa")
    print(f"Elastic modulus at {curve['temperature']}°C: {curve['E']:.2f} GPa")
    
    # Example 5: Plot property evolution
    print("\n5. Plotting property evolution:")
    handler.plot_property_evolution('R10S', 'Compressive_Strength_MPa', 
                                  include_uncertainty=True)
    
    # Example 6: Generate ABAQUS material file
    print("\n6. Generating ABAQUS material file:")
    handler.generate_abaqus_material_file('R10S', 'example_material_R10S.inp')
    
    print("\nExamples completed successfully!")

if __name__ == "__main__":
    example_usage()