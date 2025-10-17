#!/usr/bin/env python3
"""
Residual Properties Data Generator
Generates realistic residual mechanical properties after heating and cooling
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class ResidualPropertiesGenerator:
    def __init__(self):
        self.temperatures = [25, 200, 400, 600, 800]  # Exposure temperatures
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
        # Ambient properties
        self.ambient_properties = {
            'control': {
                'compressive_strength': 40.0,
                'tensile_strength': 3.5,
                'elastic_modulus': 30000,
                'upv': 4500,  # m/s
                'dynamic_modulus': 35.0  # GPa
            },
            'rubber_10': {
                'compressive_strength': 35.0,
                'tensile_strength': 3.0,
                'elastic_modulus': 28000,
                'upv': 4200,
                'dynamic_modulus': 32.0
            },
            'rubber_20': {
                'compressive_strength': 30.0,
                'tensile_strength': 2.5,
                'elastic_modulus': 25000,
                'upv': 3900,
                'dynamic_modulus': 28.0
            },
            'rubber_30': {
                'compressive_strength': 25.0,
                'tensile_strength': 2.0,
                'elastic_modulus': 22000,
                'upv': 3600,
                'dynamic_modulus': 24.0
            }
        }
    
    def calculate_residual_factor(self, property_type, temperature, mix_type):
        """Calculate residual property factor after heating and cooling"""
        # Base residual factors by temperature
        base_factors = {
            25: 1.0,
            200: 0.95,
            400: 0.80,
            600: 0.50,
            800: 0.25
        }
        
        # Property-specific adjustments
        property_adjustments = {
            'compressive_strength': 1.0,
            'tensile_strength': 0.9,  # Tensile more sensitive
            'elastic_modulus': 0.95,  # Modulus less sensitive
            'upv': 1.0,
            'dynamic_modulus': 0.95
        }
        
        # Mix-specific adjustments (rubber content effects)
        mix_adjustments = {
            'control': 0,
            'rubber_10': -0.05,
            'rubber_20': -0.10,
            'rubber_30': -0.15
        }
        
        # Calculate residual factor
        base_factor = base_factors[temperature]
        property_factor = property_adjustments[property_type]
        mix_factor = 1 + mix_adjustments[mix_type]
        
        residual_factor = base_factor * property_factor * mix_factor
        
        # Ensure reasonable bounds
        residual_factor = max(residual_factor, 0.05)  # Minimum 5% retention
        residual_factor = min(residual_factor, 1.0)   # Maximum 100% retention
        
        return residual_factor
    
    def generate_residual_properties(self, mix_type, temperature, replicate=1):
        """Generate residual properties for a specific mix and temperature"""
        ambient = self.ambient_properties[mix_type]
        
        # Add replicate variation
        replicate_factor = 1 + np.random.normal(0, 0.05)  # 5% variation
        
        properties = {}
        for prop_name, ambient_value in ambient.items():
            residual_factor = self.calculate_residual_factor(prop_name, temperature, mix_type)
            residual_value = ambient_value * residual_factor * replicate_factor
            
            # Ensure positive values
            residual_value = max(residual_value, 0.1)
            
            properties[prop_name] = residual_value
        
        # Calculate additional derived properties
        properties['residual_factor_compressive'] = properties['compressive_strength'] / ambient['compressive_strength']
        properties['residual_factor_tensile'] = properties['tensile_strength'] / ambient['tensile_strength']
        properties['residual_factor_modulus'] = properties['elastic_modulus'] / ambient['elastic_modulus']
        
        # Add metadata
        properties.update({
            'mix_type': mix_type,
            'exposure_temperature': temperature,
            'replicate': replicate,
            'cooling_method': 'Natural air cooling',
            'cooling_time': '24 hours'
        })
        
        return properties
    
    def generate_mix_data(self, mix_type, replicates=5):
        """Generate residual properties data for a mix at all temperatures"""
        mix_data = {
            'mix_type': mix_type,
            'ambient_properties': self.ambient_properties[mix_type],
            'temperatures': {}
        }
        
        for temp in self.temperatures:
            temp_data = {
                'exposure_temperature': temp,
                'replicates': []
            }
            
            for replicate in range(replicates):
                properties = self.generate_residual_properties(mix_type, temp, replicate + 1)
                temp_data['replicates'].append(properties)
            
            mix_data['temperatures'][temp] = temp_data
        
        return mix_data
    
    def generate_all_data(self):
        """Generate residual properties data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'cooling_method': 'Natural air cooling',
                'cooling_time': '24 hours to ambient temperature',
                'specimen_size': '100mm × 100mm × 100mm cubes',
                'replicates_per_temperature': 5,
                'test_standards': {
                    'compressive_strength': 'ASTM C39',
                    'tensile_strength': 'ASTM C496',
                    'elastic_modulus': 'ASTM C469',
                    'upv': 'ASTM C597',
                    'dynamic_modulus': 'ASTM C215'
                }
            },
            'mixes': {}
        }
        
        for mix in self.mixes:
            all_data['mixes'][mix] = self.generate_mix_data(mix)
        
        return all_data
    
    def save_data(self, data, output_dir):
        """Save generated data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f"{output_dir}/residual_properties_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix
        for mix, mix_data in data['mixes'].items():
            rows = []
            for temp, temp_data in mix_data['temperatures'].items():
                for replicate_data in temp_data['replicates']:
                    rows.append(replicate_data)
            
            df = pd.DataFrame(rows)
            df.to_csv(f"{output_dir}/{mix}_residual_properties.csv", index=False)
    
    def plot_residual_properties(self, data, output_dir):
        """Generate plots of residual properties"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Residual compressive strength vs temperature
        fig, ax = plt.subplots(figsize=(10, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                strengths = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    # Average across replicates
                    strengths_list = [r['compressive_strength'] for r in temp_data['replicates']]
                    temps.append(temp)
                    strengths.append(np.mean(strengths_list))
                ax.plot(temps, strengths, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Exposure Temperature (°C)')
        ax.set_ylabel('Residual Compressive Strength (MPa)')
        ax.set_title('Residual Compressive Strength vs Exposure Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/residual_compressive_strength.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Residual factor vs temperature
        fig, ax = plt.subplots(figsize=(10, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                factors = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    # Average across replicates
                    factors_list = [r['residual_factor_compressive'] for r in temp_data['replicates']]
                    temps.append(temp)
                    factors.append(np.mean(factors_list))
                ax.plot(temps, factors, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Exposure Temperature (°C)')
        ax.set_ylabel('Residual Factor')
        ax.set_title('Residual Factor vs Exposure Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/residual_factor.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: All residual properties (control mix)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            
            # Compressive strength
            ax1 = axes[0, 0]
            temps = []
            strengths = []
            for temp, temp_data in control_data['temperatures'].items():
                strengths_list = [r['compressive_strength'] for r in temp_data['replicates']]
                temps.append(temp)
                strengths.append(np.mean(strengths_list))
            ax1.plot(temps, strengths, 'o-', linewidth=2, markersize=6)
            ax1.set_xlabel('Exposure Temperature (°C)')
            ax1.set_ylabel('Compressive Strength (MPa)')
            ax1.set_title('Residual Compressive Strength')
            ax1.grid(True)
            
            # Elastic modulus
            ax2 = axes[0, 1]
            temps = []
            moduli = []
            for temp, temp_data in control_data['temperatures'].items():
                moduli_list = [r['elastic_modulus'] for r in temp_data['replicates']]
                temps.append(temp)
                moduli.append(np.mean(moduli_list))
            ax2.plot(temps, moduli, 'o-', linewidth=2, markersize=6)
            ax2.set_xlabel('Exposure Temperature (°C)')
            ax2.set_ylabel('Elastic Modulus (MPa)')
            ax2.set_title('Residual Elastic Modulus')
            ax2.grid(True)
            
            # UPV
            ax3 = axes[1, 0]
            temps = []
            upv_values = []
            for temp, temp_data in control_data['temperatures'].items():
                upv_list = [r['upv'] for r in temp_data['replicates']]
                temps.append(temp)
                upv_values.append(np.mean(upv_list))
            ax3.plot(temps, upv_values, 'o-', linewidth=2, markersize=6)
            ax3.set_xlabel('Exposure Temperature (°C)')
            ax3.set_ylabel('UPV (m/s)')
            ax3.set_title('Residual UPV')
            ax3.grid(True)
            
            # Dynamic modulus
            ax4 = axes[1, 1]
            temps = []
            dyn_moduli = []
            for temp, temp_data in control_data['temperatures'].items():
                dyn_moduli_list = [r['dynamic_modulus'] for r in temp_data['replicates']]
                temps.append(temp)
                dyn_moduli.append(np.mean(dyn_moduli_list))
            ax4.plot(temps, dyn_moduli, 'o-', linewidth=2, markersize=6)
            ax4.set_xlabel('Exposure Temperature (°C)')
            ax4.set_ylabel('Dynamic Modulus (GPa)')
            ax4.set_title('Residual Dynamic Modulus')
            ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/all_residual_properties_control.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = ResidualPropertiesGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/mechanical_testing/residual_properties")
    generator.plot_residual_properties(data, "/workspace/experimental_dataset/mechanical_testing/residual_properties")
    print("Residual properties data generation completed!")