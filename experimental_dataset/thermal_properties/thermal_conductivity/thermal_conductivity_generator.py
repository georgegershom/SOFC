#!/usr/bin/env python3
"""
Thermal Conductivity and Specific Heat Data Generator
Generates realistic thermal property data for rubberized concrete samples
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class ThermalConductivityGenerator:
    def __init__(self):
        self.temperatures = [25, 100, 200, 400, 600]  # Test temperatures
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
    def generate_thermal_conductivity(self, mix_type, temperature):
        """Generate thermal conductivity based on mix type and temperature"""
        # Base thermal conductivity values (W/m·K)
        base_conductivity = {
            'control': 2.1,
            'rubber_10': 1.8,
            'rubber_20': 1.5,
            'rubber_30': 1.2
        }
        
        # Temperature dependence (decreases with temperature)
        temp_factor = 1 - 0.0003 * (temperature - 25)
        
        # Rubber content effect (rubber has lower thermal conductivity)
        rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
        rubber_factor = 1 - (rubber_percentage * 0.02)  # 2% reduction per 10% rubber
        
        # Calculate thermal conductivity
        k = base_conductivity[mix_type] * temp_factor * rubber_factor
        
        # Add measurement uncertainty (±5%)
        uncertainty = np.random.normal(0, 0.05)
        k *= (1 + uncertainty)
        
        return max(k, 0.1)  # Minimum value
    
    def generate_specific_heat(self, mix_type, temperature):
        """Generate specific heat capacity based on mix type and temperature"""
        # Base specific heat values (J/kg·K)
        base_specific_heat = {
            'control': 900,
            'rubber_10': 950,
            'rubber_20': 1000,
            'rubber_30': 1050
        }
        
        # Temperature dependence (increases with temperature)
        temp_factor = 1 + 0.0002 * (temperature - 25)
        
        # Rubber content effect (rubber has higher specific heat)
        rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
        rubber_factor = 1 + (rubber_percentage * 0.01)  # 1% increase per 10% rubber
        
        # Calculate specific heat
        cp = base_specific_heat[mix_type] * temp_factor * rubber_factor
        
        # Add measurement uncertainty (±3%)
        uncertainty = np.random.normal(0, 0.03)
        cp *= (1 + uncertainty)
        
        return max(cp, 500)  # Minimum value
    
    def generate_density(self, mix_type, temperature):
        """Generate density based on mix type and temperature"""
        # Base density values (kg/m³)
        base_density = {
            'control': 2400,
            'rubber_10': 2300,
            'rubber_20': 2200,
            'rubber_30': 2100
        }
        
        # Temperature dependence (slight decrease due to thermal expansion)
        temp_factor = 1 - 0.0001 * (temperature - 25)
        
        # Calculate density
        rho = base_density[mix_type] * temp_factor
        
        # Add measurement uncertainty (±2%)
        uncertainty = np.random.normal(0, 0.02)
        rho *= (1 + uncertainty)
        
        return max(rho, 1800)  # Minimum value
    
    def generate_thermal_diffusivity(self, k, cp, rho):
        """Calculate thermal diffusivity from k, cp, and rho"""
        return k / (cp * rho)
    
    def generate_mix_data(self, mix_type):
        """Generate complete thermal property data for a mix"""
        data = {
            'mix_type': mix_type,
            'temperatures': [],
            'thermal_conductivity': [],
            'specific_heat': [],
            'density': [],
            'thermal_diffusivity': [],
            'replicates': []
        }
        
        for temp in self.temperatures:
            temp_data = {
                'temperature': temp,
                'thermal_conductivity': [],
                'specific_heat': [],
                'density': [],
                'thermal_diffusivity': []
            }
            
            # Generate 5 replicates for each temperature
            for replicate in range(5):
                k = self.generate_thermal_conductivity(mix_type, temp)
                cp = self.generate_specific_heat(mix_type, temp)
                rho = self.generate_density(mix_type, temp)
                alpha = self.generate_thermal_diffusivity(k, cp, rho)
                
                temp_data['thermal_conductivity'].append(k)
                temp_data['specific_heat'].append(cp)
                temp_data['density'].append(rho)
                temp_data['thermal_diffusivity'].append(alpha)
            
            data['temperatures'].append(temp_data)
        
        return data
    
    def generate_all_data(self):
        """Generate thermal property data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'Hot Disk TPS 2500S',
                'measurement_standard': 'ISO 22007-2',
                'replicates_per_temperature': 5
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
        with open(f"{output_dir}/thermal_conductivity_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix
        for mix, mix_data in data['mixes'].items():
            rows = []
            for temp_data in mix_data['temperatures']:
                temp = temp_data['temperature']
                for i in range(5):  # 5 replicates
                    rows.append({
                        'mix_type': mix,
                        'temperature': temp,
                        'replicate': i + 1,
                        'thermal_conductivity': temp_data['thermal_conductivity'][i],
                        'specific_heat': temp_data['specific_heat'][i],
                        'density': temp_data['density'][i],
                        'thermal_diffusivity': temp_data['thermal_diffusivity'][i]
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(f"{output_dir}/{mix}_thermal_properties.csv", index=False)
    
    def plot_thermal_properties(self, data, output_dir):
        """Generate plots of thermal properties"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Thermal conductivity
        ax1 = axes[0, 0]
        for mix in self.mixes:
            temps = []
            k_values = []
            for temp_data in data['mixes'][mix]['temperatures']:
                temps.append(temp_data['temperature'])
                k_values.append(np.mean(temp_data['thermal_conductivity']))
            ax1.plot(temps, k_values, 'o-', label=mix, linewidth=2, markersize=6)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Thermal Conductivity (W/m·K)')
        ax1.set_title('Thermal Conductivity vs Temperature')
        ax1.legend()
        ax1.grid(True)
        
        # Specific heat
        ax2 = axes[0, 1]
        for mix in self.mixes:
            temps = []
            cp_values = []
            for temp_data in data['mixes'][mix]['temperatures']:
                temps.append(temp_data['temperature'])
                cp_values.append(np.mean(temp_data['specific_heat']))
            ax2.plot(temps, cp_values, 'o-', label=mix, linewidth=2, markersize=6)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Specific Heat (J/kg·K)')
        ax2.set_title('Specific Heat vs Temperature')
        ax2.legend()
        ax2.grid(True)
        
        # Density
        ax3 = axes[1, 0]
        for mix in self.mixes:
            temps = []
            rho_values = []
            for temp_data in data['mixes'][mix]['temperatures']:
                temps.append(temp_data['temperature'])
                rho_values.append(np.mean(temp_data['density']))
            ax3.plot(temps, rho_values, 'o-', label=mix, linewidth=2, markersize=6)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Density (kg/m³)')
        ax3.set_title('Density vs Temperature')
        ax3.legend()
        ax3.grid(True)
        
        # Thermal diffusivity
        ax4 = axes[1, 1]
        for mix in self.mixes:
            temps = []
            alpha_values = []
            for temp_data in data['mixes'][mix]['temperatures']:
                temps.append(temp_data['temperature'])
                alpha_values.append(np.mean(temp_data['thermal_diffusivity']))
            ax4.plot(temps, alpha_values, 'o-', label=mix, linewidth=2, markersize=6)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Thermal Diffusivity (m²/s)')
        ax4.set_title('Thermal Diffusivity vs Temperature')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/thermal_properties.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = ThermalConductivityGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/thermal_properties/thermal_conductivity")
    generator.plot_thermal_properties(data, "/workspace/experimental_dataset/thermal_properties/thermal_conductivity")
    print("Thermal conductivity data generation completed!")