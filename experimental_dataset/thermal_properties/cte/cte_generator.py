#!/usr/bin/env python3
"""
Coefficient of Thermal Expansion (CTE) Data Generator
Generates realistic CTE data for rubberized concrete samples
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class CTEGenerator:
    def __init__(self):
        self.temperature_range = np.arange(25, 601, 1)  # 25°C to 600°C
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
    def generate_thermal_expansion_curve(self, mix_type):
        """Generate thermal expansion curve for a specific mix"""
        temp = self.temperature_range
        
        # Base CTE values (μm/m·°C)
        base_cte = {
            'control': 12.0,
            'rubber_10': 14.0,
            'rubber_20': 16.0,
            'rubber_30': 18.0
        }
        
        # Rubber content effect (rubber has higher CTE)
        rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
        rubber_factor = 1 + (rubber_percentage * 0.05)  # 5% increase per 10% rubber
        
        # Temperature-dependent CTE (increases with temperature)
        temp_factor = 1 + 0.0005 * (temp - 25)
        
        # Calculate instantaneous CTE
        instantaneous_cte = base_cte[mix_type] * rubber_factor * temp_factor
        
        # Add some realistic variations due to phase changes
        # Portlandite decomposition around 450°C
        portlandite_effect = np.zeros_like(temp)
        mask = (temp >= 400) & (temp <= 500)
        portlandite_effect[mask] = 2 * np.exp(-((temp[mask] - 450) / 30) ** 2)
        
        # Carbonate decomposition around 700°C (if we go that high)
        carbonate_effect = np.zeros_like(temp)
        mask = (temp >= 600) & (temp <= 800)
        carbonate_effect[mask] = 1 * np.exp(-((temp[mask] - 700) / 50) ** 2)
        
        # Combine effects
        instantaneous_cte += portlandite_effect + carbonate_effect
        
        # Add measurement noise
        noise = np.random.normal(0, 0.5, len(temp))
        instantaneous_cte += noise
        
        # Calculate cumulative thermal strain
        # Use trapezoidal integration
        thermal_strain = np.zeros_like(temp)
        for i in range(1, len(temp)):
            thermal_strain[i] = thermal_strain[i-1] + instantaneous_cte[i] * (temp[i] - temp[i-1]) / 1e6
        
        # Calculate length change percentage
        length_change_percent = thermal_strain * 100
        
        return {
            'temperature': temp,
            'instantaneous_cte': instantaneous_cte,
            'thermal_strain': thermal_strain,
            'length_change_percent': length_change_percent
        }
    
    def generate_mix_data(self, mix_type, replicates=3):
        """Generate CTE data for a mix with multiple replicates"""
        mix_data = {
            'mix_type': mix_type,
            'replicates': []
        }
        
        for replicate in range(replicates):
            cte_data = self.generate_thermal_expansion_curve(mix_type)
            mix_data['replicates'].append({
                'replicate': replicate + 1,
                'data': cte_data
            })
        
        return mix_data
    
    def generate_all_data(self):
        """Generate CTE data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'DIL 402 C (Netzsch)',
                'measurement_standard': 'ASTM E831',
                'heating_rate': '1°C/min',
                'atmosphere': 'Air',
                'replicates_per_mix': 3
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
        with open(f"{output_dir}/cte_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix and replicate
        for mix, mix_data in data['mixes'].items():
            for replicate_data in mix_data['replicates']:
                replicate_num = replicate_data['replicate']
                cte_data = replicate_data['data']
                
                df = pd.DataFrame(cte_data)
                df.to_csv(f"{output_dir}/{mix}_cte_replicate_{replicate_num}.csv", index=False)
    
    def plot_cte_curves(self, data, output_dir):
        """Generate plots of CTE curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Instantaneous CTE
        ax1 = axes[0, 0]
        for mix in self.mixes:
            if mix in data['mixes']:
                # Use first replicate for plotting
                cte_data = data['mixes'][mix]['replicates'][0]['data']
                ax1.plot(cte_data['temperature'], cte_data['instantaneous_cte'], 
                        label=mix, linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Instantaneous CTE (μm/m·°C)')
        ax1.set_title('Coefficient of Thermal Expansion vs Temperature')
        ax1.legend()
        ax1.grid(True)
        
        # Thermal strain
        ax2 = axes[0, 1]
        for mix in self.mixes:
            if mix in data['mixes']:
                cte_data = data['mixes'][mix]['replicates'][0]['data']
                ax2.plot(cte_data['temperature'], cte_data['thermal_strain'] * 1e6, 
                        label=mix, linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Thermal Strain (μm/m)')
        ax2.set_title('Thermal Strain vs Temperature')
        ax2.legend()
        ax2.grid(True)
        
        # Length change percentage
        ax3 = axes[1, 0]
        for mix in self.mixes:
            if mix in data['mixes']:
                cte_data = data['mixes'][mix]['replicates'][0]['data']
                ax3.plot(cte_data['temperature'], cte_data['length_change_percent'], 
                        label=mix, linewidth=2)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Length Change (%)')
        ax3.set_title('Length Change vs Temperature')
        ax3.legend()
        ax3.grid(True)
        
        # CTE comparison at specific temperatures
        ax4 = axes[1, 1]
        specific_temps = [100, 200, 400, 600]
        for mix in self.mixes:
            if mix in data['mixes']:
                cte_values = []
                for temp in specific_temps:
                    # Find closest temperature in data
                    cte_data = data['mixes'][mix]['replicates'][0]['data']
                    temp_idx = np.argmin(np.abs(cte_data['temperature'] - temp))
                    cte_values.append(cte_data['instantaneous_cte'][temp_idx])
                ax4.plot(specific_temps, cte_values, 'o-', label=mix, linewidth=2, markersize=6)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('CTE (μm/m·°C)')
        ax4.set_title('CTE at Specific Temperatures')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/cte_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = CTEGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/thermal_properties/cte")
    generator.plot_cte_curves(data, "/workspace/experimental_dataset/thermal_properties/cte")
    print("CTE data generation completed!")