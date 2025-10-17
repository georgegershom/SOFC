#!/usr/bin/env python3
"""
In-situ Mass Loss during Heating Data Generator
Generates realistic mass loss data during heating tests
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class MassLossGenerator:
    def __init__(self):
        self.temperature_range = np.arange(25, 801, 1)  # 25°C to 800°C
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
    def generate_mass_loss_curve(self, mix_type, heating_rate=5):
        """Generate mass loss curve during heating"""
        temp = self.temperature_range
        
        # Initial mass (kg) - varies with mix type
        initial_mass = {
            'control': 2.4,
            'rubber_10': 2.3,
            'rubber_20': 2.2,
            'rubber_30': 2.1
        }
        
        # Calculate time based on heating rate
        time = (temp - 25) / heating_rate  # minutes
        
        # Base mass loss components
        mass_loss = np.zeros_like(temp, dtype=np.float64)
        
        # Water loss (25-200°C)
        water_loss = np.zeros_like(temp)
        mask = (temp >= 25) & (temp <= 200)
        water_loss[mask] = 0.05 * (1 - np.exp(-(temp[mask] - 25) / 50))  # 5% water loss
        mass_loss += water_loss
        
        # Rubber decomposition (300-500°C)
        if 'rubber' in mix_type:
            rubber_percentage = int(mix_type.split('_')[1])
            rubber_loss = np.zeros_like(temp)
            mask = (temp >= 300) & (temp <= 500)
            rubber_loss[mask] = (rubber_percentage / 100) * 0.8 * (1 - np.exp(-(temp[mask] - 300) / 50))
            mass_loss += rubber_loss
        
        # Portlandite decomposition (400-500°C)
        portlandite_loss = np.zeros_like(temp)
        mask = (temp >= 400) & (temp <= 500)
        portlandite_loss[mask] = 0.15 * (1 - np.exp(-(temp[mask] - 400) / 30))
        mass_loss += portlandite_loss
        
        # Carbonate decomposition (600-800°C)
        carbonate_loss = np.zeros_like(temp)
        mask = (temp >= 600) & (temp <= 800)
        carbonate_loss[mask] = 0.10 * (1 - np.exp(-(temp[mask] - 600) / 40))
        mass_loss += carbonate_loss
        
        # Add measurement noise
        noise = np.random.normal(0, 0.005, len(temp))  # 0.5% noise
        mass_loss += noise
        
        # Ensure mass loss doesn't exceed 100%
        mass_loss = np.clip(mass_loss, 0, 1.0)
        
        # Calculate remaining mass
        remaining_mass = initial_mass[mix_type] * (1 - mass_loss)
        
        # Calculate mass loss rate (kg/min)
        mass_loss_rate = np.gradient(remaining_mass, time)
        
        return {
            'temperature': temp,
            'time': time,
            'initial_mass': initial_mass[mix_type],
            'remaining_mass': remaining_mass,
            'mass_loss_percent': mass_loss * 100,
            'mass_loss_rate': mass_loss_rate
        }
    
    def generate_mix_data(self, mix_type, replicates=3):
        """Generate mass loss data for a mix with multiple replicates"""
        mix_data = {
            'mix_type': mix_type,
            'heating_rate': 5,  # °C/min
            'replicates': []
        }
        
        for replicate in range(replicates):
            mass_loss_data = self.generate_mass_loss_curve(mix_type)
            mix_data['replicates'].append({
                'replicate': replicate + 1,
                'data': mass_loss_data
            })
        
        return mix_data
    
    def generate_all_data(self):
        """Generate mass loss data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'High-temperature furnace with precision balance',
                'heating_rate': '5°C/min',
                'atmosphere': 'Air',
                'replicates_per_mix': 3,
                'specimen_size': '100mm × 100mm × 100mm cubes'
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
        with open(f"{output_dir}/mass_loss_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix and replicate
        for mix, mix_data in data['mixes'].items():
            for replicate_data in mix_data['replicates']:
                replicate_num = replicate_data['replicate']
                mass_data = replicate_data['data']
                
                df = pd.DataFrame(mass_data)
                df.to_csv(f"{output_dir}/{mix}_mass_loss_replicate_{replicate_num}.csv", index=False)
    
    def plot_mass_loss_curves(self, data, output_dir):
        """Generate plots of mass loss curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Mass loss percentage vs temperature
        ax1 = axes[0, 0]
        for mix in self.mixes:
            if mix in data['mixes']:
                mass_data = data['mixes'][mix]['replicates'][0]['data']
                ax1.plot(mass_data['temperature'], mass_data['mass_loss_percent'], 
                        label=mix, linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Mass Loss (%)')
        ax1.set_title('Mass Loss vs Temperature')
        ax1.legend()
        ax1.grid(True)
        
        # Remaining mass vs temperature
        ax2 = axes[0, 1]
        for mix in self.mixes:
            if mix in data['mixes']:
                mass_data = data['mixes'][mix]['replicates'][0]['data']
                ax2.plot(mass_data['temperature'], mass_data['remaining_mass'], 
                        label=mix, linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Remaining Mass (kg)')
        ax2.set_title('Remaining Mass vs Temperature')
        ax2.legend()
        ax2.grid(True)
        
        # Mass loss rate vs temperature
        ax3 = axes[1, 0]
        for mix in self.mixes:
            if mix in data['mixes']:
                mass_data = data['mixes'][mix]['replicates'][0]['data']
                ax3.plot(mass_data['temperature'], mass_data['mass_loss_rate'], 
                        label=mix, linewidth=2)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Mass Loss Rate (kg/min)')
        ax3.set_title('Mass Loss Rate vs Temperature')
        ax3.legend()
        ax3.grid(True)
        
        # Mass loss vs time
        ax4 = axes[1, 1]
        for mix in self.mixes:
            if mix in data['mixes']:
                mass_data = data['mixes'][mix]['replicates'][0]['data']
                ax4.plot(mass_data['time'], mass_data['mass_loss_percent'], 
                        label=mix, linewidth=2)
        ax4.set_xlabel('Time (min)')
        ax4.set_ylabel('Mass Loss (%)')
        ax4.set_title('Mass Loss vs Time')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/mass_loss_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = MassLossGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/thermal_properties/mass_loss")
    generator.plot_mass_loss_curves(data, "/workspace/experimental_dataset/thermal_properties/mass_loss")
    print("Mass loss data generation completed!")