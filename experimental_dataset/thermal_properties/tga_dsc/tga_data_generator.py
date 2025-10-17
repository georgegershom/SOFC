#!/usr/bin/env python3
"""
TGA/DSC Data Generator for High-Temperature Experimental Dataset
Generates realistic TGA and DSC data for rubberized concrete samples
"""

import numpy as np
import pandas as pd
import json
from scipy import interpolate
import matplotlib.pyplot as plt
from datetime import datetime
import os

class TGADataGenerator:
    def __init__(self):
        self.temperature_range = np.arange(25, 801, 1)  # 25°C to 800°C
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30', 'raw_rubber']
        
    def generate_rubber_decomposition(self, temp):
        """Generate rubber decomposition curve (300-500°C)"""
        # Rubber decomposition typically occurs between 300-500°C
        rubber_loss = np.zeros_like(temp)
        mask = (temp >= 300) & (temp <= 500)
        rubber_loss[mask] = 80 * (1 - np.exp(-(temp[mask] - 300) / 50))
        return rubber_loss
    
    def generate_portlandite_decomposition(self, temp):
        """Generate Portlandite (Ca(OH)2) decomposition curve (~450°C)"""
        portlandite_loss = np.zeros_like(temp)
        mask = (temp >= 400) & (temp <= 500)
        portlandite_loss[mask] = 15 * (1 - np.exp(-(temp[mask] - 400) / 30))
        return portlandite_loss
    
    def generate_carbonate_decomposition(self, temp):
        """Generate carbonate decomposition curve (600-800°C)"""
        carbonate_loss = np.zeros_like(temp)
        mask = (temp >= 600) & (temp <= 800)
        carbonate_loss[mask] = 10 * (1 - np.exp(-(temp[mask] - 600) / 40))
        return carbonate_loss
    
    def generate_water_loss(self, temp):
        """Generate water loss curve (25-200°C)"""
        water_loss = np.zeros_like(temp)
        mask = (temp >= 25) & (temp <= 200)
        water_loss[mask] = 5 * (1 - np.exp(-(temp[mask] - 25) / 50))
        return water_loss
    
    def generate_tga_curve(self, mix_type, replicate=1):
        """Generate complete TGA curve for a specific mix"""
        temp = self.temperature_range
        total_loss = np.zeros_like(temp, dtype=np.float64)
        
        # Base water loss (all mixes)
        water_loss = self.generate_water_loss(temp)
        total_loss += water_loss
        
        if mix_type == 'raw_rubber':
            # Pure rubber - only rubber decomposition
            rubber_loss = self.generate_rubber_decomposition(temp)
            total_loss += rubber_loss
            
        else:
            # Concrete mixes
            rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
            
            # Portlandite decomposition (reduced with rubber content)
            portlandite_loss = self.generate_portlandite_decomposition(temp)
            portlandite_factor = 1 - (rubber_percentage * 0.1)  # Slight reduction with rubber
            total_loss += portlandite_loss * portlandite_factor
            
            # Carbonate decomposition
            carbonate_loss = self.generate_carbonate_decomposition(temp)
            total_loss += carbonate_loss
            
            # Rubber decomposition (proportional to rubber content)
            if rubber_percentage > 0:
                rubber_loss = self.generate_rubber_decomposition(temp)
                rubber_factor = rubber_percentage / 100
                total_loss += rubber_loss * rubber_factor
        
        # Add noise to simulate real measurements
        noise = np.random.normal(0, 0.1, len(temp))
        total_loss += noise
        
        # Ensure mass loss doesn't exceed 100%
        total_loss = np.clip(total_loss, 0, 100)
        
        # Calculate remaining mass percentage
        remaining_mass = 100 - total_loss
        
        return {
            'temperature': temp,
            'mass_loss_percent': total_loss,
            'remaining_mass_percent': remaining_mass,
            'mass_loss_rate': np.gradient(total_loss, temp)
        }
    
    def generate_dsc_curve(self, mix_type, replicate=1):
        """Generate DSC heat flow curve"""
        temp = self.temperature_range
        heat_flow = np.zeros_like(temp, dtype=np.float64)
        
        if mix_type == 'raw_rubber':
            # Rubber decomposition endotherm
            mask = (temp >= 300) & (temp <= 500)
            heat_flow[mask] = -50 * np.exp(-((temp[mask] - 400) / 50) ** 2)
        else:
            # Concrete mixes
            rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
            
            # Water evaporation endotherm (25-200°C)
            mask1 = (temp >= 25) & (temp <= 200)
            heat_flow[mask1] = -20 * np.exp(-((temp[mask1] - 100) / 30) ** 2)
            
            # Portlandite decomposition endotherm (~450°C)
            mask2 = (temp >= 400) & (temp <= 500)
            portlandite_factor = 1 - (rubber_percentage * 0.1)
            heat_flow[mask2] = -30 * portlandite_factor * np.exp(-((temp[mask2] - 450) / 25) ** 2)
            
            # Carbonate decomposition endotherm (600-800°C)
            mask3 = (temp >= 600) & (temp <= 800)
            heat_flow[mask3] = -25 * np.exp(-((temp[mask3] - 700) / 40) ** 2)
            
            # Rubber decomposition endotherm (if present)
            if rubber_percentage > 0:
                mask4 = (temp >= 300) & (temp <= 500)
                rubber_factor = rubber_percentage / 100
                heat_flow[mask4] += -40 * rubber_factor * np.exp(-((temp[mask4] - 400) / 50) ** 2)
        
        # Add noise
        noise = np.random.normal(0, 1, len(temp))
        heat_flow += noise
        
        return {
            'temperature': temp,
            'heat_flow_mW_mg': heat_flow,
            'cumulative_heat': np.cumsum(heat_flow) * (temp[1] - temp[0])
        }
    
    def generate_all_data(self):
        """Generate TGA and DSC data for all mixes and replicates"""
        all_data = {}
        
        for mix in self.mixes:
            all_data[mix] = {
                'tga': [],
                'dsc': [],
                'metadata': {
                    'mix_type': mix,
                    'generation_date': datetime.now().isoformat(),
                    'replicates': 3
                }
            }
            
            for replicate in range(1, 4):
                tga_data = self.generate_tga_curve(mix, replicate)
                dsc_data = self.generate_dsc_curve(mix, replicate)
                
                all_data[mix]['tga'].append({
                    'replicate': replicate,
                    'data': tga_data
                })
                all_data[mix]['dsc'].append({
                    'replicate': replicate,
                    'data': dsc_data
                })
        
        return all_data
    
    def save_data(self, data, output_dir):
        """Save generated data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f"{output_dir}/tga_dsc_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix and replicate
        for mix, mix_data in data.items():
            for i, (tga, dsc) in enumerate(zip(mix_data['tga'], mix_data['dsc'])):
                # TGA data
                tga_df = pd.DataFrame(tga['data'])
                tga_df.to_csv(f"{output_dir}/{mix}_tga_replicate_{i+1}.csv", index=False)
                
                # DSC data
                dsc_df = pd.DataFrame(dsc['data'])
                dsc_df.to_csv(f"{output_dir}/{mix}_dsc_replicate_{i+1}.csv", index=False)
    
    def plot_sample_curves(self, data, output_dir):
        """Generate sample plots"""
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # TGA curves
        ax1 = axes[0, 0]
        for mix in ['control', 'rubber_10', 'rubber_30', 'raw_rubber']:
            if mix in data:
                tga_data = data[mix]['tga'][0]['data']  # First replicate
                ax1.plot(tga_data['temperature'], tga_data['remaining_mass_percent'], 
                        label=mix, linewidth=2)
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Remaining Mass (%)')
        ax1.set_title('TGA - Mass Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        # DSC curves
        ax2 = axes[0, 1]
        for mix in ['control', 'rubber_10', 'rubber_30', 'raw_rubber']:
            if mix in data:
                dsc_data = data[mix]['dsc'][0]['data']  # First replicate
                ax2.plot(dsc_data['temperature'], dsc_data['heat_flow_mW_mg'], 
                        label=mix, linewidth=2)
        ax2.set_xlabel('Temperature (°C)')
        ax2.set_ylabel('Heat Flow (mW/mg)')
        ax2.set_title('DSC - Heat Flow Curves')
        ax2.legend()
        ax2.grid(True)
        
        # Mass loss rate
        ax3 = axes[1, 0]
        for mix in ['control', 'rubber_10', 'rubber_30', 'raw_rubber']:
            if mix in data:
                tga_data = data[mix]['tga'][0]['data']
                ax3.plot(tga_data['temperature'], tga_data['mass_loss_rate'], 
                        label=mix, linewidth=2)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Mass Loss Rate (%/°C)')
        ax3.set_title('TGA - Mass Loss Rate')
        ax3.legend()
        ax3.grid(True)
        
        # Cumulative heat
        ax4 = axes[1, 1]
        for mix in ['control', 'rubber_10', 'rubber_30', 'raw_rubber']:
            if mix in data:
                dsc_data = data[mix]['dsc'][0]['data']
                ax4.plot(dsc_data['temperature'], dsc_data['cumulative_heat'], 
                        label=mix, linewidth=2)
        ax4.set_xlabel('Temperature (°C)')
        ax4.set_ylabel('Cumulative Heat (mJ/mg)')
        ax4.set_title('DSC - Cumulative Heat')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/tga_dsc_sample_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = TGADataGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/thermal_properties/tga_dsc")
    generator.plot_sample_curves(data, "/workspace/experimental_dataset/thermal_properties/tga_dsc")
    print("TGA/DSC data generation completed!")