#!/usr/bin/env python3
"""
Gas Permeability Data Generator
Generates realistic gas permeability data at elevated temperatures
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class PermeabilityGenerator:
    def __init__(self):
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        self.temperatures = [25, 100, 200, 400, 600]  # Test temperatures
        self.pressure_differences = [0.1, 0.2, 0.5, 1.0]  # bar
        
    def calculate_permeability(self, mix_type, temperature, pressure_diff):
        """Calculate gas permeability at specific conditions"""
        # Base permeability values (m²) at ambient temperature
        base_permeability = {
            'control': 1e-16,
            'rubber_10': 2e-16,
            'rubber_20': 4e-16,
            'rubber_30': 8e-16
        }
        
        # Temperature effect - permeability increases with temperature
        temp_factor = 1 + 0.001 * (temperature - 25)  # 0.1% increase per °C
        
        # Pressure effect - higher pressure differences can cause microcracking
        pressure_factor = 1 + 0.1 * pressure_diff  # 10% increase per bar
        
        # Calculate permeability
        permeability = base_permeability[mix_type] * temp_factor * pressure_factor
        
        # Add measurement noise
        noise_factor = 1 + np.random.normal(0, 0.1)  # 10% noise
        permeability *= noise_factor
        
        # Ensure positive values
        permeability = max(permeability, 1e-20)
        
        return permeability
    
    def calculate_flow_rate(self, permeability, pressure_diff, specimen_area, specimen_length, temperature):
        """Calculate gas flow rate through specimen"""
        # Darcy's law: Q = (k * A * ΔP) / (μ * L)
        # Where: k = permeability, A = area, ΔP = pressure difference, μ = viscosity, L = length
        
        # Gas viscosity (Pa·s) - varies with temperature
        viscosity = 1.8e-5 * (1 + 0.0005 * (temperature - 25))  # Air viscosity
        
        # Convert pressure difference to Pa
        pressure_diff_pa = pressure_diff * 100000
        
        # Calculate flow rate (m³/s)
        flow_rate = (permeability * specimen_area * pressure_diff_pa) / (viscosity * specimen_length)
        
        return flow_rate
    
    def generate_permeability_data(self, mix_type, temperature, replicate=1):
        """Generate permeability data for a specific mix and temperature"""
        specimen_area = 0.01  # m² (100mm × 100mm)
        specimen_length = 0.05  # m (50mm)
        
        data = {
            'mix_type': mix_type,
            'temperature': temperature,
            'replicate': replicate,
            'specimen_area': specimen_area,
            'specimen_length': specimen_length,
            'measurements': []
        }
        
        for pressure_diff in self.pressure_differences:
            permeability = self.calculate_permeability(mix_type, temperature, pressure_diff)
            flow_rate = self.calculate_flow_rate(permeability, pressure_diff, specimen_area, specimen_length, temperature)
            
            measurement = {
                'pressure_difference': pressure_diff,
                'permeability': permeability,
                'flow_rate': flow_rate,
                'viscosity': 1.8e-5 * (1 + 0.0005 * (temperature - 25))
            }
            
            data['measurements'].append(measurement)
        
        return data
    
    def generate_mix_data(self, mix_type, replicates=5):
        """Generate permeability data for a mix at all temperatures"""
        mix_data = {
            'mix_type': mix_type,
            'temperatures': {}
        }
        
        for temp in self.temperatures:
            temp_data = {
                'temperature': temp,
                'replicates': []
            }
            
            for replicate in range(replicates):
                permeability_data = self.generate_permeability_data(mix_type, temp, replicate + 1)
                temp_data['replicates'].append(permeability_data)
            
            mix_data['temperatures'][temp] = temp_data
        
        return mix_data
    
    def generate_all_data(self):
        """Generate permeability data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'Custom permeability apparatus',
                'gas': 'Nitrogen',
                'pressure_differences': self.pressure_differences,
                'specimen_size': '100mm × 100mm × 50mm discs',
                'measurement_duration': '30 minutes per temperature',
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
        with open(f"{output_dir}/permeability_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix
        for mix, mix_data in data['mixes'].items():
            rows = []
            for temp, temp_data in mix_data['temperatures'].items():
                for replicate_data in temp_data['replicates']:
                    for measurement in replicate_data['measurements']:
                        row = {
                            'mix_type': mix,
                            'temperature': temp,
                            'replicate': replicate_data['replicate'],
                            'pressure_difference': measurement['pressure_difference'],
                            'permeability': measurement['permeability'],
                            'flow_rate': measurement['flow_rate'],
                            'viscosity': measurement['viscosity']
                        }
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(f"{output_dir}/{mix}_permeability.csv", index=False)
    
    def plot_permeability_curves(self, data, output_dir):
        """Generate plots of permeability curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Permeability vs temperature for all mixes
        fig, ax = plt.subplots(figsize=(12, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                permeabilities = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    # Average across replicates and pressure differences
                    perm_list = []
                    for replicate_data in temp_data['replicates']:
                        for measurement in replicate_data['measurements']:
                            perm_list.append(measurement['permeability'])
                    temps.append(temp)
                    permeabilities.append(np.mean(perm_list))
                
                ax.plot(temps, [p*1e15 for p in permeabilities], 'o-', label=mix, linewidth=2, markersize=6)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Permeability (×10⁻¹⁵ m²)')
        ax.set_title('Gas Permeability vs Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/permeability_vs_temperature.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Permeability vs pressure difference (control mix at different temperatures)
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            for temp in self.temperatures:
                if temp in control_data['temperatures']:
                    pressure_diffs = []
                    permeabilities = []
                    for replicate_data in control_data['temperatures'][temp]['replicates']:
                        for measurement in replicate_data['measurements']:
                            pressure_diffs.append(measurement['pressure_difference'])
                            permeabilities.append(measurement['permeability'])
                    
                    # Average across replicates
                    unique_pressures = sorted(set(pressure_diffs))
                    avg_permeabilities = []
                    for p in unique_pressures:
                        perm_values = [perm for pd, perm in zip(pressure_diffs, permeabilities) if pd == p]
                        avg_permeabilities.append(np.mean(perm_values))
                    
                    ax.plot(unique_pressures, [p*1e15 for p in avg_permeabilities], 
                           'o-', label=f'{temp}°C', linewidth=2, markersize=6)
        
        ax.set_xlabel('Pressure Difference (bar)')
        ax.set_ylabel('Permeability (×10⁻¹⁵ m²)')
        ax.set_title('Permeability vs Pressure Difference - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/permeability_vs_pressure.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Flow rate vs pressure difference
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            temp = 400  # Focus on 400°C
            if temp in control_data['temperatures']:
                pressure_diffs = []
                flow_rates = []
                for replicate_data in control_data['temperatures'][temp]['replicates']:
                    for measurement in replicate_data['measurements']:
                        pressure_diffs.append(measurement['pressure_difference'])
                        flow_rates.append(measurement['flow_rate'])
                
                # Average across replicates
                unique_pressures = sorted(set(pressure_diffs))
                avg_flow_rates = []
                for p in unique_pressures:
                    flow_values = [fr for pd, fr in zip(pressure_diffs, flow_rates) if pd == p]
                    avg_flow_rates.append(np.mean(flow_values))
                
                ax.plot(unique_pressures, [fr*1e6 for fr in avg_flow_rates], 
                       'o-', linewidth=2, markersize=6)
        
        ax.set_xlabel('Pressure Difference (bar)')
        ax.set_ylabel('Flow Rate (×10⁻⁶ m³/s)')
        ax.set_title('Flow Rate vs Pressure Difference - Control Mix at 400°C')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/flow_rate_vs_pressure.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Permeability comparison at 0.5 bar pressure difference
        fig, ax = plt.subplots(figsize=(12, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                permeabilities = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    # Get permeability at 0.5 bar pressure difference
                    perm_values = []
                    for replicate_data in temp_data['replicates']:
                        for measurement in replicate_data['measurements']:
                            if abs(measurement['pressure_difference'] - 0.5) < 0.01:
                                perm_values.append(measurement['permeability'])
                    if perm_values:
                        temps.append(temp)
                        permeabilities.append(np.mean(perm_values))
                
                ax.plot(temps, [p*1e15 for p in permeabilities], 'o-', label=mix, linewidth=2, markersize=6)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Permeability (×10⁻¹⁵ m²)')
        ax.set_title('Permeability Comparison at 0.5 bar Pressure Difference')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/permeability_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = PermeabilityGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/spalling_durability/gas_permeability")
    generator.plot_permeability_curves(data, "/workspace/experimental_dataset/spalling_durability/gas_permeability")
    print("Gas permeability data generation completed!")