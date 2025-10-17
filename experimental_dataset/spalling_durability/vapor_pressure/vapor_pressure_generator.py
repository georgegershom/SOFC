#!/usr/bin/env python3
"""
Vapor Pressure Measurement Data Generator
Generates realistic vapor pressure data during heating tests
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class VaporPressureGenerator:
    def __init__(self):
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        self.sensor_depths = [10, 20, 30, 40, 50]  # mm from surface
        self.temperature_range = np.arange(25, 801, 1)  # 25°C to 800°C
        self.heating_rate = 5  # °C/min
        
    def calculate_saturation_pressure(self, temperature):
        """Calculate water vapor saturation pressure at given temperature"""
        # Antoine equation for water vapor pressure (bar)
        if temperature < 100:
            # For temperatures below 100°C
            A, B, C = 8.07131, 1730.63, 233.426
        else:
            # For temperatures above 100°C
            A, B, C = 8.14019, 1810.94, 244.485
        
        log10_P = A - B / (C + temperature)
        P_sat = 10 ** log10_P  # bar
        return P_sat * 100000  # Convert to Pa
    
    def calculate_vapor_pressure(self, mix_type, temperature, depth, time):
        """Calculate vapor pressure at specific conditions"""
        # Base vapor pressure from saturation
        P_sat = self.calculate_saturation_pressure(temperature)
        
        # Depth effect - pressure increases with depth due to hydrostatic head
        depth_factor = 1 + (depth / 1000) * 0.1  # 10% increase per 10mm depth
        
        # Mix-specific effects
        mix_factors = {
            'control': 1.0,
            'rubber_10': 0.9,   # Rubber reduces vapor pressure
            'rubber_20': 0.8,
            'rubber_30': 0.7
        }
        
        # Temperature-dependent permeability effect
        if temperature < 100:
            # Below 100°C, vapor pressure is limited by permeability
            temp_factor = 0.1 + 0.9 * (temperature - 25) / 75
        else:
            # Above 100°C, vapor pressure approaches saturation
            temp_factor = 1.0
        
        # Time-dependent effect (pressure builds up over time)
        time_factor = 1 - np.exp(-time / 10)  # Builds up over ~10 minutes
        
        # Calculate actual vapor pressure
        vapor_pressure = P_sat * depth_factor * mix_factors[mix_type] * temp_factor * time_factor
        
        # Add measurement noise
        noise = np.random.normal(0, 1000)  # ±1 kPa noise
        vapor_pressure += noise
        
        # Ensure positive values
        vapor_pressure = max(vapor_pressure, 0)
        
        return vapor_pressure
    
    def generate_vapor_pressure_data(self, mix_type, replicate=1):
        """Generate complete vapor pressure dataset for a mix"""
        time = (self.temperature_range - 25) / self.heating_rate  # minutes
        
        data = {
            'mix_type': mix_type,
            'replicate': replicate,
            'temperature': self.temperature_range.tolist(),
            'time': time.tolist(),
            'sensor_data': {}
        }
        
        for depth in self.sensor_depths:
            sensor_data = {
                'depth': depth,
                'vapor_pressure': [],
                'saturation_pressure': []
            }
            
            for i, (temp, t) in enumerate(zip(self.temperature_range, time)):
                vapor_p = self.calculate_vapor_pressure(mix_type, temp, depth, t)
                sat_p = self.calculate_saturation_pressure(temp)
                
                sensor_data['vapor_pressure'].append(vapor_p)
                sensor_data['saturation_pressure'].append(sat_p)
            
            data['sensor_data'][f'depth_{depth}mm'] = sensor_data
        
        return data
    
    def generate_all_data(self):
        """Generate vapor pressure data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'Custom-built pressure transducers',
                'sensor_depths': self.sensor_depths,
                'heating_rate': f'{self.heating_rate}°C/min',
                'pressure_range': '0-10 bar',
                'sampling_rate': '1 Hz',
                'replicates_per_mix': 3
            },
            'mixes': {}
        }
        
        for mix in self.mixes:
            mix_data = {
                'mix_type': mix,
                'replicates': []
            }
            
            for replicate in range(3):
                vapor_data = self.generate_vapor_pressure_data(mix, replicate + 1)
                mix_data['replicates'].append(vapor_data)
            
            all_data['mixes'][mix] = mix_data
        
        return all_data
    
    def save_data(self, data, output_dir):
        """Save generated data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f"{output_dir}/vapor_pressure_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix and replicate
        for mix, mix_data in data['mixes'].items():
            for replicate_data in mix_data['replicates']:
                replicate_num = replicate_data['replicate']
                
                # Create combined DataFrame
                rows = []
                for i, (temp, time) in enumerate(zip(replicate_data['temperature'], replicate_data['time'])):
                    row = {
                        'temperature': temp,
                        'time': time,
                        'mix_type': mix,
                        'replicate': replicate_num
                    }
                    
                    # Add data for each sensor depth
                    for depth in self.sensor_depths:
                        sensor_key = f'depth_{depth}mm'
                        if sensor_key in replicate_data['sensor_data']:
                            sensor_data = replicate_data['sensor_data'][sensor_key]
                            row[f'vapor_pressure_{depth}mm'] = sensor_data['vapor_pressure'][i]
                            row[f'saturation_pressure_{depth}mm'] = sensor_data['saturation_pressure'][i]
                    
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(f"{output_dir}/{mix}_vapor_pressure_replicate_{replicate_num}.csv", index=False)
    
    def plot_vapor_pressure_curves(self, data, output_dir):
        """Generate plots of vapor pressure curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Vapor pressure vs temperature for different depths (control mix)
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']['replicates'][0]  # First replicate
            
            for depth in self.sensor_depths:
                sensor_key = f'depth_{depth}mm'
                if sensor_key in control_data['sensor_data']:
                    sensor_data = control_data['sensor_data'][sensor_key]
                    ax.plot(control_data['temperature'], 
                           [p/1000 for p in sensor_data['vapor_pressure']], 
                           label=f'{depth}mm depth', linewidth=2)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Vapor Pressure (kPa)')
        ax.set_title('Vapor Pressure vs Temperature - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/vapor_pressure_vs_temperature.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Vapor pressure vs time for different depths (control mix)
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']['replicates'][0]
            
            for depth in self.sensor_depths:
                sensor_key = f'depth_{depth}mm'
                if sensor_key in control_data['sensor_data']:
                    sensor_data = control_data['sensor_data'][sensor_key]
                    ax.plot(control_data['time'], 
                           [p/1000 for p in sensor_data['vapor_pressure']], 
                           label=f'{depth}mm depth', linewidth=2)
        
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Vapor Pressure (kPa)')
        ax.set_title('Vapor Pressure vs Time - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/vapor_pressure_vs_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Vapor pressure vs depth at different temperatures
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']['replicates'][0]
            
            specific_temps = [100, 200, 300, 400, 500]
            for temp in specific_temps:
                # Find closest temperature in data
                temp_idx = np.argmin(np.abs(np.array(control_data['temperature']) - temp))
                
                depths = []
                pressures = []
                for depth in self.sensor_depths:
                    sensor_key = f'depth_{depth}mm'
                    if sensor_key in control_data['sensor_data']:
                        sensor_data = control_data['sensor_data'][sensor_key]
                        depths.append(depth)
                        pressures.append(sensor_data['vapor_pressure'][temp_idx] / 1000)
                
                ax.plot(depths, pressures, 'o-', label=f'{temp}°C', linewidth=2, markersize=6)
        
        ax.set_xlabel('Depth (mm)')
        ax.set_ylabel('Vapor Pressure (kPa)')
        ax.set_title('Vapor Pressure vs Depth at Different Temperatures')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/vapor_pressure_vs_depth.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: Comparison between mixes at 30mm depth
        fig, ax = plt.subplots(figsize=(12, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                mix_data = data['mixes'][mix]['replicates'][0]
                sensor_key = 'depth_30mm'
                if sensor_key in mix_data['sensor_data']:
                    sensor_data = mix_data['sensor_data'][sensor_key]
                    ax.plot(mix_data['temperature'], 
                           [p/1000 for p in sensor_data['vapor_pressure']], 
                           label=mix, linewidth=2)
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Vapor Pressure (kPa)')
        ax.set_title('Vapor Pressure Comparison - 30mm Depth')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/vapor_pressure_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = VaporPressureGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/spalling_durability/vapor_pressure")
    generator.plot_vapor_pressure_curves(data, "/workspace/experimental_dataset/spalling_durability/vapor_pressure")
    print("Vapor pressure data generation completed!")