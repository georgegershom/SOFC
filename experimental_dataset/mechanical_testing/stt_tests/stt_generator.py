#!/usr/bin/env python3
"""
Stressed-Test-Temperature (STT) Tests Data Generator
Generates realistic data for specimens heated under sustained load
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class STTGenerator:
    def __init__(self):
        self.preload_levels = [0.2, 0.4, 0.6, 0.8]  # Preload as fraction of ambient strength
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
        # Ambient strength values (MPa)
        self.ambient_strength = {
            'control': 40.0,
            'rubber_10': 35.0,
            'rubber_20': 30.0,
            'rubber_30': 25.0
        }
    
    def generate_failure_temperature(self, mix_type, preload_level, replicate=1):
        """Generate critical failure temperature for a given preload level"""
        # Base failure temperatures (°C) at different preload levels
        base_failure_temps = {
            0.2: 750,
            0.4: 650,
            0.6: 550,
            0.8: 450
        }
        
        # Mix-specific adjustments
        mix_adjustments = {
            'control': 0,
            'rubber_10': -20,
            'rubber_20': -40,
            'rubber_30': -60
        }
        
        # Calculate base failure temperature
        base_temp = base_failure_temps[preload_level]
        mix_adjustment = mix_adjustments[mix_type]
        failure_temp = base_temp + mix_adjustment
        
        # Add replicate variation
        replicate_variation = np.random.normal(0, 15)  # ±15°C variation
        failure_temp += replicate_variation
        
        # Ensure reasonable bounds
        failure_temp = max(failure_temp, 200)  # Minimum 200°C
        failure_temp = min(failure_temp, 900)  # Maximum 900°C
        
        return failure_temp
    
    def generate_temperature_time_curve(self, mix_type, preload_level, failure_temp, heating_rate=5):
        """Generate temperature vs time curve up to failure"""
        # Time to failure (minutes)
        time_to_failure = (failure_temp - 25) / heating_rate
        
        # Generate time points (every 0.5 minutes)
        time_points = np.arange(0, time_to_failure + 0.5, 0.5)
        temperature_points = 25 + heating_rate * time_points
        
        # Calculate applied stress
        ambient_strength = self.ambient_strength[mix_type]
        applied_stress = ambient_strength * preload_level
        
        # Generate stress-strain data at each time point
        stress_strain_data = []
        for i, (time, temp) in enumerate(zip(time_points, temperature_points)):
            # Calculate strength at this temperature
            temp_factor = self.calculate_temperature_factor(temp)
            current_strength = ambient_strength * temp_factor
            
            # Calculate strain (assuming linear elastic behavior)
            elastic_modulus = current_strength / 0.002  # E = f'c / 0.002
            strain = applied_stress / elastic_modulus if elastic_modulus > 0 else 0
            
            stress_strain_data.append({
                'time': time,
                'temperature': temp,
                'applied_stress': applied_stress,
                'current_strength': current_strength,
                'strain': strain,
                'stress_ratio': applied_stress / current_strength if current_strength > 0 else 0
            })
        
        return stress_strain_data
    
    def calculate_temperature_factor(self, temperature):
        """Calculate strength reduction factor at given temperature"""
        if temperature <= 25:
            return 1.0
        elif temperature <= 200:
            return 1.0 - 0.15 * (temperature - 25) / 175
        elif temperature <= 400:
            return 0.85 - 0.25 * (temperature - 200) / 200
        elif temperature <= 600:
            return 0.60 - 0.25 * (temperature - 400) / 200
        else:
            return max(0.35 - 0.20 * (temperature - 600) / 300, 0.05)
    
    def generate_mix_data(self, mix_type, replicates=5):
        """Generate STT data for a mix at all preload levels"""
        mix_data = {
            'mix_type': mix_type,
            'ambient_strength': self.ambient_strength[mix_type],
            'preload_levels': {}
        }
        
        for preload in self.preload_levels:
            preload_data = {
                'preload_level': preload,
                'applied_stress': self.ambient_strength[mix_type] * preload,
                'replicates': []
            }
            
            for replicate in range(replicates):
                # Generate failure temperature
                failure_temp = self.generate_failure_temperature(mix_type, preload, replicate + 1)
                
                # Generate temperature-time curve
                temp_time_data = self.generate_temperature_time_curve(mix_type, preload, failure_temp)
                
                preload_data['replicates'].append({
                    'replicate': replicate + 1,
                    'failure_temperature': failure_temp,
                    'time_to_failure': temp_time_data[-1]['time'],
                    'temperature_time_data': temp_time_data
                })
            
            mix_data['preload_levels'][preload] = preload_data
        
        return mix_data
    
    def generate_all_data(self):
        """Generate STT data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'Instron 5982 with high-temperature furnace',
                'heating_rate': '5°C/min',
                'specimen_size': '100mm × 100mm × 100mm cubes',
                'preload_levels': self.preload_levels,
                'replicates_per_preload': 5
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
        with open(f"{output_dir}/stt_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix and preload level
        for mix, mix_data in data['mixes'].items():
            for preload, preload_data in mix_data['preload_levels'].items():
                for replicate_data in preload_data['replicates']:
                    replicate_num = replicate_data['replicate']
                    
                    # Create DataFrame with temperature-time data
                    df = pd.DataFrame(replicate_data['temperature_time_data'])
                    
                    # Add metadata
                    df['mix_type'] = mix
                    df['preload_level'] = preload
                    df['replicate'] = replicate_num
                    df['failure_temperature'] = replicate_data['failure_temperature']
                    df['time_to_failure'] = replicate_data['time_to_failure']
                    
                    filename = f"{mix}_P{preload}_R{replicate_num}_stt.csv"
                    df.to_csv(f"{output_dir}/{filename}", index=False)
    
    def plot_stt_curves(self, data, output_dir):
        """Generate plots of STT curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Failure temperature vs preload level
        fig, ax = plt.subplots(figsize=(10, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                preloads = []
                failure_temps = []
                for preload, preload_data in data['mixes'][mix]['preload_levels'].items():
                    # Average across replicates
                    temps = [r['failure_temperature'] for r in preload_data['replicates']]
                    preloads.append(preload)
                    failure_temps.append(np.mean(temps))
                ax.plot(preloads, failure_temps, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Preload Level (fraction of ambient strength)')
        ax.set_ylabel('Failure Temperature (°C)')
        ax.set_title('Failure Temperature vs Preload Level')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/failure_temperature_vs_preload.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Temperature vs time for different preload levels (control mix)
        fig, ax = plt.subplots(figsize=(10, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            for preload in self.preload_levels:
                if preload in control_data['preload_levels']:
                    # Use first replicate for plotting
                    replicate_data = control_data['preload_levels'][preload]['replicates'][0]
                    temp_time_data = replicate_data['temperature_time_data']
                    
                    time = [d['time'] for d in temp_time_data]
                    temperature = [d['temperature'] for d in temp_time_data]
                    
                    ax.plot(time, temperature, label=f'Preload {preload}', linewidth=2)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('Temperature vs Time - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/temperature_vs_time_control.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Stress ratio vs temperature
        fig, ax = plt.subplots(figsize=(10, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            for preload in self.preload_levels:
                if preload in control_data['preload_levels']:
                    # Use first replicate for plotting
                    replicate_data = control_data['preload_levels'][preload]['replicates'][0]
                    temp_time_data = replicate_data['temperature_time_data']
                    
                    temperature = [d['temperature'] for d in temp_time_data]
                    stress_ratio = [d['stress_ratio'] for d in temp_time_data]
                    
                    ax.plot(temperature, stress_ratio, label=f'Preload {preload}', linewidth=2)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Stress Ratio (applied/current strength)')
        ax.set_title('Stress Ratio vs Temperature - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/stress_ratio_vs_temperature.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = STTGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/mechanical_testing/stt_tests")
    generator.plot_stt_curves(data, "/workspace/experimental_dataset/mechanical_testing/stt_tests")
    print("STT tests data generation completed!")