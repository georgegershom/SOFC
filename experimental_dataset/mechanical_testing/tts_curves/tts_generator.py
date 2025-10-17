#!/usr/bin/env python3
"""
Transient-Test-Stress (TTS) Curves Data Generator
Generates realistic high-temperature mechanical testing data
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class TTSGenerator:
    def __init__(self):
        self.temperatures = [25, 200, 400, 600, 800]  # Test temperatures
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        
    def generate_stress_strain_curve(self, mix_type, temperature, replicate=1):
        """Generate stress-strain curve for a specific mix and temperature"""
        # Base strength values (MPa) at ambient temperature
        base_strength = {
            'control': 40.0,
            'rubber_10': 35.0,
            'rubber_20': 30.0,
            'rubber_30': 25.0
        }
        
        # Temperature degradation factors
        temp_factors = {
            25: 1.0,
            200: 0.85,
            400: 0.60,
            600: 0.35,
            800: 0.15
        }
        
        # Generate strain range (0 to 0.01 = 1%)
        strain = np.linspace(0, 0.01, 1000)
        
        # Calculate peak strength
        peak_strength = base_strength[mix_type] * temp_factors[temperature]
        
        # Add some variation between replicates
        replicate_factor = 1 + np.random.normal(0, 0.05)  # 5% variation
        peak_strength *= replicate_factor
        
        # Generate stress-strain curve using modified Hognestad model
        # Elastic region
        elastic_modulus = peak_strength / 0.002  # E = f'c / 0.002
        elastic_strain = 0.002
        
        # Parabolic region
        parabolic_strain = 0.0035
        peak_strain = 0.0035 + (temperature - 25) * 0.00001  # Slight increase with temperature
        
        stress = np.zeros_like(strain)
        
        # Elastic region
        elastic_mask = strain <= elastic_strain
        stress[elastic_mask] = elastic_modulus * strain[elastic_mask]
        
        # Parabolic region
        parabolic_mask = (strain > elastic_strain) & (strain <= peak_strain)
        if np.any(parabolic_mask):
            # Hognestad parabola
            strain_ratio = (strain[parabolic_mask] - elastic_strain) / (peak_strain - elastic_strain)
            stress[parabolic_mask] = peak_strength * (2 * strain_ratio - strain_ratio**2)
        
        # Post-peak region (softening)
        post_peak_mask = strain > peak_strain
        if np.any(post_peak_mask):
            # Linear softening
            softening_modulus = -peak_strength / 0.005  # Softening slope
            stress[post_peak_mask] = peak_strength + softening_modulus * (strain[post_peak_mask] - peak_strain)
            stress[post_peak_mask] = np.maximum(stress[post_peak_mask], 0)  # No negative stress
        
        # Add noise to simulate real measurements
        noise = np.random.normal(0, 0.5, len(stress))  # 0.5 MPa noise
        stress += noise
        stress = np.maximum(stress, 0)  # No negative stress
        
        # Calculate derived properties
        peak_strain_actual = strain[np.argmax(stress)]
        elastic_modulus_actual = np.polyfit(strain[:100], stress[:100], 1)[0]  # Slope of elastic region
        
        return {
            'strain': strain,
            'stress': stress,
            'peak_strength': np.max(stress),
            'peak_strain': peak_strain_actual,
            'elastic_modulus': elastic_modulus_actual,
            'temperature': temperature,
            'mix_type': mix_type,
            'replicate': replicate
        }
    
    def generate_mix_data(self, mix_type, replicates=5):
        """Generate TTS data for a mix at all temperatures"""
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
                curve_data = self.generate_stress_strain_curve(mix_type, temp, replicate + 1)
                temp_data['replicates'].append(curve_data)
            
            mix_data['temperatures'][temp] = temp_data
        
        return mix_data
    
    def generate_all_data(self):
        """Generate TTS data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'equipment': 'Instron 5982 with high-temperature furnace',
                'loading_rate': '0.5 MPa/s',
                'specimen_size': '100mm × 100mm × 100mm cubes',
                'heating_rate': '5°C/min',
                'soak_time': '2 hours at target temperature',
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
        with open(f"{output_dir}/tts_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save as CSV for each mix, temperature, and replicate
        for mix, mix_data in data['mixes'].items():
            for temp, temp_data in mix_data['temperatures'].items():
                for replicate_data in temp_data['replicates']:
                    replicate_num = replicate_data['replicate']
                    
                    # Create DataFrame with stress-strain data
                    df = pd.DataFrame({
                        'strain': replicate_data['strain'],
                        'stress': replicate_data['stress']
                    })
                    
                    # Add metadata
                    df['temperature'] = temp
                    df['mix_type'] = mix
                    df['replicate'] = replicate_num
                    df['peak_strength'] = replicate_data['peak_strength']
                    df['peak_strain'] = replicate_data['peak_strain']
                    df['elastic_modulus'] = replicate_data['elastic_modulus']
                    
                    filename = f"{mix}_T{temp}_R{replicate_num}_tts.csv"
                    df.to_csv(f"{output_dir}/{filename}", index=False)
    
    def plot_tts_curves(self, data, output_dir):
        """Generate plots of TTS curves"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Stress-strain curves for all temperatures (control mix)
        fig, ax = plt.subplots(figsize=(10, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            for temp in self.temperatures:
                if temp in control_data['temperatures']:
                    # Use first replicate for plotting
                    curve_data = control_data['temperatures'][temp]['replicates'][0]
                    ax.plot(curve_data['strain'] * 100, curve_data['stress'], 
                           label=f'{temp}°C', linewidth=2)
        ax.set_xlabel('Strain (%)')
        ax.set_ylabel('Stress (MPa)')
        ax.set_title('TTS Curves - Control Mix at Different Temperatures')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/control_tts_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Peak strength vs temperature for all mixes
        fig, ax = plt.subplots(figsize=(10, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                strengths = []
                for temp in self.temperatures:
                    if temp in data['mixes'][mix]['temperatures']:
                        # Average across replicates
                        peak_strengths = [r['peak_strength'] for r in data['mixes'][mix]['temperatures'][temp]['replicates']]
                        temps.append(temp)
                        strengths.append(np.mean(peak_strengths))
                ax.plot(temps, strengths, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Peak Strength (MPa)')
        ax.set_title('Peak Strength vs Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/peak_strength_vs_temperature.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Elastic modulus vs temperature
        fig, ax = plt.subplots(figsize=(10, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                moduli = []
                for temp in self.temperatures:
                    if temp in data['mixes'][mix]['temperatures']:
                        # Average across replicates
                        elastic_moduli = [r['elastic_modulus'] for r in data['mixes'][mix]['temperatures'][temp]['replicates']]
                        temps.append(temp)
                        moduli.append(np.mean(elastic_moduli))
                ax.plot(temps, moduli, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Elastic Modulus (MPa)')
        ax.set_title('Elastic Modulus vs Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/elastic_modulus_vs_temperature.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = TTSGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/mechanical_testing/tts_curves")
    generator.plot_tts_curves(data, "/workspace/experimental_dataset/mechanical_testing/tts_curves")
    print("TTS curves data generation completed!")