#!/usr/bin/env python3
"""
Spalling and Durability Dataset Generator
Pillar 2: High-Temperature Experimental Investigation - Spalling and Durability

This module generates comprehensive spalling and durability data including:
- Visual and acoustic recording analysis
- Vapor pressure measurements
- Gas permeability at elevated temperatures
- Post-exposure microstructural analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json
from datetime import datetime
import os

class SpallingDurabilityGenerator:
    def __init__(self):
        self.mix_types = {
            'Control': {'rubber_content': 0.0, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R5': {'rubber_content': 0.05, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R10': {'rubber_content': 0.10, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R15': {'rubber_content': 0.15, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R20': {'rubber_content': 0.20, 'w_c_ratio': 0.45, 'cement_type': 'OPC'}
        }
        
        # Test temperatures for spalling analysis
        self.test_temperatures = [25, 100, 200, 300, 400, 500, 600, 700, 800]
        
        # Specimen dimensions (mm)
        self.specimen_dimensions = {
            'length': 100,
            'width': 100,
            'height': 100
        }
    
    def generate_spalling_events(self, mix_type, heating_rate=1.0):
        """Generate spalling events data during heating"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Time array for heating (1°C/min heating rate)
        time = np.linspace(0, 800, 800)
        temperature = time  # 1°C/min heating rate
        
        # Spalling probability increases with temperature and decreases with rubber content
        # Rubber provides stress relief and reduces spalling tendency
        base_spalling_prob = np.zeros_like(temperature)
        
        # Critical temperature ranges for spalling
        # 200-300°C: Water vapor pressure buildup
        water_vapor_range = (temperature >= 200) & (temperature <= 300)
        water_vapor_prob = 0.3 * (1 - 0.5 * rubber_content) * water_vapor_range.astype(float)
        
        # 400-500°C: Portlandite decomposition
        portlandite_range = (temperature >= 400) & (temperature <= 500)
        portlandite_prob = 0.4 * (1 - 0.3 * rubber_content) * portlandite_range.astype(float)
        
        # 600-700°C: Carbonate decomposition
        carbonate_range = (temperature >= 600) & (temperature <= 700)
        carbonate_prob = 0.2 * (1 - 0.2 * rubber_content) * carbonate_range.astype(float)
        
        # Combine all spalling mechanisms
        spalling_prob = water_vapor_prob + portlandite_prob + carbonate_prob
        
        # Add some randomness to simulate actual spalling events
        spalling_events = []
        for i, (t, temp, prob) in enumerate(zip(time, temperature, spalling_prob)):
            if prob > 0 and np.random.random() < prob * 0.1:  # 10% of probability becomes actual event
                event_intensity = np.random.uniform(0.1, 1.0)
                spalling_events.append({
                    'time_min': t,
                    'temperature_C': temp,
                    'intensity': event_intensity,
                    'event_type': self._classify_spalling_event(temp),
                    'acoustic_amplitude': event_intensity * 100 + np.random.normal(0, 10)
                })
        
        return {
            'time': time,
            'temperature': temperature,
            'spalling_probability': spalling_prob,
            'spalling_events': spalling_events
        }
    
    def _classify_spalling_event(self, temperature):
        """Classify spalling event type based on temperature"""
        if temperature < 300:
            return 'water_vapor'
        elif temperature < 500:
            return 'portlandite_decomposition'
        else:
            return 'carbonate_decomposition'
    
    def generate_vapor_pressure_data(self, mix_type, depth_positions=[10, 25, 50, 75]):
        """Generate vapor pressure measurements at different depths"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Time and temperature arrays
        time = np.linspace(0, 800, 800)
        temperature = time  # 1°C/min heating rate
        
        vapor_pressure_data = {}
        
        for depth in depth_positions:
            # Vapor pressure increases with temperature and depth
            # Rubber content affects permeability and pressure buildup
            base_pressure = np.zeros_like(temperature)
            
            # Water vapor pressure (100-300°C)
            water_range = (temperature >= 100) & (temperature <= 300)
            water_pressure = 0.1 * (depth / 100) * (1 - 0.3 * rubber_content) * water_range.astype(float)
            
            # CO2 pressure from carbonate decomposition (600-800°C)
            co2_range = (temperature >= 600) & (temperature <= 800)
            co2_pressure = 0.5 * (depth / 100) * (1 - 0.2 * rubber_content) * co2_range.astype(float)
            
            # Combine pressures
            total_pressure = water_pressure + co2_pressure
            
            # Add noise and ensure realistic values
            noise = np.random.normal(0, 0.01, len(total_pressure))
            total_pressure += noise
            total_pressure = np.maximum(total_pressure, 0)
            
            vapor_pressure_data[f'depth_{depth}mm'] = {
                'time': time,
                'temperature': temperature,
                'vapor_pressure_MPa': total_pressure,
                'depth_mm': depth
            }
        
        return vapor_pressure_data
    
    def generate_permeability_data(self, mix_type):
        """Generate gas permeability data at elevated temperatures"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Test temperatures
        test_temps = [25, 100, 200, 300, 400, 500, 600, 700, 800]
        
        # Base permeability (m²)
        base_permeability = 1e-16 * (1 + 2 * rubber_content)  # Rubber increases permeability
        
        permeability_data = []
        
        for temp in test_temps:
            # Permeability generally increases with temperature due to microcracking
            temp_factor = 1 + 0.5 * (temp - 25) / 775
            
            # Rubber effect (rubber creates more connected porosity)
            rubber_factor = 1 + 1.5 * rubber_content * (1 + 0.2 * temp / 100)
            
            # Microcracking effect (increases significantly at high temperatures)
            cracking_factor = 1 + 2 * np.exp((temp - 400) / 200) * (temp > 400)
            
            # Calculate permeability
            permeability = base_permeability * temp_factor * rubber_factor * cracking_factor
            
            # Add experimental uncertainty
            permeability *= (1 + np.random.normal(0, 0.1))
            
            permeability_data.append({
                'temperature_C': temp,
                'permeability_m2': permeability,
                'permeability_mDarcy': permeability * 1e15  # Convert to mDarcy
            })
        
        return permeability_data
    
    def generate_microstructural_data(self, mix_type, exposure_temperature):
        """Generate post-exposure microstructural analysis data"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Damage increases with exposure temperature
        if exposure_temperature <= 200:
            damage_factor = 0.1
        elif exposure_temperature <= 400:
            damage_factor = 0.3 + 0.2 * (exposure_temperature - 200) / 200
        elif exposure_temperature <= 600:
            damage_factor = 0.5 + 0.3 * (exposure_temperature - 400) / 200
        else:
            damage_factor = 0.8 + 0.2 * (exposure_temperature - 600) / 200
        
        # Rubber content affects damage patterns
        rubber_damage_factor = 1 + 0.3 * rubber_content  # Rubber can increase damage
        
        final_damage = damage_factor * rubber_damage_factor
        final_damage = np.clip(final_damage, 0, 1)
        
        # SEM analysis data
        sem_data = {
            'microcrack_density': final_damage * 100,  # cracks/mm²
            'itz_degradation': final_damage * 80,  # percentage
            'rubber_void_morphology': {
                'void_size_increase': final_damage * 50,  # percentage increase
                'void_connectivity': final_damage * 60,  # percentage
                'rubber_decomposition': min(100, final_damage * 120)  # percentage
            },
            'cement_paste_damage': final_damage * 70,  # percentage
            'aggregate_damage': final_damage * 40  # percentage
        }
        
        # XRD analysis data
        xrd_data = {
            'portlandite_content': max(0, 20 - final_damage * 20),  # percentage
            'calcite_content': max(0, 15 - final_damage * 15),  # percentage
            'ettringite_content': max(0, 5 - final_damage * 5),  # percentage
            'new_phases': {
                'lime_content': final_damage * 10,  # percentage
                'periclase_content': final_damage * 5,  # percentage
                'amorphous_content': final_damage * 15  # percentage
            }
        }
        
        return {
            'exposure_temperature_C': exposure_temperature,
            'damage_factor': final_damage,
            'sem_analysis': sem_data,
            'xrd_analysis': xrd_data
        }
    
    def generate_acoustic_analysis(self, mix_type):
        """Generate acoustic analysis data during heating"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Time array
        time = np.linspace(0, 800, 800)
        temperature = time
        
        # Acoustic emission activity
        # Higher activity during spalling events and phase transitions
        acoustic_activity = np.zeros_like(temperature)
        
        # Water vapor release (100-300°C)
        water_range = (temperature >= 100) & (temperature <= 300)
        water_activity = 0.3 * (1 + 0.5 * rubber_content) * water_range.astype(float)
        
        # Portlandite decomposition (400-500°C)
        portlandite_range = (temperature >= 400) & (temperature <= 500)
        portlandite_activity = 0.5 * (1 + 0.3 * rubber_content) * portlandite_range.astype(float)
        
        # Carbonate decomposition (600-800°C)
        carbonate_range = (temperature >= 600) & (temperature <= 800)
        carbonate_activity = 0.4 * (1 + 0.2 * rubber_content) * carbonate_range.astype(float)
        
        # Combine activities
        acoustic_activity = water_activity + portlandite_activity + carbonate_activity
        
        # Add random events
        random_events = np.random.poisson(0.1, len(temperature))
        acoustic_activity += random_events * 0.1
        
        # Frequency analysis (Hz)
        dominant_frequencies = []
        for i, activity in enumerate(acoustic_activity):
            if activity > 0.2:
                # Higher frequencies during intense activity
                freq = 1000 + activity * 2000 + np.random.normal(0, 200)
                dominant_frequencies.append(freq)
            else:
                dominant_frequencies.append(0)
        
        return {
            'time': time,
            'temperature': temperature,
            'acoustic_activity': acoustic_activity,
            'dominant_frequency_Hz': dominant_frequencies,
            'acoustic_amplitude_dB': 20 * np.log10(acoustic_activity + 1e-6)
        }
    
    def generate_all_spalling_data(self):
        """Generate complete spalling and durability dataset"""
        all_data = {}
        
        for mix_type in self.mix_types.keys():
            print(f"Generating spalling and durability data for {mix_type}...")
            
            all_data[mix_type] = {
                'mix_properties': self.mix_types[mix_type],
                'spalling_events': self.generate_spalling_events(mix_type),
                'vapor_pressure': self.generate_vapor_pressure_data(mix_type),
                'permeability': self.generate_permeability_data(mix_type),
                'acoustic_analysis': self.generate_acoustic_analysis(mix_type),
                'microstructural_analysis': {}
            }
            
            # Microstructural analysis at different exposure temperatures
            exposure_temps = [200, 400, 600, 800]
            for exp_temp in exposure_temps:
                all_data[mix_type]['microstructural_analysis'][f'{exp_temp}C'] = self.generate_microstructural_data(mix_type, exp_temp)
        
        return all_data
    
    def save_data(self, data, output_dir='/workspace/spalling_data'):
        """Save all spalling and durability data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f'{output_dir}/spalling_durability_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files
        for mix_type, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_type}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # Spalling events
            spalling_df = pd.DataFrame(mix_data['spalling_events']['spalling_events'])
            if not spalling_df.empty:
                spalling_df.to_csv(f'{mix_dir}/spalling_events.csv', index=False)
            
            # Vapor pressure data
            vapor_dir = f'{mix_dir}/vapor_pressure'
            os.makedirs(vapor_dir, exist_ok=True)
            for depth_key, depth_data in mix_data['vapor_pressure'].items():
                df = pd.DataFrame({
                    'time_min': depth_data['time'],
                    'temperature_C': depth_data['temperature'],
                    'vapor_pressure_MPa': depth_data['vapor_pressure_MPa']
                })
                df.to_csv(f'{vapor_dir}/vapor_pressure_{depth_key}.csv', index=False)
            
            # Permeability data
            perm_df = pd.DataFrame(mix_data['permeability'])
            perm_df.to_csv(f'{mix_dir}/permeability.csv', index=False)
            
            # Acoustic analysis
            acoustic_df = pd.DataFrame(mix_data['acoustic_analysis'])
            acoustic_df.to_csv(f'{mix_dir}/acoustic_analysis.csv', index=False)
            
            # Microstructural analysis
            micro_df = pd.DataFrame([
                {
                    'exposure_temperature_C': data['exposure_temperature_C'],
                    'damage_factor': data['damage_factor'],
                    'microcrack_density': data['sem_analysis']['microcrack_density'],
                    'itz_degradation': data['sem_analysis']['itz_degradation'],
                    'portlandite_content': data['xrd_analysis']['portlandite_content'],
                    'calcite_content': data['xrd_analysis']['calcite_content']
                }
                for data in mix_data['microstructural_analysis'].values()
            ])
            micro_df.to_csv(f'{mix_dir}/microstructural_analysis.csv', index=False)
        
        print(f"Spalling and durability data saved to {output_dir}")
    
    def create_visualizations(self, data, output_dir='/workspace/spalling_data/plots'):
        """Create comprehensive visualizations of spalling data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Spalling probability vs temperature
        plt.figure(figsize=(12, 8))
        for mix_type, mix_data in data.items():
            plt.plot(mix_data['spalling_events']['temperature'], 
                    mix_data['spalling_events']['spalling_probability'], 
                    label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                    linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Spalling Probability')
        plt.title('Spalling Probability vs Temperature During Heating')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/spalling_probability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Vapor pressure at different depths
        plt.figure(figsize=(12, 8))
        mix_type = 'Control'  # Use control mix for example
        for depth_key, depth_data in data[mix_type]['vapor_pressure'].items():
            plt.plot(depth_data['temperature'], 
                    depth_data['vapor_pressure_MPa'], 
                    label=f'Depth {depth_data["depth_mm"]} mm',
                    linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Vapor Pressure (MPa)')
        plt.title('Vapor Pressure vs Temperature at Different Depths')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/vapor_pressure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Permeability vs temperature
        plt.figure(figsize=(10, 6))
        for mix_type, mix_data in data.items():
            perm_data = mix_data['permeability']
            temps = [d['temperature_C'] for d in perm_data]
            perms = [d['permeability_mDarcy'] for d in perm_data]
            plt.semilogy(temps, perms, 'o-',
                        label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                        markersize=8, linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Permeability (mDarcy)')
        plt.title('Gas Permeability vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/permeability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Microstructural damage vs exposure temperature
        plt.figure(figsize=(10, 6))
        for mix_type, mix_data in data.items():
            exp_temps = []
            damage_factors = []
            for temp_key, temp_data in mix_data['microstructural_analysis'].items():
                exp_temps.append(temp_data['exposure_temperature_C'])
                damage_factors.append(temp_data['damage_factor'])
            
            exp_temps, damage_factors = zip(*sorted(zip(exp_temps, damage_factors)))
            plt.plot(exp_temps, damage_factors, 'o-',
                    label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                    markersize=8, linewidth=2)
        
        plt.xlabel('Exposure Temperature (°C)')
        plt.ylabel('Damage Factor')
        plt.title('Microstructural Damage vs Exposure Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/microstructural_damage.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate and save spalling and durability dataset"""
    print("Generating Spalling and Durability Dataset...")
    print("=" * 60)
    
    generator = SpallingDurabilityGenerator()
    
    # Generate all spalling data
    spalling_data = generator.generate_all_spalling_data()
    
    # Save data
    generator.save_data(spalling_data)
    
    # Create visualizations
    generator.create_visualizations(spalling_data)
    
    print("\nSpalling and Durability Dataset Generation Complete!")
    print("Generated data includes:")
    print("- Spalling events and probability analysis")
    print("- Vapor pressure measurements at multiple depths")
    print("- Gas permeability at elevated temperatures")
    print("- Acoustic emission analysis")
    print("- Post-exposure microstructural analysis (SEM/XRD)")
    print(f"- Data saved to /workspace/spalling_data/")
    print(f"- Plots saved to /workspace/spalling_data/plots/")

if __name__ == "__main__":
    main()