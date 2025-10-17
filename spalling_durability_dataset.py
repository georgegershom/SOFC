#!/usr/bin/env python3
"""
Spalling and Durability Dataset Generator for Fire-Resistant Rubberized Concrete
Generates comprehensive spalling and durability data including visual/acoustic recording,
vapor pressure measurement, gas permeability, and microstructural analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import json
from datetime import datetime
import os

class SpallingDurabilityGenerator:
    def __init__(self):
        self.mix_designs = {
            'Control': {'cement': 100, 'water': 40, 'aggregate': 180, 'rubber': 0},
            'Low_Rubber': {'cement': 100, 'water': 40, 'aggregate': 160, 'rubber': 20},
            'Medium_Rubber': {'cement': 100, 'water': 40, 'aggregate': 140, 'rubber': 40},
            'High_Rubber': {'cement': 100, 'water': 40, 'aggregate': 120, 'rubber': 60}
        }
        
        # Spalling risk factors
        self.spalling_risk_factors = {
            'Control': {'moisture_content': 0.15, 'permeability': 1e-16, 'thermal_stress': 1.0},
            'Low_Rubber': {'moisture_content': 0.12, 'permeability': 2e-16, 'thermal_stress': 0.8},
            'Medium_Rubber': {'moisture_content': 0.10, 'permeability': 5e-16, 'thermal_stress': 0.6},
            'High_Rubber': {'moisture_content': 0.08, 'permeability': 1e-15, 'thermal_stress': 0.4}
        }
    
    def generate_visual_acoustic_data(self, mix_name, rubber_content):
        """Generate visual and acoustic recording data for spalling events"""
        # Time series data (2 hours at 5°C/min heating rate)
        time_points = np.linspace(0, 120, 1200)  # minutes
        temperatures = time_points * 5  # °C
        
        # Spalling probability based on mix design
        spalling_prob = self._calculate_spalling_probability(mix_name, rubber_content)
        
        # Generate spalling events
        spalling_events = self._generate_spalling_events(temperatures, spalling_prob, rubber_content)
        
        # Generate acoustic data (sound intensity during spalling)
        acoustic_intensity = self._generate_acoustic_data(temperatures, spalling_events, rubber_content)
        
        # Generate visual data (spalling intensity)
        visual_intensity = self._generate_visual_data(temperatures, spalling_events, rubber_content)
        
        return {
            'time': time_points,
            'temperature': temperatures,
            'spalling_events': spalling_events,
            'acoustic_intensity': acoustic_intensity,
            'visual_intensity': visual_intensity,
            'total_spalling_events': np.sum(spalling_events),
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def _calculate_spalling_probability(self, mix_name, rubber_content):
        """Calculate spalling probability based on mix characteristics"""
        risk_factors = self.spalling_risk_factors[mix_name]
        
        # Base spalling probability
        base_prob = 0.3
        
        # Rubber reduces spalling risk
        rubber_reduction = 1 - (rubber_content / 100) * 0.6
        
        # Moisture content effect
        moisture_factor = 1 + (risk_factors['moisture_content'] - 0.1) * 2
        
        # Permeability effect (higher permeability = lower spalling risk)
        permeability_factor = 1 - (risk_factors['permeability'] / 1e-15) * 0.5
        
        # Thermal stress effect
        thermal_stress_factor = risk_factors['thermal_stress']
        
        spalling_prob = base_prob * rubber_reduction * moisture_factor * permeability_factor * thermal_stress_factor
        
        return max(0, min(1, spalling_prob))
    
    def _generate_spalling_events(self, temperatures, spalling_prob, rubber_content):
        """Generate spalling events as binary array"""
        spalling_events = np.zeros(len(temperatures))
        
        # Critical temperature ranges for spalling
        critical_ranges = [
            (100, 150),   # Free water evaporation
            (200, 300),   # Bound water loss
            (400, 500),   # Chemical decomposition
            (600, 700)    # Severe thermal stress
        ]
        
        for temp_range in critical_ranges:
            mask = (temperatures >= temp_range[0]) & (temperatures <= temp_range[1])
            if np.any(mask):
                # Generate spalling events in this temperature range
                n_events = int(np.sum(mask) * spalling_prob * 0.1)
                if n_events > 0:
                    event_indices = np.random.choice(
                        np.where(mask)[0], 
                        size=min(n_events, np.sum(mask)), 
                        replace=False
                    )
                    spalling_events[event_indices] = 1
        
        return spalling_events
    
    def _generate_acoustic_data(self, temperatures, spalling_events, rubber_content):
        """Generate acoustic intensity data"""
        acoustic_intensity = np.zeros(len(temperatures))
        
        # Base noise level
        base_noise = 30 + np.random.normal(0, 5, len(temperatures))
        
        # Temperature-dependent noise increase
        temp_noise = (temperatures - 25) * 0.1
        
        # Spalling events create acoustic peaks
        for i, event in enumerate(spalling_events):
            if event:
                # Spalling creates high-intensity sound
                peak_intensity = 80 + np.random.normal(0, 10)
                # Sound duration (affects nearby time points)
                duration = 5  # seconds
                start_idx = max(0, i - duration)
                end_idx = min(len(temperatures), i + duration)
                acoustic_intensity[start_idx:end_idx] += peak_intensity * np.exp(-np.abs(np.arange(start_idx, end_idx) - i) / 2)
        
        # Rubber dampens sound
        rubber_damping = 1 - (rubber_content / 100) * 0.3
        
        acoustic_intensity = base_noise + temp_noise + acoustic_intensity * rubber_damping
        
        return acoustic_intensity
    
    def _generate_visual_data(self, temperatures, spalling_events, rubber_content):
        """Generate visual spalling intensity data"""
        visual_intensity = np.zeros(len(temperatures))
        
        # Base visual intensity (cracking, discoloration)
        base_intensity = (temperatures - 25) * 0.05
        
        # Spalling events create visual peaks
        for i, event in enumerate(spalling_events):
            if event:
                # Spalling creates visible damage
                peak_intensity = 50 + np.random.normal(0, 10)
                visual_intensity[i] = peak_intensity
        
        # Rubber reduces visual damage
        rubber_protection = 1 - (rubber_content / 100) * 0.2
        
        visual_intensity = (base_intensity + visual_intensity) * rubber_protection
        
        return visual_intensity
    
    def generate_vapor_pressure_data(self, mix_name, rubber_content):
        """Generate vapor pressure measurement data at different depths"""
        # Time series data
        time_points = np.linspace(0, 120, 1200)  # minutes
        temperatures = time_points * 5  # °C
        
        # Depth positions (mm from surface)
        depths = [5, 15, 25, 35, 45]  # 5 depths
        
        vapor_pressure_data = {}
        
        for depth in depths:
            # Vapor pressure increases with temperature and depth
            # Rubber affects moisture transport
            rubber_effect = 1 + (rubber_content / 100) * 0.3
            
            # Base vapor pressure (Pa)
            base_pressure = 101325  # Atmospheric pressure
            
            # Temperature-dependent vapor pressure
            temp_pressure = base_pressure * np.exp((temperatures - 25) * 0.01)
            
            # Depth effect (pressure increases with depth due to moisture accumulation)
            depth_factor = 1 + (depth / 50) * 0.5
            
            # Moisture content effect
            moisture_content = 0.15 - (rubber_content / 100) * 0.05
            moisture_factor = 1 + moisture_content * 2
            
            # Generate vapor pressure
            vapor_pressure = temp_pressure * depth_factor * moisture_factor * rubber_effect
            
            # Add realistic noise
            noise = np.random.normal(0, 1000, len(vapor_pressure))
            vapor_pressure += noise
            vapor_pressure = np.maximum(vapor_pressure, base_pressure)
            
            vapor_pressure_data[depth] = {
                'time': time_points,
                'temperature': temperatures,
                'vapor_pressure': vapor_pressure,
                'depth': depth
            }
        
        return {
            'depths': depths,
            'vapor_pressure_data': vapor_pressure_data,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def generate_gas_permeability_data(self, mix_name, rubber_content):
        """Generate gas permeability data at elevated temperatures"""
        temperatures = [25, 100, 200, 400, 600, 800]
        
        # Base permeability (m²)
        base_permeability = 1e-16
        
        # Rubber increases permeability
        rubber_factor = 1 + (rubber_content / 100) * 5
        
        permeability_data = []
        
        for temp in temperatures:
            # Permeability increases with temperature due to microcracking
            temp_factor = 1 + (temp - 25) * 0.001
            
            # Thermal damage factor
            if temp <= 200:
                damage_factor = 1.0
            elif temp <= 400:
                damage_factor = 1 + (temp - 200) * 0.002
            elif temp <= 600:
                damage_factor = 1.4 + (temp - 400) * 0.003
            else:
                damage_factor = 2.0 + (temp - 600) * 0.005
            
            # Calculate permeability
            permeability = base_permeability * rubber_factor * temp_factor * damage_factor
            
            # Add measurement uncertainty
            permeability += np.random.normal(0, permeability * 0.1)
            permeability = max(1e-18, permeability)  # Minimum permeability
            
            permeability_data.append({
                'temperature': temp,
                'permeability': permeability,
                'permeability_log': np.log10(permeability)
            })
        
        return {
            'temperatures': temperatures,
            'permeability_data': permeability_data,
            'mix_name': mix_name,
            'rubber_content': rubber_content
        }
    
    def generate_microstructural_analysis_data(self, mix_name, rubber_content):
        """Generate post-exposure microstructural analysis data"""
        exposure_temperatures = [25, 200, 400, 600, 800]
        
        microstructural_data = {}
        
        for temp in exposure_temperatures:
            # SEM analysis data
            sem_data = self._generate_sem_data(temp, rubber_content)
            
            # XRD analysis data
            xrd_data = self._generate_xrd_data(temp, rubber_content)
            
            microstructural_data[temp] = {
                'exposure_temperature': temp,
                'sem_analysis': sem_data,
                'xrd_analysis': xrd_data,
                'mix_name': mix_name,
                'rubber_content': rubber_content
            }
        
        return microstructural_data
    
    def _generate_sem_data(self, exposure_temp, rubber_content):
        """Generate SEM analysis data"""
        # Microcrack density (cracks/mm²)
        if exposure_temp <= 100:
            crack_density = 0.1
        elif exposure_temp <= 200:
            crack_density = 0.5
        elif exposure_temp <= 400:
            crack_density = 2.0
        elif exposure_temp <= 600:
            crack_density = 5.0
        else:
            crack_density = 10.0
        
        # Rubber reduces crack density
        crack_density *= (1 - (rubber_content / 100) * 0.3)
        
        # ITZ (Interfacial Transition Zone) degradation
        itz_degradation = min(1.0, (exposure_temp - 200) / 400)
        itz_degradation *= (1 - (rubber_content / 100) * 0.2)
        
        # Rubber void morphology changes
        rubber_void_size = 0.1 + (exposure_temp - 25) * 0.0001
        rubber_void_density = (rubber_content / 100) * 10 * (1 - itz_degradation)
        
        return {
            'crack_density': crack_density,
            'itz_degradation': itz_degradation,
            'rubber_void_size': rubber_void_size,
            'rubber_void_density': rubber_void_density,
            'microstructural_damage_index': itz_degradation + crack_density * 0.1
        }
    
    def _generate_xrd_data(self, exposure_temp, rubber_content):
        """Generate XRD analysis data"""
        # Phase identification and quantification
        phases = {
            'portlandite': {'2theta': 18.0, 'intensity': 100, 'decomposition_temp': 450},
            'calcite': {'2theta': 29.4, 'intensity': 80, 'decomposition_temp': 700},
            'quartz': {'2theta': 26.6, 'intensity': 60, 'decomposition_temp': 1000},
            'ettringite': {'2theta': 9.1, 'intensity': 40, 'decomposition_temp': 100},
            'C-S-H': {'2theta': 29.0, 'intensity': 120, 'decomposition_temp': 200}
        }
        
        xrd_peaks = []
        
        for phase_name, phase_data in phases.items():
            if exposure_temp < phase_data['decomposition_temp']:
                # Phase still present
                intensity = phase_data['intensity'] * (1 - (exposure_temp / phase_data['decomposition_temp']) * 0.5)
                
                # Rubber affects phase stability
                if phase_name in ['portlandite', 'ettringite']:
                    intensity *= (1 + (rubber_content / 100) * 0.1)
                
                xrd_peaks.append({
                    'phase': phase_name,
                    '2theta': phase_data['2theta'],
                    'intensity': intensity,
                    'fwhm': 0.1 + (exposure_temp / 1000) * 0.05
                })
        
        # Calculate phase change index
        phase_change_index = sum(1 - (exposure_temp / phase['decomposition_temp']) 
                               for phase in phases.values() 
                               if exposure_temp < phase['decomposition_temp']) / len(phases)
        
        return {
            'peaks': xrd_peaks,
            'phase_change_index': phase_change_index,
            'total_phases_detected': len(xrd_peaks)
        }
    
    def generate_all_spalling_durability_data(self):
        """Generate complete spalling and durability dataset"""
        all_data = {}
        
        for mix_name, composition in self.mix_designs.items():
            rubber_content = composition['rubber']
            
            print(f"Generating spalling/durability data for {mix_name} (Rubber: {rubber_content}%)")
            
            mix_data = {
                'visual_acoustic': self.generate_visual_acoustic_data(mix_name, rubber_content),
                'vapor_pressure': self.generate_vapor_pressure_data(mix_name, rubber_content),
                'gas_permeability': self.generate_gas_permeability_data(mix_name, rubber_content),
                'microstructural_analysis': self.generate_microstructural_analysis_data(mix_name, rubber_content)
            }
            
            all_data[mix_name] = mix_data
        
        return all_data
    
    def save_spalling_durability_data(self, data, output_dir='/workspace/spalling_durability_data'):
        """Save spalling and durability data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for easy access
        with open(f'{output_dir}/spalling_durability_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files for each property
        for mix_name, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_name}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # Visual/acoustic data
            va_df = pd.DataFrame(mix_data['visual_acoustic'])
            va_df.to_csv(f'{mix_dir}/visual_acoustic_data.csv', index=False)
            
            # Vapor pressure data
            vp_dir = f'{mix_dir}/vapor_pressure'
            os.makedirs(vp_dir, exist_ok=True)
            
            for depth, vp_data in mix_data['vapor_pressure']['vapor_pressure_data'].items():
                vp_df = pd.DataFrame(vp_data)
                vp_df.to_csv(f'{vp_dir}/vapor_pressure_{depth}mm.csv', index=False)
            
            # Gas permeability data
            gp_df = pd.DataFrame(mix_data['gas_permeability']['permeability_data'])
            gp_df.to_csv(f'{mix_dir}/gas_permeability.csv', index=False)
            
            # Microstructural analysis data
            ms_dir = f'{mix_dir}/microstructural_analysis'
            os.makedirs(ms_dir, exist_ok=True)
            
            for temp, ms_data in mix_data['microstructural_analysis'].items():
                # SEM data
                sem_df = pd.DataFrame([ms_data['sem_analysis']])
                sem_df.to_csv(f'{ms_dir}/sem_analysis_{temp}C.csv', index=False)
                
                # XRD data
                xrd_df = pd.DataFrame(ms_data['xrd_analysis']['peaks'])
                xrd_df.to_csv(f'{ms_dir}/xrd_analysis_{temp}C.csv', index=False)
        
        print(f"Spalling/durability data saved to {output_dir}")
    
    def create_spalling_durability_plots(self, data, output_dir='/workspace/spalling_durability_data/plots'):
        """Create visualization plots for spalling and durability data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Spalling events over time
        plt.figure(figsize=(12, 8))
        for mix_name, mix_data in data.items():
            va_data = mix_data['visual_acoustic']
            plt.plot(va_data['temperature'], va_data['spalling_events'], 
                    'o', label=f'{mix_name} (Rubber: {va_data["rubber_content"]}%)', 
                    markersize=3, alpha=0.7)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Spalling Events')
        plt.title('Spalling Events vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/spalling_events.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Vapor pressure at different depths
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, mix_name in enumerate(data.keys()):
            ax = axes[i]
            vp_data = data[mix_name]['vapor_pressure']
            
            for depth, depth_data in vp_data['vapor_pressure_data'].items():
                ax.plot(depth_data['temperature'], depth_data['vapor_pressure'] / 1000, 
                       label=f'{depth}mm depth', linewidth=2)
            
            ax.set_xlabel('Temperature (°C)')
            ax.set_ylabel('Vapor Pressure (kPa)')
            ax.set_title(f'{mix_name} - Vapor Pressure')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/vapor_pressure_depths.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gas permeability vs temperature
        plt.figure(figsize=(10, 6))
        for mix_name, mix_data in data.items():
            gp_data = mix_data['gas_permeability']
            temps = [d['temperature'] for d in gp_data['permeability_data']]
            perms = [d['permeability'] for d in gp_data['permeability_data']]
            
            plt.semilogy(temps, perms, 'o-', 
                        label=f'{mix_name} (Rubber: {gp_data["rubber_content"]}%)', 
                        linewidth=2, markersize=6)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Gas Permeability (m²)')
        plt.title('Gas Permeability vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/gas_permeability.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Microstructural damage index
        plt.figure(figsize=(10, 6))
        for mix_name, mix_data in data.items():
            temps = []
            damage_indices = []
            
            for temp, ms_data in mix_data['microstructural_analysis'].items():
                temps.append(temp)
                damage_indices.append(ms_data['sem_analysis']['microstructural_damage_index'])
            
            plt.plot(temps, damage_indices, 'o-', 
                    label=f'{mix_name} (Rubber: {mix_data["visual_acoustic"]["rubber_content"]}%)', 
                    linewidth=2, markersize=6)
        
        plt.xlabel('Exposure Temperature (°C)')
        plt.ylabel('Microstructural Damage Index')
        plt.title('Microstructural Damage vs Exposure Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/microstructural_damage.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Spalling/durability plots saved to {output_dir}")

def main():
    """Main function to generate spalling and durability dataset"""
    print("Generating Spalling and Durability Dataset")
    print("=" * 60)
    
    generator = SpallingDurabilityGenerator()
    
    # Generate all spalling/durability data
    spalling_data = generator.generate_all_spalling_durability_data()
    
    # Save data
    generator.save_spalling_durability_data(spalling_data)
    
    # Create plots
    generator.create_spalling_durability_plots(spalling_data)
    
    print("\nSpalling and Durability Dataset Generation Complete!")
    print("Generated data for 4 concrete mixes:")
    for mix_name in generator.mix_designs.keys():
        print(f"  - {mix_name}")
    
    print("\nData includes:")
    print("  - Visual and acoustic spalling event recording")
    print("  - Vapor pressure measurements at 5 depths")
    print("  - Gas permeability at elevated temperatures")
    print("  - Post-exposure microstructural analysis (SEM, XRD)")

if __name__ == "__main__":
    main()