#!/usr/bin/env python3
"""
High-Temperature Mechanical Testing Dataset Generator
Pillar 2: High-Temperature Experimental Investigation - Mechanical Properties

This module generates comprehensive mechanical testing data including:
- Transient-Test-Stress (TTS) curves at multiple temperatures
- Stressed-Test-Temperature (STT) tests
- Residual property tests after cooling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import json
from datetime import datetime
import os

class MechanicalTestingGenerator:
    def __init__(self):
        self.mix_types = {
            'Control': {'rubber_content': 0.0, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R5': {'rubber_content': 0.05, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R10': {'rubber_content': 0.10, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R15': {'rubber_content': 0.15, 'w_c_ratio': 0.45, 'cement_type': 'OPC'},
            'R20': {'rubber_content': 0.20, 'w_c_ratio': 0.45, 'cement_type': 'OPC'}
        }
        
        # Test temperatures (°C)
        self.test_temperatures = [25, 100, 200, 400, 600, 800]
        
        # Ambient properties (at 25°C)
        self.ambient_properties = {
            'compressive_strength': 45.0,  # MPa
            'tensile_strength': 4.5,      # MPa
            'modulus_elasticity': 30000,   # MPa
            'poisson_ratio': 0.2
        }
    
    def generate_tts_compressive_data(self, mix_type, temperature):
        """Generate Transient-Test-Stress compressive data at specific temperature"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Temperature effect on strength
        temp_factor = self._get_temperature_factor(temperature, 'compressive')
        
        # Rubber effect (rubber generally reduces strength but improves ductility)
        rubber_strength_factor = 1 - 0.3 * rubber_content
        rubber_ductility_factor = 1 + 0.5 * rubber_content
        
        # Peak strength at temperature
        peak_strength = self.ambient_properties['compressive_strength'] * temp_factor * rubber_strength_factor
        
        # Peak strain (ductility increases with temperature and rubber content)
        peak_strain = 0.003 * (1 + 2 * (temperature - 25) / 775) * rubber_ductility_factor
        
        # Generate stress-strain curve
        strain = np.linspace(0, peak_strain * 1.5, 1000)
        
        # Elastic modulus at temperature
        E_temp = self.ambient_properties['modulus_elasticity'] * temp_factor * (1 - 0.2 * rubber_content)
        
        # Stress-strain relationship
        stress = np.zeros_like(strain)
        
        # Linear elastic region
        elastic_limit = peak_strain * 0.3
        elastic_mask = strain <= elastic_limit
        stress[elastic_mask] = E_temp * strain[elastic_mask]
        
        # Nonlinear hardening region
        hardening_mask = (strain > elastic_limit) & (strain <= peak_strain)
        if np.any(hardening_mask):
            hardening_strain = strain[hardening_mask] - elastic_limit
            stress[hardening_mask] = (E_temp * elastic_limit + 
                                    peak_strength * (1 - np.exp(-5 * hardening_strain / peak_strain)))
        
        # Softening region
        softening_mask = strain > peak_strength
        if np.any(softening_mask):
            softening_factor = np.exp(-3 * (strain[softening_mask] - peak_strain) / peak_strain)
            stress[softening_mask] = peak_strength * softening_factor
        
        # Add experimental noise
        noise = np.random.normal(0, 0.5, len(stress))
        stress += noise
        stress = np.maximum(stress, 0)  # Ensure non-negative stress
        
        return {
            'strain': strain,
            'stress': stress,
            'peak_strength': peak_strength,
            'peak_strain': peak_strain,
            'modulus_elasticity': E_temp,
            'temperature': temperature
        }
    
    def generate_tts_tensile_data(self, mix_type, temperature):
        """Generate Transient-Test-Stress tensile data at specific temperature"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Temperature effect on tensile strength
        temp_factor = self._get_temperature_factor(temperature, 'tensile')
        
        # Rubber effect on tensile properties
        rubber_strength_factor = 1 - 0.4 * rubber_content  # Rubber reduces tensile strength more
        rubber_ductility_factor = 1 + 0.8 * rubber_content  # But improves ductility significantly
        
        # Peak tensile strength
        peak_strength = self.ambient_properties['tensile_strength'] * temp_factor * rubber_strength_factor
        
        # Peak tensile strain
        peak_strain = 0.0001 * (1 + 3 * (temperature - 25) / 775) * rubber_ductility_factor
        
        # Generate stress-strain curve
        strain = np.linspace(0, peak_strain * 2, 1000)
        
        # Tensile modulus
        E_temp = self.ambient_properties['modulus_elasticity'] * temp_factor * (1 - 0.3 * rubber_content)
        
        # Stress-strain relationship (more linear for tensile)
        stress = E_temp * strain * (1 - 0.5 * (strain / peak_strain) ** 2)
        stress = np.minimum(stress, peak_strength)
        
        # Add softening after peak
        peak_idx = np.argmin(np.abs(strain - peak_strain))
        if peak_idx < len(stress) - 1:
            softening_factor = np.exp(-2 * (strain[peak_idx:] - peak_strain) / peak_strain)
            stress[peak_idx:] = peak_strength * softening_factor
        
        # Add experimental noise
        noise = np.random.normal(0, 0.05, len(stress))
        stress += noise
        stress = np.maximum(stress, 0)
        
        return {
            'strain': strain,
            'stress': stress,
            'peak_strength': peak_strength,
            'peak_strain': peak_strain,
            'modulus_elasticity': E_temp,
            'temperature': temperature
        }
    
    def generate_stt_data(self, mix_type, stress_level):
        """Generate Stressed-Test-Temperature data at constant stress level"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Stress level as percentage of ambient strength
        stress_ratio = stress_level / 100.0
        
        # Base failure temperature (increases with rubber content due to improved ductility)
        base_failure_temp = 400 + 100 * rubber_content
        
        # Stress level effect (higher stress = lower failure temperature)
        stress_effect = -200 * stress_ratio
        
        # Critical failure temperature
        critical_temp = base_failure_temp + stress_effect
        
        # Add some variability
        critical_temp += np.random.normal(0, 20)
        critical_temp = max(100, critical_temp)  # Ensure reasonable minimum
        
        # Generate temperature vs time curve until failure
        time_to_failure = 30 + 20 * stress_ratio  # minutes
        time = np.linspace(0, time_to_failure, 1000)
        
        # Temperature ramp (heating rate decreases as temperature increases)
        temp = 25 + (critical_temp - 25) * (1 - np.exp(-time / (time_to_failure / 3)))
        
        return {
            'time': time,
            'temperature': temp,
            'critical_failure_temperature': critical_temp,
            'stress_level_percent': stress_level,
            'time_to_failure': time_to_failure
        }
    
    def generate_residual_properties(self, mix_type, exposure_temperature):
        """Generate residual properties after cooling from exposure temperature"""
        mix_props = self.mix_types[mix_type]
        rubber_content = mix_props['rubber_content']
        
        # Residual strength factors based on exposure temperature
        if exposure_temperature <= 200:
            residual_factor = 0.95 - 0.05 * (exposure_temperature - 25) / 175
        elif exposure_temperature <= 400:
            residual_factor = 0.9 - 0.3 * (exposure_temperature - 200) / 200
        elif exposure_temperature <= 600:
            residual_factor = 0.6 - 0.3 * (exposure_temperature - 400) / 200
        else:
            residual_factor = 0.3 - 0.2 * (exposure_temperature - 600) / 200
        
        # Rubber content effect on residual properties
        rubber_residual_factor = 1 + 0.2 * rubber_content  # Rubber helps retain some properties
        
        # Apply factors
        final_residual_factor = residual_factor * rubber_residual_factor
        final_residual_factor = np.clip(final_residual_factor, 0.1, 1.0)
        
        # Generate residual properties
        residual_compressive = self.ambient_properties['compressive_strength'] * final_residual_factor
        residual_tensile = self.ambient_properties['tensile_strength'] * final_residual_factor
        residual_modulus = self.ambient_properties['modulus_elasticity'] * final_residual_factor
        
        # UPV (Ultrasonic Pulse Velocity) - decreases with damage
        upv_factor = 0.8 + 0.2 * final_residual_factor
        upv = 4000 * upv_factor  # m/s
        
        # Dynamic modulus (related to UPV)
        density = 2400  # kg/m³ (assumed)
        dynamic_modulus = density * (upv ** 2) / 1e6  # MPa
        
        # Add experimental uncertainty
        noise_factor = 1 + np.random.normal(0, 0.05)
        
        return {
            'exposure_temperature': exposure_temperature,
            'residual_compressive_strength': residual_compressive * noise_factor,
            'residual_tensile_strength': residual_tensile * noise_factor,
            'residual_modulus_elasticity': residual_modulus * noise_factor,
            'upv': upv * noise_factor,
            'dynamic_modulus': dynamic_modulus * noise_factor,
            'residual_factor': final_residual_factor
        }
    
    def _get_temperature_factor(self, temperature, property_type):
        """Get temperature reduction factor for mechanical properties"""
        if temperature <= 100:
            return 1.0 - 0.1 * (temperature - 25) / 75
        elif temperature <= 200:
            return 0.9 - 0.2 * (temperature - 100) / 100
        elif temperature <= 400:
            return 0.7 - 0.3 * (temperature - 200) / 200
        elif temperature <= 600:
            return 0.4 - 0.2 * (temperature - 400) / 200
        else:
            return 0.2 - 0.1 * (temperature - 600) / 200
    
    def generate_all_mechanical_data(self):
        """Generate complete mechanical testing dataset"""
        all_data = {}
        
        for mix_type in self.mix_types.keys():
            print(f"Generating mechanical data for {mix_type}...")
            
            all_data[mix_type] = {
                'mix_properties': self.mix_types[mix_type],
                'tts_compressive': {},
                'tts_tensile': {},
                'stt_tests': {},
                'residual_properties': {}
            }
            
            # TTS Compressive tests
            for temp in self.test_temperatures:
                all_data[mix_type]['tts_compressive'][f'{temp}C'] = self.generate_tts_compressive_data(mix_type, temp)
            
            # TTS Tensile tests
            for temp in self.test_temperatures:
                all_data[mix_type]['tts_tensile'][f'{temp}C'] = self.generate_tts_tensile_data(mix_type, temp)
            
            # STT tests at different stress levels
            stress_levels = [20, 40, 60, 80]
            for stress_level in stress_levels:
                all_data[mix_type]['stt_tests'][f'{stress_level}%'] = self.generate_stt_data(mix_type, stress_level)
            
            # Residual properties after different exposure temperatures
            exposure_temps = [200, 400, 600, 800]
            for exp_temp in exposure_temps:
                all_data[mix_type]['residual_properties'][f'{exp_temp}C'] = self.generate_residual_properties(mix_type, exp_temp)
        
        return all_data
    
    def save_data(self, data, output_dir='/workspace/mechanical_data'):
        """Save all mechanical data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(f'{output_dir}/mechanical_testing_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files
        for mix_type, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_type}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # TTS Compressive data
            tts_comp_dir = f'{mix_dir}/tts_compressive'
            os.makedirs(tts_comp_dir, exist_ok=True)
            for temp_key, temp_data in mix_data['tts_compressive'].items():
                df = pd.DataFrame({
                    'strain': temp_data['strain'],
                    'stress_MPa': temp_data['stress']
                })
                df.to_csv(f'{tts_comp_dir}/tts_comp_{temp_key}.csv', index=False)
            
            # TTS Tensile data
            tts_tensile_dir = f'{mix_dir}/tts_tensile'
            os.makedirs(tts_tensile_dir, exist_ok=True)
            for temp_key, temp_data in mix_data['tts_tensile'].items():
                df = pd.DataFrame({
                    'strain': temp_data['strain'],
                    'stress_MPa': temp_data['stress']
                })
                df.to_csv(f'{tts_tensile_dir}/tts_tensile_{temp_key}.csv', index=False)
            
            # STT data
            stt_dir = f'{mix_dir}/stt_tests'
            os.makedirs(stt_dir, exist_ok=True)
            for stress_key, stress_data in mix_data['stt_tests'].items():
                df = pd.DataFrame({
                    'time_min': stress_data['time'],
                    'temperature_C': stress_data['temperature']
                })
                df.to_csv(f'{stt_dir}/stt_{stress_key}.csv', index=False)
            
            # Residual properties
            residual_df = pd.DataFrame([
                {
                    'exposure_temperature_C': data['exposure_temperature'],
                    'residual_compressive_strength_MPa': data['residual_compressive_strength'],
                    'residual_tensile_strength_MPa': data['residual_tensile_strength'],
                    'residual_modulus_elasticity_MPa': data['residual_modulus_elasticity'],
                    'upv_m_s': data['upv'],
                    'dynamic_modulus_MPa': data['dynamic_modulus'],
                    'residual_factor': data['residual_factor']
                }
                for data in mix_data['residual_properties'].values()
            ])
            residual_df.to_csv(f'{mix_dir}/residual_properties.csv', index=False)
        
        print(f"Mechanical data saved to {output_dir}")
    
    def create_visualizations(self, data, output_dir='/workspace/mechanical_data/plots'):
        """Create comprehensive visualizations of mechanical data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # TTS Compressive strength vs temperature
        plt.figure(figsize=(12, 8))
        for mix_type, mix_data in data.items():
            temps = []
            strengths = []
            for temp_key, temp_data in mix_data['tts_compressive'].items():
                temps.append(temp_data['temperature'])
                strengths.append(temp_data['peak_strength'])
            
            temps, strengths = zip(*sorted(zip(temps, strengths)))
            plt.plot(temps, strengths, 'o-', 
                    label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                    markersize=8, linewidth=2)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Compressive Strength (MPa)')
        plt.title('Transient-Test-Stress: Compressive Strength vs Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/tts_compressive_strength.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residual strength vs exposure temperature
        plt.figure(figsize=(12, 8))
        for mix_type, mix_data in data.items():
            exp_temps = []
            residual_strengths = []
            for temp_key, temp_data in mix_data['residual_properties'].items():
                exp_temps.append(temp_data['exposure_temperature'])
                residual_strengths.append(temp_data['residual_compressive_strength'])
            
            exp_temps, residual_strengths = zip(*sorted(zip(exp_temps, residual_strengths)))
            plt.plot(exp_temps, residual_strengths, 'o-',
                    label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                    markersize=8, linewidth=2)
        
        plt.xlabel('Exposure Temperature (°C)')
        plt.ylabel('Residual Compressive Strength (MPa)')
        plt.title('Residual Properties: Compressive Strength vs Exposure Temperature')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/residual_compressive_strength.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # STT critical failure temperature vs stress level
        plt.figure(figsize=(10, 6))
        for mix_type, mix_data in data.items():
            stress_levels = []
            critical_temps = []
            for stress_key, stress_data in mix_data['stt_tests'].items():
                stress_levels.append(stress_data['stress_level_percent'])
                critical_temps.append(stress_data['critical_failure_temperature'])
            
            stress_levels, critical_temps = zip(*sorted(zip(stress_levels, critical_temps)))
            plt.plot(stress_levels, critical_temps, 'o-',
                    label=f'{mix_type} (R={self.mix_types[mix_type]["rubber_content"]*100:.0f}%)',
                    markersize=8, linewidth=2)
        
        plt.xlabel('Stress Level (% of Ambient Strength)')
        plt.ylabel('Critical Failure Temperature (°C)')
        plt.title('Stressed-Test-Temperature: Critical Failure Temperature vs Stress Level')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/stt_critical_temperature.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to generate and save mechanical testing dataset"""
    print("Generating High-Temperature Mechanical Testing Dataset...")
    print("=" * 60)
    
    generator = MechanicalTestingGenerator()
    
    # Generate all mechanical data
    mechanical_data = generator.generate_all_mechanical_data()
    
    # Save data
    generator.save_data(mechanical_data)
    
    # Create visualizations
    generator.create_visualizations(mechanical_data)
    
    print("\nMechanical Testing Dataset Generation Complete!")
    print("Generated data includes:")
    print("- TTS curves at 25°C, 100°C, 200°C, 400°C, 600°C, 800°C")
    print("- STT tests at 20%, 40%, 60%, 80% stress levels")
    print("- Residual properties after exposure to 200°C, 400°C, 600°C, 800°C")
    print(f"- Data saved to /workspace/mechanical_data/")
    print(f"- Plots saved to /workspace/mechanical_data/plots/")

if __name__ == "__main__":
    main()