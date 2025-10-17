#!/usr/bin/env python3
"""
High-Temperature Mechanical Testing Dataset Generator for Fire-Resistant Rubberized Concrete
Generates comprehensive mechanical testing data including TTS curves, STT tests, and residual properties.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import json
from datetime import datetime
import os

class MechanicalTestingGenerator:
    def __init__(self):
        self.mix_designs = {
            'Control': {'cement': 100, 'water': 40, 'aggregate': 180, 'rubber': 0},
            'Low_Rubber': {'cement': 100, 'water': 40, 'aggregate': 160, 'rubber': 20},
            'Medium_Rubber': {'cement': 100, 'water': 40, 'aggregate': 140, 'rubber': 40},
            'High_Rubber': {'cement': 100, 'water': 40, 'aggregate': 120, 'rubber': 60}
        }
        
        # Ambient temperature properties (MPa)
        self.ambient_properties = {
            'Control': {'fc': 45, 'ft': 4.5, 'E': 30000, 'strain_peak': 0.002},
            'Low_Rubber': {'fc': 42, 'ft': 4.2, 'E': 28000, 'strain_peak': 0.0025},
            'Medium_Rubber': {'fc': 38, 'ft': 3.8, 'E': 25000, 'strain_peak': 0.003},
            'High_Rubber': {'fc': 32, 'ft': 3.2, 'E': 20000, 'strain_peak': 0.004}
        }
        
        # Test temperatures (°C)
        self.test_temperatures = [25, 100, 200, 400, 600, 800]
        
    def generate_tts_curves(self, mix_name, rubber_content):
        """Generate Transient-Test-Stress (TTS) curves at different temperatures"""
        tts_data = {}
        
        ambient_props = self.ambient_properties[mix_name]
        
        for temp in self.test_temperatures:
            # Temperature reduction factors
            temp_factors = self._calculate_temperature_factors(temp, rubber_content)
            
            # Generate stress-strain curve
            strain_range = np.linspace(0, 0.01, 1000)
            stress = self._generate_stress_strain_curve(
                strain_range, temp, ambient_props, temp_factors, rubber_content
            )
            
            # Calculate key properties
            peak_stress = np.max(stress)
            peak_strain_idx = np.argmax(stress)
            peak_strain = strain_range[peak_strain_idx]
            
            # Modulus of elasticity (slope of initial linear portion)
            linear_region = strain_range <= 0.001
            if np.sum(linear_region) > 10:
                E = np.polyfit(strain_range[linear_region], stress[linear_region], 1)[0]
            else:
                E = temp_factors['E'] * ambient_props['E']
            
            tts_data[temp] = {
                'temperature': temp,
                'strain': strain_range,
                'stress': stress,
                'peak_stress': peak_stress,
                'peak_strain': peak_strain,
                'modulus': E,
                'mix_name': mix_name,
                'rubber_content': rubber_content
            }
        
        return tts_data
    
    def _calculate_temperature_factors(self, temp, rubber_content):
        """Calculate temperature reduction factors for mechanical properties"""
        if temp <= 25:
            return {'fc': 1.0, 'ft': 1.0, 'E': 1.0, 'strain': 1.0}
        
        # Rubber provides some thermal protection
        rubber_protection = 1 + (rubber_content / 100) * 0.1
        
        # Compressive strength reduction
        if temp <= 100:
            fc_factor = 1.0
        elif temp <= 200:
            fc_factor = 0.95 * rubber_protection
        elif temp <= 400:
            fc_factor = 0.8 * rubber_protection
        elif temp <= 600:
            fc_factor = 0.5 * rubber_protection
        else:
            fc_factor = 0.2 * rubber_protection
        
        # Tensile strength reduction (more severe)
        if temp <= 100:
            ft_factor = 1.0
        elif temp <= 200:
            ft_factor = 0.9 * rubber_protection
        elif temp <= 400:
            ft_factor = 0.6 * rubber_protection
        elif temp <= 600:
            ft_factor = 0.3 * rubber_protection
        else:
            ft_factor = 0.1 * rubber_protection
        
        # Modulus reduction
        if temp <= 100:
            E_factor = 1.0
        elif temp <= 200:
            E_factor = 0.9 * rubber_protection
        elif temp <= 400:
            E_factor = 0.7 * rubber_protection
        elif temp <= 600:
            E_factor = 0.4 * rubber_protection
        else:
            E_factor = 0.2 * rubber_protection
        
        # Strain capacity (increases with temperature due to thermal expansion)
        strain_factor = 1 + 0.0001 * (temp - 25)
        
        return {
            'fc': fc_factor,
            'ft': ft_factor,
            'E': E_factor,
            'strain': strain_factor
        }
    
    def _generate_stress_strain_curve(self, strain, temp, ambient_props, factors, rubber_content):
        """Generate realistic stress-strain curve at given temperature"""
        fc = ambient_props['fc'] * factors['fc']
        E = ambient_props['E'] * factors['E']
        peak_strain = ambient_props['strain_peak'] * factors['strain']
        
        # Add rubber content effect on ductility
        rubber_ductility = 1 + (rubber_content / 100) * 0.5
        
        stress = np.zeros_like(strain)
        
        # Linear elastic region
        linear_mask = strain <= peak_strain
        stress[linear_mask] = E * strain[linear_mask]
        
        # Post-peak behavior (more ductile with rubber and temperature)
        post_peak_mask = strain > peak_strain
        if np.any(post_peak_mask):
            # Softening curve
            strain_normalized = (strain[post_peak_mask] - peak_strain) / peak_strain
            softening_factor = np.exp(-2 * strain_normalized * rubber_ductility)
            stress[post_peak_mask] = fc * softening_factor
        
        # Add realistic noise
        noise = np.random.normal(0, 0.5, len(stress))
        stress += noise
        stress = np.maximum(stress, 0)  # No negative stress
        
        return stress
    
    def generate_stt_tests(self, mix_name, rubber_content):
        """Generate Stressed-Test-Temperature (STT) test data"""
        ambient_fc = self.ambient_properties[mix_name]['fc']
        stress_levels = [0.2, 0.4, 0.6, 0.8]  # Percentage of ambient strength
        
        stt_data = {}
        
        for stress_level in stress_levels:
            applied_stress = ambient_fc * stress_level
            
            # Generate temperature ramp until failure
            temp_ramp = np.linspace(25, 800, 1000)
            time_ramp = np.linspace(0, 120, 1000)  # 2 hours
            
            # Calculate failure temperature
            failure_temp = self._calculate_failure_temperature(
                applied_stress, ambient_fc, rubber_content
            )
            
            # Generate stress vs temperature curve
            stress_retention = []
            for temp in temp_ramp:
                if temp <= failure_temp:
                    factors = self._calculate_temperature_factors(temp, rubber_content)
                    retained_stress = ambient_fc * factors['fc']
                    stress_retention.append(min(retained_stress, applied_stress))
                else:
                    stress_retention.append(0)
            
            stt_data[stress_level] = {
                'stress_level': stress_level,
                'applied_stress': applied_stress,
                'temperature': temp_ramp,
                'time': time_ramp,
                'stress_retention': stress_retention,
                'failure_temperature': failure_temp,
                'mix_name': mix_name,
                'rubber_content': rubber_content
            }
        
        return stt_data
    
    def _calculate_failure_temperature(self, applied_stress, ambient_fc, rubber_content):
        """Calculate critical failure temperature for given stress level"""
        stress_ratio = applied_stress / ambient_fc
        
        # Base failure temperature
        if stress_ratio <= 0.2:
            base_temp = 700
        elif stress_ratio <= 0.4:
            base_temp = 600
        elif stress_ratio <= 0.6:
            base_temp = 500
        elif stress_ratio <= 0.8:
            base_temp = 400
        else:
            base_temp = 300
        
        # Rubber provides thermal protection
        rubber_protection = 1 + (rubber_content / 100) * 0.2
        
        failure_temp = base_temp * rubber_protection
        
        # Add some variability
        failure_temp += np.random.normal(0, 20)
        
        return min(failure_temp, 800)
    
    def generate_residual_properties(self, mix_name, rubber_content):
        """Generate residual properties after cooling from high temperatures"""
        residual_data = {}
        
        ambient_props = self.ambient_properties[mix_name]
        
        for temp in self.test_temperatures:
            if temp == 25:
                # No heating, use ambient properties
                residual_data[temp] = {
                    'exposure_temperature': temp,
                    'residual_fc': ambient_props['fc'],
                    'residual_ft': ambient_props['ft'],
                    'residual_E': ambient_props['E'],
                    'residual_UPV': 4000,  # m/s
                    'residual_dynamic_modulus': ambient_props['E'],
                    'residual_strength_retention': 1.0,
                    'mix_name': mix_name,
                    'rubber_content': rubber_content
                }
                continue
            
            # Calculate residual properties after cooling
            residual_factors = self._calculate_residual_factors(temp, rubber_content)
            
            residual_fc = ambient_props['fc'] * residual_factors['fc']
            residual_ft = ambient_props['ft'] * residual_factors['ft']
            residual_E = ambient_props['E'] * residual_factors['E']
            
            # UPV decreases with damage
            residual_UPV = 4000 * residual_factors['UPV']
            
            # Dynamic modulus
            residual_dynamic_modulus = residual_E * residual_factors['dynamic']
            
            # Strength retention
            strength_retention = residual_fc / ambient_props['fc']
            
            residual_data[temp] = {
                'exposure_temperature': temp,
                'residual_fc': residual_fc,
                'residual_ft': residual_ft,
                'residual_E': residual_E,
                'residual_UPV': residual_UPV,
                'residual_dynamic_modulus': residual_dynamic_modulus,
                'residual_strength_retention': strength_retention,
                'mix_name': mix_name,
                'rubber_content': rubber_content
            }
        
        return residual_data
    
    def _calculate_residual_factors(self, exposure_temp, rubber_content):
        """Calculate residual property factors after cooling"""
        # Rubber provides some protection against thermal damage
        rubber_protection = 1 + (rubber_content / 100) * 0.15
        
        if exposure_temp <= 100:
            # Minimal damage
            fc_factor = 0.98 * rubber_protection
            ft_factor = 0.95 * rubber_protection
            E_factor = 0.95 * rubber_protection
            UPV_factor = 0.98
            dynamic_factor = 0.95
        elif exposure_temp <= 200:
            # Slight damage
            fc_factor = 0.90 * rubber_protection
            ft_factor = 0.85 * rubber_protection
            E_factor = 0.85 * rubber_protection
            UPV_factor = 0.90
            dynamic_factor = 0.85
        elif exposure_temp <= 400:
            # Moderate damage
            fc_factor = 0.70 * rubber_protection
            ft_factor = 0.60 * rubber_protection
            E_factor = 0.65 * rubber_protection
            UPV_factor = 0.75
            dynamic_factor = 0.65
        elif exposure_temp <= 600:
            # Severe damage
            fc_factor = 0.40 * rubber_protection
            ft_factor = 0.30 * rubber_protection
            E_factor = 0.35 * rubber_protection
            UPV_factor = 0.50
            dynamic_factor = 0.35
        else:
            # Very severe damage
            fc_factor = 0.20 * rubber_protection
            ft_factor = 0.15 * rubber_protection
            E_factor = 0.20 * rubber_protection
            UPV_factor = 0.30
            dynamic_factor = 0.20
        
        # Add some variability
        for key in ['fc_factor', 'ft_factor', 'E_factor', 'UPV_factor', 'dynamic_factor']:
            locals()[key] += np.random.normal(0, 0.05)
            locals()[key] = max(0, locals()[key])  # No negative values
        
        return {
            'fc': fc_factor,
            'ft': ft_factor,
            'E': E_factor,
            'UPV': UPV_factor,
            'dynamic': dynamic_factor
        }
    
    def generate_all_mechanical_data(self):
        """Generate complete mechanical testing dataset"""
        all_data = {}
        
        for mix_name, composition in self.mix_designs.items():
            rubber_content = composition['rubber']
            
            print(f"Generating mechanical data for {mix_name} (Rubber: {rubber_content}%)")
            
            mix_data = {
                'tts_curves': self.generate_tts_curves(mix_name, rubber_content),
                'stt_tests': self.generate_stt_tests(mix_name, rubber_content),
                'residual_properties': self.generate_residual_properties(mix_name, rubber_content)
            }
            
            all_data[mix_name] = mix_data
        
        return all_data
    
    def save_mechanical_data(self, data, output_dir='/workspace/mechanical_data'):
        """Save mechanical data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON for easy access
        with open(f'{output_dir}/mechanical_testing_complete.json', 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save individual CSV files for each property
        for mix_name, mix_data in data.items():
            mix_dir = f'{output_dir}/{mix_name}'
            os.makedirs(mix_dir, exist_ok=True)
            
            # TTS curves
            tts_dir = f'{mix_dir}/tts_curves'
            os.makedirs(tts_dir, exist_ok=True)
            
            for temp, tts_data in mix_data['tts_curves'].items():
                tts_df = pd.DataFrame({
                    'strain': tts_data['strain'],
                    'stress': tts_data['stress']
                })
                tts_df.to_csv(f'{tts_dir}/tts_{temp}C.csv', index=False)
            
            # STT tests
            stt_dir = f'{mix_dir}/stt_tests'
            os.makedirs(stt_dir, exist_ok=True)
            
            for stress_level, stt_data in mix_data['stt_tests'].items():
                stt_df = pd.DataFrame({
                    'temperature': stt_data['temperature'],
                    'time': stt_data['time'],
                    'stress_retention': stt_data['stress_retention']
                })
                stt_df.to_csv(f'{stt_dir}/stt_{stress_level:.1f}.csv', index=False)
            
            # Residual properties
            residual_df = pd.DataFrame(mix_data['residual_properties']).T
            residual_df.to_csv(f'{mix_dir}/residual_properties.csv', index=False)
        
        print(f"Mechanical data saved to {output_dir}")
    
    def create_mechanical_plots(self, data, output_dir='/workspace/mechanical_data/plots'):
        """Create visualization plots for mechanical data"""
        os.makedirs(output_dir, exist_ok=True)
        
        # TTS curves at different temperatures
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, temp in enumerate(self.test_temperatures):
            ax = axes[i]
            
            for mix_name, mix_data in data.items():
                tts_data = mix_data['tts_curves'][temp]
                ax.plot(tts_data['strain'] * 1000, tts_data['stress'], 
                       label=f'{mix_name} (Rubber: {tts_data["rubber_content"]}%)', 
                       linewidth=2)
            
            ax.set_xlabel('Strain (×10⁻³)')
            ax.set_ylabel('Stress (MPa)')
            ax.set_title(f'TTS Curves at {temp}°C')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/tts_curves_all_temps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # STT tests
        plt.figure(figsize=(12, 8))
        for mix_name, mix_data in data.items():
            for stress_level, stt_data in mix_data['stt_tests'].items():
                plt.plot(stt_data['temperature'], stt_data['stress_retention'], 
                        'o-', label=f'{mix_name} - {stress_level:.1f}fc', 
                        linewidth=2, markersize=4)
        
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Stress Retention (MPa)')
        plt.title('Stressed-Test-Temperature (STT) Curves')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/stt_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Residual strength retention
        plt.figure(figsize=(10, 6))
        for mix_name, mix_data in data.items():
            temps = []
            retention = []
            for temp, residual_data in mix_data['residual_properties'].items():
                temps.append(temp)
                retention.append(residual_data['residual_strength_retention'])
            
            plt.plot(temps, retention, 'o-', 
                    label=f'{mix_name} (Rubber: {mix_data["tts_curves"][25]["rubber_content"]}%)', 
                    linewidth=2, markersize=6)
        
        plt.xlabel('Exposure Temperature (°C)')
        plt.ylabel('Residual Strength Retention')
        plt.title('Residual Compressive Strength After Cooling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{output_dir}/residual_strength.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mechanical plots saved to {output_dir}")

def main():
    """Main function to generate mechanical testing dataset"""
    print("Generating High-Temperature Mechanical Testing Dataset")
    print("=" * 60)
    
    generator = MechanicalTestingGenerator()
    
    # Generate all mechanical data
    mechanical_data = generator.generate_all_mechanical_data()
    
    # Save data
    generator.save_mechanical_data(mechanical_data)
    
    # Create plots
    generator.create_mechanical_plots(mechanical_data)
    
    print("\nMechanical Testing Dataset Generation Complete!")
    print("Generated data for 4 concrete mixes:")
    for mix_name in generator.mix_designs.keys():
        print(f"  - {mix_name}")
    
    print("\nData includes:")
    print("  - TTS curves at 6 temperatures (25-800°C)")
    print("  - STT tests at 4 stress levels (20-80% of ambient strength)")
    print("  - Residual properties after cooling from high temperatures")
    print("  - Peak strength, modulus, and ductility at each temperature")

if __name__ == "__main__":
    main()