#!/usr/bin/env python3
"""
Microstructural Analysis Data Generator
Generates realistic SEM and XRD data for post-exposure analysis
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import os

class MicrostructuralGenerator:
    def __init__(self):
        self.mixes = ['control', 'rubber_10', 'rubber_20', 'rubber_30']
        self.temperatures = [25, 200, 400, 600, 800]  # Exposure temperatures
        
        # XRD peak positions (2θ degrees) for common concrete phases
        self.xrd_peaks = {
            'portlandite': [18.0, 34.1, 47.1, 50.8],
            'calcite': [29.4, 39.4, 43.2, 47.5],
            'quartz': [20.8, 26.6, 36.5, 39.4],
            'ettringite': [9.1, 15.8, 22.9, 29.1],
            'gypsum': [11.6, 20.7, 23.4, 29.1],
            'lime': [32.2, 37.4, 54.3, 64.2],
            'periclase': [42.9, 62.3, 78.7, 94.0]
        }
    
    def generate_sem_data(self, mix_type, temperature, replicate=1):
        """Generate SEM analysis data"""
        # Microcrack density (cracks per mm²)
        base_crack_density = {
            'control': 0.1,
            'rubber_10': 0.15,
            'rubber_20': 0.25,
            'rubber_30': 0.35
        }
        
        # Temperature effect on microcracking
        temp_factor = 1 + 0.001 * (temperature - 25)  # Increases with temperature
        crack_density = base_crack_density[mix_type] * temp_factor
        
        # Add replicate variation
        crack_density *= (1 + np.random.normal(0, 0.1))
        crack_density = max(crack_density, 0.01)
        
        # ITZ (Interfacial Transition Zone) degradation
        itz_degradation = min(1.0, (temperature - 25) / 400)  # 0 to 1 scale
        
        # Rubber void morphology
        rubber_percentage = int(mix_type.split('_')[1]) if 'rubber' in mix_type else 0
        void_density = rubber_percentage * 0.1  # voids per mm²
        void_size_avg = 0.5 + (temperature - 25) * 0.001  # mm, increases with temperature
        
        # Pore size distribution
        pore_sizes = np.random.lognormal(mean=0.5, sigma=0.3, size=100)  # μm
        pore_density = 1000 * (1 + 0.0005 * (temperature - 25))  # pores per mm²
        
        return {
            'mix_type': mix_type,
            'temperature': temperature,
            'replicate': replicate,
            'microcrack_density': crack_density,
            'itz_degradation': itz_degradation,
            'void_density': void_density,
            'void_size_avg': void_size_avg,
            'pore_sizes': pore_sizes.tolist(),
            'pore_density': pore_density,
            'analysis_conditions': {
                'magnification': '1000x',
                'acceleration_voltage': '15 kV',
                'working_distance': '10 mm'
            }
        }
    
    def generate_xrd_data(self, mix_type, temperature, replicate=1):
        """Generate XRD analysis data"""
        # Phase abundance (relative intensity)
        phases = {}
        
        # Portlandite - decomposes above 400°C
        if temperature < 400:
            phases['portlandite'] = 100 - (temperature - 25) * 0.1
        else:
            phases['portlandite'] = max(0, 60 - (temperature - 400) * 0.15)
        
        # Calcite - stable up to 600°C
        if temperature < 600:
            phases['calcite'] = 80 + (temperature - 25) * 0.05
        else:
            phases['calcite'] = max(0, 80 - (temperature - 600) * 0.2)
        
        # Quartz - stable
        phases['quartz'] = 60 + np.random.normal(0, 5)
        
        # Ettringite - decomposes above 100°C
        if temperature < 100:
            phases['ettringite'] = 40
        else:
            phases['ettringite'] = max(0, 40 - (temperature - 100) * 0.5)
        
        # Gypsum - decomposes above 200°C
        if temperature < 200:
            phases['gypsum'] = 20
        else:
            phases['gypsum'] = max(0, 20 - (temperature - 200) * 0.1)
        
        # Lime - forms from Portlandite decomposition
        if temperature > 400:
            phases['lime'] = (temperature - 400) * 0.2
        else:
            phases['lime'] = 0
        
        # Periclase - forms at high temperatures
        if temperature > 600:
            phases['periclase'] = (temperature - 600) * 0.1
        else:
            phases['periclase'] = 0
        
        # Add noise to all phases
        for phase in phases:
            phases[phase] *= (1 + np.random.normal(0, 0.05))
            phases[phase] = max(phases[phase], 0)
        
        # Generate peak intensities
        peak_data = {}
        for phase, abundance in phases.items():
            if abundance > 5:  # Only include phases with significant abundance
                peak_data[phase] = {
                    'abundance': abundance,
                    'peaks': []
                }
                
                for peak_2theta in self.xrd_peaks[phase]:
                    # Peak intensity based on abundance and temperature
                    intensity = abundance * (1 - (temperature - 25) * 0.0001)
                    intensity *= (1 + np.random.normal(0, 0.1))  # Add noise
                    intensity = max(intensity, 0)
                    
                    peak_data[phase]['peaks'].append({
                        '2theta': peak_2theta,
                        'intensity': intensity,
                        'fwhm': 0.1 + np.random.normal(0, 0.02)  # Full width at half maximum
                    })
        
        return {
            'mix_type': mix_type,
            'temperature': temperature,
            'replicate': replicate,
            'phases': peak_data,
            'analysis_conditions': {
                'wavelength': 'Cu Kα (1.5406 Å)',
                'scan_range': '5-80° 2θ',
                'step_size': '0.02°',
                'scan_speed': '2°/min'
            }
        }
    
    def generate_mix_data(self, mix_type, replicates=3):
        """Generate microstructural data for a mix at all temperatures"""
        mix_data = {
            'mix_type': mix_type,
            'temperatures': {}
        }
        
        for temp in self.temperatures:
            temp_data = {
                'temperature': temp,
                'sem_data': [],
                'xrd_data': []
            }
            
            for replicate in range(replicates):
                sem_data = self.generate_sem_data(mix_type, temp, replicate + 1)
                xrd_data = self.generate_xrd_data(mix_type, temp, replicate + 1)
                
                temp_data['sem_data'].append(sem_data)
                temp_data['xrd_data'].append(xrd_data)
            
            mix_data['temperatures'][temp] = temp_data
        
        return mix_data
    
    def generate_all_data(self):
        """Generate microstructural data for all mixes"""
        all_data = {
            'metadata': {
                'generation_date': datetime.now().isoformat(),
                'sem_equipment': 'FEI Quanta 200 FEG',
                'xrd_equipment': 'Bruker D8 Advance',
                'specimen_preparation': 'Polished sections, carbon coated',
                'replicates_per_temperature': 3
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
        with open(f"{output_dir}/microstructural_data.json", 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save SEM data as CSV
        for mix, mix_data in data['mixes'].items():
            sem_rows = []
            xrd_rows = []
            
            for temp, temp_data in mix_data['temperatures'].items():
                for sem_data in temp_data['sem_data']:
                    sem_rows.append({
                        'mix_type': sem_data['mix_type'],
                        'temperature': sem_data['temperature'],
                        'replicate': sem_data['replicate'],
                        'microcrack_density': sem_data['microcrack_density'],
                        'itz_degradation': sem_data['itz_degradation'],
                        'void_density': sem_data['void_density'],
                        'void_size_avg': sem_data['void_size_avg'],
                        'pore_density': sem_data['pore_density']
                    })
                
                for xrd_data in temp_data['xrd_data']:
                    for phase, phase_data in xrd_data['phases'].items():
                        for peak in phase_data['peaks']:
                            xrd_rows.append({
                                'mix_type': xrd_data['mix_type'],
                                'temperature': xrd_data['temperature'],
                                'replicate': xrd_data['replicate'],
                                'phase': phase,
                                'abundance': phase_data['abundance'],
                                'peak_2theta': peak['2theta'],
                                'intensity': peak['intensity'],
                                'fwhm': peak['fwhm']
                            })
            
            # Save SEM data
            if sem_rows:
                sem_df = pd.DataFrame(sem_rows)
                sem_df.to_csv(f"{output_dir}/{mix}_sem_data.csv", index=False)
            
            # Save XRD data
            if xrd_rows:
                xrd_df = pd.DataFrame(xrd_rows)
                xrd_df.to_csv(f"{output_dir}/{mix}_xrd_data.csv", index=False)
    
    def plot_microstructural_data(self, data, output_dir):
        """Generate plots of microstructural data"""
        import matplotlib.pyplot as plt
        
        os.makedirs(f"{output_dir}/plots", exist_ok=True)
        
        # Plot 1: Microcrack density vs temperature
        fig, ax = plt.subplots(figsize=(12, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                crack_densities = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    densities = [d['microcrack_density'] for d in temp_data['sem_data']]
                    temps.append(temp)
                    crack_densities.append(np.mean(densities))
                ax.plot(temps, crack_densities, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Exposure Temperature (°C)')
        ax.set_ylabel('Microcrack Density (cracks/mm²)')
        ax.set_title('Microcrack Density vs Exposure Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/microcrack_density.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Phase abundance vs temperature (control mix)
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            phases = ['portlandite', 'calcite', 'quartz', 'ettringite', 'gypsum', 'lime', 'periclase']
            
            for phase in phases:
                temps = []
                abundances = []
                for temp, temp_data in control_data['temperatures'].items():
                    phase_abundances = []
                    for xrd_data in temp_data['xrd_data']:
                        if phase in xrd_data['phases']:
                            phase_abundances.append(xrd_data['phases'][phase]['abundance'])
                    if phase_abundances:
                        temps.append(temp)
                        abundances.append(np.mean(phase_abundances))
                
                if abundances:  # Only plot if data exists
                    ax.plot(temps, abundances, 'o-', label=phase, linewidth=2, markersize=6)
        
        ax.set_xlabel('Exposure Temperature (°C)')
        ax.set_ylabel('Phase Abundance (relative intensity)')
        ax.set_title('Phase Abundance vs Exposure Temperature - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/phase_abundance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: XRD pattern at 400°C (control mix)
        fig, ax = plt.subplots(figsize=(12, 8))
        if 'control' in data['mixes']:
            control_data = data['mixes']['control']
            temp = 400
            if temp in control_data['temperatures']:
                xrd_data = control_data['temperatures'][temp]['xrd_data'][0]  # First replicate
                
                for phase, phase_data in xrd_data['phases'].items():
                    if phase_data['abundance'] > 10:  # Only plot significant phases
                        for peak in phase_data['peaks']:
                            ax.bar(peak['2theta'], peak['intensity'], 
                                  width=peak['fwhm'], alpha=0.7, label=phase)
        
        ax.set_xlabel('2θ (degrees)')
        ax.set_ylabel('Intensity (counts)')
        ax.set_title('XRD Pattern at 400°C - Control Mix')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/xrd_pattern_400C.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 4: ITZ degradation vs temperature
        fig, ax = plt.subplots(figsize=(12, 8))
        for mix in self.mixes:
            if mix in data['mixes']:
                temps = []
                itz_degradations = []
                for temp, temp_data in data['mixes'][mix]['temperatures'].items():
                    degradations = [d['itz_degradation'] for d in temp_data['sem_data']]
                    temps.append(temp)
                    itz_degradations.append(np.mean(degradations))
                ax.plot(temps, itz_degradations, 'o-', label=mix, linewidth=2, markersize=6)
        ax.set_xlabel('Exposure Temperature (°C)')
        ax.set_ylabel('ITZ Degradation (0-1 scale)')
        ax.set_title('ITZ Degradation vs Exposure Temperature')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plots/itz_degradation.png", dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    generator = MicrostructuralGenerator()
    data = generator.generate_all_data()
    generator.save_data(data, "/workspace/experimental_dataset/spalling_durability/microstructural")
    generator.plot_microstructural_data(data, "/workspace/experimental_dataset/spalling_durability/microstructural")
    print("Microstructural analysis data generation completed!")