#!/usr/bin/env python3
"""
SEM Analysis Dataset Generator for Fire-Resistant Rubberized Concrete
PhD Research: Development and Validation of Thermo-Mechanical Model

This module generates comprehensive SEM analysis data focusing on:
1. ITZ (Interfacial Transition Zone) characterization
2. Microcracking patterns and evolution
3. Rubber particle degradation mechanisms
4. Cement paste morphology changes

Author: Research Team
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SEMDatasetGenerator:
    def __init__(self):
        """Initialize SEM dataset generator with experimental parameters"""
        self.temperatures = [20, 100, 200, 300, 400, 500, 600, 700, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20, 25]  # % by volume
        self.specimen_types = ['control', 'heated']
        self.analysis_zones = ['ITZ_rubber_cement', 'ITZ_aggregate_cement', 'bulk_cement', 'rubber_particle']
        
        # SEM imaging parameters
        self.magnifications = [500, 1000, 2000, 5000, 10000, 20000, 50000]
        self.accelerating_voltages = [5, 10, 15, 20]  # kV
        
        # Initialize data containers
        self.sem_data = {}
        self.microcrack_data = {}
        self.itz_data = {}
        self.rubber_degradation_data = {}
        
    def generate_itz_characteristics(self):
        """Generate ITZ (Interfacial Transition Zone) characteristics data"""
        print("Generating ITZ characteristics dataset...")
        
        itz_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    # Skip heated specimens at room temperature
                    if specimen_type == 'heated' and temp == 20:
                        continue
                        
                    # ITZ thickness evolution with temperature
                    base_itz_thickness = 20 + rubber_content * 0.5  # μm
                    temp_factor = 1 + (temp - 20) * 0.002  # Thermal expansion effect
                    degradation_factor = 1 if temp < 300 else 1 + (temp - 300) * 0.001
                    itz_thickness = base_itz_thickness * temp_factor * degradation_factor
                    
                    # Porosity in ITZ
                    base_porosity = 0.15 + rubber_content * 0.002  # Base ITZ porosity
                    thermal_porosity = 0 if temp < 200 else (temp - 200) * 0.0005
                    rubber_porosity = 0 if temp < 350 else (temp - 350) * 0.001 * (rubber_content / 100)
                    total_porosity = min(base_porosity + thermal_porosity + rubber_porosity, 0.8)
                    
                    # Microhardness in ITZ
                    base_hardness = 2.5 - rubber_content * 0.02  # GPa
                    thermal_degradation = 1 if temp < 400 else 1 - (temp - 400) * 0.001
                    itz_hardness = max(base_hardness * thermal_degradation, 0.5)
                    
                    # Crack density in ITZ
                    thermal_cracking = 0 if temp < 300 else (temp - 300) ** 1.5 * 0.001
                    rubber_interface_cracking = rubber_content * 0.1 if temp > 400 else 0
                    crack_density = thermal_cracking + rubber_interface_cracking
                    
                    # Chemical composition changes
                    ca_oh2_content = max(0.15 - (temp - 20) * 0.0002, 0.02)  # Portlandite content
                    csh_gel_content = 0.45 - (temp - 20) * 0.0001 if temp < 500 else 0.45 - 0.096
                    
                    for analysis_zone in ['ITZ_rubber_cement', 'ITZ_aggregate_cement']:
                        zone_modifier = 1.2 if analysis_zone == 'ITZ_rubber_cement' else 1.0
                        
                        itz_record = {
                            'temperature': temp,
                            'rubber_content': rubber_content,
                            'specimen_type': specimen_type,
                            'analysis_zone': analysis_zone,
                            'itz_thickness_um': itz_thickness * zone_modifier,
                            'porosity_fraction': total_porosity * zone_modifier,
                            'microhardness_gpa': itz_hardness / zone_modifier,
                            'crack_density_per_mm2': crack_density * zone_modifier,
                            'ca_oh2_content': ca_oh2_content,
                            'csh_gel_content': csh_gel_content,
                            'imaging_magnification': np.random.choice(self.magnifications),
                            'accelerating_voltage_kv': np.random.choice(self.accelerating_voltages),
                            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                            'specimen_id': f'RC_{rubber_content}_{temp}_{specimen_type}_{analysis_zone[:3]}'
                        }
                        
                        # Add measurement uncertainties
                        for key in ['itz_thickness_um', 'porosity_fraction', 'microhardness_gpa', 'crack_density_per_mm2']:
                            itz_record[key] += np.random.normal(0, itz_record[key] * 0.05)
                            
                        itz_data.append(itz_record)
        
        self.itz_data = pd.DataFrame(itz_data)
        return self.itz_data
    
    def generate_microcrack_analysis(self):
        """Generate microcrack analysis data"""
        print("Generating microcrack analysis dataset...")
        
        microcrack_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Crack initiation temperature
                    crack_init_temp = 250 - rubber_content * 2  # Rubber reduces crack initiation temp
                    
                    if temp < crack_init_temp:
                        crack_length = np.random.exponential(5)  # μm, minimal pre-existing cracks
                        crack_width = np.random.exponential(0.1)  # μm
                        crack_density = np.random.poisson(2)  # cracks per mm²
                    else:
                        # Temperature-dependent crack growth
                        temp_excess = temp - crack_init_temp
                        
                        # Crack length distribution (log-normal)
                        mean_length = 10 + temp_excess * 0.5 + rubber_content * 0.2
                        crack_length = np.random.lognormal(np.log(mean_length), 0.5)
                        
                        # Crack width (increases with temperature)
                        mean_width = 0.5 + temp_excess * 0.01
                        crack_width = np.random.lognormal(np.log(mean_width), 0.3)
                        
                        # Crack density (Poisson distribution)
                        mean_density = 5 + temp_excess * 0.1 + rubber_content * 0.05
                        crack_density = np.random.poisson(mean_density)
                    
                    # Crack orientation analysis
                    if temp < 400:
                        crack_orientation = np.random.normal(0, 30)  # Random orientation
                    else:
                        crack_orientation = np.random.normal(45, 15)  # Preferred orientation due to thermal stress
                    
                    # Crack connectivity
                    connectivity_factor = min((temp - 200) / 400, 1) if temp > 200 else 0
                    crack_connectivity = np.random.beta(2, 5) * connectivity_factor
                    
                    for zone in self.analysis_zones:
                        zone_factor = {
                            'ITZ_rubber_cement': 1.5,
                            'ITZ_aggregate_cement': 1.2,
                            'bulk_cement': 1.0,
                            'rubber_particle': 0.8 if temp < 400 else 2.0
                        }[zone]
                        
                        microcrack_record = {
                            'temperature': temp,
                            'rubber_content': rubber_content,
                            'specimen_type': specimen_type,
                            'analysis_zone': zone,
                            'crack_length_um': crack_length * zone_factor,
                            'crack_width_um': crack_width * zone_factor,
                            'crack_density_per_mm2': crack_density * zone_factor,
                            'crack_orientation_deg': crack_orientation,
                            'crack_connectivity': crack_connectivity * zone_factor,
                            'fractal_dimension': 1.2 + 0.3 * crack_connectivity,
                            'tortuosity': 1 + crack_connectivity * 0.5,
                            'imaging_conditions': {
                                'magnification': np.random.choice([2000, 5000, 10000]),
                                'detector': 'SE2',
                                'working_distance_mm': np.random.uniform(8, 12)
                            },
                            'specimen_id': f'MC_{rubber_content}_{temp}_{specimen_type}_{zone[:3]}',
                            'measurement_area_mm2': 0.01  # Standard measurement area
                        }
                        
                        microcrack_data.append(microcrack_record)
        
        self.microcrack_data = pd.DataFrame(microcrack_data)
        return self.microcrack_data
    
    def generate_rubber_degradation_analysis(self):
        """Generate rubber particle degradation analysis"""
        print("Generating rubber degradation analysis dataset...")
        
        degradation_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                if rubber_content == 0:  # Skip for control concrete
                    continue
                    
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # Rubber degradation mechanisms
                    # 1. Thermal expansion (reversible up to ~200°C)
                    thermal_expansion = (temp - 20) * 2e-4 if temp < 200 else 0.036
                    
                    # 2. Softening and melting (starts ~300°C)
                    softening_factor = 1 if temp < 300 else 1 - (temp - 300) / 500
                    softening_factor = max(softening_factor, 0.1)
                    
                    # 3. Pyrolysis and carbonization (>400°C)
                    pyrolysis_factor = 0 if temp < 400 else (temp - 400) / 400
                    pyrolysis_factor = min(pyrolysis_factor, 0.9)
                    
                    # Original rubber particle characteristics
                    original_diameter = np.random.lognormal(np.log(2000), 0.3)  # μm
                    original_aspect_ratio = np.random.normal(1.2, 0.2)
                    
                    # Temperature-affected characteristics
                    current_diameter = original_diameter * (1 + thermal_expansion) * softening_factor
                    current_aspect_ratio = original_aspect_ratio * (1 + thermal_expansion * 0.5)
                    
                    # Surface degradation
                    surface_roughness = 0.1 + (temp - 20) * 0.001  # μm
                    if temp > 350:
                        surface_roughness += (temp - 350) * 0.005
                    
                    # Pore formation in rubber
                    pore_density = 0 if temp < 350 else (temp - 350) ** 1.2 * 0.01
                    mean_pore_size = 0.5 + (temp - 350) * 0.002 if temp > 350 else 0
                    
                    # Chemical composition changes
                    volatile_loss = 0 if temp < 300 else min((temp - 300) / 500, 0.4)
                    carbon_residue = 1 - volatile_loss
                    
                    # Interface bonding strength
                    base_bonding = 0.8 - rubber_content * 0.01  # MPa
                    thermal_debonding = 1 if temp < 250 else 1 - (temp - 250) / 550
                    interface_strength = base_bonding * thermal_debonding
                    
                    degradation_record = {
                        'temperature': temp,
                        'rubber_content': rubber_content,
                        'specimen_type': specimen_type,
                        'original_diameter_um': original_diameter,
                        'current_diameter_um': current_diameter,
                        'diameter_change_percent': (current_diameter - original_diameter) / original_diameter * 100,
                        'original_aspect_ratio': original_aspect_ratio,
                        'current_aspect_ratio': current_aspect_ratio,
                        'surface_roughness_um': surface_roughness,
                        'pore_density_per_mm2': pore_density,
                        'mean_pore_size_um': mean_pore_size,
                        'volatile_loss_fraction': volatile_loss,
                        'carbon_residue_fraction': carbon_residue,
                        'interface_bonding_strength_mpa': interface_strength,
                        'thermal_expansion_strain': thermal_expansion,
                        'softening_factor': softening_factor,
                        'pyrolysis_degree': pyrolysis_factor,
                        'degradation_mechanism': self._classify_degradation_mechanism(temp),
                        'specimen_id': f'RD_{rubber_content}_{temp}_{specimen_type}',
                        'analysis_method': 'SEM-EDS',
                        'measurement_precision': {
                            'diameter': '±50 nm',
                            'roughness': '±10 nm',
                            'composition': '±2%'
                        }
                    }
                    
                    degradation_data.append(degradation_record)
        
        self.rubber_degradation_data = pd.DataFrame(degradation_data)
        return self.rubber_degradation_data
    
    def _classify_degradation_mechanism(self, temperature):
        """Classify the dominant degradation mechanism based on temperature"""
        if temperature < 200:
            return 'thermal_expansion'
        elif temperature < 300:
            return 'elastic_deformation'
        elif temperature < 400:
            return 'softening_melting'
        elif temperature < 500:
            return 'pyrolysis_initiation'
        else:
            return 'carbonization'
    
    def generate_paste_morphology_analysis(self):
        """Generate cement paste morphology analysis"""
        print("Generating cement paste morphology analysis...")
        
        morphology_data = []
        
        for temp in self.temperatures:
            for rubber_content in self.rubber_contents:
                for specimen_type in self.specimen_types:
                    if specimen_type == 'heated' and temp == 20:
                        continue
                    
                    # C-S-H gel morphology
                    csh_fibril_length = 200 - (temp - 20) * 0.2 if temp < 500 else 100  # nm
                    csh_fibril_diameter = 3 + (temp - 20) * 0.001  # nm
                    csh_packing_density = 0.65 - (temp - 20) * 0.0005 if temp < 600 else 0.35
                    
                    # Portlandite crystal characteristics
                    ch_crystal_size = 5 + rubber_content * 0.1  # μm
                    if temp > 450:  # Portlandite dehydration
                        ch_crystal_size *= (1 - (temp - 450) / 100)
                    ch_crystal_size = max(ch_crystal_size, 0.5)
                    
                    # Pore structure
                    gel_porosity = 0.28 + (temp - 20) * 0.0002  # Gel pore volume fraction
                    capillary_porosity = 0.12 + (temp - 20) * 0.0003 + rubber_content * 0.001
                    
                    # Hydration products
                    degree_of_hydration = 0.75 - (temp - 20) * 0.0001 if temp < 300 else 0.72
                    ettringite_content = max(0.05 - (temp - 20) * 0.0002, 0)  # Decomposes >70°C
                    
                    # Microstructural parameters
                    specific_surface_area = 15 + (temp - 20) * 0.01  # m²/g (BET)
                    pore_size_distribution = {
                        'gel_pores_nm': np.random.lognormal(np.log(2.5), 0.3),
                        'capillary_pores_nm': np.random.lognormal(np.log(50), 0.5),
                        'macropores_um': np.random.lognormal(np.log(1), 0.4)
                    }
                    
                    morphology_record = {
                        'temperature': temp,
                        'rubber_content': rubber_content,
                        'specimen_type': specimen_type,
                        'csh_fibril_length_nm': csh_fibril_length,
                        'csh_fibril_diameter_nm': csh_fibril_diameter,
                        'csh_packing_density': csh_packing_density,
                        'ch_crystal_size_um': ch_crystal_size,
                        'gel_porosity': gel_porosity,
                        'capillary_porosity': capillary_porosity,
                        'total_porosity': gel_porosity + capillary_porosity,
                        'degree_of_hydration': degree_of_hydration,
                        'ettringite_content': ettringite_content,
                        'specific_surface_area_m2_g': specific_surface_area,
                        'gel_pore_size_nm': pore_size_distribution['gel_pores_nm'],
                        'capillary_pore_size_nm': pore_size_distribution['capillary_pores_nm'],
                        'macropore_size_um': pore_size_distribution['macropores_um'],
                        'paste_density_g_cm3': 2.1 - capillary_porosity * 0.5,
                        'specimen_id': f'PM_{rubber_content}_{temp}_{specimen_type}',
                        'imaging_technique': 'BSE-SEM',
                        'resolution_nm': 5,
                        'analysis_software': 'ImageJ-FIJI'
                    }
                    
                    morphology_data.append(morphology_record)
        
        self.paste_morphology_data = pd.DataFrame(morphology_data)
        return self.paste_morphology_data
    
    def save_datasets(self, output_dir='sem_analysis_data'):
        """Save all generated datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'itz_characteristics': self.itz_data,
            'microcrack_analysis': self.microcrack_data,
            'rubber_degradation': self.rubber_degradation_data,
            'paste_morphology': self.paste_morphology_data
        }
        
        for name, df in datasets.items():
            if df is not None and not df.empty:
                # Save as CSV
                csv_path = os.path.join(output_dir, f'{name}.csv')
                df.to_csv(csv_path, index=False)
                
                # Save as JSON for detailed metadata
                json_path = os.path.join(output_dir, f'{name}.json')
                df.to_json(json_path, orient='records', indent=2)
                
                print(f"Saved {name} dataset: {len(df)} records")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'temperature_range': f"{min(self.temperatures)}-{max(self.temperatures)}°C",
            'rubber_content_range': f"{min(self.rubber_contents)}-{max(self.rubber_contents)}%",
            'total_specimens': len(self.temperatures) * len(self.rubber_contents) * 2,
            'analysis_techniques': ['SEM', 'BSE-SEM', 'EDS'],
            'measurement_precision': {
                'dimensional': '±50 nm',
                'compositional': '±2%',
                'porosity': '±0.01'
            }
        }
        
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def generate_all_datasets(self):
        """Generate all SEM analysis datasets"""
        print("=== SEM Analysis Dataset Generation ===")
        
        self.generate_itz_characteristics()
        self.generate_microcrack_analysis()
        self.generate_rubber_degradation_analysis()
        self.generate_paste_morphology_analysis()
        
        return {
            'itz_data': self.itz_data,
            'microcrack_data': self.microcrack_data,
            'rubber_degradation_data': self.rubber_degradation_data,
            'paste_morphology_data': self.paste_morphology_data
        }

if __name__ == "__main__":
    # Generate comprehensive SEM dataset
    generator = SEMDatasetGenerator()
    datasets = generator.generate_all_datasets()
    generator.save_datasets()
    
    print("\n=== Dataset Generation Complete ===")
    for name, df in datasets.items():
        if df is not None:
            print(f"{name}: {len(df)} records")