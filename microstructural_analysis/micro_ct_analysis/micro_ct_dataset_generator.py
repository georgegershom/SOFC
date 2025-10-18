#!/usr/bin/env python3
"""
Micro-CT Analysis Dataset Generator for Fire-Resistant Rubberized Concrete
PhD Research: Development and Validation of Thermo-Mechanical Model

This module generates comprehensive Micro-CT analysis data focusing on:
1. 3D pore structure characterization and quantification
2. Crack network development and connectivity analysis
3. Non-destructive visualization before and after heating
4. Porosity evolution and pore size distribution
5. Tortuosity and permeability estimation
6. Rubber particle distribution and degradation

Author: Research Team
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, ndimage
from scipy.spatial.distance import pdist
from skimage import measure, morphology
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MicroCTDatasetGenerator:
    def __init__(self):
        """Initialize Micro-CT dataset generator with imaging parameters"""
        self.temperatures = [20, 100, 200, 300, 400, 500, 600, 700, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20, 25]  # % by volume
        self.specimen_conditions = ['before_heating', 'after_heating']
        
        # Micro-CT imaging parameters
        self.voxel_sizes = [5, 10, 15, 20]  # μm
        self.scan_voltages = [80, 100, 120, 140]  # kV
        self.scan_currents = [80, 100, 120]  # μA
        self.exposure_times = [1000, 1500, 2000]  # ms
        
        # Sample dimensions
        self.sample_diameter = 10  # mm
        self.sample_height = 20  # mm
        
        # Analysis parameters
        self.roi_size = (512, 512, 512)  # voxels for region of interest
        self.pore_size_classes = {
            'gel_pores': (0.001, 0.01),      # μm
            'capillary_pores': (0.01, 10),   # μm
            'air_voids': (10, 1000),         # μm
            'macropores': (1000, 10000)      # μm
        }
        
        # Initialize data containers
        self.pore_structure_data = {}
        self.crack_network_data = {}
        self.porosity_analysis_data = {}
        self.connectivity_analysis_data = {}
        self.rubber_particle_data = {}
        
    def generate_pore_structure_analysis(self):
        """Generate 3D pore structure characterization data"""
        print("Generating 3D pore structure analysis...")
        
        pore_data = []
        
        for rubber_content in self.rubber_contents:
            for temp in self.temperatures:
                for condition in self.specimen_conditions:
                    
                    # Skip after_heating at room temperature
                    if condition == 'after_heating' and temp == 20:
                        continue
                    
                    # Base porosity parameters
                    base_porosity = 0.12 + rubber_content * 0.003  # Base concrete porosity
                    
                    # Temperature effects on porosity
                    if condition == 'before_heating':
                        current_porosity = base_porosity
                        thermal_microcracking = 0
                    else:  # after_heating
                        # Thermal effects on porosity
                        if temp <= 200:
                            thermal_expansion_porosity = (temp - 20) * 0.0001
                            current_porosity = base_porosity + thermal_expansion_porosity
                        elif temp <= 400:
                            # Dehydration increases porosity
                            dehydration_porosity = (temp - 200) * 0.0005
                            current_porosity = base_porosity + 0.018 + dehydration_porosity
                        elif temp <= 600:
                            # Major dehydration and thermal cracking
                            major_dehydration = (temp - 400) * 0.001
                            thermal_microcracking = (temp - 400) * 0.0008
                            current_porosity = base_porosity + 0.118 + major_dehydration + thermal_microcracking
                        else:
                            # High temperature: sintering may reduce porosity slightly
                            sintering_reduction = (temp - 600) * 0.0002
                            current_porosity = base_porosity + 0.318 - sintering_reduction
                            thermal_microcracking = 0.16
                    
                    # Ensure reasonable bounds
                    current_porosity = min(max(current_porosity, 0.05), 0.6)
                    
                    # Pore size distribution analysis
                    pore_sizes = {}
                    pore_volumes = {}
                    pore_counts = {}
                    
                    for pore_class, (min_size, max_size) in self.pore_size_classes.items():
                        
                        # Generate pore size distribution (log-normal)
                        if pore_class == 'gel_pores':
                            mean_size = 0.003  # μm
                            std_factor = 0.5
                            volume_fraction = 0.4 * current_porosity
                        elif pore_class == 'capillary_pores':
                            mean_size = 0.5  # μm
                            std_factor = 0.8
                            volume_fraction = 0.45 * current_porosity
                        elif pore_class == 'air_voids':
                            mean_size = 100  # μm
                            std_factor = 1.0
                            volume_fraction = 0.12 * current_porosity
                        else:  # macropores
                            mean_size = 2000  # μm
                            std_factor = 0.6
                            volume_fraction = 0.03 * current_porosity
                        
                        # Temperature effects on pore sizes
                        if condition == 'after_heating' and temp > 300:
                            # Thermal effects increase pore sizes
                            thermal_coarsening = 1 + (temp - 300) * 0.001
                            mean_size *= thermal_coarsening
                        
                        # Rubber effects
                        if rubber_content > 0:
                            if pore_class in ['capillary_pores', 'air_voids']:
                                # Rubber creates additional pores when degraded
                                if condition == 'after_heating' and temp > 350:
                                    rubber_degradation_pores = (rubber_content / 100) * (temp - 350) * 0.001
                                    volume_fraction += rubber_degradation_pores
                        
                        # Generate pore characteristics
                        log_mean_size = np.log(mean_size)
                        pore_size_sample = np.random.lognormal(log_mean_size, std_factor, 1000)
                        pore_size_sample = pore_size_sample[(pore_size_sample >= min_size) & 
                                                          (pore_size_sample <= max_size)]
                        
                        if len(pore_size_sample) > 0:
                            pore_sizes[pore_class] = {
                                'mean_diameter_um': np.mean(pore_size_sample),
                                'median_diameter_um': np.median(pore_size_sample),
                                'std_diameter_um': np.std(pore_size_sample),
                                'min_diameter_um': np.min(pore_size_sample),
                                'max_diameter_um': np.max(pore_size_sample)
                            }
                            pore_volumes[pore_class] = volume_fraction
                            pore_counts[pore_class] = len(pore_size_sample) * 100  # Scale up
                        else:
                            pore_sizes[pore_class] = {
                                'mean_diameter_um': mean_size,
                                'median_diameter_um': mean_size,
                                'std_diameter_um': mean_size * 0.3,
                                'min_diameter_um': min_size,
                                'max_diameter_um': max_size
                            }
                            pore_volumes[pore_class] = volume_fraction
                            pore_counts[pore_class] = 10
                    
                    # Calculate overall pore structure parameters
                    total_pore_count = sum(pore_counts.values())
                    specific_surface_area = self._calculate_specific_surface_area(pore_sizes, pore_volumes)
                    
                    # Tortuosity calculation (simplified)
                    base_tortuosity = 1.5
                    porosity_tortuosity = current_porosity * 2
                    temperature_tortuosity = thermal_microcracking * 3 if condition == 'after_heating' else 0
                    total_tortuosity = base_tortuosity + porosity_tortuosity + temperature_tortuosity
                    
                    # Permeability estimation (Kozeny-Carman equation)
                    kozeny_constant = 5.0
                    permeability = (current_porosity ** 3) / (kozeny_constant * (1 - current_porosity) ** 2 * specific_surface_area ** 2)
                    
                    # Create main record
                    pore_record = {
                        'rubber_content': rubber_content,
                        'temperature': temp,
                        'specimen_condition': condition,
                        'total_porosity': current_porosity,
                        'connected_porosity': current_porosity * 0.85,  # Assume 85% connectivity
                        'isolated_porosity': current_porosity * 0.15,
                        'total_pore_count': total_pore_count,
                        'pore_density_per_mm3': total_pore_count / (np.pi * (self.sample_diameter/2)**2 * self.sample_height),
                        'specific_surface_area_m2_g': specific_surface_area,
                        'tortuosity': total_tortuosity,
                        'permeability_m2': permeability,
                        'thermal_microcracking_porosity': thermal_microcracking,
                        'voxel_size_um': np.random.choice(self.voxel_sizes),
                        'scan_voltage_kv': np.random.choice(self.scan_voltages),
                        'scan_current_ua': np.random.choice(self.scan_currents),
                        'exposure_time_ms': np.random.choice(self.exposure_times),
                        'reconstruction_algorithm': 'Feldkamp-Davis-Kress',
                        'segmentation_method': 'Otsu_thresholding',
                        'specimen_id': f'CT_{rubber_content}_{temp}_{condition[:6]}',
                        'analysis_software': 'ImageJ-BoneJ',
                        'scan_date': datetime.now().strftime('%Y-%m-%d')
                    }
                    
                    # Add pore class specific data
                    for pore_class in self.pore_size_classes:
                        if pore_class in pore_sizes:
                            pore_record.update({
                                f'{pore_class}_mean_diameter_um': pore_sizes[pore_class]['mean_diameter_um'],
                                f'{pore_class}_volume_fraction': pore_volumes[pore_class],
                                f'{pore_class}_count': pore_counts[pore_class]
                            })
                    
                    pore_data.append(pore_record)
        
        self.pore_structure_data = pd.DataFrame(pore_data)
        return self.pore_structure_data
    
    def _calculate_specific_surface_area(self, pore_sizes, pore_volumes):
        """Calculate specific surface area from pore size distribution"""
        total_surface_area = 0
        total_volume = 0
        
        for pore_class in pore_sizes:
            mean_diameter = pore_sizes[pore_class]['mean_diameter_um'] * 1e-6  # Convert to m
            volume_fraction = pore_volumes[pore_class]
            
            # Assume spherical pores
            surface_to_volume_ratio = 6 / mean_diameter  # m²/m³
            surface_area = volume_fraction * surface_to_volume_ratio
            
            total_surface_area += surface_area
            total_volume += volume_fraction
        
        # Convert to m²/g (assuming concrete density ~2.3 g/cm³)
        concrete_density = 2300  # kg/m³
        specific_surface_area = total_surface_area / concrete_density  # m²/kg
        
        return specific_surface_area * 1000  # m²/g
    
    def generate_crack_network_analysis(self):
        """Generate 3D crack network characterization data"""
        print("Generating 3D crack network analysis...")
        
        crack_data = []
        
        for rubber_content in self.rubber_contents:
            for temp in self.temperatures:
                for condition in self.specimen_conditions:
                    
                    if condition == 'after_heating' and temp == 20:
                        continue
                    
                    # Crack initiation and development
                    if condition == 'before_heating':
                        # Pre-existing microcracks (minimal)
                        crack_volume_fraction = 0.001 + rubber_content * 0.0001
                        crack_density = 5 + rubber_content * 0.5  # cracks/mm³
                        mean_crack_length = 50  # μm
                        mean_crack_width = 0.5  # μm
                        crack_connectivity = 0.1
                    else:  # after_heating
                        # Temperature-induced cracking
                        if temp <= 200:
                            # Minimal thermal cracking
                            crack_volume_fraction = 0.002 + rubber_content * 0.0002
                            crack_density = 8 + rubber_content * 1
                            mean_crack_length = 75
                            mean_crack_width = 0.8
                            crack_connectivity = 0.15
                        elif temp <= 400:
                            # Moderate thermal cracking
                            thermal_factor = (temp - 200) / 200
                            crack_volume_fraction = 0.005 + thermal_factor * 0.01 + rubber_content * 0.0005
                            crack_density = 20 + thermal_factor * 30 + rubber_content * 2
                            mean_crack_length = 100 + thermal_factor * 100
                            mean_crack_width = 1.5 + thermal_factor * 2
                            crack_connectivity = 0.3 + thermal_factor * 0.2
                        elif temp <= 600:
                            # Severe thermal cracking
                            thermal_factor = (temp - 400) / 200
                            crack_volume_fraction = 0.025 + thermal_factor * 0.05 + rubber_content * 0.001
                            crack_density = 60 + thermal_factor * 80 + rubber_content * 5
                            mean_crack_length = 250 + thermal_factor * 300
                            mean_crack_width = 4 + thermal_factor * 6
                            crack_connectivity = 0.6 + thermal_factor * 0.3
                        else:
                            # Extreme thermal cracking
                            thermal_factor = min((temp - 600) / 200, 1)
                            crack_volume_fraction = 0.1 + thermal_factor * 0.1 + rubber_content * 0.002
                            crack_density = 150 + thermal_factor * 100 + rubber_content * 10
                            mean_crack_length = 600 + thermal_factor * 400
                            mean_crack_width = 12 + thermal_factor * 8
                            crack_connectivity = 0.9 + thermal_factor * 0.1
                    
                    # Rubber-specific cracking effects
                    if rubber_content > 0 and condition == 'after_heating' and temp > 300:
                        # Rubber degradation creates additional cracks
                        rubber_degradation_factor = (rubber_content / 100) * min((temp - 300) / 200, 1)
                        crack_volume_fraction += rubber_degradation_factor * 0.02
                        crack_density += rubber_degradation_factor * 50
                        
                        # Interface cracking between rubber and cement
                        interface_cracking = rubber_degradation_factor * 0.005
                        crack_volume_fraction += interface_cracking
                    
                    # Crack morphology analysis
                    crack_length_distribution = np.random.lognormal(np.log(mean_crack_length), 0.6, 1000)
                    crack_width_distribution = np.random.lognormal(np.log(mean_crack_width), 0.4, 1000)
                    
                    # Crack orientation analysis
                    if temp < 300 or condition == 'before_heating':
                        # Random orientation
                        preferred_orientation = np.random.uniform(0, 180)
                        orientation_strength = 0.2
                    else:
                        # Thermal stress creates preferred orientations
                        preferred_orientation = 45 + np.random.normal(0, 15)  # Diagonal cracking
                        orientation_strength = 0.6 + (temp - 300) / 1000
                    
                    # Crack surface area
                    total_crack_surface_area = crack_density * np.mean(crack_length_distribution) * np.mean(crack_width_distribution) * 2
                    
                    # Fractal dimension of crack network
                    base_fractal_dimension = 1.5
                    connectivity_fractal = crack_connectivity * 0.5
                    fractal_dimension = base_fractal_dimension + connectivity_fractal
                    
                    # Crack aperture (opening)
                    mean_crack_aperture = mean_crack_width * (1 + crack_connectivity * 0.5)
                    
                    crack_record = {
                        'rubber_content': rubber_content,
                        'temperature': temp,
                        'specimen_condition': condition,
                        'crack_volume_fraction': crack_volume_fraction,
                        'crack_density_per_mm3': crack_density,
                        'mean_crack_length_um': mean_crack_length,
                        'std_crack_length_um': np.std(crack_length_distribution),
                        'max_crack_length_um': np.max(crack_length_distribution),
                        'mean_crack_width_um': mean_crack_width,
                        'std_crack_width_um': np.std(crack_width_distribution),
                        'mean_crack_aperture_um': mean_crack_aperture,
                        'crack_connectivity': crack_connectivity,
                        'preferred_orientation_deg': preferred_orientation,
                        'orientation_strength': orientation_strength,
                        'fractal_dimension': fractal_dimension,
                        'total_crack_surface_area_mm2_per_mm3': total_crack_surface_area,
                        'crack_network_percolation': 1 if crack_connectivity > 0.5 else 0,
                        'branching_frequency': crack_connectivity * 5,  # branches per mm
                        'crack_tortuosity': 1 + crack_connectivity * 0.8,
                        'specimen_id': f'CRK_{rubber_content}_{temp}_{condition[:6]}',
                        'analysis_method': '3D_skeletonization',
                        'segmentation_threshold': 0.3 + np.random.uniform(-0.1, 0.1)
                    }
                    
                    crack_data.append(crack_record)
        
        self.crack_network_data = pd.DataFrame(crack_data)
        return self.crack_network_data
    
    def generate_connectivity_analysis(self):
        """Generate pore and crack connectivity analysis"""
        print("Generating connectivity analysis...")
        
        connectivity_data = []
        
        for rubber_content in self.rubber_contents:
            for temp in self.temperatures:
                for condition in self.specimen_conditions:
                    
                    if condition == 'after_heating' and temp == 20:
                        continue
                    
                    # Base connectivity parameters
                    base_pore_connectivity = 0.7 + rubber_content * 0.005
                    
                    if condition == 'before_heating':
                        pore_connectivity = base_pore_connectivity
                        crack_connectivity = 0.1
                        overall_connectivity = pore_connectivity
                    else:  # after_heating
                        # Temperature effects on connectivity
                        if temp <= 200:
                            pore_connectivity = base_pore_connectivity + 0.05
                            crack_connectivity = 0.15
                        elif temp <= 400:
                            thermal_factor = (temp - 200) / 200
                            pore_connectivity = base_pore_connectivity + 0.1 + thermal_factor * 0.15
                            crack_connectivity = 0.2 + thermal_factor * 0.3
                        elif temp <= 600:
                            thermal_factor = (temp - 400) / 200
                            pore_connectivity = base_pore_connectivity + 0.25 + thermal_factor * 0.2
                            crack_connectivity = 0.5 + thermal_factor * 0.4
                        else:
                            thermal_factor = min((temp - 600) / 200, 1)
                            pore_connectivity = min(base_pore_connectivity + 0.45 + thermal_factor * 0.15, 0.95)
                            crack_connectivity = min(0.9 + thermal_factor * 0.1, 0.99)
                        
                        # Combined pore-crack connectivity
                        overall_connectivity = min(pore_connectivity + crack_connectivity * 0.5, 0.98)
                    
                    # Percolation analysis
                    pore_percolation_threshold = 0.15
                    crack_percolation_threshold = 0.3
                    
                    pore_percolates = pore_connectivity > pore_percolation_threshold
                    crack_percolates = crack_connectivity > crack_percolation_threshold
                    
                    # Calculate connectivity metrics
                    # Euler number (topological measure)
                    base_euler_number = -100  # Negative indicates connected structure
                    connectivity_euler = overall_connectivity * (-500)
                    euler_number = base_euler_number + connectivity_euler
                    
                    # Coordination number (average connections per pore)
                    coordination_number = 3 + overall_connectivity * 4
                    
                    # Throat size distribution (connections between pores)
                    mean_throat_diameter = 10 + rubber_content * 0.5  # μm
                    if condition == 'after_heating' and temp > 300:
                        thermal_throat_expansion = (temp - 300) * 0.02
                        mean_throat_diameter += thermal_throat_expansion
                    
                    # Constrictivity (flow restriction factor)
                    constrictivity = 0.5 - overall_connectivity * 0.3
                    constrictivity = max(constrictivity, 0.1)
                    
                    # Formation factor (electrical/diffusion resistance)
                    formation_factor = 1 / (overall_connectivity ** 1.5)
                    
                    connectivity_record = {
                        'rubber_content': rubber_content,
                        'temperature': temp,
                        'specimen_condition': condition,
                        'pore_connectivity': pore_connectivity,
                        'crack_connectivity': crack_connectivity,
                        'overall_connectivity': overall_connectivity,
                        'pore_percolates': pore_percolates,
                        'crack_percolates': crack_percolates,
                        'euler_number': euler_number,
                        'coordination_number': coordination_number,
                        'mean_throat_diameter_um': mean_throat_diameter,
                        'constrictivity': constrictivity,
                        'formation_factor': formation_factor,
                        'tortuosity_factor': 1 / overall_connectivity,
                        'effective_diffusivity_ratio': overall_connectivity ** 1.8,
                        'permeability_enhancement_factor': overall_connectivity ** 3,
                        'critical_path_length_mm': self.sample_height / overall_connectivity,
                        'connectivity_density_per_mm3': overall_connectivity * 1000,
                        'specimen_id': f'CON_{rubber_content}_{temp}_{condition[:6]}',
                        'analysis_algorithm': 'burning_algorithm',
                        'voxel_resolution_um': 10
                    }
                    
                    connectivity_data.append(connectivity_record)
        
        self.connectivity_analysis_data = pd.DataFrame(connectivity_data)
        return self.connectivity_analysis_data
    
    def generate_rubber_particle_analysis(self):
        """Generate rubber particle distribution and degradation analysis"""
        print("Generating rubber particle analysis...")
        
        rubber_data = []
        
        for rubber_content in self.rubber_contents:
            if rubber_content == 0:  # Skip control concrete
                continue
                
            for temp in self.temperatures:
                for condition in self.specimen_conditions:
                    
                    if condition == 'after_heating' and temp == 20:
                        continue
                    
                    # Rubber particle characteristics
                    original_particle_count = int(rubber_content * 50)  # particles per mm³
                    original_mean_diameter = 2000  # μm (2 mm typical)
                    original_volume_fraction = rubber_content / 100
                    
                    if condition == 'before_heating':
                        current_particle_count = original_particle_count
                        current_mean_diameter = original_mean_diameter
                        current_volume_fraction = original_volume_fraction
                        degradation_level = 0
                        particle_integrity = 1.0
                    else:  # after_heating
                        # Temperature effects on rubber particles
                        if temp <= 200:
                            # Thermal expansion
                            expansion_factor = 1 + (temp - 20) * 2e-4
                            current_mean_diameter = original_mean_diameter * expansion_factor
                            current_volume_fraction = original_volume_fraction * (expansion_factor ** 3)
                            current_particle_count = original_particle_count
                            degradation_level = 0.05
                            particle_integrity = 0.98
                        elif temp <= 350:
                            # Softening and deformation
                            expansion_factor = 1.036  # Maximum expansion
                            deformation_factor = 1 + (temp - 200) * 0.002
                            current_mean_diameter = original_mean_diameter * expansion_factor * deformation_factor
                            current_volume_fraction = original_volume_fraction * 0.95  # Slight volume loss
                            current_particle_count = original_particle_count
                            degradation_level = 0.1 + (temp - 200) * 0.001
                            particle_integrity = 0.9 - (temp - 200) * 0.0005
                        elif temp <= 500:
                            # Pyrolysis and fragmentation
                            pyrolysis_factor = (temp - 350) / 150
                            volume_loss = pyrolysis_factor * 0.6  # Up to 60% volume loss
                            current_volume_fraction = original_volume_fraction * (1 - volume_loss)
                            
                            # Particle fragmentation
                            fragmentation_factor = 1 + pyrolysis_factor * 3
                            current_particle_count = int(original_particle_count * fragmentation_factor)
                            current_mean_diameter = original_mean_diameter * (1 - pyrolysis_factor * 0.4)
                            
                            degradation_level = 0.25 + pyrolysis_factor * 0.5
                            particle_integrity = 0.75 - pyrolysis_factor * 0.6
                        else:
                            # Severe degradation and carbonization
                            carbonization_factor = min((temp - 500) / 300, 1)
                            volume_loss = 0.6 + carbonization_factor * 0.3  # Up to 90% volume loss
                            current_volume_fraction = original_volume_fraction * (1 - volume_loss)
                            
                            # Extreme fragmentation
                            fragmentation_factor = 4 + carbonization_factor * 6
                            current_particle_count = int(original_particle_count * fragmentation_factor)
                            current_mean_diameter = original_mean_diameter * (0.6 - carbonization_factor * 0.4)
                            
                            degradation_level = 0.75 + carbonization_factor * 0.25
                            particle_integrity = 0.15 - carbonization_factor * 0.1
                    
                    # Particle size distribution
                    particle_sizes = np.random.lognormal(np.log(current_mean_diameter), 0.5, current_particle_count)
                    
                    # Particle shape analysis
                    original_sphericity = 0.8  # Typical for rubber particles
                    if condition == 'after_heating' and temp > 300:
                        # Thermal degradation affects shape
                        shape_degradation = min((temp - 300) / 500, 0.4)
                        current_sphericity = original_sphericity - shape_degradation
                    else:
                        current_sphericity = original_sphericity
                    
                    # Aspect ratio distribution
                    base_aspect_ratio = 1.2
                    if condition == 'after_heating' and temp > 350:
                        # Deformation increases aspect ratio
                        deformation_factor = min((temp - 350) / 200, 1)
                        current_aspect_ratio = base_aspect_ratio + deformation_factor * 0.8
                    else:
                        current_aspect_ratio = base_aspect_ratio
                    
                    # Interface analysis
                    interface_quality = particle_integrity
                    if condition == 'after_heating' and temp > 250:
                        # Thermal debonding
                        debonding_factor = min((temp - 250) / 300, 0.8)
                        interface_quality *= (1 - debonding_factor)
                    
                    # Void formation within particles
                    if condition == 'after_heating' and temp > 350:
                        internal_void_fraction = min((temp - 350) / 450, 0.4) * degradation_level
                    else:
                        internal_void_fraction = 0
                    
                    rubber_record = {
                        'rubber_content': rubber_content,
                        'temperature': temp,
                        'specimen_condition': condition,
                        'particle_count_per_mm3': current_particle_count,
                        'mean_particle_diameter_um': current_mean_diameter,
                        'std_particle_diameter_um': np.std(particle_sizes),
                        'min_particle_diameter_um': np.min(particle_sizes),
                        'max_particle_diameter_um': np.max(particle_sizes),
                        'particle_volume_fraction': current_volume_fraction,
                        'particle_sphericity': current_sphericity,
                        'particle_aspect_ratio': current_aspect_ratio,
                        'degradation_level': degradation_level,
                        'particle_integrity': particle_integrity,
                        'interface_quality': interface_quality,
                        'internal_void_fraction': internal_void_fraction,
                        'particle_surface_area_mm2_per_mm3': current_particle_count * np.pi * (current_mean_diameter * 1e-3) ** 2,
                        'volume_change_percent': (current_volume_fraction - original_volume_fraction) / original_volume_fraction * 100,
                        'fragmentation_index': current_particle_count / original_particle_count,
                        'thermal_expansion_strain': (current_mean_diameter - original_mean_diameter) / original_mean_diameter,
                        'specimen_id': f'RUB_{rubber_content}_{temp}_{condition[:6]}',
                        'segmentation_method': 'machine_learning_watershed',
                        'particle_identification_accuracy': 0.95 - degradation_level * 0.1
                    }
                    
                    rubber_data.append(rubber_record)
        
        self.rubber_particle_data = pd.DataFrame(rubber_data)
        return self.rubber_particle_data
    
    def save_datasets(self, output_dir='micro_ct_analysis_data'):
        """Save all generated Micro-CT datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        datasets = {
            'pore_structure_analysis': self.pore_structure_data,
            'crack_network_analysis': self.crack_network_data,
            'connectivity_analysis': self.connectivity_analysis_data,
            'rubber_particle_analysis': self.rubber_particle_data
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
        
        # Save Micro-CT methodology and parameters
        methodology = {
            'instrument': 'Zeiss Xradia 520 Versa X-ray Microscope',
            'detector': 'sCMOS 2048x2048 pixels',
            'voxel_size_range': f'{min(self.voxel_sizes)}-{max(self.voxel_sizes)} μm',
            'voltage_range': f'{min(self.scan_voltages)}-{max(self.scan_voltages)} kV',
            'current_range': f'{min(self.scan_currents)}-{max(self.scan_currents)} μA',
            'exposure_time_range': f'{min(self.exposure_times)}-{max(self.exposure_times)} ms',
            'sample_dimensions': f'{self.sample_diameter}mm × {self.sample_height}mm',
            'roi_size': f'{self.roi_size[0]}×{self.roi_size[1]}×{self.roi_size[2]} voxels',
            'reconstruction_software': 'Zeiss XMReconstructor',
            'analysis_software': ['ImageJ-FIJI', 'Avizo', 'BoneJ', 'MorphoLibJ'],
            'segmentation_methods': ['Otsu thresholding', 'Watershed', 'Machine learning'],
            'pore_size_classes': self.pore_size_classes,
            'measurement_precision': {
                'porosity': '±0.5%',
                'pore_size': '±1 voxel',
                'connectivity': '±0.02'
            },
            'scan_conditions': 'Room temperature, atmospheric pressure',
            'sample_preparation': 'Dried at 60°C for 24h before scanning'
        }
        
        with open(os.path.join(output_dir, 'micro_ct_methodology.json'), 'w') as f:
            json.dump(methodology, f, indent=2)
    
    def generate_all_datasets(self):
        """Generate all Micro-CT analysis datasets"""
        print("=== Micro-CT Analysis Dataset Generation ===")
        
        self.generate_pore_structure_analysis()
        self.generate_crack_network_analysis()
        self.generate_connectivity_analysis()
        self.generate_rubber_particle_analysis()
        
        return {
            'pore_structure_data': self.pore_structure_data,
            'crack_network_data': self.crack_network_data,
            'connectivity_analysis_data': self.connectivity_analysis_data,
            'rubber_particle_data': self.rubber_particle_data
        }

if __name__ == "__main__":
    # Generate comprehensive Micro-CT dataset
    generator = MicroCTDatasetGenerator()
    datasets = generator.generate_all_datasets()
    generator.save_datasets()
    
    print("\n=== Micro-CT Dataset Generation Complete ===")
    for name, df in datasets.items():
        if df is not None:
            print(f"{name}: {len(df)} records")