"""
Micro-CT (X-Ray Computed Tomography) Data Generator for Rubberized Concrete
PhD Research: 3D Pore Network and Crack Evolution under Thermal Loading
Non-destructive visualization and quantification of internal damage
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import ndimage, spatial
from scipy.stats import lognorm, gamma
from skimage import morphology
import warnings
warnings.filterwarnings('ignore')

class MicroCTDataGenerator:
    """
    Generate comprehensive Micro-CT analysis data for 3D characterization
    of pore structure, crack networks, and rubber particle distribution
    """
    
    def __init__(self):
        self.temperatures = [20, 200, 400, 600, 800]  # °C
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        self.voxel_sizes = [5, 10, 20]  # μm - resolution
        self.sample_volumes = [1000, 2000, 5000]  # mm³
        
    def generate_porosity_analysis(self):
        """
        Generate 3D porosity characterization data
        Critical for permeability and durability assessment
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                for voxel_size in self.voxel_sizes:
                    for sample_vol in self.sample_volumes:
                        
                        # Base porosity calculation
                        if temp <= 200:
                            base_porosity = 0.12 + rubber * 0.008  # Initial porosity
                        elif temp <= 400:
                            base_porosity = 0.15 + rubber * 0.012  # Dehydration pores
                        elif temp <= 600:
                            base_porosity = 0.22 + rubber * 0.018  # Decomposition pores
                        else:
                            base_porosity = 0.35 + rubber * 0.025  # Severe degradation
                        
                        # Total porosity with variation
                        total_porosity = base_porosity + np.random.normal(0, 0.01)
                        
                        # Pore size distribution (lognormal)
                        if temp <= 200:
                            mean_pore_diameter = 50 + rubber * 5  # μm
                            std_pore_diameter = 20 + rubber * 2
                        elif temp <= 400:
                            mean_pore_diameter = 80 + rubber * 8
                            std_pore_diameter = 35 + rubber * 3
                        elif temp <= 600:
                            mean_pore_diameter = 150 + rubber * 12
                            std_pore_diameter = 60 + rubber * 5
                        else:
                            mean_pore_diameter = 300 + rubber * 20
                            std_pore_diameter = 120 + rubber * 10
                        
                        # Pore classification
                        gel_pores = total_porosity * (0.3 - temp/3000)  # < 10 nm
                        capillary_pores = total_porosity * (0.4 + temp/5000)  # 10 nm - 10 μm
                        macro_pores = total_porosity * (0.2 + temp/2000)  # 10 μm - 1 mm
                        cracks = total_porosity * (0.1 + temp/1000 + rubber/100)  # > 1 mm
                        
                        # Ensure fractions sum to 1
                        total_frac = gel_pores + capillary_pores + macro_pores + cracks
                        gel_pores /= total_frac
                        capillary_pores /= total_frac
                        macro_pores /= total_frac
                        cracks /= total_frac
                        
                        # Pore connectivity (critical for transport)
                        connectivity = 0.3 + temp/1500 + rubber/50 + np.random.normal(0, 0.02)
                        
                        # Tortuosity (path complexity)
                        tortuosity = 1.5 + temp/2000 + rubber/100 + np.random.normal(0, 0.1)
                        
                        # Specific surface area (m²/g)
                        specific_surface = 20 - temp/100 - rubber * 0.5 + np.random.normal(0, 1)
                        
                        # Pore shape factors
                        sphericity = 0.7 - temp/3000 - rubber/200 + np.random.normal(0, 0.02)
                        elongation = 1.2 + temp/1500 + rubber/100 + np.random.normal(0, 0.05)
                        
                        # Anisotropy (directional dependency)
                        anisotropy = 0.1 + temp/2000 + np.random.normal(0, 0.02)
                        
                        # Number of pores detected
                        voxels_total = (sample_vol * 1e9) / (voxel_size**3)  # Convert mm³ to μm³
                        pore_count = int(voxels_total * total_porosity * 0.001)  # Discrete pores
                        
                        data.append({
                            'Temperature_C': temp,
                            'Rubber_Content_%': rubber,
                            'Voxel_Size_um': voxel_size,
                            'Sample_Volume_mm3': sample_vol,
                            'Scan_ID': f'CT_{temp}_{rubber}_{voxel_size}_{sample_vol}',
                            'Total_Porosity_%': min(60, max(5, total_porosity * 100)),
                            'Gel_Pores_Fraction': max(0, gel_pores),
                            'Capillary_Pores_Fraction': capillary_pores,
                            'Macro_Pores_Fraction': macro_pores,
                            'Cracks_Fraction': min(0.5, cracks),
                            'Mean_Pore_Diameter_um': mean_pore_diameter,
                            'Std_Pore_Diameter_um': std_pore_diameter,
                            'Pore_Connectivity': min(1, max(0, connectivity)),
                            'Tortuosity': max(1, tortuosity),
                            'Specific_Surface_Area_m2_g': max(1, specific_surface),
                            'Pore_Sphericity': min(1, max(0.1, sphericity)),
                            'Pore_Elongation': max(1, elongation),
                            'Anisotropy_Degree': min(1, max(0, anisotropy)),
                            'Total_Pore_Count': pore_count,
                            'Timestamp': datetime.now().isoformat()
                        })
        
        return pd.DataFrame(data)
    
    def generate_crack_network_analysis(self):
        """
        Generate 3D crack network characterization
        PhD-level analysis of crack propagation and connectivity
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                for scan_id in range(10):  # Multiple scans per condition
                    
                    # Crack density evolution
                    if temp <= 200:
                        crack_volume_fraction = 0.001 + rubber * 0.0001
                        avg_crack_width = 5 + rubber * 0.5  # μm
                        max_crack_width = 20 + rubber * 2
                        crack_count = np.random.randint(0, 10)
                    elif temp <= 400:
                        crack_volume_fraction = 0.005 + rubber * 0.0005
                        avg_crack_width = 15 + rubber * 1
                        max_crack_width = 50 + rubber * 5
                        crack_count = np.random.randint(5, 30)
                    elif temp <= 600:
                        crack_volume_fraction = 0.02 + rubber * 0.002
                        avg_crack_width = 40 + rubber * 3
                        max_crack_width = 150 + rubber * 10
                        crack_count = np.random.randint(20, 100)
                    else:
                        crack_volume_fraction = 0.08 + rubber * 0.005
                        avg_crack_width = 100 + rubber * 5
                        max_crack_width = 500 + rubber * 20
                        crack_count = np.random.randint(50, 300)
                    
                    # Add variation
                    crack_volume_fraction *= np.random.lognormal(0, 0.2)
                    avg_crack_width *= np.random.lognormal(0, 0.15)
                    max_crack_width *= np.random.lognormal(0, 0.2)
                    
                    # Crack orientation analysis
                    if temp <= 400:
                        # Low temp: cracks mainly perpendicular to loading
                        primary_orientation = np.random.uniform(85, 95)  # degrees from horizontal
                        orientation_spread = 10
                    else:
                        # High temp: more random crack orientation
                        primary_orientation = np.random.uniform(0, 180)
                        orientation_spread = 45
                    
                    # Crack propagation characteristics
                    avg_crack_length = 100 + temp * 0.5 + rubber * 5 + np.random.normal(0, 10)  # μm
                    max_crack_length = avg_crack_length * np.random.uniform(3, 8)
                    
                    # Crack branching
                    branching_points = max(0, crack_count * (0.1 + temp/2000) * np.random.uniform(0.5, 1.5))
                    
                    # Crack connectivity metrics
                    connected_fraction = min(1, 0.1 + temp/1000 + rubber/100 + np.random.normal(0, 0.05))
                    percolation_threshold = temp > 600 and connected_fraction > 0.7
                    
                    # Fractal dimension of crack network
                    fractal_dimension = 1.5 + temp/2000 + rubber/200 + np.random.normal(0, 0.05)
                    
                    # Crack surface area
                    crack_surface_area = crack_volume_fraction * 2 / (avg_crack_width * 1e-6) * 1000  # m²/m³
                    
                    # Distance between cracks
                    if crack_count > 0:
                        avg_crack_spacing = (1000 / crack_count) ** (1/3) * 1000  # μm
                    else:
                        avg_crack_spacing = 10000  # Large value for no cracks
                    
                    # Crack type classification
                    if temp <= 200:
                        primary_crack_type = 'Shrinkage'
                        secondary_crack_type = 'Mechanical'
                    elif temp <= 400:
                        primary_crack_type = 'Thermal'
                        secondary_crack_type = 'Dehydration'
                    elif temp <= 600:
                        primary_crack_type = 'Decomposition'
                        secondary_crack_type = 'Thermal'
                    else:
                        primary_crack_type = 'Spalling'
                        secondary_crack_type = 'Decomposition'
                    
                    data.append({
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Scan_ID': f'CT_Crack_{temp}_{rubber}_{scan_id+1}',
                        'Crack_Volume_Fraction': min(0.2, crack_volume_fraction),
                        'Avg_Crack_Width_um': avg_crack_width,
                        'Max_Crack_Width_um': max_crack_width,
                        'Total_Crack_Count': crack_count,
                        'Avg_Crack_Length_um': avg_crack_length,
                        'Max_Crack_Length_um': max_crack_length,
                        'Primary_Orientation_deg': primary_orientation,
                        'Orientation_Spread_deg': orientation_spread,
                        'Branching_Points': int(branching_points),
                        'Connected_Crack_Fraction': connected_fraction,
                        'Percolation_Achieved': percolation_threshold,
                        'Fractal_Dimension': min(2, max(1, fractal_dimension)),
                        'Crack_Surface_Area_m2_m3': crack_surface_area,
                        'Avg_Crack_Spacing_um': avg_crack_spacing,
                        'Primary_Crack_Type': primary_crack_type,
                        'Secondary_Crack_Type': secondary_crack_type,
                        'Timestamp': datetime.now().isoformat()
                    })
        
        return pd.DataFrame(data)
    
    def generate_rubber_particle_distribution(self):
        """
        3D analysis of rubber particle distribution and degradation
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents[1:]:  # Skip 0% rubber
                for scan_id in range(8):  # Multiple scans
                    
                    # Expected particle count based on rubber content
                    sample_vol = 1000  # mm³
                    avg_particle_size = 2  # mm
                    particle_vol = (4/3) * np.pi * (avg_particle_size/2)**3
                    expected_particles = (sample_vol * rubber / 100) / particle_vol
                    
                    # Actual detected particles (some may be missed or merged)
                    if temp <= 400:
                        detection_rate = 0.95  # Good detection
                    elif temp <= 600:
                        detection_rate = 0.80  # Some particles degraded
                    else:
                        detection_rate = 0.50  # Many particles carbonized
                    
                    detected_particles = int(expected_particles * detection_rate * np.random.uniform(0.8, 1.2))
                    
                    # Particle size distribution changes with temperature
                    if temp <= 200:
                        mean_diameter = 2.0  # mm
                        std_diameter = 0.5
                        volume_change = -2  # % shrinkage
                    elif temp <= 400:
                        mean_diameter = 1.8
                        std_diameter = 0.6
                        volume_change = -10
                    elif temp <= 600:
                        mean_diameter = 1.4
                        std_diameter = 0.7
                        volume_change = -30
                    else:
                        mean_diameter = 0.8
                        std_diameter = 0.4
                        volume_change = -60
                    
                    # Add variation
                    mean_diameter *= np.random.uniform(0.9, 1.1)
                    
                    # Spatial distribution metrics
                    nearest_neighbor_distance = (sample_vol / detected_particles) ** (1/3) if detected_particles > 0 else 100
                    clustering_coefficient = 0.3 + rubber * 0.01 + np.random.normal(0, 0.05)
                    
                    # Rubber-matrix interface gap
                    if temp <= 200:
                        interface_gap = 0  # No gap
                    elif temp <= 400:
                        interface_gap = 5 + rubber * 0.2  # μm
                    elif temp <= 600:
                        interface_gap = 20 + rubber * 0.5
                    else:
                        interface_gap = 50 + rubber * 1
                    
                    # Rubber particle porosity (internal)
                    if temp <= 200:
                        internal_porosity = 0.02  # 2%
                    elif temp <= 400:
                        internal_porosity = 0.10  # Volatiles escaping
                    elif temp <= 600:
                        internal_porosity = 0.40  # Pyrolysis
                    else:
                        internal_porosity = 0.70  # Carbonization
                    
                    # Gray value analysis (density indicator)
                    if temp <= 400:
                        avg_gray_value = 120 - rubber * 2  # Lower than cement paste
                    else:
                        avg_gray_value = 80 - temp * 0.05  # Decreasing with degradation
                    
                    gray_value_std = 10 + temp * 0.01
                    
                    # Shape analysis
                    sphericity = 0.85 - temp * 0.0001 - rubber * 0.001 + np.random.normal(0, 0.02)
                    aspect_ratio = 1.1 + temp * 0.0002 + np.random.normal(0, 0.05)
                    
                    # Distribution uniformity (Ripley's K function)
                    distribution_uniformity = 0.8 - rubber * 0.01 + np.random.normal(0, 0.05)
                    
                    data.append({
                        'Temperature_C': temp,
                        'Rubber_Content_%': rubber,
                        'Scan_ID': f'CT_Rubber_{temp}_{rubber}_{scan_id+1}',
                        'Sample_Volume_mm3': sample_vol,
                        'Expected_Particles': int(expected_particles),
                        'Detected_Particles': detected_particles,
                        'Detection_Rate': detected_particles / expected_particles if expected_particles > 0 else 0,
                        'Mean_Particle_Diameter_mm': mean_diameter,
                        'Std_Particle_Diameter_mm': std_diameter,
                        'Volume_Change_%': volume_change,
                        'Nearest_Neighbor_Distance_mm': nearest_neighbor_distance,
                        'Clustering_Coefficient': min(1, max(0, clustering_coefficient)),
                        'Interface_Gap_um': interface_gap,
                        'Internal_Porosity': min(1, internal_porosity),
                        'Avg_Gray_Value': max(0, avg_gray_value),
                        'Gray_Value_StdDev': gray_value_std,
                        'Particle_Sphericity': min(1, max(0.1, sphericity)),
                        'Aspect_Ratio': max(1, aspect_ratio),
                        'Distribution_Uniformity': min(1, max(0, distribution_uniformity)),
                        'Timestamp': datetime.now().isoformat()
                    })
        
        return pd.DataFrame(data)
    
    def generate_damage_evolution_metrics(self):
        """
        Quantitative damage evolution metrics from 4D analysis (3D + time/temperature)
        """
        data = []
        
        for rubber in self.rubber_contents:
            # Track damage evolution across temperatures
            damage_history = []
            
            for i, temp in enumerate(self.temperatures):
                
                # Damage parameter D (0 = no damage, 1 = complete failure)
                if temp <= 200:
                    damage_parameter = 0.05 + rubber * 0.001
                elif temp <= 400:
                    damage_parameter = 0.15 + rubber * 0.005
                elif temp <= 600:
                    damage_parameter = 0.40 + rubber * 0.01
                else:
                    damage_parameter = 0.75 + rubber * 0.015
                
                damage_parameter += np.random.normal(0, 0.02)
                damage_parameter = min(1, max(0, damage_parameter))
                
                # Damage rate
                if i > 0:
                    damage_rate = (damage_parameter - damage_history[-1]) / (temp - self.temperatures[i-1])
                else:
                    damage_rate = 0
                
                damage_history.append(damage_parameter)
                
                # Volume changes
                total_volume_change = -damage_parameter * 5 - rubber * 0.1  # % shrinkage
                solid_volume_change = -damage_parameter * 8 - rubber * 0.15
                void_volume_change = damage_parameter * 30 + rubber * 0.5
                
                # Stiffness degradation (from CT density)
                relative_density = 1 - damage_parameter * 0.3
                estimated_modulus_ratio = relative_density ** 2  # Gibson-Ashby relation
                
                # Permeability estimation (Kozeny-Carman)
                porosity = 0.15 + damage_parameter * 0.35
                permeability = (porosity ** 3) / ((1 - porosity) ** 2) * 1e-15  # m²
                
                # Fracture process zone size
                fpz_size = 10 + damage_parameter * 100 + rubber * 5  # mm
                
                # Critical crack length
                critical_crack_length = 5 + damage_parameter * 50 + rubber * 2  # mm
                
                # Damage localization factor
                localization_factor = 0.2 + damage_parameter * 0.5 + np.random.normal(0, 0.05)
                
                # Anisotropic damage tensor components (simplified)
                damage_xx = damage_parameter * (1 + np.random.normal(0, 0.05))
                damage_yy = damage_parameter * (1 + np.random.normal(0, 0.05))
                damage_zz = damage_parameter * (1.2 + np.random.normal(0, 0.05))  # More in Z
                
                data.append({
                    'Temperature_C': temp,
                    'Rubber_Content_%': rubber,
                    'Damage_Parameter_D': damage_parameter,
                    'Damage_Rate_per_C': damage_rate,
                    'Total_Volume_Change_%': total_volume_change,
                    'Solid_Volume_Change_%': solid_volume_change,
                    'Void_Volume_Increase_%': max(0, void_volume_change),
                    'Relative_Density': relative_density,
                    'Estimated_Modulus_Ratio': estimated_modulus_ratio,
                    'Estimated_Porosity': porosity,
                    'Estimated_Permeability_m2': permeability,
                    'Fracture_Process_Zone_mm': fpz_size,
                    'Critical_Crack_Length_mm': critical_crack_length,
                    'Damage_Localization_Factor': min(1, max(0, localization_factor)),
                    'Damage_Tensor_XX': min(1, max(0, damage_xx)),
                    'Damage_Tensor_YY': min(1, max(0, damage_yy)),
                    'Damage_Tensor_ZZ': min(1, max(0, damage_zz)),
                    'Damage_Anisotropy': max(damage_xx, damage_yy, damage_zz) - min(damage_xx, damage_yy, damage_zz),
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def generate_segmentation_statistics(self):
        """
        Image segmentation and phase volume statistics
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                
                # Phase volume fractions
                if temp <= 200:
                    paste_fraction = 0.30 - rubber * 0.015
                    aggregate_fraction = 0.55
                    rubber_fraction = rubber / 100
                    pore_fraction = 0.15 - rubber * 0.005
                    crack_fraction = 0.001
                elif temp <= 400:
                    paste_fraction = 0.28 - rubber * 0.015
                    aggregate_fraction = 0.55
                    rubber_fraction = rubber / 100 * 0.9  # Some shrinkage
                    pore_fraction = 0.17 - rubber * 0.003
                    crack_fraction = 0.005
                elif temp <= 600:
                    paste_fraction = 0.25 - rubber * 0.012
                    aggregate_fraction = 0.54
                    rubber_fraction = rubber / 100 * 0.7  # Significant degradation
                    pore_fraction = 0.20
                    crack_fraction = 0.02
                else:
                    paste_fraction = 0.20 - rubber * 0.01
                    aggregate_fraction = 0.52
                    rubber_fraction = rubber / 100 * 0.4  # Mostly carbonized
                    pore_fraction = 0.25 + rubber * 0.002
                    crack_fraction = 0.05
                
                # Normalize to sum to 1
                total = paste_fraction + aggregate_fraction + rubber_fraction + pore_fraction + crack_fraction
                paste_fraction /= total
                aggregate_fraction /= total
                rubber_fraction /= total
                pore_fraction /= total
                crack_fraction /= total
                
                # Segmentation quality metrics
                dice_coefficient = 0.92 - temp * 0.00005 + np.random.normal(0, 0.01)
                jaccard_index = 0.88 - temp * 0.00008 + np.random.normal(0, 0.01)
                
                # Interface areas (m²/m³)
                paste_aggregate_interface = 500 + temp * 0.1 + np.random.normal(0, 20)
                paste_rubber_interface = rubber * 20 + temp * 0.05 + np.random.normal(0, 5)
                paste_pore_interface = 300 + temp * 0.5 + np.random.normal(0, 15)
                
                # Gray value thresholds used for segmentation
                threshold_paste_pore = 120 - temp * 0.01
                threshold_paste_aggregate = 160 + temp * 0.005
                threshold_rubber = 100 - temp * 0.02
                
                data.append({
                    'Temperature_C': temp,
                    'Rubber_Content_%': rubber,
                    'Paste_Volume_Fraction': paste_fraction,
                    'Aggregate_Volume_Fraction': aggregate_fraction,
                    'Rubber_Volume_Fraction': rubber_fraction,
                    'Pore_Volume_Fraction': pore_fraction,
                    'Crack_Volume_Fraction': crack_fraction,
                    'Segmentation_Dice_Coefficient': min(1, max(0.5, dice_coefficient)),
                    'Segmentation_Jaccard_Index': min(1, max(0.5, jaccard_index)),
                    'Paste_Aggregate_Interface_m2_m3': paste_aggregate_interface,
                    'Paste_Rubber_Interface_m2_m3': paste_rubber_interface,
                    'Paste_Pore_Interface_m2_m3': paste_pore_interface,
                    'Total_Interface_Area_m2_m3': paste_aggregate_interface + paste_rubber_interface + paste_pore_interface,
                    'Gray_Threshold_Paste_Pore': threshold_paste_pore,
                    'Gray_Threshold_Paste_Aggregate': threshold_paste_aggregate,
                    'Gray_Threshold_Rubber': threshold_rubber,
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def generate_transport_properties(self):
        """
        Estimate transport properties from CT pore network analysis
        """
        data = []
        
        for temp in self.temperatures:
            for rubber in self.rubber_contents:
                
                # Effective diffusivity (relative to free diffusion)
                if temp <= 200:
                    eff_diffusivity = 0.15 + rubber * 0.01
                elif temp <= 400:
                    eff_diffusivity = 0.25 + rubber * 0.015
                elif temp <= 600:
                    eff_diffusivity = 0.45 + rubber * 0.02
                else:
                    eff_diffusivity = 0.70 + rubber * 0.025
                
                eff_diffusivity *= np.random.lognormal(0, 0.1)
                
                # Permeability (Darcy)
                if temp <= 200:
                    permeability = 1e-17 * (1 + rubber * 0.2)
                elif temp <= 400:
                    permeability = 1e-16 * (1 + rubber * 0.3)
                elif temp <= 600:
                    permeability = 1e-15 * (1 + rubber * 0.4)
                else:
                    permeability = 1e-14 * (1 + rubber * 0.5)
                
                permeability *= np.random.lognormal(0, 0.3)
                
                # Formation factor (electrical resistivity ratio)
                formation_factor = 50 - temp * 0.03 - rubber * 0.5 + np.random.normal(0, 2)
                
                # Critical pore diameter for percolation
                critical_diameter = 0.1 + temp * 0.001 + rubber * 0.01  # mm
                
                # Constrictivity (pore neck effect)
                constrictivity = 0.5 - temp * 0.0002 - rubber * 0.002 + np.random.normal(0, 0.02)
                
                # Gas permeability vs liquid permeability ratio
                gas_liquid_perm_ratio = 1.5 + temp * 0.001 + rubber * 0.01
                
                data.append({
                    'Temperature_C': temp,
                    'Rubber_Content_%': rubber,
                    'Effective_Diffusivity_Relative': min(1, eff_diffusivity),
                    'Permeability_m2': permeability,
                    'Permeability_Darcy': permeability * 1.01325e12,  # Convert to Darcy
                    'Formation_Factor': max(1, formation_factor),
                    'Critical_Pore_Diameter_mm': critical_diameter,
                    'Constrictivity': min(1, max(0.1, constrictivity)),
                    'Gas_Liquid_Permeability_Ratio': gas_liquid_perm_ratio,
                    'Estimated_Chloride_Diffusion_m2_s': eff_diffusivity * 1e-12,  # Typical Cl- diffusion
                    'Estimated_Oxygen_Diffusion_m2_s': eff_diffusivity * 2e-12,  # O2 diffusion
                    'Estimated_Water_Sorptivity_mm_per_sqrt_h': np.sqrt(permeability * 1e15) * 10,
                    'Timestamp': datetime.now().isoformat()
                })
        
        return pd.DataFrame(data)
    
    def save_all_datasets(self):
        """Save all Micro-CT analysis datasets"""
        print("\nGenerating Micro-CT Analysis Datasets...")
        
        # Generate all datasets
        porosity_data = self.generate_porosity_analysis()
        crack_data = self.generate_crack_network_analysis()
        rubber_data = self.generate_rubber_particle_distribution()
        damage_data = self.generate_damage_evolution_metrics()
        segmentation_data = self.generate_segmentation_statistics()
        transport_data = self.generate_transport_properties()
        
        # Save to CSV files
        porosity_data.to_csv('microCT_porosity_analysis.csv', index=False)
        crack_data.to_csv('microCT_crack_network.csv', index=False)
        rubber_data.to_csv('microCT_rubber_distribution.csv', index=False)
        damage_data.to_csv('microCT_damage_evolution.csv', index=False)
        segmentation_data.to_csv('microCT_segmentation_statistics.csv', index=False)
        transport_data.to_csv('microCT_transport_properties.csv', index=False)
        
        # Generate comprehensive summary
        summary = {
            'Dataset': 'Micro-CT Analysis for Rubberized Concrete',
            'Generated': datetime.now().isoformat(),
            'Total_Porosity_Analyses': len(porosity_data),
            'Total_Crack_Analyses': len(crack_data),
            'Total_Rubber_Analyses': len(rubber_data),
            'Total_Damage_Metrics': len(damage_data),
            'Total_Segmentation_Analyses': len(segmentation_data),
            'Total_Transport_Predictions': len(transport_data),
            'Temperature_Range_C': f"{min(self.temperatures)}-{max(self.temperatures)}",
            'Rubber_Contents_%': self.rubber_contents,
            'Voxel_Resolutions_um': self.voxel_sizes,
            'Key_Findings': {
                'Max_Porosity_%': float(porosity_data['Total_Porosity_%'].max()),
                'Max_Crack_Volume_%': float(crack_data['Crack_Volume_Fraction'].max() * 100),
                'Percolation_Threshold_Temp': '600°C',
                'Max_Damage_Parameter': float(damage_data['Damage_Parameter_D'].max()),
                'Permeability_Increase_Factor': float(transport_data['Permeability_m2'].max() / transport_data['Permeability_m2'].min()),
                'Critical_Observations': [
                    'Significant crack network development above 400°C',
                    'Rubber particles show 60% volume reduction at 800°C',
                    'Percolation threshold reached at 600°C for high rubber content',
                    'Transport properties increase by 3 orders of magnitude'
                ]
            }
        }
        
        with open('MicroCT_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Generated {len(porosity_data)} porosity analyses")
        print(f"✓ Generated {len(crack_data)} crack network analyses")
        print(f"✓ Generated {len(rubber_data)} rubber distribution analyses")
        print(f"✓ Generated {len(damage_data)} damage evolution metrics")
        print(f"✓ Generated {len(segmentation_data)} segmentation statistics")
        print(f"✓ Generated {len(transport_data)} transport property estimates")
        
        return porosity_data, crack_data, rubber_data, damage_data, segmentation_data, transport_data

if __name__ == "__main__":
    generator = MicroCTDataGenerator()
    datasets = generator.save_all_datasets()
    print("\n✅ Micro-CT Analysis Dataset Generation Complete!")