#!/usr/bin/env python3
"""
Phase 3 - Microstructural and Chemical Analysis Dataset Generator
For: Development and Validation of Thermo-Mechanical Model for Fire-Resistant 
     Structural Elements Utilizing High-Performance Rubberized Concrete

This script generates comprehensive, internally consistent microstructural data
across multiple analytical techniques (SEM, XRD, TGA/DTA, Micro-CT).
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class Phase3DatasetGenerator:
    """Generate comprehensive microstructural analysis dataset"""
    
    def __init__(self):
        self.temperatures = [25, 200, 400, 600, 800]
        self.mix_ids = ['C-0', 'C-10', 'C-20', 'C-30']  # Control and rubber percentages
        self.replicates = 3
        self.output_dir = Path('/workspace/phase3_datasets')
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tracking for cross-technique consistency
        self.consistency_tracker = {}
        
    def generate_sample_ids(self):
        """Generate Phase 2 linked sample IDs"""
        sample_ids = []
        for mix in self.mix_ids:
            for temp in self.temperatures:
                for rep in range(1, self.replicates + 1):
                    sample_id = f"{mix}-28-R-{temp}-Furnace-Rep{rep}"
                    sample_ids.append({
                        'Sample_ID': sample_id,
                        'Mix_ID': mix,
                        'Temperature': temp,
                        'Replicate': rep,
                        'Rubber_Content': int(mix.split('-')[1]) if '-' in mix else 0
                    })
        return pd.DataFrame(sample_ids)
    
    def generate_sem_data(self):
        """Generate Scanning Electron Microscopy data with quantitative metrics"""
        print("Generating SEM dataset...")
        
        sem_records = []
        
        for mix in self.mix_ids:
            rubber_content = int(mix.split('-')[1]) if '-' in mix else 0
            
            for temp in self.temperatures:
                for rep in range(1, self.replicates + 1):
                    sample_id = f"{mix}-28-R-{temp}-Furnace-Rep{rep}"
                    
                    # Calculate temperature and rubber-dependent parameters
                    temp_factor = temp / 800.0
                    rubber_factor = rubber_content / 30.0
                    
                    # Generate multiple fields of view (FOV)
                    for fov in range(1, 6):  # 5 fields of view per sample
                        
                        # Pore characteristics (rubber degradation creates pores)
                        base_porosity = 5.0 + rubber_content * 0.3
                        thermal_porosity_increase = temp_factor * (8.0 + rubber_content * 0.5)
                        total_porosity = base_porosity + thermal_porosity_increase + np.random.normal(0, 0.5)
                        
                        # Pore size distribution (rubber creates larger pores)
                        pore_mean_diameter = (20 + rubber_content * 2) * (1 + temp_factor * 1.5) + np.random.normal(0, 2)
                        pore_diameter_std = pore_mean_diameter * 0.4 + np.random.normal(0, 1)
                        
                        # Crack density increases with temperature
                        crack_density = temp_factor * (2.5 + rubber_factor * 1.5) + np.random.normal(0, 0.3)
                        crack_density = max(0, crack_density)
                        
                        # Crack width (thermal expansion and rubber degradation)
                        if temp < 200:
                            crack_width = 0.0
                        else:
                            crack_width = (temp - 200) / 600 * (1.5 + rubber_factor * 0.8) + np.random.normal(0, 0.15)
                        crack_width = max(0, crack_width)
                        
                        # Interface quality (rubber-cement interface degrades with heat)
                        base_interface_quality = 85 - rubber_content * 0.5
                        interface_degradation = temp_factor * (25 + rubber_factor * 10)
                        interface_quality = base_interface_quality - interface_degradation + np.random.normal(0, 2)
                        interface_quality = np.clip(interface_quality, 20, 95)
                        
                        # Aggregate-paste bond quality
                        bond_quality = 90 - temp_factor * 20 - rubber_factor * 5 + np.random.normal(0, 2)
                        bond_quality = np.clip(bond_quality, 40, 95)
                        
                        # Rubber particle observations
                        if temp < 400:
                            rubber_morphology = "Intact" if temp < 200 else "Partially Melted"
                            rubber_particle_count = max(0, int(rubber_content * 15 * (1 - temp_factor * 0.5)))
                        else:
                            rubber_morphology = "Decomposed" if temp < 600 else "Fully Volatilized"
                            rubber_particle_count = max(0, int(rubber_content * 15 * (1 - temp_factor)))
                        
                        # Calcium hydroxide (CH) crystallinity (decomposes ~450°C)
                        if temp < 450:
                            ch_crystallinity = 75 - temp_factor * 15 + np.random.normal(0, 3)
                        else:
                            ch_crystallinity = max(0, 75 - (temp - 450) / 350 * 75 + np.random.normal(0, 5))
                        
                        # C-S-H gel transformation
                        csh_integrity = 100 - temp_factor * 35 + np.random.normal(0, 3)
                        csh_integrity = np.clip(csh_integrity, 30, 100)
                        
                        sem_records.append({
                            'Sample_ID': sample_id,
                            'Mix_ID': mix,
                            'Temperature': temp,
                            'Rubber_Content': rubber_content,
                            'Replicate': rep,
                            'Field_Of_View': fov,
                            'Analysis_Type': 'SEM',
                            'Measurement_Scale': 'Micro',
                            'Magnification': np.random.choice([500, 1000, 2000, 5000]),
                            'Porosity_Percent': round(total_porosity, 2),
                            'Pore_Mean_Diameter_um': round(pore_mean_diameter, 2),
                            'Pore_Diameter_StdDev_um': round(pore_diameter_std, 2),
                            'Pore_Count_Per_mm2': int(total_porosity * 50 * (1 + rubber_factor)),
                            'Crack_Density_mm_per_mm2': round(crack_density, 3),
                            'Crack_Mean_Width_um': round(crack_width, 2),
                            'Interface_Quality_Score': round(interface_quality, 1),
                            'Aggregate_Bond_Quality_Score': round(bond_quality, 1),
                            'Rubber_Particle_Count': rubber_particle_count,
                            'Rubber_Morphology': rubber_morphology,
                            'CH_Crystallinity_Percent': round(ch_crystallinity, 1),
                            'CSH_Integrity_Score': round(csh_integrity, 1),
                            'Microcrack_Density_per_mm2': round(temp_factor * (5 + rubber_factor * 3) + np.random.normal(0, 0.5), 2),
                            'Surface_Roughness_Ra_um': round((1 + temp_factor * 3 + rubber_factor) * (1 + np.random.normal(0, 0.1)), 2)
                        })
                        
                        # Store for consistency tracking
                        key = f"{sample_id}_{fov}"
                        if key not in self.consistency_tracker:
                            self.consistency_tracker[key] = {}
                        self.consistency_tracker[key]['porosity'] = total_porosity
                        self.consistency_tracker[key]['crack_density'] = crack_density
        
        return pd.DataFrame(sem_records)
    
    def generate_xrd_data(self):
        """Generate X-Ray Diffraction phase analysis data"""
        print("Generating XRD dataset...")
        
        xrd_records = []
        
        for mix in self.mix_ids:
            rubber_content = int(mix.split('-')[1]) if '-' in mix else 0
            
            for temp in self.temperatures:
                for rep in range(1, self.replicates + 1):
                    sample_id = f"{mix}-28-R-{temp}-Furnace-Rep{rep}"
                    
                    temp_factor = temp / 800.0
                    rubber_factor = rubber_content / 30.0
                    
                    # Phase quantification (Rietveld refinement)
                    # C3S (Alite) - decreases with hydration age and temperature
                    c3s_content = max(0, (3 + np.random.normal(0, 0.5)) * (1 - temp_factor * 0.3))
                    
                    # C2S (Belite) - more stable than C3S
                    c2s_content = max(0, (5 + np.random.normal(0, 0.7)) * (1 - temp_factor * 0.2))
                    
                    # Calcium Hydroxide (Portlandite) - decomposes 400-500°C
                    if temp < 400:
                        ch_content = 15 - rubber_factor * 2 + np.random.normal(0, 1.5)
                    elif temp < 550:
                        decomp_factor = (temp - 400) / 150
                        ch_content = (15 - rubber_factor * 2) * (1 - decomp_factor * 0.9) + np.random.normal(0, 1)
                    else:
                        ch_content = max(0, 1.5 + np.random.normal(0, 0.5))
                    
                    # CaCO3 (from carbonation and CH decomposition)
                    if temp < 400:
                        caco3_content = 3 + np.random.normal(0, 0.5)
                    elif temp < 800:
                        # Increases as CH decomposes, then decreases as CaCO3 decomposes (>600°C)
                        if temp < 650:
                            caco3_content = 3 + (temp - 400) / 250 * 8 + np.random.normal(0, 1)
                        else:
                            caco3_content = 11 - (temp - 650) / 150 * 9 + np.random.normal(0, 1)
                    else:
                        caco3_content = max(0, 2 + np.random.normal(0, 0.5))
                    
                    # CaO (quicklime) - forms from CH and CaCO3 decomposition
                    if temp < 500:
                        cao_content = 0.5 + np.random.normal(0, 0.2)
                    else:
                        cao_content = 0.5 + (temp - 500) / 300 * 8 + np.random.normal(0, 1)
                    
                    # Ettringite - decomposes ~70°C, transforms to monosulfate
                    if temp < 100:
                        ettringite = 2.5 + np.random.normal(0, 0.3)
                    else:
                        ettringite = max(0, 0.5 + np.random.normal(0, 0.2))
                    
                    # Quartz (from aggregates) - stable
                    quartz = 25 + rubber_factor * (-2) + np.random.normal(0, 2)
                    
                    # Amorphous content (C-S-H gel and decomposed phases)
                    base_amorphous = 40 - rubber_content * 0.5
                    temp_amorphous_change = temp_factor * 10  # Increases with temperature
                    amorphous_content = base_amorphous + temp_amorphous_change + np.random.normal(0, 2)
                    
                    # Normalize to 100%
                    total = c3s_content + c2s_content + ch_content + caco3_content + cao_content + ettringite + quartz + amorphous_content
                    normalization_factor = 100 / total
                    
                    xrd_records.append({
                        'Sample_ID': sample_id,
                        'Mix_ID': mix,
                        'Temperature': temp,
                        'Rubber_Content': rubber_content,
                        'Replicate': rep,
                        'Analysis_Type': 'XRD',
                        'Measurement_Scale': 'Bulk',
                        'Scan_Range_2Theta': '5-70°',
                        'Step_Size_deg': 0.02,
                        'Counting_Time_sec': 2.0,
                        'C3S_Alite_Percent': round(c3s_content * normalization_factor, 2),
                        'C2S_Belite_Percent': round(c2s_content * normalization_factor, 2),
                        'CH_Portlandite_Percent': round(ch_content * normalization_factor, 2),
                        'CaCO3_Calcite_Percent': round(caco3_content * normalization_factor, 2),
                        'CaO_Quicklime_Percent': round(cao_content * normalization_factor, 2),
                        'Ettringite_Percent': round(ettringite * normalization_factor, 2),
                        'Quartz_Percent': round(quartz * normalization_factor, 2),
                        'Amorphous_Content_Percent': round(amorphous_content * normalization_factor, 2),
                        'Crystallinity_Index': round(100 - amorphous_content * normalization_factor, 2),
                        'Peak_Intensity_CH_d001': int(max(0, 1000 * (ch_content / 15) * (1 + np.random.normal(0, 0.05)))),
                        'Peak_Width_FWHM_deg': round(0.15 + temp_factor * 0.1 + np.random.normal(0, 0.01), 3),
                        'Lattice_Parameter_Variation_Percent': round(temp_factor * 0.8 + np.random.normal(0, 0.1), 3)
                    })
        
        return pd.DataFrame(xrd_records)
    
    def generate_tga_data(self):
        """Generate TGA/DTA thermal analysis data"""
        print("Generating TGA/DTA dataset...")
        
        tga_records = []
        
        for mix in self.mix_ids:
            rubber_content = int(mix.split('-')[1]) if '-' in mix else 0
            
            # TGA is performed on unheated samples to see decomposition behavior
            for rep in range(1, self.replicates + 1):
                sample_id = f"{mix}-28-R-25-Furnace-Rep{rep}"
                
                rubber_factor = rubber_content / 30.0
                
                # Mass loss stages
                # Stage 1: Free water evaporation (30-150°C)
                free_water_loss = 2.5 - rubber_factor * 0.3 + np.random.normal(0, 0.2)
                free_water_temp = 100 + np.random.normal(0, 10)
                
                # Stage 2: Bound water and C-S-H decomposition (150-400°C)
                bound_water_loss = 3.5 - rubber_factor * 0.2 + np.random.normal(0, 0.3)
                bound_water_peak_temp = 220 + np.random.normal(0, 15)
                
                # Stage 3: Rubber decomposition (350-500°C)
                rubber_loss = rubber_content * 0.9 + np.random.normal(0, rubber_content * 0.05)
                rubber_peak_temp = 420 + np.random.normal(0, 20) if rubber_content > 0 else 0
                
                # Stage 4: CH decomposition (400-550°C)
                ch_loss = 4.5 - rubber_factor * 0.5 + np.random.normal(0, 0.4)
                ch_peak_temp = 470 + np.random.normal(0, 15)
                
                # Stage 5: CaCO3 decomposition (600-800°C)
                caco3_loss = 2.8 + np.random.normal(0, 0.3)
                caco3_peak_temp = 720 + np.random.normal(0, 20)
                
                # Total mass loss
                total_mass_loss = free_water_loss + bound_water_loss + rubber_loss + ch_loss + caco3_loss
                
                # Residual mass
                residual_mass = 100 - total_mass_loss
                
                # DTA peaks (heat flow)
                # Endothermic reactions (negative peaks)
                water_heat_flow = -(50 + np.random.normal(0, 5))  # W/g
                rubber_heat_flow = -(200 + rubber_content * 5 + np.random.normal(0, 15)) if rubber_content > 0 else 0
                ch_heat_flow = -(180 + np.random.normal(0, 15))
                caco3_heat_flow = -(150 + np.random.normal(0, 12))
                
                # Derivative mass loss rate (DTG)
                max_dtg_rate = max(rubber_loss / 50 if rubber_content > 0 else 0, ch_loss / 50) + np.random.normal(0, 0.01)
                
                tga_records.append({
                    'Sample_ID': sample_id,
                    'Mix_ID': mix,
                    'Temperature_Range': '25-900°C',
                    'Rubber_Content': rubber_content,
                    'Replicate': rep,
                    'Analysis_Type': 'TGA',
                    'Measurement_Scale': 'Bulk',
                    'Heating_Rate_C_per_min': 10,
                    'Sample_Mass_mg': round(20 + np.random.normal(0, 2), 2),
                    'Atmosphere': 'N2',
                    'Free_Water_Loss_Percent': round(free_water_loss, 2),
                    'Free_Water_Peak_Temp_C': round(free_water_temp, 1),
                    'Bound_Water_Loss_Percent': round(bound_water_loss, 2),
                    'Bound_Water_Peak_Temp_C': round(bound_water_peak_temp, 1),
                    'Rubber_Decomposition_Loss_Percent': round(rubber_loss, 2),
                    'Rubber_Peak_Temp_C': round(rubber_peak_temp, 1) if rubber_content > 0 else None,
                    'CH_Decomposition_Loss_Percent': round(ch_loss, 2),
                    'CH_Peak_Temp_C': round(ch_peak_temp, 1),
                    'CaCO3_Decomposition_Loss_Percent': round(caco3_loss, 2),
                    'CaCO3_Peak_Temp_C': round(caco3_peak_temp, 1),
                    'Total_Mass_Loss_Percent': round(total_mass_loss, 2),
                    'Residual_Mass_Percent': round(residual_mass, 2),
                    'Max_DTG_Rate_Percent_per_min': round(max_dtg_rate, 3),
                    'Water_DTA_Peak_W_per_g': round(water_heat_flow, 2),
                    'Rubber_DTA_Peak_W_per_g': round(rubber_heat_flow, 2) if rubber_content > 0 else None,
                    'CH_DTA_Peak_W_per_g': round(ch_heat_flow, 2),
                    'CaCO3_DTA_Peak_W_per_g': round(caco3_heat_flow, 2),
                    'Onset_Decomposition_Temp_C': round(min(free_water_temp, rubber_peak_temp if rubber_peak_temp > 0 else 1000), 1),
                    'Final_Decomposition_Temp_C': round(caco3_peak_temp + 50, 1)
                })
        
        return pd.DataFrame(tga_records)
    
    def generate_microct_data(self):
        """Generate Micro-CT 3D spatial analysis data"""
        print("Generating Micro-CT dataset...")
        
        microct_records = []
        
        for mix in self.mix_ids:
            rubber_content = int(mix.split('-')[1]) if '-' in mix else 0
            
            for temp in self.temperatures:
                for rep in range(1, self.replicates + 1):
                    sample_id = f"{mix}-28-R-{temp}-Furnace-Rep{rep}"
                    
                    temp_factor = temp / 800.0
                    rubber_factor = rubber_content / 30.0
                    
                    # 3D porosity analysis
                    base_porosity_3d = 6.0 + rubber_content * 0.4
                    thermal_increase = temp_factor * (10.0 + rubber_content * 0.6)
                    total_porosity_3d = base_porosity_3d + thermal_increase + np.random.normal(0, 0.6)
                    
                    # Connected vs isolated porosity
                    connected_ratio = 0.6 + temp_factor * 0.25 + rubber_factor * 0.1 + np.random.normal(0, 0.05)
                    connected_ratio = np.clip(connected_ratio, 0.4, 0.95)
                    connected_porosity = total_porosity_3d * connected_ratio
                    isolated_porosity = total_porosity_3d * (1 - connected_ratio)
                    
                    # Pore size distribution (3D)
                    pore_volume_mean = (35 + rubber_content * 3) * (1 + temp_factor * 2) + np.random.normal(0, 5)
                    pore_volume_std = pore_volume_mean * 0.6
                    
                    # Pore sphericity (rubber creates less spherical pores)
                    pore_sphericity = 0.7 - rubber_factor * 0.15 - temp_factor * 0.1 + np.random.normal(0, 0.03)
                    pore_sphericity = np.clip(pore_sphericity, 0.3, 0.85)
                    
                    # Crack network analysis
                    crack_volume_fraction = temp_factor * (0.8 + rubber_factor * 0.5) + np.random.normal(0, 0.1)
                    crack_volume_fraction = max(0, crack_volume_fraction)
                    
                    # Tortuosity (path complexity)
                    tortuosity = 1.5 + temp_factor * 1.2 + rubber_factor * 0.3 + np.random.normal(0, 0.1)
                    
                    # Specific surface area
                    specific_surface_area = (0.5 + rubber_factor * 0.2) * (1 + temp_factor * 1.5) + np.random.normal(0, 0.05)
                    
                    # Anisotropy index (directional preference)
                    anisotropy = 1.0 + temp_factor * 0.3 + rubber_factor * 0.2 + np.random.normal(0, 0.05)
                    
                    # Rubber particle distribution (for non-zero rubber content)
                    if rubber_content > 0 and temp < 500:
                        rubber_particle_volume = rubber_content * 0.8 * (1 - temp_factor * 0.7) + np.random.normal(0, rubber_content * 0.05)
                        rubber_particle_count_3d = int(rubber_content * 200 * (1 - temp_factor * 0.6))
                        rubber_distribution_uniformity = 0.75 - temp_factor * 0.2 + np.random.normal(0, 0.05)
                    else:
                        rubber_particle_volume = 0
                        rubber_particle_count_3d = 0
                        rubber_distribution_uniformity = None
                    
                    # Interface area (aggregate-paste and rubber-paste)
                    interface_area_density = 2.5 + rubber_factor * 0.5 + np.random.normal(0, 0.2)  # mm²/mm³
                    
                    # Damage metrics
                    damage_parameter = temp_factor * (25 + rubber_factor * 10) + np.random.normal(0, 2)
                    
                    microct_records.append({
                        'Sample_ID': sample_id,
                        'Mix_ID': mix,
                        'Temperature': temp,
                        'Rubber_Content': rubber_content,
                        'Replicate': rep,
                        'Analysis_Type': 'MicroCT',
                        'Measurement_Scale': 'Meso',
                        'Voxel_Size_um': 10,
                        'Scan_Volume_mm3': 8**3,  # 8mm cube
                        'Total_Porosity_3D_Percent': round(total_porosity_3d, 2),
                        'Connected_Porosity_Percent': round(connected_porosity, 2),
                        'Isolated_Porosity_Percent': round(isolated_porosity, 2),
                        'Connectivity_Ratio': round(connected_ratio, 3),
                        'Pore_Count_Total': int(total_porosity_3d * 500 * (1 + rubber_factor)),
                        'Pore_Volume_Mean_um3': round(pore_volume_mean, 1),
                        'Pore_Volume_StdDev_um3': round(pore_volume_std, 1),
                        'Pore_Sphericity_Mean': round(pore_sphericity, 3),
                        'Pore_Elongation_Index': round(1 / pore_sphericity - 1, 3),
                        'Crack_Volume_Fraction_Percent': round(crack_volume_fraction, 3),
                        'Crack_Network_Length_mm': round(temp_factor * (15 + rubber_factor * 8) + np.random.normal(0, 1), 2),
                        'Tortuosity_Factor': round(tortuosity, 3),
                        'Specific_Surface_Area_mm2_per_mm3': round(specific_surface_area, 3),
                        'Anisotropy_Index': round(anisotropy, 3),
                        'Rubber_Particle_Volume_Percent': round(rubber_particle_volume, 2) if rubber_content > 0 else 0,
                        'Rubber_Particle_Count': rubber_particle_count_3d,
                        'Rubber_Distribution_Uniformity': round(rubber_distribution_uniformity, 3) if rubber_distribution_uniformity else None,
                        'Interface_Area_Density_mm2_per_mm3': round(interface_area_density, 3),
                        'Damage_Parameter_Percent': round(damage_parameter, 2),
                        'Fractal_Dimension': round(2.5 + temp_factor * 0.3 + np.random.normal(0, 0.05), 3),
                        'Euler_Number': int(-100 * total_porosity_3d * (1 + rubber_factor))
                    })
                    
                    # Consistency tracking
                    key = f"{sample_id}_CT"
                    if key not in self.consistency_tracker:
                        self.consistency_tracker[key] = {}
                    self.consistency_tracker[key]['porosity_3d'] = total_porosity_3d
                    self.consistency_tracker[key]['crack_volume'] = crack_volume_fraction
        
        return pd.DataFrame(microct_records)
    
    def generate_integrated_summary(self, sem_df, xrd_df, tga_df, microct_df):
        """Generate cross-technique integrated analysis summary"""
        print("Generating integrated cross-technique summary...")
        
        summary_records = []
        
        for mix in self.mix_ids:
            rubber_content = int(mix.split('-')[1]) if '-' in mix else 0
            
            for temp in self.temperatures:
                # Average across replicates
                sem_subset = sem_df[(sem_df['Mix_ID'] == mix) & (sem_df['Temperature'] == temp)]
                xrd_subset = xrd_df[(xrd_df['Mix_ID'] == mix) & (xrd_df['Temperature'] == temp)]
                microct_subset = microct_df[(microct_df['Mix_ID'] == mix) & (microct_df['Temperature'] == temp)]
                
                if len(sem_subset) > 0:
                    summary_records.append({
                        'Mix_ID': mix,
                        'Temperature': temp,
                        'Rubber_Content': rubber_content,
                        'SEM_Avg_Porosity_Percent': round(sem_subset['Porosity_Percent'].mean(), 2),
                        'SEM_Avg_Crack_Density': round(sem_subset['Crack_Density_mm_per_mm2'].mean(), 3),
                        'SEM_Avg_Interface_Quality': round(sem_subset['Interface_Quality_Score'].mean(), 1),
                        'XRD_CH_Content_Percent': round(xrd_subset['CH_Portlandite_Percent'].mean(), 2),
                        'XRD_Amorphous_Content_Percent': round(xrd_subset['Amorphous_Content_Percent'].mean(), 2),
                        'XRD_Crystallinity_Index': round(xrd_subset['Crystallinity_Index'].mean(), 2),
                        'MicroCT_Total_Porosity_Percent': round(microct_subset['Total_Porosity_3D_Percent'].mean(), 2),
                        'MicroCT_Connected_Porosity_Percent': round(microct_subset['Connected_Porosity_Percent'].mean(), 2),
                        'MicroCT_Crack_Volume_Fraction': round(microct_subset['Crack_Volume_Fraction_Percent'].mean(), 3),
                        'MicroCT_Tortuosity': round(microct_subset['Tortuosity_Factor'].mean(), 3),
                        'Porosity_Consistency_SEM_vs_CT_Ratio': round(sem_subset['Porosity_Percent'].mean() / microct_subset['Total_Porosity_3D_Percent'].mean(), 3),
                        'Degradation_Severity_Index': round(
                            (sem_subset['Crack_Density_mm_per_mm2'].mean() * 10 +
                             microct_subset['Damage_Parameter_Percent'].mean() +
                             (100 - sem_subset['Interface_Quality_Score'].mean())) / 3, 2
                        )
                    })
        
        return pd.DataFrame(summary_records)
    
    def export_datasets(self, sem_df, xrd_df, tga_df, microct_df, summary_df):
        """Export all datasets in multiple formats"""
        print("\nExporting datasets...")
        
        # CSV exports
        sem_df.to_csv(self.output_dir / 'phase3_sem_data.csv', index=False)
        xrd_df.to_csv(self.output_dir / 'phase3_xrd_data.csv', index=False)
        tga_df.to_csv(self.output_dir / 'phase3_tga_data.csv', index=False)
        microct_df.to_csv(self.output_dir / 'phase3_microct_data.csv', index=False)
        summary_df.to_csv(self.output_dir / 'phase3_integrated_summary.csv', index=False)
        
        # JSON export (hierarchical structure)
        json_data = {
            'metadata': {
                'title': 'Phase 3 - Microstructural and Chemical Analysis Dataset',
                'research': 'Fire-Resistant Rubberized Concrete Thermo-Mechanical Model',
                'generation_date': datetime.now().isoformat(),
                'temperatures_C': self.temperatures,
                'mix_designs': self.mix_ids,
                'replicates': self.replicates
            },
            'sem_data': sem_df.to_dict(orient='records'),
            'xrd_data': xrd_df.to_dict(orient='records'),
            'tga_data': tga_df.to_dict(orient='records'),
            'microct_data': microct_df.to_dict(orient='records'),
            'integrated_summary': summary_df.to_dict(orient='records')
        }
        
        with open(self.output_dir / 'phase3_complete_dataset.json', 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Statistics summary
        stats_summary = {
            'SEM': {
                'total_samples': len(sem_df),
                'total_fields_of_view': len(sem_df),
                'unique_samples': sem_df['Sample_ID'].nunique(),
                'temperature_range': f"{sem_df['Temperature'].min()}-{sem_df['Temperature'].max()}°C",
                'porosity_range': f"{sem_df['Porosity_Percent'].min():.2f}-{sem_df['Porosity_Percent'].max():.2f}%",
                'crack_density_range': f"{sem_df['Crack_Density_mm_per_mm2'].min():.3f}-{sem_df['Crack_Density_mm_per_mm2'].max():.3f} mm/mm²"
            },
            'XRD': {
                'total_samples': len(xrd_df),
                'unique_samples': xrd_df['Sample_ID'].nunique(),
                'ch_content_range': f"{xrd_df['CH_Portlandite_Percent'].min():.2f}-{xrd_df['CH_Portlandite_Percent'].max():.2f}%",
                'crystallinity_range': f"{xrd_df['Crystallinity_Index'].min():.2f}-{xrd_df['Crystallinity_Index'].max():.2f}%"
            },
            'TGA': {
                'total_samples': len(tga_df),
                'mass_loss_range': f"{tga_df['Total_Mass_Loss_Percent'].min():.2f}-{tga_df['Total_Mass_Loss_Percent'].max():.2f}%",
                'rubber_decomposition_range': f"{tga_df['Rubber_Decomposition_Loss_Percent'].min():.2f}-{tga_df['Rubber_Decomposition_Loss_Percent'].max():.2f}%"
            },
            'MicroCT': {
                'total_samples': len(microct_df),
                'porosity_3d_range': f"{microct_df['Total_Porosity_3D_Percent'].min():.2f}-{microct_df['Total_Porosity_3D_Percent'].max():.2f}%",
                'tortuosity_range': f"{microct_df['Tortuosity_Factor'].min():.3f}-{microct_df['Tortuosity_Factor'].max():.3f}"
            }
        }
        
        with open(self.output_dir / 'dataset_statistics.json', 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        return stats_summary
    
    def generate_all_datasets(self):
        """Main method to generate all Phase 3 datasets"""
        print("=" * 70)
        print("Phase 3 - Microstructural and Chemical Analysis Dataset Generation")
        print("Fire-Resistant Rubberized Concrete Research")
        print("=" * 70)
        print()
        
        # Generate all datasets
        sem_df = self.generate_sem_data()
        xrd_df = self.generate_xrd_data()
        tga_df = self.generate_tga_data()
        microct_df = self.generate_microct_data()
        summary_df = self.generate_integrated_summary(sem_df, xrd_df, tga_df, microct_df)
        
        # Export everything
        stats = self.export_datasets(sem_df, xrd_df, tga_df, microct_df, summary_df)
        
        print("\n" + "=" * 70)
        print("Dataset Generation Complete!")
        print("=" * 70)
        print(f"\nOutput Directory: {self.output_dir}")
        print(f"\nDataset Statistics:")
        print(f"  SEM Records: {len(sem_df)} ({sem_df['Sample_ID'].nunique()} unique samples)")
        print(f"  XRD Records: {len(xrd_df)} ({xrd_df['Sample_ID'].nunique()} unique samples)")
        print(f"  TGA Records: {len(tga_df)} ({tga_df['Sample_ID'].nunique()} unique samples)")
        print(f"  Micro-CT Records: {len(microct_df)} ({microct_df['Sample_ID'].nunique()} unique samples)")
        print(f"  Integrated Summary: {len(summary_df)} conditions")
        print(f"\nTotal Data Points: {len(sem_df) + len(xrd_df) + len(tga_df) + len(microct_df)}")
        print(f"\nFiles Generated:")
        print(f"  - phase3_sem_data.csv")
        print(f"  - phase3_xrd_data.csv")
        print(f"  - phase3_tga_data.csv")
        print(f"  - phase3_microct_data.csv")
        print(f"  - phase3_integrated_summary.csv")
        print(f"  - phase3_complete_dataset.json")
        print(f"  - dataset_statistics.json")
        
        return sem_df, xrd_df, tga_df, microct_df, summary_df


if __name__ == "__main__":
    # Generate complete Phase 3 dataset
    generator = Phase3DatasetGenerator()
    sem, xrd, tga, microct, summary = generator.generate_all_datasets()
    
    print("\n✓ Dataset generation successful!")
    print(f"✓ All files saved to: {generator.output_dir}")
