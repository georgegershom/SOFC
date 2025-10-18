#!/usr/bin/env python3
"""
Phase 3 Sample Data Viewer
Display representative data samples from each analysis technique
"""

import pandas as pd
from pathlib import Path

def display_sample_records():
    """Display sample records from each dataset"""
    
    data_dir = Path('/workspace/phase3_datasets')
    
    print("=" * 80)
    print("PHASE 3 DATASET - SAMPLE DATA RECORDS")
    print("=" * 80)
    print()
    
    # Load datasets
    sem = pd.read_csv(data_dir / 'phase3_sem_data.csv')
    xrd = pd.read_csv(data_dir / 'phase3_xrd_data.csv')
    tga = pd.read_csv(data_dir / 'phase3_tga_data.csv')
    microct = pd.read_csv(data_dir / 'phase3_microct_data.csv')
    
    # SEM Sample
    print("1. SEM DATA SAMPLE")
    print("-" * 80)
    print("Sample: C-20-28-R-600-Furnace-Rep1, Field of View 1")
    print()
    sample_sem = sem[(sem['Mix_ID'] == 'C-20') & 
                     (sem['Temperature'] == 600) & 
                     (sem['Replicate'] == 1) & 
                     (sem['Field_Of_View'] == 1)]
    
    if len(sample_sem) > 0:
        record = sample_sem.iloc[0]
        print(f"  Porosity: {record['Porosity_Percent']:.2f}%")
        print(f"  Pore Mean Diameter: {record['Pore_Mean_Diameter_um']:.2f} µm")
        print(f"  Pore Count: {record['Pore_Count_Per_mm2']:.0f} pores/mm²")
        print(f"  Crack Density: {record['Crack_Density_mm_per_mm2']:.3f} mm/mm²")
        print(f"  Crack Mean Width: {record['Crack_Mean_Width_um']:.2f} µm")
        print(f"  Interface Quality Score: {record['Interface_Quality_Score']:.1f}/100")
        print(f"  Aggregate Bond Quality: {record['Aggregate_Bond_Quality_Score']:.1f}/100")
        print(f"  Rubber Morphology: {record['Rubber_Morphology']}")
        print(f"  Rubber Particle Count: {record['Rubber_Particle_Count']:.0f}")
        print(f"  CH Crystallinity: {record['CH_Crystallinity_Percent']:.1f}%")
        print(f"  C-S-H Integrity: {record['CSH_Integrity_Score']:.1f}/100")
        print(f"  Microcrack Density: {record['Microcrack_Density_per_mm2']:.2f} /mm²")
        print(f"  Surface Roughness: {record['Surface_Roughness_Ra_um']:.2f} µm")
    
    print()
    print()
    
    # XRD Sample
    print("2. XRD DATA SAMPLE")
    print("-" * 80)
    print("Sample: C-20-28-R-600-Furnace-Rep1")
    print()
    sample_xrd = xrd[(xrd['Mix_ID'] == 'C-20') & 
                     (xrd['Temperature'] == 600) & 
                     (xrd['Replicate'] == 1)]
    
    if len(sample_xrd) > 0:
        record = sample_xrd.iloc[0]
        print("  Phase Composition (Rietveld Refinement):")
        print(f"    C3S (Alite): {record['C3S_Alite_Percent']:.2f}%")
        print(f"    C2S (Belite): {record['C2S_Belite_Percent']:.2f}%")
        print(f"    CH (Portlandite): {record['CH_Portlandite_Percent']:.2f}%")
        print(f"    CaCO3 (Calcite): {record['CaCO3_Calcite_Percent']:.2f}%")
        print(f"    CaO (Quicklime): {record['CaO_Quicklime_Percent']:.2f}%")
        print(f"    Ettringite: {record['Ettringite_Percent']:.2f}%")
        print(f"    Quartz: {record['Quartz_Percent']:.2f}%")
        print(f"    Amorphous (C-S-H): {record['Amorphous_Content_Percent']:.2f}%")
        print()
        print(f"  Crystallinity Index: {record['Crystallinity_Index']:.2f}%")
        print(f"  Peak Intensity (CH d001): {record['Peak_Intensity_CH_d001']:.0f} counts")
        print(f"  Peak Width (FWHM): {record['Peak_Width_FWHM_deg']:.3f}°")
        print(f"  Lattice Parameter Variation: {record['Lattice_Parameter_Variation_Percent']:.3f}%")
    
    print()
    print()
    
    # TGA Sample
    print("3. TGA/DTA DATA SAMPLE")
    print("-" * 80)
    print("Sample: C-20-28-R-25-Furnace-Rep1 (unheated, analyzed 25-900°C)")
    print()
    sample_tga = tga[(tga['Mix_ID'] == 'C-20') & (tga['Replicate'] == 1)]
    
    if len(sample_tga) > 0:
        record = sample_tga.iloc[0]
        print(f"  Heating Rate: {record['Heating_Rate_C_per_min']:.0f}°C/min")
        print(f"  Atmosphere: {record['Atmosphere']}")
        print(f"  Sample Mass: {record['Sample_Mass_mg']:.2f} mg")
        print()
        print("  Mass Loss Stages:")
        print(f"    Free Water (30-150°C):")
        print(f"      Loss: {record['Free_Water_Loss_Percent']:.2f}%")
        print(f"      Peak Temp: {record['Free_Water_Peak_Temp_C']:.1f}°C")
        print()
        print(f"    Bound Water (150-400°C):")
        print(f"      Loss: {record['Bound_Water_Loss_Percent']:.2f}%")
        print(f"      Peak Temp: {record['Bound_Water_Peak_Temp_C']:.1f}°C")
        print()
        print(f"    Rubber Decomposition (350-500°C):")
        print(f"      Loss: {record['Rubber_Decomposition_Loss_Percent']:.2f}%")
        if pd.notna(record['Rubber_Peak_Temp_C']):
            print(f"      Peak Temp: {record['Rubber_Peak_Temp_C']:.1f}°C")
        print()
        print(f"    CH Dehydroxylation (400-550°C):")
        print(f"      Loss: {record['CH_Decomposition_Loss_Percent']:.2f}%")
        print(f"      Peak Temp: {record['CH_Peak_Temp_C']:.1f}°C")
        print()
        print(f"    CaCO3 Decarbonation (600-800°C):")
        print(f"      Loss: {record['CaCO3_Decomposition_Loss_Percent']:.2f}%")
        print(f"      Peak Temp: {record['CaCO3_Peak_Temp_C']:.1f}°C")
        print()
        print(f"  Total Mass Loss: {record['Total_Mass_Loss_Percent']:.2f}%")
        print(f"  Residual Mass: {record['Residual_Mass_Percent']:.2f}%")
        print(f"  Max DTG Rate: {record['Max_DTG_Rate_Percent_per_min']:.3f} %/min")
        print()
        print("  DTA Heat Flow Peaks:")
        print(f"    Water: {record['Water_DTA_Peak_W_per_g']:.2f} W/g (endothermic)")
        if pd.notna(record['Rubber_DTA_Peak_W_per_g']):
            print(f"    Rubber: {record['Rubber_DTA_Peak_W_per_g']:.2f} W/g (endothermic)")
        print(f"    CH: {record['CH_DTA_Peak_W_per_g']:.2f} W/g (endothermic)")
        print(f"    CaCO3: {record['CaCO3_DTA_Peak_W_per_g']:.2f} W/g (endothermic)")
    
    print()
    print()
    
    # Micro-CT Sample
    print("4. MICRO-CT DATA SAMPLE")
    print("-" * 80)
    print("Sample: C-20-28-R-600-Furnace-Rep1")
    print()
    sample_ct = microct[(microct['Mix_ID'] == 'C-20') & 
                        (microct['Temperature'] == 600) & 
                        (microct['Replicate'] == 1)]
    
    if len(sample_ct) > 0:
        record = sample_ct.iloc[0]
        print(f"  Scan Parameters:")
        print(f"    Voxel Size: {record['Voxel_Size_um']:.0f} µm")
        print(f"    Scan Volume: {record['Scan_Volume_mm3']:.0f} mm³")
        print()
        print(f"  3D Porosity Analysis:")
        print(f"    Total Porosity: {record['Total_Porosity_3D_Percent']:.2f}%")
        print(f"    Connected Porosity: {record['Connected_Porosity_Percent']:.2f}%")
        print(f"    Isolated Porosity: {record['Isolated_Porosity_Percent']:.2f}%")
        print(f"    Connectivity Ratio: {record['Connectivity_Ratio']:.3f}")
        print()
        print(f"  Pore Characteristics:")
        print(f"    Total Pore Count: {record['Pore_Count_Total']:.0f}")
        print(f"    Mean Pore Volume: {record['Pore_Volume_Mean_um3']:.1f} µm³")
        print(f"    Pore Volume Std Dev: {record['Pore_Volume_StdDev_um3']:.1f} µm³")
        print(f"    Mean Sphericity: {record['Pore_Sphericity_Mean']:.3f}")
        print(f"    Elongation Index: {record['Pore_Elongation_Index']:.3f}")
        print()
        print(f"  Crack Network:")
        print(f"    Crack Volume Fraction: {record['Crack_Volume_Fraction_Percent']:.3f}%")
        print(f"    Crack Network Length: {record['Crack_Network_Length_mm']:.2f} mm")
        print()
        print(f"  Network Topology:")
        print(f"    Tortuosity Factor: {record['Tortuosity_Factor']:.3f}")
        print(f"    Specific Surface Area: {record['Specific_Surface_Area_mm2_per_mm3']:.3f} mm²/mm³")
        print(f"    Anisotropy Index: {record['Anisotropy_Index']:.3f}")
        print(f"    Fractal Dimension: {record['Fractal_Dimension']:.3f}")
        print()
        print(f"  Interface Analysis:")
        print(f"    Interface Area Density: {record['Interface_Area_Density_mm2_per_mm3']:.3f} mm²/mm³")
        if record['Rubber_Content'] > 0:
            print(f"    Rubber Particle Volume: {record['Rubber_Particle_Volume_Percent']:.2f}%")
            print(f"    Rubber Particle Count: {record['Rubber_Particle_Count']:.0f}")
            if pd.notna(record['Rubber_Distribution_Uniformity']):
                print(f"    Distribution Uniformity: {record['Rubber_Distribution_Uniformity']:.3f}")
        print()
        print(f"  Damage Assessment:")
        print(f"    Damage Parameter: {record['Damage_Parameter_Percent']:.2f}%")
        print(f"    Euler Number: {record['Euler_Number']:.0f}")
    
    print()
    print("=" * 80)
    print("END OF SAMPLE DATA")
    print("=" * 80)


if __name__ == "__main__":
    display_sample_records()
