#!/usr/bin/env python3
"""
Simple Data Analysis Script for Rubberized Concrete Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_and_analyze_data():
    """Load and analyze the synthetic dataset"""
    data_dir = Path("/workspace")
    
    print("Loading datasets...")
    
    # Load main datasets
    mix_proportions = pd.read_csv(data_dir / "mix_proportions_fresh_properties.csv")
    mechanical_props = pd.read_csv(data_dir / "mechanical_properties.csv")
    thermal_props = pd.read_csv(data_dir / "thermal_properties.csv")
    specimen_data = pd.read_csv(data_dir / "specimen_preparation_testing.csv")
    
    print("✓ Datasets loaded successfully")
    
    # Add CR_Content_% to mechanical and thermal properties
    mechanical_props = mechanical_props.merge(
        mix_proportions[['Mix_ID', 'CR_Content_%']], 
        on='Mix_ID', 
        how='left'
    )
    
    thermal_props = thermal_props.merge(
        mix_proportions[['Mix_ID', 'CR_Content_%']], 
        on='Mix_ID', 
        how='left'
    )
    
    print("\n" + "="*60)
    print("SYNTHETIC DATASET SUMMARY")
    print("="*60)
    
    print(f"\n1. MIX DESIGNS: {len(mix_proportions)} total mixes")
    print(f"   - Control mixes: {len(mix_proportions[mix_proportions['CR_Content_%'] == 0])}")
    print(f"   - Rubberized mixes: {len(mix_proportions[mix_proportions['CR_Content_%'] > 0])}")
    print(f"   - Rubber content range: {mix_proportions['CR_Content_%'].min()}% - {mix_proportions['CR_Content_%'].max()}%")
    
    print(f"\n2. MECHANICAL PROPERTIES: {len(mechanical_props)} data points")
    print(f"   - Testing ages: {sorted(mechanical_props['Age_days'].unique())} days")
    print(f"   - Compressive strength range: {mechanical_props['Compressive_Strength_MPa'].min():.1f} - {mechanical_props['Compressive_Strength_MPa'].max():.1f} MPa")
    
    print(f"\n3. THERMAL PROPERTIES: {len(thermal_props)} data points")
    print(f"   - Temperature range: {thermal_props['Temperature_C'].min()}°C - {thermal_props['Temperature_C'].max()}°C")
    print(f"   - Thermal conductivity range: {thermal_props['Thermal_Conductivity_W_mK'].min():.2f} - {thermal_props['Thermal_Conductivity_W_mK'].max():.2f} W/m·K")
    
    print(f"\n4. SPECIMEN DATA: {len(specimen_data)} specimens")
    print(f"   - Density range: {specimen_data['Density_kg_m3'].min()} - {specimen_data['Density_kg_m3'].max()} kg/m³")
    
    # Analyze rubber effects
    print("\n" + "="*60)
    print("RUBBER CONTENT EFFECTS ANALYSIS")
    print("="*60)
    
    # Fresh properties vs rubber content
    fresh_analysis = mix_proportions.groupby('CR_Content_%').agg({
        'Fresh_Density_kg_m3': 'mean',
        'Slump_mm': 'mean',
        'Air_Content_%': 'mean'
    }).round(2)
    
    print("\n1. FRESH PROPERTIES vs RUBBER CONTENT:")
    print(fresh_analysis)
    
    # Mechanical properties at 28 days
    mech_28d = mechanical_props[mechanical_props['Age_days'] == 28]
    mech_analysis = mech_28d.groupby('CR_Content_%').agg({
        'Compressive_Strength_MPa': 'mean',
        'Modulus_of_Elasticity_GPa': 'mean'
    }).round(2)
    
    print("\n2. MECHANICAL PROPERTIES vs RUBBER CONTENT (28-day):")
    print(mech_analysis)
    
    # Thermal properties at 20°C
    thermal_20c = thermal_props[thermal_props['Temperature_C'] == 20]
    thermal_analysis = thermal_20c.groupby('CR_Content_%').agg({
        'Thermal_Conductivity_W_mK': 'mean',
        'Fire_Resistance_Rating_min': 'mean'
    }).round(3)
    
    print("\n3. THERMAL PROPERTIES vs RUBBER CONTENT (20°C):")
    print(thermal_analysis)
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Compressive strength vs age
    ax1 = axes[0, 0]
    for mix in mechanical_props['Mix_ID'].unique():
        mix_data = mechanical_props[mechanical_props['Mix_ID'] == mix]
        ax1.plot(mix_data['Age_days'], mix_data['Compressive_Strength_MPa'], 
                marker='o', label=mix, linewidth=2)
    ax1.set_xlabel('Age (days)')
    ax1.set_ylabel('Compressive Strength (MPa)')
    ax1.set_title('Compressive Strength Development')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Fresh density vs rubber content
    ax2 = axes[0, 1]
    density_data = mix_proportions.groupby('CR_Content_%')['Fresh_Density_kg_m3'].mean()
    ax2.plot(density_data.index, density_data.values, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Rubber Content (%)')
    ax2.set_ylabel('Fresh Density (kg/m³)')
    ax2.set_title('Fresh Density vs Rubber Content')
    ax2.grid(True, alpha=0.3)
    
    # 3. Thermal conductivity vs temperature
    ax3 = axes[0, 2]
    for mix in thermal_props['Mix_ID'].unique():
        mix_data = thermal_props[thermal_props['Mix_ID'] == mix]
        ax3.plot(mix_data['Temperature_C'], mix_data['Thermal_Conductivity_W_mK'], 
                marker='s', label=mix, linewidth=2)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Thermal Conductivity (W/m·K)')
    ax3.set_title('Thermal Conductivity vs Temperature')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Fire resistance vs rubber content
    ax4 = axes[1, 0]
    fire_resistance = thermal_props.groupby('CR_Content_%')['Fire_Resistance_Rating_min'].mean()
    ax4.bar(fire_resistance.index, fire_resistance.values, color='red', alpha=0.7)
    ax4.set_xlabel('Rubber Content (%)')
    ax4.set_ylabel('Fire Resistance Rating (min)')
    ax4.set_title('Fire Resistance vs Rubber Content')
    ax4.grid(True, alpha=0.3)
    
    # 5. Modulus of elasticity vs rubber content
    ax5 = axes[1, 1]
    mod_data = mechanical_props[mechanical_props['Age_days'] == 28]
    mod_analysis = mod_data.groupby('CR_Content_%')['Modulus_of_Elasticity_GPa'].mean()
    ax5.plot(mod_analysis.index, mod_analysis.values, 'go-', linewidth=2, markersize=8)
    ax5.set_xlabel('Rubber Content (%)')
    ax5.set_ylabel('Modulus of Elasticity (GPa)')
    ax5.set_title('Modulus of Elasticity vs Rubber Content (28-day)')
    ax5.grid(True, alpha=0.3)
    
    # 6. Workability vs rubber content
    ax6 = axes[1, 2]
    workability_data = mix_proportions.groupby('CR_Content_%')['Slump_mm'].mean()
    ax6.plot(workability_data.index, workability_data.values, 'mo-', linewidth=2, markersize=8)
    ax6.set_xlabel('Rubber Content (%)')
    ax6.set_ylabel('Slump (mm)')
    ax6.set_title('Workability vs Rubber Content')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(data_dir / 'synthetic_dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved as 'synthetic_dataset_analysis.png'")
    
    # Export summary results
    summary_results = {
        'dataset_info': {
            'total_mixes': len(mix_proportions),
            'total_specimens': len(specimen_data),
            'total_mechanical_tests': len(mechanical_props),
            'total_thermal_tests': len(thermal_props)
        },
        'rubber_content_effects': fresh_analysis.to_dict(),
        'mechanical_properties_28d': mech_analysis.to_dict(),
        'thermal_properties_20c': thermal_analysis.to_dict()
    }
    
    with open(data_dir / 'analysis_results.json', 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    print("✓ Analysis results exported to 'analysis_results.json'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Generated files:")
    print("  - synthetic_dataset_analysis.png (visualizations)")
    print("  - analysis_results.json (summary statistics)")
    print("  - All original CSV datasets")
    print("  - complete_dataset.json (comprehensive JSON format)")

if __name__ == "__main__":
    load_and_analyze_data()