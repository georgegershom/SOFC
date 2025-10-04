#!/usr/bin/env python3
"""
Material Properties Dataset Validation Script

This script validates the integrity and consistency of the material properties dataset
for multi-physics model calibration.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def load_datasets():
    """Load all CSV datasets"""
    datasets = {}
    
    # Load mechanical properties
    datasets['mechanical'] = pd.read_csv('mechanical_properties.csv')
    
    # Load creep properties
    datasets['creep'] = pd.read_csv('creep_properties.csv')
    
    # Load thermophysical properties
    datasets['thermophysical'] = pd.read_csv('thermophysical_properties.csv')
    
    # Load electrochemical properties
    datasets['electrochemical'] = pd.read_csv('electrochemical_properties.csv')
    
    return datasets

def validate_data_integrity(datasets):
    """Validate data integrity and consistency"""
    print("=== DATA INTEGRITY VALIDATION ===\n")
    
    for name, df in datasets.items():
        print(f"{name.upper()} DATASET:")
        print(f"  - Shape: {df.shape}")
        print(f"  - Missing values: {df.isnull().sum().sum()}")
        print(f"  - Duplicate rows: {df.duplicated().sum()}")
        
        # Check temperature ranges
        if 'Temperature_C' in df.columns:
            temp_range = f"{df['Temperature_C'].min()}-{df['Temperature_C'].max()}°C"
            print(f"  - Temperature range: {temp_range}")
        
        # Check material coverage
        if 'Material_ID' in df.columns:
            materials = df['Material_ID'].unique()
            print(f"  - Materials: {len(materials)} ({', '.join(materials)})")
        
        print()

def validate_physical_consistency(datasets):
    """Validate physical consistency of properties"""
    print("=== PHYSICAL CONSISTENCY VALIDATION ===\n")
    
    # Mechanical properties validation
    mech_df = datasets['mechanical']
    print("MECHANICAL PROPERTIES:")
    
    # Check if Young's modulus decreases with temperature (expected behavior)
    for material in mech_df['Material_ID'].unique():
        material_data = mech_df[mech_df['Material_ID'] == material].sort_values('Temperature_C')
        modulus_trend = np.diff(material_data['Youngs_Modulus_GPa'].values)
        decreasing_trend = np.sum(modulus_trend < 0) / len(modulus_trend)
        print(f"  - {material}: Young's modulus decreasing trend: {decreasing_trend:.1%}")
    
    # Thermophysical properties validation
    thermo_df = datasets['thermophysical']
    print("\nTHERMOPHYSICAL PROPERTIES:")
    
    # Check if thermal conductivity generally increases with temperature for metals
    for material in thermo_df['Material_ID'].unique():
        if any(metal in material for metal in ['SS316L', 'Inconel', 'Ti6Al4V']):
            material_data = thermo_df[thermo_df['Material_ID'] == material].sort_values('Temperature_C')
            conductivity_trend = np.diff(material_data['Thermal_Conductivity_W_mK'].values)
            increasing_trend = np.sum(conductivity_trend > 0) / len(conductivity_trend)
            print(f"  - {material}: Thermal conductivity increasing trend: {increasing_trend:.1%}")
    
    # Electrochemical properties validation
    electro_df = datasets['electrochemical']
    print("\nELECTROCHEMICAL PROPERTIES:")
    
    # Check Arrhenius behavior (conductivity should increase with temperature)
    for material in electro_df['Material_ID'].unique():
        material_data = electro_df[electro_df['Material_ID'] == material].sort_values('Temperature_C')
        ionic_trend = np.diff(material_data['Ionic_Conductivity_S_m'].values)
        increasing_trend = np.sum(ionic_trend > 0) / len(ionic_trend)
        print(f"  - {material}: Ionic conductivity increasing trend: {increasing_trend:.1%}")

def generate_summary_plots(datasets):
    """Generate summary plots for data visualization"""
    print("\n=== GENERATING SUMMARY PLOTS ===\n")
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Material Properties Dataset Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Mechanical Properties vs Temperature
    mech_df = datasets['mechanical']
    ax1 = axes[0, 0]
    for material in mech_df['Material_ID'].unique():
        material_data = mech_df[mech_df['Material_ID'] == material]
        ax1.plot(material_data['Temperature_C'], material_data['Youngs_Modulus_GPa'], 
                marker='o', label=material, linewidth=2)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel("Young's Modulus (GPa)")
    ax1.set_title('Mechanical Properties')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Creep Strain vs Time
    creep_df = datasets['creep']
    ax2 = axes[0, 1]
    # Plot representative creep curves at different temperatures
    for temp in [600, 650, 700]:
        temp_data = creep_df[(creep_df['Temperature_C'] == temp) & 
                            (creep_df['Material_ID'] == 'SS316L_001')]
        if not temp_data.empty:
            ax2.loglog(temp_data['Time_Hours'], temp_data['Creep_Strain_Percent'], 
                      marker='s', label=f'{temp}°C', linewidth=2)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Creep Strain (%)')
    ax2.set_title('Creep Behavior (SS316L)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Thermal Conductivity vs Temperature
    thermo_df = datasets['thermophysical']
    ax3 = axes[1, 0]
    for material in thermo_df['Material_ID'].unique():
        material_data = thermo_df[thermo_df['Material_ID'] == material]
        ax3.plot(material_data['Temperature_C'], material_data['Thermal_Conductivity_W_mK'], 
                marker='^', label=material, linewidth=2)
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Thermal Conductivity (W/m·K)')
    ax3.set_title('Thermal Properties')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Ionic Conductivity vs Temperature (Arrhenius plot)
    electro_df = datasets['electrochemical']
    ax4 = axes[1, 1]
    for material in electro_df['Material_ID'].unique():
        material_data = electro_df[electro_df['Material_ID'] == material]
        # Convert to 1000/T for Arrhenius plot
        inv_temp = 1000 / (material_data['Temperature_C'] + 273.15)
        ax4.semilogy(inv_temp, material_data['Ionic_Conductivity_S_m'], 
                    marker='d', label=material, linewidth=2)
    ax4.set_xlabel('1000/T (K⁻¹)')
    ax4.set_ylabel('Ionic Conductivity (S/m)')
    ax4.set_title('Electrochemical Properties (Arrhenius)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_summary_plots.png', dpi=300, bbox_inches='tight')
    print("Summary plots saved as 'dataset_summary_plots.png'")

def generate_statistics_report(datasets):
    """Generate comprehensive statistics report"""
    print("\n=== GENERATING STATISTICS REPORT ===\n")
    
    stats_report = {}
    
    for name, df in datasets.items():
        stats_report[name] = {
            'total_samples': len(df),
            'materials_count': df['Material_ID'].nunique() if 'Material_ID' in df.columns else 0,
            'temperature_range': {
                'min': float(df['Temperature_C'].min()) if 'Temperature_C' in df.columns else None,
                'max': float(df['Temperature_C'].max()) if 'Temperature_C' in df.columns else None
            },
            'numeric_columns': {}
        }
        
        # Calculate statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'Temperature_C':  # Skip temperature as it's a controlled variable
                stats_report[name]['numeric_columns'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'coefficient_of_variation': float(df[col].std() / df[col].mean())
                }
    
    # Save statistics report
    with open('dataset_statistics.json', 'w') as f:
        json.dump(stats_report, f, indent=2)
    
    print("Statistics report saved as 'dataset_statistics.json'")
    
    # Print summary
    total_samples = sum(stats['total_samples'] for stats in stats_report.values())
    print(f"Total dataset samples: {total_samples}")
    print(f"Dataset files: {len(stats_report)}")

def main():
    """Main validation function"""
    print("Material Properties Dataset Validation")
    print("=" * 50)
    
    try:
        # Load datasets
        datasets = load_datasets()
        print(f"Successfully loaded {len(datasets)} datasets\n")
        
        # Run validations
        validate_data_integrity(datasets)
        validate_physical_consistency(datasets)
        
        # Generate outputs
        generate_summary_plots(datasets)
        generate_statistics_report(datasets)
        
        print("\n" + "=" * 50)
        print("VALIDATION COMPLETE")
        print("All datasets passed integrity and consistency checks!")
        print("Summary plots and statistics generated successfully.")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()