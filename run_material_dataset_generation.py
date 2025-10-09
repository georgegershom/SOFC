#!/usr/bin/env python3
"""
Main Execution Script for SOFC Material Property Dataset Generation
================================================================

This script orchestrates the complete generation, validation, and export
of the comprehensive SOFC material property dataset.

Author: Generated for SOFC Research
Date: 2025-10-09
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from material_property_dataset import SOFCMaterialDatabase, main as run_main_database
from experimental_data_generator import (
    generate_comprehensive_experimental_dataset,
    export_experimental_data
)
from data_validation_and_analysis import (
    MaterialPropertyValidator,
    UncertaintyQuantification,
    DataVisualization,
    generate_comprehensive_validation_report
)

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

def print_header(title: str):
    """Print formatted header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def print_progress(message: str):
    """Print progress message."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def main():
    """Main execution function."""
    
    print_header("SOFC Material Property Dataset Generation")
    print("Comprehensive material property database for Solid Oxide Fuel Cells")
    print("Including YSZ, Ni, Ni-YSZ composites, and critical interfaces")
    
    start_time = time.time()
    
    # Step 1: Initialize and populate material database
    print_header("Step 1: Material Property Database Creation")
    print_progress("Initializing SOFC material database...")
    
    database = SOFCMaterialDatabase()
    
    print_progress("Database initialized successfully!")
    print(f"Materials available: {list(database.materials.keys())}")
    print(f"Interfaces available: {list(database.interfaces.keys())}")
    
    # Step 2: Generate experimental data
    print_header("Step 2: Synthetic Experimental Data Generation")
    print_progress("Generating comprehensive experimental datasets...")
    
    experimental_data = generate_comprehensive_experimental_dataset(database)
    
    print_progress("Experimental data generation completed!")
    print(f"Generated data for {len(experimental_data)} materials/interfaces")
    
    # Step 3: Data validation and analysis
    print_header("Step 3: Data Validation and Quality Assessment")
    print_progress("Performing comprehensive data validation...")
    
    validation_report = generate_comprehensive_validation_report(database, experimental_data)
    
    print_progress("Validation completed!")
    print(f"Validation success rate: {validation_report['summary']['validation_success_rate']:.1%}")
    print(f"Overall confidence level: {validation_report['summary']['overall_confidence']:.1%}")
    
    # Step 4: Export all data
    print_header("Step 4: Data Export and Documentation")
    print_progress("Exporting material property database...")
    
    # Export main database
    csv_data = database.export_to_csv("sofc_material_properties_complete.csv")
    json_data = database.export_to_json("sofc_material_properties_complete.json")
    
    print_progress("Exporting experimental datasets...")
    
    # Export experimental data
    experimental_summary = export_experimental_data(experimental_data, "sofc_experimental")
    
    print_progress("Exporting validation report...")
    
    # Export validation report
    with open("sofc_validation_report.json", 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the validation report
        def deep_convert(obj):
            if isinstance(obj, dict):
                return {k: deep_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [deep_convert(v) for v in obj]
            else:
                return convert_numpy(obj)
        
        json.dump(deep_convert(validation_report), f, indent=2, default=str)
    
    # Step 5: Generate summary statistics and visualizations
    print_header("Step 5: Summary Statistics and Key Insights")
    
    # Material property summary
    print("\n📊 MATERIAL PROPERTY SUMMARY")
    print("-" * 50)
    
    materials_summary = []
    for material_name in ['YSZ', 'Ni', 'Ni-YSZ_40', 'Ni-YSZ_50']:
        try:
            props = database.get_material_properties(material_name)
            elastic = props['elastic_1073K']
            fracture = props['fracture']
            thermal = props['thermal']
            
            materials_summary.append({
                'Material': material_name,
                'E (GPa)': f"{elastic.youngs_modulus_GPa.value:.1f} ± {elastic.youngs_modulus_GPa.uncertainty:.1f}",
                'ν': f"{elastic.poissons_ratio.value:.3f} ± {elastic.poissons_ratio.uncertainty:.3f}",
                'K_Ic (MPa√m)': f"{fracture.fracture_toughness_MPa_sqrt_m.value:.1f} ± {fracture.fracture_toughness_MPa_sqrt_m.uncertainty:.1f}",
                'CTE (×10⁻⁶/K)': f"{thermal.thermal_expansion_coefficient_K_inv.value*1e6:.1f} ± {thermal.thermal_expansion_coefficient_K_inv.uncertainty*1e6:.1f}"
            })
        except Exception as e:
            print(f"Warning: Could not process {material_name}: {e}")
    
    summary_df = pd.DataFrame(materials_summary)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("sofc_materials_summary.csv", index=False)
    
    # Interface property summary
    print("\n🔗 INTERFACE PROPERTY SUMMARY")
    print("-" * 50)
    
    interface_summary = []
    for interface_name in ['anode_electrolyte', 'Ni_YSZ']:
        try:
            interface_props = database.get_interface_properties(interface_name)
            fracture = interface_props['fracture']
            
            interface_summary.append({
                'Interface': interface_name.replace('_', '/'),
                'G_c (J/m²)': f"{fracture.critical_energy_release_rate_J_m2.value:.1f} ± {fracture.critical_energy_release_rate_J_m2.uncertainty:.1f}",
                'K_Ic (MPa√m)': f"{fracture.fracture_toughness_MPa_sqrt_m.value:.1f} ± {fracture.fracture_toughness_MPa_sqrt_m.uncertainty:.1f}",
                'Criticality': 'HIGH' if 'anode' in interface_name else 'MEDIUM'
            })
        except Exception as e:
            print(f"Warning: Could not process {interface_name}: {e}")
    
    interface_df = pd.DataFrame(interface_summary)
    print(interface_df.to_string(index=False))
    interface_df.to_csv("sofc_interfaces_summary.csv", index=False)
    
    # Experimental data summary
    print("\n🧪 EXPERIMENTAL DATA SUMMARY")
    print("-" * 50)
    
    exp_summary = []
    for material_name, data in experimental_data.items():
        if 'nanoindentation' in data:
            stats = data['nanoindentation']['analysis_results']['statistics']
            exp_summary.append({
                'Material': material_name,
                'Measurements': stats['n_valid_measurements'],
                'Success Rate': f"{stats['success_rate']:.1%}",
                'Mean E (GPa)': f"{stats['youngs_modulus']['mean_GPa']:.1f}",
                'CV (%)': f"{stats['youngs_modulus']['cv_percent']:.1f}",
                'Data Quality': 'GOOD' if stats['youngs_modulus']['cv_percent'] < 10 else 'FAIR'
            })
    
    if exp_summary:
        exp_df = pd.DataFrame(exp_summary)
        print(exp_df.to_string(index=False))
        exp_df.to_csv("sofc_experimental_summary.csv", index=False)
    
    # Step 6: Key insights and recommendations
    print_header("Step 6: Key Insights and Recommendations")
    
    print("\n🔍 KEY INSIGHTS:")
    print("-" * 30)
    print("1. Interface Properties are Critical:")
    print("   • Anode/electrolyte interface has lowest fracture toughness (1.1 MPa√m)")
    print("   • Interface failure dominates SOFC reliability")
    print("   • Thermal cycling creates maximum stress at interfaces")
    
    print("\n2. Material Property Hierarchy:")
    print("   • Ni: High toughness (85 MPa√m) but high CTE (16.8×10⁻⁶/K)")
    print("   • YSZ: Moderate toughness (2.2 MPa√m) but lower CTE (10.8×10⁻⁶/K)")
    print("   • Ni-YSZ: Balanced properties depend on volume fraction")
    
    print("\n3. Chemical Expansion Concerns:")
    print("   • Ni oxidation causes 21% volume expansion")
    print("   • YSZ constraint reduces but doesn't eliminate expansion")
    print("   • Critical for redox cycling durability")
    
    print("\n4. Temperature Dependencies:")
    print("   • Young's modulus decreases ~20% from RT to 800°C")
    print("   • CTE mismatch drives residual stresses")
    print("   • Interface properties most temperature-sensitive")
    
    print("\n📋 RECOMMENDATIONS:")
    print("-" * 30)
    print("1. Prioritize interface toughening strategies")
    print("2. Optimize Ni volume fraction (40-50% recommended)")
    print("3. Implement graded compositions near interfaces")
    print("4. Control redox cycling to prevent Ni oxidation")
    print("5. Use temperature-dependent properties in FE models")
    print("6. Validate critical interface properties experimentally")
    
    # Step 7: Dataset statistics
    print_header("Step 7: Dataset Statistics")
    
    total_properties = len(csv_data)
    total_materials = len(database.materials)
    total_interfaces = len(database.interfaces)
    
    print(f"\n📈 DATASET STATISTICS:")
    print(f"   • Total property records: {total_properties:,}")
    print(f"   • Bulk materials: {total_materials}")
    print(f"   • Interface systems: {total_interfaces}")
    print(f"   • Temperature points: {len(database.temperature_dependencies['temperatures_K'])}")
    print(f"   • Composite variations: {sum(1 for k in database.materials.keys() if 'Ni-YSZ' in k)}")
    
    # Calculate file sizes
    file_stats = []
    for filename in ['sofc_material_properties_complete.csv', 
                    'sofc_material_properties_complete.json',
                    'sofc_validation_report.json']:
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            file_stats.append(f"   • {filename}: {size_mb:.2f} MB")
    
    print(f"\n💾 EXPORTED FILES:")
    for stat in file_stats:
        print(stat)
    
    # Completion
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("Dataset Generation Complete!")
    print(f"⏱️  Total execution time: {duration:.1f} seconds")
    print(f"✅ Successfully generated comprehensive SOFC material property dataset")
    print(f"📁 All files exported to current directory")
    print(f"🔬 Ready for finite element modeling and failure analysis")
    
    return {
        'database': database,
        'experimental_data': experimental_data,
        'validation_report': validation_report,
        'execution_time': duration
    }

if __name__ == "__main__":
    try:
        results = main()
        print(f"\n🎉 Dataset generation completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)