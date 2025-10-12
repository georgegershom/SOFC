#!/usr/bin/env python3
"""
Stratified Flow Acoustics Dataset Summary
Comprehensive summary and validation of the generated dataset.
"""

import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

def generate_dataset_summary():
    """Generate comprehensive dataset summary."""
    
    print("="*80)
    print("STRATIFIED FLOW ACOUSTICS DATASET - COMPREHENSIVE SUMMARY")
    print("PhD Thesis: Study on the Attenuation Mechanisms in Stratified Flows")
    print("="*80)
    
    # Load main dataset
    print("\n1. MAIN DATASET ANALYSIS")
    print("-" * 40)
    
    with open('stratified_flow_acoustics_dataset.json', 'r') as f:
        main_dataset = json.load(f)
    
    print(f"Dataset file size: {os.path.getsize('stratified_flow_acoustics_dataset.json') / (1024*1024):.2f} MB")
    print(f"Number of experiments: {len(main_dataset['flow_regime_data'])}")
    print(f"Generated on: {main_dataset['metadata']['generated_date']}")
    
    # Flow regime statistics
    flow_df = pd.DataFrame(main_dataset['flow_regime_data'])
    print(f"\nFlow Regime Statistics:")
    print(f"  Void fraction range: {flow_df['void_fraction'].min():.3f} - {flow_df['void_fraction'].max():.3f}")
    print(f"  Gas velocity range: {flow_df['superficial_gas_velocity'].min():.3f} - {flow_df['superficial_gas_velocity'].max():.3f} m/s")
    print(f"  Liquid velocity range: {flow_df['superficial_liquid_velocity'].min():.3f} - {flow_df['superficial_liquid_velocity'].max():.3f} m/s")
    print(f"  Temperature range: {flow_df['temperature'].min():.1f} - {flow_df['temperature'].max():.1f} °C")
    print(f"  Pressure range: {flow_df['pressure'].min()/1000:.1f} - {flow_df['pressure'].max()/1000:.1f} kPa")
    
    print(f"\nFlow Pattern Distribution:")
    pattern_counts = flow_df['flow_pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        percentage = (count / len(flow_df)) * 100
        print(f"  {pattern}: {count} cases ({percentage:.1f}%)")
    
    # Acoustic data statistics
    print(f"\nAcoustic Data Statistics:")
    snr_values = [exp['snr_db'] for exp in main_dataset['acoustic_transmission_data']]
    print(f"  SNR range: {min(snr_values):.1f} - {max(snr_values):.1f} dB")
    print(f"  Average SNR: {np.mean(snr_values):.1f} ± {np.std(snr_values):.1f} dB")
    print(f"  Signal sampling rate: 1000 Hz")
    print(f"  Signal duration: 1.0 seconds")
    print(f"  Frequency range: 10 Hz - 10 kHz")
    
    # Attenuation statistics
    atten_df = pd.DataFrame(main_dataset['attenuation_metrics_data'])
    print(f"\nAttenuation Statistics:")
    print(f"  Average attenuation coefficient: {atten_df['average_attenuation'].mean():.4f}")
    print(f"  Attenuation range: {atten_df['average_attenuation'].min():.4f} - {atten_df['average_attenuation'].max():.4f}")
    print(f"  Frequency resolution: 100 points (log scale)")
    
    # Fluid properties statistics
    fluid_df = pd.DataFrame(main_dataset['fluid_properties_data'])
    print(f"\nFluid Properties Statistics:")
    print(f"  Liquid density range: {fluid_df['liquid_density'].min():.1f} - {fluid_df['liquid_density'].max():.1f} kg/m³")
    print(f"  Gas density range: {fluid_df['gas_density'].min():.3f} - {fluid_df['gas_density'].max():.3f} kg/m³")
    print(f"  Liquid viscosity range: {fluid_df['liquid_viscosity'].min():.2e} - {fluid_df['liquid_viscosity'].max():.2e} Pa·s")
    print(f"  Gas viscosity range: {fluid_df['gas_viscosity'].min():.2e} - {fluid_df['gas_viscosity'].max():.2e} Pa·s")
    
    # Turbulence statistics
    turb_df = pd.DataFrame(main_dataset['turbulence_data'])
    print(f"\nTurbulence Statistics:")
    print(f"  Gas TKE range: {turb_df['turbulent_kinetic_energy_gas'].min():.2e} - {turb_df['turbulent_kinetic_energy_gas'].max():.2e} m²/s²")
    print(f"  Liquid TKE range: {turb_df['turbulent_kinetic_energy_liquid'].min():.2e} - {turb_df['turbulent_kinetic_energy_liquid'].max():.2e} m²/s²")
    print(f"  Interface shear stress range: {turb_df['shear_stress_interface'].min():.2e} - {turb_df['shear_stress_interface'].max():.2e} Pa")
    
    # Load additional scenarios
    print(f"\n2. ADDITIONAL SCENARIOS DATASET")
    print("-" * 40)
    
    with open('additional_scenarios_dataset.json', 'r') as f:
        additional_dataset = json.load(f)
    
    print(f"Additional scenarios file size: {os.path.getsize('additional_scenarios_dataset.json') / (1024*1024):.2f} MB")
    print(f"Total additional scenarios: {len(additional_dataset['scenarios'])}")
    
    # Count scenario types
    scenario_types = {}
    for scenario in additional_dataset['scenarios']:
        scenario_type = scenario['scenario_type']
        scenario_types[scenario_type] = scenario_types.get(scenario_type, 0) + 1
    
    print(f"\nAdditional Scenario Types:")
    for scenario_type, count in scenario_types.items():
        print(f"  {scenario_type}: {count} scenarios")
    
    # CSV exports summary
    print(f"\n3. CSV EXPORTS SUMMARY")
    print("-" * 40)
    
    csv_files = [
        'flow_regime_data.csv',
        'fluid_properties_data.csv',
        'attenuation_metrics_data.csv',
        'turbulence_data.csv',
        'acoustic_transmission_metadata.csv',
        'comprehensive_summary.csv'
    ]
    
    total_csv_size = 0
    for csv_file in csv_files:
        file_path = f'csv_exports/{csv_file}'
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            total_csv_size += file_size
            print(f"  {csv_file}: {file_size / 1024:.1f} KB")
    
    print(f"  Total CSV size: {total_csv_size / (1024*1024):.2f} MB")
    
    # Generated plots summary
    print(f"\n4. GENERATED ANALYSIS PLOTS")
    print("-" * 40)
    
    plot_files = [
        'flow_regime_correlations.png',
        'attenuation_analysis.png',
        'frequency_dependent_attenuation.png',
        'acoustic_signal_analysis.png',
        'turbulence_analysis.png'
    ]
    
    total_plot_size = 0
    for plot_file in plot_files:
        if os.path.exists(plot_file):
            file_size = os.path.getsize(plot_file)
            total_plot_size += file_size
            print(f"  {plot_file}: {file_size / 1024:.1f} KB")
    
    print(f"  Total plot size: {total_plot_size / (1024*1024):.2f} MB")
    
    # Dataset validation
    print(f"\n5. DATASET VALIDATION")
    print("-" * 40)
    
    # Check for missing values
    missing_values = flow_df.isnull().sum().sum()
    print(f"  Missing values in flow data: {missing_values}")
    
    # Check data ranges
    print(f"  Data range validation:")
    print(f"    Void fraction in [0,1]: {flow_df['void_fraction'].between(0, 1).all()}")
    print(f"    Velocities positive: {(flow_df['superficial_gas_velocity'] >= 0).all() and (flow_df['superficial_liquid_velocity'] >= 0).all()}")
    print(f"    Temperature reasonable: {flow_df['temperature'].between(0, 100).all()}")
    print(f"    Pressure positive: {(flow_df['pressure'] > 0).all()}")
    
    # Check acoustic data consistency
    acoustic_consistency = len(main_dataset['acoustic_transmission_data']) == len(main_dataset['flow_regime_data'])
    print(f"    Acoustic data consistency: {acoustic_consistency}")
    
    # Research applications
    print(f"\n6. RESEARCH APPLICATIONS")
    print("-" * 40)
    print("This dataset is suitable for:")
    print("  • Machine learning model training for attenuation prediction")
    print("  • Computational fluid dynamics validation")
    print("  • Acoustic modeling algorithm development")
    print("  • Flow regime identification studies")
    print("  • PhD thesis research on multiphase flow acoustics")
    print("  • Boundary case analysis and extreme condition studies")
    print("  • Frequency-dependent attenuation mechanism analysis")
    print("  • Temperature and pressure effect studies")
    
    # File structure
    print(f"\n7. COMPLETE FILE STRUCTURE")
    print("-" * 40)
    print("Generated files:")
    print("  Main Dataset:")
    print("    • stratified_flow_acoustics_dataset.json (75.1 MB)")
    print("    • additional_scenarios_dataset.json (0.2 MB)")
    print("  Analysis Scripts:")
    print("    • stratified_flow_acoustics_dataset.py")
    print("    • analyze_stratified_flow_data.py")
    print("    • export_dataset_to_csv.py")
    print("    • generate_additional_scenarios.py")
    print("    • dataset_summary.py")
    print("  CSV Exports:")
    for csv_file in csv_files:
        print(f"    • csv_exports/{csv_file}")
    print("  Analysis Plots:")
    for plot_file in plot_files:
        print(f"    • {plot_file}")
    print("  Documentation:")
    print("    • README.md")
    print("    • requirements.txt")
    
    # Final summary
    print(f"\n" + "="*80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total dataset size: {(os.path.getsize('stratified_flow_acoustics_dataset.json') + os.path.getsize('additional_scenarios_dataset.json')) / (1024*1024):.2f} MB")
    print(f"Total experiments: {len(main_dataset['flow_regime_data'])}")
    print(f"Additional scenarios: {len(additional_dataset['scenarios'])}")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    generate_dataset_summary()