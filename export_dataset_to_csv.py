#!/usr/bin/env python3
"""
Export Stratified Flow Acoustics Dataset to CSV Format
Converts the JSON dataset to multiple CSV files for easier analysis.
"""

import json
import pandas as pd
import numpy as np
import os

def export_dataset_to_csv(json_file='stratified_flow_acoustics_dataset.json'):
    """Export the JSON dataset to multiple CSV files."""
    
    print("Loading dataset from JSON...")
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    # Create output directory
    output_dir = 'csv_exports'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Exporting flow regime data...")
    flow_df = pd.DataFrame(dataset['flow_regime_data'])
    flow_df.to_csv(f'{output_dir}/flow_regime_data.csv', index=False)
    
    print("Exporting fluid properties data...")
    fluid_df = pd.DataFrame(dataset['fluid_properties_data'])
    fluid_df.to_csv(f'{output_dir}/fluid_properties_data.csv', index=False)
    
    print("Exporting attenuation metrics data...")
    atten_df = pd.DataFrame(dataset['attenuation_metrics_data'])
    atten_df.to_csv(f'{output_dir}/attenuation_metrics_data.csv', index=False)
    
    print("Exporting turbulence data...")
    turb_df = pd.DataFrame(dataset['turbulence_data'])
    turb_df.to_csv(f'{output_dir}/turbulence_data.csv', index=False)
    
    # Export acoustic data (simplified - just metadata)
    print("Exporting acoustic transmission metadata...")
    acoustic_metadata = []
    for i, exp in enumerate(dataset['acoustic_transmission_data']):
        acoustic_metadata.append({
            'experiment_id': exp['experiment_id'],
            'snr_db': exp['snr_db'],
            'attenuation_coefficient': exp['attenuation_coefficient']
        })
    
    acoustic_df = pd.DataFrame(acoustic_metadata)
    acoustic_df.to_csv(f'{output_dir}/acoustic_transmission_metadata.csv', index=False)
    
    # Create a comprehensive summary file
    print("Creating comprehensive summary...")
    summary_data = []
    for i in range(len(flow_df)):
        summary_data.append({
            'experiment_id': i,
            'void_fraction': flow_df.iloc[i]['void_fraction'],
            'superficial_gas_velocity': flow_df.iloc[i]['superficial_gas_velocity'],
            'superficial_liquid_velocity': flow_df.iloc[i]['superficial_liquid_velocity'],
            'flow_pattern': flow_df.iloc[i]['flow_pattern'],
            'interface_height': flow_df.iloc[i]['interface_height'],
            'wave_amplitude': flow_df.iloc[i]['wave_amplitude'],
            'temperature': flow_df.iloc[i]['temperature'],
            'pressure': flow_df.iloc[i]['pressure'],
            'liquid_density': fluid_df.iloc[i]['liquid_density'],
            'gas_density': fluid_df.iloc[i]['gas_density'],
            'liquid_viscosity': fluid_df.iloc[i]['liquid_viscosity'],
            'gas_viscosity': fluid_df.iloc[i]['gas_viscosity'],
            'snr_db': acoustic_df.iloc[i]['snr_db'],
            'average_attenuation': atten_df.iloc[i]['average_attenuation'],
            'turbulent_kinetic_energy_gas': turb_df.iloc[i]['turbulent_kinetic_energy_gas'],
            'turbulent_kinetic_energy_liquid': turb_df.iloc[i]['turbulent_kinetic_energy_liquid'],
            'shear_stress_interface': turb_df.iloc[i]['shear_stress_interface']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f'{output_dir}/comprehensive_summary.csv', index=False)
    
    print(f"\nExport completed successfully!")
    print(f"Files saved in '{output_dir}/' directory:")
    print("- flow_regime_data.csv")
    print("- fluid_properties_data.csv") 
    print("- attenuation_metrics_data.csv")
    print("- turbulence_data.csv")
    print("- acoustic_transmission_metadata.csv")
    print("- comprehensive_summary.csv")
    
    # Print basic statistics
    print(f"\nDataset Statistics:")
    print(f"Total experiments: {len(summary_df)}")
    print(f"Void fraction range: {summary_df['void_fraction'].min():.3f} - {summary_df['void_fraction'].max():.3f}")
    print(f"Gas velocity range: {summary_df['superficial_gas_velocity'].min():.3f} - {summary_df['superficial_gas_velocity'].max():.3f} m/s")
    print(f"Liquid velocity range: {summary_df['superficial_liquid_velocity'].min():.3f} - {summary_df['superficial_liquid_velocity'].max():.3f} m/s")
    print(f"Temperature range: {summary_df['temperature'].min():.1f} - {summary_df['temperature'].max():.1f} °C")
    print(f"Pressure range: {summary_df['pressure'].min()/1000:.1f} - {summary_df['pressure'].max()/1000:.1f} kPa")
    print(f"SNR range: {summary_df['snr_db'].min():.1f} - {summary_df['snr_db'].max():.1f} dB")
    print(f"Average attenuation: {summary_df['average_attenuation'].mean():.4f}")

if __name__ == "__main__":
    export_dataset_to_csv()