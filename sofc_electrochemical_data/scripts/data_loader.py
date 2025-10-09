#!/usr/bin/env python3
"""
SOFC Data Loader and Preprocessor
Utility functions for loading and preprocessing SOFC electrochemical data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

class SOFCDataLoader:
    """Class for loading and managing SOFC electrochemical data"""
    
    def __init__(self, base_path='..'):
        """Initialize data loader with base path"""
        self.base_path = Path(base_path)
        self.iv_data = {}
        self.eis_data = {}
        self.overpotential_data = {}
        
    def load_all_data(self):
        """Load all available datasets"""
        self.load_iv_curves()
        self.load_eis_data()
        self.load_overpotentials()
        return self
    
    def load_iv_curves(self):
        """Load IV curve data for all temperatures"""
        iv_path = self.base_path / 'iv_curves'
        for file in iv_path.glob('*.csv'):
            temp = file.stem.split('_')[-1].replace('C', '°C')
            self.iv_data[temp] = pd.read_csv(file)
        print(f"Loaded {len(self.iv_data)} IV curve datasets")
        return self.iv_data
    
    def load_eis_data(self):
        """Load EIS data for all conditions"""
        eis_path = self.base_path / 'eis_data'
        for file in eis_path.glob('*.csv'):
            condition = file.stem
            self.eis_data[condition] = pd.read_csv(file)
        print(f"Loaded {len(self.eis_data)} EIS datasets")
        return self.eis_data
    
    def load_overpotentials(self):
        """Load overpotential data for all temperatures"""
        overpotential_path = self.base_path / 'overpotentials'
        for file in overpotential_path.glob('*.csv'):
            temp = file.stem.split('_')[-1].replace('C', '°C')
            self.overpotential_data[temp] = pd.read_csv(file)
        print(f"Loaded {len(self.overpotential_data)} overpotential datasets")
        return self.overpotential_data
    
    def get_performance_at_current(self, current_density, temperature='800°C'):
        """Get performance parameters at specific current density"""
        if temperature not in self.iv_data:
            raise ValueError(f"Temperature {temperature} not available")
        
        data = self.iv_data[temperature]
        idx = np.argmin(np.abs(data['Current_Density_A_cm2'] - current_density))
        
        return {
            'Current Density': data.iloc[idx]['Current_Density_A_cm2'],
            'Voltage': data.iloc[idx]['Voltage_V'],
            'Power Density': data.iloc[idx]['Power_Density_W_cm2'],
            'Fuel Utilization': data.iloc[idx]['Fuel_Utilization_%'],
            'Air Utilization': data.iloc[idx]['Air_Utilization_%']
        }
    
    def get_overpotentials_at_current(self, current_density, temperature='800°C'):
        """Get overpotentials at specific current density"""
        if temperature not in self.overpotential_data:
            raise ValueError(f"Temperature {temperature} not available")
        
        data = self.overpotential_data[temperature]
        idx = np.argmin(np.abs(data['Current_Density_A_cm2'] - current_density))
        
        return {
            'Anode Overpotential': data.iloc[idx]['Anode_Overpotential_mV'],
            'Cathode Overpotential': data.iloc[idx]['Cathode_Overpotential_mV'],
            'Ohmic Overpotential': data.iloc[idx]['Ohmic_Overpotential_mV'],
            'Total Overpotential': data.iloc[idx]['Total_Overpotential_mV'],
            'Ni Oxidation Risk': data.iloc[idx]['Ni_Oxidation_Risk'],
            'Local pO2': data.iloc[idx]['Local_pO2_atm'],
            'Stress': data.iloc[idx]['Stress_MPa']
        }
    
    def calculate_efficiency(self, temperature='800°C'):
        """Calculate voltage and fuel efficiency"""
        if temperature not in self.iv_data:
            raise ValueError(f"Temperature {temperature} not available")
        
        data = self.iv_data[temperature]
        
        # Thermodynamic voltage (HHV basis for H2)
        V_th = 1.23  # V at 25°C
        
        # Voltage efficiency
        voltage_eff = data['Voltage_V'] / V_th * 100
        
        # Add to dataframe
        data['Voltage_Efficiency_%'] = voltage_eff
        
        # Fuel efficiency (already have utilization, calculate overall)
        data['Overall_Efficiency_%'] = voltage_eff * data['Fuel_Utilization_%'] / 100
        
        return data
    
    def interpolate_data(self, current_densities, temperature='800°C', dataset='iv'):
        """Interpolate data at specific current densities"""
        if dataset == 'iv':
            source_data = self.iv_data.get(temperature)
        elif dataset == 'overpotential':
            source_data = self.overpotential_data.get(temperature)
        else:
            raise ValueError("Dataset must be 'iv' or 'overpotential'")
        
        if source_data is None:
            raise ValueError(f"No data available for {temperature}")
        
        # Create interpolated dataframe
        interpolated = pd.DataFrame()
        interpolated['Current_Density_A_cm2'] = current_densities
        
        # Interpolate each column
        for col in source_data.columns:
            if col != 'Current_Density_A_cm2':
                if source_data[col].dtype in [np.float64, np.int64]:
                    interpolated[col] = np.interp(
                        current_densities,
                        source_data['Current_Density_A_cm2'],
                        source_data[col]
                    )
        
        return interpolated
    
    def export_to_json(self, output_file='sofc_data.json'):
        """Export all data to JSON format"""
        all_data = {
            'iv_curves': {k: v.to_dict('records') for k, v in self.iv_data.items()},
            'eis_data': {k: v.to_dict('records') for k, v in self.eis_data.items()},
            'overpotentials': {k: v.to_dict('records') for k, v in self.overpotential_data.items()}
        }
        
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        
        print(f"Data exported to {output_file}")
        return output_file
    
    def get_safe_operating_boundary(self, risk_threshold='Medium'):
        """Get safe operating current limits for each temperature"""
        risk_levels = ['Low', 'Medium', 'High', 'Very High']
        if risk_threshold not in risk_levels:
            raise ValueError(f"Risk threshold must be one of {risk_levels}")
        
        max_threshold_idx = risk_levels.index(risk_threshold)
        safe_limits = {}
        
        for temp, data in self.overpotential_data.items():
            # Find maximum current where risk is at or below threshold
            safe_data = data[data['Ni_Oxidation_Risk'].isin(risk_levels[:max_threshold_idx+1])]
            if not safe_data.empty:
                safe_limits[temp] = {
                    'Max_Current_Density': safe_data['Current_Density_A_cm2'].max(),
                    'Max_Stress': safe_data['Stress_MPa'].max(),
                    'Max_pO2': safe_data['Local_pO2_atm'].max()
                }
        
        return safe_limits


def example_usage():
    """Example of how to use the data loader"""
    # Initialize loader
    loader = SOFCDataLoader()
    
    # Load all data
    loader.load_all_data()
    
    # Get performance at specific point
    print("\nPerformance at 0.5 A/cm² and 800°C:")
    perf = loader.get_performance_at_current(0.5, '800°C')
    for key, value in perf.items():
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Get overpotentials
    print("\nOverpotentials at 0.5 A/cm² and 800°C:")
    overpot = loader.get_overpotentials_at_current(0.5, '800°C')
    for key, value in overpot.items():
        if isinstance(value, float):
            if 'pO2' in key:
                print(f"  {key}: {value:.2e}")
            else:
                print(f"  {key}: {value:.1f}")
        else:
            print(f"  {key}: {value}")
    
    # Get safe operating boundaries
    print("\nSafe Operating Boundaries (Risk ≤ Medium):")
    safe_limits = loader.get_safe_operating_boundary('Medium')
    for temp, limits in safe_limits.items():
        print(f"  {temp}:")
        for key, value in limits.items():
            if 'pO2' in key:
                print(f"    {key}: {value:.2e}")
            else:
                print(f"    {key}: {value:.3f}")
    
    # Calculate efficiency
    print("\nCalculating efficiency for 800°C...")
    eff_data = loader.calculate_efficiency('800°C')
    print(f"  Max overall efficiency: {eff_data['Overall_Efficiency_%'].max():.1f}%")
    
    # Export to JSON
    # loader.export_to_json('sofc_complete_dataset.json')


if __name__ == "__main__":
    example_usage()