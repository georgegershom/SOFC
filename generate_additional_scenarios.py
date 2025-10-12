#!/usr/bin/env python3
"""
Generate Additional Experimental Scenarios
Creates specialized test cases for specific research questions.
"""

import numpy as np
import pandas as pd
import json
from stratified_flow_acoustics_dataset import StratifiedFlowAcousticsDataset

class AdditionalScenarioGenerator:
    """Generate additional experimental scenarios for specific research questions."""
    
    def __init__(self):
        self.base_generator = StratifiedFlowAcousticsDataset(n_points=100)
    
    def generate_extreme_conditions(self):
        """Generate extreme flow conditions for boundary case analysis."""
        print("Generating extreme conditions dataset...")
        
        # Extreme void fractions
        extreme_void = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
        
        # Extreme velocities
        extreme_gas_vel = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0])
        extreme_liquid_vel = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Generate all combinations
        scenarios = []
        for vf in extreme_void:
            for ug in extreme_gas_vel:
                for ul in extreme_liquid_vel:
                    scenarios.append({
                        'void_fraction': vf,
                        'superficial_gas_velocity': ug,
                        'superficial_liquid_velocity': ul,
                        'scenario_type': 'extreme_conditions'
                    })
        
        return scenarios
    
    def generate_frequency_sweep(self):
        """Generate frequency sweep experiments."""
        print("Generating frequency sweep dataset...")
        
        # Fixed flow conditions
        base_conditions = [
            {'void_fraction': 0.3, 'superficial_gas_velocity': 1.0, 'superficial_liquid_velocity': 0.5},
            {'void_fraction': 0.5, 'superficial_gas_velocity': 2.0, 'superficial_liquid_velocity': 1.0},
            {'void_fraction': 0.7, 'superficial_gas_velocity': 3.0, 'superficial_liquid_velocity': 1.5}
        ]
        
        # Frequency ranges
        freq_ranges = [
            np.logspace(1, 2, 20),  # 10-100 Hz
            np.logspace(2, 3, 20),  # 100-1000 Hz
            np.logspace(3, 4, 20),  # 1000-10000 Hz
        ]
        
        scenarios = []
        for condition in base_conditions:
            for freq_range in freq_ranges:
                for freq in freq_range:
                    scenarios.append({
                        **condition,
                        'frequency': freq,
                        'scenario_type': 'frequency_sweep'
                    })
        
        return scenarios
    
    def generate_temperature_pressure_matrix(self):
        """Generate temperature-pressure matrix experiments."""
        print("Generating temperature-pressure matrix...")
        
        # Temperature range (extended)
        temperatures = np.linspace(5, 50, 10)  # 5-50°C
        
        # Pressure range (extended)
        pressures = np.array([101325, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])  # 1-10 bar
        
        # Fixed flow conditions
        base_flow = {'void_fraction': 0.5, 'superficial_gas_velocity': 2.0, 'superficial_liquid_velocity': 1.0}
        
        scenarios = []
        for temp in temperatures:
            for press in pressures:
                scenarios.append({
                    **base_flow,
                    'temperature': temp,
                    'pressure': press,
                    'scenario_type': 'temperature_pressure_matrix'
                })
        
        return scenarios
    
    def generate_interface_roughness_study(self):
        """Generate interface roughness parameter study."""
        print("Generating interface roughness study...")
        
        # Interface roughness parameters
        roughness_amplitudes = np.linspace(0.001, 0.01, 10)  # 1-10 mm
        roughness_wavelengths = np.linspace(0.01, 0.1, 10)   # 1-10 cm
        
        # Base flow conditions
        base_flow = {'void_fraction': 0.4, 'superficial_gas_velocity': 1.5, 'superficial_liquid_velocity': 0.8}
        
        scenarios = []
        for amp in roughness_amplitudes:
            for wl in roughness_wavelengths:
                scenarios.append({
                    **base_flow,
                    'interface_roughness_amplitude': amp,
                    'interface_roughness_wavelength': wl,
                    'scenario_type': 'interface_roughness_study'
                })
        
        return scenarios
    
    def generate_complete_additional_dataset(self):
        """Generate complete additional scenarios dataset."""
        print("Generating complete additional scenarios dataset...")
        print("="*60)
        
        # Generate all scenario types
        extreme_scenarios = self.generate_extreme_conditions()
        freq_scenarios = self.generate_frequency_sweep()
        temp_press_scenarios = self.generate_temperature_pressure_matrix()
        roughness_scenarios = self.generate_interface_roughness_study()
        
        # Combine all scenarios
        all_scenarios = extreme_scenarios + freq_scenarios + temp_press_scenarios + roughness_scenarios
        
        print(f"Generated {len(all_scenarios)} additional scenarios:")
        print(f"- Extreme conditions: {len(extreme_scenarios)}")
        print(f"- Frequency sweeps: {len(freq_scenarios)}")
        print(f"- Temperature-pressure matrix: {len(temp_press_scenarios)}")
        print(f"- Interface roughness study: {len(roughness_scenarios)}")
        
        # Create comprehensive dataset
        additional_dataset = {
            'metadata': {
                'title': 'Additional Stratified Flow Acoustics Scenarios',
                'description': 'Specialized experimental scenarios for boundary case analysis',
                'total_scenarios': len(all_scenarios),
                'scenario_types': ['extreme_conditions', 'frequency_sweep', 'temperature_pressure_matrix', 'interface_roughness_study']
            },
            'scenarios': all_scenarios
        }
        
        return additional_dataset
    
    def save_additional_dataset(self, dataset, filename='additional_scenarios_dataset.json'):
        """Save additional scenarios dataset."""
        print(f"Saving additional scenarios to {filename}...")
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert the dataset
        dataset_serializable = convert_numpy_types(dataset)
        
        with open(filename, 'w') as f:
            json.dump(dataset_serializable, f, indent=2)
        
        print(f"Additional scenarios saved successfully!")
        print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")

def main():
    """Main function to generate additional scenarios."""
    print("Additional Stratified Flow Acoustics Scenarios Generator")
    print("="*60)
    
    generator = AdditionalScenarioGenerator()
    dataset = generator.generate_complete_additional_dataset()
    generator.save_additional_dataset(dataset)
    
    print("\nAdditional scenarios generation completed!")
    print("This dataset includes specialized test cases for:")
    print("- Extreme flow conditions")
    print("- Frequency sweep analysis")
    print("- Temperature-pressure effects")
    print("- Interface roughness studies")

if __name__ == "__main__":
    import os
    main()