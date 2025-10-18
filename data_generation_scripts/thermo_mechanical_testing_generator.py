#!/usr/bin/env python3
"""
Thermo-Mechanical Testing Data Generator for Fire-Resistant Rubberized Concrete
Generates comprehensive mechanical and thermal testing data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import json
from datetime import datetime
import os

class ThermoMechanicalTestingGenerator:
    def __init__(self, seed=42):
        """Initialize the generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.testing_data = {}
        
    def generate_mechanical_properties_data(self, mix_designs, n_specimens=6):
        """Generate mechanical properties data for all mixes"""
        print("Generating mechanical properties data...")
        
        # Test conditions
        temperatures = [20, 100, 200, 300, 400, 500, 600, 700, 800]  # °C
        ages = [7, 14, 28, 56, 90, 180]  # days
        loading_rates = [0.25, 0.5, 1.0]  # MPa/s
        heating_rates = [5, 10, 20]  # °C/min
        
        mechanical_data = {}
        
        for mix_id, mix_design in mix_designs.items():
            print(f"Processing mix {mix_id}...")
            mix_data = {
                'compressive_strength': {},
                'tensile_strength': {},
                'flexural_strength': {},
                'elastic_modulus': {},
                'poissons_ratio': {}
            }
            
            # Generate data for each test condition
            for age in ages:
                for temp in temperatures:
                    for loading_rate in loading_rates:
                        for heating_rate in heating_rates:
                            condition_key = f"age_{age}_temp_{temp}_loading_{loading_rate}_heating_{heating_rate}"
                            
                            # Compressive strength
                            mix_data['compressive_strength'][condition_key] = self._generate_compressive_strength(
                                mix_design, age, temp, loading_rate, heating_rate, n_specimens
                            )
                            
                            # Tensile strength
                            mix_data['tensile_strength'][condition_key] = self._generate_tensile_strength(
                                mix_design, age, temp, loading_rate, heating_rate, n_specimens
                            )
                            
                            # Flexural strength
                            mix_data['flexural_strength'][condition_key] = self._generate_flexural_strength(
                                mix_design, age, temp, loading_rate, heating_rate, n_specimens
                            )
                            
                            # Elastic modulus
                            mix_data['elastic_modulus'][condition_key] = self._generate_elastic_modulus(
                                mix_design, age, temp, loading_rate, heating_rate, n_specimens
                            )
                            
                            # Poisson's ratio
                            mix_data['poissons_ratio'][condition_key] = self._generate_poissons_ratio(
                                mix_design, age, temp, loading_rate, heating_rate, n_specimens
                            )
            
            mechanical_data[mix_id] = mix_data
        
        self.testing_data['mechanical_properties'] = mechanical_data
        return mechanical_data
    
    def _generate_compressive_strength(self, mix_design, age, temperature, loading_rate, heating_rate, n_specimens):
        """Generate compressive strength data"""
        # Base strength at 28 days, 20°C
        base_strength = 40  # MPa
        
        # Age factor (strength development)
        age_factor = self._calculate_age_factor(age)
        
        # Temperature factor (strength degradation)
        temp_factor = self._calculate_temperature_factor(temperature)
        
        # Mix design factors
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 20  # Higher w/c = lower strength
        rubber_factor = -mix_design['rubber_fine_content'] * 0.8  # Rubber reduces strength
        cement_factor = (mix_design['cement_content'] - 400) * 0.05  # More cement = higher strength
        fiber_factor = mix_design['fiber_content'] * 2  # Fibers increase strength
        
        # Loading rate factor
        loading_factor = (loading_rate - 0.5) * 5  # Higher loading rate = higher strength
        
        # Heating rate factor
        heating_factor = -(heating_rate - 10) * 0.5  # Higher heating rate = lower strength
        
        # Calculate strength
        strength = (base_strength + wc_factor + rubber_factor + cement_factor + fiber_factor + 
                   loading_factor + heating_factor) * age_factor * temp_factor
        
        # Add random variation for each specimen
        strengths = []
        for _ in range(n_specimens):
            specimen_strength = strength + np.random.normal(0, strength * 0.08)  # 8% COV
            strengths.append(max(5, specimen_strength))  # Minimum 5 MPa
        
        return {
            'values': strengths,
            'mean': np.mean(strengths),
            'std': np.std(strengths),
            'cov': np.std(strengths) / np.mean(strengths),
            'min': np.min(strengths),
            'max': np.max(strengths)
        }
    
    def _generate_tensile_strength(self, mix_design, age, temperature, loading_rate, heating_rate, n_specimens):
        """Generate tensile strength data"""
        # Base tensile strength (typically 10% of compressive)
        base_tensile = 4  # MPa
        
        # Age factor
        age_factor = self._calculate_age_factor(age)
        
        # Temperature factor
        temp_factor = self._calculate_temperature_factor(temperature)
        
        # Mix design factors
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 2
        rubber_factor = -mix_design['rubber_fine_content'] * 0.1
        cement_factor = (mix_design['cement_content'] - 400) * 0.005
        fiber_factor = mix_design['fiber_content'] * 0.5  # Fibers significantly improve tensile
        
        # Loading rate factor
        loading_factor = (loading_rate - 0.5) * 0.5
        
        # Heating rate factor
        heating_factor = -(heating_rate - 10) * 0.05
        
        # Calculate tensile strength
        tensile = (base_tensile + wc_factor + rubber_factor + cement_factor + fiber_factor + 
                  loading_factor + heating_factor) * age_factor * temp_factor
        
        # Add random variation
        tensile_strengths = []
        for _ in range(n_specimens):
            specimen_tensile = tensile + np.random.normal(0, tensile * 0.12)  # 12% COV
            tensile_strengths.append(max(1, specimen_tensile))  # Minimum 1 MPa
        
        return {
            'values': tensile_strengths,
            'mean': np.mean(tensile_strengths),
            'std': np.std(tensile_strengths),
            'cov': np.std(tensile_strengths) / np.mean(tensile_strengths),
            'min': np.min(tensile_strengths),
            'max': np.max(tensile_strengths)
        }
    
    def _generate_flexural_strength(self, mix_design, age, temperature, loading_rate, heating_rate, n_specimens):
        """Generate flexural strength data"""
        # Base flexural strength (typically 15% of compressive)
        base_flexural = 6  # MPa
        
        # Age factor
        age_factor = self._calculate_age_factor(age)
        
        # Temperature factor
        temp_factor = self._calculate_temperature_factor(temperature)
        
        # Mix design factors
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 3
        rubber_factor = -mix_design['rubber_fine_content'] * 0.15
        cement_factor = (mix_design['cement_content'] - 400) * 0.008
        fiber_factor = mix_design['fiber_content'] * 1.2  # Fibers significantly improve flexural
        
        # Loading rate factor
        loading_factor = (loading_rate - 0.5) * 0.8
        
        # Heating rate factor
        heating_factor = -(heating_rate - 10) * 0.08
        
        # Calculate flexural strength
        flexural = (base_flexural + wc_factor + rubber_factor + cement_factor + fiber_factor + 
                   loading_factor + heating_factor) * age_factor * temp_factor
        
        # Add random variation
        flexural_strengths = []
        for _ in range(n_specimens):
            specimen_flexural = flexural + np.random.normal(0, flexural * 0.10)  # 10% COV
            flexural_strengths.append(max(2, specimen_flexural))  # Minimum 2 MPa
        
        return {
            'values': flexural_strengths,
            'mean': np.mean(flexural_strengths),
            'std': np.std(flexural_strengths),
            'cov': np.std(flexural_strengths) / np.mean(flexural_strengths),
            'min': np.min(flexural_strengths),
            'max': np.max(flexural_strengths)
        }
    
    def _generate_elastic_modulus(self, mix_design, age, temperature, loading_rate, heating_rate, n_specimens):
        """Generate elastic modulus data"""
        # Base elastic modulus
        base_modulus = 30  # GPa
        
        # Age factor
        age_factor = self._calculate_age_factor(age)
        
        # Temperature factor
        temp_factor = self._calculate_temperature_factor(temperature)
        
        # Mix design factors
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 15
        rubber_factor = -mix_design['rubber_fine_content'] * 0.5
        cement_factor = (mix_design['cement_content'] - 400) * 0.03
        fiber_factor = mix_design['fiber_content'] * 1.5
        
        # Loading rate factor
        loading_factor = (loading_rate - 0.5) * 2
        
        # Heating rate factor
        heating_factor = -(heating_rate - 10) * 0.2
        
        # Calculate elastic modulus
        modulus = (base_modulus + wc_factor + rubber_factor + cement_factor + fiber_factor + 
                  loading_factor + heating_factor) * age_factor * temp_factor
        
        # Add random variation
        moduli = []
        for _ in range(n_specimens):
            specimen_modulus = modulus + np.random.normal(0, modulus * 0.06)  # 6% COV
            moduli.append(max(10, specimen_modulus))  # Minimum 10 GPa
        
        return {
            'values': moduli,
            'mean': np.mean(moduli),
            'std': np.std(moduli),
            'cov': np.std(moduli) / np.mean(moduli),
            'min': np.min(moduli),
            'max': np.max(moduli)
        }
    
    def _generate_poissons_ratio(self, mix_design, age, temperature, loading_rate, heating_rate, n_specimens):
        """Generate Poisson's ratio data"""
        # Base Poisson's ratio
        base_poisson = 0.20
        
        # Age factor (slight increase with age)
        age_factor = 1 + (age - 28) * 0.001
        
        # Temperature factor (increases with temperature)
        temp_factor = 1 + (temperature - 20) * 0.0001
        
        # Mix design factors
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 0.05
        rubber_factor = mix_design['rubber_fine_content'] * 0.002
        cement_factor = -(mix_design['cement_content'] - 400) * 0.0001
        fiber_factor = -mix_design['fiber_content'] * 0.01
        
        # Loading rate factor
        loading_factor = -(loading_rate - 0.5) * 0.02
        
        # Heating rate factor
        heating_factor = (heating_rate - 10) * 0.001
        
        # Calculate Poisson's ratio
        poisson = (base_poisson + wc_factor + rubber_factor + cement_factor + fiber_factor + 
                  loading_factor + heating_factor) * age_factor * temp_factor
        
        # Add random variation
        poissons = []
        for _ in range(n_specimens):
            specimen_poisson = poisson + np.random.normal(0, 0.01)  # Small variation
            poissons.append(max(0.15, min(0.25, specimen_poisson)))  # Clamp between 0.15-0.25
        
        return {
            'values': poissons,
            'mean': np.mean(poissons),
            'std': np.std(poissons),
            'cov': np.std(poissons) / np.mean(poissons),
            'min': np.min(poissons),
            'max': np.max(poissons)
        }
    
    def _calculate_age_factor(self, age):
        """Calculate strength development factor based on age"""
        # Typical concrete strength development curve
        if age <= 28:
            return age / 28
        else:
            return 1 + 0.1 * np.log(age / 28)
    
    def _calculate_temperature_factor(self, temperature):
        """Calculate strength retention factor based on temperature"""
        if temperature <= 100:
            return 1.0
        elif temperature <= 300:
            return 1 - 0.2 * (temperature - 100) / 200
        elif temperature <= 500:
            return 0.8 - 0.3 * (temperature - 300) / 200
        elif temperature <= 700:
            return 0.5 - 0.3 * (temperature - 500) / 200
        else:
            return max(0.1, 0.2 - 0.1 * (temperature - 700) / 100)
    
    def generate_thermal_properties_data(self, mix_designs):
        """Generate thermal properties data for all mixes"""
        print("Generating thermal properties data...")
        
        temperatures = np.linspace(20, 800, 100)  # °C
        thermal_data = {}
        
        for mix_id, mix_design in mix_designs.items():
            print(f"Processing thermal properties for mix {mix_id}...")
            
            mix_thermal = {
                'thermal_conductivity': self._generate_thermal_conductivity(mix_design, temperatures),
                'specific_heat': self._generate_specific_heat(mix_design, temperatures),
                'thermal_expansion': self._generate_thermal_expansion(mix_design, temperatures),
                'thermal_diffusivity': self._generate_thermal_diffusivity(mix_design, temperatures)
            }
            
            thermal_data[mix_id] = mix_thermal
        
        self.testing_data['thermal_properties'] = thermal_data
        return thermal_data
    
    def _generate_thermal_conductivity(self, mix_design, temperatures):
        """Generate thermal conductivity data"""
        # Base thermal conductivity
        base_conductivity = 2.0  # W/m·K
        
        # Temperature effect (decreases with temperature)
        temp_factor = 1 - 0.3 * (temperatures - 20) / 780
        
        # Mix design factors
        rubber_factor = -mix_design['rubber_fine_content'] * 0.02  # Rubber reduces conductivity
        cement_factor = (mix_design['cement_content'] - 400) * 0.001
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 0.5
        fiber_factor = mix_design['fiber_content'] * 0.1  # Fibers increase conductivity
        
        # Calculate conductivity
        conductivity = (base_conductivity + rubber_factor + cement_factor + wc_factor + fiber_factor) * temp_factor
        
        # Add random variation
        conductivity += np.random.normal(0, 0.05, len(temperatures))
        
        return {
            'temperature': temperatures.tolist(),
            'conductivity': conductivity.tolist(),
            'mean': np.mean(conductivity),
            'std': np.std(conductivity)
        }
    
    def _generate_specific_heat(self, mix_design, temperatures):
        """Generate specific heat data"""
        # Base specific heat
        base_heat = 0.9  # kJ/kg·K
        
        # Temperature effect (increases with temperature)
        temp_factor = 1 + 0.4 * (temperatures - 20) / 780
        
        # Mix design factors
        rubber_factor = mix_design['rubber_fine_content'] * 0.01  # Rubber increases specific heat
        cement_factor = (mix_design['cement_content'] - 400) * 0.0005
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 0.2
        fiber_factor = mix_design['fiber_content'] * 0.05
        
        # Calculate specific heat
        specific_heat = (base_heat + rubber_factor + cement_factor + wc_factor + fiber_factor) * temp_factor
        
        # Add random variation
        specific_heat += np.random.normal(0, 0.02, len(temperatures))
        
        return {
            'temperature': temperatures.tolist(),
            'specific_heat': specific_heat.tolist(),
            'mean': np.mean(specific_heat),
            'std': np.std(specific_heat)
        }
    
    def _generate_thermal_expansion(self, mix_design, temperatures):
        """Generate thermal expansion coefficient data"""
        # Base thermal expansion coefficient
        base_expansion = 10e-6  # 1/K
        
        # Temperature effect (increases with temperature)
        temp_factor = 1 + 0.5 * (temperatures - 20) / 780
        
        # Mix design factors
        rubber_factor = mix_design['rubber_fine_content'] * 0.5e-6  # Rubber increases expansion
        cement_factor = (mix_design['cement_content'] - 400) * 0.1e-6
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 2e-6
        fiber_factor = -mix_design['fiber_content'] * 0.2e-6  # Fibers reduce expansion
        
        # Calculate expansion coefficient
        expansion = (base_expansion + rubber_factor + cement_factor + wc_factor + fiber_factor) * temp_factor
        
        # Add random variation
        expansion += np.random.normal(0, 0.5e-6, len(temperatures))
        
        return {
            'temperature': temperatures.tolist(),
            'expansion_coefficient': expansion.tolist(),
            'mean': np.mean(expansion),
            'std': np.std(expansion)
        }
    
    def _generate_thermal_diffusivity(self, mix_design, temperatures):
        """Generate thermal diffusivity data"""
        # Thermal diffusivity = thermal_conductivity / (density * specific_heat)
        # This is calculated from the other thermal properties
        
        # Get conductivity and specific heat
        conductivity_data = self._generate_thermal_conductivity(mix_design, temperatures)
        specific_heat_data = self._generate_specific_heat(mix_design, temperatures)
        
        # Calculate density (simplified)
        base_density = 2400  # kg/m³
        rubber_factor = -mix_design['rubber_fine_content'] * 10
        density = base_density + rubber_factor
        
        # Calculate diffusivity
        diffusivity = np.array(conductivity_data['conductivity']) / (density * np.array(specific_heat_data['specific_heat']))
        
        return {
            'temperature': temperatures.tolist(),
            'diffusivity': diffusivity.tolist(),
            'mean': np.mean(diffusivity),
            'std': np.std(diffusivity)
        }
    
    def generate_fire_resistance_data(self, mix_designs):
        """Generate fire resistance testing data"""
        print("Generating fire resistance data...")
        
        fire_durations = [30, 60, 90, 120, 180]  # minutes
        fire_data = {}
        
        for mix_id, mix_design in mix_designs.items():
            print(f"Processing fire resistance for mix {mix_id}...")
            
            mix_fire = {
                'fire_resistance_rating': self._generate_fire_resistance_rating(mix_design, fire_durations),
                'spalling_resistance': self._generate_spalling_resistance(mix_design),
                'smoke_production': self._generate_smoke_production(mix_design),
                'heat_release_rate': self._generate_heat_release_rate(mix_design)
            }
            
            fire_data[mix_id] = mix_fire
        
        self.testing_data['fire_resistance'] = fire_data
        return fire_data
    
    def _generate_fire_resistance_rating(self, mix_design, durations):
        """Generate fire resistance rating data"""
        # Base fire resistance
        base_resistance = 60  # minutes
        
        # Mix design factors
        rubber_factor = -mix_design['rubber_fine_content'] * 2  # Rubber reduces fire resistance
        cement_factor = (mix_design['cement_content'] - 400) * 0.1
        wc_factor = -(mix_design['w_c_ratio'] - 0.45) * 20
        fiber_factor = mix_design['fiber_content'] * 5  # Fibers improve fire resistance
        fire_retardant_factor = mix_design['fire_retardant_content'] * 3
        
        # Calculate resistance for each duration
        resistances = []
        for duration in durations:
            resistance = (base_resistance + rubber_factor + cement_factor + wc_factor + 
                        fiber_factor + fire_retardant_factor) * (duration / 60)
            resistance += np.random.normal(0, 5)  # Add random variation
            resistances.append(max(15, resistance))  # Minimum 15 minutes
        
        return {
            'durations': durations,
            'resistances': resistances,
            'mean': np.mean(resistances),
            'std': np.std(resistances)
        }
    
    def _generate_spalling_resistance(self, mix_design):
        """Generate spalling resistance data"""
        # Base spalling resistance (higher is better)
        base_resistance = 0.8
        
        # Mix design factors
        rubber_factor = mix_design['rubber_fine_content'] * 0.01  # Rubber improves spalling resistance
        cement_factor = (mix_design['cement_content'] - 400) * 0.001
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 0.1
        fiber_factor = mix_design['fiber_content'] * 0.05  # Fibers improve spalling resistance
        
        # Calculate resistance
        resistance = base_resistance + rubber_factor + cement_factor + wc_factor + fiber_factor
        resistance += np.random.normal(0, 0.05)
        
        return {
            'resistance': max(0.3, min(1.0, resistance)),
            'spalling_probability': 1 - resistance
        }
    
    def _generate_smoke_production(self, mix_design):
        """Generate smoke production data"""
        # Base smoke production
        base_smoke = 0.5  # m²/s
        
        # Mix design factors
        rubber_factor = mix_design['rubber_fine_content'] * 0.02  # Rubber increases smoke
        cement_factor = -(mix_design['cement_content'] - 400) * 0.001
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 0.1
        fire_retardant_factor = -mix_design['fire_retardant_content'] * 0.05
        
        # Calculate smoke production
        smoke = base_smoke + rubber_factor + cement_factor + wc_factor + fire_retardant_factor
        smoke += np.random.normal(0, 0.05)
        
        return {
            'smoke_production_rate': max(0.1, smoke),
            'smoke_density': smoke * 2  # Proportional to production rate
        }
    
    def _generate_heat_release_rate(self, mix_design):
        """Generate heat release rate data"""
        # Base heat release rate
        base_hrr = 200  # kW/m²
        
        # Mix design factors
        rubber_factor = mix_design['rubber_fine_content'] * 5  # Rubber increases HRR
        cement_factor = -(mix_design['cement_content'] - 400) * 0.1
        wc_factor = (mix_design['w_c_ratio'] - 0.45) * 50
        fire_retardant_factor = -mix_design['fire_retardant_content'] * 10
        
        # Calculate heat release rate
        hrr = base_hrr + rubber_factor + cement_factor + wc_factor + fire_retardant_factor
        hrr += np.random.normal(0, 20)
        
        return {
            'peak_hrr': max(50, hrr),
            'total_heat_release': hrr * 0.5  # Proportional to peak HRR
        }
    
    def save_data(self, filename='thermo_mechanical_testing_data.json'):
        """Save generated data to JSON file"""
        data_to_save = {
            'testing_data': self.testing_data,
            'generation_date': datetime.now().isoformat(),
            'total_mixes': len(self.testing_data.get('mechanical_properties', {}))
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        serializable_data = convert_numpy(data_to_save)
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Thermo-mechanical testing data saved to {filename}")
    
    def generate_summary_report(self):
        """Generate summary report of the testing data"""
        report = {
            'generation_date': datetime.now().isoformat(),
            'total_mixes_tested': len(self.testing_data.get('mechanical_properties', {})),
            'test_types': list(self.testing_data.keys()),
            'data_points_per_test': {}
        }
        
        for test_type, test_data in self.testing_data.items():
            if isinstance(test_data, dict):
                total_points = sum(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in test_data.values())
                report['data_points_per_test'][test_type] = total_points
        
        return report

def main():
    """Main function to generate thermo-mechanical testing data"""
    print("Starting Thermo-Mechanical Testing Data Generation...")
    print("=" * 60)
    
    # Initialize generator
    generator = ThermoMechanicalTestingGenerator(seed=42)
    
    # Load mix designs (assuming they exist from previous script)
    # In a real scenario, you would load this from the mix design generator
    print("Note: This script requires mix design data from the mix design generator.")
    print("Please run the mix design generator first to create the necessary input data.")
    
    # For demonstration, create a simple mix design
    sample_mix_designs = {
        1: {
            'cement_content': 400,
            'w_c_ratio': 0.45,
            'rubber_fine_content': 15,
            'rubber_coarse_content': 0,
            'fiber_content': 1.0,
            'fire_retardant_content': 4
        }
    }
    
    # Generate mechanical properties data
    generator.generate_mechanical_properties_data(sample_mix_designs)
    
    # Generate thermal properties data
    generator.generate_thermal_properties_data(sample_mix_designs)
    
    # Generate fire resistance data
    generator.generate_fire_resistance_data(sample_mix_designs)
    
    # Save data
    generator.save_data('thermo_mechanical_testing_data.json')
    
    # Generate summary report
    report = generator.generate_summary_report()
    print("\nSummary Report:")
    print(json.dumps(report, indent=2))
    
    print("\nThermo-mechanical testing data generation completed successfully!")

if __name__ == "__main__":
    main()