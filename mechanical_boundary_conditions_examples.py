#!/usr/bin/env python3
"""
Mechanical Boundary Conditions Dataset - Usage Examples

This script demonstrates how to use the mechanical boundary conditions dataset
for SOFC electrolyte fracture analysis. It includes examples for:
1. Loading the dataset
2. Extracting specific parameters
3. Setting up FEA boundary conditions
4. Calculating derived parameters
5. Validating against experimental data

Author: AI Assistant
Date: 2024-01-15
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

class SOFCBoundaryConditions:
    """
    Class for handling SOFC mechanical boundary conditions dataset
    """
    
    def __init__(self, dataset_path: str):
        """Initialize with dataset file path"""
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        self.parameters_df = None
        self.load_parameters_csv()
    
    def load_parameters_csv(self, csv_path: str = "mechanical_boundary_conditions_parameters.csv"):
        """Load parameters from CSV file"""
        try:
            self.parameters_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.parameters_df)} parameters from CSV")
        except FileNotFoundError:
            print(f"CSV file {csv_path} not found. Using JSON data only.")
    
    def get_fixture_data(self, fixture_id: str) -> Dict:
        """Get data for specific fixture type"""
        fixtures = self.data['fixture_types']['data']
        for fixture in fixtures:
            if fixture['fixture_id'] == fixture_id:
                return fixture
        raise ValueError(f"Fixture {fixture_id} not found")
    
    def get_pressure_conditions(self, pressure_id: str) -> Dict:
        """Get data for specific pressure condition"""
        pressures = self.data['stack_pressure_conditions']['data']
        for pressure in pressures:
            if pressure['pressure_id'] == pressure_id:
                return pressure
        raise ValueError(f"Pressure condition {pressure_id} not found")
    
    def get_constraint_data(self, constraint_id: str) -> Dict:
        """Get data for specific constraint type"""
        constraints = self.data['external_constraints']['data']
        for constraint in constraints:
            if constraint['constraint_id'] == constraint_id:
                return constraint
        raise ValueError(f"Constraint {constraint_id} not found")
    
    def get_load_data(self, load_id: str) -> Dict:
        """Get data for specific load type"""
        loads = self.data['applied_loads']['data']
        for load in loads:
            if load['load_id'] == load_id:
                return load
        raise ValueError(f"Load {load_id} not found")
    
    def get_validation_data(self, validation_id: str) -> Dict:
        """Get data for specific validation test"""
        validations = self.data['validation_data']['data']
        for validation in validations:
            if validation['validation_id'] == validation_id:
                return validation
        raise ValueError(f"Validation {validation_id} not found")
    
    def calculate_thermal_stress(self, temperature: float, reference_temp: float = 25.0) -> float:
        """
        Calculate thermal stress due to temperature change
        
        Args:
            temperature: Current temperature (°C)
            reference_temp: Reference temperature (°C)
        
        Returns:
            Thermal stress (MPa)
        """
        # Get material properties
        cte = 10.5e-6  # K^-1 (YSZ)
        youngs_modulus = 200e3  # MPa (at room temperature)
        
        # Calculate thermal strain
        delta_temp = temperature - reference_temp
        thermal_strain = cte * delta_temp
        
        # Calculate thermal stress (assuming constrained expansion)
        thermal_stress = youngs_modulus * thermal_strain
        
        return thermal_stress
    
    def calculate_assembly_pressure_effects(self, pressure: float) -> Dict[str, float]:
        """
        Calculate effects of assembly pressure on SOFC cell
        
        Args:
            pressure: Assembly pressure (MPa)
        
        Returns:
            Dictionary of calculated effects
        """
        # Get assembly pressure data
        assembly_data = self.get_pressure_conditions("PRES_001")
        
        # Calculate contact stress (simplified)
        contact_area = 100e-4  # m^2 (100 cm^2)
        total_force = pressure * 1e6 * contact_area  # N
        
        # Calculate stress concentration at edges
        edge_reduction = 0.2  # 20% reduction at edges
        edge_pressure = pressure * (1 - edge_reduction)
        
        # Calculate stress concentration at corners
        corner_reduction = 0.4  # 40% reduction at corners
        corner_pressure = pressure * (1 - corner_reduction)
        
        return {
            'total_force': total_force,
            'contact_stress': pressure,
            'edge_pressure': edge_pressure,
            'corner_pressure': corner_pressure,
            'pressure_uniformity': f"±{assembly_data['pressure_range']['recommended']*100/2:.1f}%"
        }
    
    def calculate_creep_strain_rate(self, stress: float, temperature: float) -> float:
        """
        Calculate creep strain rate using Norton-Bailey law
        
        Args:
            stress: Applied stress (MPa)
            temperature: Temperature (°C)
        
        Returns:
            Creep strain rate (s^-1)
        """
        # Creep parameters for 8YSZ
        B = 8.5e-12  # s^-1 MPa^-n
        n = 1.8      # stress exponent
        Q = 385e3    # J/mol (activation energy)
        R = 8.314    # J/mol·K (gas constant)
        
        # Convert temperature to Kelvin
        T_K = temperature + 273.15
        
        # Calculate creep strain rate
        creep_rate = B * (stress ** n) * np.exp(-Q / (R * T_K))
        
        return creep_rate
    
    def calculate_fatigue_life(self, stress_range: float, mean_stress: float, 
                             temperature: float = 800.0) -> float:
        """
        Calculate fatigue life using modified Goodman relation
        
        Args:
            stress_range: Stress range (MPa)
            mean_stress: Mean stress (MPa)
            temperature: Temperature (°C)
        
        Returns:
            Fatigue life (cycles)
        """
        # Material properties
        ultimate_strength = 200  # MPa (approximate for YSZ)
        fatigue_strength_coeff = 0.9
        fatigue_strength_exp = -0.1
        
        # Temperature factor (reduces strength at high temperature)
        temp_factor = 1.0 - 0.3 * (temperature - 25) / 775  # Linear reduction
        
        # Modified Goodman relation
        effective_stress_range = stress_range / (1 - mean_stress / ultimate_strength)
        
        # Basquin equation
        fatigue_life = (fatigue_strength_coeff * ultimate_strength * temp_factor / effective_stress_range) ** (1 / fatigue_strength_exp)
        
        return max(1, fatigue_life)  # Minimum 1 cycle
    
    def generate_fea_boundary_conditions(self, load_case: str = "steady_state") -> Dict:
        """
        Generate boundary conditions for FEA simulation
        
        Args:
            load_case: Type of loading case ("steady_state", "thermal_cycling", "assembly")
        
        Returns:
            Dictionary of FEA boundary conditions
        """
        if load_case == "steady_state":
            return {
                "mechanical": {
                    "bottom_support": "Fixed in Z-direction, free in X-Y",
                    "lateral_support": "Symmetry boundary conditions",
                    "top_loading": "Applied pressure: 0.2 MPa",
                    "pressure_distribution": "Uniform with ±5% variation"
                },
                "thermal": {
                    "operating_temperature": "800°C",
                    "convective_htc": "100 W/m²·K",
                    "ambient_temperature": "25°C",
                    "internal_heat_generation": "1500 W/m²"
                },
                "electrical": {
                    "current_density": "0.5 A/cm²",
                    "fuel_utilization": "85%",
                    "air_utilization": "25%"
                }
            }
        
        elif load_case == "thermal_cycling":
            return {
                "mechanical": {
                    "bottom_support": "Fixed in Z-direction, free in X-Y",
                    "lateral_support": "Symmetry boundary conditions",
                    "top_loading": "Applied pressure: 0.15-0.2 MPa (cycling)",
                    "pressure_cycling": "0.1 to 0.2 MPa"
                },
                "thermal": {
                    "temperature_cycle": "25°C to 800°C",
                    "heating_rate": "5°C/min",
                    "cooling_rate": "2°C/min",
                    "dwell_time": "2 hours at 800°C"
                },
                "electrical": {
                    "current_density": "0.5 A/cm² (during operation)",
                    "shutdown_sequence": "Current reduction before cooling"
                }
            }
        
        elif load_case == "assembly":
            return {
                "mechanical": {
                    "bottom_support": "Fixed in Z-direction, free in X-Y",
                    "lateral_support": "Symmetry boundary conditions",
                    "top_loading": "Applied pressure: 0.2 MPa",
                    "loading_rate": "0.01 MPa/s",
                    "dwell_time": "30 minutes"
                },
                "thermal": {
                    "temperature": "25°C (room temperature)",
                    "thermal_expansion": "Considered in material properties"
                },
                "electrical": {
                    "current_density": "0 A/cm² (no electrical loading)"
                }
            }
        
        else:
            raise ValueError(f"Unknown load case: {load_case}")
    
    def plot_pressure_distribution(self, pressure_type: str = "assembly"):
        """
        Plot pressure distribution across SOFC cell
        
        Args:
            pressure_type: Type of pressure ("assembly", "operational", "thermal")
        """
        # Create coordinate grid
        x = np.linspace(0, 100, 50)  # mm
        y = np.linspace(0, 100, 50)  # mm
        X, Y = np.meshgrid(x, y)
        
        if pressure_type == "assembly":
            # Assembly pressure with edge and corner effects
            base_pressure = 0.2  # MPa
            edge_reduction = 0.2
            corner_reduction = 0.4
            
            # Calculate distance from edges
            dist_from_edge = np.minimum(np.minimum(x, 100-x), np.minimum(y, 100-y))
            
            # Calculate pressure reduction factor
            reduction_factor = np.ones_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    if dist_from_edge[i] < 10:  # Edge region
                        reduction_factor[j, i] = 1 - edge_reduction
                    if dist_from_edge[i] < 5:   # Corner region
                        reduction_factor[j, i] = 1 - corner_reduction
            
            pressure = base_pressure * reduction_factor
        
        elif pressure_type == "operational":
            # Operational pressure with slight variations
            base_pressure = 0.2  # MPa
            variation = 0.05 * np.sin(2 * np.pi * X / 50) * np.cos(2 * np.pi * Y / 50)
            pressure = base_pressure + variation
        
        else:
            # Thermal expansion pressure
            base_pressure = 0.35  # MPa
            pressure = base_pressure * np.ones_like(X)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        contour = plt.contourf(X, Y, pressure, levels=20, cmap='viridis')
        plt.colorbar(contour, label='Pressure (MPa)')
        plt.xlabel('X Position (mm)')
        plt.ylabel('Y Position (mm)')
        plt.title(f'{pressure_type.title()} Pressure Distribution')
        plt.axis('equal')
        plt.show()
    
    def validate_against_experimental(self, parameter: str, calculated_value: float) -> Dict:
        """
        Validate calculated parameter against experimental data
        
        Args:
            parameter: Parameter name to validate
            calculated_value: Calculated value
        
        Returns:
            Validation results
        """
        # Get validation data
        validation_data = self.data['validation_data']['data']
        
        for validation in validation_data:
            if parameter in validation['results']:
                exp_value = validation['results'][parameter]
                if isinstance(exp_value, str) and '-' in exp_value:
                    # Handle range values
                    exp_min, exp_max = map(float, exp_value.split('-'))
                    exp_mean = (exp_min + exp_max) / 2
                    exp_uncertainty = (exp_max - exp_min) / 2
                else:
                    exp_mean = float(exp_value)
                    exp_uncertainty = 0.1 * exp_mean  # Assume 10% uncertainty
                
                error = abs(calculated_value - exp_mean) / exp_mean * 100
                
                return {
                    'parameter': parameter,
                    'calculated_value': calculated_value,
                    'experimental_value': exp_mean,
                    'experimental_uncertainty': exp_uncertainty,
                    'error_percent': error,
                    'within_uncertainty': error < exp_uncertainty / exp_mean * 100,
                    'validation_id': validation['validation_id']
                }
        
        return {
            'parameter': parameter,
            'calculated_value': calculated_value,
            'experimental_value': None,
            'error_percent': None,
            'within_uncertainty': None,
            'validation_id': None
        }

def main():
    """Main function demonstrating dataset usage"""
    print("SOFC Mechanical Boundary Conditions Dataset - Usage Examples")
    print("=" * 60)
    
    # Initialize dataset
    dataset = SOFCBoundaryConditions("mechanical_boundary_conditions_dataset.json")
    
    # Example 1: Get fixture data
    print("\n1. Fixture Data Example:")
    fixture_data = dataset.get_fixture_data("FIX_001")
    print(f"Fixture: {fixture_data['name']}")
    print(f"Load Capacity: {fixture_data['mechanical_properties']['load_capacity']} N")
    print(f"Stiffness: {fixture_data['mechanical_properties']['stiffness']} N/mm")
    
    # Example 2: Calculate thermal stress
    print("\n2. Thermal Stress Calculation:")
    temp_stress = dataset.calculate_thermal_stress(800, 25)
    print(f"Thermal stress at 800°C: {temp_stress:.1f} MPa")
    
    # Example 3: Calculate assembly pressure effects
    print("\n3. Assembly Pressure Effects:")
    pressure_effects = dataset.calculate_assembly_pressure_effects(0.2)
    for key, value in pressure_effects.items():
        print(f"{key}: {value}")
    
    # Example 4: Calculate creep strain rate
    print("\n4. Creep Strain Rate Calculation:")
    creep_rate = dataset.calculate_creep_strain_rate(100, 800)
    print(f"Creep strain rate at 100 MPa, 800°C: {creep_rate:.2e} s^-1")
    
    # Example 5: Calculate fatigue life
    print("\n5. Fatigue Life Calculation:")
    fatigue_life = dataset.calculate_fatigue_life(80, 100, 800)
    print(f"Fatigue life (80 MPa range, 100 MPa mean): {fatigue_life:.0f} cycles")
    
    # Example 6: Generate FEA boundary conditions
    print("\n6. FEA Boundary Conditions (Steady State):")
    fea_conditions = dataset.generate_fea_boundary_conditions("steady_state")
    print("Mechanical BCs:")
    for key, value in fea_conditions["mechanical"].items():
        print(f"  {key}: {value}")
    
    # Example 7: Validate against experimental data
    print("\n7. Validation Against Experimental Data:")
    validation = dataset.validate_against_experimental("measured_pressure", 0.19)
    if validation['experimental_value']:
        print(f"Parameter: {validation['parameter']}")
        print(f"Calculated: {validation['calculated_value']:.3f} MPa")
        print(f"Experimental: {validation['experimental_value']:.3f} ± {validation['experimental_uncertainty']:.3f} MPa")
        print(f"Error: {validation['error_percent']:.1f}%")
        print(f"Within uncertainty: {validation['within_uncertainty']}")
    
    print("\n" + "=" * 60)
    print("Dataset usage examples completed successfully!")

if __name__ == "__main__":
    main()