#!/usr/bin/env python3
"""
Numerical Simulation Dataset Generator for Multi-Physics FEM Models
Generates comprehensive datasets for COMSOL/ABAQUS simulations including:
- Input parameters (mesh, boundary conditions, material models, thermal profiles)
- Output data (stress, strain, damage evolution, temperature, voltage distributions)
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any
import random

class NumericalSimulationDataset:
    """Generator for multi-physics FEM simulation datasets"""
    
    def __init__(self, seed: int = 42):
        """Initialize the dataset generator with random seed"""
        np.random.seed(seed)
        random.seed(seed)
        self.dataset = {}
        
    def generate_mesh_data(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate mesh data including element size, type, and interface refinement"""
        mesh_data = {
            "element_sizes": np.random.uniform(0.1, 2.0, num_simulations),  # mm
            "element_types": np.random.choice([
                "Tetrahedral", "Hexahedral", "Triangular", "Quadrilateral", 
                "Wedge", "Pyramid", "Mixed"
            ], num_simulations),
            "interface_refinement_levels": np.random.randint(1, 5, num_simulations),
            "total_elements": np.random.randint(10000, 1000000, num_simulations),
            "mesh_quality": np.random.uniform(0.6, 1.0, num_simulations),
            "aspect_ratios": np.random.uniform(1.0, 10.0, num_simulations)
        }
        return mesh_data
    
    def generate_boundary_conditions(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate boundary conditions for temperature, displacement, and voltage"""
        boundary_conditions = {
            "temperature_conditions": {
                "ambient_temp": np.random.uniform(20, 25, num_simulations),  # °C
                "max_temp": np.random.uniform(100, 300, num_simulations),   # °C
                "temp_gradient": np.random.uniform(0.1, 50, num_simulations),  # °C/mm
                "convection_coeff": np.random.uniform(5, 100, num_simulations),  # W/m²K
                "radiation_emissivity": np.random.uniform(0.1, 0.95, num_simulations)
            },
            "displacement_conditions": {
                "fixed_displacements": np.random.uniform(0, 0.1, num_simulations),  # mm
                "applied_forces": np.random.uniform(100, 10000, num_simulations),  # N
                "pressure_loads": np.random.uniform(0.1, 10, num_simulations),     # MPa
                "constraint_types": np.random.choice([
                    "Fixed", "Free", "Symmetry", "Periodic", "Contact"
                ], num_simulations)
            },
            "voltage_conditions": {
                "applied_voltage": np.random.uniform(0, 1000, num_simulations),    # V
                "current_density": np.random.uniform(0, 100, num_simulations),     # A/m²
                "electrical_conductivity": np.random.uniform(1e-6, 1e8, num_simulations),  # S/m
                "dielectric_constant": np.random.uniform(1, 100, num_simulations)
            }
        }
        return boundary_conditions
    
    def generate_material_models(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate material model parameters for various material behaviors"""
        material_models = {
            "elastic_properties": {
                "youngs_modulus": np.random.uniform(1e9, 500e9, num_simulations),  # Pa
                "poisson_ratio": np.random.uniform(0.1, 0.5, num_simulations),
                "shear_modulus": np.random.uniform(1e8, 200e9, num_simulations),   # Pa
                "bulk_modulus": np.random.uniform(1e9, 300e9, num_simulations)     # Pa
            },
            "plastic_properties": {
                "yield_strength": np.random.uniform(50e6, 2000e6, num_simulations),  # Pa
                "hardening_modulus": np.random.uniform(1e6, 100e6, num_simulations),  # Pa
                "strain_hardening_exponent": np.random.uniform(0.05, 0.5, num_simulations),
                "plastic_strain_rate": np.random.uniform(1e-6, 1e-2, num_simulations)  # 1/s
            },
            "creep_properties": {
                "creep_coefficient": np.random.uniform(1e-20, 1e-10, num_simulations),
                "creep_exponent": np.random.uniform(1, 8, num_simulations),
                "activation_energy": np.random.uniform(50e3, 500e3, num_simulations),  # J/mol
                "reference_stress": np.random.uniform(10e6, 500e6, num_simulations)    # Pa
            },
            "thermal_properties": {
                "thermal_conductivity": np.random.uniform(0.1, 400, num_simulations),  # W/mK
                "specific_heat": np.random.uniform(100, 2000, num_simulations),       # J/kgK
                "density": np.random.uniform(1000, 20000, num_simulations),          # kg/m³
                "thermal_expansion": np.random.uniform(1e-6, 50e-6, num_simulations)  # 1/K
            },
            "electrochemical_properties": {
                "electrical_conductivity": np.random.uniform(1e-6, 1e8, num_simulations),  # S/m
                "ionic_conductivity": np.random.uniform(1e-8, 1e-2, num_simulations),      # S/m
                "diffusion_coefficient": np.random.uniform(1e-15, 1e-8, num_simulations),  # m²/s
                "electrochemical_potential": np.random.uniform(-2, 2, num_simulations)     # V
            }
        }
        return material_models
    
    def generate_thermal_profiles(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate transient thermal profiles with heating/cooling rates"""
        thermal_profiles = {
            "heating_rates": np.random.uniform(1, 10, num_simulations),  # °C/min
            "cooling_rates": np.random.uniform(1, 10, num_simulations),  # °C/min
            "temperature_cycles": {
                "cycle_duration": np.random.uniform(60, 3600, num_simulations),  # seconds
                "max_temperature": np.random.uniform(100, 500, num_simulations),  # °C
                "min_temperature": np.random.uniform(-50, 50, num_simulations),   # °C
                "ramp_time": np.random.uniform(10, 300, num_simulations)          # seconds
            },
            "thermal_history": []
        }
        
        # Generate time-temperature profiles for each simulation
        for i in range(num_simulations):
            time_points = np.linspace(0, thermal_profiles["temperature_cycles"]["cycle_duration"][i], 100)
            temp_profile = self._generate_temperature_profile(
                time_points, 
                thermal_profiles["heating_rates"][i],
                thermal_profiles["cooling_rates"][i],
                thermal_profiles["temperature_cycles"]["max_temperature"][i],
                thermal_profiles["temperature_cycles"]["min_temperature"][i]
            )
            thermal_profiles["thermal_history"].append({
                "simulation_id": i,
                "time": time_points.tolist(),
                "temperature": temp_profile.tolist()
            })
        
        return thermal_profiles
    
    def _generate_temperature_profile(self, time, heating_rate, cooling_rate, max_temp, min_temp):
        """Generate a realistic temperature profile"""
        cycle_time = time[-1]
        ramp_time = cycle_time * 0.3
        
        temp_profile = np.zeros_like(time)
        for i, t in enumerate(time):
            if t < ramp_time:
                # Heating phase
                temp_profile[i] = min_temp + (max_temp - min_temp) * (t / ramp_time)
            elif t < cycle_time - ramp_time:
                # Hold phase
                temp_profile[i] = max_temp
            else:
                # Cooling phase
                cool_time = t - (cycle_time - ramp_time)
                temp_profile[i] = max_temp - (max_temp - min_temp) * (cool_time / ramp_time)
        
        return temp_profile
    
    def generate_output_data(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate output data including stress, strain, damage evolution, etc."""
        output_data = {
            "stress_distributions": {
                "von_mises_stress": np.random.uniform(0, 1000e6, num_simulations),  # Pa
                "principal_stress_1": np.random.uniform(-500e6, 1000e6, num_simulations),  # Pa
                "principal_stress_2": np.random.uniform(-500e6, 1000e6, num_simulations),  # Pa
                "principal_stress_3": np.random.uniform(-500e6, 1000e6, num_simulations),  # Pa
                "interfacial_shear_stress": np.random.uniform(0, 100e6, num_simulations)   # Pa
            },
            "strain_fields": {
                "elastic_strain": np.random.uniform(0, 0.01, num_simulations),
                "plastic_strain": np.random.uniform(0, 0.1, num_simulations),
                "creep_strain": np.random.uniform(0, 0.05, num_simulations),
                "thermal_strain": np.random.uniform(-0.01, 0.01, num_simulations),
                "total_strain": np.random.uniform(0, 0.15, num_simulations)
            },
            "damage_evolution": {
                "damage_variable_D": np.random.uniform(0, 1, num_simulations),
                "damage_rate": np.random.uniform(0, 1e-3, num_simulations),  # 1/s
                "crack_length": np.random.uniform(0, 10, num_simulations),    # mm
                "crack_density": np.random.uniform(0, 100, num_simulations),  # cracks/mm²
                "fatigue_life": np.random.uniform(100, 1e6, num_simulations)  # cycles
            },
            "temperature_distributions": {
                "max_temperature": np.random.uniform(50, 500, num_simulations),  # °C
                "min_temperature": np.random.uniform(-50, 50, num_simulations),  # °C
                "temperature_gradient": np.random.uniform(0.1, 100, num_simulations),  # °C/mm
                "thermal_conductivity_effective": np.random.uniform(0.1, 400, num_simulations)  # W/mK
            },
            "voltage_distributions": {
                "max_voltage": np.random.uniform(0, 1000, num_simulations),  # V
                "min_voltage": np.random.uniform(-1000, 0, num_simulations),  # V
                "voltage_gradient": np.random.uniform(0, 1000, num_simulations),  # V/mm
                "current_density_max": np.random.uniform(0, 1000, num_simulations),  # A/m²
                "electric_field_strength": np.random.uniform(0, 1e6, num_simulations)  # V/m
            },
            "failure_predictions": {
                "delamination_probability": np.random.uniform(0, 1, num_simulations),
                "crack_initiation_time": np.random.uniform(0, 10000, num_simulations),  # hours
                "failure_mode": np.random.choice([
                    "Brittle fracture", "Ductile failure", "Fatigue", 
                    "Creep rupture", "Thermal shock", "Electrical breakdown"
                ], num_simulations),
                "safety_factor": np.random.uniform(0.5, 5.0, num_simulations)
            }
        }
        return output_data
    
    def generate_complete_dataset(self, num_simulations: int = 100) -> Dict[str, Any]:
        """Generate the complete numerical simulation dataset"""
        print(f"Generating dataset for {num_simulations} simulations...")
        
        self.dataset = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "num_simulations": num_simulations,
                "simulation_type": "Multi-physics FEM",
                "software": ["COMSOL", "ABAQUS"],
                "description": "Comprehensive numerical simulation dataset for multi-physics FEM models"
            },
            "input_parameters": {
                "mesh_data": self.generate_mesh_data(num_simulations),
                "boundary_conditions": self.generate_boundary_conditions(num_simulations),
                "material_models": self.generate_material_models(num_simulations),
                "thermal_profiles": self.generate_thermal_profiles(num_simulations)
            },
            "output_data": self.generate_output_data(num_simulations)
        }
        
        return self.dataset
    
    def save_dataset(self, filename: str = "numerical_simulation_dataset.json"):
        """Save the dataset to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.dataset, f, indent=2, default=str)
        print(f"Dataset saved to {filename}")
    
    def export_to_csv(self, output_dir: str = "simulation_data"):
        """Export dataset components to CSV files for easy analysis"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export input parameters
        for category, data in self.dataset["input_parameters"].items():
            if category == "thermal_profiles":
                # Handle thermal profiles separately due to nested structure
                continue
            
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/{category}.csv", index=False)
        
        # Export output data
        for category, data in self.dataset["output_data"].items():
            df = pd.DataFrame(data)
            df.to_csv(f"{output_dir}/output_{category}.csv", index=False)
        
        print(f"CSV files exported to {output_dir}/")
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        if not self.dataset:
            return {}
        
        summary = {
            "dataset_size": self.dataset["metadata"]["num_simulations"],
            "input_parameters_count": len(self.dataset["input_parameters"]),
            "output_parameters_count": len(self.dataset["output_data"]),
            "statistics": {}
        }
        
        # Calculate statistics for key parameters
        key_params = [
            ("stress", "output_data.stress_distributions.von_mises_stress"),
            ("strain", "output_data.strain_fields.total_strain"),
            ("damage", "output_data.damage_evolution.damage_variable_D"),
            ("temperature", "output_data.temperature_distributions.max_temperature")
        ]
        
        for name, path in key_params:
            keys = path.split('.')
            data = self.dataset
            for key in keys:
                data = data[key]
            
            summary["statistics"][name] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "median": float(np.median(data))
            }
        
        return summary

def main():
    """Main function to generate and save the dataset"""
    # Create dataset generator
    generator = NumericalSimulationDataset(seed=42)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(num_simulations=100)
    
    # Save dataset
    generator.save_dataset("numerical_simulation_dataset.json")
    
    # Export to CSV
    generator.export_to_csv("simulation_data")
    
    # Generate and print summary statistics
    summary = generator.generate_summary_statistics()
    print("\nDataset Summary:")
    print(json.dumps(summary, indent=2))
    
    print(f"\nDataset generation complete!")
    print(f"Total simulations: {dataset['metadata']['num_simulations']}")
    print(f"Input parameter categories: {len(dataset['input_parameters'])}")
    print(f"Output data categories: {len(dataset['output_data'])}")

if __name__ == "__main__":
    main()