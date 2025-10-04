#!/usr/bin/env python3
"""
Molecular Dynamics Simulation Data Generator
Generates synthetic MD trajectory and property data for materials modeling
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt

class MDSimulationDataGenerator:
    """Generate synthetic molecular dynamics simulation data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.materials = ['Al', 'Cu', 'Fe', 'Ni', 'Ti', 'Mg', 'Zn']
        self.grain_boundary_types = ['tilt', 'twist', 'mixed', 'twin']
        self.dislocation_types = ['edge', 'screw', 'mixed']
        
    def generate_grain_boundary_sliding_data(self, n_simulations: int = 200) -> pd.DataFrame:
        """Generate grain boundary sliding simulation data"""
        data = []
        
        for i in range(n_simulations):
            material = np.random.choice(self.materials)
            gb_type = np.random.choice(self.grain_boundary_types)
            
            # Simulation parameters
            temperature = np.random.uniform(300, 1200)  # K
            applied_stress = np.random.uniform(10, 500)  # MPa
            gb_angle = np.random.uniform(5, 180)  # degrees
            gb_energy = np.random.uniform(0.5, 3.0)  # J/m^2
            
            # Generate sliding resistance based on material properties
            base_resistance = self._get_gb_sliding_resistance(material, gb_type, temperature)
            
            # Stress-dependent sliding rate (power law)
            stress_exponent = np.random.uniform(2, 5)
            sliding_rate = base_resistance * (applied_stress / 100) ** stress_exponent
            
            # Add thermal effects
            thermal_factor = np.exp(-0.5 * (1000/temperature)) * np.random.normal(1, 0.1)
            sliding_rate *= thermal_factor
            
            # Generate time series data for this simulation
            time_steps = np.linspace(0, 100, 1000)  # ps
            displacement = np.cumsum(sliding_rate * 0.1 * np.random.normal(1, 0.05, len(time_steps)))
            
            # Calculate derived properties
            max_displacement = np.max(displacement)
            avg_sliding_rate = np.mean(np.diff(displacement) / np.diff(time_steps))
            
            data.append({
                'simulation_id': f'GB_slide_{i:04d}',
                'material': material,
                'gb_type': gb_type,
                'temperature_K': temperature,
                'applied_stress_MPa': applied_stress,
                'gb_angle_deg': gb_angle,
                'gb_energy_J_m2': gb_energy,
                'sliding_resistance_Pa_s': base_resistance,
                'stress_exponent': stress_exponent,
                'max_displacement_A': max_displacement,
                'avg_sliding_rate_A_ps': avg_sliding_rate,
                'simulation_time_ps': 100,
                'timestep_fs': 1.0,
                'ensemble': np.random.choice(['NVT', 'NPT', 'NVE']),
                'box_size_A': np.random.uniform(50, 200),
                'n_atoms': np.random.randint(10000, 100000)
            })
            
        return pd.DataFrame(data)
    
    def generate_dislocation_mobility_data(self, n_simulations: int = 150) -> pd.DataFrame:
        """Generate dislocation mobility simulation data"""
        data = []
        
        for i in range(n_simulations):
            material = np.random.choice(self.materials)
            disl_type = np.random.choice(self.dislocation_types)
            
            # Simulation conditions
            temperature = np.random.uniform(300, 1200)  # K
            applied_stress = np.random.uniform(1, 200)  # MPa
            dislocation_density = np.random.uniform(1e12, 1e16)  # m^-2
            
            # Burgers vector magnitude (material dependent)
            burgers_vector = self._get_burgers_vector(material)
            
            # Calculate dislocation mobility
            base_mobility = self._get_dislocation_mobility(material, disl_type, temperature)
            
            # Stress and density effects
            stress_factor = applied_stress / 10  # Linear approximation for low stress
            density_factor = 1 / (1 + np.sqrt(dislocation_density / 1e14))  # Density hardening
            
            mobility = base_mobility * stress_factor * density_factor
            
            # Generate velocity profile over time
            time_steps = np.linspace(0, 50, 500)  # ps
            
            # Add fluctuations due to thermal motion and obstacles
            velocity_fluctuations = np.random.normal(0, 0.1 * mobility, len(time_steps))
            velocity = mobility + velocity_fluctuations
            velocity = np.maximum(velocity, 0)  # Physical constraint
            
            # Calculate displacement
            displacement = np.cumsum(velocity * 0.1)  # Convert to Angstroms
            
            # Derived properties
            avg_velocity = np.mean(velocity)
            max_velocity = np.max(velocity)
            total_displacement = displacement[-1]
            
            data.append({
                'simulation_id': f'Disl_mob_{i:04d}',
                'material': material,
                'dislocation_type': disl_type,
                'temperature_K': temperature,
                'applied_stress_MPa': applied_stress,
                'dislocation_density_m2': dislocation_density,
                'burgers_vector_A': burgers_vector,
                'base_mobility_m2_Pa_s': base_mobility,
                'avg_velocity_A_ps': avg_velocity,
                'max_velocity_A_ps': max_velocity,
                'total_displacement_A': total_displacement,
                'simulation_time_ps': 50,
                'timestep_fs': 0.5,
                'crystal_orientation': f"[{np.random.randint(0,3)}{np.random.randint(0,3)}{np.random.randint(1,4)}]",
                'slip_system': np.random.choice(['primary', 'secondary', 'cross_slip']),
                'n_atoms': np.random.randint(50000, 500000)
            })
            
        return pd.DataFrame(data)
    
    def generate_trajectory_data(self, n_trajectories: int = 5) -> Dict[str, Any]:
        """Generate sample MD trajectory data"""
        trajectories = {}
        
        for i in range(n_trajectories):
            traj_id = f'traj_{i:04d}'
            
            # Simulation parameters
            n_atoms = np.random.randint(100, 1000)
            n_steps = np.random.randint(100, 500)
            
            # Generate atomic positions over time
            positions = np.random.uniform(-25, 25, (n_steps, n_atoms, 3))
            
            # Add realistic motion (small displacements between steps)
            for step in range(1, n_steps):
                displacement = np.random.normal(0, 0.1, (n_atoms, 3))
                positions[step] = positions[step-1] + displacement
            
            # Generate forces
            forces = np.random.normal(0, 0.5, (n_steps, n_atoms, 3))
            
            # Generate energies
            kinetic_energy = np.random.uniform(100, 1000, n_steps)
            potential_energy = np.random.uniform(-5000, -1000, n_steps)
            total_energy = kinetic_energy + potential_energy
            
            # Generate temperature and pressure
            temperature = np.random.normal(800, 50, n_steps)
            pressure = np.random.normal(0, 10, n_steps)
            
            trajectories[traj_id] = {
                'metadata': {
                    'material': np.random.choice(self.materials),
                    'n_atoms': n_atoms,
                    'n_steps': n_steps,
                    'timestep_fs': 1.0,
                    'total_time_ps': n_steps * 0.001,
                    'ensemble': np.random.choice(['NVT', 'NPT', 'NVE']),
                    'box_dimensions_A': [50.0, 50.0, 50.0]
                },
                'positions_A': positions.tolist(),  # Convert to list for JSON serialization
                'forces_eV_A': forces.tolist(),
                'kinetic_energy_eV': kinetic_energy.tolist(),
                'potential_energy_eV': potential_energy.tolist(),
                'total_energy_eV': total_energy.tolist(),
                'temperature_K': temperature.tolist(),
                'pressure_GPa': pressure.tolist()
            }
            
        return trajectories
    
    def generate_force_field_parameters(self) -> pd.DataFrame:
        """Generate force field parameters used in MD simulations"""
        data = []
        
        for material in self.materials:
            # Lennard-Jones parameters
            epsilon = np.random.uniform(0.01, 0.5)  # eV
            sigma = np.random.uniform(2.0, 4.0)  # Angstrom
            
            # EAM parameters (for metals)
            lattice_constant = np.random.uniform(3.0, 5.0)  # Angstrom
            cohesive_energy = np.random.uniform(2.0, 8.0)  # eV
            bulk_modulus = np.random.uniform(50, 300)  # GPa
            
            # Embedded atom method parameters
            rho_cutoff = np.random.uniform(8.0, 12.0)  # Angstrom
            phi_cutoff = np.random.uniform(10.0, 15.0)  # Angstrom
            
            data.append({
                'material': material,
                'potential_type': 'EAM',
                'epsilon_eV': epsilon,
                'sigma_A': sigma,
                'lattice_constant_A': lattice_constant,
                'cohesive_energy_eV': cohesive_energy,
                'bulk_modulus_GPa': bulk_modulus,
                'rho_cutoff_A': rho_cutoff,
                'phi_cutoff_A': phi_cutoff,
                'atomic_mass_amu': self._get_atomic_mass(material),
                'melting_point_K': self._get_melting_point(material)
            })
            
        return pd.DataFrame(data)
    
    def _get_gb_sliding_resistance(self, material: str, gb_type: str, temperature: float) -> float:
        """Get grain boundary sliding resistance"""
        base_resistances = {
            'Al': {'tilt': 1e-8, 'twist': 1.5e-8, 'mixed': 1.2e-8, 'twin': 0.5e-8},
            'Cu': {'tilt': 1.5e-8, 'twist': 2e-8, 'mixed': 1.8e-8, 'twin': 0.8e-8},
            'Fe': {'tilt': 5e-8, 'twist': 7e-8, 'mixed': 6e-8, 'twin': 2e-8},
            'Ni': {'tilt': 3e-8, 'twist': 4e-8, 'mixed': 3.5e-8, 'twin': 1.2e-8},
            'Ti': {'tilt': 8e-8, 'twist': 12e-8, 'mixed': 10e-8, 'twin': 3e-8},
            'Mg': {'tilt': 0.8e-8, 'twist': 1.2e-8, 'mixed': 1e-8, 'twin': 0.3e-8},
            'Zn': {'tilt': 0.6e-8, 'twist': 1e-8, 'mixed': 0.8e-8, 'twin': 0.2e-8}
        }
        
        base = base_resistances[material][gb_type]
        # Temperature dependence (Arrhenius-like)
        temp_factor = np.exp(2000/temperature)  # Simplified activation
        return base * temp_factor * np.random.normal(1, 0.2)
    
    def _get_dislocation_mobility(self, material: str, disl_type: str, temperature: float) -> float:
        """Get dislocation mobility"""
        base_mobilities = {
            'Al': {'edge': 1e-4, 'screw': 0.8e-4, 'mixed': 0.9e-4},
            'Cu': {'edge': 0.8e-4, 'screw': 0.6e-4, 'mixed': 0.7e-4},
            'Fe': {'edge': 0.3e-4, 'screw': 0.2e-4, 'mixed': 0.25e-4},
            'Ni': {'edge': 0.5e-4, 'screw': 0.4e-4, 'mixed': 0.45e-4},
            'Ti': {'edge': 0.2e-4, 'screw': 0.15e-4, 'mixed': 0.18e-4},
            'Mg': {'edge': 1.5e-4, 'screw': 1.2e-4, 'mixed': 1.35e-4},
            'Zn': {'edge': 1.8e-4, 'screw': 1.5e-4, 'mixed': 1.65e-4}
        }
        
        base = base_mobilities[material][disl_type]
        # Temperature dependence
        temp_factor = (temperature / 300) ** 0.5
        return base * temp_factor * np.random.normal(1, 0.15)
    
    def _get_burgers_vector(self, material: str) -> float:
        """Get Burgers vector magnitude"""
        burgers_vectors = {
            'Al': 2.86, 'Cu': 2.56, 'Fe': 2.48, 'Ni': 2.49, 
            'Ti': 2.95, 'Mg': 3.21, 'Zn': 2.66
        }
        return burgers_vectors[material] * np.random.normal(1, 0.02)
    
    def _get_atomic_mass(self, material: str) -> float:
        """Get atomic mass in amu"""
        masses = {
            'Al': 26.98, 'Cu': 63.55, 'Fe': 55.85, 'Ni': 58.69,
            'Ti': 47.87, 'Mg': 24.31, 'Zn': 65.38
        }
        return masses[material]
    
    def _get_melting_point(self, material: str) -> float:
        """Get melting point in K"""
        melting_points = {
            'Al': 933, 'Cu': 1358, 'Fe': 1811, 'Ni': 1728,
            'Ti': 1941, 'Mg': 923, 'Zn': 693
        }
        return melting_points[material]

def main():
    """Generate MD simulation dataset"""
    print("Generating MD Simulation Dataset...")
    
    # Create output directory
    os.makedirs('atomic_simulation_dataset/md_data', exist_ok=True)
    
    # Initialize generator
    generator = MDSimulationDataGenerator(seed=42)
    
    # Generate grain boundary sliding data
    print("Generating grain boundary sliding data...")
    gb_sliding = generator.generate_grain_boundary_sliding_data(n_simulations=200)
    gb_sliding.to_csv('atomic_simulation_dataset/md_data/grain_boundary_sliding.csv', index=False)
    
    # Generate dislocation mobility data
    print("Generating dislocation mobility data...")
    disl_mobility = generator.generate_dislocation_mobility_data(n_simulations=150)
    disl_mobility.to_csv('atomic_simulation_dataset/md_data/dislocation_mobility.csv', index=False)
    
    # Generate force field parameters
    print("Generating force field parameters...")
    ff_params = generator.generate_force_field_parameters()
    ff_params.to_csv('atomic_simulation_dataset/md_data/force_field_parameters.csv', index=False)
    
    # Generate sample trajectory data (smaller subset due to size)
    print("Generating sample trajectory data...")
    trajectories = generator.generate_trajectory_data(n_trajectories=3)
    
    with open('atomic_simulation_dataset/md_data/sample_trajectories.json', 'w') as f:
        json.dump(trajectories, f, indent=2)
    
    print("MD simulation dataset generated successfully!")
    print("Files created:")
    print("- grain_boundary_sliding.csv")
    print("- dislocation_mobility.csv")
    print("- force_field_parameters.csv")
    print("- sample_trajectories.json")

if __name__ == "__main__":
    main()