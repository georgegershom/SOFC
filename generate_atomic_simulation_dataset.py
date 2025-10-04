#!/usr/bin/env python3
"""
Atomic-Scale Simulation Dataset Generator
Generates synthetic DFT and MD simulation data for quantum-enhanced materials modeling
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class AtomicSimulationDataGenerator:
    """Generate synthetic atomic-scale simulation data for materials science"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.materials = ['Al', 'Cu', 'Fe', 'Ni', 'Ti', 'Mg', 'Zn']
        self.defect_types = ['vacancy', 'interstitial', 'substitutional', 'dislocation', 'grain_boundary']
        self.crystal_structures = ['fcc', 'bcc', 'hcp']
        
    def generate_dft_formation_energies(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate DFT formation energy data for various defects"""
        data = []
        
        for i in range(n_samples):
            material = np.random.choice(self.materials)
            defect_type = np.random.choice(self.defect_types)
            crystal_structure = np.random.choice(self.crystal_structures)
            
            # Base formation energy depends on material and defect type
            base_energy = self._get_base_formation_energy(material, defect_type)
            
            # Add realistic variations
            temperature = np.random.uniform(300, 1200)  # K
            concentration = np.random.uniform(1e-6, 1e-2)  # atomic fraction
            
            # Temperature and concentration effects
            thermal_correction = 0.1 * np.log(temperature/300) * np.random.normal(1, 0.1)
            conc_correction = 0.05 * np.log(concentration/1e-4) * np.random.normal(1, 0.05)
            
            formation_energy = base_energy + thermal_correction + conc_correction
            
            # Add computational uncertainty
            formation_energy += np.random.normal(0, 0.02)
            
            data.append({
                'material': material,
                'defect_type': defect_type,
                'crystal_structure': crystal_structure,
                'formation_energy_eV': formation_energy,
                'temperature_K': temperature,
                'defect_concentration': concentration,
                'supercell_size': np.random.choice([64, 108, 216, 512]),
                'k_points': f"{np.random.randint(4,12)}x{np.random.randint(4,12)}x{np.random.randint(4,12)}",
                'exchange_correlation': np.random.choice(['PBE', 'PBEsol', 'HSE06', 'LDA']),
                'cutoff_energy_eV': np.random.uniform(400, 800),
                'convergence_threshold': np.random.uniform(1e-6, 1e-4)
            })
            
        return pd.DataFrame(data)
    
    def generate_activation_barriers(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate activation energy barriers for diffusion processes"""
        data = []
        
        diffusion_mechanisms = ['vacancy_migration', 'interstitial_migration', 'grain_boundary_diffusion', 
                              'surface_diffusion', 'pipe_diffusion']
        
        for i in range(n_samples):
            material = np.random.choice(self.materials)
            mechanism = np.random.choice(diffusion_mechanisms)
            
            # Base activation energy
            base_barrier = self._get_base_activation_energy(material, mechanism)
            
            # Environmental effects
            stress = np.random.uniform(0, 500)  # MPa
            grain_size = np.random.uniform(10, 1000)  # nm
            temperature = np.random.uniform(300, 1200)  # K
            
            # Stress and grain size effects
            stress_effect = -0.1 * stress / 100 * np.random.normal(1, 0.1)
            grain_size_effect = 0.05 * np.log(grain_size/100) * np.random.normal(1, 0.05)
            
            activation_barrier = base_barrier + stress_effect + grain_size_effect
            activation_barrier = max(0.1, activation_barrier)  # Physical constraint
            
            # Calculate attempt frequency (Debye frequency ~ 10^13 Hz)
            attempt_frequency = np.random.uniform(1e12, 1e14)
            
            data.append({
                'material': material,
                'diffusion_mechanism': mechanism,
                'activation_barrier_eV': activation_barrier,
                'attempt_frequency_Hz': attempt_frequency,
                'applied_stress_MPa': stress,
                'grain_size_nm': grain_size,
                'temperature_K': temperature,
                'migration_path_length_A': np.random.uniform(2, 6),
                'coordination_number': np.random.randint(4, 12),
                'elastic_modulus_GPa': np.random.uniform(50, 300)
            })
            
        return pd.DataFrame(data)
    
    def generate_surface_energies(self, n_samples: int = 300) -> pd.DataFrame:
        """Generate surface energy data for different crystallographic planes"""
        data = []
        
        miller_indices = [(1,0,0), (1,1,0), (1,1,1), (2,1,0), (2,1,1), (3,1,0)]
        
        for i in range(n_samples):
            material = np.random.choice(self.materials)
            crystal_structure = np.random.choice(self.crystal_structures)
            miller = miller_indices[np.random.randint(len(miller_indices))]
            
            # Base surface energy
            base_surface_energy = self._get_base_surface_energy(material, miller, crystal_structure)
            
            # Environmental effects
            temperature = np.random.uniform(300, 1200)
            atmosphere = np.random.choice(['vacuum', 'air', 'inert', 'reducing'])
            
            # Temperature effect (surface energy typically decreases with T)
            temp_effect = -0.0001 * (temperature - 300) * np.random.normal(1, 0.1)
            
            # Atmosphere effect
            atm_effects = {'vacuum': 0, 'air': 0.05, 'inert': 0.01, 'reducing': -0.02}
            atm_effect = atm_effects[atmosphere] * np.random.normal(1, 0.2)
            
            surface_energy = base_surface_energy + temp_effect + atm_effect
            surface_energy = max(0.1, surface_energy)  # Physical constraint
            
            data.append({
                'material': material,
                'crystal_structure': crystal_structure,
                'miller_indices': f"({miller[0]}{miller[1]}{miller[2]})",
                'surface_energy_J_m2': surface_energy,
                'temperature_K': temperature,
                'atmosphere': atmosphere,
                'surface_area_A2': np.random.uniform(100, 1000),
                'relaxation_layers': np.random.randint(3, 10),
                'slab_thickness_A': np.random.uniform(15, 40),
                'vacuum_thickness_A': np.random.uniform(15, 30)
            })
            
        return pd.DataFrame(data)
    
    def _get_base_formation_energy(self, material: str, defect_type: str) -> float:
        """Get realistic base formation energies"""
        # Typical formation energies in eV
        formation_energies = {
            'Al': {'vacancy': 0.68, 'interstitial': 2.8, 'substitutional': 0.5, 'dislocation': 1.2, 'grain_boundary': 0.8},
            'Cu': {'vacancy': 1.28, 'interstitial': 3.1, 'substitutional': 0.7, 'dislocation': 1.5, 'grain_boundary': 1.0},
            'Fe': {'vacancy': 2.17, 'interstitial': 4.2, 'substitutional': 1.2, 'dislocation': 2.1, 'grain_boundary': 1.5},
            'Ni': {'vacancy': 1.79, 'interstitial': 3.8, 'substitutional': 0.9, 'dislocation': 1.8, 'grain_boundary': 1.2},
            'Ti': {'vacancy': 2.05, 'interstitial': 4.5, 'substitutional': 1.5, 'dislocation': 2.3, 'grain_boundary': 1.7},
            'Mg': {'vacancy': 0.79, 'interstitial': 2.2, 'substitutional': 0.4, 'dislocation': 1.0, 'grain_boundary': 0.6},
            'Zn': {'vacancy': 0.64, 'interstitial': 2.1, 'substitutional': 0.3, 'dislocation': 0.9, 'grain_boundary': 0.5}
        }
        return formation_energies[material][defect_type] * np.random.normal(1, 0.1)
    
    def _get_base_activation_energy(self, material: str, mechanism: str) -> float:
        """Get realistic base activation energies"""
        # Typical activation energies in eV
        activation_energies = {
            'Al': {'vacancy_migration': 0.61, 'interstitial_migration': 0.15, 'grain_boundary_diffusion': 0.45, 
                   'surface_diffusion': 0.30, 'pipe_diffusion': 0.35},
            'Cu': {'vacancy_migration': 0.71, 'interstitial_migration': 0.18, 'grain_boundary_diffusion': 0.55,
                   'surface_diffusion': 0.35, 'pipe_diffusion': 0.40},
            'Fe': {'vacancy_migration': 1.35, 'interstitial_migration': 0.35, 'grain_boundary_diffusion': 1.10,
                   'surface_diffusion': 0.70, 'pipe_diffusion': 0.85},
            'Ni': {'vacancy_migration': 1.04, 'interstitial_migration': 0.25, 'grain_boundary_diffusion': 0.85,
                   'surface_diffusion': 0.55, 'pipe_diffusion': 0.65},
            'Ti': {'vacancy_migration': 1.55, 'interstitial_migration': 0.45, 'grain_boundary_diffusion': 1.25,
                   'surface_diffusion': 0.80, 'pipe_diffusion': 0.95},
            'Mg': {'vacancy_migration': 0.52, 'interstitial_migration': 0.12, 'grain_boundary_diffusion': 0.38,
                   'surface_diffusion': 0.25, 'pipe_diffusion': 0.30},
            'Zn': {'vacancy_migration': 0.45, 'interstitial_migration': 0.10, 'grain_boundary_diffusion': 0.32,
                   'surface_diffusion': 0.20, 'pipe_diffusion': 0.25}
        }
        return activation_energies[material][mechanism] * np.random.normal(1, 0.15)
    
    def _get_base_surface_energy(self, material: str, miller: Tuple[int,int,int], crystal_structure: str) -> float:
        """Get realistic base surface energies"""
        # Surface energies in J/m^2
        base_energies = {
            'Al': 1.16, 'Cu': 1.83, 'Fe': 2.90, 'Ni': 2.38, 'Ti': 2.10, 'Mg': 0.76, 'Zn': 0.99
        }
        
        # Miller index factors (higher index = higher energy)
        miller_factor = (miller[0]**2 + miller[1]**2 + miller[2]**2) / 3.0
        
        # Crystal structure factors
        structure_factors = {'fcc': 1.0, 'bcc': 1.1, 'hcp': 0.95}
        
        return base_energies[material] * miller_factor * structure_factors[crystal_structure] * np.random.normal(1, 0.1)

def main():
    """Generate the complete atomic-scale simulation dataset"""
    print("Generating Atomic-Scale Simulation Dataset...")
    
    # Create output directory
    os.makedirs('atomic_simulation_dataset', exist_ok=True)
    
    # Initialize generator
    generator = AtomicSimulationDataGenerator(seed=42)
    
    # Generate DFT data
    print("Generating DFT formation energy data...")
    dft_formation = generator.generate_dft_formation_energies(n_samples=1000)
    dft_formation.to_csv('atomic_simulation_dataset/dft_formation_energies.csv', index=False)
    
    print("Generating activation barrier data...")
    activation_barriers = generator.generate_activation_barriers(n_samples=500)
    activation_barriers.to_csv('atomic_simulation_dataset/activation_barriers.csv', index=False)
    
    print("Generating surface energy data...")
    surface_energies = generator.generate_surface_energies(n_samples=300)
    surface_energies.to_csv('atomic_simulation_dataset/surface_energies.csv', index=False)
    
    # Generate summary statistics
    summary = {
        'dataset_info': {
            'generated_date': datetime.now().isoformat(),
            'total_samples': len(dft_formation) + len(activation_barriers) + len(surface_energies),
            'dft_formation_samples': len(dft_formation),
            'activation_barrier_samples': len(activation_barriers),
            'surface_energy_samples': len(surface_energies)
        },
        'materials_covered': generator.materials,
        'defect_types': generator.defect_types,
        'crystal_structures': generator.crystal_structures
    }
    
    with open('atomic_simulation_dataset/dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Dataset generated successfully!")
    print(f"Total samples: {summary['dataset_info']['total_samples']}")
    print("Files created:")
    print("- dft_formation_energies.csv")
    print("- activation_barriers.csv") 
    print("- surface_energies.csv")
    print("- dataset_summary.json")

if __name__ == "__main__":
    main()