#!/usr/bin/env python3
"""
Generate synthetic DFT calculation data for material modeling.
Includes formation energies, activation barriers, and surface energies.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

class DFTDataGenerator:
    """Generate realistic DFT calculation outputs"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.timestamp = datetime.now().isoformat()
        
        # Material parameters (example: Ni-based superalloy)
        self.lattice_param = 3.524  # Angstroms (FCC Ni)
        self.cohesive_energy = -4.44  # eV/atom (Ni)
        
    def generate_vacancy_formation_energies(self, n_configs: int = 50) -> Dict:
        """Generate vacancy formation energies for different configurations"""
        data = {
            'calculation_type': 'vacancy_formation',
            'method': 'DFT-PBE+U',
            'timestamp': self.timestamp,
            'units': {'energy': 'eV', 'distance': 'Angstrom'},
            'configurations': []
        }
        
        for i in range(n_configs):
            # Base vacancy formation energy with variations
            E_vac_base = 1.39  # eV (typical for Ni)
            
            # Add variations based on local environment
            local_strain = np.random.normal(0, 0.02)  # Local strain effect
            chemical_env = np.random.normal(0, 0.15)  # Chemical environment
            
            E_formation = E_vac_base + local_strain * 2.5 + chemical_env
            
            config = {
                'config_id': f'vac_{i:04d}',
                'supercell_size': [3, 3, 3],
                'total_atoms': 107,  # 108 - 1 vacancy
                'vacancy_site': {
                    'type': np.random.choice(['bulk', 'near_gb', 'near_surface']),
                    'coordinates': [
                        np.random.uniform(0, 10.572),
                        np.random.uniform(0, 10.572),
                        np.random.uniform(0, 10.572)
                    ]
                },
                'formation_energy': float(E_formation),
                'relaxation_volume': float(np.random.uniform(0.2, 0.4)),  # Vacancy volume/atomic volume
                'magnetic_moment': float(np.random.uniform(0.5, 0.7)),
                'convergence': {
                    'energy_tolerance': 1e-6,
                    'force_tolerance': 0.01,
                    'iterations': int(np.random.randint(30, 80))
                }
            }
            data['configurations'].append(config)
        
        return data
    
    def generate_dislocation_energies(self, n_configs: int = 30) -> Dict:
        """Generate dislocation and stacking fault energies"""
        data = {
            'calculation_type': 'dislocation_energy',
            'method': 'DFT-PBE',
            'timestamp': self.timestamp,
            'units': {'energy': 'mJ/m^2', 'distance': 'Angstrom'},
            'configurations': []
        }
        
        dislocation_types = ['edge', 'screw', 'mixed']
        
        for i in range(n_configs):
            disl_type = np.random.choice(dislocation_types)
            
            # Stacking fault energies (mJ/m^2)
            if disl_type == 'edge':
                E_sf = np.random.normal(125, 10)  # Edge dislocation
                core_radius = np.random.uniform(5, 8)
            elif disl_type == 'screw':
                E_sf = np.random.normal(150, 12)  # Screw dislocation
                core_radius = np.random.uniform(3, 5)
            else:  # mixed
                E_sf = np.random.normal(137, 11)
                core_radius = np.random.uniform(4, 6)
            
            config = {
                'config_id': f'disl_{i:04d}',
                'dislocation_type': disl_type,
                'burgers_vector': [self.lattice_param/2, self.lattice_param/2, 0],
                'slip_plane': [1, 1, 1],
                'stacking_fault_energy': float(E_sf),
                'core_energy': float(E_sf * core_radius * 0.1),  # eV/Angstrom
                'core_radius': float(core_radius),
                'peierls_stress': float(np.random.uniform(50, 200)),  # MPa
                'elastic_interaction': {
                    'shear_modulus': float(np.random.uniform(75, 85)),  # GPa
                    'poisson_ratio': 0.31
                }
            }
            data['configurations'].append(config)
        
        return data
    
    def generate_grain_boundary_energies(self, n_configs: int = 40) -> Dict:
        """Generate grain boundary energies for different misorientations"""
        data = {
            'calculation_type': 'grain_boundary_energy',
            'method': 'DFT-PBE',
            'timestamp': self.timestamp,
            'units': {'energy': 'J/m^2', 'angle': 'degrees'},
            'configurations': []
        }
        
        gb_types = ['tilt', 'twist', 'mixed']
        
        for i in range(n_configs):
            gb_type = np.random.choice(gb_types)
            misorientation = np.random.uniform(5, 60)  # degrees
            
            # GB energy depends on misorientation (Read-Shockley model for low angles)
            if misorientation < 15:
                E_gb_base = 0.324 * misorientation * (1 - np.log(misorientation/15))
            else:
                E_gb_base = np.random.uniform(0.5, 1.5)  # J/m^2 for high angle GBs
            
            # Add some scatter
            E_gb = E_gb_base * np.random.uniform(0.9, 1.1)
            
            config = {
                'config_id': f'gb_{i:04d}',
                'gb_type': gb_type,
                'misorientation_angle': float(misorientation),
                'rotation_axis': [int(x) for x in np.random.randint(0, 4, 3)],
                'gb_plane': [int(x) for x in np.random.randint(1, 5, 3)],
                'gb_energy': float(E_gb),
                'excess_volume': float(np.random.uniform(0.05, 0.15)),  # Angstrom
                'segregation_energy': {
                    'Cr': float(np.random.uniform(-0.5, -0.1)),
                    'Mo': float(np.random.uniform(-0.3, 0.1)),
                    'Al': float(np.random.uniform(-0.2, 0.2))
                },
                'mobility_prefactor': float(np.exp(np.random.uniform(-2, 2)))
            }
            data['configurations'].append(config)
        
        return data
    
    def generate_activation_barriers(self, n_paths: int = 60) -> Dict:
        """Generate activation energy barriers for diffusion processes"""
        data = {
            'calculation_type': 'activation_barriers',
            'method': 'DFT-NEB',  # Nudged Elastic Band
            'timestamp': self.timestamp,
            'units': {'energy': 'eV', 'distance': 'Angstrom'},
            'diffusion_paths': []
        }
        
        mechanisms = ['vacancy', 'interstitial', 'interstitialcy', 'solute_drag']
        
        for i in range(n_paths):
            mechanism = np.random.choice(mechanisms)
            
            # Base activation energies (eV)
            if mechanism == 'vacancy':
                E_act_base = np.random.uniform(1.0, 1.4)
                prefactor = 1e-4  # cm^2/s
            elif mechanism == 'interstitial':
                E_act_base = np.random.uniform(0.1, 0.3)
                prefactor = 1e-3
            elif mechanism == 'interstitialcy':
                E_act_base = np.random.uniform(0.3, 0.6)
                prefactor = 5e-4
            else:  # solute_drag
                E_act_base = np.random.uniform(1.5, 2.5)
                prefactor = 1e-5
            
            # Generate reaction coordinate
            n_images = 11
            reaction_coord = np.linspace(0, 1, n_images)
            
            # Generate energy profile along path
            energy_profile = []
            for rc in reaction_coord:
                if rc == 0 or rc == 1:
                    energy = 0
                else:
                    # Parabolic barrier with some asymmetry
                    energy = E_act_base * (4 * rc * (1 - rc) + 0.1 * np.sin(2 * np.pi * rc))
                energy_profile.append(float(energy))
            
            path = {
                'path_id': f'path_{i:04d}',
                'mechanism': mechanism,
                'migrating_species': np.random.choice(['Ni', 'Cr', 'Al', 'vacancy']),
                'initial_site': {
                    'type': np.random.choice(['octahedral', 'tetrahedral', 'substitutional']),
                    'coordinates': list(np.random.uniform(0, 10, 3))
                },
                'final_site': {
                    'type': np.random.choice(['octahedral', 'tetrahedral', 'substitutional']),
                    'coordinates': list(np.random.uniform(0, 10, 3))
                },
                'activation_energy': float(E_act_base),
                'attempt_frequency': float(1e13 * np.random.uniform(0.5, 2)),  # Hz
                'prefactor': float(prefactor),
                'reaction_coordinate': list(reaction_coord),
                'energy_profile': energy_profile,
                'transition_state_index': int(np.argmax(energy_profile)),
                'temperature_range': [600, 1200]  # K
            }
            data['diffusion_paths'].append(path)
        
        return data
    
    def generate_surface_energies(self, n_surfaces: int = 25) -> Dict:
        """Generate surface and interface energies"""
        data = {
            'calculation_type': 'surface_energy',
            'method': 'DFT-PBE',
            'timestamp': self.timestamp,
            'units': {'energy': 'J/m^2', 'distance': 'Angstrom'},
            'surfaces': []
        }
        
        miller_indices = [[1,0,0], [1,1,0], [1,1,1], [2,1,0], [2,1,1], [3,1,0]]
        
        for i in range(n_surfaces):
            miller = miller_indices[i % len(miller_indices)]
            
            # Surface energy depends on Miller indices
            if miller == [1,1,1]:  # Close-packed
                E_surf_base = np.random.uniform(1.8, 2.0)
            elif miller == [1,0,0]:  # Less stable
                E_surf_base = np.random.uniform(2.2, 2.5)
            else:
                E_surf_base = np.random.uniform(2.0, 2.3)
            
            # Add environmental effects
            oxidation = np.random.choice([False, True], p=[0.7, 0.3])
            if oxidation:
                E_surf = E_surf_base * np.random.uniform(0.6, 0.8)
            else:
                E_surf = E_surf_base
            
            surface = {
                'surface_id': f'surf_{i:04d}',
                'miller_indices': miller,
                'surface_energy': float(E_surf),
                'work_function': float(np.random.uniform(4.5, 5.5)),  # eV
                'surface_stress': [
                    float(np.random.uniform(-0.5, 0.5)),
                    float(np.random.uniform(-0.5, 0.5))
                ],  # N/m
                'reconstruction': np.random.choice(['1x1', '2x1', '2x2', 'c(2x2)']),
                'oxidation_state': bool(oxidation),
                'adsorption_sites': {
                    'top': float(np.random.uniform(-2, -0.5)),
                    'bridge': float(np.random.uniform(-2.5, -1)),
                    'hollow': float(np.random.uniform(-3, -1.5))
                },
                'interlayer_spacing': float(self.lattice_param / np.sqrt(sum([m**2 for m in miller])))
            }
            data['surfaces'].append(surface)
        
        return data
    
    def save_data(self, data: Dict, filename: str, directory: str):
        """Save data to JSON file"""
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data.get('configurations', data.get('diffusion_paths', data.get('surfaces', []))))} entries to {filepath}")
    
    def generate_all_dft_data(self, output_dir: str = 'dft_calculations'):
        """Generate all DFT calculation datasets"""
        print("Generating DFT calculation data...")
        
        # Create subdirectories
        os.makedirs(f"{output_dir}/defect_energies", exist_ok=True)
        os.makedirs(f"{output_dir}/activation_barriers", exist_ok=True)
        os.makedirs(f"{output_dir}/surface_energies", exist_ok=True)
        
        # Generate and save datasets
        print("\n1. Generating vacancy formation energies...")
        vacancy_data = self.generate_vacancy_formation_energies(50)
        self.save_data(vacancy_data, "vacancy_formation.json", f"{output_dir}/defect_energies")
        
        print("\n2. Generating dislocation energies...")
        dislocation_data = self.generate_dislocation_energies(30)
        self.save_data(dislocation_data, "dislocation_energies.json", f"{output_dir}/defect_energies")
        
        print("\n3. Generating grain boundary energies...")
        gb_data = self.generate_grain_boundary_energies(40)
        self.save_data(gb_data, "grain_boundary_energies.json", f"{output_dir}/defect_energies")
        
        print("\n4. Generating activation barriers...")
        barrier_data = self.generate_activation_barriers(60)
        self.save_data(barrier_data, "diffusion_barriers.json", f"{output_dir}/activation_barriers")
        
        print("\n5. Generating surface energies...")
        surface_data = self.generate_surface_energies(25)
        self.save_data(surface_data, "surface_energies.json", f"{output_dir}/surface_energies")
        
        # Generate summary statistics
        self.generate_summary_statistics(output_dir)
        
    def generate_summary_statistics(self, output_dir: str):
        """Generate summary statistics for all DFT data"""
        summary = {
            'generated_at': self.timestamp,
            'total_calculations': 0,
            'calculation_types': {},
            'computational_cost_estimate': {
                'cpu_hours': 0,
                'memory_gb': 0
            }
        }
        
        # Count calculations and estimate computational cost
        calc_types = [
            ('defect_energies/vacancy_formation.json', 50, 100, 16),
            ('defect_energies/dislocation_energies.json', 30, 200, 32),
            ('defect_energies/grain_boundary_energies.json', 40, 150, 24),
            ('activation_barriers/diffusion_barriers.json', 60, 300, 16),
            ('surface_energies/surface_energies.json', 25, 80, 16)
        ]
        
        for filename, count, cpu_per_calc, mem_per_calc in calc_types:
            calc_type = filename.split('/')[0]
            summary['calculation_types'][calc_type] = count
            summary['total_calculations'] += count
            summary['computational_cost_estimate']['cpu_hours'] += count * cpu_per_calc
            summary['computational_cost_estimate']['memory_gb'] = max(
                summary['computational_cost_estimate']['memory_gb'],
                mem_per_calc
            )
        
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary: Generated {summary['total_calculations']} DFT calculations")
        print(f"Estimated computational cost: {summary['computational_cost_estimate']['cpu_hours']:,} CPU-hours")


if __name__ == "__main__":
    generator = DFTDataGenerator(seed=42)
    generator.generate_all_dft_data()