#!/usr/bin/env python3
"""
Atomic-Scale Simulation Dataset Generator
Generates realistic DFT and MD simulation data for quantum-enhanced inputs
"""

import numpy as np
import pandas as pd
import json
import h5py
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy import stats
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

@dataclass
class SimulationParameters:
    """Parameters for atomic-scale simulations"""
    temperature: float  # K
    pressure: float     # GPa
    crystal_structure: str  # FCC, BCC, HCP
    lattice_parameter: float  # Angstrom
    supercell_size: Tuple[int, int, int]
    k_point_density: float
    cutoff_energy: float  # eV
    exchange_correlation: str  # PBE, LDA, etc.

@dataclass
class DFTResults:
    """DFT calculation results"""
    formation_energy: float  # eV/atom
    activation_barrier: float  # eV
    surface_energy: float  # J/m²
    bulk_modulus: float  # GPa
    elastic_constants: List[float]  # C11, C12, C44
    band_gap: float  # eV
    magnetic_moment: float  # μB/atom

@dataclass
class MDResults:
    """Molecular dynamics simulation results"""
    grain_boundary_energy: float  # J/m²
    sliding_resistance: float  # MPa
    dislocation_mobility: float  # m/s/MPa
    diffusion_coefficient: float  # m²/s
    viscosity: float  # Pa·s
    stress_strain_data: List[Tuple[float, float]]

class AtomicSimulationDatasetGenerator:
    """Generator for atomic-scale simulation datasets"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.materials_db = self._initialize_materials_database()
        
    def _initialize_materials_database(self) -> Dict:
        """Initialize realistic materials properties database"""
        return {
            'Ni': {
                'lattice_param': 3.52,  # Angstrom
                'formation_energy_range': (-0.5, 0.2),  # eV/atom
                'activation_barrier_range': (0.8, 2.5),  # eV
                'surface_energy_range': (1.8, 2.5),  # J/m²
                'bulk_modulus_range': (180, 220),  # GPa
                'melting_point': 1728,  # K
            },
            'Al': {
                'lattice_param': 4.05,
                'formation_energy_range': (-0.3, 0.1),
                'activation_barrier_range': (0.6, 1.8),
                'surface_energy_range': (1.0, 1.5),
                'bulk_modulus_range': (70, 90),
                'melting_point': 933,
            },
            'Fe': {
                'lattice_param': 2.87,
                'formation_energy_range': (-0.4, 0.3),
                'activation_barrier_range': (1.0, 2.8),
                'surface_energy_range': (2.0, 2.8),
                'bulk_modulus_range': (160, 200),
                'melting_point': 1811,
            },
            'Cu': {
                'lattice_param': 3.61,
                'formation_energy_range': (-0.2, 0.1),
                'activation_barrier_range': (0.7, 2.0),
                'surface_energy_range': (1.5, 2.0),
                'bulk_modulus_range': (130, 160),
                'melting_point': 1358,
            }
        }
    
    def generate_dft_data(self, material: str, num_simulations: int = 1000) -> List[Dict]:
        """Generate DFT simulation data"""
        if material not in self.materials_db:
            raise ValueError(f"Material {material} not in database")
        
        material_props = self.materials_db[material]
        dft_data = []
        
        for i in range(num_simulations):
            # Generate simulation parameters
            temp = np.random.uniform(300, 1500)  # K
            pressure = np.random.uniform(0, 10)  # GPa
            
            # Generate DFT results with realistic correlations
            formation_energy = np.random.normal(
                np.mean(material_props['formation_energy_range']),
                np.std(material_props['formation_energy_range']) / 3
            )
            
            # Activation barrier correlates with formation energy
            activation_barrier = np.random.normal(
                np.mean(material_props['activation_barrier_range']),
                np.std(material_props['activation_barrier_range']) / 3
            ) + 0.3 * formation_energy
            
            surface_energy = np.random.normal(
                np.mean(material_props['surface_energy_range']),
                np.std(material_props['surface_energy_range']) / 3
            )
            
            bulk_modulus = np.random.normal(
                np.mean(material_props['bulk_modulus_range']),
                np.std(material_props['bulk_modulus_range']) / 3
            )
            
            # Elastic constants (correlated with bulk modulus)
            c11 = bulk_modulus * np.random.uniform(2.5, 3.5)
            c12 = bulk_modulus * np.random.uniform(0.8, 1.2)
            c44 = bulk_modulus * np.random.uniform(0.3, 0.7)
            
            # Band gap (metals have small/zero band gap)
            band_gap = np.random.exponential(0.1) if material in ['Ni', 'Fe'] else np.random.exponential(0.05)
            
            # Magnetic moment (only for magnetic materials)
            magnetic_moment = np.random.normal(0.6, 0.2) if material == 'Fe' else 0.0
            
            dft_result = {
                'simulation_id': f"DFT_{material}_{i:04d}",
                'material': material,
                'parameters': {
                    'temperature': temp,
                    'pressure': pressure,
                    'crystal_structure': 'FCC',
                    'lattice_parameter': material_props['lattice_param'],
                    'supercell_size': [4, 4, 4],
                    'k_point_density': np.random.uniform(0.02, 0.1),
                    'cutoff_energy': np.random.uniform(400, 600),
                    'exchange_correlation': np.random.choice(['PBE', 'LDA', 'PBE0'])
                },
                'results': {
                    'formation_energy': formation_energy,
                    'activation_barrier': activation_barrier,
                    'surface_energy': surface_energy,
                    'bulk_modulus': bulk_modulus,
                    'elastic_constants': [c11, c12, c44],
                    'band_gap': band_gap,
                    'magnetic_moment': magnetic_moment
                },
                'metadata': {
                    'convergence_criteria': 1e-6,
                    'total_energy': np.random.uniform(-1000, -500),
                    'forces_converged': True,
                    'stress_converged': True
                }
            }
            
            dft_data.append(dft_result)
        
        return dft_data
    
    def generate_md_data(self, material: str, num_simulations: int = 500) -> List[Dict]:
        """Generate MD simulation data"""
        if material not in self.materials_db:
            raise ValueError(f"Material {material} not in database")
        
        material_props = self.materials_db[material]
        md_data = []
        
        for i in range(num_simulations):
            # Generate simulation parameters
            temp = np.random.uniform(300, 1200)  # K
            pressure = np.random.uniform(0, 5)  # GPa
            
            # Generate MD results with temperature dependence
            temp_factor = temp / material_props['melting_point']
            
            # Grain boundary energy (decreases with temperature)
            gb_energy = np.random.normal(0.8, 0.2) * (1 - 0.3 * temp_factor)
            
            # Sliding resistance (decreases with temperature)
            sliding_resistance = np.random.normal(200, 50) * (1 - 0.5 * temp_factor)
            
            # Dislocation mobility (increases with temperature)
            dislocation_mobility = np.random.exponential(1e-6) * np.exp(temp_factor * 2)
            
            # Diffusion coefficient (Arrhenius behavior)
            d0 = np.random.uniform(1e-5, 1e-4)  # m²/s
            q = np.random.uniform(1.5, 2.5)  # eV
            diffusion_coeff = d0 * np.exp(-q / (8.617e-5 * temp))
            
            # Viscosity (decreases with temperature)
            viscosity = np.random.exponential(1e-3) * np.exp(1000 / temp)
            
            # Generate stress-strain curve
            strain_points = np.linspace(0, 0.1, 50)
            stress_points = self._generate_stress_strain_curve(strain_points, temp_factor)
            
            md_result = {
                'simulation_id': f"MD_{material}_{i:04d}",
                'material': material,
                'parameters': {
                    'temperature': temp,
                    'pressure': pressure,
                    'timestep': np.random.uniform(1e-15, 2e-15),  # s
                    'total_time': np.random.uniform(1e-9, 1e-8),  # s
                    'ensemble': np.random.choice(['NVT', 'NPT', 'NVE']),
                    'boundary_conditions': 'periodic'
                },
                'results': {
                    'grain_boundary_energy': gb_energy,
                    'sliding_resistance': sliding_resistance,
                    'dislocation_mobility': dislocation_mobility,
                    'diffusion_coefficient': diffusion_coeff,
                    'viscosity': viscosity,
                    'stress_strain_data': list(zip(strain_points, stress_points))
                },
                'metadata': {
                    'equilibration_time': np.random.uniform(0.1, 0.5),
                    'production_time': np.random.uniform(0.5, 0.9),
                    'thermostat': 'Nose-Hoover',
                    'barostat': 'Parrinello-Rahman'
                }
            }
            
            md_data.append(md_result)
        
        return md_data
    
    def _generate_stress_strain_curve(self, strain_points: np.ndarray, temp_factor: float) -> np.ndarray:
        """Generate realistic stress-strain curve"""
        # Elastic region
        young_modulus = np.random.uniform(100, 300)  # GPa
        yield_strength = np.random.uniform(200, 800)  # MPa
        
        # Temperature effect on yield strength
        yield_strength *= (1 - 0.4 * temp_factor)
        
        stress_points = []
        for strain in strain_points:
            if strain < yield_strength / young_modulus:
                # Elastic region
                stress = young_modulus * strain * 1000  # Convert to MPa
            else:
                # Plastic region with work hardening
                plastic_strain = strain - yield_strength / young_modulus
                hardening_modulus = np.random.uniform(1000, 5000)  # MPa
                stress = yield_strength + hardening_modulus * plastic_strain
            
            # Add noise
            stress += np.random.normal(0, stress * 0.02)
            stress_points.append(max(0, stress))
        
        return np.array(stress_points)
    
    def generate_defect_data(self, material: str, num_simulations: int = 300) -> List[Dict]:
        """Generate defect formation energy data"""
        if material not in self.materials_db:
            raise ValueError(f"Material {material} not in database")
        
        defect_data = []
        defect_types = ['vacancy', 'interstitial', 'antisite', 'dislocation', 'grain_boundary']
        
        for i in range(num_simulations):
            defect_type = np.random.choice(defect_types)
            
            # Defect-specific formation energies
            if defect_type == 'vacancy':
                formation_energy = np.random.normal(1.5, 0.3)
            elif defect_type == 'interstitial':
                formation_energy = np.random.normal(3.0, 0.5)
            elif defect_type == 'antisite':
                formation_energy = np.random.normal(0.8, 0.2)
            elif defect_type == 'dislocation':
                formation_energy = np.random.normal(0.5, 0.1)
            else:  # grain_boundary
                formation_energy = np.random.normal(0.3, 0.05)
            
            defect_result = {
                'simulation_id': f"DEFECT_{material}_{i:04d}",
                'material': material,
                'defect_type': defect_type,
                'formation_energy': formation_energy,
                'migration_barrier': np.random.normal(0.8, 0.2),
                'binding_energy': np.random.normal(-0.5, 0.3),
                'volume_change': np.random.normal(0.1, 0.05),  # Ω/Ω₀
                'metadata': {
                    'supercell_size': [6, 6, 6],
                    'k_points': [4, 4, 4],
                    'convergence': 1e-6
                }
            }
            
            defect_data.append(defect_result)
        
        return defect_data
    
    def save_dataset(self, data: List[Dict], filename: str, format: str = 'json'):
        """Save dataset to file"""
        if format == 'json':
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == 'hdf5':
            with h5py.File(filename, 'w') as f:
                self._save_to_hdf5(data, f)
        elif format == 'csv':
            # Flatten data for CSV
            flattened_data = self._flatten_data(data)
            df = pd.DataFrame(flattened_data)
            df.to_csv(filename, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _save_to_hdf5(self, data: List[Dict], h5file):
        """Save data to HDF5 format"""
        for i, item in enumerate(data):
            group = h5file.create_group(f'simulation_{i}')
            self._dict_to_hdf5(item, group)
    
    def _dict_to_hdf5(self, d: Dict, group):
        """Recursively save dictionary to HDF5"""
        for key, value in d.items():
            if isinstance(value, dict):
                subgroup = group.create_group(key)
                self._dict_to_hdf5(value, subgroup)
            elif isinstance(value, list):
                if value and isinstance(value[0], (int, float)):
                    group.create_dataset(key, data=np.array(value))
                else:
                    subgroup = group.create_group(key)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_group = subgroup.create_group(f'item_{i}')
                            self._dict_to_hdf5(item, item_group)
                        else:
                            # Convert to string for HDF5 compatibility
                            subgroup.create_dataset(f'item_{i}', data=str(item))
            else:
                # Convert strings to bytes for HDF5 compatibility
                if isinstance(value, str):
                    group.create_dataset(key, data=value.encode('utf-8'))
                else:
                    group.create_dataset(key, data=value)
    
    def _flatten_data(self, data: List[Dict]) -> List[Dict]:
        """Flatten nested data for CSV export"""
        flattened = []
        for item in data:
            flat_item = {}
            self._flatten_dict(item, flat_item)
            flattened.append(flat_item)
        return flattened
    
    def _flatten_dict(self, d: Dict, flat_dict: Dict, prefix: str = ''):
        """Recursively flatten dictionary"""
        for key, value in d.items():
            new_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                self._flatten_dict(value, flat_dict, new_key)
            elif isinstance(value, list):
                if value and isinstance(value[0], (int, float)):
                    flat_dict[new_key] = str(value)
                else:
                    flat_dict[new_key] = str(value)
            else:
                flat_dict[new_key] = value

def main():
    """Generate comprehensive atomic simulation dataset"""
    generator = AtomicSimulationDatasetGenerator(seed=42)
    
    # Generate data for different materials
    materials = ['Ni', 'Al', 'Fe', 'Cu']
    
    print("Generating atomic-scale simulation dataset...")
    
    # Create output directory
    output_dir = Path("atomic_simulation_data")
    output_dir.mkdir(exist_ok=True)
    
    all_dft_data = []
    all_md_data = []
    all_defect_data = []
    
    for material in materials:
        print(f"Generating data for {material}...")
        
        # Generate DFT data
        dft_data = generator.generate_dft_data(material, num_simulations=250)
        all_dft_data.extend(dft_data)
        
        # Generate MD data
        md_data = generator.generate_md_data(material, num_simulations=125)
        all_md_data.extend(md_data)
        
        # Generate defect data
        defect_data = generator.generate_defect_data(material, num_simulations=75)
        all_defect_data.extend(defect_data)
    
    # Save datasets
    print("Saving datasets...")
    
    # Save as JSON
    generator.save_dataset(all_dft_data, output_dir / "dft_simulations.json")
    generator.save_dataset(all_md_data, output_dir / "md_simulations.json")
    generator.save_dataset(all_defect_data, output_dir / "defect_simulations.json")
    
    # Save as HDF5
    generator.save_dataset(all_dft_data, output_dir / "dft_simulations.h5", format='hdf5')
    generator.save_dataset(all_md_data, output_dir / "md_simulations.h5", format='hdf5')
    generator.save_dataset(all_defect_data, output_dir / "defect_simulations.h5", format='hdf5')
    
    # Save as CSV
    generator.save_dataset(all_dft_data, output_dir / "dft_simulations.csv", format='csv')
    generator.save_dataset(all_md_data, output_dir / "md_simulations.csv", format='csv')
    generator.save_dataset(all_defect_data, output_dir / "defect_simulations.csv", format='csv')
    
    # Create summary statistics
    create_summary_statistics(all_dft_data, all_md_data, all_defect_data, output_dir)
    
    print(f"Dataset generation complete! Files saved to {output_dir}")
    print(f"Total simulations: {len(all_dft_data) + len(all_md_data) + len(all_defect_data)}")

def create_summary_statistics(dft_data, md_data, defect_data, output_dir):
    """Create summary statistics and visualizations"""
    
    # Create summary report
    summary = {
        'dataset_info': {
            'total_dft_simulations': len(dft_data),
            'total_md_simulations': len(md_data),
            'total_defect_simulations': len(defect_data),
            'materials': list(set([item['material'] for item in dft_data])),
            'generation_date': pd.Timestamp.now().isoformat()
        },
        'dft_statistics': {
            'formation_energy_range': [
                min([item['results']['formation_energy'] for item in dft_data]),
                max([item['results']['formation_energy'] for item in dft_data])
            ],
            'activation_barrier_range': [
                min([item['results']['activation_barrier'] for item in dft_data]),
                max([item['results']['activation_barrier'] for item in dft_data])
            ],
            'surface_energy_range': [
                min([item['results']['surface_energy'] for item in dft_data]),
                max([item['results']['surface_energy'] for item in dft_data])
            ]
        },
        'md_statistics': {
            'temperature_range': [
                min([item['parameters']['temperature'] for item in md_data]),
                max([item['parameters']['temperature'] for item in md_data])
            ],
            'diffusion_coefficient_range': [
                min([item['results']['diffusion_coefficient'] for item in md_data]),
                max([item['results']['diffusion_coefficient'] for item in md_data])
            ]
        }
    }
    
    with open(output_dir / "dataset_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    
    # DFT formation energy distribution
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Formation energies by material
    materials = list(set([item['material'] for item in dft_data]))
    for i, material in enumerate(materials):
        material_data = [item['results']['formation_energy'] for item in dft_data if item['material'] == material]
        axes[0, 0].hist(material_data, alpha=0.7, label=material, bins=30)
    axes[0, 0].set_xlabel('Formation Energy (eV/atom)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('DFT Formation Energy Distribution')
    axes[0, 0].legend()
    
    # Activation barriers vs formation energies
    formation_energies = [item['results']['formation_energy'] for item in dft_data]
    activation_barriers = [item['results']['activation_barrier'] for item in dft_data]
    axes[0, 1].scatter(formation_energies, activation_barriers, alpha=0.6)
    axes[0, 1].set_xlabel('Formation Energy (eV/atom)')
    axes[0, 1].set_ylabel('Activation Barrier (eV)')
    axes[0, 1].set_title('Formation Energy vs Activation Barrier')
    
    # MD temperature vs diffusion coefficient
    temperatures = [item['parameters']['temperature'] for item in md_data]
    diffusion_coeffs = [item['results']['diffusion_coefficient'] for item in md_data]
    axes[1, 0].scatter(temperatures, diffusion_coeffs, alpha=0.6)
    axes[1, 0].set_xlabel('Temperature (K)')
    axes[1, 0].set_ylabel('Diffusion Coefficient (m²/s)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_title('Temperature vs Diffusion Coefficient')
    
    # Defect formation energies by type
    defect_types = list(set([item['defect_type'] for item in defect_data]))
    defect_energies = {dt: [item['formation_energy'] for item in defect_data if item['defect_type'] == dt] 
                      for dt in defect_types}
    axes[1, 1].boxplot([defect_energies[dt] for dt in defect_types], labels=defect_types)
    axes[1, 1].set_ylabel('Formation Energy (eV)')
    axes[1, 1].set_title('Defect Formation Energies by Type')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_summary.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()