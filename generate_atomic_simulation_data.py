"""
Atomic-Scale Simulation Data Generator
Generates synthetic DFT and MD simulation data for quantum-enhanced materials modeling.

This script creates physically realistic synthetic data for:
1. DFT Calculations: Formation energies, activation barriers, surface energies
2. MD Simulations: Grain boundary sliding, dislocation mobility
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class DFTDataGenerator:
    """Generates synthetic Density Functional Theory calculation outputs."""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.materials = ['Fe', 'Ni', 'Al', 'Cu', 'Ti', 'Cr', 'Steel-316', 'Inconel-718']
        self.defect_types = ['vacancy', 'interstitial', 'substitutional', 'divacancy']
        
    def generate_defect_formation_energies(self, n_samples=500):
        """
        Generate formation energies for various defects.
        Units: eV (electron volts)
        
        Typical ranges:
        - Vacancies: 0.5-3.0 eV
        - Interstitials: 2.0-6.0 eV
        - Grain boundaries: 0.3-2.0 J/m²
        """
        data = []
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            defect_type = np.random.choice(self.defect_types)
            temperature = np.random.uniform(300, 1500)  # K
            
            # Base formation energy depends on defect type
            if defect_type == 'vacancy':
                base_energy = np.random.uniform(0.5, 3.0)
            elif defect_type == 'interstitial':
                base_energy = np.random.uniform(2.0, 6.0)
            elif defect_type == 'substitutional':
                base_energy = np.random.uniform(0.3, 2.5)
            else:  # divacancy
                base_energy = np.random.uniform(0.8, 4.5)
            
            # Add temperature dependence
            thermal_correction = -0.0001 * (temperature - 300)
            formation_energy = base_energy + thermal_correction
            
            # Additional properties
            data.append({
                'material': material,
                'defect_type': defect_type,
                'temperature_K': round(temperature, 2),
                'formation_energy_eV': round(formation_energy, 4),
                'lattice_parameter_A': round(np.random.uniform(2.8, 4.2), 4),
                'defect_volume_A3': round(np.random.uniform(10, 50), 3),
                'charge_state': np.random.choice([-2, -1, 0, 1, 2]),
                'convergence_tolerance': round(np.random.uniform(1e-6, 1e-5), 8),
                'k_points': f"{np.random.choice([4, 6, 8])}x{np.random.choice([4, 6, 8])}x{np.random.choice([4, 6, 8])}",
                'energy_cutoff_eV': np.random.choice([400, 500, 600, 700]),
            })
        
        return pd.DataFrame(data)
    
    def generate_grain_boundary_energies(self, n_samples=200):
        """
        Generate grain boundary energies.
        Units: J/m² (Joules per square meter)
        
        Typical range: 0.2-2.0 J/m²
        """
        data = []
        
        # Common grain boundary types
        gb_types = ['tilt', 'twist', 'mixed', 'symmetric_tilt', 'asymmetric_tilt']
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            gb_type = np.random.choice(gb_types)
            
            # Misorientation angle (degrees)
            misorientation = np.random.uniform(5, 60)
            
            # GB energy typically has a minimum around 30-40 degrees
            theta_opt = 35.0
            base_energy = 0.3 + 0.8 * (1 - np.exp(-((misorientation - theta_opt) / 20) ** 2))
            
            # Add some noise
            gb_energy = base_energy + np.random.normal(0, 0.1)
            gb_energy = max(0.2, min(2.0, gb_energy))  # Clamp to physical range
            
            data.append({
                'material': material,
                'gb_type': gb_type,
                'misorientation_deg': round(misorientation, 2),
                'gb_energy_J_m2': round(gb_energy, 4),
                'gb_width_nm': round(np.random.uniform(0.5, 2.0), 3),
                'grain_size_um': round(np.random.uniform(1, 100), 2),
                'segregation_energy_eV': round(np.random.uniform(-0.5, 0.5), 4),
                'diffusivity_enhancement': round(10 ** np.random.uniform(1, 4), 2),
            })
        
        return pd.DataFrame(data)
    
    def generate_activation_barriers(self, n_samples=400):
        """
        Generate activation energy barriers for diffusion processes.
        Units: eV
        
        Typical ranges:
        - Vacancy migration: 0.5-1.5 eV
        - Interstitial migration: 0.1-0.8 eV
        - Substitutional diffusion: 1.0-3.0 eV
        """
        data = []
        
        diffusion_mechanisms = [
            'vacancy_migration', 
            'interstitial_migration',
            'substitutional_diffusion',
            'grain_boundary_diffusion',
            'dislocation_pipe_diffusion',
            'solute_drag'
        ]
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            mechanism = np.random.choice(diffusion_mechanisms)
            temperature = np.random.uniform(300, 1500)  # K
            
            # Base activation energy depends on mechanism
            if mechanism == 'vacancy_migration':
                Q0 = np.random.uniform(0.5, 1.5)
            elif mechanism == 'interstitial_migration':
                Q0 = np.random.uniform(0.1, 0.8)
            elif mechanism == 'substitutional_diffusion':
                Q0 = np.random.uniform(1.0, 3.0)
            elif mechanism == 'grain_boundary_diffusion':
                Q0 = np.random.uniform(0.4, 1.2)
            elif mechanism == 'dislocation_pipe_diffusion':
                Q0 = np.random.uniform(0.3, 1.0)
            else:  # solute_drag
                Q0 = np.random.uniform(0.6, 2.0)
            
            # Calculate diffusion coefficient using Arrhenius equation
            # D = D0 * exp(-Q/kT)
            k_B = 8.617e-5  # eV/K
            D0 = 10 ** np.random.uniform(-5, -2)  # m²/s
            D = D0 * np.exp(-Q0 / (k_B * temperature))
            
            data.append({
                'material': material,
                'mechanism': mechanism,
                'temperature_K': round(temperature, 2),
                'activation_energy_eV': round(Q0, 4),
                'pre_exponential_factor_m2_s': round(D0, 8),
                'diffusion_coefficient_m2_s': round(D, 12),
                'attempt_frequency_THz': round(np.random.uniform(1, 50), 3),
                'migration_path_length_A': round(np.random.uniform(2, 6), 3),
                'saddle_point_energy_eV': round(Q0 + np.random.uniform(0, 0.3), 4),
            })
        
        return pd.DataFrame(data)
    
    def generate_surface_energies(self, n_samples=300):
        """
        Generate surface energies for different crystallographic orientations.
        Units: J/m²
        
        Typical range: 0.5-3.0 J/m²
        """
        data = []
        
        # Common surface orientations
        orientations = ['(100)', '(110)', '(111)', '(112)', '(210)', '(211)']
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            orientation = np.random.choice(orientations)
            
            # Different orientations have different surface energies
            # (111) typically has lowest energy for FCC
            base_energies = {
                '(100)': (1.5, 2.5),
                '(110)': (1.8, 2.8),
                '(111)': (1.0, 2.0),
                '(112)': (1.6, 2.6),
                '(210)': (1.7, 2.7),
                '(211)': (1.6, 2.6),
            }
            
            e_min, e_max = base_energies[orientation]
            surface_energy = np.random.uniform(e_min, e_max)
            
            # Temperature effect
            temperature = np.random.uniform(0, 1500)  # K
            surface_energy -= 0.0002 * temperature  # Slight decrease with T
            
            data.append({
                'material': material,
                'surface_orientation': orientation,
                'surface_energy_J_m2': round(surface_energy, 4),
                'temperature_K': round(temperature, 2),
                'work_of_adhesion_J_m2': round(surface_energy * np.random.uniform(1.5, 2.5), 4),
                'surface_stress_N_m': round(surface_energy + np.random.uniform(-0.3, 0.3), 4),
                'atomic_density_per_A2': round(np.random.uniform(0.1, 0.2), 4),
                'surface_relaxation_percent': round(np.random.uniform(-5, 2), 2),
            })
        
        return pd.DataFrame(data)


class MDDataGenerator:
    """Generates synthetic Molecular Dynamics simulation outputs."""
    
    def __init__(self, seed=42):
        np.random.seed(seed + 100)  # Different seed from DFT
        self.materials = ['Fe', 'Ni', 'Al', 'Cu', 'Ti', 'Cr', 'Steel-316', 'Inconel-718']
        
    def generate_grain_boundary_sliding_data(self, n_samples=350):
        """
        Generate grain boundary sliding simulation data.
        
        Key outputs:
        - Shear stress vs displacement
        - Sliding resistance
        - Activation energy for sliding
        """
        data = []
        
        gb_types = ['tilt', 'twist', 'mixed', 'symmetric_tilt', 'asymmetric_tilt']
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            gb_type = np.random.choice(gb_types)
            temperature = np.random.uniform(300, 1500)  # K
            
            # Shear stress (MPa)
            applied_shear_stress = np.random.uniform(10, 500)
            
            # Critical shear stress for sliding (temperature dependent)
            tau_c = 100 * (1500 - temperature) / 1200  # Decreases with T
            tau_c += np.random.normal(0, 10)
            
            # Sliding displacement (nm)
            if applied_shear_stress > tau_c:
                displacement = (applied_shear_stress - tau_c) * np.random.uniform(0.1, 1.0)
            else:
                displacement = applied_shear_stress / tau_c * np.random.uniform(0.01, 0.1)
            
            # Sliding velocity
            sliding_velocity = displacement / np.random.uniform(100, 1000)  # nm/ps
            
            data.append({
                'material': material,
                'gb_type': gb_type,
                'temperature_K': round(temperature, 2),
                'misorientation_deg': round(np.random.uniform(5, 60), 2),
                'applied_shear_stress_MPa': round(applied_shear_stress, 2),
                'critical_shear_stress_MPa': round(max(0, tau_c), 2),
                'sliding_displacement_nm': round(max(0, displacement), 4),
                'sliding_velocity_nm_ps': round(sliding_velocity, 6),
                'activation_energy_eV': round(np.random.uniform(0.3, 1.5), 4),
                'simulation_time_ps': round(np.random.uniform(100, 1000), 2),
                'box_size_nm': round(np.random.uniform(10, 50), 2),
                'num_atoms': np.random.choice([10000, 50000, 100000, 500000]),
                'timestep_fs': np.random.choice([0.5, 1.0, 2.0]),
            })
        
        return pd.DataFrame(data)
    
    def generate_dislocation_mobility_data(self, n_samples=400):
        """
        Generate dislocation mobility simulation data.
        
        Key outputs:
        - Dislocation velocity vs stress
        - Mobility coefficients
        - Interaction energies
        """
        data = []
        
        dislocation_types = ['edge', 'screw', 'mixed', 'prismatic', 'basal']
        slip_systems = ['<110>{111}', '<111>{110}', '<100>{001}', '<110>{001}']
        
        for _ in range(n_samples):
            material = np.random.choice(self.materials)
            disl_type = np.random.choice(dislocation_types)
            slip_system = np.random.choice(slip_systems)
            temperature = np.random.uniform(300, 1500)  # K
            
            # Applied stress (MPa)
            stress = np.random.uniform(10, 1000)
            
            # Peierls stress (resistance to motion)
            peierls_stress = np.random.uniform(10, 200) * (1500 - temperature) / 1200
            
            # Dislocation velocity (m/s)
            # v = M * (τ - τ_p)^m where M is mobility, m is stress exponent
            if stress > peierls_stress:
                mobility = 10 ** np.random.uniform(-6, -3)
                stress_exponent = np.random.uniform(1, 3)
                velocity = mobility * (stress - peierls_stress) ** stress_exponent
            else:
                velocity = 0
            
            # Burgers vector magnitude
            burgers_vector = np.random.uniform(2.5, 3.5)  # Angstroms
            
            data.append({
                'material': material,
                'dislocation_type': disl_type,
                'slip_system': slip_system,
                'temperature_K': round(temperature, 2),
                'applied_stress_MPa': round(stress, 2),
                'peierls_stress_MPa': round(max(0, peierls_stress), 2),
                'dislocation_velocity_m_s': round(velocity, 8),
                'mobility_coefficient': round(10 ** np.random.uniform(-6, -3), 10),
                'stress_exponent': round(np.random.uniform(1, 3), 3),
                'burgers_vector_A': round(burgers_vector, 3),
                'dislocation_density_m2': round(10 ** np.random.uniform(12, 15), 2),
                'line_tension_eV_A': round(np.random.uniform(1, 5), 3),
                'core_energy_eV_A': round(np.random.uniform(0.5, 3), 3),
                'interaction_energy_eV': round(np.random.uniform(-0.5, 0.5), 4),
                'simulation_time_ps': round(np.random.uniform(50, 500), 2),
                'num_atoms': np.random.choice([50000, 100000, 500000, 1000000]),
            })
        
        return pd.DataFrame(data)
    
    def generate_trajectory_data(self, n_atoms=1000, n_steps=100):
        """
        Generate a sample MD trajectory.
        This is a simplified version - real trajectories would be much larger.
        """
        # Generate initial positions (random fcc lattice-like structure)
        positions = np.random.uniform(-10, 10, size=(n_steps, n_atoms, 3))
        
        # Generate velocities (Maxwell-Boltzmann-like distribution)
        velocities = np.random.normal(0, 1, size=(n_steps, n_atoms, 3))
        
        # Generate forces
        forces = np.random.normal(0, 0.1, size=(n_steps, n_atoms, 3))
        
        # Calculate energies
        timesteps = np.arange(n_steps) * 2.0  # fs
        potential_energy = -1000 + np.random.normal(0, 10, n_steps)
        kinetic_energy = np.sum(velocities**2, axis=(1, 2)) / 2
        total_energy = potential_energy + kinetic_energy
        
        trajectory = {
            'timesteps_fs': timesteps.tolist(),
            'potential_energy_eV': potential_energy.tolist(),
            'kinetic_energy_eV': kinetic_energy.tolist(),
            'total_energy_eV': total_energy.tolist(),
            'temperature_K': (kinetic_energy / (1.5 * n_atoms * 8.617e-5)).tolist(),
            'pressure_GPa': np.random.uniform(0, 0.1, n_steps).tolist(),
            'num_atoms': n_atoms,
            'num_steps': n_steps,
        }
        
        return trajectory


def generate_complete_dataset(output_dir='atomic_simulation_data'):
    """Generate complete atomic-scale simulation dataset."""
    
    print("=" * 70)
    print("ATOMIC-SCALE SIMULATION DATA GENERATOR")
    print("Quantum-Enhanced Materials Modeling")
    print("=" * 70)
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize generators
    dft_gen = DFTDataGenerator(seed=42)
    md_gen = MDDataGenerator(seed=42)
    
    print("📊 Generating DFT Calculation Data...")
    print("-" * 70)
    
    # Generate DFT data
    print("  ⚛️  Defect formation energies...")
    defect_energies = dft_gen.generate_defect_formation_energies(n_samples=500)
    defect_energies.to_csv(output_path / 'dft_defect_formation_energies.csv', index=False)
    print(f"     ✓ Generated {len(defect_energies)} samples")
    
    print("  🔗 Grain boundary energies...")
    gb_energies = dft_gen.generate_grain_boundary_energies(n_samples=200)
    gb_energies.to_csv(output_path / 'dft_grain_boundary_energies.csv', index=False)
    print(f"     ✓ Generated {len(gb_energies)} samples")
    
    print("  🎯 Activation energy barriers...")
    activation_barriers = dft_gen.generate_activation_barriers(n_samples=400)
    activation_barriers.to_csv(output_path / 'dft_activation_barriers.csv', index=False)
    print(f"     ✓ Generated {len(activation_barriers)} samples")
    
    print("  📐 Surface energies...")
    surface_energies = dft_gen.generate_surface_energies(n_samples=300)
    surface_energies.to_csv(output_path / 'dft_surface_energies.csv', index=False)
    print(f"     ✓ Generated {len(surface_energies)} samples")
    
    print()
    print("🔬 Generating MD Simulation Data...")
    print("-" * 70)
    
    # Generate MD data
    print("  🔀 Grain boundary sliding...")
    gb_sliding = md_gen.generate_grain_boundary_sliding_data(n_samples=350)
    gb_sliding.to_csv(output_path / 'md_grain_boundary_sliding.csv', index=False)
    print(f"     ✓ Generated {len(gb_sliding)} samples")
    
    print("  ↔️  Dislocation mobility...")
    disl_mobility = md_gen.generate_dislocation_mobility_data(n_samples=400)
    disl_mobility.to_csv(output_path / 'md_dislocation_mobility.csv', index=False)
    print(f"     ✓ Generated {len(disl_mobility)} samples")
    
    print("  🎬 Sample trajectory...")
    trajectory = md_gen.generate_trajectory_data(n_atoms=1000, n_steps=100)
    with open(output_path / 'md_sample_trajectory.json', 'w') as f:
        json.dump(trajectory, f, indent=2)
    print(f"     ✓ Generated trajectory with {trajectory['num_atoms']} atoms, {trajectory['num_steps']} steps")
    
    print()
    print("📈 Generating Summary Statistics...")
    print("-" * 70)
    
    # Generate summary
    summary = {
        'generation_date': datetime.now().isoformat(),
        'datasets': {
            'dft_defect_formation_energies': {
                'samples': len(defect_energies),
                'file': 'dft_defect_formation_energies.csv',
                'description': 'Formation energies for vacancies, interstitials, and other defects',
                'key_columns': list(defect_energies.columns),
                'statistics': {
                    'mean_formation_energy_eV': float(defect_energies['formation_energy_eV'].mean()),
                    'std_formation_energy_eV': float(defect_energies['formation_energy_eV'].std()),
                    'temperature_range_K': [float(defect_energies['temperature_K'].min()), 
                                           float(defect_energies['temperature_K'].max())],
                }
            },
            'dft_grain_boundary_energies': {
                'samples': len(gb_energies),
                'file': 'dft_grain_boundary_energies.csv',
                'description': 'Grain boundary energies for various misorientation angles',
                'key_columns': list(gb_energies.columns),
                'statistics': {
                    'mean_gb_energy_J_m2': float(gb_energies['gb_energy_J_m2'].mean()),
                    'std_gb_energy_J_m2': float(gb_energies['gb_energy_J_m2'].std()),
                    'misorientation_range_deg': [float(gb_energies['misorientation_deg'].min()), 
                                                 float(gb_energies['misorientation_deg'].max())],
                }
            },
            'dft_activation_barriers': {
                'samples': len(activation_barriers),
                'file': 'dft_activation_barriers.csv',
                'description': 'Activation energy barriers for diffusion mechanisms',
                'key_columns': list(activation_barriers.columns),
                'statistics': {
                    'mean_activation_energy_eV': float(activation_barriers['activation_energy_eV'].mean()),
                    'std_activation_energy_eV': float(activation_barriers['activation_energy_eV'].std()),
                    'mechanisms': list(activation_barriers['mechanism'].unique()),
                }
            },
            'dft_surface_energies': {
                'samples': len(surface_energies),
                'file': 'dft_surface_energies.csv',
                'description': 'Surface energies for different crystallographic orientations',
                'key_columns': list(surface_energies.columns),
                'statistics': {
                    'mean_surface_energy_J_m2': float(surface_energies['surface_energy_J_m2'].mean()),
                    'std_surface_energy_J_m2': float(surface_energies['surface_energy_J_m2'].std()),
                    'orientations': list(surface_energies['surface_orientation'].unique()),
                }
            },
            'md_grain_boundary_sliding': {
                'samples': len(gb_sliding),
                'file': 'md_grain_boundary_sliding.csv',
                'description': 'Grain boundary sliding resistance and displacement data',
                'key_columns': list(gb_sliding.columns),
                'statistics': {
                    'mean_critical_shear_stress_MPa': float(gb_sliding['critical_shear_stress_MPa'].mean()),
                    'mean_sliding_velocity_nm_ps': float(gb_sliding['sliding_velocity_nm_ps'].mean()),
                }
            },
            'md_dislocation_mobility': {
                'samples': len(disl_mobility),
                'file': 'md_dislocation_mobility.csv',
                'description': 'Dislocation velocity and mobility coefficients',
                'key_columns': list(disl_mobility.columns),
                'statistics': {
                    'mean_peierls_stress_MPa': float(disl_mobility['peierls_stress_MPa'].mean()),
                    'dislocation_types': list(disl_mobility['dislocation_type'].unique()),
                }
            },
            'md_sample_trajectory': {
                'file': 'md_sample_trajectory.json',
                'description': 'Sample MD trajectory data (positions, velocities, energies)',
                'num_atoms': trajectory['num_atoms'],
                'num_steps': trajectory['num_steps'],
            }
        },
        'total_samples': (
            len(defect_energies) + len(gb_energies) + len(activation_barriers) + 
            len(surface_energies) + len(gb_sliding) + len(disl_mobility)
        ),
        'materials_covered': sorted(list(set(
            defect_energies['material'].unique().tolist() +
            gb_energies['material'].unique().tolist()
        ))),
        'usage_notes': {
            'DFT_data': 'Can be used to parameterize phase-field models, crystal plasticity models, or train ML surrogate models',
            'MD_data': 'Provides kinetic information for grain boundary sliding and dislocation dynamics',
            'integration': 'These atomic-scale properties can be upscaled into continuum models for creep simulation',
            'units': {
                'energy': 'eV (electron volts)',
                'surface_energy': 'J/m²',
                'temperature': 'K (Kelvin)',
                'stress': 'MPa (Megapascals)',
                'length': 'Angstroms (Å) or nm',
                'time': 'fs (femtoseconds) or ps (picoseconds)',
            }
        }
    }
    
    with open(output_path / 'dataset_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  ✓ Summary saved to dataset_summary.json")
    
    print()
    print("=" * 70)
    print("✅ DATASET GENERATION COMPLETE")
    print("=" * 70)
    print(f"📁 Output directory: {output_path.absolute()}")
    print(f"📊 Total samples generated: {summary['total_samples']}")
    print(f"🗂️  Total files created: {len(summary['datasets']) + 1}")
    print()
    print("Files created:")
    for dataset_name, dataset_info in summary['datasets'].items():
        if 'file' in dataset_info:
            print(f"  • {dataset_info['file']}")
            if 'samples' in dataset_info:
                print(f"    ({dataset_info['samples']} samples)")
    print(f"  • dataset_summary.json")
    print()
    print("🎯 Next steps:")
    print("  1. Review the generated datasets")
    print("  2. Use DFT data to parameterize your continuum models")
    print("  3. Integrate MD data into your creep simulation framework")
    print("  4. Train ML surrogate models if needed")
    print()
    
    return summary


if __name__ == '__main__':
    summary = generate_complete_dataset()
