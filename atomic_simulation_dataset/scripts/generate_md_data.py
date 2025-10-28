#!/usr/bin/env python3
"""
Generate synthetic Molecular Dynamics simulation data for material modeling.
Includes grain boundary sliding and dislocation mobility simulations.
"""

import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd

class MDDataGenerator:
    """Generate realistic MD simulation outputs"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.timestamp = datetime.now().isoformat()
        
        # Simulation parameters
        self.temperature_range = [600, 1200]  # K
        self.timestep = 1.0  # fs
        self.box_size = [100, 100, 100]  # Angstroms
        
    def generate_grain_boundary_sliding(self, n_simulations: int = 20) -> Dict:
        """Generate grain boundary sliding simulation data"""
        data = {
            'simulation_type': 'grain_boundary_sliding',
            'method': 'MD-EAM',  # Embedded Atom Method
            'timestamp': self.timestamp,
            'units': {
                'time': 'ps',
                'distance': 'Angstrom',
                'stress': 'MPa',
                'energy': 'eV',
                'temperature': 'K'
            },
            'simulations': []
        }
        
        for i in range(n_simulations):
            temperature = np.random.uniform(*self.temperature_range)
            applied_stress = np.random.uniform(50, 500)  # MPa
            misorientation = np.random.uniform(5, 60)  # degrees
            
            # Simulation time and data points
            total_time = 1000  # ps
            n_steps = 1000
            time_points = np.linspace(0, total_time, n_steps)
            
            # Generate sliding displacement profile
            # Creep-like behavior: initial elastic + steady-state sliding
            elastic_strain = applied_stress / 80000  # E ~ 80 GPa
            creep_rate = applied_stress**3 * np.exp(-250000/(8.314*temperature)) * 1e-10
            
            displacement = []
            velocity = []
            shear_stress = []
            potential_energy = []
            
            for t in time_points:
                # Displacement with some noise
                d = elastic_strain * 10 + creep_rate * t + np.random.normal(0, 0.01)
                displacement.append(float(d))
                
                # Instantaneous velocity
                v = creep_rate + np.random.normal(0, creep_rate * 0.1)
                velocity.append(float(v))
                
                # Stress response
                s = applied_stress + np.random.normal(0, 5)
                shear_stress.append(float(s))
                
                # Potential energy fluctuations
                pe = -4.45 * 1000 + temperature * 0.001 + np.random.normal(0, 0.5)
                potential_energy.append(float(pe))
            
            # Calculate sliding resistance metrics
            avg_sliding_rate = float(np.mean(velocity[len(velocity)//2:]))
            activation_volume = float(np.random.uniform(10, 50))  # b^3
            
            sim = {
                'simulation_id': f'gb_slide_{i:04d}',
                'temperature': float(temperature),
                'applied_stress': float(applied_stress),
                'grain_boundary': {
                    'type': np.random.choice(['tilt', 'twist', 'mixed']),
                    'misorientation_angle': float(misorientation),
                    'gb_plane': [int(x) for x in np.random.randint(1, 5, 3)],
                    'area': float(np.random.uniform(8000, 10000))  # Angstrom^2
                },
                'system_size': {
                    'atoms': int(np.random.randint(50000, 100000)),
                    'box_dimensions': self.box_size
                },
                'trajectory_data': {
                    'time': list(time_points),
                    'displacement': displacement,
                    'velocity': velocity,
                    'shear_stress': shear_stress,
                    'potential_energy': potential_energy
                },
                'analysis': {
                    'average_sliding_rate': avg_sliding_rate,
                    'sliding_resistance': float(applied_stress / avg_sliding_rate) if avg_sliding_rate > 0 else float('inf'),
                    'activation_volume': activation_volume,
                    'activation_energy': float(np.random.uniform(1.5, 3.0)),  # eV
                    'viscosity': float(applied_stress / (avg_sliding_rate * 1e-10)) if avg_sliding_rate > 0 else float('inf')
                },
                'computational_details': {
                    'timestep': self.timestep,
                    'total_time': total_time,
                    'n_steps': n_steps,
                    'thermostat': 'Nose-Hoover',
                    'barostat': 'NPT'
                }
            }
            data['simulations'].append(sim)
        
        return data
    
    def generate_dislocation_mobility(self, n_simulations: int = 25) -> Dict:
        """Generate dislocation mobility simulation data"""
        data = {
            'simulation_type': 'dislocation_mobility',
            'method': 'MD-EAM',
            'timestamp': self.timestamp,
            'units': {
                'time': 'ps',
                'distance': 'Angstrom',
                'velocity': 'Angstrom/ps',
                'stress': 'MPa',
                'energy': 'eV'
            },
            'simulations': []
        }
        
        dislocation_types = ['edge', 'screw', 'mixed']
        
        for i in range(n_simulations):
            temperature = np.random.uniform(*self.temperature_range)
            disl_type = np.random.choice(dislocation_types)
            applied_stress = np.random.uniform(100, 1000)  # MPa
            
            # Simulation parameters
            total_time = 500  # ps
            n_steps = 500
            time_points = np.linspace(0, total_time, n_steps)
            
            # Dislocation velocity model (thermally activated)
            if disl_type == 'edge':
                v0 = 1000  # Angstrom/ps
                activation_energy = 1.2  # eV
            elif disl_type == 'screw':
                v0 = 500
                activation_energy = 1.5
            else:  # mixed
                v0 = 750
                activation_energy = 1.35
            
            # Generate dislocation position and velocity
            position = []
            velocity = []
            line_tension = []
            interaction_energy = []
            
            current_pos = 0
            for j, t in enumerate(time_points):
                # Velocity with thermal activation and stress dependence
                v_thermal = v0 * np.exp(-activation_energy * 11600 / temperature)
                v_stress = v_thermal * (applied_stress / 100) ** 2
                v_instant = v_stress * (1 + np.random.normal(0, 0.2))
                
                velocity.append(float(v_instant))
                
                # Update position
                if j > 0:
                    current_pos += v_instant * (time_points[j] - time_points[j-1])
                position.append(float(current_pos))
                
                # Line tension fluctuations
                lt = 0.5 * 80e9 * (2.86e-10)**2 * (1 + np.random.normal(0, 0.05))  # Gb^2
                line_tension.append(float(lt * 1.6e-19 / (1e-10)**2))  # Convert to eV/Angstrom
                
                # Interaction energy with obstacles
                ie = -0.1 * np.sin(2 * np.pi * current_pos / 50) + np.random.normal(0, 0.02)
                interaction_energy.append(float(ie))
            
            # Calculate mobility metrics
            avg_velocity = float(np.mean(velocity[len(velocity)//4:]))
            mobility = float(avg_velocity / applied_stress) if applied_stress > 0 else 0
            
            # Identify pinning events (velocity drops)
            velocity_array = np.array(velocity)
            pinning_events = np.where(np.diff(velocity_array) < -0.5 * np.std(velocity_array))[0]
            
            sim = {
                'simulation_id': f'disl_mob_{i:04d}',
                'temperature': float(temperature),
                'applied_stress': float(applied_stress),
                'dislocation': {
                    'type': disl_type,
                    'burgers_vector': [2.86/2, 2.86/2, 0] if disl_type != 'screw' else [2.86, 0, 0],
                    'line_direction': [1, 1, 0] if disl_type == 'edge' else [0, 0, 1],
                    'slip_plane': [1, 1, 1],
                    'character_angle': float(0 if disl_type == 'edge' else 90 if disl_type == 'screw' else 45)
                },
                'system_size': {
                    'atoms': int(np.random.randint(100000, 200000)),
                    'box_dimensions': [200, 200, 50]  # Larger box for dislocation motion
                },
                'trajectory_data': {
                    'time': list(time_points),
                    'position': position,
                    'velocity': velocity,
                    'line_tension': line_tension,
                    'interaction_energy': interaction_energy
                },
                'analysis': {
                    'average_velocity': avg_velocity,
                    'mobility': mobility,
                    'drag_coefficient': float(1/mobility) if mobility > 0 else float('inf'),
                    'activation_energy': float(activation_energy),
                    'peierls_stress': float(np.random.uniform(50, 200)),
                    'n_pinning_events': int(len(pinning_events)),
                    'average_jump_distance': float(np.random.uniform(5, 20))  # Angstroms
                },
                'interactions': {
                    'with_point_defects': bool(np.random.choice([True, False])),
                    'with_precipitates': bool(np.random.choice([True, False])),
                    'with_other_dislocations': bool(np.random.choice([True, False]))
                },
                'computational_details': {
                    'timestep': self.timestep,
                    'total_time': total_time,
                    'n_steps': n_steps,
                    'potential': 'EAM-Mishin',
                    'boundary_conditions': 'periodic'
                }
            }
            data['simulations'].append(sim)
        
        return data
    
    def generate_cross_slip_events(self, n_simulations: int = 15) -> Dict:
        """Generate cross-slip event simulation data"""
        data = {
            'simulation_type': 'cross_slip_events',
            'method': 'MD-EAM',
            'timestamp': self.timestamp,
            'units': {
                'time': 'ps',
                'distance': 'Angstrom',
                'energy': 'eV',
                'stress': 'MPa'
            },
            'simulations': []
        }
        
        for i in range(n_simulations):
            temperature = np.random.uniform(*self.temperature_range)
            stress = np.random.uniform(200, 800)
            
            # Cross-slip activation energy
            E_cross_slip = np.random.uniform(0.8, 1.5)  # eV
            
            # Generate cross-slip event data
            event_time = np.random.uniform(50, 200)  # ps
            
            sim = {
                'simulation_id': f'cross_slip_{i:04d}',
                'temperature': float(temperature),
                'applied_stress': float(stress),
                'initial_plane': [1, 1, 1],
                'cross_slip_plane': [1, -1, 1],
                'screw_dislocation_length': float(np.random.uniform(50, 150)),  # Angstroms
                'event_details': {
                    'nucleation_time': float(event_time),
                    'completion_time': float(event_time + np.random.uniform(5, 20)),
                    'activation_energy': float(E_cross_slip),
                    'critical_stress': float(np.random.uniform(300, 600)),
                    'constriction_width': float(np.random.uniform(5, 15))  # Angstroms
                },
                'energy_barrier': {
                    'forward_barrier': float(E_cross_slip),
                    'reverse_barrier': float(E_cross_slip * np.random.uniform(0.8, 1.2)),
                    'saddle_point_configuration': 'Fleischer-type'
                },
                'success_rate': float(np.exp(-E_cross_slip * 11600 / temperature)),
                'computational_details': {
                    'n_atoms': int(np.random.randint(50000, 100000)),
                    'simulation_time': 250,  # ps
                    'method': 'adaptive-timestep MD'
                }
            }
            data['simulations'].append(sim)
        
        return data
    
    def generate_dislocation_interactions(self, n_simulations: int = 20) -> Dict:
        """Generate dislocation-dislocation interaction data"""
        data = {
            'simulation_type': 'dislocation_interactions',
            'method': 'MD-EAM',
            'timestamp': self.timestamp,
            'units': {
                'distance': 'Angstrom',
                'force': 'eV/Angstrom',
                'energy': 'eV',
                'stress': 'MPa'
            },
            'simulations': []
        }
        
        interaction_types = ['parallel', 'perpendicular', 'junction', 'annihilation']
        
        for i in range(n_simulations):
            interaction = np.random.choice(interaction_types)
            separation = np.random.uniform(10, 100)  # Angstroms
            
            # Calculate interaction force and energy
            if interaction == 'parallel':
                force_magnitude = 1.0 / separation  # Simplified 1/r dependence
                energy = -np.log(separation / 5)
            elif interaction == 'perpendicular':
                force_magnitude = 0.5 / separation
                energy = -0.5 * np.log(separation / 5)
            elif interaction == 'junction':
                force_magnitude = 2.0 / separation
                energy = -2 * np.log(separation / 5)
            else:  # annihilation
                force_magnitude = 3.0 / separation if separation < 20 else 0
                energy = -10 * np.exp(-separation / 10)
            
            sim = {
                'simulation_id': f'disl_interact_{i:04d}',
                'interaction_type': interaction,
                'dislocation_1': {
                    'type': np.random.choice(['edge', 'screw', 'mixed']),
                    'burgers_vector': [2.86/2, 2.86/2, 0],
                    'line_direction': list(np.random.randn(3))
                },
                'dislocation_2': {
                    'type': np.random.choice(['edge', 'screw', 'mixed']),
                    'burgers_vector': [2.86/2, -2.86/2, 0],
                    'line_direction': list(np.random.randn(3))
                },
                'separation_distance': float(separation),
                'interaction_force': {
                    'magnitude': float(force_magnitude),
                    'direction': list(np.random.randn(3))
                },
                'interaction_energy': float(energy),
                'stress_field': {
                    'max_stress': float(np.random.uniform(500, 2000)),
                    'stress_range': float(np.random.uniform(50, 200))  # Angstroms
                },
                'reaction_products': {
                    'forms_junction': interaction == 'junction',
                    'junction_strength': float(np.random.uniform(0.5, 2.0)) if interaction == 'junction' else 0,
                    'annihilates': interaction == 'annihilation' and separation < 20
                },
                'computational_details': {
                    'n_atoms': int(np.random.randint(100000, 300000)),
                    'box_size': [300, 300, 100],
                    'relaxation_steps': int(np.random.randint(10000, 50000))
                }
            }
            data['simulations'].append(sim)
        
        return data
    
    def generate_thermal_activation_data(self, n_temps: int = 10) -> Dict:
        """Generate temperature-dependent activation data"""
        data = {
            'analysis_type': 'thermal_activation',
            'timestamp': self.timestamp,
            'temperature_range': self.temperature_range,
            'measurements': []
        }
        
        temperatures = np.linspace(*self.temperature_range, n_temps)
        
        for T in temperatures:
            # Arrhenius behavior for various processes
            vacancy_diffusion = 1e-4 * np.exp(-1.39 * 11600 / T)
            dislocation_velocity = 1000 * np.exp(-1.2 * 11600 / T)
            gb_sliding_rate = 1e-6 * np.exp(-2.5 * 11600 / T)
            
            measurement = {
                'temperature': float(T),
                'vacancy_diffusion_coefficient': float(vacancy_diffusion),
                'dislocation_velocity': float(dislocation_velocity),
                'gb_sliding_rate': float(gb_sliding_rate),
                'creep_rate': float(gb_sliding_rate * 1e3),
                'phonon_drag_coefficient': float(1e-4 * T),
                'stacking_fault_width': float(5 + 0.01 * T)
            }
            data['measurements'].append(measurement)
        
        return data
    
    def save_data(self, data: Dict, filename: str, directory: str):
        """Save data to JSON file"""
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        n_entries = len(data.get('simulations', data.get('measurements', [])))
        print(f"Saved {n_entries} entries to {filepath}")
    
    def save_trajectory_lammps_format(self, sim_data: Dict, output_dir: str):
        """Save trajectory data in LAMMPS dump format"""
        sim_id = sim_data['simulation_id']
        filepath = os.path.join(output_dir, f"{sim_id}_trajectory.lammpstrj")
        
        with open(filepath, 'w') as f:
            # Simplified LAMMPS trajectory header
            n_atoms = sim_data['system_size']['atoms']
            box = sim_data['system_size']['box_dimensions']
            
            for i, t in enumerate(sim_data['trajectory_data']['time']):
                f.write(f"ITEM: TIMESTEP\n{int(t*1000)}\n")  # Convert ps to fs
                f.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
                f.write(f"ITEM: BOX BOUNDS pp pp pp\n")
                f.write(f"0.0 {box[0]}\n0.0 {box[1]}\n0.0 {box[2]}\n")
                f.write("ITEM: ATOMS id type x y z\n")
                
                # Generate some dummy atom positions for demonstration
                # In real simulation, these would be actual atom coordinates
                for atom_id in range(min(100, n_atoms)):  # Only write first 100 for demo
                    atom_type = 1
                    x = np.random.uniform(0, box[0])
                    y = np.random.uniform(0, box[1])
                    z = np.random.uniform(0, box[2])
                    f.write(f"{atom_id+1} {atom_type} {x:.4f} {y:.4f} {z:.4f}\n")
    
    def generate_all_md_data(self, output_dir: str = 'md_simulations'):
        """Generate all MD simulation datasets"""
        print("Generating MD simulation data...")
        
        # Create subdirectories
        os.makedirs(f"{output_dir}/grain_boundary", exist_ok=True)
        os.makedirs(f"{output_dir}/dislocation", exist_ok=True)
        os.makedirs(f"{output_dir}/trajectories", exist_ok=True)
        
        # Generate datasets
        print("\n1. Generating grain boundary sliding simulations...")
        gb_data = self.generate_grain_boundary_sliding(20)
        self.save_data(gb_data, "gb_sliding.json", f"{output_dir}/grain_boundary")
        
        # Save sample trajectories
        for sim in gb_data['simulations'][:3]:  # Save first 3 trajectories
            self.save_trajectory_lammps_format(sim, f"{output_dir}/trajectories")
        
        print("\n2. Generating dislocation mobility simulations...")
        disl_data = self.generate_dislocation_mobility(25)
        self.save_data(disl_data, "dislocation_mobility.json", f"{output_dir}/dislocation")
        
        print("\n3. Generating cross-slip events...")
        cross_slip_data = self.generate_cross_slip_events(15)
        self.save_data(cross_slip_data, "cross_slip_events.json", f"{output_dir}/dislocation")
        
        print("\n4. Generating dislocation interaction data...")
        interaction_data = self.generate_dislocation_interactions(20)
        self.save_data(interaction_data, "dislocation_interactions.json", f"{output_dir}/dislocation")
        
        print("\n5. Generating thermal activation analysis...")
        thermal_data = self.generate_thermal_activation_data(10)
        self.save_data(thermal_data, "thermal_activation.json", output_dir)
        
        # Generate summary
        self.generate_summary_statistics(output_dir)
    
    def generate_summary_statistics(self, output_dir: str):
        """Generate summary statistics for MD simulations"""
        summary = {
            'generated_at': self.timestamp,
            'simulation_counts': {
                'grain_boundary_sliding': 20,
                'dislocation_mobility': 25,
                'cross_slip_events': 15,
                'dislocation_interactions': 20,
                'thermal_activation_points': 10
            },
            'total_simulations': 80,
            'temperature_range': self.temperature_range,
            'computational_cost_estimate': {
                'gpu_hours': 2000,
                'cpu_hours': 10000,
                'storage_gb': 50
            }
        }
        
        with open(f"{output_dir}/summary_statistics.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSummary: Generated {summary['total_simulations']} MD simulations")
        print(f"Estimated computational cost: {summary['computational_cost_estimate']['gpu_hours']:,} GPU-hours")


if __name__ == "__main__":
    generator = MDDataGenerator(seed=42)
    generator.generate_all_md_data()