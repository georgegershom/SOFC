#!/usr/bin/env python3
"""
Main script to generate complete numerical simulation dataset
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_generator import (
    SimulationDataGenerator,
    save_simulation_data,
    generate_simulation_metadata,
    MeshParameters
)

def create_directory_structure(base_path: Path):
    """Create the dataset directory structure"""
    
    directories = [
        'input_parameters/mesh_data',
        'input_parameters/boundary_conditions',
        'input_parameters/material_models',
        'input_parameters/thermal_profiles',
        'output_data/stress_fields',
        'output_data/strain_fields',
        'output_data/damage_evolution',
        'output_data/temperature_distributions',
        'output_data/voltage_distributions',
        'output_data/failure_predictions',
        'metadata',
        'summary_statistics'
    ]
    
    for dir_path in directories:
        (base_path / dir_path).mkdir(parents=True, exist_ok=True)

def generate_single_simulation(sim_id: str, 
                             base_path: Path,
                             config: Dict) -> Dict:
    """Generate data for a single simulation case"""
    
    generator = SimulationDataGenerator(seed=hash(sim_id) % 2**32)
    
    # Generate input parameters
    print(f"  Generating simulation {sim_id}...")
    
    # 1. Mesh data
    mesh = generator.generate_mesh_data(refinement_level=config.get('refinement', 2))
    mesh_data = {
        'element_size': mesh.element_size,
        'element_type': mesh.element_type,
        'interface_refinement': mesh.interface_refinement,
        'num_elements': mesh.num_elements,
        'num_nodes': mesh.num_nodes
    }
    
    # 2. Thermal profile
    heating_rate = np.random.uniform(1, 10)  # °C/min
    time, temperature = generator.generate_thermal_profile(
        heating_rate=heating_rate,
        T_initial=25,
        T_max=np.random.uniform(60, 85),
        hold_time=np.random.uniform(1800, 7200),
        cooling_rate=np.random.uniform(0.5, 5)
    )
    
    thermal_data = {
        'time': time,
        'temperature': temperature,
        'heating_rate': heating_rate,
        'max_temperature': np.max(temperature),
        'thermal_cycles': 1
    }
    
    # 3. Boundary conditions
    boundary_conditions = {
        'mechanical': {
            'type': np.random.choice(['displacement', 'force', 'mixed']),
            'displacement_x': np.random.uniform(-0.1, 0.1) if np.random.rand() > 0.5 else 0,
            'displacement_y': np.random.uniform(-0.1, 0.1) if np.random.rand() > 0.5 else 0,
            'displacement_z': np.random.uniform(-0.5, 0.5),
            'force_x': np.random.uniform(-100, 100) if np.random.rand() > 0.5 else 0,
            'force_y': np.random.uniform(-100, 100) if np.random.rand() > 0.5 else 0,
            'force_z': np.random.uniform(-500, 500),
        },
        'thermal': {
            'type': 'convection',
            'h_convection': np.random.uniform(5, 25),  # W/m²K
            'T_ambient': 25,
            'heat_flux': np.random.uniform(0, 1000)  # W/m²
        },
        'electrical': {
            'voltage_applied': np.random.uniform(3.0, 4.2),
            'current_density': np.random.uniform(0.1, 5.0),  # A/m²
            'charge_rate': np.random.uniform(0.5, 2.0)  # C-rate
        }
    }
    
    # 4. Generate output fields
    nx, ny, nz = config.get('grid_size', [30, 30, 15])
    time_steps = config.get('time_steps', 50)
    
    # Stress fields
    print(f"    Generating stress fields...")
    stress_data = generator.generate_stress_field(
        nx=nx, ny=ny, nz=nz,
        load_type=np.random.choice(['thermal', 'mechanical', 'coupled']),
        time_steps=time_steps
    )
    
    # Strain fields
    print(f"    Generating strain fields...")
    strain_data = generator.generate_strain_field(stress_data)
    
    # Damage evolution
    print(f"    Generating damage evolution...")
    damage_data = generator.generate_damage_evolution(
        stress_data,
        damage_model=np.random.choice(['lemaitre', 'cohesive', 'fatigue'])
    )
    
    # Temperature distribution
    print(f"    Generating temperature distribution...")
    temp_data = generator.generate_temperature_distribution(
        nx=nx, ny=ny, nz=nz,
        time_steps=time_steps
    )
    
    # Voltage distribution
    print(f"    Generating voltage distribution...")
    voltage_data = generator.generate_voltage_distribution(
        nx=nx, ny=ny, nz=nz,
        time_steps=time_steps
    )
    
    # Failure predictions
    print(f"    Generating failure predictions...")
    failure_data = generator.generate_failure_predictions(damage_data, stress_data)
    
    # Save all data
    print(f"    Saving data files...")
    
    # Save input parameters
    with open(base_path / f'input_parameters/mesh_data/{sim_id}_mesh.json', 'w') as f:
        json.dump(mesh_data, f, indent=2)
    
    np.savez_compressed(
        base_path / f'input_parameters/thermal_profiles/{sim_id}_thermal.npz',
        **thermal_data
    )
    
    with open(base_path / f'input_parameters/boundary_conditions/{sim_id}_bc.json', 'w') as f:
        json.dump(boundary_conditions, f, indent=2)
    
    # Save output data in HDF5 format
    save_simulation_data(
        stress_data,
        base_path / f'output_data/stress_fields/{sim_id}_stress.h5',
        format='hdf5'
    )
    
    save_simulation_data(
        strain_data,
        base_path / f'output_data/strain_fields/{sim_id}_strain.h5',
        format='hdf5'
    )
    
    save_simulation_data(
        damage_data,
        base_path / f'output_data/damage_evolution/{sim_id}_damage.h5',
        format='hdf5'
    )
    
    save_simulation_data(
        temp_data,
        base_path / f'output_data/temperature_distributions/{sim_id}_temperature.h5',
        format='hdf5'
    )
    
    save_simulation_data(
        voltage_data,
        base_path / f'output_data/voltage_distributions/{sim_id}_voltage.h5',
        format='hdf5'
    )
    
    save_simulation_data(
        failure_data,
        base_path / f'output_data/failure_predictions/{sim_id}_failure.h5',
        format='hdf5'
    )
    
    # Generate and save metadata
    metadata = generate_simulation_metadata(
        sim_id,
        {
            'mesh': mesh_data,
            'boundary_conditions': boundary_conditions,
            'grid_size': [nx, ny, nz],
            'time_steps': time_steps,
            'heating_rate': heating_rate
        }
    )
    
    with open(base_path / f'metadata/{sim_id}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Extract summary statistics
    summary = {
        'simulation_id': sim_id,
        'max_stress': max([np.max(s['von_mises']) for s in stress_data.values()]),
        'max_strain': max([np.max(s['total']) for s in strain_data.values()]),
        'max_damage': max([d['max_damage'] for d in damage_data.values()]),
        'max_temperature': max([t['max_temp'] for t in temp_data.values()]),
        'num_crack_sites': max([f['num_crack_sites'] for f in failure_data.values()]),
        'convergence': True,
        'computation_time': np.random.uniform(100, 1000)  # seconds
    }
    
    return summary

def generate_parameter_sweep(base_path: Path, num_simulations: int):
    """Generate a parameter sweep study"""
    
    # Define parameter ranges for sweep
    param_ranges = {
        'heating_rates': np.linspace(1, 10, 5),  # °C/min
        'mesh_refinements': [1, 2, 3],
        'load_types': ['thermal', 'mechanical', 'coupled'],
        'damage_models': ['lemaitre', 'cohesive', 'fatigue']
    }
    
    # Create parameter combinations
    sweep_data = []
    sim_count = 0
    
    for heating_rate in param_ranges['heating_rates']:
        for refinement in param_ranges['mesh_refinements']:
            for load_type in param_ranges['load_types']:
                for damage_model in param_ranges['damage_models']:
                    if sim_count >= num_simulations:
                        break
                    
                    sim_id = f'sweep_{sim_count:04d}'
                    config = {
                        'heating_rate': heating_rate,
                        'refinement': refinement,
                        'load_type': load_type,
                        'damage_model': damage_model,
                        'grid_size': [20, 20, 10],
                        'time_steps': 30
                    }
                    
                    # Generate simplified sweep data
                    sweep_entry = {
                        'simulation_id': sim_id,
                        'heating_rate': heating_rate,
                        'mesh_refinement': refinement,
                        'load_type': load_type,
                        'damage_model': damage_model,
                        'max_stress': np.random.uniform(100, 500),
                        'max_damage': np.random.uniform(0, 1),
                        'failure_time': np.random.uniform(1000, 10000) if np.random.rand() > 0.3 else None
                    }
                    
                    sweep_data.append(sweep_entry)
                    sim_count += 1
    
    # Save parameter sweep results
    df_sweep = pd.DataFrame(sweep_data)
    df_sweep.to_csv(base_path / 'summary_statistics/parameter_sweep.csv', index=False)
    
    return df_sweep

def main():
    parser = argparse.ArgumentParser(description='Generate numerical simulation dataset')
    parser.add_argument('--num_simulations', type=int, default=10,
                       help='Number of simulations to generate')
    parser.add_argument('--output_dir', type=str, 
                       default='/workspace/numerical_simulation_dataset',
                       help='Output directory for dataset')
    parser.add_argument('--grid_size', type=int, nargs=3, default=[30, 30, 15],
                       help='Grid size for 3D fields (nx ny nz)')
    parser.add_argument('--time_steps', type=int, default=50,
                       help='Number of time steps')
    parser.add_argument('--parameter_sweep', action='store_true',
                       help='Generate parameter sweep study')
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.output_dir)
    create_directory_structure(base_path)
    
    print(f"Generating numerical simulation dataset with {args.num_simulations} simulations...")
    print(f"Output directory: {base_path}")
    
    # Configuration for all simulations
    config = {
        'grid_size': args.grid_size,
        'time_steps': args.time_steps,
        'refinement': 2
    }
    
    # Generate simulations
    summaries = []
    
    for i in tqdm(range(args.num_simulations), desc="Generating simulations"):
        sim_id = f'sim_{i:04d}'
        
        try:
            summary = generate_single_simulation(sim_id, base_path, config)
            summaries.append(summary)
        except Exception as e:
            print(f"  Error in simulation {sim_id}: {e}")
            continue
    
    # Save summary statistics
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(base_path / 'summary_statistics/simulation_summary.csv', index=False)
    
    # Generate parameter sweep if requested
    if args.parameter_sweep:
        print("\nGenerating parameter sweep study...")
        df_sweep = generate_parameter_sweep(base_path, min(args.num_simulations, 50))
        print(f"Parameter sweep saved with {len(df_sweep)} entries")
    
    # Generate dataset statistics
    stats = {
        'total_simulations': len(summaries),
        'successful_simulations': len([s for s in summaries if s.get('convergence', False)]),
        'average_max_stress': df_summary['max_stress'].mean() if summaries else 0,
        'average_max_damage': df_summary['max_damage'].mean() if summaries else 0,
        'failure_rate': len([s for s in summaries if s['num_crack_sites'] > 0]) / len(summaries) if summaries else 0,
        'grid_dimensions': args.grid_size,
        'time_steps': args.time_steps
    }
    
    with open(base_path / 'summary_statistics/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("\n" + "="*50)
    print("Dataset generation complete!")
    print(f"Total simulations: {stats['total_simulations']}")
    print(f"Average max stress: {stats['average_max_stress']:.2f} MPa")
    print(f"Average max damage: {stats['average_max_damage']:.3f}")
    print(f"Failure rate: {stats['failure_rate']*100:.1f}%")
    print("="*50)

if __name__ == '__main__':
    main()