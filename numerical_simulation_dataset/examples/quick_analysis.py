#!/usr/bin/env python3
"""
Quick analysis script to demonstrate loading and analyzing the numerical simulation dataset
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.visualization import SimulationVisualizer

def main():
    """Main analysis function"""
    
    # Set data path
    data_path = Path(__file__).parent.parent
    
    # 1. Load summary statistics
    print("=" * 50)
    print("📊 DATASET SUMMARY")
    print("=" * 50)
    
    with open(data_path / 'summary_statistics/dataset_stats.json', 'r') as f:
        stats = json.load(f)
    
    print(f"Total Simulations: {stats['total_simulations']}")
    print(f"Successful Simulations: {stats['successful_simulations']}")
    print(f"Average Max Stress: {stats['average_max_stress']:.2f} MPa")
    print(f"Average Max Damage: {stats['average_max_damage']:.3f}")
    print(f"Failure Rate: {stats['failure_rate']*100:.1f}%")
    print(f"Grid Size: {stats['grid_dimensions']}")
    print(f"Time Steps: {stats['time_steps']}")
    
    # 2. Load simulation summary
    print("\n" + "=" * 50)
    print("📈 SIMULATION RESULTS")
    print("=" * 50)
    
    df_summary = pd.read_csv(data_path / 'summary_statistics/simulation_summary.csv')
    print("\nSimulation Summary:")
    print(df_summary.to_string())
    
    # 3. Load and analyze a specific simulation
    sim_id = 'sim_0000'
    print(f"\n" + "=" * 50)
    print(f"🔍 DETAILED ANALYSIS: {sim_id}")
    print("=" * 50)
    
    # Load mesh data
    with open(data_path / f'input_parameters/mesh_data/{sim_id}_mesh.json', 'r') as f:
        mesh_data = json.load(f)
    
    print("\nMesh Configuration:")
    for key, value in mesh_data.items():
        print(f"  {key}: {value}")
    
    # Load boundary conditions
    with open(data_path / f'input_parameters/boundary_conditions/{sim_id}_bc.json', 'r') as f:
        bc_data = json.load(f)
    
    print("\nBoundary Conditions:")
    print(f"  Mechanical Type: {bc_data['mechanical']['type']}")
    print(f"  Applied Voltage: {bc_data['electrical']['voltage_applied']:.2f} V")
    print(f"  Charge Rate: {bc_data['electrical']['charge_rate']:.2f} C")
    
    # Load stress data
    stress_file = data_path / f'output_data/stress_fields/{sim_id}_stress.h5'
    with h5py.File(stress_file, 'r') as f:
        print("\nStress Data Structure:")
        print(f"  Time steps available: {len([k for k in f.keys() if k.startswith('t_')])}")
        
        # Analyze final time step
        final_time = 't_19'
        if final_time in f:
            von_mises = f[final_time]['von_mises'][:]
            print(f"\nVon Mises Stress at {final_time}:")
            print(f"    Shape: {von_mises.shape}")
            print(f"    Min: {np.min(von_mises):.2f} MPa")
            print(f"    Max: {np.max(von_mises):.2f} MPa")
            print(f"    Mean: {np.mean(von_mises):.2f} MPa")
            print(f"    Std: {np.std(von_mises):.2f} MPa")
    
    # Load damage data
    damage_file = data_path / f'output_data/damage_evolution/{sim_id}_damage.h5'
    with h5py.File(damage_file, 'r') as f:
        # Track damage evolution
        damage_evolution = []
        for t in range(20):
            time_key = f't_{t}'
            if time_key in f:
                max_damage = f[time_key].attrs.get('max_damage', 0)
                damage_evolution.append(max_damage)
        
        print(f"\nDamage Evolution:")
        print(f"  Initial Damage: {damage_evolution[0]:.4f}")
        print(f"  Final Damage: {damage_evolution[-1]:.4f}")
        print(f"  Max Rate of Change: {np.max(np.diff(damage_evolution)):.4f}")
    
    # 4. Create a simple visualization
    print("\n" + "=" * 50)
    print("📊 CREATING VISUALIZATION")
    print("=" * 50)
    
    visualizer = SimulationVisualizer(data_path)
    
    # Create stress contour plot
    try:
        fig = visualizer.plot_stress_contour(
            sim_id, 
            time_step=10,
            stress_type='von_mises',
            slice_axis='z'
        )
        
        # Save figure
        output_file = data_path / f'examples/{sim_id}_stress_contour.png'
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved stress contour to: {output_file}")
        plt.close(fig)
    except Exception as e:
        print(f"Could not create visualization: {e}")
    
    # 5. Data format examples
    print("\n" + "=" * 50)
    print("💾 DATA ACCESS EXAMPLES")
    print("=" * 50)
    
    print("\nTo load stress data in Python:")
    print("```python")
    print("import h5py")
    print("with h5py.File('output_data/stress_fields/sim_0000_stress.h5', 'r') as f:")
    print("    von_mises = f['t_10']['von_mises'][:]  # Load von Mises stress at t=10")
    print("    coords = f['t_10']['coordinates']")
    print("    x = coords['x'][:]")
    print("    y = coords['y'][:]")
    print("    z = coords['z'][:]")
    print("```")
    
    print("\nTo load in MATLAB:")
    print("```matlab")
    print("data = h5read('output_data/stress_fields/sim_0000_stress.h5', '/t_10/von_mises');")
    print("```")
    
    print("\n" + "=" * 50)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 50)

if __name__ == '__main__':
    main()