"""
Visualization and Analysis Script for Atomic-Scale Simulation Data
Creates plots and statistical summaries of the generated data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_dft_data(data_dir='atomic_simulation_data'):
    """Analyze and visualize DFT calculation results."""
    
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("DFT DATA ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data
    defect_energies = pd.read_csv(data_path / 'dft_defect_formation_energies.csv')
    gb_energies = pd.read_csv(data_path / 'dft_grain_boundary_energies.csv')
    activation_barriers = pd.read_csv(data_path / 'dft_activation_barriers.csv')
    surface_energies = pd.read_csv(data_path / 'dft_surface_energies.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DFT Calculation Results', fontsize=16, fontweight='bold')
    
    # 1. Defect Formation Energies by Type
    ax1 = axes[0, 0]
    defect_energies.boxplot(column='formation_energy_eV', by='defect_type', ax=ax1)
    ax1.set_title('Defect Formation Energies by Type')
    ax1.set_xlabel('Defect Type')
    ax1.set_ylabel('Formation Energy (eV)')
    plt.sca(ax1)
    plt.xticks(rotation=45)
    
    # 2. Grain Boundary Energy vs Misorientation
    ax2 = axes[0, 1]
    for gb_type in gb_energies['gb_type'].unique():
        data = gb_energies[gb_energies['gb_type'] == gb_type]
        ax2.scatter(data['misorientation_deg'], data['gb_energy_J_m2'], 
                   label=gb_type, alpha=0.6, s=50)
    ax2.set_title('Grain Boundary Energy vs Misorientation Angle')
    ax2.set_xlabel('Misorientation Angle (degrees)')
    ax2.set_ylabel('GB Energy (J/m²)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Activation Barriers by Mechanism
    ax3 = axes[1, 0]
    mechanism_data = activation_barriers.groupby('mechanism')['activation_energy_eV'].agg(['mean', 'std'])
    mechanism_data.plot(kind='bar', y='mean', yerr='std', ax=ax3, legend=False)
    ax3.set_title('Activation Energy Barriers by Diffusion Mechanism')
    ax3.set_xlabel('Mechanism')
    ax3.set_ylabel('Activation Energy (eV)')
    plt.sca(ax3)
    plt.xticks(rotation=45, ha='right')
    
    # 4. Surface Energy by Orientation
    ax4 = axes[1, 1]
    surface_energies.boxplot(column='surface_energy_J_m2', by='surface_orientation', ax=ax4)
    ax4.set_title('Surface Energies by Crystallographic Orientation')
    ax4.set_xlabel('Surface Orientation')
    ax4.set_ylabel('Surface Energy (J/m²)')
    
    plt.tight_layout()
    plt.savefig(data_path / 'dft_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved DFT analysis plot: dft_analysis.png")
    
    # Print statistics
    print()
    print("DEFECT FORMATION ENERGIES:")
    print(defect_energies.groupby('defect_type')['formation_energy_eV'].describe())
    print()
    print("ACTIVATION BARRIERS:")
    print(activation_barriers.groupby('mechanism')['activation_energy_eV'].describe())
    print()
    
    plt.close()


def analyze_md_data(data_dir='atomic_simulation_data'):
    """Analyze and visualize MD simulation results."""
    
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("MD SIMULATION ANALYSIS")
    print("=" * 70)
    print()
    
    # Load data
    gb_sliding = pd.read_csv(data_path / 'md_grain_boundary_sliding.csv')
    disl_mobility = pd.read_csv(data_path / 'md_dislocation_mobility.csv')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Molecular Dynamics Simulation Results', fontsize=16, fontweight='bold')
    
    # 1. GB Sliding: Stress vs Displacement
    ax1 = axes[0, 0]
    scatter = ax1.scatter(gb_sliding['applied_shear_stress_MPa'], 
                         gb_sliding['sliding_displacement_nm'],
                         c=gb_sliding['temperature_K'], 
                         cmap='coolwarm', alpha=0.6, s=50)
    ax1.set_title('Grain Boundary Sliding: Stress vs Displacement')
    ax1.set_xlabel('Applied Shear Stress (MPa)')
    ax1.set_ylabel('Sliding Displacement (nm)')
    cbar1 = plt.colorbar(scatter, ax=ax1)
    cbar1.set_label('Temperature (K)')
    ax1.grid(True, alpha=0.3)
    
    # 2. GB Sliding Velocity Distribution
    ax2 = axes[0, 1]
    gb_sliding['sliding_velocity_nm_ps'].hist(bins=30, ax=ax2, edgecolor='black')
    ax2.set_title('Distribution of Grain Boundary Sliding Velocities')
    ax2.set_xlabel('Sliding Velocity (nm/ps)')
    ax2.set_ylabel('Frequency')
    ax2.set_yscale('log')
    
    # 3. Dislocation Velocity vs Stress
    ax3 = axes[1, 0]
    for disl_type in disl_mobility['dislocation_type'].unique()[:3]:  # Plot first 3 types
        data = disl_mobility[disl_mobility['dislocation_type'] == disl_type]
        ax3.scatter(data['applied_stress_MPa'], 
                   data['dislocation_velocity_m_s'],
                   label=disl_type, alpha=0.6, s=50)
    ax3.set_title('Dislocation Velocity vs Applied Stress')
    ax3.set_xlabel('Applied Stress (MPa)')
    ax3.set_ylabel('Dislocation Velocity (m/s)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Peierls Stress Distribution
    ax4 = axes[1, 1]
    disl_mobility.boxplot(column='peierls_stress_MPa', by='dislocation_type', ax=ax4)
    ax4.set_title('Peierls Stress by Dislocation Type')
    ax4.set_xlabel('Dislocation Type')
    ax4.set_ylabel('Peierls Stress (MPa)')
    plt.sca(ax4)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(data_path / 'md_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved MD analysis plot: md_analysis.png")
    
    # Print statistics
    print()
    print("GRAIN BOUNDARY SLIDING:")
    print(gb_sliding[['critical_shear_stress_MPa', 'sliding_displacement_nm', 
                      'sliding_velocity_nm_ps']].describe())
    print()
    print("DISLOCATION MOBILITY:")
    print(disl_mobility[['peierls_stress_MPa', 'dislocation_velocity_m_s']].describe())
    print()
    
    plt.close()


def analyze_trajectory(data_dir='atomic_simulation_data'):
    """Analyze MD trajectory data."""
    
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("MD TRAJECTORY ANALYSIS")
    print("=" * 70)
    print()
    
    # Load trajectory
    with open(data_path / 'md_sample_trajectory.json', 'r') as f:
        trajectory = json.load(f)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('MD Trajectory Analysis', fontsize=16, fontweight='bold')
    
    timesteps = np.array(trajectory['timesteps_fs'])
    
    # 1. Energy vs Time
    ax1 = axes[0, 0]
    ax1.plot(timesteps, trajectory['total_energy_eV'], label='Total', linewidth=2)
    ax1.plot(timesteps, trajectory['potential_energy_eV'], label='Potential', alpha=0.7)
    ax1.plot(timesteps, trajectory['kinetic_energy_eV'], label='Kinetic', alpha=0.7)
    ax1.set_title('Energy Evolution')
    ax1.set_xlabel('Time (fs)')
    ax1.set_ylabel('Energy (eV)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Temperature vs Time
    ax2 = axes[0, 1]
    ax2.plot(timesteps, trajectory['temperature_K'], color='red', linewidth=1.5)
    ax2.set_title('Temperature Evolution')
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Temperature (K)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Pressure vs Time
    ax3 = axes[1, 0]
    ax3.plot(timesteps, trajectory['pressure_GPa'], color='green', linewidth=1.5)
    ax3.set_title('Pressure Evolution')
    ax3.set_xlabel('Time (fs)')
    ax3.set_ylabel('Pressure (GPa)')
    ax3.grid(True, alpha=0.3)
    
    # 4. Energy Conservation Check
    ax4 = axes[1, 1]
    total_energy = np.array(trajectory['total_energy_eV'])
    energy_drift = total_energy - total_energy[0]
    ax4.plot(timesteps, energy_drift, color='purple', linewidth=2)
    ax4.set_title('Energy Conservation (Drift from Initial)')
    ax4.set_xlabel('Time (fs)')
    ax4.set_ylabel('Energy Drift (eV)')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(data_path / 'trajectory_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved trajectory analysis plot: trajectory_analysis.png")
    
    # Print statistics
    print()
    print(f"Simulation details:")
    print(f"  Number of atoms: {trajectory['num_atoms']}")
    print(f"  Number of steps: {trajectory['num_steps']}")
    print(f"  Total simulation time: {timesteps[-1]} fs")
    print()
    print(f"Temperature statistics:")
    print(f"  Mean: {np.mean(trajectory['temperature_K']):.2f} K")
    print(f"  Std: {np.std(trajectory['temperature_K']):.2f} K")
    print()
    print(f"Energy conservation:")
    print(f"  Total energy drift: {energy_drift[-1]:.4f} eV")
    print(f"  Relative drift: {100 * energy_drift[-1] / total_energy[0]:.4f}%")
    print()
    
    plt.close()


def generate_data_summary(data_dir='atomic_simulation_data'):
    """Generate comprehensive data summary report."""
    
    data_path = Path(data_dir)
    
    print("=" * 70)
    print("COMPREHENSIVE DATA SUMMARY")
    print("=" * 70)
    print()
    
    # Load summary
    with open(data_path / 'dataset_summary.json', 'r') as f:
        summary = json.load(f)
    
    print(f"Generation Date: {summary['generation_date']}")
    print(f"Total Samples: {summary['total_samples']}")
    print(f"Materials Covered: {', '.join(summary['materials_covered'])}")
    print()
    
    print("DATASETS:")
    print("-" * 70)
    for dataset_name, info in summary['datasets'].items():
        print(f"\n{dataset_name}:")
        print(f"  File: {info['file']}")
        print(f"  Description: {info['description']}")
        if 'samples' in info:
            print(f"  Samples: {info['samples']}")
        if 'statistics' in info:
            print(f"  Key Statistics:")
            for stat_name, stat_value in info['statistics'].items():
                print(f"    - {stat_name}: {stat_value}")
    
    print()
    print("=" * 70)
    print("USAGE NOTES:")
    print("=" * 70)
    for note_type, note in summary['usage_notes'].items():
        if isinstance(note, dict):
            print(f"\n{note_type}:")
            for k, v in note.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n{note_type}: {note}")
    print()


if __name__ == '__main__':
    data_dir = 'atomic_simulation_data'
    
    try:
        # Analyze DFT data
        analyze_dft_data(data_dir)
        
        # Analyze MD data
        analyze_md_data(data_dir)
        
        # Analyze trajectory
        analyze_trajectory(data_dir)
        
        # Generate summary
        generate_data_summary(data_dir)
        
        print()
        print("=" * 70)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 70)
        print()
        print("Generated plots:")
        print("  • dft_analysis.png")
        print("  • md_analysis.png")
        print("  • trajectory_analysis.png")
        print()
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
