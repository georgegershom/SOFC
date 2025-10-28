#!/usr/bin/env python
"""
SOFC Results Visualization Script
==================================
Generate publication-quality plots from simulation results.

Usage:
  python visualize_results.py Job_SOFC_HR1_results.npz
  python visualize_results.py --all  # Process all NPZ files in directory
"""

import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# Plotting style
plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# ============================================================================
# CONFIGURATION
# ============================================================================

# Interface y-coordinates (meters)
Y_ANODE_TOP = 0.4e-3
Y_ELYTE_TOP = 0.5e-3
Y_CATH_TOP = 0.9e-3

# Delamination thresholds
TAU_CRIT = {'AE': 25.0, 'EC': 20.0, 'CI': 30.0}  # MPa

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_thermal_history(data, output_prefix):
    """Plot temperature evolution."""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    time_min = data['time'] / 60.0
    temp_elyte = data['temperature'][:, :].mean(axis=1) - 273.15  # Average temp, convert to C
    
    ax.plot(time_min, temp_elyte, 'r-', linewidth=2, label='Average Temperature')
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Temperature [°C]')
    ax.set_title('Thermal Cycle Profile')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_thermal_history.png')
    print(f"  Saved: {output_prefix}_thermal_history.png")
    plt.close()


def plot_stress_evolution(data, output_prefix):
    """Plot stress evolution."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time_min = data['time'] / 60.0
    max_vm = np.max(data['von_mises'], axis=1) / 1e6  # Convert to MPa
    
    # Von Mises stress
    ax1.plot(time_min, max_vm, 'b-', linewidth=2)
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Max von Mises Stress [MPa]')
    ax1.set_title('Maximum Stress Evolution')
    ax1.grid(True, alpha=0.3)
    
    # Shear stresses at interfaces
    shear_ae = np.array(data['interface_shear_AE']) / 1e6
    shear_ec = np.array(data['interface_shear_EC']) / 1e6
    shear_ci = np.array(data['interface_shear_CI']) / 1e6
    
    ax2.plot(time_min, shear_ae, 'r-', linewidth=2, label='Anode-Electrolyte')
    ax2.plot(time_min, shear_ec, 'g-', linewidth=2, label='Electrolyte-Cathode')
    ax2.plot(time_min, shear_ci, 'b-', linewidth=2, label='Cathode-Interconnect')
    
    ax2.axhline(TAU_CRIT['AE'], color='r', linestyle='--', alpha=0.5)
    ax2.axhline(TAU_CRIT['EC'], color='g', linestyle='--', alpha=0.5)
    ax2.axhline(TAU_CRIT['CI'], color='b', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Interface Shear Stress [MPa]')
    ax2.set_title('Interface Shear Stresses')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_stress_evolution.png')
    print(f"  Saved: {output_prefix}_stress_evolution.png")
    plt.close()


def plot_delamination_risk(data, output_prefix):
    """Plot delamination risk metrics."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    time_min = data['time'] / 60.0
    
    ax.plot(time_min, data['delamination_risk_AE'], 'r-', linewidth=2, 
            label='Anode-Electrolyte', marker='o', markersize=3, markevery=10)
    ax.plot(time_min, data['delamination_risk_EC'], 'g-', linewidth=2,
            label='Electrolyte-Cathode', marker='s', markersize=3, markevery=10)
    ax.plot(time_min, data['delamination_risk_CI'], 'b-', linewidth=2,
            label='Cathode-Interconnect', marker='^', markersize=3, markevery=10)
    
    ax.axhline(1.0, color='k', linestyle='--', linewidth=1.5, label='Critical Threshold')
    ax.fill_between(time_min, 1.0, ax.get_ylim()[1], alpha=0.1, color='red')
    
    ax.set_xlabel('Time [min]')
    ax.set_ylabel('Delamination Risk')
    ax.set_title('Interface Delamination Risk Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Annotate if any exceeded
    max_risks = {
        'AE': np.max(data['delamination_risk_AE']),
        'EC': np.max(data['delamination_risk_EC']),
        'CI': np.max(data['delamination_risk_CI'])
    }
    
    exceeded = [k for k, v in max_risks.items() if v > 1.0]
    if exceeded:
        ax.text(0.02, 0.98, f"⚠ Critical: {', '.join(exceeded)}", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    else:
        ax.text(0.02, 0.98, "✓ All interfaces OK", 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_delamination_risk.png')
    print(f"  Saved: {output_prefix}_delamination_risk.png")
    plt.close()


def plot_damage_evolution(data, output_prefix):
    """Plot damage and crack depth evolution."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    time_min = data['time'] / 60.0
    max_damage = np.max(data['damage_D'], axis=1)
    crack_depth = data['crack_depth_um']
    
    # Damage evolution
    ax1.plot(time_min, max_damage, 'r-', linewidth=2)
    ax1.axhline(0.2, color='orange', linestyle='--', alpha=0.7, label='Damage threshold (0.2)')
    ax1.axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Complete damage (1.0)')
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Maximum Damage D')
    ax1.set_title('Damage Evolution')
    ax1.set_ylim([0, min(1.1, max_damage.max() * 1.2)])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Crack depth
    ax2.plot(time_min, crack_depth, 'b-', linewidth=2)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Crack Depth [μm]')
    ax2.set_title('Electrolyte Crack Depth')
    ax2.grid(True, alpha=0.3)
    
    # Annotate max crack depth
    max_depth_idx = np.argmax(crack_depth)
    max_depth = crack_depth[max_depth_idx]
    max_depth_time = time_min[max_depth_idx]
    
    if max_depth > 0:
        ax2.annotate(f'Max: {max_depth:.2f} μm\n@ {max_depth_time:.1f} min',
                     xy=(max_depth_time, max_depth),
                     xytext=(10, 10), textcoords='offset points',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_damage_evolution.png')
    print(f"  Saved: {output_prefix}_damage_evolution.png")
    plt.close()


def plot_strain_accumulation(data, output_prefix):
    """Plot plastic and creep strain accumulation."""
    
    if 'peeq' not in data or 'ceeq' not in data:
        print("  Skipping strain plot (PEEQ/CEEQ not available)")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    time_min = data['time'] / 60.0
    max_peeq = np.max(data['peeq'], axis=1)
    max_ceeq = np.max(data['ceeq'], axis=1)
    
    # Plastic strain
    ax1.plot(time_min, max_peeq * 100, 'r-', linewidth=2)
    ax1.set_xlabel('Time [min]')
    ax1.set_ylabel('Max Equivalent Plastic Strain [%]')
    ax1.set_title('Plastic Strain Accumulation')
    ax1.grid(True, alpha=0.3)
    
    # Creep strain
    ax2.plot(time_min, max_ceeq * 100, 'b-', linewidth=2)
    ax2.set_xlabel('Time [min]')
    ax2.set_ylabel('Max Equivalent Creep Strain [%]')
    ax2.set_title('Creep Strain Accumulation')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_strain_accumulation.png')
    print(f"  Saved: {output_prefix}_strain_accumulation.png")
    plt.close()


def plot_2d_field_snapshot(data, output_prefix, frame_idx=-1):
    """Plot 2D contour snapshots of key fields."""
    
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Get coordinates and field values
    coords = data['elem_centers']
    x = coords[:, 0] * 1e3  # Convert to mm
    y = coords[:, 1] * 1e3
    
    vm_stress = data['von_mises'][frame_idx] / 1e6  # MPa
    temperature = data['temperature'][frame_idx] - 273.15  # C
    damage = data['damage_D'][frame_idx]
    
    time_val = data['time'][frame_idx] / 60.0  # min
    
    # Von Mises stress
    ax1 = fig.add_subplot(gs[0, 0])
    sc1 = ax1.scatter(x, y, c=vm_stress, s=2, cmap='jet', vmin=0)
    ax1.set_xlabel('x [mm]')
    ax1.set_ylabel('y [mm]')
    ax1.set_title(f'von Mises Stress [MPa] @ t={time_val:.1f} min')
    plt.colorbar(sc1, ax=ax1)
    add_layer_lines(ax1)
    
    # Temperature
    ax2 = fig.add_subplot(gs[0, 1])
    sc2 = ax2.scatter(x, y, c=temperature, s=2, cmap='hot')
    ax2.set_xlabel('x [mm]')
    ax2.set_ylabel('y [mm]')
    ax2.set_title(f'Temperature [°C] @ t={time_val:.1f} min')
    plt.colorbar(sc2, ax=ax2)
    add_layer_lines(ax2)
    
    # Damage
    ax3 = fig.add_subplot(gs[1, 0])
    sc3 = ax3.scatter(x, y, c=damage, s=2, cmap='Reds', vmin=0, vmax=1)
    ax3.set_xlabel('x [mm]')
    ax3.set_ylabel('y [mm]')
    ax3.set_title(f'Damage D @ t={time_val:.1f} min')
    plt.colorbar(sc3, ax=ax3)
    add_layer_lines(ax3)
    
    # PEEQ (if available)
    ax4 = fig.add_subplot(gs[1, 1])
    if 'peeq' in data:
        peeq = data['peeq'][frame_idx] * 100  # percent
        sc4 = ax4.scatter(x, y, c=peeq, s=2, cmap='viridis', vmin=0)
        ax4.set_title(f'Plastic Strain PEEQ [%] @ t={time_val:.1f} min')
        plt.colorbar(sc4, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'PEEQ not available', ha='center', va='center',
                 transform=ax4.transAxes)
        ax4.set_title('Plastic Strain (N/A)')
    
    ax4.set_xlabel('x [mm]')
    ax4.set_ylabel('y [mm]')
    add_layer_lines(ax4)
    
    plt.savefig(f'{output_prefix}_field_snapshot.png')
    print(f"  Saved: {output_prefix}_field_snapshot.png")
    plt.close()


def add_layer_lines(ax):
    """Add horizontal lines marking layer boundaries."""
    ax.axhline(Y_ANODE_TOP * 1e3, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(Y_ELYTE_TOP * 1e3, color='white', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(Y_CATH_TOP * 1e3, color='white', linestyle='--', linewidth=0.8, alpha=0.5)


def plot_comparison(data_dict, output_prefix='comparison'):
    """Compare results from multiple scenarios."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['r', 'g', 'b', 'orange', 'purple']
    
    for idx, (scenario, data) in enumerate(data_dict.items()):
        color = colors[idx % len(colors)]
        time_min = data['time'] / 60.0
        
        # Max von Mises
        ax = axes[0, 0]
        max_vm = np.max(data['von_mises'], axis=1) / 1e6
        ax.plot(time_min, max_vm, color=color, linewidth=2, label=scenario)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Max von Mises [MPa]')
        ax.set_title('Stress Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Max damage
        ax = axes[0, 1]
        max_damage = np.max(data['damage_D'], axis=1)
        ax.plot(time_min, max_damage, color=color, linewidth=2, label=scenario)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Max Damage D')
        ax.set_title('Damage Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Crack depth
        ax = axes[1, 0]
        crack_depth = data['crack_depth_um']
        ax.plot(time_min, crack_depth, color=color, linewidth=2, label=scenario)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Crack Depth [μm]')
        ax.set_title('Crack Depth Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Delamination risk (worst case)
        ax = axes[1, 1]
        max_risk = np.maximum(
            data['delamination_risk_AE'],
            np.maximum(data['delamination_risk_EC'], data['delamination_risk_CI'])
        )
        ax.plot(time_min, max_risk, color=color, linewidth=2, label=scenario)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time [min]')
        ax.set_ylabel('Max Delamination Risk')
        ax.set_title('Worst-Case Delamination Risk')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_scenarios.png')
    print(f"  Saved: {output_prefix}_scenarios.png")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def process_npz_file(npz_path):
    """Process a single NPZ file and generate all plots."""
    
    print(f"\nProcessing: {npz_path}")
    print("-" * 70)
    
    # Load data
    data = np.load(npz_path)
    
    # Output prefix (remove .npz extension)
    output_prefix = os.path.splitext(npz_path)[0]
    
    # Generate plots
    print("Generating plots...")
    
    plot_thermal_history(data, output_prefix)
    plot_stress_evolution(data, output_prefix)
    plot_delamination_risk(data, output_prefix)
    plot_damage_evolution(data, output_prefix)
    plot_strain_accumulation(data, output_prefix)
    plot_2d_field_snapshot(data, output_prefix, frame_idx=-1)
    
    print(f"\nCompleted: {npz_path}")
    print("=" * 70)


def main():
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_results.py <npz_file>")
        print("  python visualize_results.py --all")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        # Process all NPZ files in current directory
        npz_files = glob.glob('*_results.npz')
        
        if not npz_files:
            print("No *_results.npz files found in current directory.")
            sys.exit(1)
        
        print(f"Found {len(npz_files)} NPZ files")
        
        # Process each file
        data_dict = {}
        for npz_file in sorted(npz_files):
            process_npz_file(npz_file)
            
            # Load for comparison
            scenario_name = os.path.basename(npz_file).replace('Job_SOFC_', '').replace('_results.npz', '')
            data_dict[scenario_name] = np.load(npz_file)
        
        # Generate comparison plot if multiple scenarios
        if len(data_dict) > 1:
            print("\nGenerating comparison plots...")
            plot_comparison(data_dict, output_prefix='SOFC_comparison')
        
    else:
        # Process single file
        npz_path = sys.argv[1]
        
        if not os.path.exists(npz_path):
            print(f"ERROR: File not found: {npz_path}")
            sys.exit(1)
        
        process_npz_file(npz_path)
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
