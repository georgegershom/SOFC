#!/usr/bin/env python3
"""
SOFC Experimental Data Explorer
Quick exploration and analysis of the generated experimental datasets
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Load all experimental data"""
    print("🔬 Loading SOFC Experimental Data...")
    
    with open('/workspace/sofc_experimental_data/dic_data.json', 'r') as f:
        dic_data = json.load(f)
    
    with open('/workspace/sofc_experimental_data/xrd_data.json', 'r') as f:
        xrd_data = json.load(f)
    
    with open('/workspace/sofc_experimental_data/post_mortem_data.json', 'r') as f:
        post_mortem_data = json.load(f)
    
    print("✅ Data loaded successfully!")
    return dic_data, xrd_data, post_mortem_data

def explore_dic_data(dic_data):
    """Explore DIC data statistics"""
    print("\n📊 Digital Image Correlation (DIC) Data Analysis")
    print("=" * 50)
    
    # Sintering data analysis
    sintering_data = dic_data['sintering']
    temps = [data['temperature'] for data in sintering_data]
    max_strains = [data['max_strain'] for data in sintering_data]
    mean_strains = [data['mean_strain'] for data in sintering_data]
    
    print(f"🌡️  Temperature Range: {min(temps):.0f}°C - {max(temps):.0f}°C")
    print(f"📈 Maximum Strain Range: {min(max_strains):.4f} - {max(max_strains):.4f}")
    print(f"📊 Mean Strain Range: {min(mean_strains):.4f} - {max(mean_strains):.4f}")
    print(f"🔢 Data Points: {len(sintering_data)}")
    
    # Thermal cycling analysis
    thermal_data = dic_data['thermal_cycling']
    cycles = set([data['cycle'] for data in thermal_data])
    print(f"🔄 Thermal Cycles: {len(cycles)} cycles")
    
    # Startup/shutdown analysis
    startup_data = dic_data['startup_shutdown']
    startup_cycles = set([data['cycle'] for data in startup_data])
    print(f"🚀 Startup/Shutdown Cycles: {len(startup_cycles)} cycles")
    
    # Speckle patterns
    speckle_data = dic_data['speckle_patterns']
    print(f"📸 Speckle Patterns: {len(speckle_data)} images")
    
    # Lagrangian tensors
    tensor_data = dic_data['lagrangian_tensors']
    print(f"📐 Strain Tensors: {len(tensor_data)} measurement points")
    
    return {
        'temperature_range': (min(temps), max(temps)),
        'strain_range': (min(max_strains), max(max_strains)),
        'data_points': len(sintering_data),
        'thermal_cycles': len(cycles),
        'startup_cycles': len(startup_cycles)
    }

def explore_xrd_data(xrd_data):
    """Explore XRD data statistics"""
    print("\n🔬 Synchrotron X-ray Diffraction (XRD) Data Analysis")
    print("=" * 55)
    
    # Residual stress analysis
    residual_data = xrd_data['residual_stresses']
    stresses = [data['stress'] for data in residual_data]
    positions = [data['position'] for data in residual_data]
    layers = [data['layer'] for data in residual_data]
    
    print(f"📏 Cross-section Length: {max(positions):.1f} mm")
    print(f"💪 Stress Range: {min(stresses):.1f} - {max(stresses):.1f} MPa")
    print(f"📊 Average Stress: {np.mean(stresses):.1f} MPa")
    print(f"🔢 Measurement Points: {len(residual_data)}")
    
    # Layer analysis
    layer_counts = {}
    layer_stresses = {}
    for layer in set(layers):
        layer_counts[layer] = layers.count(layer)
        layer_stresses[layer] = [s for s, l in zip(stresses, layers) if l == layer]
        print(f"  {layer.capitalize()}: {layer_counts[layer]} points, "
              f"avg stress: {np.mean(layer_stresses[layer]):.1f} MPa")
    
    # Lattice strain analysis
    lattice_data = xrd_data['lattice_strains']
    materials = set([data['material'] for data in lattice_data])
    print(f"🧪 Materials Analyzed: {', '.join(materials)}")
    print(f"🌡️  Temperature Points: {len(set([data['temperature'] for data in lattice_data]))}")
    
    # Microcrack analysis
    microcrack_data = xrd_data['microcrack_data']
    crack_initiated = sum([1 for data in microcrack_data if data['crack_initiated']])
    print(f"💥 Microcrack Tests: {len(microcrack_data)}")
    print(f"🔴 Cracks Initiated: {crack_initiated} ({crack_initiated/len(microcrack_data)*100:.1f}%)")
    
    return {
        'stress_range': (min(stresses), max(stresses)),
        'materials': materials,
        'crack_initiation_rate': crack_initiated/len(microcrack_data)
    }

def explore_post_mortem_data(post_mortem_data):
    """Explore post-mortem analysis data statistics"""
    print("\n🔍 Post-Mortem Analysis Data")
    print("=" * 35)
    
    # SEM analysis
    sem_data = post_mortem_data['sem_images']
    crack_densities = [data['crack_density'] for data in sem_data]
    magnifications = [data['magnification'] for data in sem_data]
    
    print(f"📸 SEM Images: {len(sem_data)}")
    print(f"💥 Crack Density Range: {min(crack_densities):.2f} - {max(crack_densities):.2f} cracks/mm²")
    print(f"🔍 Magnification Range: {min(magnifications)}× - {max(magnifications)}×")
    
    # EDS analysis
    eds_data = post_mortem_data['eds_scans']
    print(f"🧪 EDS Line Scans: {len(eds_data)}")
    print(f"📏 Scan Length: {eds_data[0]['step_size'] * len(eds_data[0]['scan_data']):.1f} μm")
    
    # Nano-indentation analysis
    nano_data = post_mortem_data['nano_indentation']
    youngs_modulus = [data['youngs_modulus'] for data in nano_data]
    hardness = [data['hardness'] for data in nano_data]
    phases = [data['phase'] for data in nano_data]
    
    print(f"🔬 Nano-indentation Points: {len(nano_data)}")
    print(f"💪 Young's Modulus Range: {min(youngs_modulus):.1f} - {max(youngs_modulus):.1f} GPa")
    print(f"🪨 Hardness Range: {min(hardness):.1f} - {max(hardness):.1f} GPa")
    
    # Phase analysis
    phase_counts = {}
    for phase in set(phases):
        phase_counts[phase] = phases.count(phase)
        phase_moduli = [m for m, p in zip(youngs_modulus, phases) if p == phase]
        print(f"  {phase}: {phase_counts[phase]} points, "
              f"avg E: {np.mean(phase_moduli):.1f} GPa")
    
    return {
        'crack_density_range': (min(crack_densities), max(crack_densities)),
        'youngs_modulus_range': (min(youngs_modulus), max(youngs_modulus)),
        'phases': set(phases)
    }

def create_quick_plots(dic_data, xrd_data, post_mortem_data):
    """Create quick summary plots"""
    print("\n📊 Creating Quick Summary Plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SOFC Experimental Data Quick Summary', fontsize=16, fontweight='bold')
    
    # DIC strain vs temperature
    ax1 = axes[0, 0]
    sintering_data = dic_data['sintering']
    temps = [data['temperature'] for data in sintering_data]
    max_strains = [data['max_strain'] for data in sintering_data]
    ax1.plot(temps, max_strains, 'r-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Maximum Strain')
    ax1.set_title('DIC: Strain vs Temperature')
    ax1.grid(True, alpha=0.3)
    
    # XRD residual stress
    ax2 = axes[0, 1]
    residual_data = xrd_data['residual_stresses']
    positions = [data['position'] for data in residual_data]
    stresses = [data['stress'] for data in residual_data]
    ax2.plot(positions, stresses, 'b-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Position (mm)')
    ax2.set_ylabel('Residual Stress (MPa)')
    ax2.set_title('XRD: Residual Stress Profile')
    ax2.grid(True, alpha=0.3)
    
    # Post-mortem crack density
    ax3 = axes[1, 0]
    sem_data = post_mortem_data['sem_images']
    crack_densities = [data['crack_density'] for data in sem_data]
    ax3.hist(crack_densities, bins=8, alpha=0.7, color='green', edgecolor='black')
    ax3.set_xlabel('Crack Density (cracks/mm²)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Post-Mortem: Crack Density Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Nano-indentation Young's modulus
    ax4 = axes[1, 1]
    nano_data = post_mortem_data['nano_indentation']
    youngs_modulus = [data['youngs_modulus'] for data in nano_data]
    phases = [data['phase'] for data in nano_data]
    
    # Box plot by phase
    phase_data = {}
    for phase in set(phases):
        phase_data[phase] = [m for m, p in zip(youngs_modulus, phases) if p == phase]
    
    box_data = [phase_data[phase] for phase in phase_data.keys()]
    box_labels = list(phase_data.keys())
    
    bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4.set_ylabel('Young\'s Modulus (GPa)')
    ax4.set_title('Nano-indentation: Young\'s Modulus by Phase')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/sofc_experimental_data/quick_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Quick summary plot saved as 'quick_summary.png'")

def main():
    """Main exploration function"""
    print("🔬 SOFC Experimental Data Explorer")
    print("=" * 40)
    
    # Load data
    dic_data, xrd_data, post_mortem_data = load_data()
    
    # Explore each dataset
    dic_stats = explore_dic_data(dic_data)
    xrd_stats = explore_xrd_data(xrd_data)
    post_mortem_stats = explore_post_mortem_data(post_mortem_data)
    
    # Create quick plots
    create_quick_plots(dic_data, xrd_data, post_mortem_data)
    
    # Summary
    print("\n📋 Dataset Summary")
    print("=" * 20)
    print(f"🌡️  Temperature Range: {dic_stats['temperature_range'][0]:.0f}°C - {dic_stats['temperature_range'][1]:.0f}°C")
    print(f"📈 Strain Range: {dic_stats['strain_range'][0]:.4f} - {dic_stats['strain_range'][1]:.4f}")
    print(f"💪 Stress Range: {xrd_stats['stress_range'][0]:.1f} - {xrd_stats['stress_range'][1]:.1f} MPa")
    print(f"💥 Crack Initiation Rate: {xrd_stats['crack_initiation_rate']*100:.1f}%")
    print(f"🔬 Materials Analyzed: {len(xrd_stats['materials'])}")
    print(f"📊 Total Data Points: {dic_stats['data_points'] + len(xrd_data['residual_stresses']) + len(post_mortem_data['nano_indentation'])}")
    
    print("\n✅ Data exploration complete!")
    print("📁 Check the generated plots and summary files in the data directory.")

if __name__ == "__main__":
    main()