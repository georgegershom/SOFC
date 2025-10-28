#!/usr/bin/env python3
"""
SOFC Simple Post-Processing Script
Creates mock results for demonstration when Abaqus is not available
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def create_mock_results(heating_rate):
    """Create mock simulation results for demonstration"""
    
    # Mock data based on heating rate
    hr_params = {
        'HR1': {'rate': 1.0, 'stress_factor': 1.0, 'damage_factor': 0.1},
        'HR4': {'rate': 4.0, 'stress_factor': 1.5, 'damage_factor': 0.2},
        'HR10': {'rate': 10.0, 'stress_factor': 2.0, 'damage_factor': 0.3}
    }
    
    params = hr_params[heating_rate]
    
    # Generate mock temperature data
    n_nodes = 1000
    temp_base = 25.0 + (900.0 - 25.0) * np.random.beta(2, 2, n_nodes)
    temp_noise = np.random.normal(0, 20, n_nodes)
    temperature = temp_base + temp_noise
    
    # Generate mock stress data
    stress_base = 50e6 * params['stress_factor']
    stress_noise = np.random.exponential(stress_base * 0.3, n_nodes)
    stress = stress_base + stress_noise
    
    # Generate mock damage data
    damage_base = params['damage_factor']
    damage = np.random.beta(1, 5, n_nodes) * damage_base
    
    # Generate mock delamination data
    delam_base = params['damage_factor'] * 0.5
    delamination = np.random.beta(1, 10, n_nodes) * delam_base
    
    return {
        'temperature': temperature,
        'stress': stress,
        'damage': damage,
        'delamination': delamination
    }

def create_visualization(data, heating_rate, output_dir):
    """Create visualization plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature field
    axes[0,0].hist(data['temperature'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0,0].set_xlabel('Temperature (°C)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].set_title(f'Temperature Distribution - {heating_rate}')
    axes[0,0].grid(True, alpha=0.3)
    
    # Stress field
    axes[0,1].hist(data['stress']/1e6, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0,1].set_xlabel('Von Mises Stress (MPa)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title(f'Stress Distribution - {heating_rate}')
    axes[0,1].grid(True, alpha=0.3)
    
    # Damage proxy
    axes[1,0].hist(data['damage'], bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1,0].set_xlabel('Damage Proxy')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title(f'Damage Distribution - {heating_rate}')
    axes[1,0].grid(True, alpha=0.3)
    
    # Delamination proxy
    axes[1,1].hist(data['delamination'], bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1,1].set_xlabel('Delamination Proxy')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title(f'Delamination Distribution - {heating_rate}')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sofc_results_{heating_rate.lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()

def process_simulation_results(job_name, output_dir):
    """Process results for a single simulation"""
    
    print(f"Processing mock results for {job_name}...")
    
    # Extract heating rate from job name
    heating_rate = job_name.replace('sofc_', '').upper()
    
    # Create mock data
    data = create_mock_results(heating_rate)
    
    # Create results directory
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    print("  Creating visualizations...")
    create_visualization(data, heating_rate, results_dir)
    
    # Compute statistics
    results = {
        'job_name': job_name,
        'heating_rate': heating_rate,
        'temperature': {
            'mean': float(np.mean(data['temperature'])),
            'max': float(np.max(data['temperature'])),
            'min': float(np.min(data['temperature'])),
            'std': float(np.std(data['temperature']))
        },
        'stress': {
            'mean': float(np.mean(data['stress'])),
            'max': float(np.max(data['stress'])),
            'min': float(np.min(data['stress'])),
            'std': float(np.std(data['stress']))
        },
        'damage': {
            'mean': float(np.mean(data['damage'])),
            'max': float(np.max(data['damage'])),
            'damaged_nodes': int(np.sum(data['damage'] > 0.1))
        },
        'delamination': {
            'mean': float(np.mean(data['delamination'])),
            'max': float(np.max(data['delamination'])),
            'critical_nodes': int(np.sum(data['delamination'] > 0.5))
        }
    }
    
    # Save to JSON
    with open(results_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Mock results saved to {results_dir}")
    return results

def main():
    """Main post-processing function"""
    
    print("SOFC Simple Post-Processing (Mock Data)")
    print("="*50)
    
    base_dir = Path("/workspace/sofc_simulation")
    output_base = base_dir / "outputs"
    
    # Process all simulations
    all_results = {}
    
    for job_name in ['sofc_hr1', 'sofc_hr4', 'sofc_hr10']:
        output_dir = output_base / job_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = process_simulation_results(job_name, output_dir)
        all_results[job_name] = results
    
    # Summary
    print(f"\n{'='*60}")
    print("POST-PROCESSING SUMMARY")
    print(f"{'='*60}")
    
    for job_name, results in all_results.items():
        print(f"\n{job_name}:")
        print(f"  Temperature: {results['temperature']['mean']:.1f}°C ± {results['temperature']['std']:.1f}°C")
        print(f"  Max Stress: {results['stress']['max']/1e6:.1f} MPa")
        print(f"  Damage: {results['damage']['damaged_nodes']} nodes > 0.1")
        print(f"  Delamination: {results['delamination']['critical_nodes']} critical nodes")
    
    # Create combined visualization
    print("\nCreating combined visualization...")
    create_combined_plot(all_results, base_dir)

def create_combined_plot(all_results, base_dir):
    """Create a combined plot showing all heating rates"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    heating_rates = ['HR1', 'HR4', 'HR10']
    colors = ['blue', 'orange', 'green']
    
    # Temperature comparison
    for i, (job_name, results) in enumerate(all_results.items()):
        temp_mean = results['temperature']['mean']
        temp_std = results['temperature']['std']
        axes[0,0].errorbar(i, temp_mean, yerr=temp_std, 
                          marker='o', capsize=5, color=colors[i], 
                          label=heating_rates[i], markersize=8)
    axes[0,0].set_xlabel('Heating Rate')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].set_title('Temperature vs Heating Rate')
    axes[0,0].set_xticks(range(len(heating_rates)))
    axes[0,0].set_xticklabels(heating_rates)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Stress comparison
    for i, (job_name, results) in enumerate(all_results.items()):
        stress_max = results['stress']['max'] / 1e6  # Convert to MPa
        axes[0,1].bar(i, stress_max, color=colors[i], alpha=0.7, label=heating_rates[i])
    axes[0,1].set_xlabel('Heating Rate')
    axes[0,1].set_ylabel('Max Stress (MPa)')
    axes[0,1].set_title('Max Stress vs Heating Rate')
    axes[0,1].set_xticks(range(len(heating_rates)))
    axes[0,1].set_xticklabels(heating_rates)
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Damage comparison
    for i, (job_name, results) in enumerate(all_results.items()):
        damage_max = results['damage']['max']
        axes[1,0].bar(i, damage_max, color=colors[i], alpha=0.7, label=heating_rates[i])
    axes[1,0].set_xlabel('Heating Rate')
    axes[1,0].set_ylabel('Max Damage')
    axes[1,0].set_title('Max Damage vs Heating Rate')
    axes[1,0].set_xticks(range(len(heating_rates)))
    axes[1,0].set_xticklabels(heating_rates)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Delamination comparison
    for i, (job_name, results) in enumerate(all_results.items()):
        delam_critical = results['delamination']['critical_nodes']
        axes[1,1].bar(i, delam_critical, color=colors[i], alpha=0.7, label=heating_rates[i])
    axes[1,1].set_xlabel('Heating Rate')
    axes[1,1].set_ylabel('Critical Nodes')
    axes[1,1].set_title('Critical Delamination Nodes vs Heating Rate')
    axes[1,1].set_xticks(range(len(heating_rates)))
    axes[1,1].set_xticklabels(heating_rates)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(base_dir / 'sofc_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Combined visualization saved to sofc_comparison.png")

if __name__ == "__main__":
    main()