#!/usr/bin/env python3
"""
SOFC Simulation Results Analysis
===============================

Detailed analysis and visualization of SOFC simulation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def load_results(heating_rate):
    """Load simulation results for a given heating rate"""
    filename = f"/workspace/sofc_results_{heating_rate.lower()}/sofc_simulation_{heating_rate.lower()}.npz"
    
    if not os.path.exists(filename):
        print(f"Results file not found: {filename}")
        return None
    
    data = np.load(filename, allow_pickle=True)
    return {key: data[key] for key in data.keys()}

def analyze_thermal_behavior(results_dict):
    """Analyze thermal behavior across heating rates"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Thermal Behavior Analysis', fontsize=16)
    
    colors = {'hr1': 'blue', 'hr4': 'green', 'hr10': 'red'}
    
    # Temperature evolution at bottom and top
    ax = axes[0, 0]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        T_bottom = results['temperature'][:, 0] - 273.15
        T_top = results['temperature'][:, -1] - 273.15
        
        ax.plot(times_hr, T_bottom, color=colors[hr], linestyle='-', 
                label=f'{hr.upper()} Bottom', linewidth=2)
        ax.plot(times_hr, T_top, color=colors[hr], linestyle='--', 
                label=f'{hr.upper()} Top', alpha=0.7)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Temperature gradients
    ax = axes[0, 1]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        T_gradient = (results['temperature'][:, -1] - results['temperature'][:, 0]) / 0.001  # K/m
        
        ax.plot(times_hr, T_gradient, color=colors[hr], 
                label=f'{hr.upper()}', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Temperature Gradient (K/m)')
    ax.set_title('Through-Thickness Temperature Gradient')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final temperature profiles
    ax = axes[1, 0]
    for hr, results in results_dict.items():
        y_mm = results['coordinates'] * 1000
        T_final = results['temperature'][-1] - 273.15
        
        ax.plot(T_final, y_mm, color=colors[hr], marker='o', 
                label=f'{hr.upper()}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Final Temperature Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add layer boundaries
    for y_bound in [0.4, 0.5, 0.9]:
        ax.axhline(y=y_bound, color='gray', linestyle=':', alpha=0.5)
    
    # Heating/cooling rates
    ax = axes[1, 1]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        T_bottom = results['temperature'][:, 0] - 273.15
        
        # Calculate heating rate (dT/dt)
        if len(times_hr) > 1:
            dt = np.diff(times_hr)
            dT = np.diff(T_bottom)
            heating_rate = dT / dt  # °C/hour
            
            ax.plot(times_hr[:-1], heating_rate, color=colors[hr], 
                    label=f'{hr.upper()}', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Heating Rate (°C/hour)')
    ax.set_title('Instantaneous Heating Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/thermal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_mechanical_behavior(results_dict):
    """Analyze mechanical behavior across heating rates"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Mechanical Behavior Analysis', fontsize=16)
    
    colors = {'hr1': 'blue', 'hr4': 'green', 'hr10': 'red'}
    
    # Stress evolution
    ax = axes[0, 0]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        max_stress = np.max(np.abs(results['stress']), axis=1) / 1e6  # MPa
        
        ax.plot(times_hr, max_stress, color=colors[hr], 
                label=f'{hr.upper()}', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Maximum Stress (MPa)')
    ax.set_title('Maximum Stress Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Strain evolution
    ax = axes[0, 1]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        max_strain = np.max(np.abs(results['strain']), axis=1) * 100  # %
        
        ax.plot(times_hr, max_strain, color=colors[hr], 
                label=f'{hr.upper()}', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Maximum Strain (%)')
    ax.set_title('Maximum Strain Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final stress profiles
    ax = axes[1, 0]
    for hr, results in results_dict.items():
        # Element centers for plotting
        coords = results['coordinates']
        y_elem = []
        for i in range(len(coords) - 1):
            y_elem.append(0.5 * (coords[i] + coords[i+1]))
        y_elem_mm = np.array(y_elem) * 1000
        
        final_stress = results['stress'][-1] / 1e6  # MPa
        
        ax.plot(final_stress, y_elem_mm, color=colors[hr], marker='o', 
                label=f'{hr.upper()}', linewidth=2, markersize=4)
    
    ax.set_xlabel('Stress (MPa)')
    ax.set_ylabel('Height (mm)')
    ax.set_title('Final Stress Profiles')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add layer boundaries
    for y_bound in [0.4, 0.5, 0.9]:
        ax.axhline(y=y_bound, color='gray', linestyle=':', alpha=0.5)
    
    # Damage evolution
    ax = axes[1, 1]
    for hr, results in results_dict.items():
        times_hr = results['times'] / 3600
        max_damage = np.max(results['damage'], axis=1)
        
        ax.plot(times_hr, max_damage, color=colors[hr], 
                label=f'{hr.upper()}', linewidth=2)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Maximum Damage')
    ax.set_title('Damage Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/workspace/mechanical_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results_dict):
    """Create summary table of key results"""
    print("\n" + "="*80)
    print("SOFC SIMULATION RESULTS SUMMARY")
    print("="*80)
    
    print(f"{'Parameter':<30} {'HR1 (1°C/min)':<15} {'HR4 (4°C/min)':<15} {'HR10 (10°C/min)':<15}")
    print("-"*80)
    
    for hr, results in results_dict.items():
        if hr == 'hr1':
            # Total simulation time
            total_time = results['times'][-1] / 3600  # hours
            max_temp = np.max(results['temperature']) - 273.15  # °C
            max_stress = np.max(np.abs(results['stress'])) / 1e6  # MPa
            max_strain = np.max(np.abs(results['strain'])) * 100  # %
            max_damage = np.max(results['damage'])
            
            print(f"{'Total Time (hours)':<30} {total_time:<15.1f}", end="")
        elif hr == 'hr4':
            total_time = results['times'][-1] / 3600
            max_temp = np.max(results['temperature']) - 273.15
            max_stress = np.max(np.abs(results['stress'])) / 1e6
            max_strain = np.max(np.abs(results['strain'])) * 100
            max_damage = np.max(results['damage'])
            
            print(f" {total_time:<15.1f}", end="")
        elif hr == 'hr10':
            total_time = results['times'][-1] / 3600
            max_temp = np.max(results['temperature']) - 273.15
            max_stress = np.max(np.abs(results['stress'])) / 1e6
            max_strain = np.max(np.abs(results['strain'])) * 100
            max_damage = np.max(results['damage'])
            
            print(f" {total_time:<15.1f}")
    
    # Print other parameters
    parameters = [
        ('Max Temperature (°C)', lambda r: np.max(r['temperature']) - 273.15, '.1f'),
        ('Max Stress (MPa)', lambda r: np.max(np.abs(r['stress'])) / 1e6, '.0f'),
        ('Max Strain (%)', lambda r: np.max(np.abs(r['strain'])) * 100, '.2f'),
        ('Max Damage', lambda r: np.max(r['damage']), '.3f'),
        ('Final Damage Elements', lambda r: np.sum(r['damage'][-1] > 0.5), '.0f'),
    ]
    
    for param_name, param_func, fmt in parameters:
        print(f"{param_name:<30}", end="")
        for hr in ['hr1', 'hr4', 'hr10']:
            value = param_func(results_dict[hr])
            print(f" {value:<15{fmt}}", end="")
        print()
    
    print("-"*80)
    
    # Efficiency analysis
    print("\nEFFICIENCY ANALYSIS:")
    hr1_time = results_dict['hr1']['times'][-1] / 3600
    for hr in ['hr4', 'hr10']:
        hr_time = results_dict[hr]['times'][-1] / 3600
        speedup = hr1_time / hr_time
        print(f"{hr.upper()} is {speedup:.1f}x faster than HR1")
    
    print("\nDAMAGE ASSESSMENT:")
    for hr, results in results_dict.items():
        final_damage = results['damage'][-1]
        damaged_elements = np.sum(final_damage > 0.1)
        severely_damaged = np.sum(final_damage > 0.8)
        print(f"{hr.upper()}: {damaged_elements} elements with damage > 10%, {severely_damaged} with damage > 80%")

def main():
    """Main analysis function"""
    print("SOFC Simulation Results Analysis")
    print("="*50)
    
    # Load all results
    heating_rates = ['hr1', 'hr4', 'hr10']
    results_dict = {}
    
    for hr in heating_rates:
        results = load_results(hr)
        if results is not None:
            results_dict[hr] = results
            print(f"Loaded results for {hr.upper()}")
        else:
            print(f"Failed to load results for {hr.upper()}")
    
    if not results_dict:
        print("No results found!")
        return
    
    # Perform analyses
    print("\nGenerating thermal behavior analysis...")
    analyze_thermal_behavior(results_dict)
    
    print("Generating mechanical behavior analysis...")
    analyze_mechanical_behavior(results_dict)
    
    print("Creating summary table...")
    create_summary_table(results_dict)
    
    print("\nAnalysis completed!")
    print("Generated files:")
    print("  - thermal_analysis.png")
    print("  - mechanical_analysis.png")

if __name__ == "__main__":
    main()