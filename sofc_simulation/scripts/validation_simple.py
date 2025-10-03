#!/usr/bin/env python3
"""
SOFC Simple Validation Script
Validates mock simulation results against expected behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def load_expected_behavior():
    """Load expected behavior patterns for validation"""
    
    expected = {
        'HR1': {
            'temperature_range': (25, 900),
            'stress_range': (30e6, 80e6),  # Pa
            'damage_range': (0.05, 0.15),
            'delamination_critical': (5, 20)
        },
        'HR4': {
            'temperature_range': (25, 900),
            'stress_range': (50e6, 120e6),  # Pa
            'damage_range': (0.10, 0.25),
            'delamination_critical': (15, 35)
        },
        'HR10': {
            'temperature_range': (25, 900),
            'stress_range': (80e6, 150e6),  # Pa
            'damage_range': (0.15, 0.35),
            'delamination_critical': (30, 60)
        }
    }
    
    return expected

def validate_temperature_response(results, expected, heating_rate):
    """Validate temperature response"""
    
    print(f"Validating temperature response for {heating_rate}...")
    
    temp_mean = results['temperature']['mean']
    temp_max = results['temperature']['max']
    temp_min = results['temperature']['min']
    
    expected_min, expected_max = expected['temperature_range']
    
    # Validation criteria
    temp_max_valid = expected_min <= temp_max <= expected_max + 50  # Allow some overshoot
    temp_min_valid = expected_min - 10 <= temp_min <= expected_min + 10  # Allow some undershoot
    
    validation_results = {
        'temperature_max_valid': temp_max_valid,
        'temperature_min_valid': temp_min_valid,
        'temperature_max': temp_max,
        'temperature_min': temp_min,
        'expected_range': expected['temperature_range']
    }
    
    print(f"  Temperature max: {temp_max:.1f}°C (expected: {expected_min}-{expected_max}°C)")
    print(f"  Temperature min: {temp_min:.1f}°C (expected: ~{expected_min}°C)")
    print(f"  Max valid: {'✓' if temp_max_valid else '✗'}")
    print(f"  Min valid: {'✓' if temp_min_valid else '✗'}")
    
    return validation_results

def validate_stress_response(results, expected, heating_rate):
    """Validate stress response"""
    
    print(f"Validating stress response for {heating_rate}...")
    
    stress_max = results['stress']['max']
    stress_mean = results['stress']['mean']
    
    expected_min, expected_max = expected['stress_range']
    
    # Validation criteria
    stress_valid = expected_min <= stress_max <= expected_max
    
    validation_results = {
        'stress_valid': stress_valid,
        'stress_max': stress_max,
        'stress_mean': stress_mean,
        'expected_range': expected['stress_range']
    }
    
    print(f"  Stress max: {stress_max/1e6:.1f} MPa (expected: {expected_min/1e6:.1f}-{expected_max/1e6:.1f} MPa)")
    print(f"  Stress valid: {'✓' if stress_valid else '✗'}")
    
    return validation_results

def validate_damage_response(results, expected, heating_rate):
    """Validate damage response"""
    
    print(f"Validating damage response for {heating_rate}...")
    
    damage_max = results['damage']['max']
    damage_mean = results['damage']['mean']
    damaged_nodes = results['damage']['damaged_nodes']
    
    expected_min, expected_max = expected['damage_range']
    
    # Validation criteria
    damage_valid = expected_min <= damage_max <= expected_max
    
    validation_results = {
        'damage_valid': damage_valid,
        'damage_max': damage_max,
        'damage_mean': damage_mean,
        'damaged_nodes': damaged_nodes,
        'expected_range': expected['damage_range']
    }
    
    print(f"  Damage max: {damage_max:.3f} (expected: {expected_min:.3f}-{expected_max:.3f})")
    print(f"  Damaged nodes: {damaged_nodes}")
    print(f"  Damage valid: {'✓' if damage_valid else '✗'}")
    
    return validation_results

def validate_delamination_response(results, expected, heating_rate):
    """Validate delamination response"""
    
    print(f"Validating delamination response for {heating_rate}...")
    
    delam_max = results['delamination']['max']
    delam_mean = results['delamination']['mean']
    critical_nodes = results['delamination']['critical_nodes']
    
    expected_min, expected_max = expected['delamination_critical']
    
    # Validation criteria
    delam_valid = expected_min <= critical_nodes <= expected_max
    
    validation_results = {
        'delamination_valid': delam_valid,
        'delamination_max': delam_max,
        'delamination_mean': delam_mean,
        'critical_nodes': critical_nodes,
        'expected_range': expected['delamination_critical']
    }
    
    print(f"  Critical nodes: {critical_nodes} (expected: {expected_min}-{expected_max})")
    print(f"  Delamination valid: {'✓' if delam_valid else '✗'}")
    
    return validation_results

def create_validation_plots(validation_results, output_dir):
    """Create validation plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    heating_rates = ['HR1', 'HR4', 'HR10']
    colors = ['blue', 'orange', 'green']
    
    # Temperature validation
    temp_maxes = [validation_results[hr]['temperature']['temperature_max'] for hr in heating_rates]
    temp_mins = [validation_results[hr]['temperature']['temperature_min'] for hr in heating_rates]
    
    x_pos = np.arange(len(heating_rates))
    width = 0.35
    
    axes[0,0].bar(x_pos - width/2, temp_maxes, width, label='Max Temp', color='red', alpha=0.7)
    axes[0,0].bar(x_pos + width/2, temp_mins, width, label='Min Temp', color='blue', alpha=0.7)
    axes[0,0].set_xlabel('Heating Rate')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].set_title('Temperature Validation')
    axes[0,0].set_xticks(x_pos)
    axes[0,0].set_xticklabels(heating_rates)
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].axhline(y=900, color='red', linestyle='--', alpha=0.7, label='Target Max')
    axes[0,0].axhline(y=25, color='blue', linestyle='--', alpha=0.7, label='Target Min')
    
    # Stress validation
    stress_maxes = [validation_results[hr]['stress']['stress_max']/1e6 for hr in heating_rates]
    expected_ranges = [validation_results[hr]['stress']['expected_range'] for hr in heating_rates]
    
    axes[0,1].bar(heating_rates, stress_maxes, color=colors, alpha=0.7)
    for i, (hr, stress_max, expected_range) in enumerate(zip(heating_rates, stress_maxes, expected_ranges)):
        axes[0,1].errorbar(i, stress_max, yerr=0, marker='o', color='black', markersize=6)
        axes[0,1].axhspan(expected_range[0]/1e6, expected_range[1]/1e6, alpha=0.2, color=colors[i])
    axes[0,1].set_xlabel('Heating Rate')
    axes[0,1].set_ylabel('Max Stress (MPa)')
    axes[0,1].set_title('Stress Validation')
    axes[0,1].grid(True, alpha=0.3)
    
    # Damage validation
    damage_maxes = [validation_results[hr]['damage']['damage_max'] for hr in heating_rates]
    damage_ranges = [validation_results[hr]['damage']['expected_range'] for hr in heating_rates]
    
    axes[1,0].bar(heating_rates, damage_maxes, color=colors, alpha=0.7)
    for i, (hr, damage_max, expected_range) in enumerate(zip(heating_rates, damage_maxes, damage_ranges)):
        axes[1,0].errorbar(i, damage_max, yerr=0, marker='o', color='black', markersize=6)
        axes[1,0].axhspan(expected_range[0], expected_range[1], alpha=0.2, color=colors[i])
    axes[1,0].set_xlabel('Heating Rate')
    axes[1,0].set_ylabel('Max Damage')
    axes[1,0].set_title('Damage Validation')
    axes[1,0].grid(True, alpha=0.3)
    
    # Delamination validation
    critical_nodes = [validation_results[hr]['delamination']['critical_nodes'] for hr in heating_rates]
    delam_ranges = [validation_results[hr]['delamination']['expected_range'] for hr in heating_rates]
    
    axes[1,1].bar(heating_rates, critical_nodes, color=colors, alpha=0.7)
    for i, (hr, critical, expected_range) in enumerate(zip(heating_rates, critical_nodes, delam_ranges)):
        axes[1,1].errorbar(i, critical, yerr=0, marker='o', color='black', markersize=6)
        axes[1,1].axhspan(expected_range[0], expected_range[1], alpha=0.2, color=colors[i])
    axes[1,1].set_xlabel('Heating Rate')
    axes[1,1].set_ylabel('Critical Nodes')
    axes[1,1].set_title('Delamination Validation')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main validation function"""
    
    print("SOFC Simple Validation")
    print("="*50)
    
    # Load expected behavior
    expected_behavior = load_expected_behavior()
    
    # Load simulation results
    base_dir = Path("/workspace/sofc_simulation")
    validation_results = {}
    
    for heating_rate in ['HR1', 'HR4', 'HR10']:
        results_file = base_dir / "outputs" / f"sofc_{heating_rate.lower()}" / "results" / "results.json"
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                sim_results = json.load(f)
            
            print(f"\n{'='*60}")
            print(f"Validating {heating_rate}")
            print(f"{'='*60}")
            
            # Run all validations
            validation_results[heating_rate] = {
                'temperature': validate_temperature_response(sim_results, expected_behavior[heating_rate], heating_rate),
                'stress': validate_stress_response(sim_results, expected_behavior[heating_rate], heating_rate),
                'damage': validate_damage_response(sim_results, expected_behavior[heating_rate], heating_rate),
                'delamination': validate_delamination_response(sim_results, expected_behavior[heating_rate], heating_rate)
            }
        else:
            print(f"Warning: Results file {results_file} not found")
    
    # Create validation plots
    validation_dir = base_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    create_validation_plots(validation_results, validation_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    for heating_rate, results in validation_results.items():
        print(f"\n{heating_rate}:")
        
        # Count validations
        valid_count = 0
        total_count = 0
        
        for category, validation in results.items():
            for key, value in validation.items():
                if key.endswith('_valid'):
                    total_count += 1
                    if value:
                        valid_count += 1
        
        print(f"  Validation: {valid_count}/{total_count} passed")
        
        # Individual results
        temp_valid = results['temperature']['temperature_max_valid'] and results['temperature']['temperature_min_valid']
        stress_valid = results['stress']['stress_valid']
        damage_valid = results['damage']['damage_valid']
        delam_valid = results['delamination']['delamination_valid']
        
        print(f"  Temperature: {'✓' if temp_valid else '✗'}")
        print(f"  Stress: {'✓' if stress_valid else '✗'}")
        print(f"  Damage: {'✓' if damage_valid else '✗'}")
        print(f"  Delamination: {'✓' if delam_valid else '✗'}")
    
    # Save validation results
    with open(validation_dir / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\nValidation results saved to {validation_dir}")

if __name__ == "__main__":
    main()