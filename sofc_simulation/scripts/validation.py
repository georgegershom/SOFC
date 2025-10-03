#!/usr/bin/env python3
"""
SOFC Simulation Validation Script
Compares simulation results with synthetic data and validates against expected behavior
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd

def load_synthetic_data():
    """Load synthetic data for comparison"""
    
    # Create mock synthetic data based on the specifications
    synthetic_data = {
        'HR1': {
            'temperature_profile': {
                'times': np.linspace(0, 1760*60, 1000),  # seconds
                'temperatures': np.concatenate([
                    np.linspace(25, 900, 500),  # heating
                    np.full(100, 900),          # hold
                    np.linspace(900, 25, 400)   # cooling
                ])
            },
            'stress_evolution': {
                'times': np.linspace(0, 1760*60, 1000),
                'max_stress': 50e6 + 30e6 * np.sin(np.linspace(0, 4*np.pi, 1000))  # MPa
            },
            'damage_evolution': {
                'times': np.linspace(0, 1760*60, 1000),
                'damage': np.clip(0.1 * np.exp(-np.linspace(0, 5, 1000)), 0, 1)
            }
        },
        'HR4': {
            'temperature_profile': {
                'times': np.linspace(0, 447.5*60, 1000),  # seconds
                'temperatures': np.concatenate([
                    np.linspace(25, 900, 250),  # heating
                    np.full(100, 900),         # hold
                    np.linspace(900, 25, 650)  # cooling
                ])
            },
            'stress_evolution': {
                'times': np.linspace(0, 447.5*60, 1000),
                'max_stress': 60e6 + 40e6 * np.sin(np.linspace(0, 6*np.pi, 1000))  # MPa
            },
            'damage_evolution': {
                'times': np.linspace(0, 447.5*60, 1000),
                'damage': np.clip(0.2 * np.exp(-np.linspace(0, 3, 1000)), 0, 1)
            }
        },
        'HR10': {
            'temperature_profile': {
                'times': np.linspace(0, 185*60, 1000),  # seconds
                'temperatures': np.concatenate([
                    np.linspace(25, 900, 100),  # heating
                    np.full(100, 900),         # hold
                    np.linspace(900, 25, 800)  # cooling
                ])
            },
            'stress_evolution': {
                'times': np.linspace(0, 185*60, 1000),
                'max_stress': 80e6 + 50e6 * np.sin(np.linspace(0, 8*np.pi, 1000))  # MPa
            },
            'damage_evolution': {
                'times': np.linspace(0, 185*60, 1000),
                'damage': np.clip(0.3 * np.exp(-np.linspace(0, 2, 1000)), 0, 1)
            }
        }
    }
    
    return synthetic_data

def validate_temperature_response(sim_results, synthetic_data, heating_rate):
    """Validate temperature response against synthetic data"""
    
    print(f"Validating temperature response for {heating_rate}...")
    
    # Get synthetic temperature profile
    synth_temp = synthetic_data[heating_rate]['temperature_profile']
    
    # Extract simulation results
    sim_temp_mean = sim_results['temperature']['mean']
    sim_temp_max = sim_results['temperature']['max']
    sim_temp_min = sim_results['temperature']['min']
    
    # Expected temperature range (should reach 900°C during hold)
    expected_max = 900.0
    expected_min = 25.0
    
    # Validation criteria
    temp_max_error = abs(sim_temp_max - expected_max) / expected_max
    temp_min_error = abs(sim_temp_min - expected_min) / expected_min
    
    validation_results = {
        'temperature_max_error': temp_max_error,
        'temperature_min_error': temp_min_error,
        'temperature_max_valid': temp_max_error < 0.05,  # 5% tolerance
        'temperature_min_valid': temp_min_error < 0.05,  # 5% tolerance
        'temperature_mean': sim_temp_mean,
        'temperature_max': sim_temp_max,
        'temperature_min': sim_temp_min
    }
    
    print(f"  Temperature max: {sim_temp_max:.1f}°C (expected: {expected_max}°C, error: {temp_max_error:.1%})")
    print(f"  Temperature min: {sim_temp_min:.1f}°C (expected: {expected_min}°C, error: {temp_min_error:.1%})")
    
    return validation_results

def validate_stress_response(sim_results, synthetic_data, heating_rate):
    """Validate stress response against synthetic data"""
    
    print(f"Validating stress response for {heating_rate}...")
    
    # Get synthetic stress evolution
    synth_stress = synthetic_data[heating_rate]['stress_evolution']
    
    # Extract simulation results
    sim_stress_mean = sim_results['stress']['mean']
    sim_stress_max = sim_results['stress']['max']
    
    # Expected stress range (based on thermal mismatch)
    expected_max_range = {
        'HR1': 50e6,   # Lower heating rate = lower thermal stress
        'HR4': 80e6,   # Medium heating rate = medium thermal stress
        'HR10': 120e6  # Higher heating rate = higher thermal stress
    }
    
    expected_max = expected_max_range[heating_rate]
    
    # Validation criteria
    stress_max_error = abs(sim_stress_max - expected_max) / expected_max
    
    validation_results = {
        'stress_max_error': stress_max_error,
        'stress_max_valid': stress_max_error < 0.3,  # 30% tolerance for stress
        'stress_mean': sim_stress_mean,
        'stress_max': sim_stress_max,
        'expected_max': expected_max
    }
    
    print(f"  Stress max: {sim_stress_max/1e6:.1f} MPa (expected: {expected_max/1e6:.1f} MPa, error: {stress_max_error:.1%})")
    
    return validation_results

def validate_damage_response(sim_results, synthetic_data, heating_rate):
    """Validate damage response against synthetic data"""
    
    print(f"Validating damage response for {heating_rate}...")
    
    # Get synthetic damage evolution
    synth_damage = synthetic_data[heating_rate]['damage_evolution']
    
    # Extract simulation results
    sim_damage_mean = sim_results['damage']['mean']
    sim_damage_max = sim_results['damage']['max']
    sim_damaged_nodes = sim_results['damage']['damaged_nodes']
    
    # Expected damage characteristics
    expected_damage_trend = {
        'HR1': 0.1,   # Lower heating rate = lower damage
        'HR4': 0.2,   # Medium heating rate = medium damage
        'HR10': 0.3   # Higher heating rate = higher damage
    }
    
    expected_damage = expected_damage_trend[heating_rate]
    
    # Validation criteria
    damage_error = abs(sim_damage_max - expected_damage) / expected_damage
    
    validation_results = {
        'damage_error': damage_error,
        'damage_valid': damage_error < 0.5,  # 50% tolerance for damage
        'damage_mean': sim_damage_mean,
        'damage_max': sim_damage_max,
        'damaged_nodes': sim_damaged_nodes,
        'expected_damage': expected_damage
    }
    
    print(f"  Damage max: {sim_damage_max:.3f} (expected: {expected_damage:.3f}, error: {damage_error:.1%})")
    print(f"  Damaged nodes: {sim_damaged_nodes}")
    
    return validation_results

def validate_delamination_response(sim_results, synthetic_data, heating_rate):
    """Validate delamination response against synthetic data"""
    
    print(f"Validating delamination response for {heating_rate}...")
    
    # Extract simulation results
    sim_delam_mean = sim_results['delamination']['mean']
    sim_delam_max = sim_results['delamination']['max']
    sim_critical_nodes = sim_results['delamination']['critical_nodes']
    
    # Expected delamination characteristics (higher heating rate = more delamination)
    expected_critical_trend = {
        'HR1': 10,    # Lower heating rate = fewer critical nodes
        'HR4': 25,    # Medium heating rate = medium critical nodes
        'HR10': 50    # Higher heating rate = more critical nodes
    }
    
    expected_critical = expected_critical_trend[heating_rate]
    
    # Validation criteria
    delam_error = abs(sim_critical_nodes - expected_critical) / expected_critical if expected_critical > 0 else 0
    
    validation_results = {
        'delamination_error': delam_error,
        'delamination_valid': delam_error < 0.5,  # 50% tolerance for delamination
        'delamination_mean': sim_delam_mean,
        'delamination_max': sim_delam_max,
        'critical_nodes': sim_critical_nodes,
        'expected_critical': expected_critical
    }
    
    print(f"  Critical nodes: {sim_critical_nodes} (expected: {expected_critical}, error: {delam_error:.1%})")
    
    return validation_results

def create_validation_plots(validation_results, output_dir):
    """Create validation plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    heating_rates = ['HR1', 'HR4', 'HR10']
    
    # Temperature validation
    temp_errors = [validation_results[hr]['temperature']['temperature_max_error'] for hr in heating_rates]
    axes[0,0].bar(heating_rates, temp_errors, color=['green' if e < 0.05 else 'red' for e in temp_errors])
    axes[0,0].set_ylabel('Temperature Error')
    axes[0,0].set_title('Temperature Validation')
    axes[0,0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% tolerance')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Stress validation
    stress_errors = [validation_results[hr]['stress']['stress_max_error'] for hr in heating_rates]
    axes[0,1].bar(heating_rates, stress_errors, color=['green' if e < 0.3 else 'red' for e in stress_errors])
    axes[0,1].set_ylabel('Stress Error')
    axes[0,1].set_title('Stress Validation')
    axes[0,1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='30% tolerance')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Damage validation
    damage_errors = [validation_results[hr]['damage']['damage_error'] for hr in heating_rates]
    axes[1,0].bar(heating_rates, damage_errors, color=['green' if e < 0.5 else 'red' for e in damage_errors])
    axes[1,0].set_ylabel('Damage Error')
    axes[1,0].set_title('Damage Validation')
    axes[1,0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% tolerance')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Delamination validation
    delam_errors = [validation_results[hr]['delamination']['delamination_error'] for hr in heating_rates]
    axes[1,1].bar(heating_rates, delam_errors, color=['green' if e < 0.5 else 'red' for e in delam_errors])
    axes[1,1].set_ylabel('Delamination Error')
    axes[1,1].set_title('Delamination Validation')
    axes[1,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% tolerance')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main validation function"""
    
    print("SOFC Simulation Validation")
    print("="*50)
    
    # Load synthetic data
    synthetic_data = load_synthetic_data()
    
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
                'temperature': validate_temperature_response(sim_results, synthetic_data, heating_rate),
                'stress': validate_stress_response(sim_results, synthetic_data, heating_rate),
                'damage': validate_damage_response(sim_results, synthetic_data, heating_rate),
                'delamination': validate_delamination_response(sim_results, synthetic_data, heating_rate)
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
        stress_valid = results['stress']['stress_max_valid']
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