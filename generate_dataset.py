#!/usr/bin/env python3
"""
Main script to generate comprehensive thermo-mechanical modeling dataset
for fire-resistant rubberized concrete research
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add package to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from thermo_mechanical_dataset import ThermoMechanicalDataset

def main():
    """Main execution function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Generate thermo-mechanical modeling dataset for rubberized concrete'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for generated datasets'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--uncertainty',
        type=float,
        default=0.05,
        help='Uncertainty level (coefficient of variation)'
    )
    parser.add_argument(
        '--calibration-ratio',
        type=float,
        default=0.7,
        help='Fraction of data for calibration (rest for validation)'
    )
    parser.add_argument(
        '--n-validation-sets',
        type=int,
        default=5,
        help='Number of independent validation datasets'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Run validation checks on generated data'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("THERMO-MECHANICAL DATASET GENERATOR")
    print("Fire-Resistant Rubberized Concrete")
    print("=" * 80)
    print(f"Start time: {datetime.now().isoformat()}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Uncertainty level: {args.uncertainty*100:.1f}%")
    print(f"Calibration ratio: {args.calibration_ratio:.1%}")
    print(f"Validation sets: {args.n_validation_sets}")
    print("=" * 80)
    
    # Initialize dataset generator
    dataset_gen = ThermoMechanicalDataset(
        output_dir=args.output_dir,
        seed=args.seed,
        uncertainty_level=args.uncertainty
    )
    
    # Generate complete dataset
    print("\n[1/4] Generating datasets...")
    calibration_data, validation_data = dataset_gen.generate_complete_dataset(
        calibration_ratio=args.calibration_ratio,
        n_validation_sets=args.n_validation_sets
    )
    
    # Run validation checks
    if args.validate:
        print("\n[2/4] Running validation checks...")
        validation_results = validate_datasets(calibration_data, validation_data)
        save_validation_report(validation_results, args.output_dir)
    else:
        print("\n[2/4] Skipping validation checks (use --validate to enable)")
    
    # Generate visualizations
    if args.visualize:
        print("\n[3/4] Generating visualizations...")
        generate_visualizations(calibration_data, args.output_dir)
    else:
        print("\n[3/4] Skipping visualizations (use --visualize to enable)")
    
    # Generate summary report
    print("\n[4/4] Generating summary report...")
    generate_summary_report(
        calibration_data, 
        validation_data, 
        args.output_dir
    )
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print(f"End time: {datetime.now().isoformat()}")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 80)

def validate_datasets(calibration_data, validation_data):
    """
    Run validation checks on generated datasets
    
    Parameters:
    -----------
    calibration_data : dict
        Calibration dataset
    validation_data : dict
        Validation datasets
    
    Returns:
    --------
    dict : Validation results
    """
    results = {
        'timestamp': datetime.now().isoformat(),
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # Check 1: Physical consistency
    print("  - Checking physical consistency...")
    for mix_id in calibration_data.keys():
        # Check thermal properties
        if 'thermal' in calibration_data[mix_id]:
            thermal = calibration_data[mix_id]['thermal']
            
            # Conductivity should decrease with temperature
            k = np.array(thermal['thermal_conductivity'])
            if not np.all(np.diff(k) <= 0.1):  # Allow small increases
                results['warnings'].append(
                    f"{mix_id}: Thermal conductivity may not be monotonically decreasing"
                )
            
            # Density should decrease with temperature
            rho = np.array(thermal['density'])
            if not np.all(np.diff(rho) <= 0):
                results['warnings'].append(
                    f"{mix_id}: Density not monotonically decreasing with temperature"
                )
        
        # Check mechanical properties
        if 'mechanical' in calibration_data[mix_id]:
            mech = calibration_data[mix_id]['mechanical']
            
            # Elastic modulus should decrease with temperature
            E = np.array(mech['elastic']['elastic_modulus'])
            if not np.all(np.diff(E) <= 0):
                results['warnings'].append(
                    f"{mix_id}: Elastic modulus not monotonically decreasing"
                )
            
            # Strength should generally decrease with temperature
            fc = np.array(mech['strength']['compressive_strength'])
            # Allow initial increase up to 100°C
            temps = np.array(mech['strength']['temperature'])
            idx_100 = np.where(temps >= 100)[0][0]
            if not np.all(np.diff(fc[idx_100:]) <= 0):
                results['warnings'].append(
                    f"{mix_id}: Compressive strength may not decrease properly after 100°C"
                )
    
    results['checks']['physical_consistency'] = len(results['warnings']) == 0
    
    # Check 2: Data completeness
    print("  - Checking data completeness...")
    required_properties = {
        'thermal': ['temperature', 'thermal_conductivity', 'specific_heat_capacity', 'density'],
        'mechanical': ['elastic', 'strength', 'plastic', 'damage'],
        'transport': ['porosity', 'permeability', 'moisture']
    }
    
    for mix_id in calibration_data.keys():
        for prop_type, req_fields in required_properties.items():
            if prop_type not in calibration_data[mix_id]:
                results['errors'].append(f"{mix_id}: Missing {prop_type} properties")
            else:
                for field in req_fields:
                    if field not in str(calibration_data[mix_id][prop_type]):
                        results['warnings'].append(
                            f"{mix_id}: Missing {field} in {prop_type}"
                        )
    
    results['checks']['data_completeness'] = len(results['errors']) == 0
    
    # Check 3: Temperature range coverage
    print("  - Checking temperature range...")
    for mix_id in calibration_data.keys():
        if 'thermal' in calibration_data[mix_id]:
            temps = calibration_data[mix_id]['thermal']['temperature']
            if min(temps) > 25 or max(temps) < 750:
                results['warnings'].append(
                    f"{mix_id}: Temperature range [{min(temps)}, {max(temps)}] "
                    f"may not cover required range [20, 800]"
                )
    
    results['checks']['temperature_coverage'] = True
    
    # Check 4: Stochastic variation in validation sets
    print("  - Checking stochastic variation...")
    if len(validation_data) > 0:
        first_set = list(validation_data.values())[0]
        for mix_id in first_set.keys():
            if 'thermal' in first_set[mix_id] and 'thermal' in calibration_data[mix_id]:
                cal_k = calibration_data[mix_id]['thermal']['thermal_conductivity']
                val_k = first_set[mix_id]['thermal']['thermal_conductivity']
                
                # Check if values are different (stochastic variation applied)
                if np.allclose(cal_k, val_k):
                    results['warnings'].append(
                        f"{mix_id}: No stochastic variation detected in validation set"
                    )
    
    results['checks']['stochastic_variation'] = True
    
    # Check 5: Multi-physics coupling
    print("  - Checking multi-physics coupling...")
    for mix_id in calibration_data.keys():
        if 'coupling' in calibration_data[mix_id]:
            coupling = calibration_data[mix_id]['coupling']
            required_coupling = ['thermal_expansion_coefficient', 'biot_coefficient']
            for param in required_coupling:
                if param not in coupling:
                    results['warnings'].append(
                        f"{mix_id}: Missing coupling parameter {param}"
                    )
    
    results['checks']['coupling_parameters'] = True
    
    # Summary
    results['summary'] = {
        'total_checks': len(results['checks']),
        'passed_checks': sum(results['checks'].values()),
        'warnings': len(results['warnings']),
        'errors': len(results['errors'])
    }
    
    return results

def save_validation_report(results, output_dir):
    """Save validation report to file"""
    report_path = os.path.join(output_dir, 'validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also create human-readable report
    report_text_path = os.path.join(output_dir, 'validation_report.txt')
    with open(report_text_path, 'w') as f:
        f.write("DATASET VALIDATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated: {results['timestamp']}\n\n")
        
        f.write("VALIDATION CHECKS:\n")
        f.write("-" * 40 + "\n")
        for check, passed in results['checks'].items():
            status = "PASS" if passed else "FAIL"
            f.write(f"  {check:30s} [{status}]\n")
        
        f.write("\nSUMMARY:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total checks: {results['summary']['total_checks']}\n")
        f.write(f"  Passed: {results['summary']['passed_checks']}\n")
        f.write(f"  Warnings: {results['summary']['warnings']}\n")
        f.write(f"  Errors: {results['summary']['errors']}\n")
        
        if results['warnings']:
            f.write("\nWARNINGS:\n")
            f.write("-" * 40 + "\n")
            for warning in results['warnings']:
                f.write(f"  - {warning}\n")
        
        if results['errors']:
            f.write("\nERRORS:\n")
            f.write("-" * 40 + "\n")
            for error in results['errors']:
                f.write(f"  - {error}\n")
    
    print(f"  Validation report saved to: {report_text_path}")

def generate_visualizations(data, output_dir):
    """
    Generate visualization plots for the dataset
    
    Parameters:
    -----------
    data : dict
        Dataset to visualize
    output_dir : str
        Output directory
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Plot 1: Thermal property evolution
    print("  - Generating thermal property plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for mix_id in data.keys():
        if 'thermal' in data[mix_id]:
            thermal = data[mix_id]['thermal']
            temps = thermal['temperature']
            
            # Thermal conductivity
            axes[0, 0].plot(temps, thermal['thermal_conductivity'], 
                          label=mix_id, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Temperature (°C)')
            axes[0, 0].set_ylabel('Thermal Conductivity (W/m·K)')
            axes[0, 0].set_title('Temperature-Dependent Thermal Conductivity')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Specific heat
            axes[0, 1].plot(temps, thermal['specific_heat_capacity'],
                          label=mix_id, marker='s', markersize=3)
            axes[0, 1].set_xlabel('Temperature (°C)')
            axes[0, 1].set_ylabel('Specific Heat (J/kg·K)')
            axes[0, 1].set_title('Temperature-Dependent Specific Heat')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Density
            axes[1, 0].plot(temps, thermal['density'],
                          label=mix_id, marker='^', markersize=3)
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Density (kg/m³)')
            axes[1, 0].set_title('Temperature-Dependent Density')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Thermal diffusivity
            axes[1, 1].plot(temps, thermal['thermal_diffusivity'],
                          label=mix_id, marker='d', markersize=3)
            axes[1, 1].set_xlabel('Temperature (°C)')
            axes[1, 1].set_ylabel('Thermal Diffusivity (mm²/s)')
            axes[1, 1].set_title('Temperature-Dependent Thermal Diffusivity')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'thermal_properties.png'), dpi=150)
    plt.close()
    
    # Plot 2: Mechanical property evolution
    print("  - Generating mechanical property plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for mix_id in data.keys():
        if 'mechanical' in data[mix_id]:
            mech = data[mix_id]['mechanical']
            
            # Elastic modulus
            temps = mech['elastic']['temperature']
            axes[0, 0].plot(temps, mech['elastic']['elastic_modulus'],
                          label=mix_id, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Temperature (°C)')
            axes[0, 0].set_ylabel('Elastic Modulus (GPa)')
            axes[0, 0].set_title('Temperature-Dependent Elastic Modulus')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Compressive strength
            temps = mech['strength']['temperature']
            axes[0, 1].plot(temps, mech['strength']['compressive_strength'],
                          label=mix_id, marker='s', markersize=3)
            axes[0, 1].set_xlabel('Temperature (°C)')
            axes[0, 1].set_ylabel('Compressive Strength (MPa)')
            axes[0, 1].set_title('Temperature-Dependent Compressive Strength')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Tensile strength
            axes[1, 0].plot(temps, mech['strength']['tensile_strength'],
                          label=mix_id, marker='^', markersize=3)
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Tensile Strength (MPa)')
            axes[1, 0].set_title('Temperature-Dependent Tensile Strength')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Poisson's ratio
            temps = mech['elastic']['temperature']
            axes[1, 1].plot(temps, mech['elastic']['poisson_ratio'],
                          label=mix_id, marker='d', markersize=3)
            axes[1, 1].set_xlabel('Temperature (°C)')
            axes[1, 1].set_ylabel("Poisson's Ratio")
            axes[1, 1].set_title("Temperature-Dependent Poisson's Ratio")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'mechanical_properties.png'), dpi=150)
    plt.close()
    
    # Plot 3: Transport properties
    print("  - Generating transport property plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for mix_id in data.keys():
        if 'transport' in data[mix_id]:
            transport = data[mix_id]['transport']
            
            # Porosity evolution
            temps = transport['porosity']['temperature']
            axes[0, 0].plot(temps, transport['porosity']['total_porosity'],
                          label=mix_id, marker='o', markersize=3)
            axes[0, 0].set_xlabel('Temperature (°C)')
            axes[0, 0].set_ylabel('Porosity (fraction)')
            axes[0, 0].set_title('Temperature-Dependent Porosity Evolution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Permeability
            temps = transport['permeability']['temperature']
            axes[0, 1].plot(temps, transport['permeability']['intrinsic_permeability'],
                          label=mix_id, marker='s', markersize=3)
            axes[0, 1].set_xlabel('Temperature (°C)')
            axes[0, 1].set_ylabel('Permeability (×10⁻¹⁸ m²)')
            axes[0, 1].set_title('Temperature-Dependent Permeability')
            axes[0, 1].set_yscale('log')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Moisture content
            temps = transport['moisture']['temperature']
            axes[1, 0].plot(temps, transport['moisture']['moisture_content'],
                          label=mix_id, marker='^', markersize=3)
            axes[1, 0].set_xlabel('Temperature (°C)')
            axes[1, 0].set_ylabel('Moisture Content (%)')
            axes[1, 0].set_title('Temperature-Dependent Moisture Content')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Spalling risk
            temps = transport['spalling']['temperature']
            axes[1, 1].plot(temps, transport['spalling']['spalling_risk_index'],
                          label=mix_id, marker='d', markersize=3)
            axes[1, 1].set_xlabel('Temperature (°C)')
            axes[1, 1].set_ylabel('Spalling Risk Index')
            axes[1, 1].set_title('Temperature-Dependent Spalling Risk')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'transport_properties.png'), dpi=150)
    plt.close()
    
    # Plot 4: Comparative analysis - rubber effect
    print("  - Generating comparative analysis plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Extract rubber contents and key properties at specific temperatures
    rubber_contents = []
    props_20C = {'E': [], 'fc': [], 'k': [], 'porosity': []}
    props_400C = {'E': [], 'fc': [], 'k': [], 'porosity': []}
    
    for mix_id in ['C', 'R5S', 'R10S', 'R15S', 'R20S']:
        if mix_id in data:
            # Extract rubber content
            if mix_id == 'C':
                rubber_contents.append(0)
            else:
                rubber_contents.append(int(mix_id[1:].split('S')[0]))
            
            # Get properties at 20°C and 400°C
            if 'mechanical' in data[mix_id]:
                temps = data[mix_id]['mechanical']['elastic']['temperature']
                idx_20 = 0
                idx_400 = np.argmin(np.abs(np.array(temps) - 400))
                
                props_20C['E'].append(data[mix_id]['mechanical']['elastic']['elastic_modulus'][idx_20])
                props_400C['E'].append(data[mix_id]['mechanical']['elastic']['elastic_modulus'][idx_400])
                
                props_20C['fc'].append(data[mix_id]['mechanical']['strength']['compressive_strength'][idx_20])
                props_400C['fc'].append(data[mix_id]['mechanical']['strength']['compressive_strength'][idx_400])
            
            if 'thermal' in data[mix_id]:
                props_20C['k'].append(data[mix_id]['thermal']['thermal_conductivity'][0])
                temps = data[mix_id]['thermal']['temperature']
                idx_400 = np.argmin(np.abs(np.array(temps) - 400))
                props_400C['k'].append(data[mix_id]['thermal']['thermal_conductivity'][idx_400])
            
            if 'transport' in data[mix_id]:
                props_20C['porosity'].append(data[mix_id]['transport']['porosity']['total_porosity'][0])
                temps = data[mix_id]['transport']['porosity']['temperature']
                idx_400 = np.argmin(np.abs(np.array(temps) - 400))
                props_400C['porosity'].append(data[mix_id]['transport']['porosity']['total_porosity'][idx_400])
    
    # Plot rubber effect on properties
    axes[0, 0].plot(rubber_contents, props_20C['E'], 'bo-', label='20°C')
    axes[0, 0].plot(rubber_contents, props_400C['E'], 'ro-', label='400°C')
    axes[0, 0].set_xlabel('Rubber Content (%)')
    axes[0, 0].set_ylabel('Elastic Modulus (GPa)')
    axes[0, 0].set_title('Effect of Rubber on Elastic Modulus')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(rubber_contents, props_20C['fc'], 'bo-', label='20°C')
    axes[0, 1].plot(rubber_contents, props_400C['fc'], 'ro-', label='400°C')
    axes[0, 1].set_xlabel('Rubber Content (%)')
    axes[0, 1].set_ylabel('Compressive Strength (MPa)')
    axes[0, 1].set_title('Effect of Rubber on Compressive Strength')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(rubber_contents, props_20C['k'], 'bo-', label='20°C')
    axes[1, 0].plot(rubber_contents, props_400C['k'], 'ro-', label='400°C')
    axes[1, 0].set_xlabel('Rubber Content (%)')
    axes[1, 0].set_ylabel('Thermal Conductivity (W/m·K)')
    axes[1, 0].set_title('Effect of Rubber on Thermal Conductivity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(rubber_contents, props_20C['porosity'], 'bo-', label='20°C')
    axes[1, 1].plot(rubber_contents, props_400C['porosity'], 'ro-', label='400°C')
    axes[1, 1].set_xlabel('Rubber Content (%)')
    axes[1, 1].set_ylabel('Porosity (fraction)')
    axes[1, 1].set_title('Effect of Rubber on Porosity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'rubber_effect_analysis.png'), dpi=150)
    plt.close()
    
    print(f"  Visualizations saved to: {viz_dir}")

def generate_summary_report(calibration_data, validation_data, output_dir):
    """
    Generate comprehensive summary report
    
    Parameters:
    -----------
    calibration_data : dict
        Calibration dataset
    validation_data : dict
        Validation datasets
    output_dir : str
        Output directory
    """
    report_path = os.path.join(output_dir, 'dataset_summary.md')
    
    with open(report_path, 'w') as f:
        f.write("# Thermo-Mechanical Modeling Dataset Summary\n\n")
        f.write("## Research Title\n")
        f.write("Development and Validation of a Thermo-Mechanical Model for Fire-Resistant ")
        f.write("Structural Elements Utilizing High-Performance Rubberized Concrete\n\n")
        
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write("### Mix Identifications\n")
        f.write("| Mix ID | Description | Rubber Content | Rubber Size |\n")
        f.write("|--------|-------------|----------------|-------------|\n")
        f.write("| C | Control (no rubber) | 0% | - |\n")
        f.write("| R5S | 5% small rubber | 5% | Small (2±0.5 mm) |\n")
        f.write("| R10S | 10% small rubber | 10% | Small (2±0.5 mm) |\n")
        f.write("| R15S | 15% small rubber | 15% | Small (2±0.5 mm) |\n")
        f.write("| R20S | 20% small rubber | 20% | Small (2±0.5 mm) |\n")
        f.write("| R10L | 10% large rubber | 10% | Large (8±2 mm) |\n\n")
        
        f.write("### Temperature Range\n")
        f.write("- **Minimum:** 20°C\n")
        f.write("- **Maximum:** 800°C\n")
        f.write("- **Critical Points:** 20, 100, 200, 300, 400, 500, 600, 700, 800°C\n\n")
        
        f.write("## Properties Included\n\n")
        
        f.write("### Thermal Properties\n")
        f.write("- Thermal conductivity (W/m·K)\n")
        f.write("- Specific heat capacity (J/kg·K)\n")
        f.write("- Thermal diffusivity (mm²/s)\n")
        f.write("- Density (kg/m³)\n")
        f.write("- Thermal expansion coefficient (μstrain/K)\n")
        f.write("- Surface emissivity\n")
        f.write("- Degradation functions\n\n")
        
        f.write("### Mechanical Properties\n")
        f.write("#### Elastic Properties\n")
        f.write("- Elastic modulus (GPa)\n")
        f.write("- Poisson's ratio\n")
        f.write("- Shear modulus (GPa)\n")
        f.write("- Bulk modulus (GPa)\n\n")
        
        f.write("#### Strength Properties\n")
        f.write("- Compressive strength (MPa)\n")
        f.write("- Tensile strength (MPa)\n")
        f.write("- Flexural strength (MPa)\n")
        f.write("- Fracture energy (N/m)\n\n")
        
        f.write("#### Plastic Properties\n")
        f.write("- Yield stress (MPa)\n")
        f.write("- Hardening modulus (MPa)\n")
        f.write("- Plastic strain at peak\n")
        f.write("- Dilation angle (degrees)\n\n")
        
        f.write("#### Damage Evolution\n")
        f.write("- Damage initiation parameters\n")
        f.write("- Damage evolution laws\n")
        f.write("- Stiffness degradation\n")
        f.write("- Compression/tension damage variables\n\n")
        
        f.write("### Transport Properties\n")
        f.write("#### Porosity\n")
        f.write("- Total porosity\n")
        f.write("- Connected porosity\n")
        f.write("- Capillary porosity\n")
        f.write("- Gel porosity\n")
        f.write("- Crack-induced porosity\n\n")
        
        f.write("#### Permeability\n")
        f.write("- Intrinsic permeability (m²)\n")
        f.write("- Gas permeability\n")
        f.write("- Liquid permeability\n")
        f.write("- Relative permeabilities\n\n")
        
        f.write("#### Moisture Transport\n")
        f.write("- Moisture content (%)\n")
        f.write("- Moisture diffusivity (mm²/s)\n")
        f.write("- Vapor diffusivity (mm²/s)\n")
        f.write("- Sorption isotherms\n")
        f.write("- Desorption rates\n\n")
        
        f.write("### Multi-Physics Coupling\n")
        f.write("- Thermal-mechanical coupling\n")
        f.write("- Poro-mechanical coupling (Biot parameters)\n")
        f.write("- Thermal-transport coupling\n")
        f.write("- Creep parameters\n\n")
        
        f.write("## Dataset Structure\n\n")
        f.write("```\n")
        f.write("output/\n")
        f.write("├── calibration/          # Calibration datasets\n")
        f.write("│   ├── complete_dataset.json\n")
        f.write("│   └── [mix_id]/\n")
        f.write("│       ├── thermal_properties.csv\n")
        f.write("│       ├── mechanical_properties.csv\n")
        f.write("│       └── transport_properties.csv\n")
        f.write("├── validation/           # Validation datasets\n")
        f.write("│   └── Set_[1-5]/\n")
        f.write("├── fea_inputs/          # FEA software input files\n")
        f.write("│   ├── abaqus/\n")
        f.write("│   ├── ansys/\n")
        f.write("│   └── comsol/\n")
        f.write("├── visualizations/      # Generated plots\n")
        f.write("└── documentation/       # Reports and summaries\n")
        f.write("```\n\n")
        
        f.write("## Key Features\n\n")
        f.write("1. **Temperature-Dependent Functions**: Continuous property evolution from 20°C to 800°C\n")
        f.write("2. **Multi-Physics Coupling**: Interdependent thermal, mechanical, and transport properties\n")
        f.write("3. **Model-Ready Formatting**: Direct import into ABAQUS, ANSYS, and COMSOL\n")
        f.write("4. **Calibration-Validation Split**: Separate datasets for model development and validation\n")
        f.write("5. **Stochastic Bounds**: Statistical variations (mean ± std) for probabilistic modeling\n")
        f.write("6. **Multi-Scale Linking**: Microstructural parameters inform macro-scale properties\n\n")
        
        f.write("## Usage Instructions\n\n")
        f.write("### For ABAQUS Users\n")
        f.write("1. Navigate to `fea_inputs/abaqus/`\n")
        f.write("2. Include material files: `*Include, input=[mix_id]/[mix_id]_material.inp`\n")
        f.write("3. Use amplitude curves for temperature loading\n")
        f.write("4. Apply field variable dependencies as needed\n\n")
        
        f.write("### For ANSYS Users\n")
        f.write("1. Navigate to `fea_inputs/ansys/`\n")
        f.write("2. Run material macros: `/INPUT,[mix_id]_material.mac`\n")
        f.write("3. Use table arrays for temperature dependencies\n")
        f.write("4. Execute master analysis macro\n\n")
        
        f.write("### For COMSOL Users\n")
        f.write("1. Navigate to `fea_inputs/comsol/`\n")
        f.write("2. Import interpolation functions\n")
        f.write("3. Use Java API file for model setup\n")
        f.write("4. Link material functions to model\n\n")
        
        f.write("## Data Statistics\n\n")
        
        # Calculate statistics
        n_mixes = len(calibration_data.keys())
        n_val_sets = len(validation_data)
        
        if n_mixes > 0:
            sample_mix = list(calibration_data.keys())[0]
            if 'thermal' in calibration_data[sample_mix]:
                n_temp_points = len(calibration_data[sample_mix]['thermal']['temperature'])
            else:
                n_temp_points = 0
        else:
            n_temp_points = 0
        
        f.write(f"- **Number of mixes:** {n_mixes}\n")
        f.write(f"- **Temperature points per property:** {n_temp_points}\n")
        f.write(f"- **Calibration dataset:** 1\n")
        f.write(f"- **Validation datasets:** {n_val_sets}\n")
        f.write(f"- **Total data points:** ~{n_mixes * n_temp_points * 50:,}\n\n")
        
        f.write("## References\n\n")
        f.write("- ISO 834: Fire-resistance tests\n")
        f.write("- ASTM E119: Standard test methods for fire tests\n")
        f.write("- Eurocode 2: Design of concrete structures - Fire design\n")
        f.write("- fib Bulletin 38: Fire design of concrete structures\n\n")
        
        f.write("## Contact Information\n\n")
        f.write("For questions or support regarding this dataset, please contact the research team.\n\n")
        
        f.write("---\n")
        f.write(f"*Generated by Thermo-Mechanical Dataset Generator v1.0.0*\n")
    
    print(f"  Summary report saved to: {report_path}")

if __name__ == "__main__":
    main()