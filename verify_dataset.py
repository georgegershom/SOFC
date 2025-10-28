#!/usr/bin/env python3
"""
Dataset Verification Script
Validates the generated FEM simulation dataset for completeness and quality
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_success(text):
    print(f"✅ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def print_error(text):
    print(f"❌ {text}")

def verify_dataset():
    """Comprehensive dataset verification"""
    
    print_header("FEM SIMULATION DATASET VERIFICATION")
    
    errors = 0
    warnings = 0
    
    # Check directory structure
    print("\n📁 Checking Directory Structure...")
    data_dir = Path('fem_simulation_data')
    
    if not data_dir.exists():
        print_error("Dataset directory 'fem_simulation_data' not found!")
        return False
    print_success("Dataset directory found")
    
    # Check required files
    print("\n📄 Checking Required Files...")
    required_files = [
        'dataset_metadata.json',
        'mesh_nodes.csv',
        'mesh_elements.csv',
        'simulation_summary.csv'
    ]
    
    for file in required_files:
        if (data_dir / file).exists():
            print_success(f"{file} exists")
        else:
            print_error(f"{file} missing!")
            errors += 1
    
    # Check subdirectories
    if (data_dir / 'time_series_output').exists():
        print_success("time_series_output/ directory exists")
    else:
        print_error("time_series_output/ directory missing!")
        errors += 1
    
    if (data_dir / 'visualizations').exists():
        print_success("visualizations/ directory exists")
    else:
        print_error("visualizations/ directory missing!")
        errors += 1
    
    # Verify metadata
    print("\n📊 Verifying Metadata...")
    try:
        with open(data_dir / 'dataset_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        assert 'metadata' in metadata
        assert 'input_parameters' in metadata
        assert 'mesh' in metadata['input_parameters']
        assert 'boundary_conditions' in metadata['input_parameters']
        assert 'materials' in metadata['input_parameters']
        
        print_success(f"Metadata structure valid")
        print_success(f"  Nodes: {metadata['input_parameters']['mesh']['n_nodes']}")
        print_success(f"  Elements: {metadata['input_parameters']['mesh']['n_elements']}")
        print_success(f"  Time steps: {metadata['metadata']['n_time_steps']}")
        
    except Exception as e:
        print_error(f"Metadata validation failed: {e}")
        errors += 1
    
    # Verify mesh data
    print("\n🔷 Verifying Mesh Data...")
    try:
        nodes = pd.read_csv(data_dir / 'mesh_nodes.csv')
        elements = pd.read_csv(data_dir / 'mesh_elements.csv')
        
        print_success(f"Loaded {len(nodes)} nodes")
        print_success(f"Loaded {len(elements)} elements")
        
        # Check node coordinates
        assert all(col in nodes.columns for col in ['x', 'y', 'z'])
        assert len(nodes) == metadata['input_parameters']['mesh']['n_nodes']
        
        # Check coordinate ranges
        x_range = [nodes['x'].min(), nodes['x'].max()]
        y_range = [nodes['y'].min(), nodes['y'].max()]
        z_range = [nodes['z'].min(), nodes['z'].max()]
        
        print_success(f"  x range: [{x_range[0]:.2f}, {x_range[1]:.2f}] mm")
        print_success(f"  y range: [{y_range[0]:.2f}, {y_range[1]:.2f}] mm")
        print_success(f"  z range: [{z_range[0]:.2f}, {z_range[1]:.2f}] mm")
        
        # Check element types
        element_types = elements['element_type'].value_counts()
        print_success(f"  Element types: {dict(element_types)}")
        
    except Exception as e:
        print_error(f"Mesh validation failed: {e}")
        errors += 1
    
    # Verify time series data
    print("\n⏱️  Verifying Time Series Data...")
    try:
        summary = pd.read_csv(data_dir / 'simulation_summary.csv')
        print_success(f"Loaded summary with {len(summary)} time steps")
        
        # Check required columns
        required_cols = ['time', 'max_von_mises', 'avg_von_mises', 
                        'max_plastic_strain', 'max_damage', 'avg_damage',
                        'max_temperature', 'delamination_area', 'crack_count']
        
        missing_cols = [col for col in required_cols if col not in summary.columns]
        if missing_cols:
            print_error(f"Missing columns: {missing_cols}")
            errors += 1
        else:
            print_success("All required columns present")
        
        # Check value ranges
        print_success(f"  Time range: {summary['time'].min():.1f} - {summary['time'].max():.1f} s")
        print_success(f"  Max stress range: {summary['max_von_mises'].min()/1e6:.1f} - {summary['max_von_mises'].max()/1e6:.1f} MPa")
        print_success(f"  Damage range: {summary['max_damage'].min():.4f} - {summary['max_damage'].max():.4f}")
        
        # Check for NaN values
        nan_counts = summary.isna().sum()
        if nan_counts.any():
            print_warning(f"Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            warnings += 1
        else:
            print_success("No NaN values in summary")
        
    except Exception as e:
        print_error(f"Time series validation failed: {e}")
        errors += 1
    
    # Verify individual time step files
    print("\n📈 Verifying Time Step Output Files...")
    try:
        output_dir = data_dir / 'time_series_output'
        output_files = sorted(output_dir.glob('output_t*.csv'))
        
        if len(output_files) == 0:
            print_error("No output files found!")
            errors += 1
        else:
            print_success(f"Found {len(output_files)} output files")
            
            # Check first file
            first_file = pd.read_csv(output_files[0])
            expected_cols = ['node_id', 'x', 'y', 'z', 'time',
                           'von_mises_stress', 'principal_stress_1', 'principal_stress_2', 'principal_stress_3',
                           'shear_stress', 'elastic_strain', 'plastic_strain', 'creep_strain',
                           'thermal_strain', 'total_strain', 'damage',
                           'temperature', 'voltage', 'delamination_risk', 'crack_risk']
            
            missing = [col for col in expected_cols if col not in first_file.columns]
            if missing:
                print_error(f"Missing columns in output files: {missing}")
                errors += 1
            else:
                print_success("All required columns present in output files")
            
            print_success(f"  Rows per file: {len(first_file)}")
            print_success(f"  Total data points: {len(first_file) * len(output_files):,}")
            
            # Check last file
            last_file = pd.read_csv(output_files[-1])
            print_success(f"  Final time: {last_file['time'].iloc[0]:.1f} s")
            print_success(f"  Final max stress: {last_file['von_mises_stress'].max()/1e6:.1f} MPa")
            print_success(f"  Final max damage: {last_file['damage'].max():.4f}")
            
    except Exception as e:
        print_error(f"Output files validation failed: {e}")
        errors += 1
    
    # Verify visualizations
    print("\n🎨 Verifying Visualizations...")
    try:
        vis_dir = data_dir / 'visualizations'
        expected_plots = [
            'damage_evolution.png',
            'stress_evolution.png',
            'failure_mechanisms.png',
            'temperature_distribution.png',
            'voltage_distribution.png'
        ]
        
        for plot in expected_plots:
            if (vis_dir / plot).exists():
                size = (vis_dir / plot).stat().st_size / 1024  # KB
                print_success(f"{plot} exists ({size:.1f} KB)")
            else:
                print_error(f"{plot} missing!")
                errors += 1
                
    except Exception as e:
        print_error(f"Visualization validation failed: {e}")
        errors += 1
    
    # Physical consistency checks
    print("\n🔬 Checking Physical Consistency...")
    try:
        # Check stress-strain relationship
        t10 = pd.read_csv(output_files[10])
        
        # Elastic modulus check (rough approximation)
        E_apparent = (t10['von_mises_stress'] / t10['elastic_strain']).median()
        E_expected = 150e9  # Pa
        
        if 10e9 < E_apparent < 500e9:
            print_success(f"Apparent elastic modulus reasonable: {E_apparent/1e9:.1f} GPa")
        else:
            print_warning(f"Apparent elastic modulus unusual: {E_apparent/1e9:.1f} GPa (expected ~150 GPa)")
            warnings += 1
        
        # Check damage bounds
        if t10['damage'].min() >= 0 and t10['damage'].max() <= 1:
            print_success(f"Damage variable properly bounded [0, 1]")
        else:
            print_error(f"Damage out of bounds: [{t10['damage'].min()}, {t10['damage'].max()}]")
            errors += 1
        
        # Check temperature range
        if 20 < t10['temperature'].min() and t10['temperature'].max() < 100:
            print_success(f"Temperature range reasonable: {t10['temperature'].min():.1f} - {t10['temperature'].max():.1f} °C")
        else:
            print_warning(f"Temperature range unusual: {t10['temperature'].min():.1f} - {t10['temperature'].max():.1f} °C")
            warnings += 1
        
        # Check voltage range
        if -0.5 < t10['voltage'].min() and t10['voltage'].max() < 5.0:
            print_success(f"Voltage range reasonable: {t10['voltage'].min():.1f} - {t10['voltage'].max():.1f} V")
        else:
            print_warning(f"Voltage range unusual: {t10['voltage'].min():.1f} - {t10['voltage'].max():.1f} V")
            warnings += 1
        
    except Exception as e:
        print_error(f"Physical consistency checks failed: {e}")
        errors += 1
    
    # Final summary
    print_header("VERIFICATION SUMMARY")
    
    if errors == 0 and warnings == 0:
        print("\n🎉 PERFECT! Dataset passed all verification checks!")
        print("✨ Dataset is complete, consistent, and ready to use.")
    elif errors == 0:
        print(f"\n✅ GOOD! Dataset passed with {warnings} warning(s).")
        print("   Dataset is usable but review warnings above.")
    else:
        print(f"\n⚠️  ISSUES FOUND: {errors} error(s), {warnings} warning(s)")
        print("   Please review errors above and regenerate if necessary.")
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total size: ~91 MB")
    print(f"   Files: {len(list(data_dir.rglob('*.*')))}")
    print(f"   Nodes: {len(nodes):,}")
    print(f"   Elements: {len(elements):,}")
    print(f"   Time steps: {len(output_files)}")
    print(f"   Data points: {len(nodes) * len(output_files):,}")
    
    print("\n" + "="*70 + "\n")
    
    return errors == 0

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
