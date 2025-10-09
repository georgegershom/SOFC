#!/usr/bin/env python3
"""
Dataset Verification Script
Checks the integrity and completeness of the SOFC electrochemical dataset
"""

import csv
import os
from pathlib import Path

def verify_dataset():
    """Verify that all expected files exist and have proper structure"""
    
    print("SOFC Electrochemical Dataset Verification")
    print("="*50)
    
    base_path = Path('.')
    errors = []
    warnings = []
    
    # Expected file structure
    expected_files = {
        'IV Curves': [
            'iv_curves/iv_curves_700C.csv',
            'iv_curves/iv_curves_750C.csv',
            'iv_curves/iv_curves_800C.csv'
        ],
        'EIS Data': [
            'eis_data/eis_800C_0.5A_cm2.csv',
            'eis_data/eis_750C_0.5A_cm2.csv'
        ],
        'Overpotentials': [
            'overpotentials/overpotentials_700C.csv',
            'overpotentials/overpotentials_750C.csv',
            'overpotentials/overpotentials_800C.csv'
        ],
        'Scripts': [
            'scripts/visualize_iv_curves.py',
            'scripts/visualize_eis.py',
            'scripts/visualize_overpotentials.py',
            'scripts/data_loader.py',
            'scripts/generate_report.py',
            'scripts/requirements.txt'
        ],
        'Documentation': [
            'README.md',
            'docs/data_dictionary.md'
        ]
    }
    
    # Check each category
    for category, files in expected_files.items():
        print(f"\nChecking {category}...")
        print("-"*30)
        
        for file_path in files:
            full_path = base_path / file_path
            if full_path.exists():
                # Check if it's a CSV file
                if file_path.endswith('.csv'):
                    try:
                        with open(full_path, 'r') as f:
                            reader = csv.reader(f)
                            header = next(reader)
                            row_count = sum(1 for row in reader)
                            print(f"  ✓ {file_path} ({row_count} rows, {len(header)} columns)")
                            
                            # Verify minimum rows
                            if row_count < 10:
                                warnings.append(f"{file_path} has only {row_count} rows")
                    except Exception as e:
                        errors.append(f"Error reading {file_path}: {e}")
                        print(f"  ✗ {file_path} - Error reading file")
                else:
                    # Just check file size for non-CSV files
                    size = full_path.stat().st_size
                    print(f"  ✓ {file_path} ({size:,} bytes)")
            else:
                errors.append(f"Missing file: {file_path}")
                print(f"  ✗ {file_path} - MISSING")
    
    # Summary
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    if errors:
        print("\n❌ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\n✓ All required files present")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    # Data statistics
    print("\n📊 DATASET STATISTICS:")
    print("-"*30)
    
    total_csv_files = 0
    total_rows = 0
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.csv'):
                total_csv_files += 1
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r') as f:
                        rows = sum(1 for line in f) - 1  # Subtract header
                        total_rows += rows
                except:
                    pass
    
    print(f"  Total CSV files: {total_csv_files}")
    print(f"  Total data rows: {total_rows}")
    
    # Check data ranges for IV curves
    print("\n📈 IV CURVE RANGES:")
    print("-"*30)
    
    for temp in ['700C', '750C', '800C']:
        file_path = base_path / f'iv_curves/iv_curves_{temp}.csv'
        if file_path.exists():
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    min_v = float(rows[-1]['Voltage_V'])
                    max_v = float(rows[0]['Voltage_V'])
                    max_i = float(rows[-1]['Current_Density_A_cm2'])
                    max_p = max(float(r['Power_Density_W_cm2']) for r in rows)
                    print(f"  {temp}: V={min_v:.3f}-{max_v:.3f}V, I=0-{max_i:.3f}A/cm², P_max={max_p:.3f}W/cm²")
    
    print("\n✅ Dataset verification complete!")
    
    return len(errors) == 0

if __name__ == "__main__":
    success = verify_dataset()
    exit(0 if success else 1)