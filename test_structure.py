#!/usr/bin/env python3
"""
Test script to verify the dataset generator structure without dependencies
"""

import sys
import os

# Test imports
print("Testing dataset generator structure...")
print("=" * 60)

# Check if all module files exist
modules = [
    'thermo_mechanical_dataset/__init__.py',
    'thermo_mechanical_dataset/core.py',
    'thermo_mechanical_dataset/thermal_properties.py',
    'thermo_mechanical_dataset/mechanical_properties.py',
    'thermo_mechanical_dataset/transport_properties.py',
    'thermo_mechanical_dataset/fea_exporters.py'
]

all_exist = True
for module in modules:
    exists = os.path.exists(module)
    status = "✓" if exists else "✗"
    print(f"{status} {module}")
    if not exists:
        all_exist = False

print("=" * 60)

# Check main scripts
scripts = [
    'generate_dataset.py',
    'example_usage.py',
    'requirements.txt',
    'README.md'
]

print("\nMain files:")
for script in scripts:
    exists = os.path.exists(script)
    status = "✓" if exists else "✗"
    print(f"{status} {script}")

print("=" * 60)

# Try importing modules (without numpy dependencies)
try:
    # This will fail on numpy imports but shows the structure is correct
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import thermo_mechanical_dataset
    print("\n✓ Package structure is valid")
except ImportError as e:
    if 'numpy' in str(e):
        print(f"\n✓ Package structure is valid (numpy dependency expected: {e})")
    else:
        print(f"\n✗ Package import error: {e}")

print("\n" + "=" * 60)
print("DATASET GENERATOR STRUCTURE TEST COMPLETE")
print("=" * 60)
print("\nTo use the generator, install dependencies:")
print("  pip install -r requirements.txt")
print("\nThen run:")
print("  python3 generate_dataset.py --help")
print("=" * 60)