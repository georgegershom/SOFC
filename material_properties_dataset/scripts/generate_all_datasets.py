#!/usr/bin/env python3
"""
Master script to generate all material properties datasets
"""

import os
import sys
import subprocess

def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Generate all datasets"""
    
    print("🚀 Material Properties Dataset Generator")
    print("=" * 60)
    print("This will generate synthetic material properties data for:")
    print("  • Mechanical Properties")
    print("  • Creep Properties")
    print("  • Thermo-physical Properties")
    print("  • Electrochemical Properties")
    print("=" * 60)
    
    # List of scripts to run
    scripts = [
        'generate_mechanical_properties.py',
        'generate_creep_properties.py',
        'generate_thermophysical_properties.py',
        'generate_electrochemical_properties.py'
    ]
    
    success_count = 0
    
    for script in scripts:
        if run_script(script):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✅ Dataset generation complete!")
    print(f"   Successfully generated: {success_count}/{len(scripts)} datasets")
    print("=" * 60)
    
    # List generated files
    print("\n📁 Generated files:")
    directories = ['../mechanical', '../creep', '../thermophysical', '../electrochemical']
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n  {directory}:")
            for file in sorted(os.listdir(directory)):
                file_path = os.path.join(directory, file)
                file_size = os.path.getsize(file_path)
                if file_size > 1024 * 1024:
                    size_str = f"{file_size / (1024 * 1024):.1f} MB"
                elif file_size > 1024:
                    size_str = f"{file_size / 1024:.1f} KB"
                else:
                    size_str = f"{file_size} B"
                print(f"    • {file} ({size_str})")

if __name__ == "__main__":
    main()