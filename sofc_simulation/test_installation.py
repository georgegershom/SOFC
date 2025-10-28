#!/usr/bin/env python
"""
Test script to verify SOFC simulation package installation
"""

import os
import sys
import json
from pathlib import Path

def test_directory_structure():
    """Check if all required directories exist"""
    print("Testing directory structure...")
    
    required_dirs = [
        'inp',
        'scripts',
        'post_processing',
        'materials',
        'docs'
    ]
    
    base_dir = Path(__file__).parent
    missing = []
    
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            missing.append(dir_name)
    
    return len(missing) == 0

def test_required_files():
    """Check if all required files exist"""
    print("\nTesting required files...")
    
    required_files = [
        'inp/sofc_main.inp',
        'scripts/create_sofc_model.py',
        'post_processing/damage_analysis.py',
        'post_processing/visualize_results.py',
        'materials/material_database.json',
        'run_simulation.py',
        'README.md'
    ]
    
    base_dir = Path(__file__).parent
    missing = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            size_kb = full_path.stat().st_size / 1024
            print(f"  ✓ {file_path} ({size_kb:.1f} KB)")
        else:
            print(f"  ✗ {file_path} (missing)")
            missing.append(file_path)
    
    return len(missing) == 0

def test_material_database():
    """Verify material database integrity"""
    print("\nTesting material database...")
    
    base_dir = Path(__file__).parent
    db_path = base_dir / 'materials' / 'material_database.json'
    
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
        
        # Check for required materials
        required_materials = ['Ni_YSZ', 'YSZ_8', 'LSM', 'Ferritic_Steel']
        for mat in required_materials:
            if mat in data['materials']:
                props = len(data['materials'][mat])
                print(f"  ✓ {mat}: {props} property groups")
            else:
                print(f"  ✗ {mat}: missing")
                return False
        
        # Check interfaces
        if 'interfaces' in data:
            print(f"  ✓ Interfaces: {len(data['interfaces'])} defined")
        else:
            print("  ✗ Interfaces: missing")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error reading database: {e}")
        return False

def test_python_syntax():
    """Check Python scripts for syntax errors"""
    print("\nTesting Python syntax...")
    
    python_files = [
        'run_simulation.py',
        'scripts/create_sofc_model.py',
        'post_processing/damage_analysis.py',
        'post_processing/visualize_results.py'
    ]
    
    base_dir = Path(__file__).parent
    errors = []
    
    for file_path in python_files:
        full_path = base_dir / file_path
        if not full_path.exists():
            continue
        
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            compile(code, str(full_path), 'exec')
            print(f"  ✓ {file_path}")
        except SyntaxError as e:
            print(f"  ✗ {file_path}: {e}")
            errors.append(file_path)
    
    return len(errors) == 0

def test_inp_file():
    """Verify INP file structure"""
    print("\nTesting Abaqus input file...")
    
    base_dir = Path(__file__).parent
    inp_path = base_dir / 'inp' / 'sofc_main.inp'
    
    if not inp_path.exists():
        print("  ✗ INP file not found")
        return False
    
    required_keywords = [
        '*Heading',
        '*Part',
        '*Material',
        '*Step',
        '*Boundary',
        '*Output'
    ]
    
    with open(inp_path, 'r') as f:
        content = f.read()
    
    missing = []
    for keyword in required_keywords:
        if keyword in content:
            count = content.count(keyword)
            print(f"  ✓ {keyword}: {count} occurrence(s)")
        else:
            print(f"  ✗ {keyword}: missing")
            missing.append(keyword)
    
    return len(missing) == 0

def check_abaqus():
    """Check if Abaqus is available"""
    print("\nChecking Abaqus installation...")
    
    import subprocess
    
    try:
        result = subprocess.run(['abaqus', 'information=version'],
                              capture_output=True, text=True,
                              timeout=5)
        if result.returncode == 0:
            print(f"  ✓ Abaqus found")
            return True
        else:
            print(f"  ⚠ Abaqus command failed")
            return False
    except FileNotFoundError:
        print("  ⚠ Abaqus not found in PATH")
        print("    Set up with: export PATH=/path/to/abaqus/bin:$PATH")
        return False
    except subprocess.TimeoutExpired:
        print("  ⚠ Abaqus command timed out")
        return False
    except Exception as e:
        print(f"  ⚠ Error checking Abaqus: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("SOFC SIMULATION PACKAGE - INSTALLATION TEST")
    print("="*60)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_required_files),
        ("Material Database", test_material_database),
        ("Python Syntax", test_python_syntax),
        ("INP File Structure", test_inp_file),
        ("Abaqus Installation", check_abaqus)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n✗ Error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for test_name, passed_test in results:
        status = "PASS" if passed_test else "FAIL"
        symbol = "✓" if passed_test else "✗"
        print(f"{symbol} {test_name:.<40} {status}")
    
    print("-"*60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Package is ready to use.")
        print("\nNext steps:")
        print("  1. Run a test simulation:")
        print("     python run_simulation.py --rate HR1")
        print("  2. Or generate input files only:")
        print("     python run_simulation.py --generate-only")
    else:
        print("\n⚠ Some tests failed. Please check the issues above.")
        print("\nYou can still:")
        print("  - Generate input files: python run_simulation.py --generate-only")
        print("  - Review documentation: cat README.md")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())