#!/usr/bin/env python3
"""
Export and package the complete dataset for distribution
"""

import os
import zipfile
import json
from datetime import datetime

def create_dataset_package():
    """Create a comprehensive ZIP package of all datasets"""
    
    # Define the files to include
    files_to_include = []
    
    # Walk through the directory and collect all relevant files
    for root, dirs, files in os.walk('.'):
        # Skip hidden directories and __pycache__
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith(('.csv', '.json', '.md', '.txt', '.py')):
                file_path = os.path.join(root, file)
                files_to_include.append(file_path)
    
    # Create the ZIP file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"rubberized_concrete_fire_dataset_{timestamp}.zip"
    
    print(f"Creating dataset package: {zip_filename}")
    print("-" * 50)
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in files_to_include:
            # Add file to zip with relative path
            arcname = file_path[2:] if file_path.startswith('./') else file_path
            zipf.write(file_path, arcname)
            print(f"Added: {arcname}")
    
    # Get file size
    file_size = os.path.getsize(zip_filename)
    file_size_mb = file_size / (1024 * 1024)
    
    print("-" * 50)
    print(f"✓ Dataset package created successfully!")
    print(f"  Filename: {zip_filename}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Files included: {len(files_to_include)}")
    
    # Create a manifest file
    manifest = {
        "package_name": zip_filename,
        "creation_date": datetime.now().isoformat(),
        "total_files": len(files_to_include),
        "size_bytes": file_size,
        "size_mb": round(file_size_mb, 2),
        "contents": {
            "csv_files": len([f for f in files_to_include if f.endswith('.csv')]),
            "json_files": len([f for f in files_to_include if f.endswith('.json')]),
            "python_scripts": len([f for f in files_to_include if f.endswith('.py')]),
            "documentation": len([f for f in files_to_include if f.endswith('.md')])
        },
        "datasets": {
            "mix_designs": "12 concrete mix designs with varying rubber content",
            "ambient_tests": "108 specimens tested at 7, 28, and 56 days",
            "high_temp_residual": "360 specimens tested after heating to 200-800°C",
            "in_situ_tests": "216 specimens tested at elevated temperatures",
            "spalling_data": "840 pore pressure and spalling measurements",
            "stress_strain": "3000 stress-strain data points at various temperatures"
        }
    }
    
    manifest_filename = f"manifest_{timestamp}.json"
    with open(manifest_filename, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  Manifest saved as: {manifest_filename}")
    
    return zip_filename, manifest

def generate_citation_file():
    """Generate a CITATION.cff file for the dataset"""
    
    citation = """cff-version: 1.2.0
message: "If you use this dataset, please cite it as below."
authors:
  - name: "Research Team"
    affiliation: "University/Institution"
title: "Comprehensive Experimental Dataset for Thermo-Mechanical Modeling of Fire-Resistant Rubberized Concrete"
version: 1.0.0
date-released: 2024-10-18
url: "https://github.com/[repository]"
doi: "10.xxxxx/xxxxx"
keywords:
  - rubberized concrete
  - fire resistance
  - thermo-mechanical properties
  - elevated temperature
  - spalling behavior
  - experimental dataset
license: "CC-BY-4.0"
"""
    
    with open("CITATION.cff", 'w') as f:
        f.write(citation)
    
    print("\n✓ Citation file (CITATION.cff) created")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RUBBERIZED CONCRETE FIRE RESISTANCE DATASET PACKAGER")
    print("="*60 + "\n")
    
    # Generate citation file
    generate_citation_file()
    
    # Create the dataset package
    zip_file, manifest = create_dataset_package()
    
    print("\n" + "="*60)
    print("PACKAGE READY FOR DISTRIBUTION")
    print("="*60)
    print(f"\nDataset is ready for download: {zip_file}")
    print("\nTo use this dataset:")
    print("1. Extract the ZIP file to your working directory")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run analysis: python scripts/data_analyzer.py")
    print("4. Generate visualizations: python scripts/data_visualizer.py")
    print("\nPlease cite this dataset using the information in CITATION.cff")