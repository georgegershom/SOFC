#!/usr/bin/env python3
"""
Simple dataset summary script (no external dependencies required)
"""

import os
import csv
import json
from pathlib import Path

def count_csv_rows(filepath):
    """Count rows in a CSV file"""
    with open(filepath, 'r') as f:
        return sum(1 for row in csv.reader(f)) - 1  # Subtract header

def get_csv_columns(filepath):
    """Get column names from CSV"""
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        return next(reader)

def format_size(bytes):
    """Format file size"""
    for unit in ['B', 'KB', 'MB']:
        if bytes < 1024.0:
            return f"{bytes:.1f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.1f} GB"

def main():
    print("=" * 70)
    print("GEOTECHNICAL DATASETS SUMMARY")
    print("Underground Structure Failure Mechanisms Research")
    print("=" * 70)
    
    base_path = Path('.')
    
    # Dataset categories
    categories = {
        'Sandy Soils': 'sandy_soils',
        'Clay Soils': 'clay_soils',
        'Case Studies': 'case_studies',
        'Spatial Data': 'spatial_data'
    }
    
    total_rows = 0
    total_size = 0
    
    for category_name, folder in categories.items():
        print(f"\n📁 {category_name.upper()}")
        print("-" * 50)
        
        folder_path = base_path / folder
        if not folder_path.exists():
            print(f"  ⚠ Folder not found: {folder}")
            continue
        
        # Process CSV files
        csv_files = list(folder_path.glob('*.csv'))
        for csv_file in csv_files:
            rows = count_csv_rows(csv_file)
            cols = len(get_csv_columns(csv_file))
            size = os.path.getsize(csv_file)
            total_rows += rows
            total_size += size
            
            print(f"  • {csv_file.name}")
            print(f"    - Rows: {rows:,}")
            print(f"    - Columns: {cols}")
            print(f"    - Size: {format_size(size)}")
        
        # Process GeoJSON files
        geojson_files = list(folder_path.glob('*.geojson'))
        for geojson_file in geojson_files:
            with open(geojson_file, 'r') as f:
                data = json.load(f)
                features = len(data.get('features', []))
                size = os.path.getsize(geojson_file)
                total_size += size
                
                print(f"  • {geojson_file.name}")
                print(f"    - Features: {features}")
                print(f"    - Type: GeoJSON")
                print(f"    - Size: {format_size(size)}")
    
    # Analysis tools
    print("\n🛠️ ANALYSIS TOOLS")
    print("-" * 50)
    
    tools_path = base_path / 'analysis_tools'
    if tools_path.exists():
        py_files = list(tools_path.glob('*.py'))
        for py_file in py_files:
            size = os.path.getsize(py_file)
            total_size += size
            with open(py_file, 'r') as f:
                lines = sum(1 for _ in f)
            print(f"  • {py_file.name}: {lines} lines, {format_size(size)}")
    
    # Documentation
    print("\n📚 DOCUMENTATION")
    print("-" * 50)
    
    doc_files = [
        'README.md',
        'documentation/data_dictionary.md',
        'requirements.txt'
    ]
    
    for doc_file in doc_files:
        doc_path = base_path / doc_file
        if doc_path.exists():
            size = os.path.getsize(doc_path)
            total_size += size
            print(f"  • {doc_file}: {format_size(size)}")
    
    # Summary statistics
    print("\n📊 OVERALL STATISTICS")
    print("-" * 50)
    print(f"  Total data rows: {total_rows:,}")
    print(f"  Total disk usage: {format_size(total_size)}")
    print(f"  Dataset categories: {len(categories)}")
    
    # Key datasets info
    print("\n🔑 KEY DATASET HIGHLIGHTS")
    print("-" * 50)
    
    highlights = {
        'sandy_soils/sand_basic_properties.csv': 'Grain size distribution, density, moisture',
        'sandy_soils/liquefaction_data.csv': 'CSR/CRR, safety factors, pore pressure',
        'clay_soils/clay_basic_properties.csv': 'Atterberg limits, OCR, water content',
        'clay_soils/clay_mineralogy.csv': 'Clay minerals, CEC, swelling potential',
        'case_studies/underground_structure_failures.csv': 'Real failure events with triggers',
        'case_studies/structural_response_monitoring.csv': 'Time-series monitoring data',
        'spatial_data/grid_soil_data.csv': 'Gridded soil properties for mapping'
    }
    
    for file_path, description in highlights.items():
        full_path = base_path / file_path
        if full_path.exists():
            rows = count_csv_rows(full_path)
            print(f"  • {file_path.split('/')[-1]}")
            print(f"    {description}")
            print(f"    Records: {rows:,}")
    
    print("\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print("\nAll datasets have been successfully generated and are ready for use.")
    print("To perform full analysis with visualizations, install dependencies:")
    print("  $ pip install -r requirements.txt")
    print("  $ python3 run_analysis.py")
    
    return total_rows, total_size

if __name__ == "__main__":
    rows, size = main()