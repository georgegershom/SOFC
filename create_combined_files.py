#!/usr/bin/env python3
"""
Helper Script: Create Combined CSV Files and ZIP Archive

This script reconstructs the complete CSV files and ZIP archive from the
individual per-second CSV files. Run this after cloning the repository
to get the full dataset in convenient formats.

Usage:
    python3 create_combined_files.py
"""

import os
import pandas as pd
import zipfile
from pathlib import Path

def combine_group_files(group_dir):
    """Combine per-second CSV files into a complete file"""
    group_name = os.path.basename(group_dir)
    print(f"\n[{group_name}]")
    
    # Find all per-second files
    second_files = sorted([f for f in os.listdir(group_dir) 
                          if f.startswith(f"{group_name}_second_") and f.endswith('.csv')])
    
    if not second_files:
        print(f"  ⚠ No per-second files found")
        return None
    
    print(f"  Found {len(second_files)} per-second files")
    
    # Read and combine all files
    dfs = []
    for filename in second_files:
        filepath = os.path.join(group_dir, filename)
        df = pd.read_csv(filepath)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Extract leak location from filename
    leak_location = second_files[0].split('_')[-1].replace('.csv', '')
    
    # Save combined file
    output_filename = f"{group_name}_complete_{leak_location}.csv"
    output_path = os.path.join(group_dir, output_filename)
    combined_df.to_csv(output_path, index=False, float_format='%.6f')
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ Created {output_filename} ({file_size_mb:.2f} MB)")
    
    return output_path


def create_zip_archive(base_dir):
    """Create ZIP archive with all CSV files and metadata"""
    print(f"\n📦 Creating ZIP archive...")
    
    zip_path = os.path.join(base_dir, 'acoustic_pressure_data_all_groups.zip')
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all CSV files
        for group_dir in ['Group_01', 'Group_02', 'Group_03', 'Group_04']:
            group_path = os.path.join(base_dir, group_dir)
            if os.path.exists(group_path):
                for filename in os.listdir(group_path):
                    if filename.endswith('.csv'):
                        filepath = os.path.join(group_path, filename)
                        arcname = os.path.join(group_dir, filename)
                        zipf.write(filepath, arcname)
        
        # Add metadata
        metadata_file = os.path.join(base_dir, 'experiment_metadata.json')
        if os.path.exists(metadata_file):
            zipf.write(metadata_file, 'experiment_metadata.json')
        
        # Add README
        readme_file = os.path.join(base_dir, 'README.md')
        if os.path.exists(readme_file):
            zipf.write(readme_file, 'README.md')
    
    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  ✓ Created acoustic_pressure_data_all_groups.zip ({zip_size_mb:.2f} MB)")
    
    return zip_path


def main():
    """Main execution"""
    print("=" * 70)
    print("CREATE COMBINED FILES AND ZIP ARCHIVE")
    print("=" * 70)
    
    base_dir = '/workspace/acoustic_pressure_dataset'
    
    if not os.path.exists(base_dir):
        print(f"\n❌ Error: Dataset directory not found: {base_dir}")
        return
    
    print(f"\n📁 Working directory: {base_dir}")
    
    # Combine files for each group
    print(f"\n🔄 Combining per-second files into complete datasets...")
    combined_files = []
    
    for group_name in ['Group_01', 'Group_02', 'Group_03', 'Group_04']:
        group_dir = os.path.join(base_dir, group_name)
        if os.path.exists(group_dir):
            combined_file = combine_group_files(group_dir)
            if combined_file:
                combined_files.append(combined_file)
    
    # Create ZIP archive
    zip_file = create_zip_archive(base_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\n✅ Created {len(combined_files)} complete CSV files")
    print(f"✅ Created ZIP archive")
    print(f"\n📂 All files available in: {base_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
