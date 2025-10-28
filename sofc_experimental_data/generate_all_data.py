#!/usr/bin/env python3
"""
Main script to generate all SOFC experimental datasets
Run this script to create complete synthetic experimental data
"""

import sys
import os
import time
from datetime import datetime

# Import data generators
from generate_dic_data import DICDataGenerator
from generate_xrd_data import XRDDataGenerator
from generate_postmortem_data import PostMortemDataGenerator

def print_header():
    """Print script header"""
    print("\n" + "="*70)
    print(" SOFC EXPERIMENTAL DATA GENERATOR")
    print(" Generating Synthetic Datasets for DIC, XRD, and Post-Mortem Analysis")
    print("="*70)
    print(f" Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

def print_section(title):
    """Print section header"""
    print("\n" + "-"*60)
    print(f" {title}")
    print("-"*60)

def generate_all_datasets():
    """Generate all experimental datasets"""
    
    print_header()
    
    # Track timing
    start_time = time.time()
    
    try:
        # Generate DIC data
        print_section("GENERATING DIGITAL IMAGE CORRELATION (DIC) DATA")
        dic_gen = DICDataGenerator()
        dic_gen.run_all()
        print("✓ DIC data generation completed successfully")
        
        # Generate XRD data
        print_section("GENERATING SYNCHROTRON XRD DATA")
        xrd_gen = XRDDataGenerator()
        xrd_gen.run_all()
        print("✓ XRD data generation completed successfully")
        
        # Generate post-mortem data
        print_section("GENERATING POST-MORTEM ANALYSIS DATA")
        pm_gen = PostMortemDataGenerator()
        pm_gen.run_all()
        print("✓ Post-mortem data generation completed successfully")
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        
        # Count generated files
        total_files = 0
        total_size = 0
        for root, dirs, files in os.walk('.'):
            if '.git' not in root and '__pycache__' not in root:
                total_files += len(files)
                for file in files:
                    filepath = os.path.join(root, file)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        
        # Print summary
        print_section("GENERATION COMPLETE")
        print(f"✓ Total files generated: {total_files}")
        print(f"✓ Total data size: {total_size / (1024*1024):.2f} MB")
        print(f"✓ Generation time: {elapsed_time:.2f} seconds")
        print(f"✓ Completion time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print("\n" + "="*70)
        print(" SUCCESS: All experimental datasets generated successfully!")
        print(" Run 'python visualize_data.py' to create visualizations")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate datasets")
        print(f"   Error details: {str(e)}")
        print("\nPlease check the error message and try again.")
        return False

def main():
    """Main execution function"""
    success = generate_all_datasets()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()