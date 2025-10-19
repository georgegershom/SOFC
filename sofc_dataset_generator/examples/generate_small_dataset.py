#!/usr/bin/env python3
"""
Example: Generate Small SOFC Dataset

This example demonstrates how to generate a small synthetic dataset
for testing and validation purposes.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from generate_dataset import SOFCDatasetGenerator


def main():
    """Generate a small dataset for testing"""
    print("SOFC Synthetic Dataset Generator - Small Dataset Example")
    print("=" * 60)
    
    # Create dataset generator
    generator = SOFCDatasetGenerator(
        output_dir='./small_dataset',
        random_seed=42
    )
    
    # Generate small dataset
    print("\nGenerating small dataset (10 samples)...")
    generator.generate_dataset(
        n_samples=10,
        strategy='lhs',
        use_creep=True,
        save_intermediate=True
    )
    
    print("\nSmall dataset generation completed!")
    print("Check the './small_dataset' directory for results.")


if __name__ == "__main__":
    main()