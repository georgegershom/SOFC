#!/usr/bin/env python3
"""
Quick generation script for synthetic synchrotron data (smaller dataset)
"""

from generate_synchrotron_data import SyntheticSynchrotronDataGenerator

# Create generator with smaller dimensions for quick demo
generator = SyntheticSynchrotronDataGenerator(
    output_dir="synchrotron_data",
    random_seed=42
)

# Use smaller dimensions for faster generation
generator.experiment["image_dimensions"] = [128, 128, 128]  # Reduced from 512
generator.experiment["num_time_steps"] = 6  # Reduced from 11

print("Generating smaller demo dataset...")
print(f"Dimensions: {generator.experiment['image_dimensions']}")
print(f"Time steps: {generator.experiment['num_time_steps']}")

# Generate all data
summary_file = generator.generate_all()

print("\n" + "="*70)
print("DEMO DATASET GENERATED SUCCESSFULLY!")
print("="*70)
print("\nTo generate full-size data, use:")
print("  python3 generate_synchrotron_data.py")
print("="*70)
