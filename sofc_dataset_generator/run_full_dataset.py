#!/usr/bin/env python3
"""
Generate Full SOFC Synthetic Dataset

This script generates the complete synthetic dataset with 500-1000+ samples
as specified in the research article requirements.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generate_dataset import SOFCDatasetGenerator


def load_config(config_path: str = 'config/dataset_config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Generate full SOFC dataset"""
    parser = argparse.ArgumentParser(description='Generate full SOFC synthetic dataset')
    parser.add_argument('--config', type=str, default='config/dataset_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--samples', type=int, default=None,
                       help='Override number of samples from config')
    parser.add_argument('--strategy', type=str, default=None,
                       help='Override sampling strategy from config')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Override output directory from config')
    parser.add_argument('--no_creep', action='store_true',
                       help='Disable creep effects')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.samples:
        config['dataset']['n_samples'] = args.samples
    if args.strategy:
        config['dataset']['sampling_strategy'] = args.strategy
    if args.output_dir:
        config['dataset']['output_dir'] = args.output_dir
    if args.no_creep:
        config['dataset']['use_creep'] = False
    
    # Print configuration
    print("SOFC Synthetic Dataset Generator - Full Dataset")
    print("=" * 60)
    print(f"Configuration loaded from: {args.config}")
    print(f"Number of samples: {config['dataset']['n_samples']}")
    print(f"Sampling strategy: {config['dataset']['sampling_strategy']}")
    print(f"Output directory: {config['dataset']['output_dir']}")
    print(f"Creep effects: {'Enabled' if config['dataset']['use_creep'] else 'Disabled'}")
    print(f"Random seed: {config['dataset']['random_seed']}")
    
    # Create output directory
    output_dir = config['dataset']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration to output directory
    config_file = os.path.join(output_dir, 'dataset_config.yaml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    print(f"Configuration saved to: {config_file}")
    
    # Create dataset generator
    generator = SOFCDatasetGenerator(
        output_dir=output_dir,
        random_seed=config['dataset']['random_seed']
    )
    
    # Generate dataset
    print(f"\nStarting dataset generation...")
    print(f"This may take several hours for large datasets.")
    print(f"Progress will be saved every 50 samples.")
    
    generator.generate_dataset(
        n_samples=config['dataset']['n_samples'],
        strategy=config['dataset']['sampling_strategy'],
        use_creep=config['dataset']['use_creep'],
        save_intermediate=config['dataset']['save_intermediate']
    )
    
    print(f"\nFull dataset generation completed!")
    print(f"Dataset saved to: {output_dir}")
    print(f"Check the following files:")
    print(f"  - sofc_dataset.h5 (main dataset)")
    print(f"  - parameters.csv (parameter matrix)")
    print(f"  - dataset_metadata.json (dataset statistics)")
    print(f"  - doe_matrix.csv (DOE matrix)")


if __name__ == "__main__":
    main()