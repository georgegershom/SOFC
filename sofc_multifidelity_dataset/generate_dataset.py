#!/usr/bin/env python3
"""
Main script to generate the complete multi-fidelity SOFC dataset.

Usage:
    python generate_dataset.py --config config.yaml --output ./data
    python generate_dataset.py --fidelity low --samples 1000
    python generate_dataset.py --all --parallel
"""

import argparse
import yaml
import os
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.generators import (
    LowFidelityGenerator,
    MidFidelityGenerator, 
    HighFidelityGenerator,
    ExperimentalDataGenerator
)
from src.utils.data_utils import DatasetManager, save_config


def generate_low_fidelity(config: Dict, n_samples: Optional[int] = None, 
                         output_path: str = "./data") -> Dict:
    """Generate low-fidelity dataset."""
    print("\n" + "="*60)
    print("GENERATING LOW-FIDELITY DATASET")
    print("="*60)
    
    if n_samples is None:
        n_samples = config['dataset']['low_fidelity']['n_samples']
    
    generator = LowFidelityGenerator(config)
    start_time = time.time()
    
    report = generator.generate_dataset(n_samples, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per sample: {elapsed/n_samples:.3f} seconds")
    
    return report


def generate_mid_fidelity(config: Dict, n_samples: Optional[int] = None,
                         output_path: str = "./data") -> Dict:
    """Generate mid-fidelity dataset."""
    print("\n" + "="*60)
    print("GENERATING MID-FIDELITY DATASET")
    print("="*60)
    
    if n_samples is None:
        n_samples = config['dataset']['mid_fidelity']['n_samples']
    
    generator = MidFidelityGenerator(config)
    start_time = time.time()
    
    report = generator.generate_dataset(n_samples, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per sample: {elapsed/n_samples:.3f} seconds")
    
    return report


def generate_high_fidelity(config: Dict, n_samples: Optional[int] = None,
                          output_path: str = "./data") -> Dict:
    """Generate high-fidelity dataset."""
    print("\n" + "="*60)
    print("GENERATING HIGH-FIDELITY DATASET")
    print("="*60)
    
    if n_samples is None:
        n_samples = config['dataset']['high_fidelity']['n_samples']
    
    generator = HighFidelityGenerator(config)
    start_time = time.time()
    
    report = generator.generate_dataset(n_samples, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per sample: {elapsed/n_samples:.3f} seconds")
    
    return report


def generate_experimental(config: Dict, n_samples: Optional[int] = None,
                         output_path: str = "./data") -> Dict:
    """Generate experimental dataset."""
    print("\n" + "="*60)
    print("GENERATING EXPERIMENTAL DATASET")
    print("="*60)
    
    if n_samples is None:
        n_samples = config['dataset']['experimental']['n_samples']
    
    generator = ExperimentalDataGenerator(config)
    start_time = time.time()
    
    report = generator.generate_dataset(n_samples, output_path)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    print(f"Average time per sample: {elapsed/n_samples:.3f} seconds")
    
    return report


def generate_all_datasets(config: Dict, output_path: str = "./data",
                         parallel: bool = False) -> Dict:
    """Generate all fidelity levels."""
    print("\n" + "="*60)
    print("GENERATING COMPLETE MULTI-FIDELITY DATASET")
    print("="*60)
    
    reports = {}
    total_start = time.time()
    
    if parallel and mp.cpu_count() >= 4:
        print(f"\nUsing parallel generation with {mp.cpu_count()} cores")
        
        with mp.Pool(processes=4) as pool:
            # Submit jobs
            jobs = []
            jobs.append(pool.apply_async(generate_low_fidelity, (config, None, output_path)))
            jobs.append(pool.apply_async(generate_mid_fidelity, (config, None, output_path)))
            jobs.append(pool.apply_async(generate_high_fidelity, (config, None, output_path)))
            jobs.append(pool.apply_async(generate_experimental, (config, None, output_path)))
            
            # Collect results
            fidelities = ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']
            for fidelity, job in zip(fidelities, jobs):
                reports[fidelity] = job.get()
    else:
        # Sequential generation
        reports['low_fidelity'] = generate_low_fidelity(config, None, output_path)
        reports['mid_fidelity'] = generate_mid_fidelity(config, None, output_path)
        reports['high_fidelity'] = generate_high_fidelity(config, None, output_path)
        reports['experimental'] = generate_experimental(config, None, output_path)
    
    total_elapsed = time.time() - total_start
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    
    total_samples = sum(r['n_samples'] for r in reports.values())
    total_size = sum(r['file_size_mb'] for r in reports.values())
    
    print(f"\nSummary:")
    print(f"  Total samples generated: {total_samples:,}")
    print(f"  Total dataset size: {total_size:.1f} MB")
    print(f"  Total generation time: {total_elapsed/60:.1f} minutes")
    
    print(f"\nBreakdown by fidelity:")
    for fidelity, report in reports.items():
        print(f"  {fidelity:20s}: {report['n_samples']:6d} samples, {report['file_size_mb']:8.1f} MB")
    
    return reports


def validate_dataset(output_path: str = "./data"):
    """Validate generated datasets."""
    print("\n" + "="*60)
    print("VALIDATING DATASETS")
    print("="*60)
    
    manager = DatasetManager(output_path)
    fidelities = ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']
    
    all_valid = True
    for fidelity in fidelities:
        try:
            report = manager.validate_dataset(fidelity)
            print(f"\n{fidelity}:")
            print(f"  ✓ File exists: {manager.file_paths[fidelity]}")
            print(f"  ✓ Samples: {report['n_samples']}")
            print(f"  ✓ Size: {report['file_size_mb']:.1f} MB")
            
            # Check for NaN or Inf values
            has_issues = False
            for group_name, datasets in report['datasets'].items():
                for name, stats in datasets.items():
                    if 'mean' in stats:
                        if np.isnan(stats['mean']) or np.isinf(stats['mean']):
                            print(f"  ⚠ Warning: NaN/Inf in {group_name}/{name}")
                            has_issues = True
            
            if not has_issues:
                print(f"  ✓ Data integrity check passed")
                
        except Exception as e:
            print(f"\n{fidelity}:")
            print(f"  ✗ Validation failed: {e}")
            all_valid = False
    
    if all_valid:
        print("\n✓ All datasets validated successfully!")
    else:
        print("\n⚠ Some datasets have issues. Please check the logs.")
    
    return all_valid


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate multi-fidelity SOFC dataset')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='./data',
                       help='Output directory for datasets')
    parser.add_argument('--fidelity', type=str, choices=['low', 'mid', 'high', 'experimental', 'all'],
                       help='Fidelity level to generate')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples (overrides config)')
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel generation (for --all)')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing datasets')
    parser.add_argument('--quick-test', action='store_true',
                       help='Generate small test dataset (10 samples each)')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Save configuration to output directory
    save_config(config, os.path.join(args.output, 'config_used.yaml'))
    
    # Validate only mode
    if args.validate_only:
        validate_dataset(args.output)
        return
    
    # Quick test mode
    if args.quick_test:
        print("\nQuick test mode: Generating 10 samples per fidelity level")
        config['dataset']['low_fidelity']['n_samples'] = 10
        config['dataset']['mid_fidelity']['n_samples'] = 10
        config['dataset']['high_fidelity']['n_samples'] = 10
        config['dataset']['experimental']['n_samples'] = 10
        generate_all_datasets(config, args.output, args.parallel)
        validate_dataset(args.output)
        return
    
    # Generate datasets
    if args.fidelity == 'all' or args.fidelity is None:
        generate_all_datasets(config, args.output, args.parallel)
    elif args.fidelity == 'low':
        generate_low_fidelity(config, args.samples, args.output)
    elif args.fidelity == 'mid':
        generate_mid_fidelity(config, args.samples, args.output)
    elif args.fidelity == 'high':
        generate_high_fidelity(config, args.samples, args.output)
    elif args.fidelity == 'experimental':
        generate_experimental(config, args.samples, args.output)
    
    # Validate after generation
    print("\nValidating generated datasets...")
    validate_dataset(args.output)
    
    print("\n✓ Dataset generation complete!")
    print(f"  Datasets saved to: {os.path.abspath(args.output)}")
    print(f"  Configuration saved to: {os.path.join(args.output, 'config_used.yaml')}")


if __name__ == "__main__":
    main()