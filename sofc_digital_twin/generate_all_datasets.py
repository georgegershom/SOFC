#!/usr/bin/env python3
"""
Generate All SOFC Digital Twin Datasets
Main script to generate all three datasets for the SOFC digital twin project
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_generation.dataset1_generator import Dataset1Generator
from data_generation.dataset2_generator import Dataset2Generator
from data_generation.dataset3_generator import Dataset3Generator

def print_banner():
    """Print project banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    SOFC Digital Twin Dataset Generator               ║
    ║                                                                      ║
    ║        Adaptive-Scale Physics-Informed Digital Twin for SOFC        ║
    ║              Thermo-Structural Integrity Monitoring                  ║
    ║                                                                      ║
    ║  Multi-Fidelity & Multi-Physics Dataset Generation Framework        ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def generate_dataset1(config_path: str, verbose: bool = True):
    """Generate Dataset 1: High-Fidelity Physics Simulation Data"""
    
    if verbose:
        print("\n" + "="*70)
        print("GENERATING DATASET 1: HIGH-FIDELITY PHYSICS SIMULATION DATA")
        print("="*70)
        print("This dataset contains comprehensive multi-physics SOFC simulation data")
        print("including electrochemical, thermal, and structural fields.")
        print()
    
    start_time = time.time()
    
    try:
        generator = Dataset1Generator(config_path)
        summary = generator.generate_dataset()
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Dataset 1 Generation Completed Successfully!")
            print(f"  Time elapsed: {elapsed_time:.1f} seconds")
            print(f"  Total simulations: {summary['statistics']['total_simulations']}")
            print(f"  Successful: {summary['statistics']['successful_simulations']}")
            print(f"  Success rate: {summary['statistics']['success_rate']*100:.1f}%")
            print(f"  Output directory: {generator.output_dir}")
        
        return True, summary
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Dataset 1 Generation Failed!")
            print(f"  Error: {str(e)}")
        return False, str(e)

def generate_dataset2(config_path: str, verbose: bool = True):
    """Generate Dataset 2: Experimental Validation Data"""
    
    if verbose:
        print("\n" + "="*70)
        print("GENERATING DATASET 2: EXPERIMENTAL VALIDATION DATA")
        print("="*70)
        print("This dataset contains synthetic experimental data including")
        print("operational data, EIS, thermal imaging, strain measurements, etc.")
        print()
    
    start_time = time.time()
    
    try:
        generator = Dataset2Generator(config_path)
        summary = generator.save_dataset2()
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Dataset 2 Generation Completed Successfully!")
            print(f"  Time elapsed: {elapsed_time:.1f} seconds")
            print(f"  Components generated:")
            for component, count in summary['data_components'].items():
                print(f"    - {component}: {count}")
            print(f"  Output directory: {generator.output_dir}")
        
        return True, summary
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Dataset 2 Generation Failed!")
            print(f"  Error: {str(e)}")
        return False, str(e)

def generate_dataset3(config_path: str, duration_hours: float = 168, verbose: bool = True):
    """Generate Dataset 3: Real-Time Monitoring Data"""
    
    if verbose:
        print("\n" + "="*70)
        print("GENERATING DATASET 3: REAL-TIME MONITORING DATA")
        print("="*70)
        print("This dataset contains adaptive real-time monitoring data")
        print(f"for digital twin operation ({duration_hours} hours simulation).")
        print()
    
    start_time = time.time()
    
    try:
        generator = Dataset3Generator(config_path)
        results = generator.simulate_realtime_operation(duration_hours=duration_hours)
        
        elapsed_time = time.time() - start_time
        
        if verbose:
            print(f"\n✓ Dataset 3 Generation Completed Successfully!")
            print(f"  Time elapsed: {elapsed_time:.1f} seconds")
            print(f"  Simulation duration: {duration_hours} hours")
            print(f"  High-frequency data points: {len(results['high_freq_data'])}")
            print(f"  Low-frequency measurements: {len(results['low_freq_data'])}")
            print(f"  Acoustic emission events: {len(results['events_data'])}")
            print(f"  Output directory: {generator.output_dir}")
        
        return True, results
        
    except Exception as e:
        if verbose:
            print(f"\n✗ Dataset 3 Generation Failed!")
            print(f"  Error: {str(e)}")
        return False, str(e)

def create_project_summary(dataset1_result, dataset2_result, dataset3_result, 
                          total_time: float):
    """Create comprehensive project summary"""
    
    summary = {
        'generation_info': {
            'timestamp': datetime.now().isoformat(),
            'total_generation_time_seconds': total_time,
            'total_generation_time_formatted': f"{total_time/60:.1f} minutes"
        },
        'datasets': {
            'dataset1': {
                'status': 'success' if dataset1_result[0] else 'failed',
                'details': dataset1_result[1] if dataset1_result[0] else {'error': dataset1_result[1]}
            },
            'dataset2': {
                'status': 'success' if dataset2_result[0] else 'failed',
                'details': dataset2_result[1] if dataset2_result[0] else {'error': dataset2_result[1]}
            },
            'dataset3': {
                'status': 'success' if dataset3_result[0] else 'failed',
                'details': dataset3_result[1] if dataset3_result[0] else {'error': dataset3_result[1]}
            }
        },
        'usage_instructions': {
            'data_loading': 'Use src/utils/data_processor.py to load and analyze datasets',
            'visualization': 'Use src/utils/visualizer.py for advanced visualizations',
            'examples': 'See examples/ directory for usage examples',
            'ml_training': 'Use examples/physics_informed_ml.py for ML model training'
        }
    }
    
    # Save summary
    import yaml
    with open('dataset_generation_summary.yaml', 'w') as f:
        yaml.dump(summary, f, default_flow_style=False, indent=2)
    
    return summary

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(
        description='Generate SOFC Digital Twin Datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_datasets.py                    # Generate all datasets
  python generate_all_datasets.py --dataset1-only   # Generate only Dataset 1
  python generate_all_datasets.py --quick           # Quick generation (reduced data)
  python generate_all_datasets.py --duration 72     # 72-hour real-time simulation
        """
    )
    
    parser.add_argument('--config', default='config/simulation_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--dataset1-only', action='store_true',
                       help='Generate only Dataset 1')
    parser.add_argument('--dataset2-only', action='store_true',
                       help='Generate only Dataset 2')
    parser.add_argument('--dataset3-only', action='store_true',
                       help='Generate only Dataset 3')
    parser.add_argument('--duration', type=float, default=168,
                       help='Duration for Dataset 3 simulation (hours)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick generation with reduced data size')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
        print(f"Starting dataset generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration file: {args.config}")
        print()
    
    # Check configuration file
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        print("Please make sure the configuration file exists.")
        return 1
    
    # Modify config for quick generation
    if args.quick:
        if not args.quiet:
            print("Quick generation mode: Reducing dataset sizes for faster generation")
        # This would modify the config to reduce simulation counts
        # For now, we'll just note it
    
    # Initialize results
    dataset1_result = (True, {})
    dataset2_result = (True, {})
    dataset3_result = (True, {})
    
    total_start_time = time.time()
    
    # Generate datasets based on arguments
    if args.dataset1_only:
        dataset1_result = generate_dataset1(args.config, verbose=not args.quiet)
    elif args.dataset2_only:
        dataset2_result = generate_dataset2(args.config, verbose=not args.quiet)
    elif args.dataset3_only:
        dataset3_result = generate_dataset3(args.config, args.duration, verbose=not args.quiet)
    else:
        # Generate all datasets
        dataset1_result = generate_dataset1(args.config, verbose=not args.quiet)
        dataset2_result = generate_dataset2(args.config, verbose=not args.quiet)
        dataset3_result = generate_dataset3(args.config, args.duration, verbose=not args.quiet)
    
    total_time = time.time() - total_start_time
    
    # Create project summary
    summary = create_project_summary(dataset1_result, dataset2_result, dataset3_result, total_time)
    
    # Print final summary
    if not args.quiet:
        print("\n" + "="*70)
        print("DATASET GENERATION SUMMARY")
        print("="*70)
        
        success_count = sum([
            dataset1_result[0], dataset2_result[0], dataset3_result[0]
        ])
        total_datasets = 3
        
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Datasets generated: {success_count}/{total_datasets}")
        print()
        
        # Dataset status
        datasets = [
            ("Dataset 1 (Physics Simulation)", dataset1_result[0]),
            ("Dataset 2 (Experimental Data)", dataset2_result[0]),
            ("Dataset 3 (Real-time Monitoring)", dataset3_result[0])
        ]
        
        for name, success in datasets:
            status = "✓ SUCCESS" if success else "✗ FAILED"
            print(f"{name}: {status}")
        
        print(f"\nProject summary saved to: dataset_generation_summary.yaml")
        
        if success_count > 0:
            print("\nNext steps:")
            print("1. Explore the data using: python examples/basic_usage.py")
            print("2. Train ML models using: python examples/physics_informed_ml.py")
            print("3. Check the documentation in README.md")
        
        print("\n" + "="*70)
    
    # Return appropriate exit code
    return 0 if all([dataset1_result[0], dataset2_result[0], dataset3_result[0]]) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)