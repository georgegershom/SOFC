#!/usr/bin/env python
"""
Quick Start Script for SOFC Multi-Fidelity Dataset Generation
"""

import os
import sys
import time
import argparse
from generate_dataset import SOFCDatasetGenerator
from visualize_data import SOFCDataVisualizer
from validate_data import SOFCDataValidator

def print_header():
    """Print welcome header"""
    print("\n" + "="*70)
    print(" "*15 + "SOFC MULTI-FIDELITY DATASET GENERATOR")
    print(" "*10 + "Multi-Scale Modeling & Deep Learning for SOFCs")
    print("="*70)

def quick_generate(small: bool = True):
    """Quick dataset generation"""
    print("\n📊 GENERATING SOFC DATASET...")
    print("-"*50)
    
    # Set parameters based on size
    if small:
        n_lf, n_mf, n_hf = 100, 20, 5
        time_points = 50
        print("Mode: SMALL (for quick testing)")
    else:
        n_lf, n_mf, n_hf = 1000, 100, 10
        time_points = 100
        print("Mode: FULL")
    
    print(f"Samples: LF={n_lf}, MF={n_mf}, HF={n_hf}")
    print(f"Time points: {time_points}")
    
    # Create generator
    generator = SOFCDatasetGenerator('datasets')
    
    # Generate dataset
    start_time = time.time()
    dataset_path = generator.generate_complete_dataset(
        n_lf=n_lf,
        n_mf=n_mf,
        n_hf=n_hf,
        time_points=time_points,
        max_hours=10000,
        n_jobs=4  # Limit parallel jobs for stability
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ Dataset generated in {elapsed_time:.1f} seconds")
    print(f"📁 Location: {dataset_path}")
    
    return dataset_path

def quick_validate(dataset_path: str):
    """Quick validation of generated dataset"""
    print("\n🔍 VALIDATING DATASET...")
    print("-"*50)
    
    validator = SOFCDataValidator(dataset_path)
    results = validator.validate_dataset()
    
    # Check for critical errors
    if validator.errors:
        print(f"\n⚠️  Found {len(validator.errors)} errors - please review")
    else:
        print("\n✅ Dataset validation passed!")
    
    return results

def quick_visualize(dataset_path: str):
    """Quick visualization of dataset"""
    print("\n📈 GENERATING VISUALIZATIONS...")
    print("-"*50)
    
    visualizer = SOFCDataVisualizer(dataset_path, 'visualizations')
    
    # Generate key visualizations
    print("Creating plots...")
    visualizer.plot_parameter_distributions()
    visualizer.plot_degradation_curves()
    visualizer.create_summary_report()
    
    print("\n✅ Visualizations saved to: visualizations/")

def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Quick start for SOFC dataset generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python quickstart.py --small           # Generate small test dataset
  python quickstart.py --full            # Generate full dataset
  python quickstart.py --validate only   # Validate existing dataset
        """
    )
    
    parser.add_argument('--small', action='store_true', 
                       help='Generate small dataset for testing (default)')
    parser.add_argument('--full', action='store_true',
                       help='Generate full-size dataset')
    parser.add_argument('--validate', type=str, metavar='PATH',
                       help='Validate existing dataset at PATH')
    parser.add_argument('--visualize', type=str, metavar='PATH',
                       help='Visualize existing dataset at PATH')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization step')
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Determine mode
    if args.validate:
        # Validate only mode
        if args.validate == 'only':
            # Find most recent dataset
            import glob
            datasets = glob.glob('datasets/sofc_dataset_*.h5')
            if not datasets:
                print("❌ No datasets found in datasets/ directory")
                sys.exit(1)
            dataset_path = sorted(datasets)[-1]
            print(f"Using most recent dataset: {dataset_path}")
        else:
            dataset_path = args.validate
        
        quick_validate(dataset_path)
        if not args.skip_viz:
            quick_visualize(dataset_path)
    
    elif args.visualize:
        # Visualize only mode
        quick_visualize(args.visualize)
    
    else:
        # Generate mode
        small_mode = not args.full
        
        print("\n📋 WORKFLOW:")
        print("  1. Generate multi-fidelity samples")
        print("  2. Compute degradation responses")
        print("  3. Validate dataset integrity")
        print("  4. Create visualizations")
        print("\nStarting generation...\n")
        
        # Generate dataset
        dataset_path = quick_generate(small=small_mode)
        
        # Validate
        quick_validate(dataset_path)
        
        # Visualize (unless skipped)
        if not args.skip_viz:
            quick_visualize(dataset_path)
        
        # Print summary
        print("\n" + "="*70)
        print("✨ DATASET GENERATION COMPLETE!")
        print("="*70)
        print("\n📦 Generated files:")
        print(f"  • Dataset: {dataset_path}")
        print(f"  • Metadata: datasets/metadata.json")
        print(f"  • Inputs CSV: datasets/inputs_*.csv")
        print(f"  • Validation: validation_report.json")
        if not args.skip_viz:
            print(f"  • Visualizations: visualizations/")
        
        print("\n🚀 Next steps:")
        print("  1. Review the validation report")
        print("  2. Explore visualizations in visualizations/")
        print("  3. Load dataset in Python:")
        print("     >>> import h5py")
        print(f"     >>> f = h5py.File('{dataset_path}', 'r')")
        print("     >>> print(list(f.keys()))")
        
        print("\n📚 For more examples, run:")
        print("     python examples.py")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)