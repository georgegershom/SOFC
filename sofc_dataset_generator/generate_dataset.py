#!/usr/bin/env python3
"""
SOFC Dataset Generation Script

Main entry point for generating synthetic SOFC warp-stress datasets
for machine learning applications.

Usage:
    python generate_dataset.py --n_samples 1000 --output_dir ./output
"""

import argparse
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.dataset_generator import create_dataset_generator


def main():
    """Main function for dataset generation."""
    
    parser = argparse.ArgumentParser(
        description="Generate synthetic SOFC warp-stress dataset for ML training"
    )
    
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./sofc_dataset_output",
        help="Output directory for dataset files (default: ./sofc_dataset_output)"
    )
    
    parser.add_argument(
        "--mesh_resolution", 
        type=float, 
        default=2e-3,
        help="FEA mesh resolution in meters (default: 2e-3)"
    )
    
    parser.add_argument(
        "--grid_resolution", 
        type=float, 
        default=1e-3,
        help="Output grid resolution in meters (default: 1e-3)"
    )
    
    parser.add_argument(
        "--n_workers", 
        type=int, 
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="sofc_warp_stress_dataset",
        help="Name for the generated dataset (default: sofc_warp_stress_dataset)"
    )
    
    parser.add_argument(
        "--no_parallel", 
        action="store_true",
        help="Disable parallel processing"
    )
    
    parser.add_argument(
        "--validation_samples", 
        type=int, 
        default=100,
        help="Number of validation samples to generate (default: 100)"
    )
    
    parser.add_argument(
        "--visualizations", 
        type=int, 
        default=10,
        help="Number of sample visualizations to create (default: 10)"
    )
    
    parser.add_argument(
        "--test_run", 
        action="store_true",
        help="Run with reduced samples for testing (10 samples)"
    )
    
    args = parser.parse_args()
    
    # Adjust parameters for test run
    if args.test_run:
        args.n_samples = 10
        args.validation_samples = 5
        args.visualizations = 3
        args.mesh_resolution = 5e-3  # Coarser mesh for speed
        print("Running in test mode with reduced parameters")
    
    print("=" * 60)
    print("SOFC Dataset Generation")
    print("=" * 60)
    print(f"Number of samples: {args.n_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Mesh resolution: {args.mesh_resolution*1000:.1f} mm")
    print(f"Grid resolution: {args.grid_resolution*1000:.1f} mm")
    print(f"Parallel processing: {'Disabled' if args.no_parallel else 'Enabled'}")
    print(f"Workers: {args.n_workers or 'Auto-detect'}")
    print("=" * 60)
    
    # Create dataset generator
    generator = create_dataset_generator(
        output_directory=args.output_dir,
        n_samples=args.n_samples,
        mesh_resolution=args.mesh_resolution,
        grid_resolution=args.grid_resolution,
        n_workers=args.n_workers
    )
    
    try:
        # Generate main dataset
        print("\n🚀 Starting main dataset generation...")
        start_time = time.time()
        
        results = generator.generate_dataset(
            dataset_name=args.dataset_name,
            description=f"Synthetic SOFC warp-stress dataset with {args.n_samples} samples generated for ML training",
            parallel=not args.no_parallel
        )
        
        main_time = time.time() - start_time
        
        print(f"\n✅ Main dataset generation completed!")
        print(f"   Success rate: {results['success_rate']:.1%}")
        print(f"   Generation time: {main_time:.1f} seconds")
        print(f"   Dataset size: {results['metadata'].file_size_mb:.1f} MB")
        
        # Generate validation dataset
        if args.validation_samples > 0:
            print(f"\n🔍 Generating validation dataset ({args.validation_samples} samples)...")
            validation_results = generator.generate_validation_dataset(
                n_validation_samples=args.validation_samples,
                dataset_name=f"{args.dataset_name}_validation"
            )
            
            print(f"✅ Validation dataset completed!")
            print(f"   Success rate: {validation_results['success_rate']:.1%}")
        
        # Export visualizations
        if args.visualizations > 0:
            print(f"\n🎨 Creating {args.visualizations} sample visualizations...")
            generator.export_sample_visualizations(n_visualizations=args.visualizations)
            print("✅ Visualizations completed!")
        
        # Print final summary
        print("\n" + "=" * 60)
        print("DATASET GENERATION SUMMARY")
        print("=" * 60)
        print(f"📊 Main dataset: {results['metadata'].n_samples} samples")
        print(f"📁 Output directory: {Path(args.output_dir).absolute()}")
        print(f"💾 Dataset files:")
        print(f"   - {args.dataset_name}.h5 (HDF5 format)")
        print(f"   - {args.dataset_name}.npz (NumPy format)")
        print(f"   - {args.dataset_name}_ml_ready.npz (ML-ready format)")
        print(f"   - {args.dataset_name}_metadata.csv (Metadata)")
        
        if args.validation_samples > 0:
            print(f"🔍 Validation dataset: {validation_results['n_samples']} samples")
        
        if args.visualizations > 0:
            print(f"🎨 Visualizations: {args.visualizations} samples in visualizations/")
        
        print(f"⏱️  Total time: {time.time() - start_time:.1f} seconds")
        print("=" * 60)
        
        # Print usage examples
        print("\n📖 USAGE EXAMPLES:")
        print("   # Load dataset in Python:")
        print(f"   import h5py")
        print(f"   with h5py.File('{args.dataset_name}.h5', 'r') as f:")
        print(f"       # Access samples...")
        print()
        print("   # Load ML-ready dataset:")
        print(f"   import numpy as np")
        print(f"   data = np.load('{args.dataset_name}_ml_ready.npz')")
        print(f"   X_warp = data['X_warp']")
        print(f"   X_stress = data['X_stress']")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Generation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)