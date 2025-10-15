#!/usr/bin/env python3
"""
Quick Runner Script for SOFC Dataset Generation
===============================================

This script provides a simple interface to run the complete SOFC dataset generation
with predefined configurations for different use cases.

Usage:
    python run_dataset_generation.py [--config CONFIG] [--quick]

Configurations:
    - quick: Fast generation for testing (1000 analytical, 50 FEA, 100 sintering)
    - standard: Standard generation (5000 analytical, 100 FEA, 200 sintering)
    - comprehensive: Full generation (10000 analytical, 200 FEA, 500 sintering)
    - research: Research-grade generation (20000 analytical, 500 FEA, 1000 sintering)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from generate_complete_dataset import CompleteSOFCDatasetGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Predefined configurations
CONFIGURATIONS = {
    'quick': {
        'analytical_samples': 1000,
        'fenics_samples': 50,
        'sintering_samples': 100,
        'augmented_samples': 500,
        'description': 'Quick generation for testing (5-10 minutes)'
    },
    'standard': {
        'analytical_samples': 5000,
        'fenics_samples': 100,
        'sintering_samples': 200,
        'augmented_samples': 1000,
        'description': 'Standard generation for development (30-60 minutes)'
    },
    'comprehensive': {
        'analytical_samples': 10000,
        'fenics_samples': 200,
        'sintering_samples': 500,
        'augmented_samples': 2000,
        'description': 'Comprehensive generation for production (2-4 hours)'
    },
    'research': {
        'analytical_samples': 20000,
        'fenics_samples': 500,
        'sintering_samples': 1000,
        'augmented_samples': 5000,
        'description': 'Research-grade generation for publication (6-12 hours)'
    }
}

def run_dataset_generation(config_name: str, output_dir: str = None, skip_fenics: bool = False):
    """Run dataset generation with specified configuration"""
    
    if config_name not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(CONFIGURATIONS.keys())}")
    
    config = CONFIGURATIONS[config_name]
    
    if output_dir is None:
        output_dir = f"sofc_dataset_{config_name}"
    
    logger.info(f"Starting SOFC dataset generation with '{config_name}' configuration")
    logger.info(f"Description: {config['description']}")
    logger.info(f"Output directory: {output_dir}")
    
    # Print configuration details
    print("\n" + "="*60)
    print(f"SOFC DATASET GENERATION - {config_name.upper()} CONFIGURATION")
    print("="*60)
    print(f"Description: {config['description']}")
    print(f"Analytical samples: {config['analytical_samples']:,}")
    print(f"FEA validation samples: {config['fenics_samples']:,}")
    print(f"Sintering samples: {config['sintering_samples']:,}")
    print(f"Augmented samples per method: {config['augmented_samples']:,}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Initialize generator
    generator = CompleteSOFCDatasetGenerator(output_dir=output_dir)
    
    try:
        # Generate analytical dataset
        logger.info("Generating analytical dataset...")
        generator.generate_analytical_dataset(n_samples=config['analytical_samples'])
        
        # Generate FEA validation dataset
        if not skip_fenics:
            logger.info("Generating FEA validation dataset...")
            generator.generate_fenics_validation_dataset(n_samples=config['fenics_samples'])
        else:
            logger.info("Skipping FEA validation dataset")
        
        # Generate sintering dataset
        logger.info("Generating sintering dataset...")
        generator.generate_sintering_dataset(n_samples=config['sintering_samples'])
        
        # Generate augmented dataset
        logger.info("Generating augmented dataset...")
        generator.generate_augmented_dataset(n_samples_per_method=config['augmented_samples'])
        
        # Combine all datasets
        logger.info("Combining all datasets...")
        generator.combine_all_datasets()
        
        # Train ML models
        logger.info("Training ML models...")
        generator.train_ml_models()
        
        # Create visualizations
        logger.info("Creating visualizations...")
        generator.create_comprehensive_visualizations()
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        generator.generate_comprehensive_report()
        
        # Print success message
        print("\n" + "="*60)
        print("DATASET GENERATION COMPLETE! 🎉")
        print("="*60)
        print(f"Total samples generated: {generator.generation_stats['total_samples']:,}")
        print(f"Output directory: {output_dir}")
        print(f"Key files:")
        print(f"  - Complete dataset: {output_dir}/complete_sofc_dataset.csv")
        print(f"  - HDF5 dataset: {output_dir}/complete_sofc_dataset.h5")
        print(f"  - Visualizations: {output_dir}/visualizations/")
        print(f"  - Report: {output_dir}/comprehensive_report.md")
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        print(f"\n❌ Dataset generation failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Generate SOFC residual stress dataset with predefined configurations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dataset_generation.py --config quick
  python run_dataset_generation.py --config standard --output-dir my_dataset
  python run_dataset_generation.py --config comprehensive --skip-fenics
  python run_dataset_generation.py --quick
        """
    )
    
    parser.add_argument('--config', type=str, default='standard',
                       choices=list(CONFIGURATIONS.keys()),
                       help='Configuration to use for dataset generation')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for generated dataset')
    parser.add_argument('--skip-fenics', action='store_true',
                       help='Skip FEA validation (faster generation)')
    parser.add_argument('--quick', action='store_true',
                       help='Use quick configuration (equivalent to --config quick)')
    
    args = parser.parse_args()
    
    # Handle quick flag
    if args.quick:
        config_name = 'quick'
    else:
        config_name = args.config
    
    # Run dataset generation
    success = run_dataset_generation(
        config_name=config_name,
        output_dir=args.output_dir,
        skip_fenics=args.skip_fenics
    )
    
    if success:
        print("\n✅ Dataset generation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Dataset generation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()