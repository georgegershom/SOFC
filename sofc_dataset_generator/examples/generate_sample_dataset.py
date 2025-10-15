#!/usr/bin/env python3
"""
Example: Generate Sample SOFC Dataset

This script demonstrates how to generate a small sample dataset
for testing and development purposes.
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main_generator import create_dataset_generator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Generate a small sample dataset."""
    
    # Configuration for small test dataset
    config = {
        'dataset': {
            'name': 'SOFC_Sample_Dataset',
            'description': 'Small sample dataset for testing and development',
            'output_directory': './sample_dataset',
            'n_samples': 10  # Small number for testing
        },
        'doe': {
            'sampling_method': 'latin_hypercube',
            'seed': 42,
            'stratification': False,  # Disable for small dataset
            'optimization': 'maximin'
        },
        'fea': {
            'mesh_resolution': 5e-3,  # Coarser mesh for speed
            'element_order': 1,
            'n_time_steps': 20,       # Fewer time steps for speed
            'solver_type': 'simplified',
            'enable_thermal': True,
            'enable_mechanical': True,
            'coupling_type': 'sequential'
        },
        'warp_extraction': {
            'grid_resolution': [32, 32]  # Lower resolution for speed
        },
        'stress_extraction': {
            'voxel_resolution': [16, 16, 8]  # Lower resolution for speed
        },
        'parallel': {
            'enable': False,  # Disable for debugging
            'n_processes': 1,
            'chunk_size': 5
        },
        'output': {
            'save_intermediate': True,  # Save for inspection
            'export_formats': ['hdf5'],
            'compression': True
        }
    }
    
    logger.info("Creating dataset generator...")
    
    # Create generator with custom config
    generator = create_dataset_generator()
    generator.config = config
    
    # Validate configuration
    logger.info("Validating configuration...")
    issues = generator.validate_config()
    if issues:
        logger.error("Configuration issues found:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return 1
    
    logger.info("Configuration validated successfully!")
    
    # Generate dataset
    logger.info("Starting dataset generation...")
    try:
        dataset_id = generator.generate_dataset(resume=False)
        
        logger.info(f"Dataset generation completed!")
        logger.info(f"Dataset ID: {dataset_id}")
        
        # Get progress summary
        progress = generator.get_progress()
        logger.info(f"Final progress: {progress}")
        
        # Get dataset summary
        if generator.dataset_manager:
            summary = generator.dataset_manager.get_dataset_summary()
            logger.info(f"Dataset summary:")
            logger.info(f"  - Total samples: {summary['n_samples']}")
            logger.info(f"  - Storage size: {summary['storage_info']['total_size_mb']:.2f} MB")
            logger.info(f"  - Dataset root: {summary['storage_info']['dataset_root']}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())