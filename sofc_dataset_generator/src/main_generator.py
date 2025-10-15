"""
Main SOFC Dataset Generator

This module orchestrates the complete dataset generation process:
1. Generate DOE matrix
2. Run FEA simulations for each DOE point
3. Extract warp and stress fields
4. Store paired data for ML training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import yaml
import json
from datetime import datetime
import logging
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import all components
from .doe.parameter_space import create_sofc_parameter_space
from .doe.doe_generator import create_doe_generator, DOEConfiguration
from .materials.material_models import create_sofc_material_model
from .geometry.mesh_generator import create_mesh_generator
from .fea.fea_solver import create_fea_solver, FEAConfiguration
from .extraction.warp_extractor import create_warp_extractor
from .extraction.stress_extractor import create_stress_extractor
from .utils.dataset_manager import create_dataset_manager


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SOFCDatasetGenerator:
    """Main class for generating SOFC warp-stress paired datasets."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize dataset generator.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config = self._load_config(config_file)
        
        # Initialize components
        self.parameter_space = create_sofc_parameter_space()
        self.material_model = create_sofc_material_model()
        
        # Initialize extractors
        self.warp_extractor = create_warp_extractor(
            grid_resolution=tuple(self.config['warp_extraction']['grid_resolution'])
        )
        self.stress_extractor = create_stress_extractor(
            voxel_resolution=tuple(self.config['stress_extraction']['voxel_resolution'])
        )
        
        # Dataset manager (initialized when needed)
        self.dataset_manager = None
        
        # Progress tracking
        self.progress = {
            'total_samples': 0,
            'completed_samples': 0,
            'failed_samples': 0,
            'current_sample': None
        }
    
    def _load_config(self, config_file: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        
        default_config = {
            'dataset': {
                'name': 'SOFC_Warp_Stress_Dataset',
                'description': 'Synthetic dataset of paired warp and stress fields for SOFC plates',
                'output_directory': './sofc_dataset',
                'n_samples': 1000
            },
            'doe': {
                'sampling_method': 'latin_hypercube',
                'seed': 42,
                'stratification': True,
                'optimization': 'maximin'
            },
            'fea': {
                'mesh_resolution': 2e-3,  # 2 mm elements
                'element_order': 1,
                'n_time_steps': 50,
                'solver_type': 'simplified',
                'enable_thermal': True,
                'enable_mechanical': True,
                'coupling_type': 'sequential'
            },
            'warp_extraction': {
                'grid_resolution': [64, 64]
            },
            'stress_extraction': {
                'voxel_resolution': [32, 32, 16]
            },
            'parallel': {
                'enable': True,
                'n_processes': None,  # Use all available cores
                'chunk_size': 10
            },
            'output': {
                'save_intermediate': False,
                'export_formats': ['hdf5'],
                'compression': True
            }
        }
        
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                user_config = yaml.safe_load(f)
            
            # Merge configurations (user config overrides defaults)
            config = self._merge_configs(default_config, user_config)
        else:
            config = default_config
        
        return config
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def generate_dataset(self, resume: bool = False) -> str:
        """Generate the complete dataset.
        
        Args:
            resume: Whether to resume from existing dataset
            
        Returns:
            Dataset ID
        """
        logger.info("Starting SOFC dataset generation...")
        
        # Initialize dataset manager
        output_dir = Path(self.config['dataset']['output_directory'])
        self.dataset_manager = create_dataset_manager(output_dir)
        
        if resume and (output_dir / "metadata.json").exists():
            logger.info("Resuming existing dataset...")
            self.dataset_manager.load_dataset()
            dataset_id = self.dataset_manager.metadata.dataset_id
            existing_samples = len(self.dataset_manager.samples)
            logger.info(f"Found {existing_samples} existing samples")
        else:
            logger.info("Creating new dataset...")
            # Create new dataset
            dataset_id = self.dataset_manager.create_dataset(
                dataset_name=self.config['dataset']['name'],
                description=self.config['dataset']['description'],
                parameter_ranges=self.parameter_space.summary().to_dict(),
                doe_method=self.config['doe']['sampling_method']
            )
            existing_samples = 0
        
        # Generate DOE matrix
        n_total_samples = self.config['dataset']['n_samples']
        n_new_samples = n_total_samples - existing_samples
        
        if n_new_samples <= 0:
            logger.info("Dataset already complete!")
            return dataset_id
        
        logger.info(f"Generating {n_new_samples} new samples...")
        
        doe_config = DOEConfiguration(
            n_samples=n_new_samples,
            sampling_method=self.config['doe']['sampling_method'],
            seed=self.config['doe']['seed'],
            stratification=self.config['doe']['stratification'],
            optimization=self.config['doe']['optimization']
        )
        
        doe_generator = create_doe_generator(doe_config=doe_config)
        doe_matrix = doe_generator.generate_doe_matrix()
        
        logger.info(f"Generated DOE matrix with {len(doe_matrix)} samples")
        
        # Update progress tracking
        self.progress['total_samples'] = len(doe_matrix)
        self.progress['completed_samples'] = 0
        self.progress['failed_samples'] = 0
        
        # Generate samples
        if self.config['parallel']['enable']:
            self._generate_samples_parallel(doe_matrix)
        else:
            self._generate_samples_sequential(doe_matrix)
        
        # Calculate dataset statistics
        logger.info("Calculating dataset statistics...")
        self.dataset_manager.calculate_dataset_statistics()
        
        # Export dataset for ML
        logger.info("Exporting dataset for ML...")
        exported_files = self.dataset_manager.export_for_ml(
            export_format='hdf5',
            include_raw_data=True,
            train_split=0.8
        )
        
        logger.info(f"Dataset generation completed!")
        logger.info(f"Dataset ID: {dataset_id}")
        logger.info(f"Total samples: {self.progress['completed_samples']}")
        logger.info(f"Failed samples: {self.progress['failed_samples']}")
        logger.info(f"Exported files: {list(exported_files.keys())}")
        
        return dataset_id
    
    def _generate_samples_sequential(self, doe_matrix: pd.DataFrame):
        """Generate samples sequentially (single-threaded)."""
        
        for i, (_, doe_params) in enumerate(tqdm(doe_matrix.iterrows(), 
                                                total=len(doe_matrix),
                                                desc="Generating samples")):
            
            self.progress['current_sample'] = i
            
            try:
                sample_id = self._generate_single_sample(doe_params.to_dict(), i)
                self.progress['completed_samples'] += 1
                logger.debug(f"Completed sample {i}: {sample_id}")
                
            except Exception as e:
                self.progress['failed_samples'] += 1
                logger.error(f"Failed to generate sample {i}: {str(e)}")
                logger.debug(traceback.format_exc())
    
    def _generate_samples_parallel(self, doe_matrix: pd.DataFrame):
        """Generate samples in parallel (multi-threaded)."""
        
        n_processes = self.config['parallel']['n_processes']
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        chunk_size = self.config['parallel']['chunk_size']
        
        logger.info(f"Using {n_processes} processes for parallel generation")
        
        # Split DOE matrix into chunks
        chunks = [doe_matrix.iloc[i:i+chunk_size] for i in range(0, len(doe_matrix), chunk_size)]
        
        with ProcessPoolExecutor(max_processes=n_processes) as executor:
            # Submit chunks
            future_to_chunk = {}
            for chunk_idx, chunk in enumerate(chunks):
                future = executor.submit(self._process_chunk, chunk, chunk_idx)
                future_to_chunk[future] = chunk_idx
            
            # Process results
            with tqdm(total=len(doe_matrix), desc="Generating samples") as pbar:
                for future in as_completed(future_to_chunk):
                    chunk_idx = future_to_chunk[future]
                    
                    try:
                        chunk_results = future.result()
                        
                        # Update progress
                        for result in chunk_results:
                            if result['success']:
                                self.progress['completed_samples'] += 1
                            else:
                                self.progress['failed_samples'] += 1
                            
                            pbar.update(1)
                        
                        logger.info(f"Completed chunk {chunk_idx}")
                        
                    except Exception as e:
                        logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
                        self.progress['failed_samples'] += len(chunks[chunk_idx])
                        pbar.update(len(chunks[chunk_idx]))
    
    def _process_chunk(self, chunk: pd.DataFrame, chunk_idx: int) -> List[Dict[str, Any]]:
        """Process a chunk of DOE samples."""
        
        results = []
        
        for i, (_, doe_params) in enumerate(chunk.iterrows()):
            sample_idx = chunk_idx * self.config['parallel']['chunk_size'] + i
            
            try:
                sample_id = self._generate_single_sample(doe_params.to_dict(), sample_idx)
                results.append({
                    'sample_idx': sample_idx,
                    'sample_id': sample_id,
                    'success': True,
                    'error': None
                })
                
            except Exception as e:
                results.append({
                    'sample_idx': sample_idx,
                    'sample_id': None,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def _generate_single_sample(self, doe_parameters: Dict[str, Any], sample_idx: int) -> str:
        """Generate a single sample (DOE point -> FEA -> extraction -> storage).
        
        Args:
            doe_parameters: DOE parameters for this sample
            sample_idx: Sample index for logging
            
        Returns:
            Sample ID
        """
        
        logger.debug(f"Generating sample {sample_idx}")
        
        # 1. Create mesh
        mesh_generator = create_mesh_generator(
            mesh_resolution=self.config['fea']['mesh_resolution']
        )
        mesh = mesh_generator.generate_mesh(doe_parameters)
        
        # 2. Setup and run FEA simulation
        fea_config = FEAConfiguration(
            mesh_resolution=self.config['fea']['mesh_resolution'],
            element_order=self.config['fea']['element_order'],
            n_time_steps=self.config['fea']['n_time_steps'],
            solver_type=self.config['fea']['solver_type'],
            enable_thermal=self.config['fea']['enable_thermal'],
            enable_mechanical=self.config['fea']['enable_mechanical'],
            coupling_type=self.config['fea']['coupling_type']
        )
        
        fea_solver = create_fea_solver(fea_config)
        fea_solver.setup_problem(doe_parameters, self.material_model, mesh_generator)
        
        simulation_results = fea_solver.solve()
        
        # 3. Extract warp field
        warp_data = self.warp_extractor.extract_warp_field(
            mesh, 
            simulation_results.final_displacement,
            doe_parameters
        )
        
        # 4. Extract stress field
        stress_data = self.stress_extractor.extract_stress_field(
            mesh,
            simulation_results.final_stress,
            doe_parameters
        )
        
        # 5. Prepare simulation metadata
        simulation_metadata = {
            'simulation_time': simulation_results.simulation_time,
            'n_nodes': len(simulation_results.coordinates),
            'n_elements': len(simulation_results.connectivity) if simulation_results.connectivity is not None else 0,
            'fea_config': {
                'mesh_resolution': fea_config.mesh_resolution,
                'n_time_steps': fea_config.n_time_steps,
                'solver_type': fea_config.solver_type
            },
            'convergence_achieved': len(simulation_results.convergence_history) > 0,
            'sample_index': sample_idx,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        # 6. Store sample in dataset
        sample_id = self.dataset_manager.add_sample(
            doe_parameters=doe_parameters,
            warp_data=warp_data,
            stress_data=stress_data,
            simulation_metadata=simulation_metadata
        )
        
        return sample_id
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current generation progress."""
        progress = self.progress.copy()
        
        if progress['total_samples'] > 0:
            progress['completion_percentage'] = (
                progress['completed_samples'] / progress['total_samples'] * 100
            )
        else:
            progress['completion_percentage'] = 0.0
        
        return progress
    
    def save_config(self, filepath: Union[str, Path]):
        """Save current configuration to file."""
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, indent=2)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Check required sections
        required_sections = ['dataset', 'doe', 'fea', 'warp_extraction', 'stress_extraction']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required configuration section: {section}")
        
        # Check dataset config
        if 'dataset' in self.config:
            if self.config['dataset'].get('n_samples', 0) <= 0:
                issues.append("Dataset n_samples must be positive")
        
        # Check DOE config
        if 'doe' in self.config:
            valid_methods = ['latin_hypercube', 'sobol', 'random', 'full_factorial']
            if self.config['doe'].get('sampling_method') not in valid_methods:
                issues.append(f"Invalid DOE sampling method. Must be one of: {valid_methods}")
        
        # Check FEA config
        if 'fea' in self.config:
            if self.config['fea'].get('mesh_resolution', 0) <= 0:
                issues.append("FEA mesh_resolution must be positive")
            
            if self.config['fea'].get('n_time_steps', 0) <= 0:
                issues.append("FEA n_time_steps must be positive")
        
        return issues


def create_dataset_generator(config_file: Optional[Union[str, Path]] = None) -> SOFCDatasetGenerator:
    """Factory function to create dataset generator."""
    return SOFCDatasetGenerator(config_file)


def main():
    """Main entry point for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate SOFC warp-stress paired dataset')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--resume', action='store_true', help='Resume existing dataset generation')
    parser.add_argument('--validate', action='store_true', help='Validate configuration only')
    parser.add_argument('--n-samples', type=int, help='Override number of samples')
    parser.add_argument('--output-dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    # Create generator
    generator = create_dataset_generator(args.config)
    
    # Override config if specified
    if args.n_samples:
        generator.config['dataset']['n_samples'] = args.n_samples
    
    if args.output_dir:
        generator.config['dataset']['output_directory'] = args.output_dir
    
    # Validate configuration
    issues = generator.validate_config()
    if issues:
        logger.error("Configuration validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        return 1
    
    if args.validate:
        logger.info("Configuration validation passed!")
        return 0
    
    # Generate dataset
    try:
        dataset_id = generator.generate_dataset(resume=args.resume)
        logger.info(f"Dataset generation completed successfully!")
        logger.info(f"Dataset ID: {dataset_id}")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())