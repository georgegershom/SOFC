"""
Main SOFC Dataset Generator

This module orchestrates the complete dataset generation process:
1. Generate DOE matrix
2. Run FEA simulations for each scenario
3. Extract warp and stress fields
4. Create paired dataset for ML training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import time
import logging
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

# Import all modules
from .doe.doe_generator import create_doe_generator, DOEConfiguration
from .fea.fea_solver import create_fea_solver, FEAConfiguration
from .materials.material_models import create_sofc_material_model
from .geometry.mesh_generator import create_mesh_generator
from .extraction.warp_extractor import create_warp_extractor
from .extraction.stress_extractor import create_stress_extractor
from .utils.data_manager import create_dataset_manager, DatasetSample


class SOFCDatasetGenerator:
    """Main class for generating SOFC warp-stress datasets."""
    
    def __init__(self, 
                 output_directory: Union[str, Path],
                 n_samples: int = 1000,
                 mesh_resolution: float = 2e-3,
                 grid_resolution: float = 1e-3,
                 n_workers: int = None):
        """Initialize dataset generator.
        
        Args:
            output_directory: Directory to store generated datasets
            n_samples: Number of samples to generate
            mesh_resolution: FEA mesh resolution in meters
            grid_resolution: Output grid resolution in meters
            n_workers: Number of parallel workers (None = auto)
        """
        
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_samples = n_samples
        self.mesh_resolution = mesh_resolution
        self.grid_resolution = grid_resolution
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Progress tracking
        self.completed_samples = 0
        self.failed_samples = 0
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"dataset_generation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting SOFC dataset generation with {self.n_samples} samples")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Mesh resolution: {self.mesh_resolution*1000:.1f} mm")
        self.logger.info(f"Grid resolution: {self.grid_resolution*1000:.1f} mm")
        self.logger.info(f"Number of workers: {self.n_workers}")
    
    def _initialize_components(self):
        """Initialize all components for dataset generation."""
        
        # DOE generator
        doe_config = DOEConfiguration(
            n_samples=self.n_samples,
            sampling_method="latin_hypercube",
            seed=42
        )
        self.doe_generator = create_doe_generator(doe_config=doe_config)
        
        # FEA solver configuration
        fea_config = FEAConfiguration(
            mesh_resolution=self.mesh_resolution,
            n_time_steps=50,
            solver_type="simplified",  # Use simplified solver for speed
            enable_thermal=True,
            enable_mechanical=True
        )
        self.fea_config = fea_config
        
        # Material model
        self.material_model = create_sofc_material_model()
        
        # Mesh generator
        self.mesh_generator = create_mesh_generator(self.mesh_resolution)
        
        # Extractors
        self.warp_extractor = create_warp_extractor(self.grid_resolution)
        self.stress_extractor = create_stress_extractor(self.grid_resolution)
        
        # Data manager
        self.data_manager = create_dataset_manager(self.output_dir)
        
        self.logger.info("All components initialized successfully")
    
    def generate_dataset(self, 
                        dataset_name: str = "sofc_warp_stress_dataset",
                        description: str = "",
                        parallel: bool = True) -> Dict[str, Any]:
        """Generate the complete dataset.
        
        Args:
            dataset_name: Name for the dataset
            description: Description of the dataset
            parallel: Whether to use parallel processing
            
        Returns:
            Dictionary with dataset statistics and metadata
        """
        
        start_time = time.time()
        self.logger.info(f"Starting dataset generation: {dataset_name}")
        
        # Generate DOE matrix
        self.logger.info("Generating DOE matrix...")
        doe_matrix = self.doe_generator.generate_doe_matrix()
        self.logger.info(f"Generated DOE matrix with {len(doe_matrix)} scenarios")
        
        # Validate DOE matrix
        validation_results = self.doe_generator.validate_doe_matrix(doe_matrix)
        if not all(validation_results.values()):
            self.logger.warning(f"DOE matrix validation issues: {validation_results}")
        
        # Convert DOE matrix to list of parameter dictionaries
        doe_scenarios = []
        for idx, row in doe_matrix.iterrows():
            scenario = {
                'scenario_id': f"scenario_{idx:06d}",
                **row.to_dict()
            }
            doe_scenarios.append(scenario)
        
        # Generate samples
        if parallel and self.n_workers > 1:
            self._generate_samples_parallel(doe_scenarios)
        else:
            self._generate_samples_sequential(doe_scenarios)
        
        # Create dataset from samples
        self.logger.info("Creating final dataset...")
        metadata = self.data_manager.create_dataset_from_samples(
            dataset_name=dataset_name,
            description=description or f"SOFC warp-stress dataset with {self.n_samples} samples",
            tags=["SOFC", "warp", "stress", "FEA", "ML"]
        )
        
        # Create ML-ready dataset
        self.logger.info("Creating ML-ready dataset...")
        ml_dataset = self.data_manager.create_ml_ready_dataset(dataset_name)
        
        # Generate statistics
        dataset_stats = self.data_manager.get_dataset_statistics()
        
        # Final summary
        total_time = time.time() - start_time
        self.logger.info(f"Dataset generation completed in {total_time:.2f} seconds")
        self.logger.info(f"Successfully generated {self.completed_samples} samples")
        self.logger.info(f"Failed samples: {self.failed_samples}")
        
        return {
            'metadata': metadata,
            'statistics': dataset_stats,
            'ml_dataset_info': {k: v.shape if hasattr(v, 'shape') else type(v).__name__ 
                               for k, v in ml_dataset.items()},
            'generation_time': total_time,
            'success_rate': self.completed_samples / len(doe_scenarios) if doe_scenarios else 0.0
        }
    
    def _generate_samples_sequential(self, doe_scenarios: List[Dict[str, Any]]):
        """Generate samples sequentially."""
        
        self.logger.info("Generating samples sequentially...")
        
        for scenario in tqdm(doe_scenarios, desc="Generating samples"):
            try:
                sample = self._generate_single_sample(scenario)
                if sample:
                    self.data_manager.add_sample(sample)
                    self.completed_samples += 1
                else:
                    self.failed_samples += 1
            except Exception as e:
                self.logger.error(f"Failed to generate sample {scenario['scenario_id']}: {str(e)}")
                self.failed_samples += 1
    
    def _generate_samples_parallel(self, doe_scenarios: List[Dict[str, Any]]):
        """Generate samples in parallel."""
        
        self.logger.info(f"Generating samples in parallel with {self.n_workers} workers...")
        
        # Split scenarios into chunks for better load balancing
        chunk_size = max(1, len(doe_scenarios) // (self.n_workers * 4))
        scenario_chunks = [doe_scenarios[i:i + chunk_size] 
                          for i in range(0, len(doe_scenarios), chunk_size)]
        
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(self._generate_sample_chunk, chunk): chunk 
                for chunk in scenario_chunks
            }
            
            # Process completed chunks
            for future in tqdm(as_completed(future_to_chunk), 
                             total=len(scenario_chunks), 
                             desc="Processing chunks"):
                try:
                    chunk_results = future.result()
                    for sample in chunk_results:
                        if sample:
                            self.data_manager.add_sample(sample)
                            self.completed_samples += 1
                        else:
                            self.failed_samples += 1
                except Exception as e:
                    self.logger.error(f"Chunk processing failed: {str(e)}")
                    chunk = future_to_chunk[future]
                    self.failed_samples += len(chunk)
    
    def _generate_sample_chunk(self, scenarios: List[Dict[str, Any]]) -> List[Optional[DatasetSample]]:
        """Generate a chunk of samples (for parallel processing)."""
        
        # Re-initialize components in worker process
        self._initialize_components()
        
        results = []
        for scenario in scenarios:
            try:
                sample = self._generate_single_sample(scenario)
                results.append(sample)
            except Exception as e:
                # Log error and continue
                print(f"Error generating sample {scenario['scenario_id']}: {str(e)}")
                results.append(None)
        
        return results
    
    def _generate_single_sample(self, doe_parameters: Dict[str, Any]) -> Optional[DatasetSample]:
        """Generate a single dataset sample."""
        
        scenario_id = doe_parameters['scenario_id']
        
        try:
            # Step 1: Setup FEA problem
            fea_solver = create_fea_solver(self.fea_config)
            
            # Create geometry info for extractors
            geometry_info = {
                'total_thickness': (
                    doe_parameters.get('geometry.anode_thickness', 500e-6) +
                    doe_parameters.get('geometry.electrolyte_thickness', 15e-6) +
                    doe_parameters.get('geometry.cathode_thickness', 40e-6)
                ),
                'layers': {
                    'anode': {
                        'z_bottom': 0.0,
                        'z_top': doe_parameters.get('geometry.anode_thickness', 500e-6)
                    },
                    'electrolyte': {
                        'z_bottom': doe_parameters.get('geometry.anode_thickness', 500e-6),
                        'z_top': (doe_parameters.get('geometry.anode_thickness', 500e-6) + 
                                 doe_parameters.get('geometry.electrolyte_thickness', 15e-6))
                    },
                    'cathode': {
                        'z_bottom': (doe_parameters.get('geometry.anode_thickness', 500e-6) + 
                                   doe_parameters.get('geometry.electrolyte_thickness', 15e-6)),
                        'z_top': (doe_parameters.get('geometry.anode_thickness', 500e-6) + 
                                doe_parameters.get('geometry.electrolyte_thickness', 15e-6) +
                                doe_parameters.get('geometry.cathode_thickness', 40e-6))
                    }
                }
            }
            
            fea_solver.setup_problem(doe_parameters, self.material_model, self.mesh_generator)
            
            # Step 2: Run FEA simulation
            simulation_results = fea_solver.solve()
            
            # Step 3: Extract warp field
            warp_data = self.warp_extractor.extract_warp_field(simulation_results, geometry_info)
            
            # Step 4: Extract stress field
            stress_data = self.stress_extractor.extract_stress_field(simulation_results, geometry_info)
            
            # Step 5: Create dataset sample
            sample = DatasetSample(
                sample_id=scenario_id,
                doe_parameters=doe_parameters,
                warp_data=warp_data,
                stress_data=stress_data,
                metadata={
                    'mesh_resolution': self.mesh_resolution,
                    'grid_resolution': self.grid_resolution,
                    'simulation_time': simulation_results.simulation_time
                },
                timestamp=datetime.now().isoformat()
            )
            
            return sample
            
        except Exception as e:
            error_msg = f"Error generating sample {scenario_id}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            return None
    
    def generate_validation_dataset(self, 
                                  n_validation_samples: int = 100,
                                  dataset_name: str = "sofc_validation_dataset") -> Dict[str, Any]:
        """Generate a smaller validation dataset with different parameter ranges."""
        
        self.logger.info(f"Generating validation dataset with {n_validation_samples} samples")
        
        # Create validation DOE with different sampling
        validation_doe_config = DOEConfiguration(
            n_samples=n_validation_samples,
            sampling_method="sobol",  # Use Sobol for validation
            seed=123  # Different seed
        )
        
        validation_doe_generator = create_doe_generator(doe_config=validation_doe_config)
        validation_doe_matrix = validation_doe_generator.generate_doe_matrix()
        
        # Convert to scenarios
        validation_scenarios = []
        for idx, row in validation_doe_matrix.iterrows():
            scenario = {
                'scenario_id': f"validation_{idx:06d}",
                **row.to_dict()
            }
            validation_scenarios.append(scenario)
        
        # Generate validation samples
        original_n_samples = self.n_samples
        self.n_samples = n_validation_samples
        
        # Reset counters
        self.completed_samples = 0
        self.failed_samples = 0
        
        # Generate samples
        self._generate_samples_sequential(validation_scenarios)  # Use sequential for validation
        
        # Create validation dataset
        metadata = self.data_manager.create_dataset_from_samples(
            dataset_name=dataset_name,
            description=f"SOFC validation dataset with {n_validation_samples} samples",
            tags=["SOFC", "validation", "warp", "stress"]
        )
        
        # Restore original settings
        self.n_samples = original_n_samples
        
        return {
            'metadata': metadata,
            'n_samples': self.completed_samples,
            'success_rate': self.completed_samples / len(validation_scenarios)
        }
    
    def export_sample_visualizations(self, n_visualizations: int = 10):
        """Export visualizations for a subset of samples."""
        
        if not self.data_manager.samples:
            self.logger.warning("No samples available for visualization")
            return
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Select samples for visualization
        n_viz = min(n_visualizations, len(self.data_manager.samples))
        viz_samples = self.data_manager.samples[:n_viz]
        
        self.logger.info(f"Generating visualizations for {n_viz} samples...")
        
        for i, sample in enumerate(tqdm(viz_samples, desc="Creating visualizations")):
            try:
                # Warp field visualization
                warp_fig = self.warp_extractor.visualize_warp_field(
                    sample.warp_data, 
                    save_path=viz_dir / f"warp_field_{sample.sample_id}.png"
                )
                warp_fig.close()
                
                # Stress field visualization
                stress_fig = self.stress_extractor.visualize_stress_field(
                    sample.stress_data,
                    save_path=viz_dir / f"stress_field_{sample.sample_id}.png"
                )
                stress_fig.close()
                
            except Exception as e:
                self.logger.error(f"Failed to create visualization for {sample.sample_id}: {str(e)}")
        
        self.logger.info(f"Visualizations saved to {viz_dir}")


def create_dataset_generator(output_directory: Union[str, Path],
                           n_samples: int = 1000,
                           mesh_resolution: float = 2e-3,
                           grid_resolution: float = 1e-3,
                           n_workers: int = None) -> SOFCDatasetGenerator:
    """Factory function to create dataset generator."""
    
    return SOFCDatasetGenerator(
        output_directory=output_directory,
        n_samples=n_samples,
        mesh_resolution=mesh_resolution,
        grid_resolution=grid_resolution,
        n_workers=n_workers
    )


if __name__ == "__main__":
    # Example usage
    generator = create_dataset_generator(
        output_directory="sofc_dataset_output",
        n_samples=100,  # Small test dataset
        mesh_resolution=3e-3,  # 3mm mesh
        grid_resolution=2e-3,  # 2mm grid
        n_workers=2
    )
    
    # Generate main dataset
    results = generator.generate_dataset(
        dataset_name="sofc_test_dataset",
        description="Test SOFC dataset for validation",
        parallel=True
    )
    
    print(f"Dataset generation completed!")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Generation time: {results['generation_time']:.2f} seconds")
    
    # Generate validation dataset
    validation_results = generator.generate_validation_dataset(
        n_validation_samples=20,
        dataset_name="sofc_validation_test"
    )
    
    print(f"Validation dataset success rate: {validation_results['success_rate']:.2%}")
    
    # Export visualizations
    generator.export_sample_visualizations(n_visualizations=5)