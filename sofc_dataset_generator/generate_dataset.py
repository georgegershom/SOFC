#!/usr/bin/env python3
"""
SOFC Synthetic Dataset Generator

Main script for generating synthetic datasets for ML-augmented inverse modeling
of residual stress quantification from warped SOFC plates.

This script implements the "Virtual DOE" methodology described in the research article,
generating 500-1000+ manufacturing scenarios with paired warp and stress data.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import h5py
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.materials.sofc_materials import SOFCMaterials
from src.materials.creep_models import NortonBaileyCreep
from src.doe.doe_generator import DOEGenerator
from src.doe.sofc_parameters import SOFCParameters
from src.fea.mesh_generator import SOFCMeshGenerator, MeshParameters
from src.fea.fea_solver import FEASolver
from src.fea.warp_analysis import WarpAnalyzer
from src.fea.stress_analysis import StressAnalyzer


class SOFCDatasetGenerator:
    """Main class for generating SOFC synthetic datasets"""
    
    def __init__(self, output_dir: str = './dataset', random_seed: int = 42):
        self.output_dir = output_dir
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.materials = SOFCMaterials()
        self.doe_generator = DOEGenerator()
        self.creep_model = NortonBaileyCreep()
        
        # Dataset storage
        self.dataset = {
            'warp_data': [],
            'stress_data': [],
            'parameters': [],
            'metadata': []
        }
        
        print(f"SOFC Dataset Generator initialized")
        print(f"Output directory: {output_dir}")
        print(f"Random seed: {random_seed}")
    
    def generate_dataset(self, n_samples: int = 100, 
                        strategy: str = 'lhs',
                        use_creep: bool = True,
                        save_intermediate: bool = True):
        """
        Generate complete synthetic dataset
        
        Args:
            n_samples: Number of samples to generate
            strategy: DOE sampling strategy ('lhs', 'sobol', 'random')
            use_creep: Whether to include creep effects
            save_intermediate: Whether to save intermediate results
        """
        print(f"\nGenerating SOFC synthetic dataset with {n_samples} samples")
        print(f"Sampling strategy: {strategy}")
        print(f"Creep effects: {'Enabled' if use_creep else 'Disabled'}")
        
        # Generate DOE matrix
        print("\n1. Generating Design of Experiments matrix...")
        doe_df = self.doe_generator.generate_doe(
            n_samples=n_samples,
            strategy=strategy,
            random_state=self.random_seed,
            save_to_file=os.path.join(self.output_dir, 'doe_matrix.csv') if save_intermediate else None
        )
        
        # Generate manufacturing parameters
        manufacturing_params = self.doe_generator.generate_manufacturing_parameters(doe_df)
        
        print(f"Generated {len(manufacturing_params)} parameter combinations")
        
        # Process each sample
        print("\n2. Running FEA simulations...")
        for i, mfg_params in enumerate(tqdm(manufacturing_params, desc="Processing samples")):
            try:
                # Generate sample data
                sample_data = self._generate_single_sample(mfg_params, i, use_creep)
                
                # Store in dataset
                self.dataset['warp_data'].append(sample_data['warp_data'])
                self.dataset['stress_data'].append(sample_data['stress_data'])
                self.dataset['parameters'].append(sample_data['parameters'])
                self.dataset['metadata'].append(sample_data['metadata'])
                
                # Save intermediate results if requested
                if save_intermediate and (i + 1) % 50 == 0:
                    self._save_intermediate_results(i + 1)
                
            except Exception as e:
                print(f"\nError processing sample {i}: {str(e)}")
                continue
        
        # Finalize dataset
        print("\n3. Finalizing dataset...")
        self._finalize_dataset()
        
        # Save complete dataset
        print("\n4. Saving complete dataset...")
        self._save_complete_dataset()
        
        print(f"\nDataset generation completed!")
        print(f"Generated {len(self.dataset['warp_data'])} samples")
        print(f"Dataset saved to: {self.output_dir}")
    
    def _generate_single_sample(self, mfg_params, sample_id: int, use_creep: bool) -> dict:
        """Generate a single sample with warp and stress data"""
        
        # Create mesh with current parameters
        mesh_params = self._create_mesh_parameters(mfg_params)
        mesh_gen = SOFCMeshGenerator(mesh_params)
        mesh = mesh_gen.generate_mesh()
        
        # Create material properties with variations
        material_props = self._create_material_properties(mfg_params)
        
        # Create FEA solver
        solver = FEASolver(mesh, material_props, self.creep_model if use_creep else None)
        
        # Create temperature field
        temperature_field = self._create_temperature_field(mesh, mfg_params)
        
        # Run simulation
        results = solver.solve_thermo_mechanical(
            temperature_field=temperature_field,
            boundary_conditions={'bottom_fixed_z': 0.0},
            assembly_pressure=mfg_params.values['assembly_pressure'],
            use_creep=use_creep,
            time_steps=[0, 3600, 7200] if use_creep else None  # 0, 1h, 2h
        )
        
        # Analyze warp
        warp_analyzer = WarpAnalyzer(mesh, results)
        top_warp = warp_analyzer.analyze_surface_warp('top')
        bottom_warp = warp_analyzer.analyze_surface_warp('bottom')
        
        # Generate height maps
        top_height_map, top_x, top_y = top_warp.get_height_map()
        bottom_height_map, bottom_x, bottom_y = bottom_warp.get_height_map()
        
        # Analyze stress
        stress_analyzer = StressAnalyzer(mesh, results)
        electrolyte_stress = stress_analyzer.analyze_stress_field('electrolyte')
        anode_stress = stress_analyzer.analyze_stress_field('anode')
        cathode_stress = stress_analyzer.analyze_stress_field('cathode')
        
        # Generate stress maps
        electrolyte_stress_maps = stress_analyzer.generate_stress_maps('electrolyte')
        
        # Compile sample data
        sample_data = {
            'warp_data': {
                'top_height_map': top_height_map,
                'top_height_x': top_x,
                'top_height_y': top_y,
                'bottom_height_map': bottom_height_map,
                'bottom_height_x': bottom_x,
                'bottom_height_y': bottom_y,
                'top_warp_metrics': warp_analyzer.analyze_warp_patterns('top'),
                'bottom_warp_metrics': warp_analyzer.analyze_warp_patterns('bottom'),
                'top_point_cloud': top_warp.get_point_cloud(),
                'bottom_point_cloud': bottom_warp.get_point_cloud()
            },
            'stress_data': {
                'electrolyte_stress_tensor': electrolyte_stress.stress_tensor,
                'electrolyte_von_mises': electrolyte_stress.von_mises_stress,
                'electrolyte_principal_stresses': electrolyte_stress.principal_stresses,
                'electrolyte_stress_maps': electrolyte_stress_maps,
                'anode_stress_tensor': anode_stress.stress_tensor,
                'cathode_stress_tensor': cathode_stress.stress_tensor,
                'stress_metrics': {
                    'electrolyte': stress_analyzer.analyze_stress_distribution('electrolyte'),
                    'anode': stress_analyzer.analyze_stress_distribution('anode'),
                    'cathode': stress_analyzer.analyze_stress_distribution('cathode')
                }
            },
            'parameters': mfg_params.to_dict(),
            'metadata': {
                'sample_id': sample_id,
                'mesh_stats': mesh_gen.get_mesh_statistics(),
                'simulation_time': results.simulation_time,
                'convergence_info': results.convergence_info,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        return sample_data
    
    def _create_mesh_parameters(self, mfg_params) -> MeshParameters:
        """Create mesh parameters from manufacturing parameters"""
        return MeshParameters(
            cell_length=mfg_params.values['cell_length'],
            cell_width=mfg_params.values['cell_width'],
            electrolyte_thickness=mfg_params.values['electrolyte_thickness'],
            anode_thickness=mfg_params.values['anode_thickness'],
            cathode_thickness=mfg_params.values['cathode_thickness'],
            interconnect_thickness=mfg_params.values['interconnect_thickness']
        )
    
    def _create_material_properties(self, mfg_params) -> dict:
        """Create material properties with variations"""
        # Base temperature for property evaluation
        operating_temp = mfg_params.values['operating_temperature']
        
        # Get base properties
        electrolyte_base = self.materials.get_all_properties('8YSZ', operating_temp)
        anode_base = self.materials.get_all_properties('NiYSZ', operating_temp)
        cathode_base = self.materials.get_all_properties('LSM', operating_temp)
        interconnect_base = self.materials.get_all_properties('Crofer22APU', operating_temp)
        
        # Apply variations
        def apply_variation(base_props, variation_key, property_key):
            if variation_key in mfg_params.values:
                variation = mfg_params.values[variation_key] / 100.0  # Convert % to decimal
                base_props[property_key] *= (1.0 + variation)
            return base_props
        
        # Electrolyte variations
        electrolyte_props = apply_variation(electrolyte_base.copy(), 
                                          'electrolyte_youngs_modulus_variation', 'youngs_modulus')
        electrolyte_props = apply_variation(electrolyte_props, 
                                          'electrolyte_cte_variation', 'cte')
        
        # Anode variations
        anode_props = apply_variation(anode_base.copy(), 
                                    'anode_youngs_modulus_variation', 'youngs_modulus')
        anode_props = apply_variation(anode_props, 
                                    'anode_cte_variation', 'cte')
        
        # Cathode variations
        cathode_props = apply_variation(cathode_base.copy(), 
                                      'cathode_youngs_modulus_variation', 'youngs_modulus')
        cathode_props = apply_variation(cathode_props, 
                                      'cathode_cte_variation', 'cte')
        
        return {
            'electrolyte': electrolyte_props,
            'anode': anode_props,
            'cathode': cathode_props,
            'interconnect': interconnect_base
        }
    
    def _create_temperature_field(self, mesh, mfg_params) -> np.ndarray:
        """Create temperature field with thermal gradients"""
        nodes = mesh['nodes']
        n_nodes = len(nodes)
        
        # Base operating temperature
        base_temp = mfg_params.values['operating_temperature']
        
        # Create thermal gradient
        max_gradient = mfg_params.values['thermal_gradient']
        
        # Simple linear gradient from center to edges
        center_x = np.mean(nodes[:, 0])
        center_y = np.mean(nodes[:, 1])
        
        # Distance from center
        distances = np.sqrt((nodes[:, 0] - center_x)**2 + (nodes[:, 1] - center_y)**2)
        max_distance = np.max(distances)
        
        # Temperature variation
        temp_variation = max_gradient * (1.0 - distances / max_distance)
        temperature_field = base_temp + temp_variation
        
        return temperature_field
    
    def _save_intermediate_results(self, n_completed: int):
        """Save intermediate results"""
        intermediate_file = os.path.join(self.output_dir, f'intermediate_results_{n_completed}.npz')
        
        # Save current progress
        np.savez(intermediate_file,
                warp_data=self.dataset['warp_data'],
                stress_data=self.dataset['stress_data'],
                parameters=self.dataset['parameters'],
                metadata=self.dataset['metadata'],
                n_completed=n_completed)
        
        print(f"Intermediate results saved: {n_completed} samples completed")
    
    def _finalize_dataset(self):
        """Finalize dataset with summary statistics"""
        n_samples = len(self.dataset['warp_data'])
        
        # Compute dataset statistics
        dataset_stats = {
            'n_samples': n_samples,
            'generation_time': datetime.now().isoformat(),
            'random_seed': self.random_seed,
            'warp_statistics': self._compute_warp_statistics(),
            'stress_statistics': self._compute_stress_statistics(),
            'parameter_statistics': self._compute_parameter_statistics()
        }
        
        self.dataset['dataset_stats'] = dataset_stats
    
    def _compute_warp_statistics(self) -> dict:
        """Compute warp field statistics across all samples"""
        if not self.dataset['warp_data']:
            return {}
        
        max_warps = []
        rms_warps = []
        
        for sample in self.dataset['warp_data']:
            max_warps.append(sample['top_warp_metrics']['max_warp'])
            rms_warps.append(sample['top_warp_metrics']['rms_warp'])
        
        return {
            'max_warp_mean': np.mean(max_warps),
            'max_warp_std': np.std(max_warps),
            'max_warp_min': np.min(max_warps),
            'max_warp_max': np.max(max_warps),
            'rms_warp_mean': np.mean(rms_warps),
            'rms_warp_std': np.std(rms_warps)
        }
    
    def _compute_stress_statistics(self) -> dict:
        """Compute stress field statistics across all samples"""
        if not self.dataset['stress_data']:
            return {}
        
        max_von_mises = []
        max_principal = []
        
        for sample in self.dataset['stress_data']:
            max_von_mises.append(np.max(sample['electrolyte_von_mises']))
            max_principal.append(np.max(sample['electrolyte_principal_stresses'][:, 0]))
        
        return {
            'max_von_mises_mean': np.mean(max_von_mises),
            'max_von_mises_std': np.std(max_von_mises),
            'max_principal_mean': np.mean(max_principal),
            'max_principal_std': np.std(max_principal)
        }
    
    def _compute_parameter_statistics(self) -> dict:
        """Compute parameter statistics across all samples"""
        if not self.dataset['parameters']:
            return {}
        
        # Convert to DataFrame for easy statistics
        param_df = pd.DataFrame(self.dataset['parameters'])
        
        stats = {}
        for col in param_df.columns:
            stats[f'{col}_mean'] = float(param_df[col].mean())
            stats[f'{col}_std'] = float(param_df[col].std())
            stats[f'{col}_min'] = float(param_df[col].min())
            stats[f'{col}_max'] = float(param_df[col].max())
        
        return stats
    
    def _save_complete_dataset(self):
        """Save complete dataset in multiple formats"""
        # Save as HDF5 (recommended for large datasets)
        h5_file = os.path.join(self.output_dir, 'sofc_dataset.h5')
        with h5py.File(h5_file, 'w') as f:
            # Create groups
            warp_group = f.create_group('warp_data')
            stress_group = f.create_group('stress_data')
            param_group = f.create_group('parameters')
            meta_group = f.create_group('metadata')
            
            # Save warp data
            for i, sample in enumerate(self.dataset['warp_data']):
                sample_group = warp_group.create_group(f'sample_{i:04d}')
                sample_group.create_dataset('top_height_map', data=sample['top_height_map'])
                sample_group.create_dataset('top_height_x', data=sample['top_height_x'])
                sample_group.create_dataset('top_height_y', data=sample['top_height_y'])
                sample_group.create_dataset('bottom_height_map', data=sample['bottom_height_map'])
                sample_group.create_dataset('top_point_cloud', data=sample['top_point_cloud'])
                sample_group.create_dataset('bottom_point_cloud', data=sample['bottom_point_cloud'])
                
                # Save metrics as attributes
                for key, value in sample['top_warp_metrics'].items():
                    sample_group.attrs[f'top_{key}'] = value
                for key, value in sample['bottom_warp_metrics'].items():
                    sample_group.attrs[f'bottom_{key}'] = value
            
            # Save stress data
            for i, sample in enumerate(self.dataset['stress_data']):
                sample_group = stress_group.create_group(f'sample_{i:04d}')
                sample_group.create_dataset('electrolyte_stress_tensor', data=sample['electrolyte_stress_tensor'])
                sample_group.create_dataset('electrolyte_von_mises', data=sample['electrolyte_von_mises'])
                sample_group.create_dataset('electrolyte_principal_stresses', data=sample['electrolyte_principal_stresses'])
                sample_group.create_dataset('anode_stress_tensor', data=sample['anode_stress_tensor'])
                sample_group.create_dataset('cathode_stress_tensor', data=sample['cathode_stress_tensor'])
                
                # Save stress maps
                stress_maps = sample['electrolyte_stress_maps']
                sample_group.create_dataset('von_mises_map', data=stress_maps['von_mises_map'])
                sample_group.create_dataset('principal_1_map', data=stress_maps['principal_1_map'])
                sample_group.create_dataset('x_grid', data=stress_maps['x_grid'])
                sample_group.create_dataset('y_grid', data=stress_maps['y_grid'])
            
            # Save parameters
            param_df = pd.DataFrame(self.dataset['parameters'])
            param_df.to_hdf(h5_file, 'parameters/dataframe', mode='a')
            
            # Save metadata
            for i, meta in enumerate(self.dataset['metadata']):
                meta_group.create_group(f'sample_{i:04d}')
                for key, value in meta.items():
                    if isinstance(value, (int, float, str)):
                        meta_group[f'sample_{i:04d}'].attrs[key] = value
        
        # Save as JSON (for metadata and small data)
        json_file = os.path.join(self.output_dir, 'dataset_metadata.json')
        with open(json_file, 'w') as f:
            json.dump(self.dataset['dataset_stats'], f, indent=2)
        
        # Save parameters as CSV
        param_file = os.path.join(self.output_dir, 'parameters.csv')
        param_df = pd.DataFrame(self.dataset['parameters'])
        param_df.to_csv(param_file, index=False)
        
        print(f"Dataset saved in multiple formats:")
        print(f"  - HDF5: {h5_file}")
        print(f"  - JSON metadata: {json_file}")
        print(f"  - Parameters CSV: {param_file}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate SOFC synthetic dataset')
    parser.add_argument('--samples', type=int, default=100, 
                       help='Number of samples to generate (default: 100)')
    parser.add_argument('--output_dir', type=str, default='./dataset',
                       help='Output directory (default: ./dataset)')
    parser.add_argument('--strategy', type=str, default='lhs',
                       choices=['lhs', 'sobol', 'random', 'halton', 'stratified'],
                       help='DOE sampling strategy (default: lhs)')
    parser.add_argument('--no_creep', action='store_true',
                       help='Disable creep effects')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no_intermediate', action='store_true',
                       help='Disable intermediate saving')
    
    args = parser.parse_args()
    
    # Create dataset generator
    generator = SOFCDatasetGenerator(
        output_dir=args.output_dir,
        random_seed=args.seed
    )
    
    # Generate dataset
    generator.generate_dataset(
        n_samples=args.samples,
        strategy=args.strategy,
        use_creep=not args.no_creep,
        save_intermediate=not args.no_intermediate
    )


if __name__ == "__main__":
    main()