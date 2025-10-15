"""
Main dataset generation pipeline for multi-fidelity SOFC modeling
"""

import numpy as np
import pandas as pd
import h5py
import json
import argparse
from tqdm import tqdm
from datetime import datetime
import os
from typing import Dict, List, Optional
from joblib import Parallel, delayed

from parameters import SOFCParameters
from sampling import SOFCSampler
from degradation_models import SOFCDegradationModels

class SOFCDatasetGenerator:
    """Generate comprehensive multi-fidelity SOFC dataset"""
    
    def __init__(self, output_dir: str = 'datasets'):
        self.parameters = SOFCParameters()
        self.sampler = SOFCSampler(self.parameters)
        self.models = SOFCDegradationModels()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_complete_dataset(self,
                                 n_lf: int = 10000,
                                 n_mf: int = 1000,
                                 n_hf: int = 100,
                                 time_points: int = 100,
                                 max_hours: float = 10000,
                                 n_jobs: int = -1) -> str:
        """
        Generate complete multi-fidelity dataset
        
        Args:
            n_lf: Number of low-fidelity samples
            n_mf: Number of medium-fidelity samples
            n_hf: Number of high-fidelity samples
            time_points: Number of time points for degradation
            max_hours: Maximum operation time in hours
            n_jobs: Number of parallel jobs (-1 for all cores)
        
        Returns:
            Path to generated dataset
        """
        print(f"Generating multi-fidelity SOFC dataset...")
        print(f"Samples: LF={n_lf}, MF={n_mf}, HF={n_hf}")
        print(f"Time points: {time_points} over {max_hours} hours")
        
        # Generate time vector
        time_hours = np.linspace(0, max_hours, time_points)
        
        # Generate multi-fidelity samples
        print("\n1. Generating parameter samples...")
        samples = self.sampler.multi_fidelity_sampling(n_lf, n_mf, n_hf, method='lhs')
        
        # Add noise to simulate measurement uncertainty
        for fidelity in samples:
            noise_level = {'LF': 0.05, 'MF': 0.03, 'HF': 0.01}[fidelity]
            samples[fidelity] = self.sampler.add_noise_to_samples(
                samples[fidelity], noise_level
            )
        
        # Generate responses for each fidelity level
        results = {}
        
        for fidelity in ['LF', 'MF', 'HF']:
            print(f"\n2. Generating {fidelity} responses...")
            df = samples[fidelity]
            
            # Process in parallel
            responses = Parallel(n_jobs=n_jobs)(
                delayed(self._generate_sample_response)(
                    row.to_dict(), time_hours, fidelity
                ) for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{fidelity} samples")
            )
            
            results[fidelity] = {
                'inputs': df,
                'responses': responses
            }
        
        # Save dataset
        print("\n3. Saving dataset...")
        dataset_path = self._save_dataset(results, time_hours)
        
        # Generate metadata
        metadata = self._generate_metadata(n_lf, n_mf, n_hf, time_points, max_hours)
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset generated successfully!")
        print(f"Location: {dataset_path}")
        print(f"Metadata: {metadata_path}")
        
        return dataset_path
    
    def _generate_sample_response(self,
                                 params: Dict,
                                 time_hours: np.ndarray,
                                 fidelity: str) -> Dict:
        """Generate response for a single sample"""
        
        # Remove non-parameter columns
        params_clean = {k: v for k, v in params.items() 
                       if k not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']}
        
        # Compute degradation
        degradation = self.models.compute_voltage_degradation(params_clean, time_hours, fidelity)
        
        # Compute IV curve at initial and final time
        current_range = np.linspace(0, 1.5, 50)
        iv_initial = self.models.compute_electrochemical_response(params_clean, current_range)
        
        # Modify parameters for aged cell
        params_aged = params_clean.copy()
        if fidelity in ['MF', 'HF']:
            # Update parameters based on degradation
            if 'anode_material.anode_porosity' in params_aged:
                params_aged['anode_material.anode_porosity'] *= 0.9
            if 'anode_material.ni_particle_size' in params_aged:
                params_aged['anode_material.ni_particle_size'] *= 1.5
        
        iv_final = self.models.compute_electrochemical_response(params_aged, current_range)
        
        # Compute thermal stress
        temp_profile = np.linspace(
            params_clean.get('system.temperature', 1023) - 50,
            params_clean.get('system.temperature', 1023) + 50,
            10
        )
        thermal_stress = self.models.compute_thermal_stress(params_clean, temp_profile)
        
        # For HF, add microstructure
        microstructure = None
        if fidelity == 'HF':
            microstructure = self.models.generate_synthetic_microstructure(
                params_clean, 
                grid_size=(20, 20, 20)
            )
        
        return {
            'degradation': degradation,
            'iv_initial': iv_initial,
            'iv_final': iv_final,
            'thermal_stress': thermal_stress,
            'microstructure': microstructure
        }
    
    def _save_dataset(self, 
                     results: Dict,
                     time_hours: np.ndarray) -> str:
        """Save dataset to HDF5 format"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sofc_dataset_{timestamp}.h5'
        filepath = os.path.join(self.output_dir, filename)
        
        with h5py.File(filepath, 'w') as f:
            # Save time vector
            f.create_dataset('time_hours', data=time_hours)
            
            # Save data for each fidelity level
            for fidelity in ['LF', 'MF', 'HF']:
                grp = f.create_group(fidelity)
                
                # Save inputs
                inputs_grp = grp.create_group('inputs')
                df_inputs = results[fidelity]['inputs']
                
                for col in df_inputs.columns:
                    if df_inputs[col].dtype != 'object':
                        inputs_grp.create_dataset(col, data=df_inputs[col].values)
                
                # Save responses
                responses_grp = grp.create_group('responses')
                responses = results[fidelity]['responses']
                
                # Organize response data
                for i, response in enumerate(responses):
                    sample_grp = responses_grp.create_group(f'sample_{i:05d}')
                    
                    # Save degradation data
                    deg_grp = sample_grp.create_group('degradation')
                    for key, value in response['degradation'].items():
                        if isinstance(value, np.ndarray):
                            deg_grp.create_dataset(key, data=value)
                    
                    # Save IV curves
                    iv_grp = sample_grp.create_group('iv_curves')
                    for key, value in response['iv_initial'].items():
                        if isinstance(value, np.ndarray):
                            iv_grp.create_dataset(f'initial_{key}', data=value)
                    for key, value in response['iv_final'].items():
                        if isinstance(value, np.ndarray):
                            iv_grp.create_dataset(f'final_{key}', data=value)
                    
                    # Save thermal stress
                    stress_grp = sample_grp.create_group('thermal_stress')
                    for key, value in response['thermal_stress'].items():
                        if isinstance(value, np.ndarray):
                            stress_grp.create_dataset(key, data=value)
                    
                    # Save microstructure for HF
                    if fidelity == 'HF' and response['microstructure'] is not None:
                        sample_grp.create_dataset('microstructure', 
                                                data=response['microstructure'],
                                                compression='gzip')
        
        # Also save as CSV for easy access
        for fidelity in ['LF', 'MF', 'HF']:
            csv_path = os.path.join(self.output_dir, f'inputs_{fidelity}.csv')
            results[fidelity]['inputs'].to_csv(csv_path, index=False)
        
        return filepath
    
    def _generate_metadata(self,
                          n_lf: int,
                          n_mf: int,
                          n_hf: int,
                          time_points: int,
                          max_hours: float) -> Dict:
        """Generate dataset metadata"""
        
        metadata = {
            'dataset_info': {
                'title': 'Multi-Fidelity Digital Twin for SOFCs Dataset',
                'description': 'Synthetic dataset for multi-scale modeling and deep learning',
                'version': '1.0',
                'creation_date': datetime.now().isoformat(),
                'author': 'SOFC Dataset Generator',
            },
            'dataset_statistics': {
                'n_samples': {
                    'LF': n_lf,
                    'MF': n_mf,
                    'HF': n_hf,
                    'total': n_lf + n_mf + n_hf
                },
                'time_points': time_points,
                'max_hours': max_hours,
            },
            'parameters': {
                'LF': self.parameters.get_parameter_info('LF'),
                'MF': self.parameters.get_parameter_info('MF'),
                'HF': self.parameters.get_parameter_info('HF'),
            },
            'fidelity_descriptions': {
                'LF': 'Low-fidelity lumped parameter models',
                'MF': 'Medium-fidelity 1D/2D models with degradation mechanisms',
                'HF': 'High-fidelity 3D models with microstructural evolution'
            },
            'response_variables': {
                'degradation': ['voltage', 'ASR', 'power_density', 'degradation_rate', 'efficiency'],
                'iv_curves': ['current_density', 'voltage', 'power_density'],
                'thermal_stress': ['stress_anode', 'stress_cathode', 'von_mises_stress'],
                'microstructure': ['3D_phase_distribution'] 
            }
        }
        
        return metadata

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Generate multi-fidelity SOFC dataset')
    parser.add_argument('--n_lf', type=int, default=10000,
                       help='Number of low-fidelity samples')
    parser.add_argument('--n_mf', type=int, default=1000,
                       help='Number of medium-fidelity samples')
    parser.add_argument('--n_hf', type=int, default=100,
                       help='Number of high-fidelity samples')
    parser.add_argument('--time_points', type=int, default=100,
                       help='Number of time points for degradation')
    parser.add_argument('--max_hours', type=float, default=10000,
                       help='Maximum operation time in hours')
    parser.add_argument('--output', type=str, default='datasets',
                       help='Output directory')
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of parallel jobs (-1 for all cores)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = SOFCDatasetGenerator(args.output)
    
    # Generate dataset
    dataset_path = generator.generate_complete_dataset(
        n_lf=args.n_lf,
        n_mf=args.n_mf,
        n_hf=args.n_hf,
        time_points=args.time_points,
        max_hours=args.max_hours,
        n_jobs=args.n_jobs
    )
    
    print(f"\nDataset generation complete!")
    print(f"Dataset saved to: {dataset_path}")

if __name__ == '__main__':
    main()