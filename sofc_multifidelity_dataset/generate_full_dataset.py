#!/usr/bin/env python3
"""
Main script to generate the complete SOFC Multi-Fidelity Dataset
for PhD thesis: Multi-Scale Modeling and Deep Learning for Predicting Thermo-Mechanical Degradation
"""

import os
import sys
import time
import yaml
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from generators.low_fidelity import generate_lf_dataset
from generators.mid_fidelity import generate_mf_dataset  
from generators.high_fidelity import generate_hf_dataset
from generators.experimental import generate_experimental_dataset
from visualization.visualize_data import SOFCDataVisualizer


class MultiFidelityDatasetGenerator:
    """Master class to orchestrate multi-fidelity dataset generation"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize generator with configuration"""
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Track generation progress
        self.generation_log = {
            'start_time': datetime.now(),
            'datasets': {},
            'errors': []
        }
    
    def generate_low_fidelity(self, n_samples: int = None):
        """Generate low-fidelity dataset"""
        print("\n" + "="*60)
        print("GENERATING LOW-FIDELITY DATASET")
        print("="*60)
        
        try:
            start_time = time.time()
            
            if n_samples is None:
                n_samples = self.config['dataset']['low_fidelity']['n_samples']
            
            df, filepath = generate_lf_dataset(self.config_path, n_samples)
            
            elapsed = time.time() - start_time
            
            self.generation_log['datasets']['low_fidelity'] = {
                'filepath': filepath,
                'n_samples': len(df),
                'n_features': len(df.columns),
                'generation_time': elapsed,
                'samples_per_second': len(df) / elapsed
            }
            
            print(f"\n✓ Low-fidelity dataset complete!")
            print(f"  - Samples: {len(df)}")
            print(f"  - Time: {elapsed:.1f} seconds")
            print(f"  - Rate: {len(df)/elapsed:.1f} samples/sec")
            
            return df, filepath
            
        except Exception as e:
            error_msg = f"Failed to generate LF dataset: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.generation_log['errors'].append(error_msg)
            return None, None
    
    def generate_mid_fidelity(self, n_samples: int = None, lf_path: str = None):
        """Generate mid-fidelity dataset"""
        print("\n" + "="*60)
        print("GENERATING MID-FIDELITY DATASET")
        print("="*60)
        
        try:
            start_time = time.time()
            
            if n_samples is None:
                n_samples = self.config['dataset']['mid_fidelity']['n_samples']
            
            if lf_path is None:
                lf_path = os.path.join(self.data_dir, 'lf_dataset.h5')
            
            df, fields, filepath = generate_mf_dataset(self.config_path, n_samples, lf_path)
            
            elapsed = time.time() - start_time
            
            self.generation_log['datasets']['mid_fidelity'] = {
                'filepath': filepath,
                'n_samples': len(df),
                'n_features': len(df.columns),
                'n_2d_fields': len(fields),
                'generation_time': elapsed,
                'samples_per_second': len(df) / elapsed
            }
            
            print(f"\n✓ Mid-fidelity dataset complete!")
            print(f"  - Samples: {len(df)}")
            print(f"  - 2D Fields: {len(fields)} types")
            print(f"  - Time: {elapsed:.1f} seconds")
            print(f"  - Rate: {len(df)/elapsed:.2f} samples/sec")
            
            return df, fields, filepath
            
        except Exception as e:
            error_msg = f"Failed to generate MF dataset: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.generation_log['errors'].append(error_msg)
            return None, None, None
    
    def generate_high_fidelity(self, n_samples: int = None, mf_path: str = None):
        """Generate high-fidelity dataset"""
        print("\n" + "="*60)
        print("GENERATING HIGH-FIDELITY DATASET")
        print("="*60)
        print("NOTE: This will take significant time due to 3D simulations...")
        
        try:
            start_time = time.time()
            
            if n_samples is None:
                n_samples = self.config['dataset']['high_fidelity']['n_samples']
            
            if mf_path is None:
                mf_path = os.path.join(self.data_dir, 'mf_dataset.h5')
            
            df, fields, filepath = generate_hf_dataset(self.config_path, n_samples, mf_path)
            
            elapsed = time.time() - start_time
            
            self.generation_log['datasets']['high_fidelity'] = {
                'filepath': filepath,
                'n_samples': len(df),
                'n_features': len(df.columns),
                'n_3d_fields': len(fields[0]) if fields else 0,
                'generation_time': elapsed,
                'samples_per_second': len(df) / elapsed if elapsed > 0 else 0
            }
            
            print(f"\n✓ High-fidelity dataset complete!")
            print(f"  - Samples: {len(df)}")
            print(f"  - 3D Fields: {len(fields[0]) if fields else 0} types")
            print(f"  - Time: {elapsed:.1f} seconds")
            print(f"  - Rate: {elapsed/len(df) if len(df) > 0 else 0:.1f} sec/sample")
            
            return df, fields, filepath
            
        except Exception as e:
            error_msg = f"Failed to generate HF dataset: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.generation_log['errors'].append(error_msg)
            return None, None, None
    
    def generate_experimental(self, n_samples: int = None):
        """Generate experimental dataset"""
        print("\n" + "="*60)
        print("GENERATING EXPERIMENTAL DATASET")
        print("="*60)
        
        try:
            start_time = time.time()
            
            if n_samples is None:
                n_samples = self.config['dataset']['experimental']['n_samples']
            
            df, exp_data, filepath = generate_experimental_dataset(self.config_path, n_samples)
            
            elapsed = time.time() - start_time
            
            self.generation_log['datasets']['experimental'] = {
                'filepath': filepath,
                'n_samples': len(df),
                'n_measurements': len(exp_data),
                'measurement_types': list(exp_data.keys()),
                'generation_time': elapsed,
                'samples_per_second': len(df) / elapsed
            }
            
            print(f"\n✓ Experimental dataset complete!")
            print(f"  - Samples: {len(df)}")
            print(f"  - Measurement types: {len(exp_data)}")
            print(f"  - Time: {elapsed:.1f} seconds")
            print(f"  - Rate: {len(df)/elapsed:.2f} samples/sec")
            
            return df, exp_data, filepath
            
        except Exception as e:
            error_msg = f"Failed to generate experimental dataset: {str(e)}"
            print(f"\n✗ {error_msg}")
            self.generation_log['errors'].append(error_msg)
            return None, None, None
    
    def generate_all(self, scale_factor: float = 1.0, visualize: bool = True):
        """Generate all datasets with optional scaling"""
        
        print("\n" + "="*70)
        print("SOFC MULTI-FIDELITY DATASET GENERATION")
        print("PhD Thesis: Multi-Scale Modeling & Deep Learning for SOFC Degradation")
        print("="*70)
        print(f"\nConfiguration: {self.config_path}")
        print(f"Scale factor: {scale_factor}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Calculate sample sizes
        n_lf = int(self.config['dataset']['low_fidelity']['n_samples'] * scale_factor)
        n_mf = int(self.config['dataset']['mid_fidelity']['n_samples'] * scale_factor)
        n_hf = int(self.config['dataset']['high_fidelity']['n_samples'] * scale_factor)
        n_exp = int(self.config['dataset']['experimental']['n_samples'] * scale_factor)
        
        print(f"\nPlanned dataset sizes:")
        print(f"  - Low Fidelity: {n_lf} samples")
        print(f"  - Mid Fidelity: {n_mf} samples")
        print(f"  - High Fidelity: {n_hf} samples")
        print(f"  - Experimental: {n_exp} samples")
        
        total_start = time.time()
        
        # Generate datasets in sequence
        # Low Fidelity
        lf_df, lf_path = self.generate_low_fidelity(n_lf)
        
        # Mid Fidelity (uses LF for intelligent sampling)
        mf_df, mf_fields, mf_path = self.generate_mid_fidelity(n_mf, lf_path)
        
        # High Fidelity (uses MF for critical point selection)
        hf_df, hf_fields, hf_path = self.generate_high_fidelity(n_hf, mf_path)
        
        # Experimental (independent but follows similar conditions)
        exp_df, exp_data, exp_path = self.generate_experimental(n_exp)
        
        total_elapsed = time.time() - total_start
        
        # Summary
        print("\n" + "="*70)
        print("GENERATION COMPLETE")
        print("="*70)
        
        print(f"\nTotal generation time: {total_elapsed/60:.1f} minutes")
        
        print("\nDatasets generated:")
        for fidelity, info in self.generation_log['datasets'].items():
            print(f"\n{fidelity.upper().replace('_', ' ')}:")
            for key, value in info.items():
                if key != 'filepath':
                    print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        if self.generation_log['errors']:
            print(f"\n⚠ Errors encountered: {len(self.generation_log['errors'])}")
            for error in self.generation_log['errors']:
                print(f"  - {error}")
        
        # Save generation log
        import json
        log_path = os.path.join(self.data_dir, 'generation_log.json')
        self.generation_log['end_time'] = datetime.now()
        self.generation_log['total_time_seconds'] = total_elapsed
        
        with open(log_path, 'w') as f:
            json.dump(self.generation_log, f, indent=2, default=str)
        print(f"\nGeneration log saved to: {log_path}")
        
        # Visualize if requested
        if visualize:
            print("\n" + "="*70)
            print("GENERATING VISUALIZATIONS")
            print("="*70)
            
            visualizer = SOFCDataVisualizer(self.data_dir)
            report_dir = visualizer.generate_summary_report()
            print(f"\nVisualizations saved to: {report_dir}/")
        
        return self.generation_log
    
    def validate_datasets(self):
        """Validate generated datasets for consistency"""
        print("\n" + "="*60)
        print("VALIDATING DATASETS")
        print("="*60)
        
        import h5py
        import pandas as pd
        
        validations = []
        
        # Check file existence
        files_to_check = [
            ('Low Fidelity CSV', 'lf_dataset.csv'),
            ('Low Fidelity HDF5', 'lf_dataset.h5'),
            ('Mid Fidelity CSV', 'mf_dataset_scalars.csv'),
            ('Mid Fidelity HDF5', 'mf_dataset.h5'),
            ('High Fidelity CSV', 'hf_dataset_scalars.csv'),
            ('High Fidelity HDF5', 'hf_dataset.h5'),
            ('Experimental CSV', 'experimental_dataset_summary.csv'),
            ('Experimental HDF5', 'experimental_dataset.h5')
        ]
        
        for name, filename in files_to_check:
            filepath = os.path.join(self.data_dir, filename)
            exists = os.path.exists(filepath)
            size = os.path.getsize(filepath) / 1e6 if exists else 0
            
            validations.append({
                'dataset': name,
                'exists': exists,
                'size_mb': size
            })
            
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {'Found' if exists else 'Missing'} ({size:.1f} MB)")
        
        # Check data consistency
        print("\nChecking data consistency...")
        
        # Load CSVs and check key metrics
        try:
            lf_df = pd.read_csv(os.path.join(self.data_dir, 'lf_dataset.csv'))
            mf_df = pd.read_csv(os.path.join(self.data_dir, 'mf_dataset_scalars.csv'))
            
            # Check value ranges
            checks = [
                ('Voltage range', 0 < lf_df['voltage'].mean() < 1.5),
                ('Temperature range', 800 < lf_df['temperature_average'].mean() < 1300),
                ('Stress reasonable', lf_df['stress_maximum'].max() < 1e9),
                ('Lifetime positive', lf_df['estimated_lifetime_hours'].min() > 0)
            ]
            
            for check_name, passed in checks:
                status = "✓" if passed else "✗"
                print(f"  {status} {check_name}: {'PASS' if passed else 'FAIL'}")
            
        except Exception as e:
            print(f"  ⚠ Could not validate CSV data: {e}")
        
        return validations


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description='Generate SOFC Multi-Fidelity Dataset for PhD Research'
    )
    
    parser.add_argument(
        '--config', type=str, default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    parser.add_argument(
        '--scale', type=float, default=0.01,
        help='Scale factor for dataset sizes (default: 0.01 for quick testing, use 1.0 for full)'
    )
    
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--validate-only', action='store_true',
        help='Only validate existing datasets'
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = MultiFidelityDatasetGenerator(args.config)
    
    if args.validate_only:
        # Just validate existing datasets
        generator.validate_datasets()
    else:
        # Generate all datasets
        log = generator.generate_all(
            scale_factor=args.scale,
            visualize=not args.no_viz
        )
        
        # Validate after generation
        generator.validate_datasets()
        
        print("\n" + "="*70)
        print("DATASET GENERATION COMPLETE!")
        print("="*70)
        print("\nYour SOFC multi-fidelity dataset is ready for:")
        print("  ✓ Multi-fidelity machine learning")
        print("  ✓ Digital twin development")
        print("  ✓ Degradation prediction models")
        print("  ✓ PhD thesis research")
        print("\nGood luck with your research!")
        print("="*70)


if __name__ == "__main__":
    main()