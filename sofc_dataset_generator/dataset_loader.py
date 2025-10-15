#!/usr/bin/env python3
"""
Example script to load and explore the SOFC dataset
"""

import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_sofc_dataset(filepath):
    """Load and explore SOFC dataset"""
    
    print("="*60)
    print("SOFC MULTI-FIDELITY DATASET LOADER")
    print("="*60)
    
    with h5py.File(filepath, 'r') as f:
        print(f"\nDataset: {filepath}")
        print(f"Top-level groups: {list(f.keys())}")
        
        # Load time vector
        time = f['time_hours'][:]
        print(f"\nTime points: {len(time)} (0 to {time[-1]:.0f} hours)")
        
        # Explore each fidelity level
        for fidelity in ['LF', 'MF', 'HF']:
            if fidelity in f:
                print(f"\n{fidelity} Fidelity Level:")
                print("-"*40)
                
                # Count samples
                n_samples = len(f[fidelity]['responses'].keys())
                print(f"  Number of samples: {n_samples}")
                
                # Count parameters
                n_params = len(f[fidelity]['inputs'].keys())
                print(f"  Number of input parameters: {n_params}")
                
                # Show sample parameters
                print(f"  Sample parameters:")
                param_names = list(f[fidelity]['inputs'].keys())[:5]
                for param in param_names:
                    data = f[fidelity]['inputs'][param][:]
                    print(f"    • {param}: [{np.min(data):.3f}, {np.max(data):.3f}]")
                
                # Load first sample response
                if n_samples > 0:
                    sample_0 = f[fidelity]['responses']['sample_00000']
                    
                    print(f"\n  First sample response:")
                    if 'voltage' in sample_0:
                        voltage = sample_0['voltage'][:]
                        print(f"    • Initial voltage: {voltage[0]:.3f} V")
                        print(f"    • Final voltage: {voltage[-1]:.3f} V")
                        print(f"    • Degradation: {(voltage[0]-voltage[-1])*1000:.1f} mV")
                    
                    # Show attributes
                    if sample_0.attrs:
                        print(f"    • Attributes: {list(sample_0.attrs.keys())}")
        
        print("\n" + "="*60)
        
        # Create a simple plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
            if fidelity in f:
                # Plot first 3 samples
                n_plot = min(3, len(f[fidelity]['responses'].keys()))
                for i in range(n_plot):
                    sample_key = f'sample_{i:05d}'
                    if sample_key in f[fidelity]['responses']:
                        if 'voltage' in f[fidelity]['responses'][sample_key]:
                            voltage = f[fidelity]['responses'][sample_key]['voltage'][:]
                            axes[idx].plot(time, voltage, label=f'Sample {i}', linewidth=2)
                
                axes[idx].set_xlabel('Time (hours)')
                axes[idx].set_ylabel('Voltage (V)')
                axes[idx].set_title(f'{fidelity} Fidelity')
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()
        
        plt.suptitle('SOFC Voltage Degradation - Multi-Fidelity Comparison')
        plt.tight_layout()
        plt.savefig('dataset_preview.png', dpi=150)
        print(f"Preview plot saved as 'dataset_preview.png'")
        
    return time

def load_input_parameters(csv_path):
    """Load input parameters from CSV"""
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} samples with {len(df.columns)} columns")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nBasic statistics:")
    print(df.describe())
    
    return df

if __name__ == '__main__':
    import glob
    import os
    
    # Find the most recent dataset
    datasets = glob.glob('datasets/sofc_test_dataset_*.h5')
    if datasets:
        latest_dataset = sorted(datasets)[-1]
        print(f"Loading dataset: {latest_dataset}")
        
        # Load HDF5 dataset
        time = load_sofc_dataset(latest_dataset)
        
        # Load CSV inputs
        print("\n" + "="*60)
        print("LOADING INPUT PARAMETERS FROM CSV")
        print("="*60)
        
        if os.path.exists('datasets/test_inputs_LF.csv'):
            print("\nLow-Fidelity Inputs:")
            df_lf = load_input_parameters('datasets/test_inputs_LF.csv')
    else:
        print("No dataset found. Please run generate_test_dataset.py first.")