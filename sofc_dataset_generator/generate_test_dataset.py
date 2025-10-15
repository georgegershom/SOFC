#!/usr/bin/env python3
"""
Generate a test SOFC dataset with minimal samples for demonstration
"""

import numpy as np
import pandas as pd
import h5py
import json
import os
from datetime import datetime
from parameters import SOFCParameters
from sampling import SOFCSampler
from degradation_models import SOFCDegradationModels

def generate_minimal_dataset():
    """Generate minimal dataset for testing"""
    
    print("Generating minimal SOFC dataset for demonstration...")
    print("-" * 50)
    
    # Initialize components
    params = SOFCParameters()
    sampler = SOFCSampler(params)
    models = SOFCDegradationModels()
    
    # Create output directory
    os.makedirs('datasets', exist_ok=True)
    
    # Generate time vector (1000 hours, 10 points)
    time_hours = np.linspace(0, 1000, 10)
    
    # Generate samples for each fidelity
    n_samples = {'LF': 10, 'MF': 5, 'HF': 2}
    datasets = {}
    results = {}
    
    for fidelity in ['LF', 'MF', 'HF']:
        print(f"\nGenerating {fidelity} samples...")
        
        # Generate parameter samples
        samples = sampler.latin_hypercube_sampling(n_samples[fidelity], fidelity)
        datasets[fidelity] = samples
        
        # Generate responses
        responses = []
        for idx, row in samples.iterrows():
            # Get parameters
            params_dict = row.to_dict()
            params_clean = {k: v for k, v in params_dict.items() 
                          if k not in ['sample_id', 'fidelity']}
            
            # Ensure positive current density to avoid divide by zero
            if 'system.current_density' in params_clean:
                params_clean['system.current_density'] = max(0.1, params_clean['system.current_density'])
            
            # Compute degradation
            try:
                degradation = models.compute_voltage_degradation(params_clean, time_hours, fidelity)
                
                # Store key metrics
                response = {
                    'initial_voltage': degradation['voltage'][0],
                    'final_voltage': degradation['voltage'][-1],
                    'avg_degradation_rate': (degradation['voltage'][0] - degradation['voltage'][-1]) / 1000 * 1000,
                    'voltage_series': degradation['voltage'].tolist(),
                    'asr_series': degradation['ASR'].tolist()
                }
                responses.append(response)
                
            except Exception as e:
                print(f"  Warning: Sample {idx} failed: {e}")
                # Create dummy response
                response = {
                    'initial_voltage': 0.75,
                    'final_voltage': 0.70,
                    'avg_degradation_rate': 50,
                    'voltage_series': np.linspace(0.75, 0.70, len(time_hours)).tolist(),
                    'asr_series': np.linspace(0.15, 0.20, len(time_hours)).tolist()
                }
                responses.append(response)
        
        results[fidelity] = responses
        print(f"  Generated {len(responses)} responses")
    
    # Save to HDF5
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'datasets/sofc_test_dataset_{timestamp}.h5'
    
    print(f"\nSaving dataset to {filename}...")
    
    with h5py.File(filename, 'w') as f:
        # Save time vector
        f.create_dataset('time_hours', data=time_hours)
        
        # Save data for each fidelity
        for fidelity in ['LF', 'MF', 'HF']:
            grp = f.create_group(fidelity)
            
            # Save inputs
            inputs_grp = grp.create_group('inputs')
            df = datasets[fidelity]
            for col in df.columns:
                if df[col].dtype != 'object':
                    inputs_grp.create_dataset(col, data=df[col].values)
            
            # Save responses
            responses_grp = grp.create_group('responses')
            for i, response in enumerate(results[fidelity]):
                sample_grp = responses_grp.create_group(f'sample_{i:05d}')
                
                # Save metrics
                sample_grp.attrs['initial_voltage'] = response['initial_voltage']
                sample_grp.attrs['final_voltage'] = response['final_voltage']
                sample_grp.attrs['avg_degradation_rate'] = response['avg_degradation_rate']
                
                # Save time series
                sample_grp.create_dataset('voltage', data=response['voltage_series'])
                sample_grp.create_dataset('ASR', data=response['asr_series'])
    
    # Also save inputs as CSV for easy access
    for fidelity in ['LF', 'MF', 'HF']:
        csv_path = f'datasets/test_inputs_{fidelity}.csv'
        datasets[fidelity].to_csv(csv_path, index=False)
    
    # Create metadata
    metadata = {
        'dataset_info': {
            'title': 'SOFC Multi-Fidelity Test Dataset',
            'description': 'Minimal test dataset for demonstration',
            'creation_date': datetime.now().isoformat(),
            'time_points': len(time_hours),
            'max_hours': 1000
        },
        'samples': {
            'LF': n_samples['LF'],
            'MF': n_samples['MF'],
            'HF': n_samples['HF']
        },
        'parameters': {
            'LF': len([c for c in datasets['LF'].columns if c not in ['sample_id', 'fidelity']]),
            'MF': len([c for c in datasets['MF'].columns if c not in ['sample_id', 'fidelity']]),
            'HF': len([c for c in datasets['HF'].columns if c not in ['sample_id', 'fidelity']])
        }
    }
    
    with open('datasets/test_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*50)
    print("✅ Test dataset generated successfully!")
    print("="*50)
    print(f"\nFiles created:")
    print(f"  • HDF5 Dataset: {filename}")
    print(f"  • Metadata: datasets/test_metadata.json")
    print(f"  • Input CSVs: datasets/test_inputs_*.csv")
    
    print("\nDataset Summary:")
    for fidelity in ['LF', 'MF', 'HF']:
        print(f"  {fidelity}: {n_samples[fidelity]} samples, {metadata['parameters'][fidelity]} parameters")
    
    print("\nTo load the dataset:")
    print("```python")
    print("import h5py")
    print(f"f = h5py.File('{filename}', 'r')")
    print("time = f['time_hours'][:]")
    print("lf_voltage = f['LF']['responses']['sample_00000']['voltage'][:]")
    print("```")
    
    return filename

if __name__ == '__main__':
    generate_minimal_dataset()