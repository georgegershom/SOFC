"""
Usage examples for SOFC multi-fidelity dataset
"""

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from parameters import SOFCParameters
from sampling import SOFCSampler
from degradation_models import SOFCDegradationModels

def example_1_explore_parameters():
    """Example 1: Explore parameter space"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Exploring SOFC Parameter Space")
    print("="*60)
    
    # Initialize parameters
    params = SOFCParameters()
    
    # Get parameter info for each fidelity level
    for fidelity in ['LF', 'MF', 'HF']:
        param_info = params.get_parameter_info(fidelity)
        print(f"\n{fidelity} Fidelity: {len(param_info)} parameters")
        
        # Show first 5 parameters
        print("Sample parameters:")
        for p in param_info[:5]:
            print(f"  - {p['name']:30s} [{p['min']:.2e} - {p['max']:.2e}] {p['unit']}")
    
    # Get nominal parameter vector
    nominal_hf = params.get_parameter_vector('HF')
    print(f"\nTotal HF parameters: {len(nominal_hf)}")

def example_2_generate_samples():
    """Example 2: Generate parameter samples using different methods"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Generating Parameter Samples")
    print("="*60)
    
    # Initialize sampler
    params = SOFCParameters()
    sampler = SOFCSampler(params)
    
    # Generate Latin Hypercube samples
    print("\nGenerating Latin Hypercube samples...")
    lhs_samples = sampler.latin_hypercube_sampling(n_samples=100, fidelity='MF')
    print(f"Generated {len(lhs_samples)} LHS samples with {len(lhs_samples.columns)} parameters")
    
    # Generate Sobol sequence samples
    print("\nGenerating Sobol sequence samples...")
    sobol_samples = sampler.sobol_sequence_sampling(n_samples=100, fidelity='MF')
    print(f"Generated {len(sobol_samples)} Sobol samples")
    
    # Show sample statistics
    print("\nSample statistics for key parameters:")
    key_params = ['system.temperature', 'system.current_density', 'system.fuel_utilization']
    for param in key_params:
        if param in lhs_samples.columns:
            print(f"\n{param}:")
            print(f"  Mean: {lhs_samples[param].mean():.3f}")
            print(f"  Std:  {lhs_samples[param].std():.3f}")
            print(f"  Range: [{lhs_samples[param].min():.3f}, {lhs_samples[param].max():.3f}]")

def example_3_compute_degradation():
    """Example 3: Compute degradation for a single operating point"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Computing Degradation Response")
    print("="*60)
    
    # Initialize models
    models = SOFCDegradationModels()
    params_obj = SOFCParameters()
    
    # Get nominal parameters
    params = params_obj.get_parameter_vector('HF')
    
    # Define time vector (10000 hours)
    time_hours = np.linspace(0, 10000, 100)
    
    # Compute degradation at different fidelities
    print("\nComputing degradation at different fidelity levels...")
    
    results = {}
    for fidelity in ['LF', 'MF', 'HF']:
        print(f"\n{fidelity} Fidelity:")
        degradation = models.compute_voltage_degradation(params, time_hours, fidelity)
        results[fidelity] = degradation
        
        # Print summary
        initial_voltage = degradation['voltage'][0]
        final_voltage = degradation['voltage'][-1]
        avg_degradation_rate = (initial_voltage - final_voltage) / 10000 * 1000  # mV/1000h
        
        print(f"  Initial voltage: {initial_voltage:.3f} V")
        print(f"  Final voltage: {final_voltage:.3f} V")
        print(f"  Average degradation rate: {avg_degradation_rate:.2f} mV/1000h")
        
        # Print available outputs
        print(f"  Available outputs: {list(degradation.keys())}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for fidelity, result in results.items():
        plt.plot(time_hours, result['voltage'], label=f'{fidelity} Fidelity', linewidth=2)
    
    plt.xlabel('Time (hours)')
    plt.ylabel('Voltage (V)')
    plt.title('Voltage Degradation: Multi-Fidelity Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('degradation_comparison.png', dpi=150)
    print("\nPlot saved as 'degradation_comparison.png'")

def example_4_multi_fidelity_dataset():
    """Example 4: Generate and explore multi-fidelity dataset"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Multi-Fidelity Dataset Generation")
    print("="*60)
    
    # Initialize components
    params = SOFCParameters()
    sampler = SOFCSampler(params)
    
    # Generate multi-fidelity samples with nested structure
    print("\nGenerating multi-fidelity samples (nested structure)...")
    datasets = sampler.multi_fidelity_sampling(
        n_lf=1000,
        n_mf=100,
        n_hf=10,
        method='lhs'
    )
    
    # Show dataset structure
    print("\nDataset structure:")
    for fidelity in ['LF', 'MF', 'HF']:
        df = datasets[fidelity]
        print(f"\n{fidelity} Fidelity:")
        print(f"  Samples: {len(df)}")
        print(f"  Parameters: {len([c for c in df.columns if c not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']])}")
        
        # Check parent relationships
        if 'parent_fidelity' in df.columns:
            parent_fids = df['parent_fidelity'].value_counts()
            if len(parent_fids) > 0:
                print(f"  Parent samples from: {parent_fids.to_dict()}")
    
    # Visualize parameter coverage
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Select two parameters for visualization
    param1 = 'system.temperature'
    param2 = 'system.current_density'
    
    for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
        df = datasets[fidelity]
        if param1 in df.columns and param2 in df.columns:
            axes[idx].scatter(df[param1], df[param2], alpha=0.6, s=30)
            axes[idx].set_xlabel('Temperature (K)')
            axes[idx].set_ylabel('Current Density (A/cm²)')
            axes[idx].set_title(f'{fidelity} Fidelity (n={len(df)})')
            axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Fidelity Parameter Space Coverage')
    plt.tight_layout()
    plt.savefig('multifidelity_coverage.png', dpi=150)
    print("\nPlot saved as 'multifidelity_coverage.png'")

def example_5_load_and_analyze():
    """Example 5: Load and analyze generated dataset"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Loading and Analyzing HDF5 Dataset")
    print("="*60)
    
    # This example assumes a dataset has been generated
    # For demonstration, we'll show the structure
    
    print("""
To load a generated dataset:

```python
import h5py
import numpy as np

# Open HDF5 file
with h5py.File('datasets/sofc_dataset.h5', 'r') as f:
    # Load time vector
    time = f['time_hours'][:]
    
    # Load LF data
    lf_inputs = {}
    for key in f['LF']['inputs'].keys():
        lf_inputs[key] = f['LF']['inputs'][key][:]
    
    # Load response for first sample
    sample_0 = f['LF']['responses']['sample_00000']
    voltage = sample_0['degradation']['voltage'][:]
    asr = sample_0['degradation']['ASR'][:]
    
    # For HF data with microstructure
    if 'HF' in f:
        microstructure = f['HF']['responses']['sample_00000']['microstructure'][:]
        print(f"Microstructure shape: {microstructure.shape}")
```
    """)

def example_6_thermal_stress():
    """Example 6: Compute thermal stress distribution"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Thermal Stress Analysis")
    print("="*60)
    
    # Initialize models and parameters
    models = SOFCDegradationModels()
    params_obj = SOFCParameters()
    params = params_obj.get_parameter_vector('HF')
    
    # Create temperature profile (thermal gradient)
    base_temp = params.get('system.temperature', 1023)
    temp_profile = np.linspace(base_temp - 100, base_temp + 100, 50)
    
    # Compute thermal stress
    print("\nComputing thermal stress distribution...")
    stress_results = models.compute_thermal_stress(params, temp_profile)
    
    # Print results
    print("\nThermal Stress Results:")
    for key, value in stress_results.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    Min: {np.min(value):.2e} Pa")
            print(f"    Max: {np.max(value):.2e} Pa")
            print(f"    Mean: {np.mean(value):.2e} Pa")
    
    # Plot stress distribution
    plt.figure(figsize=(10, 6))
    plt.plot(temp_profile - 273.15, stress_results['stress_anode'] / 1e6, label='Anode', linewidth=2)
    plt.plot(temp_profile - 273.15, stress_results['stress_cathode'] / 1e6, label='Cathode', linewidth=2)
    plt.plot(temp_profile - 273.15, stress_results['von_mises_stress'] / 1e6, label='Von Mises', linewidth=2, linestyle='--')
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Stress (MPa)')
    plt.title('Thermal Stress Distribution in SOFC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('thermal_stress.png', dpi=150)
    print("\nPlot saved as 'thermal_stress.png'")

def run_all_examples():
    """Run all examples"""
    examples = [
        example_1_explore_parameters,
        example_2_generate_samples,
        example_3_compute_degradation,
        example_4_multi_fidelity_dataset,
        example_5_load_and_analyze,
        example_6_thermal_stress
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
            continue

if __name__ == '__main__':
    print("=" * 70)
    print("SOFC MULTI-FIDELITY DATASET - USAGE EXAMPLES")
    print("=" * 70)
    
    run_all_examples()
    
    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)