"""
Example Usage of Multi-Fidelity SOFC Dataset
Demonstrates how to load and work with the dataset
"""

import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt

def example_1_load_csv_data():
    """Example 1: Load and explore CSV data"""
    print("\n" + "="*70)
    print("Example 1: Loading CSV Data")
    print("="*70)
    
    # Load low-fidelity data
    df_lf = pd.read_csv('sofc_multifidelity_dataset/phase1_LF/phase1_LF_complete.csv')
    
    print(f"\nDataset shape: {df_lf.shape}")
    print(f"Columns: {df_lf.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df_lf.head())
    
    print(f"\nBasic statistics:")
    print(df_lf[['operating_temperature_K', 'current_density_A_cm2', 
                 'voltage_V', 'avg_von_mises_stress_MPa']].describe())
    
    return df_lf


def example_2_load_hdf5_spatial():
    """Example 2: Load spatial fields from HDF5"""
    print("\n" + "="*70)
    print("Example 2: Loading Spatial Fields from HDF5")
    print("="*70)
    
    with h5py.File('sofc_multifidelity_dataset/phase3_HF/phase3_HF_complete.h5', 'r') as f:
        print(f"\nAvailable groups: {list(f.keys())}")
        
        # Load scalar outputs
        stress = f['scalar_outputs/max_stress_MPa'][:]
        damage = f['scalar_outputs/crack_initiation_indicator'][:]
        
        print(f"\nStress array shape: {stress.shape}")
        print(f"Damage array shape: {damage.shape}")
        print(f"Stress range: {stress.min():.2f} - {stress.max():.2f} MPa")
        
        # Load spatial field
        if 'spatial_fields_2D_slices' in f:
            print(f"\nSpatial fields available: {list(f['spatial_fields_2D_slices'].keys())}")
            
            # Load first sample
            T_field = f['spatial_fields_2D_slices/sample_0/temperature_K'][:]
            sigma_field = f['spatial_fields_2D_slices/sample_0/stress_MPa'][:]
            damage_field = f['spatial_fields_2D_slices/sample_0/damage_indicator'][:]
            
            print(f"\nTemperature field shape: {T_field.shape}")
            print(f"Temperature range: {T_field.min():.1f} - {T_field.max():.1f} K")
            
            return T_field, sigma_field, damage_field
    
    return None, None, None


def example_3_analyze_degradation():
    """Example 3: Analyze degradation trends"""
    print("\n" + "="*70)
    print("Example 3: Analyzing Degradation Trends")
    print("="*70)
    
    df = pd.read_csv('sofc_multifidelity_dataset/phase3_HF/phase3_HF_complete.csv')
    
    # Group by temperature
    temp_bins = pd.cut(df['operating_temperature_K'], bins=3, labels=['Low', 'Medium', 'High'])
    
    print("\nAverage degradation by temperature:")
    degradation_summary = df.groupby(temp_bins).agg({
        'Ni_coarsening_percent': 'mean',
        'TPB_loss_percent': 'mean',
        'crack_initiation_indicator': 'mean',
        'delamination_indicator': 'mean',
        'time_to_failure_hours': 'median'
    })
    
    print(degradation_summary)
    
    # Correlation with cycles
    correlation = df[['cycles', 'Ni_particle_size_current_nm', 'TPB_loss_percent',
                      'crack_initiation_indicator']].corr()
    
    print("\nCorrelation with cycles:")
    print(correlation['cycles'].sort_values(ascending=False))
    
    return degradation_summary


def example_4_experimental_data():
    """Example 4: Work with experimental data"""
    print("\n" + "="*70)
    print("Example 4: Exploring Experimental Data")
    print("="*70)
    
    # Load summary
    df_summary = pd.read_csv('sofc_multifidelity_dataset/phase4_experimental/experimental_summary.csv')
    
    print(f"\nNumber of cells: {len(df_summary)}")
    print(f"\nTest conditions:")
    print(df_summary[['cell_id', 'test_condition', 'operating_temperature_K', 
                      'test_duration_hours']].head())
    
    # Load I-V curves
    df_iv = pd.read_csv('sofc_multifidelity_dataset/phase4_experimental/IV_curves_timeseries.csv')
    
    # Get data for first cell
    cell_1 = df_iv[df_iv['cell_id'] == 'SOFC_EXP_001']
    time_points = sorted(cell_1['time_hours'].unique())
    
    print(f"\nCell 1 characterized at times: {time_points} hours")
    
    # Calculate degradation rate
    t0_data = cell_1[cell_1['time_hours'] == 0]
    tf_data = cell_1[cell_1['time_hours'] == time_points[-1]]
    
    if len(t0_data) > 0 and len(tf_data) > 0:
        V_initial = t0_data[t0_data['current_density_A_cm2'] == 1.0]['voltage_V'].values
        V_final = tf_data[tf_data['current_density_A_cm2'] == 1.0]['voltage_V'].values
        
        if len(V_initial) > 0 and len(V_final) > 0:
            degradation = ((V_initial[0] - V_final[0]) / time_points[-1]) * 1000
            print(f"\nDegradation rate at 1.0 A/cm²: {degradation:.3f} mV/kh")
    
    return df_summary, df_iv


def example_5_visualize_sample():
    """Example 5: Quick visualization"""
    print("\n" + "="*70)
    print("Example 5: Creating Quick Visualization")
    print("="*70)
    
    # Load data
    df = pd.read_csv('sofc_multifidelity_dataset/phase1_LF/phase1_LF_complete.csv')
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SOFC Dataset: Quick Analysis', fontsize=14, fontweight='bold')
    
    # Plot 1: Temperature vs Stress
    ax = axes[0, 0]
    scatter = ax.scatter(df['operating_temperature_K'], df['avg_von_mises_stress_MPa'],
                        c=df['current_density_A_cm2'], cmap='viridis', alpha=0.5, s=10)
    ax.set_xlabel('Operating Temperature (K)')
    ax.set_ylabel('Von Mises Stress (MPa)')
    ax.set_title('Temperature vs Stress')
    plt.colorbar(scatter, ax=ax, label='Current Density (A/cm²)')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: I-V Characteristic
    ax = axes[0, 1]
    # Sample at different temps
    for T in [873, 973, 1073]:
        subset = df[abs(df['operating_temperature_K'] - T) < 10]
        if len(subset) > 0:
            subset_sorted = subset.sort_values('current_density_A_cm2')
            ax.plot(subset_sorted['current_density_A_cm2'].values[:50], 
                   subset_sorted['voltage_V'].values[:50], 
                   'o-', alpha=0.7, label=f'T={T}K', markersize=3)
    ax.set_xlabel('Current Density (A/cm²)')
    ax.set_ylabel('Voltage (V)')
    ax.set_title('I-V Characteristics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Degradation vs Cycles
    ax = axes[1, 0]
    scatter = ax.scatter(df['cycles'], df['Ni_particle_size_nm'],
                        c=df['operating_temperature_K'], cmap='hot', alpha=0.5, s=10)
    ax.set_xlabel('Thermal Cycles')
    ax.set_ylabel('Ni Particle Size (nm)')
    ax.set_title('Ni Coarsening vs Cycles')
    plt.colorbar(scatter, ax=ax, label='Temperature (K)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Life Prediction
    ax = axes[1, 1]
    scatter = ax.scatter(df['avg_von_mises_stress_MPa'], 
                        np.log10(df['time_to_failure_hours']),
                        c=df['thermal_cycling_rate_K_min'], cmap='coolwarm', alpha=0.5, s=10)
    ax.set_xlabel('Von Mises Stress (MPa)')
    ax.set_ylabel('log₁₀(Time to Failure) (hours)')
    ax.set_title('Life Prediction')
    plt.colorbar(scatter, ax=ax, label='Cycling Rate (K/min)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150, bbox_inches='tight')
    print("\n✓ Visualization saved as 'quick_analysis.png'")
    plt.close()


def example_6_multifidelity_comparison():
    """Example 6: Compare across fidelities"""
    print("\n" + "="*70)
    print("Example 6: Multi-Fidelity Comparison")
    print("="*70)
    
    # Load all fidelities
    df_lf = pd.read_csv('sofc_multifidelity_dataset/phase1_LF/phase1_LF_complete.csv')
    df_mf = pd.read_csv('sofc_multifidelity_dataset/phase2_MF/phase2_MF_global.csv')
    df_hf = pd.read_csv('sofc_multifidelity_dataset/phase3_HF/phase3_HF_complete.csv')
    
    print("\nDataset sizes:")
    print(f"  Low-Fidelity:  {len(df_lf):,} samples")
    print(f"  Mid-Fidelity:  {len(df_mf):,} samples")
    print(f"  High-Fidelity: {len(df_hf):,} samples")
    
    print("\nStress statistics (MPa):")
    print(f"  LF - Mean: {df_lf['avg_von_mises_stress_MPa'].mean():.2f}, "
          f"Std: {df_lf['avg_von_mises_stress_MPa'].std():.2f}")
    print(f"  MF - Mean: {df_mf['avg_stress_MPa'].mean():.2f}, "
          f"Std: {df_mf['avg_stress_MPa'].std():.2f}")
    print(f"  HF - Mean: {df_hf['von_mises_stress_MPa'].mean():.2f}, "
          f"Std: {df_hf['von_mises_stress_MPa'].std():.2f}")
    
    print("\nDamage indicators:")
    print(f"  LF - Crack probability: {df_lf['crack_probability'].mean():.3f}")
    print(f"  HF - Crack initiation:  {df_hf['crack_initiation_indicator'].mean():.3f}")
    print(f"  HF - Cracks present:    {df_hf['crack_present'].sum()} / {len(df_hf)} "
          f"({df_hf['crack_present'].sum()/len(df_hf)*100:.1f}%)")
    
    print("\nOperating condition coverage:")
    print(f"  Temperature range (K):")
    print(f"    LF: {df_lf['operating_temperature_K'].min():.0f} - "
          f"{df_lf['operating_temperature_K'].max():.0f}")
    print(f"    HF: {df_hf['operating_temperature_K'].min():.0f} - "
          f"{df_hf['operating_temperature_K'].max():.0f}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print(" "*20 + "SOFC DATASET USAGE EXAMPLES")
    print("="*80)
    
    # Run examples
    df_lf = example_1_load_csv_data()
    T_field, sigma_field, damage_field = example_2_load_hdf5_spatial()
    degradation_summary = example_3_analyze_degradation()
    df_summary, df_iv = example_4_experimental_data()
    example_5_visualize_sample()
    example_6_multifidelity_comparison()
    
    print("\n" + "="*80)
    print(" "*25 + "EXAMPLES COMPLETE!")
    print("="*80)
    print("\n✅ All examples executed successfully!")
    print("✅ Check 'quick_analysis.png' for visualization\n")


if __name__ == "__main__":
    main()
