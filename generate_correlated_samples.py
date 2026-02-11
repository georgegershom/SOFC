"""
Synthetic Correlation Generator for Sensitivity Analysis
Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness in SOCs

This script generates correlated fracture energy samples to fill the gap of MISSING_DATASET_01
(Interfacial Property Correlation Matrix).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_correlated_fracture_energies(n_samples, mu1, sigma1, mu2, sigma2, rho):
    """
    Generates correlated Gc values for YSZ|GDC (1) and GDC|LSCF (2)
    to fill the gap of MISSING_DATASET_01.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    mu1 : float
        Mean fracture energy for YSZ|GDC interface (J/m²)
    sigma1 : float
        Standard deviation for YSZ|GDC interface (J/m²)
    mu2 : float
        Mean fracture energy for GDC|LSCF interface (J/m²)
    sigma2 : float
        Standard deviation for GDC|LSCF interface (J/m²)
    rho : float
        Correlation coefficient (-1 to 1)
        
    Returns:
    --------
    DataFrame with columns ['Gc_YSZ_GDC', 'Gc_GDC_LSCF']
    """
    mean = [mu1, mu2]
    cov = [[sigma1**2, rho * sigma1 * sigma2], 
           [rho * sigma1 * sigma2, sigma2**2]]
    
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Ensure positive values (fracture energy cannot be negative)
    samples = np.maximum(samples, 0.01)
    
    df = pd.DataFrame(samples, columns=['Gc_YSZ_GDC', 'Gc_GDC_LSCF'])
    return df


def generate_stochastic_inputs(n_samples=500, temperature='RT', correlation=0.0):
    """
    Generate stochastic input parameters for Abaqus UEL simulations.
    
    Parameters:
    -----------
    n_samples : int
        Number of parameter sets to generate
    temperature : str
        'RT' for room temperature (25°C) or 'HT' for high temperature (800°C)
    correlation : float
        Correlation coefficient between interfaces
        
    Returns:
    --------
    DataFrame with input parameters
    """
    # Material properties based on 04_uncertainty_material_properties.csv
    if temperature == 'RT':
        # Room Temperature (25°C)
        mu1, sigma1 = 2.15, 0.42  # YSZ|GDC
        mu2, sigma2 = 1.02, 0.21  # GDC|LSCF
        temp_val = 25
    else:  # High Temperature (800°C)
        # High-T Scaling: Proportional Variance Assumption
        mu1, sigma1 = 1.89, 0.37  # YSZ|GDC
        mu2, sigma2 = 0.88, 0.18  # GDC|LSCF
        temp_val = 800
    
    # Generate correlated samples
    df = generate_correlated_fracture_energies(n_samples, mu1, sigma1, mu2, sigma2, correlation)
    
    # Add sample ID and temperature
    df.insert(0, 'Sample_ID', [f'S{i+1:04d}' for i in range(n_samples)])
    df['Temperature_C'] = temp_val
    df['Correlation_Coeff'] = correlation
    
    return df


def plot_correlation_comparison(save_dir='.'):
    """
    Create scatter plots showing independent vs. correlated assumptions.
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate datasets
    df_independent = generate_correlated_fracture_energies(500, 2.15, 0.42, 1.02, 0.21, 0.0)
    df_correlated = generate_correlated_fracture_energies(500, 2.15, 0.42, 1.02, 0.21, 0.5)
    df_strong_corr = generate_correlated_fracture_energies(500, 2.15, 0.42, 1.02, 0.21, 0.8)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Independent (rho = 0.0)
    axes[0].scatter(df_independent['Gc_YSZ_GDC'], df_independent['Gc_GDC_LSCF'], 
                    alpha=0.5, s=20, c='blue', label='Independent (ρ=0.0)')
    axes[0].set_xlabel('Gc (YSZ|GDC) [J/m²]', fontsize=12)
    axes[0].set_ylabel('Gc (GDC|LSCF) [J/m²]', fontsize=12)
    axes[0].set_title('Series A: Current Assumption\n(Uncorrelated)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Moderate Correlation (rho = 0.5)
    axes[1].scatter(df_correlated['Gc_YSZ_GDC'], df_correlated['Gc_GDC_LSCF'], 
                    alpha=0.5, s=20, c='orange', label='Moderate (ρ=0.5)')
    axes[1].set_xlabel('Gc (YSZ|GDC) [J/m²]', fontsize=12)
    axes[1].set_ylabel('Gc (GDC|LSCF) [J/m²]', fontsize=12)
    axes[1].set_title('Series B: Hypothetical Reality\n(Process Defects)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Plot 3: Strong Correlation (rho = 0.8)
    axes[2].scatter(df_strong_corr['Gc_YSZ_GDC'], df_strong_corr['Gc_GDC_LSCF'], 
                    alpha=0.5, s=20, c='red', label='Strong (ρ=0.8)')
    axes[2].set_xlabel('Gc (YSZ|GDC) [J/m²]', fontsize=12)
    axes[2].set_ylabel('Gc (GDC|LSCF) [J/m²]', fontsize=12)
    axes[2].set_title('Sensitivity Case\n(Strong Correlation)', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/correlation_comparison_scatter.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/correlation_comparison_scatter.png")
    return fig


def plot_distributions(save_dir='.'):
    """
    Create histograms showing the distribution of fracture energies.
    """
    np.random.seed(42)
    
    # Generate data
    df_rt = generate_stochastic_inputs(500, temperature='RT', correlation=0.0)
    df_ht = generate_stochastic_inputs(500, temperature='HT', correlation=0.0)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # YSZ|GDC at RT
    axes[0, 0].hist(df_rt['Gc_YSZ_GDC'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(df_rt['Gc_YSZ_GDC'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean = {df_rt["Gc_YSZ_GDC"].mean():.2f} J/m²')
    axes[0, 0].set_xlabel('Gc (YSZ|GDC) [J/m²]', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('YSZ|GDC Interface @ 25°C', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # YSZ|GDC at 800°C
    axes[0, 1].hist(df_ht['Gc_YSZ_GDC'], bins=30, alpha=0.7, color='coral', edgecolor='black')
    axes[0, 1].axvline(df_ht['Gc_YSZ_GDC'].mean(), color='darkred', linestyle='--', 
                       linewidth=2, label=f'Mean = {df_ht["Gc_YSZ_GDC"].mean():.2f} J/m²')
    axes[0, 1].set_xlabel('Gc (YSZ|GDC) [J/m²]', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('YSZ|GDC Interface @ 800°C', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # GDC|LSCF at RT
    axes[1, 0].hist(df_rt['Gc_GDC_LSCF'], bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
    axes[1, 0].axvline(df_rt['Gc_GDC_LSCF'].mean(), color='darkgreen', linestyle='--', 
                       linewidth=2, label=f'Mean = {df_rt["Gc_GDC_LSCF"].mean():.2f} J/m²')
    axes[1, 0].set_xlabel('Gc (GDC|LSCF) [J/m²]', fontsize=11)
    axes[1, 0].set_ylabel('Frequency', fontsize=11)
    axes[1, 0].set_title('GDC|LSCF Interface @ 25°C', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # GDC|LSCF at 800°C
    axes[1, 1].hist(df_ht['Gc_GDC_LSCF'], bins=30, alpha=0.7, color='goldenrod', edgecolor='black')
    axes[1, 1].axvline(df_ht['Gc_GDC_LSCF'].mean(), color='darkgoldenrod', linestyle='--', 
                       linewidth=2, label=f'Mean = {df_ht["Gc_GDC_LSCF"].mean():.2f} J/m²')
    axes[1, 1].set_xlabel('Gc (GDC|LSCF) [J/m²]', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('GDC|LSCF Interface @ 800°C', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/fracture_energy_distributions.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {save_dir}/fracture_energy_distributions.png")
    return fig


if __name__ == "__main__":
    print("="*80)
    print("Probabilistic Failure Maps: Uncertainty Quantification of Interfacial Toughness")
    print("="*80)
    
    # Generate baseline dataset (uncorrelated, rho=0.0)
    print("\n[1] Generating baseline dataset (ρ=0.0, Independent)...")
    df_baseline_rt = generate_stochastic_inputs(n_samples=500, temperature='RT', correlation=0.0)
    df_baseline_ht = generate_stochastic_inputs(n_samples=500, temperature='HT', correlation=0.0)
    
    print("\n--- Room Temperature (25°C) ---")
    print(df_baseline_rt.head())
    print(f"\nStatistics:")
    print(df_baseline_rt[['Gc_YSZ_GDC', 'Gc_GDC_LSCF']].describe())
    
    print("\n--- High Temperature (800°C) ---")
    print(df_baseline_ht.head())
    print(f"\nStatistics:")
    print(df_baseline_ht[['Gc_YSZ_GDC', 'Gc_GDC_LSCF']].describe())
    
    # Generate sensitivity datasets
    print("\n[2] Generating sensitivity datasets (ρ=0.5, Process Defects)...")
    df_sensitivity = generate_stochastic_inputs(n_samples=500, temperature='RT', correlation=0.5)
    print(df_sensitivity.head())
    print(f"\nActual correlation: {df_sensitivity[['Gc_YSZ_GDC', 'Gc_GDC_LSCF']].corr().iloc[0,1]:.3f}")
    
    # Save CSV files
    print("\n[3] Saving CSV files...")
    df_baseline_rt.to_csv('stochastic_inputs_RT_uncorrelated.csv', index=False)
    print("Saved: stochastic_inputs_RT_uncorrelated.csv")
    
    df_baseline_ht.to_csv('stochastic_inputs_HT_uncorrelated.csv', index=False)
    print("Saved: stochastic_inputs_HT_uncorrelated.csv")
    
    df_sensitivity.to_csv('stochastic_inputs_RT_correlated_rho050.csv', index=False)
    print("Saved: stochastic_inputs_RT_correlated_rho050.csv")
    
    # Generate visualizations
    print("\n[4] Generating visualizations...")
    plot_correlation_comparison()
    plot_distributions()
    
    print("\n" + "="*80)
    print("COMPLETE: All datasets and figures generated successfully!")
    print("="*80)
