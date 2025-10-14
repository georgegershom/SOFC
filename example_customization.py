"""
Example: How to Customize Figure 4a.2 for Your Specific Material

This script demonstrates common customization scenarios:
1. Using your own material parameters
2. Adding your experimental data
3. Changing stress-temperature test matrix
4. Modifying visual styling
5. Adjusting thresholds and criteria
"""

import numpy as np
from generate_figure_4a2_advanced import CreepDamageModel, create_figure_4a2
import matplotlib.pyplot as plt


# ============================================================================
# EXAMPLE 1: Custom Material Parameters
# ============================================================================

class CustomTBCModel(CreepDamageModel):
    """
    Custom TBC material with your fitted parameters
    Example: Yttria-stabilized zirconia with bond coat interface
    """
    
    def __init__(self):
        super().__init__()
        
        # YOUR FITTED CREEP PARAMETERS (replace with your values)
        self.A = 2.5e-10      # Your fitted A from Arrhenius plot
        self.n = 3.8          # Your fitted n from log-log stress plot
        self.Q = 310e3        # Your Q from 1/T slope [J/mol]
        
        # YOUR DAMAGE PARAMETERS (from long-hold experiments)
        self.B = 8.0e-7       # Calibrated from constant-load tests
        self.m = 2.2          # Stress exponent from damage rate data
        self.k = 1.8          # From late-stage damage saturation
        
        # YOUR THRESHOLD CRITERIA (from micrograph analysis)
        self.D_c = 0.35       # Critical damage from crack density correlation
        self.eps_dot_c_star = 1e-6  # From DIC strain rate threshold
        
        # YOUR DESIGN REQUIREMENT
        self.t_target = 90    # Required dwell time [min]
        
        # Temperature-dependent Gc from your Mode I tests
        self.G_c_base = 30.0       # J/m²
        self.G_c_slope = 0.20      # %/°C


# ============================================================================
# EXAMPLE 2: Using Your Experimental Data
# ============================================================================

def load_your_experimental_data():
    """
    Replace this with your actual data loading
    Format: arrays of (time, DIC_area, XRD_depth) for each test condition
    """
    
    # Example structure - replace with your CSV/Excel reading
    experimental_data = {
        (100, 1000): {  # (stress MPa, temp °C)
            'time': np.array([0, 10, 20, 30, 40, 50, 60]),  # minutes
            'DIC_area': np.array([0.5, 1.2, 2.8, 8.5, 22.3, 45.2, 68.1]),  # %
            'XRD_depth': np.array([0.1, 0.2, 0.5, 1.8, 4.2, 7.8, 11.3]),  # µm
            'DIC_error': np.array([0.2, 0.3, 0.4, 1.0, 2.1, 3.5, 5.2]),
            'XRD_error': np.array([0.05, 0.08, 0.12, 0.25, 0.45, 0.68, 0.92])
        },
        # Add more conditions...
    }
    
    return experimental_data


# ============================================================================
# EXAMPLE 3: Custom Test Matrix
# ============================================================================

def define_your_test_matrix():
    """
    Define the specific (σ, T) conditions you tested
    Returns list of (stress, temperature, color) tuples
    """
    
    # Option A: Grid of conditions (factorial design)
    stresses = [80, 100, 120]        # MPa
    temperatures = [950, 1000, 1050] # °C
    colors = ['#E41A1C', '#377EB8', '#4DAF4A', 
              '#984EA3', '#FF7F00', '#FFFF33']
    
    test_matrix = []
    idx = 0
    for sigma in stresses:
        for T in temperatures:
            test_matrix.append((sigma, T, colors[idx % len(colors)]))
            idx += 1
    
    # Option B: Specific critical conditions you tested
    # test_matrix = [
    #     (85, 920, '#E41A1C'),   # Safe baseline
    #     (105, 980, '#377EB8'),  # Near threshold
    #     (125, 1040, '#4DAF4A'), # Expected failure
    # ]
    
    return test_matrix


# ============================================================================
# EXAMPLE 4: Custom Visualization Styling
# ============================================================================

def apply_your_lab_style():
    """
    Apply your lab/company style guide to plots
    """
    
    plt.rcParams.update({
        # Fonts (match your institution's style guide)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 11,
        
        # Colors (your brand colors)
        'axes.prop_cycle': plt.cycler(color=[
            '#003366',  # Dark blue
            '#CC0000',  # Red
            '#00AA44',  # Green
            '#FF8800',  # Orange
            '#8800CC',  # Purple
        ]),
        
        # Line styles (thicker for presentations)
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        
        # Grid (lighter for cleaner look)
        'grid.alpha': 0.2,
        'grid.linestyle': ':',
        
        # Figure size (adjust for your layout)
        'figure.figsize': (18, 14),  # Larger for poster
    })


# ============================================================================
# EXAMPLE 5: Adjusted Hazard Map Range
# ============================================================================

def compute_custom_hazard_map(model, sigma_range, T_range, resolution=60):
    """
    Compute hazard map over your specific operating range
    
    Args:
        model: Your custom CreepDamageModel instance
        sigma_range: (min, max) stress in MPa
        T_range: (min, max) temperature in °C
        resolution: Grid points per axis
    """
    
    sigma_grid = np.linspace(sigma_range[0], sigma_range[1], resolution)
    T_grid = np.linspace(T_range[0], T_range[1], resolution)
    
    # ... (computation logic similar to Panel C in main code)
    
    return sigma_grid, T_grid, None  # Placeholder


# ============================================================================
# EXAMPLE 6: Multi-Material Comparison
# ============================================================================

def compare_materials():
    """
    Generate comparison figure for multiple material systems
    """
    
    materials = {
        'Standard YSZ': CreepDamageModel(),
        'Modified YSZ': CustomTBCModel(),
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, (name, model) in enumerate(materials.items()):
        ax = axes[idx]
        
        # Plot creep curves for comparison
        sigma, T = 100, 1000
        t, eps_c = model.integrate_creep(sigma, T, 120)
        
        ax.plot(t, eps_c * 1e6, linewidth=2.5, label=name)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Creep Strain (µε)')
        ax.set_title(f'{name} at {T}°C, {sigma} MPa')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Material_Comparison.png', dpi=300)
    print("✓ Material comparison saved")


# ============================================================================
# EXAMPLE 7: Sensitivity Analysis
# ============================================================================

def sensitivity_analysis():
    """
    Analyze sensitivity to key parameters
    """
    
    model = CreepDamageModel()
    base_D_c = model.D_c
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Vary critical damage threshold
    D_c_values = [0.2, 0.25, 0.3, 0.35, 0.4]
    sigma, T = 100, 1000
    
    for D_c in D_c_values:
        model.D_c = D_c
        t, D = model.integrate_damage(sigma, T, 120)
        
        ax.plot(t, D, linewidth=2, label=f'$D_c$ = {D_c}', alpha=0.8)
        ax.axhline(D_c, linestyle='--', alpha=0.5, color=ax.lines[-1].get_color())
    
    ax.set_xlabel('Time (min)', fontsize=12)
    ax.set_ylabel('Damage $D$', fontsize=12)
    ax.set_title(f'Sensitivity to $D_c$ at {T}°C, {sigma} MPa', fontsize=13)
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('Sensitivity_Dc.png', dpi=300)
    print("✓ Sensitivity analysis saved")
    
    # Restore original value
    model.D_c = base_D_c


# ============================================================================
# EXAMPLE 8: Export Data for Further Analysis
# ============================================================================

def export_simulation_data():
    """
    Export simulation results to CSV for external analysis
    (e.g., import into Excel, MATLAB, or statistical software)
    """
    
    model = CreepDamageModel()
    
    # Generate data for key conditions
    conditions = [(80, 900), (100, 1000), (120, 1100)]
    
    with open('simulation_results.csv', 'w') as f:
        f.write('Stress_MPa,Temperature_C,Time_min,Creep_Strain,Damage,Notes\n')
        
        for sigma, T in conditions:
            t, eps_c = model.integrate_creep(sigma, T, 120, dt=1.0)
            t_D, D = model.integrate_damage(sigma, T, 120, dt=1.0)
            
            for i in range(len(t)):
                f.write(f'{sigma},{T},{t[i]:.2f},{eps_c[i]:.6e},{D[i]:.6f},\n')
    
    print("✓ Data exported to simulation_results.csv")


# ============================================================================
# EXAMPLE 9: Uncertainty Quantification
# ============================================================================

def uncertainty_quantification(n_samples=100):
    """
    Monte Carlo uncertainty propagation through model
    Accounts for parameter uncertainty (e.g., from regression)
    """
    
    # Parameter uncertainties (standard deviations)
    param_uncertainty = {
        'A': 0.2e-10,     # ±20% of A
        'n': 0.2,         # ±0.2 in n
        'Q': 15e3,        # ±15 kJ/mol
        'D_c': 0.025,     # ±0.025
    }
    
    sigma, T = 100, 1000
    t_nuc_samples = []
    
    for i in range(n_samples):
        model = CreepDamageModel()
        
        # Perturb parameters (normal distribution)
        model.A += np.random.normal(0, param_uncertainty['A'])
        model.n += np.random.normal(0, param_uncertainty['n'])
        model.Q += np.random.normal(0, param_uncertainty['Q'])
        model.D_c += np.random.normal(0, param_uncertainty['D_c'])
        
        # Compute nucleation time
        t, D = model.integrate_damage(sigma, T, 120)
        t_D = model.find_damage_threshold_time(t, D)
        t_nuc_samples.append(t_D)
    
    # Statistics
    t_nuc_mean = np.mean(t_nuc_samples)
    t_nuc_std = np.std(t_nuc_samples)
    t_nuc_95 = np.percentile(t_nuc_samples, [2.5, 97.5])
    
    print(f"\n{'='*60}")
    print(f"Uncertainty Quantification: {n_samples} samples")
    print(f"{'='*60}")
    print(f"Condition: {sigma} MPa, {T}°C")
    print(f"Mean t_nuc: {t_nuc_mean:.2f} min")
    print(f"Std deviation: {t_nuc_std:.2f} min")
    print(f"95% CI: [{t_nuc_95[0]:.2f}, {t_nuc_95[1]:.2f}] min")
    print(f"{'='*60}\n")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(t_nuc_samples, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(t_nuc_mean, color='red', linewidth=2.5, label=f'Mean = {t_nuc_mean:.1f} min')
    plt.axvline(t_nuc_95[0], color='orange', linewidth=2, linestyle='--', label='95% CI')
    plt.axvline(t_nuc_95[1], color='orange', linewidth=2, linestyle='--')
    plt.xlabel('Nucleation Time $t_{nuc}$ (min)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'Uncertainty in $t_{{nuc}}$ at {T}°C, {sigma} MPa', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Uncertainty_Analysis.png', dpi=300)
    print("✓ Uncertainty histogram saved")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("CUSTOMIZATION EXAMPLES FOR FIGURE 4a.2")
    print("="*70 + "\n")
    
    # Example 1: Use custom material
    print("Example 1: Custom material parameters")
    custom_model = CustomTBCModel()
    print(f"  Custom A = {custom_model.A:.2e} s⁻¹ MPa⁻ⁿ")
    print(f"  Custom n = {custom_model.n}")
    print(f"  Custom Q = {custom_model.Q/1000:.0f} kJ/mol")
    print(f"  Custom D_c = {custom_model.D_c}")
    print(f"  Custom t_target = {custom_model.t_target} min")
    
    # Example 2: Load your data structure
    print("\nExample 2: Experimental data structure")
    exp_data = load_your_experimental_data()
    print(f"  Loaded {len(exp_data)} test conditions")
    
    # Example 3: Define test matrix
    print("\nExample 3: Custom test matrix")
    test_matrix = define_your_test_matrix()
    print(f"  Defined {len(test_matrix)} test points")
    for i, (s, t, c) in enumerate(test_matrix[:3]):
        print(f"    {i+1}. {s} MPa, {t}°C (color: {c})")
    
    # Example 4: Apply custom styling
    print("\nExample 4: Custom visualization style")
    apply_your_lab_style()
    print("  ✓ Applied custom matplotlib style")
    
    # Example 5: Material comparison
    print("\nExample 5: Multi-material comparison")
    compare_materials()
    
    # Example 6: Sensitivity analysis
    print("\nExample 6: Sensitivity analysis")
    sensitivity_analysis()
    
    # Example 7: Export data
    print("\nExample 7: Export simulation data")
    export_simulation_data()
    
    # Example 8: Uncertainty quantification
    print("\nExample 8: Uncertainty quantification")
    uncertainty_quantification(n_samples=50)  # Use 50 for speed
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED")
    print("="*70)
    print("\nGenerated files:")
    print("  • Material_Comparison.png")
    print("  • Sensitivity_Dc.png")
    print("  • Uncertainty_Analysis.png")
    print("  • simulation_results.csv")
    print("\nNext steps:")
    print("  1. Replace CustomTBCModel parameters with your fitted values")
    print("  2. Load your experimental data in load_your_experimental_data()")
    print("  3. Adjust test_matrix to match your conditions")
    print("  4. Re-run generate_figure_4a2_advanced.py with custom model")
    print("="*70 + "\n")
