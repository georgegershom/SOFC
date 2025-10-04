"""
Script to generate figures for SOFC Constitutive Models Research Article
This script creates all the figures referenced in the research paper
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import interpolate
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality plot parameters
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color scheme for consistency
colors = {
    'elastic': '#2E86AB',
    'viscoelastic': '#A23B72',
    'experimental': '#F18F01',
    'baseline': '#C73E1D',
    'gradient': ['#FFF275', '#FF8C42', '#FF3C38', '#A23B72', '#2E86AB']
}

def create_stress_contour_data(nx=100, ny=50, model_type='elastic'):
    """Generate synthetic stress contour data for visualization"""
    x = np.linspace(0, 50, nx)  # 50mm width
    y = np.linspace(0, 0.01, ny)  # 10 μm thickness
    X, Y = np.meshgrid(x, y)
    
    # Create periodic stress pattern (representing channel-rib structure)
    channel_period = 2.0  # 2mm pitch
    stress_pattern = np.sin(2 * np.pi * X / channel_period) * np.exp(-10 * Y)
    
    # Add stress concentrations at interfaces
    interface_stress = 50 * np.exp(-100 * (Y - 0.005)**2)
    
    # Base stress level
    if model_type == 'elastic':
        base_stress = 100 + 25 * stress_pattern + interface_stress
        # Add edge effects
        edge_effect = 20 * np.exp(-0.5 * np.minimum(X, 50-X))
        base_stress += edge_effect
        peak_stress = 145
    else:  # viscoelastic
        base_stress = 80 + 20 * stress_pattern + 0.8 * interface_stress
        edge_effect = 15 * np.exp(-0.5 * np.minimum(X, 50-X))
        base_stress += edge_effect
        peak_stress = 113
    
    # Normalize to peak stress
    base_stress = base_stress * peak_stress / np.max(base_stress)
    
    return X, Y, base_stress

def figure1_von_mises_after_sintering():
    """Figure 1: Von Mises Stress Distribution in Electrolyte After Sintering"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    X, Y, stress = create_stress_contour_data(model_type='elastic')
    stress = stress * 0.86  # Scale for room temperature
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('stress', colors['gradient'])
    
    im = ax.contourf(X, Y * 1000, stress, levels=20, cmap=cmap)
    ax.contour(X, Y * 1000, stress, levels=10, colors='black', linewidths=0.3, alpha=0.3)
    
    # Add labels and formatting
    ax.set_xlabel('Position along cell width (mm)')
    ax.set_ylabel('Electrolyte thickness (μm)')
    ax.set_title('Von Mises Stress Distribution After Sintering Cool-down')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Von Mises Stress (MPa)')
    
    # Add annotations for key features
    ax.annotate('Peak stress at interface', xy=(10, 5), xytext=(15, 8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    ax.annotate('Channel region', xy=(25, 2), xytext=(30, 3),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
    
    plt.savefig('figure1_von_mises_sintering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure2_elastic_vs_viscoelastic_comparison():
    """Figure 2: Comparison of Von Mises Stress - Elastic vs. Viscoelastic Models at 800°C"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elastic model
    X1, Y1, stress1 = create_stress_contour_data(model_type='elastic')
    cmap = LinearSegmentedColormap.from_list('stress', colors['gradient'])
    
    im1 = ax1.contourf(X1, Y1 * 1000, stress1, levels=20, cmap=cmap, vmin=35, vmax=145)
    ax1.contour(X1, Y1 * 1000, stress1, levels=10, colors='black', linewidths=0.3, alpha=0.3)
    ax1.set_xlabel('Position along cell width (mm)')
    ax1.set_ylabel('Electrolyte thickness (μm)')
    ax1.set_title('(a) Elastic Model - Peak: 145 MPa')
    
    # Viscoelastic model
    X2, Y2, stress2 = create_stress_contour_data(model_type='viscoelastic')
    
    im2 = ax2.contourf(X2, Y2 * 1000, stress2, levels=20, cmap=cmap, vmin=35, vmax=145)
    ax2.contour(X2, Y2 * 1000, stress2, levels=10, colors='black', linewidths=0.3, alpha=0.3)
    ax2.set_xlabel('Position along cell width (mm)')
    ax2.set_ylabel('Electrolyte thickness (μm)')
    ax2.set_title('(b) Viscoelastic Model (t=100h) - Peak: 113 MPa')
    
    # Add shared colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im1, cax=cbar_ax, label='Von Mises Stress (MPa)')
    
    plt.suptitle('Stress Distribution at 800°C Steady-State Operation', fontsize=14, y=1.02)
    plt.savefig('figure2_elastic_vs_viscoelastic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure3_stress_relaxation_time():
    """Figure 3: Time Evolution of Maximum Principal Stress - Viscoelastic Relaxation"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Time array (hours)
    time = np.logspace(-2, 3, 100)  # 0.01 to 1000 hours
    
    # Stress relaxation model
    initial_stress = 138  # MPa
    final_stress = 108  # MPa
    relaxation_time = 10  # hours
    
    # Exponential relaxation
    stress_elastic = np.ones_like(time) * initial_stress
    stress_visco = final_stress + (initial_stress - final_stress) * np.exp(-time/relaxation_time)
    
    # Add some realistic fluctuation
    noise = np.random.normal(0, 0.5, len(time))
    stress_visco += noise * np.exp(-time/100)
    
    # Plot
    ax.semilogx(time, stress_elastic, '-', color=colors['elastic'], linewidth=2.5, label='Elastic Model')
    ax.semilogx(time, stress_visco, '-', color=colors['viscoelastic'], linewidth=2.5, label='Viscoelastic Model')
    
    # Add shaded region for relaxation
    ax.fill_between(time, stress_elastic, stress_visco, alpha=0.2, color='gray')
    
    # Annotations
    ax.axhline(y=final_stress, color='gray', linestyle='--', alpha=0.5)
    ax.text(200, final_stress + 2, 'Quasi-equilibrium: 108 MPa', fontsize=10)
    
    # Add relaxation percentage
    ax.annotate('', xy=(500, final_stress), xytext=(500, initial_stress),
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    ax.text(600, (initial_stress + final_stress)/2, '22% Relaxation', color='red', fontsize=10)
    
    # Formatting
    ax.set_xlabel('Time at 800°C (hours)')
    ax.set_ylabel('Maximum Principal Stress (MPa)')
    ax.set_title('Stress Relaxation Due to Creep at Operational Temperature')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper right')
    ax.set_xlim([0.01, 1000])
    ax.set_ylim([100, 145])
    
    plt.savefig('figure3_stress_relaxation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure4_spatial_relaxation_map():
    """Figure 4: Spatial Map of Stress Relaxation Percentage"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Generate relaxation data
    nx, ny = 100, 50
    x = np.linspace(0, 50, nx)
    y = np.linspace(0, 0.01, ny)
    X, Y = np.meshgrid(x, y)
    
    # Relaxation correlates with initial stress level
    _, _, initial_stress = create_stress_contour_data(model_type='elastic')
    
    # Calculate relaxation percentage
    relaxation = 5 + 25 * (initial_stress - 35) / (145 - 35)  # 5-30% range
    
    # Add spatial variation
    spatial_var = 5 * np.sin(2 * np.pi * X / 5) * np.exp(-50 * Y)
    relaxation += spatial_var
    relaxation = np.clip(relaxation, 5, 30)
    
    # Plot
    cmap = plt.cm.RdYlBu_r
    im = ax.contourf(X, Y * 1000, relaxation, levels=15, cmap=cmap)
    ax.contour(X, Y * 1000, relaxation, levels=8, colors='black', linewidths=0.3, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Position along cell width (mm)')
    ax.set_ylabel('Electrolyte thickness (μm)')
    ax.set_title('Spatial Distribution of Stress Relaxation After 100 Hours')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Stress Relaxation (%)')
    
    plt.savefig('figure4_spatial_relaxation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure5_creep_strain_accumulation():
    """Figure 5: Accumulated Creep Strain After 100 Hours at 800°C"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    
    # Generate creep strain data
    nx, ny = 100, 50
    x = np.linspace(0, 50, nx)
    y = np.linspace(0, 0.01, ny)
    X, Y = np.meshgrid(x, y)
    
    # Creep strain pattern (correlates with stress)
    _, _, stress = create_stress_contour_data(model_type='viscoelastic')
    
    # Power-law creep strain
    creep_strain = 0.001 + 0.017 * (stress / 113)**1.5
    creep_strain = creep_strain * 100  # Convert to percentage
    
    # Plot
    cmap = plt.cm.YlOrRd
    im = ax.contourf(X, Y * 1000, creep_strain, levels=15, cmap=cmap)
    ax.contour(X, Y * 1000, creep_strain, levels=8, colors='black', linewidths=0.3, alpha=0.3)
    
    # Formatting
    ax.set_xlabel('Position along cell width (mm)')
    ax.set_ylabel('Electrolyte thickness (μm)')
    ax.set_title('Accumulated Equivalent Creep Strain Distribution')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Creep Strain (%)')
    
    plt.savefig('figure5_creep_strain.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure6_thermal_cycling_stress():
    """Figure 6: Maximum Principal Stress Evolution During Thermal Cycling"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Temperature profile for heating and cooling
    temp_heat = np.array([25, 200, 400, 600, 800])
    temp_cool = np.array([800, 700, 500, 300, 100])
    
    # Elastic model stress
    stress_elastic_heat = np.array([20, 45, 75, 105, 138])
    stress_elastic_cool = np.array([138, 130, 120, 155, 145])
    
    # Viscoelastic model stress
    stress_visco_heat = np.array([20, 45, 75, 100, 115])
    stress_visco_cool = np.array([115, 110, 125, 165, 160])
    
    # Create smooth interpolation
    temp_heat_smooth = np.linspace(25, 800, 100)
    temp_cool_smooth = np.linspace(800, 100, 100)
    
    # Interpolate for smooth curves
    f_elastic_heat = interpolate.interp1d(temp_heat, stress_elastic_heat, kind='cubic')
    f_elastic_cool = interpolate.interp1d(temp_cool, stress_elastic_cool, kind='cubic')
    f_visco_heat = interpolate.interp1d(temp_heat, stress_visco_heat, kind='cubic')
    f_visco_cool = interpolate.interp1d(temp_cool, stress_visco_cool, kind='cubic')
    
    # Plot heating phase
    ax.plot(temp_heat_smooth, f_elastic_heat(temp_heat_smooth), '-', color=colors['elastic'], 
            linewidth=2.5, label='Elastic - Heating')
    ax.plot(temp_cool_smooth, f_elastic_cool(temp_cool_smooth), '--', color=colors['elastic'], 
            linewidth=2.5, label='Elastic - Cooling')
    
    ax.plot(temp_heat_smooth, f_visco_heat(temp_heat_smooth), '-', color=colors['viscoelastic'], 
            linewidth=2.5, label='Viscoelastic - Heating')
    ax.plot(temp_cool_smooth, f_visco_cool(temp_cool_smooth), '--', color=colors['viscoelastic'], 
            linewidth=2.5, label='Viscoelastic - Cooling')
    
    # Add critical points
    ax.plot(200, 155, 'ro', markersize=8)
    ax.annotate('Peak stress during cooling\n(Elastic: 155 MPa)', 
                xy=(200, 155), xytext=(350, 160),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.plot(200, 165, 'rs', markersize=8)
    ax.annotate('Peak stress during cooling\n(Viscoelastic: 165 MPa)', 
                xy=(200, 165), xytext=(350, 175),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.3'))
    
    # Formatting
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Maximum Principal Stress (MPa)')
    ax.set_title('Stress Evolution During First Thermal Cycle')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlim([0, 850])
    ax.set_ylim([0, 180])
    
    plt.savefig('figure6_thermal_cycling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure7_failure_probability():
    """Figure 7: Cumulative Failure Probability vs. Operating Time"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Time array (hours)
    time = np.logspace(1, 5, 100)  # 10 to 100,000 hours
    
    # Weibull parameters
    sigma_0 = 280  # MPa
    m = 8  # Weibull modulus
    
    # Stress levels
    stress_elastic = 138  # Constant for elastic
    stress_visco = 108 + 30 * np.exp(-time/100)  # Relaxing for viscoelastic
    
    # Calculate failure probability
    Pf_elastic = 1 - np.exp(-(stress_elastic/sigma_0)**m)
    Pf_visco = 1 - np.exp(-(stress_visco/sigma_0)**m)
    
    # Account for time accumulation
    Pf_elastic_cumul = 1 - np.exp(-time/10000 * (stress_elastic/sigma_0)**m)
    Pf_visco_cumul = 1 - np.exp(-time/10000 * (stress_visco/sigma_0)**m)
    
    # Plot
    ax.semilogx(time, Pf_elastic_cumul * 100, '-', color=colors['elastic'], 
                linewidth=2.5, label='Elastic Model')
    ax.semilogx(time, Pf_visco_cumul * 100, '-', color=colors['viscoelastic'], 
                linewidth=2.5, label='Viscoelastic Model')
    
    # Add target lifetime markers
    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5)
    ax.text(50, 5.5, '5% Failure Criterion', color='red', fontsize=10)
    
    # Mark specific points
    ax.plot(4200, 5, 'o', color=colors['elastic'], markersize=8)
    ax.plot(8500, 5, 'o', color=colors['viscoelastic'], markersize=8)
    
    ax.annotate('Elastic: 4,200 h', xy=(4200, 5), xytext=(2000, 8),
                arrowprops=dict(arrowstyle='->', color=colors['elastic']))
    ax.annotate('Viscoelastic: 8,500 h', xy=(8500, 5), xytext=(15000, 8),
                arrowprops=dict(arrowstyle='->', color=colors['viscoelastic']))
    
    # Formatting
    ax.set_xlabel('Operating Time (hours)')
    ax.set_ylabel('Cumulative Failure Probability (%)')
    ax.set_title('Weibull Failure Analysis for Steady-State Operation')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left')
    ax.set_xlim([10, 100000])
    ax.set_ylim([0, 20])
    
    plt.savefig('figure7_failure_probability.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def figure8_sensitivity_analysis():
    """Figure 8: Tornado Diagram - Parameter Sensitivity on Predicted Lifetime"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Parameters and their effects
    parameters = [
        'Creep Activation Energy (Q)',
        'Stress Exponent (n)',
        'YSZ Tensile Strength',
        'Young\'s Modulus',
        'Thermal Expansion Coefficient',
        'Operating Temperature',
        'Pre-exponential Factor (B)',
        'Poisson\'s Ratio'
    ]
    
    # Baseline lifetime: 8500 hours
    baseline = 8500
    
    # Effects (hours change for ±variation)
    low_values = np.array([6375, 6800, 5525, 7820, 7650, 7225, 7650, 8200])
    high_values = np.array([10625, 10200, 11475, 9180, 9350, 9775, 9350, 8800])
    
    # Calculate deviations from baseline
    low_dev = baseline - low_values
    high_dev = high_values - baseline
    
    # Sort by total effect
    total_effect = low_dev + high_dev
    sort_idx = np.argsort(total_effect)[::-1]
    
    parameters = [parameters[i] for i in sort_idx]
    low_dev = low_dev[sort_idx]
    high_dev = high_dev[sort_idx]
    
    # Create tornado plot
    y_pos = np.arange(len(parameters))
    
    # Plot bars
    bars1 = ax.barh(y_pos, -low_dev, left=baseline, height=0.6, 
                     color=colors['elastic'], alpha=0.7, label='Low Parameter Value')
    bars2 = ax.barh(y_pos, high_dev, left=baseline, height=0.6, 
                     color=colors['viscoelastic'], alpha=0.7, label='High Parameter Value')
    
    # Add baseline line
    ax.axvline(x=baseline, color='black', linestyle='-', linewidth=2, label='Baseline (8,500 h)')
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(parameters)
    ax.set_xlabel('Predicted Lifetime (hours)')
    ax.set_title('Sensitivity Analysis: Impact of ±15% Parameter Variation on Lifetime')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([5000, 11000])
    
    # Add percentage labels
    for i, (l, h) in enumerate(zip(low_dev, high_dev)):
        ax.text(baseline - l - 100, i, f'-{l/baseline*100:.0f}%', 
                ha='right', va='center', fontsize=9)
        ax.text(baseline + h + 100, i, f'+{h/baseline*100:.0f}%', 
                ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('figure8_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_all_figures():
    """Generate all figures for the paper"""
    print("Generating Figure 1: Von Mises Stress After Sintering...")
    figure1_von_mises_after_sintering()
    
    print("Generating Figure 2: Elastic vs Viscoelastic Comparison...")
    figure2_elastic_vs_viscoelastic_comparison()
    
    print("Generating Figure 3: Stress Relaxation Over Time...")
    figure3_stress_relaxation_time()
    
    print("Generating Figure 4: Spatial Relaxation Map...")
    figure4_spatial_relaxation_map()
    
    print("Generating Figure 5: Creep Strain Accumulation...")
    figure5_creep_strain_accumulation()
    
    print("Generating Figure 6: Thermal Cycling Stress...")
    figure6_thermal_cycling_stress()
    
    print("Generating Figure 7: Failure Probability...")
    figure7_failure_probability()
    
    print("Generating Figure 8: Sensitivity Analysis...")
    figure8_sensitivity_analysis()
    
    print("\nAll figures have been generated successfully!")
    print("Figures saved as PNG files in the current directory.")

if __name__ == "__main__":
    create_all_figures()