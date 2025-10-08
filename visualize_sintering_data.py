#!/usr/bin/env python3
"""
Visualization script for sintering process and microstructure dataset
Generates comprehensive plots for analysis and optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all dataset files"""
    sintering_params = pd.read_csv('sintering_parameters.csv')
    microstructure = pd.read_csv('microstructure_results.csv')
    grain_dist = pd.read_csv('grain_size_distribution.csv')
    pore_dist = pd.read_csv('pore_size_distribution.csv')
    grain_boundary = pd.read_csv('grain_boundary_characteristics.csv')
    
    # Merge main datasets
    full_data = pd.merge(sintering_params, microstructure, on='Sample_ID')
    
    return full_data, grain_dist, pore_dist, grain_boundary

def plot_process_microstructure_correlations(data):
    """Create correlation matrix between process parameters and microstructure"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature vs Relative Density
    ax = axes[0, 0]
    ax.scatter(data['Max_Temperature_C'], data['Final_Relative_Density_percent'], 
               alpha=0.6, s=50)
    z = np.polyfit(data['Max_Temperature_C'], data['Final_Relative_Density_percent'], 2)
    p = np.poly1d(z)
    x_line = np.linspace(data['Max_Temperature_C'].min(), data['Max_Temperature_C'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)
    ax.set_xlabel('Maximum Temperature (°C)', fontsize=12)
    ax.set_ylabel('Final Relative Density (%)', fontsize=12)
    ax.set_title('Temperature Effect on Densification', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Hold Time vs Grain Size
    ax = axes[0, 1]
    ax.scatter(data['Hold_Time_min'], data['Avg_Grain_Size_um'], 
               alpha=0.6, s=50)
    z = np.polyfit(data['Hold_Time_min'], data['Avg_Grain_Size_um'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(data['Hold_Time_min'].min(), data['Hold_Time_min'].max(), 100)
    ax.plot(x_line, p(x_line), "r-", alpha=0.8, linewidth=2)
    ax.set_xlabel('Hold Time (min)', fontsize=12)
    ax.set_ylabel('Average Grain Size (μm)', fontsize=12)
    ax.set_title('Hold Time Effect on Grain Growth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Applied Pressure vs Porosity
    ax = axes[1, 0]
    pressure_data = data[data['Applied_Pressure_MPa'] > 0]
    if len(pressure_data) > 0:
        ax.scatter(pressure_data['Applied_Pressure_MPa'], 
                  pressure_data['Final_Porosity_percent'], 
                  alpha=0.6, s=50)
        ax.set_xlabel('Applied Pressure (MPa)', fontsize=12)
        ax.set_ylabel('Final Porosity (%)', fontsize=12)
        ax.set_title('Pressure Effect on Porosity', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No pressure-assisted sintering data', 
                ha='center', va='center', transform=ax.transAxes)
    ax.grid(True, alpha=0.3)
    
    # Heating Rate vs Grain Boundary Density
    ax = axes[1, 1]
    ax.scatter(data['Temperature_Ramp_Rate_C_min'], 
              data['Grain_Boundary_Density_mm_per_mm2'], 
              alpha=0.6, s=50)
    ax.set_xlabel('Heating Rate (°C/min)', fontsize=12)
    ax.set_ylabel('Grain Boundary Density (mm/mm²)', fontsize=12)
    ax.set_title('Heating Rate Effect on GB Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('process_microstructure_correlations.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_grain_size_distributions(grain_dist):
    """Plot grain size distribution for selected samples"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    samples = ['S001', 'S005', 'S020', 'S029']
    titles = ['Low Temp (1200°C)', 'Medium Temp (1400°C)', 
              'With Pressure (40 MPa)', 'High Temp (1500°C)']
    
    for idx, (sample, title) in enumerate(zip(samples, titles)):
        ax = axes[idx // 2, idx % 2]
        sample_data = grain_dist[grain_dist['Sample_ID'] == sample]
        
        if not sample_data.empty:
            # Extract size ranges and frequencies
            sizes = []
            freqs = []
            for _, row in sample_data.iterrows():
                size_range = row['Size_Class_um']
                if '-' in size_range:
                    min_size, max_size = map(float, size_range.split('-'))
                    sizes.append((min_size + max_size) / 2)
                    freqs.append(row['Frequency'])
            
            ax.bar(sizes, freqs, width=np.diff(sizes)[0] if len(sizes) > 1 else 1, 
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel('Grain Size (μm)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{sample}: {title}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Grain Size Distributions Under Different Conditions', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('grain_size_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_pore_evolution(data):
    """Plot porosity evolution with temperature"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Total porosity vs temperature
    ax = axes[0]
    temps = sorted(data['Max_Temperature_C'].unique())
    avg_porosity = []
    std_porosity = []
    for temp in temps:
        temp_data = data[data['Max_Temperature_C'] == temp]
        avg_porosity.append(temp_data['Final_Porosity_percent'].mean())
        std_porosity.append(temp_data['Final_Porosity_percent'].std())
    
    ax.errorbar(temps, avg_porosity, yerr=std_porosity, 
                marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    ax.set_xlabel('Maximum Temperature (°C)', fontsize=12)
    ax.set_ylabel('Final Porosity (%)', fontsize=12)
    ax.set_title('Porosity Evolution with Temperature', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Open vs Closed Porosity
    ax = axes[1]
    ax.scatter(data['Open_Porosity_percent'], data['Closed_Porosity_percent'], 
               c=data['Max_Temperature_C'], cmap='coolwarm', s=50, alpha=0.7)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=11)
    ax.set_xlabel('Open Porosity (%)', fontsize=12)
    ax.set_ylabel('Closed Porosity (%)', fontsize=12)
    ax.set_title('Open vs Closed Porosity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Pore size vs Temperature
    ax = axes[2]
    sc = ax.scatter(data['Max_Temperature_C'], data['Avg_Pore_Size_um'], 
                    c=data['Final_Relative_Density_percent'], 
                    cmap='viridis', s=50, alpha=0.7)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Relative Density (%)', fontsize=11)
    ax.set_xlabel('Maximum Temperature (°C)', fontsize=12)
    ax.set_ylabel('Average Pore Size (μm)', fontsize=12)
    ax.set_title('Pore Size Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pore_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimization_landscape(data):
    """Create 3D surface plot for optimization visualization"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 6))
    
    # Create grid for interpolation
    temp_range = np.linspace(data['Max_Temperature_C'].min(), 
                            data['Max_Temperature_C'].max(), 20)
    time_range = np.linspace(data['Hold_Time_min'].min(), 
                            data['Hold_Time_min'].max(), 20)
    
    TEMP, TIME = np.meshgrid(temp_range, time_range)
    
    # Interpolate density values
    from scipy.interpolate import griddata
    points = data[['Max_Temperature_C', 'Hold_Time_min']].values
    density_values = data['Final_Relative_Density_percent'].values
    DENSITY = griddata(points, density_values, (TEMP, TIME), method='cubic')
    
    # 3D Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(TEMP, TIME, DENSITY, cmap='viridis', 
                            alpha=0.8, edgecolor='none')
    ax1.scatter(data['Max_Temperature_C'], data['Hold_Time_min'], 
               data['Final_Relative_Density_percent'], 
               c='red', s=20, alpha=0.5)
    ax1.set_xlabel('Temperature (°C)', fontsize=11)
    ax1.set_ylabel('Hold Time (min)', fontsize=11)
    ax1.set_zlabel('Relative Density (%)', fontsize=11)
    ax1.set_title('Density Optimization Landscape', fontsize=13, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(TEMP, TIME, DENSITY, levels=15, cmap='viridis')
    ax2.contour(TEMP, TIME, DENSITY, levels=15, colors='black', 
                alpha=0.3, linewidths=0.5)
    ax2.scatter(data['Max_Temperature_C'], data['Hold_Time_min'], 
               c='red', s=20, alpha=0.5, edgecolors='black')
    ax2.set_xlabel('Temperature (°C)', fontsize=11)
    ax2.set_ylabel('Hold Time (min)', fontsize=11)
    ax2.set_title('Density Contour Map', fontsize=13, fontweight='bold')
    fig.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_grain_boundary_analysis(gb_data, full_data):
    """Analyze grain boundary characteristics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GB Type distribution for a sample
    ax = axes[0, 0]
    sample_gb = gb_data[gb_data['Sample_ID'] == 'S005']
    if not sample_gb.empty:
        gb_types = sample_gb.groupby('GB_Type')['Frequency_Percent'].sum()
        ax.pie(gb_types.values, labels=gb_types.index, autopct='%1.1f%%', 
               startangle=90)
        ax.set_title('Grain Boundary Type Distribution (S005)', 
                    fontsize=13, fontweight='bold')
    
    # Misorientation angle distribution
    ax = axes[0, 1]
    ax.hist(gb_data['Misorientation_Angle_deg'], bins=20, 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Misorientation Angle (degrees)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Misorientation Angle Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # GB Energy vs Temperature
    ax = axes[1, 0]
    # Merge GB data with temperature data
    gb_temp = pd.merge(gb_data, full_data[['Sample_ID', 'Max_Temperature_C']], 
                       on='Sample_ID')
    ax.scatter(gb_temp['Max_Temperature_C'], gb_temp['GB_Energy_J_m2'], 
               alpha=0.3, s=20)
    ax.set_xlabel('Maximum Temperature (°C)', fontsize=11)
    ax.set_ylabel('GB Energy (J/m²)', fontsize=11)
    ax.set_title('GB Energy vs Sintering Temperature', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # GB Mobility correlation
    ax = axes[1, 1]
    ax.scatter(gb_data['GB_Energy_J_m2'], 
              np.log10(gb_data['GB_Mobility_m4_Js']), 
              alpha=0.5, s=20)
    ax.set_xlabel('GB Energy (J/m²)', fontsize=11)
    ax.set_ylabel('log₁₀(GB Mobility) (m⁴/J·s)', fontsize=11)
    ax.set_title('GB Energy-Mobility Relationship', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('grain_boundary_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_statistics(data):
    """Generate summary statistics and save to file"""
    summary = []
    
    summary.append("="*60)
    summary.append("SINTERING DATASET SUMMARY STATISTICS")
    summary.append("="*60)
    summary.append("")
    
    # Process parameters statistics
    summary.append("PROCESS PARAMETERS:")
    summary.append("-"*40)
    params = ['Max_Temperature_C', 'Hold_Time_min', 'Temperature_Ramp_Rate_C_min', 
              'Applied_Pressure_MPa', 'Initial_Green_Density_percent']
    for param in params:
        summary.append(f"{param}:")
        summary.append(f"  Mean: {data[param].mean():.2f}")
        summary.append(f"  Std:  {data[param].std():.2f}")
        summary.append(f"  Min:  {data[param].min():.2f}")
        summary.append(f"  Max:  {data[param].max():.2f}")
        summary.append("")
    
    # Microstructure results statistics
    summary.append("MICROSTRUCTURE RESULTS:")
    summary.append("-"*40)
    results = ['Final_Relative_Density_percent', 'Final_Porosity_percent', 
               'Avg_Grain_Size_um', 'Avg_Pore_Size_um', 'Grain_Boundary_Density_mm_per_mm2']
    for result in results:
        summary.append(f"{result}:")
        summary.append(f"  Mean: {data[result].mean():.2f}")
        summary.append(f"  Std:  {data[result].std():.2f}")
        summary.append(f"  Min:  {data[result].min():.2f}")
        summary.append(f"  Max:  {data[result].max():.2f}")
        summary.append("")
    
    # Correlation analysis
    summary.append("KEY CORRELATIONS:")
    summary.append("-"*40)
    corr1 = data['Max_Temperature_C'].corr(data['Final_Relative_Density_percent'])
    summary.append(f"Temperature vs Density: {corr1:.3f}")
    corr2 = data['Hold_Time_min'].corr(data['Avg_Grain_Size_um'])
    summary.append(f"Hold Time vs Grain Size: {corr2:.3f}")
    corr3 = data['Final_Relative_Density_percent'].corr(data['Final_Porosity_percent'])
    summary.append(f"Density vs Porosity: {corr3:.3f}")
    
    # Save to file
    with open('dataset_summary.txt', 'w') as f:
        f.write('\n'.join(summary))
    
    print('\n'.join(summary))

def main():
    """Main execution function"""
    print("Loading sintering dataset...")
    full_data, grain_dist, pore_dist, grain_boundary = load_data()
    
    print(f"Dataset loaded: {len(full_data)} samples")
    print("\nGenerating visualizations...")
    
    # Generate all plots
    print("1. Process-Microstructure Correlations...")
    plot_process_microstructure_correlations(full_data)
    
    print("2. Grain Size Distributions...")
    plot_grain_size_distributions(grain_dist)
    
    print("3. Pore Evolution Analysis...")
    plot_pore_evolution(full_data)
    
    print("4. Optimization Landscape...")
    plot_optimization_landscape(full_data)
    
    print("5. Grain Boundary Analysis...")
    plot_grain_boundary_analysis(grain_boundary, full_data)
    
    print("6. Generating Summary Statistics...")
    generate_summary_statistics(full_data)
    
    print("\nAll visualizations complete! Check the generated PNG files.")
    print("Summary statistics saved to 'dataset_summary.txt'")

if __name__ == "__main__":
    main()