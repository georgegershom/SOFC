"""
Visualization Script for Sintering Process & Microstructure Dataset
This script creates comprehensive visualizations showing relationships between
sintering parameters and resulting microstructure characteristics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_data():
    """Load the sintering dataset."""
    df = pd.read_csv('sintering_process_microstructure_dataset.csv')
    return df

def create_correlation_heatmap(df):
    """Create correlation heatmap for key numerical features."""
    
    # Select key numerical features
    key_features = [
        'Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
        'Initial_Relative_Density', 'Final_Relative_Density', 
        'Final_Porosity_percent', 'Mean_Grain_Size_um', 
        'Mean_Pore_Size_um', 'Densification_Percent'
    ]
    
    # Create shortened labels for better display
    labels = [
        'Hold Temp', 'Hold Time', 'Pressure',
        'Init. Density', 'Final Density', 
        'Porosity', 'Grain Size', 
        'Pore Size', 'Densification'
    ]
    
    corr_matrix = df[key_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=1,
                xticklabels=labels, yticklabels=labels,
                vmin=-1, vmax=1, cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Correlation Matrix: Sintering Parameters & Microstructure', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('01_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 01_correlation_heatmap.png")
    plt.close()

def create_parameter_vs_microstructure_plots(df):
    """Create scatter plots showing key parameter-microstructure relationships."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Temperature vs Final Density
    ax = axes[0, 0]
    scatter = ax.scatter(df['Hold_Temperature_C'], df['Final_Relative_Density'],
                        c=df['Applied_Pressure_MPa'], cmap='viridis', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Hold Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Final Relative Density', fontweight='bold')
    ax.set_title('Temperature Effect on Densification', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Applied Pressure (MPa)')
    ax.grid(True, alpha=0.3)
    
    # 2. Temperature vs Grain Size
    ax = axes[0, 1]
    scatter = ax.scatter(df['Hold_Temperature_C'], df['Mean_Grain_Size_um'],
                        c=df['Hold_Time_hours'], cmap='plasma', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Hold Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Mean Grain Size (μm)', fontweight='bold')
    ax.set_title('Temperature Effect on Grain Growth', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hold Time (hours)')
    ax.grid(True, alpha=0.3)
    
    # 3. Hold Time vs Grain Size
    ax = axes[0, 2]
    scatter = ax.scatter(df['Hold_Time_hours'], df['Mean_Grain_Size_um'],
                        c=df['Hold_Temperature_C'], cmap='coolwarm', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Hold Time (hours)', fontweight='bold')
    ax.set_ylabel('Mean Grain Size (μm)', fontweight='bold')
    ax.set_title('Time Effect on Grain Growth', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hold Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    # 4. Pressure vs Porosity
    ax = axes[1, 0]
    scatter = ax.scatter(df['Applied_Pressure_MPa'], df['Final_Porosity_percent'],
                        c=df['Hold_Temperature_C'], cmap='coolwarm', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Applied Pressure (MPa)', fontweight='bold')
    ax.set_ylabel('Final Porosity (%)', fontweight='bold')
    ax.set_title('Pressure Effect on Porosity', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hold Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    # 5. Initial vs Final Density
    ax = axes[1, 1]
    scatter = ax.scatter(df['Initial_Relative_Density'], df['Final_Relative_Density'],
                        c=df['Hold_Temperature_C'], cmap='coolwarm', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.plot([0.4, 0.7], [0.4, 0.7], 'k--', alpha=0.5, label='No densification')
    ax.set_xlabel('Initial Relative Density', fontweight='bold')
    ax.set_ylabel('Final Relative Density', fontweight='bold')
    ax.set_title('Densification Progress', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hold Temperature (°C)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Final Density vs Grain Size
    ax = axes[1, 2]
    scatter = ax.scatter(df['Final_Relative_Density'], df['Mean_Grain_Size_um'],
                        c=df['Hold_Temperature_C'], cmap='coolwarm', 
                        s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Final Relative Density', fontweight='bold')
    ax.set_ylabel('Mean Grain Size (μm)', fontweight='bold')
    ax.set_title('Density-Grain Size Relationship', fontweight='bold')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Hold Temperature (°C)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_parameter_microstructure_relationships.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 02_parameter_microstructure_relationships.png")
    plt.close()

def create_atmosphere_comparison(df):
    """Compare microstructure results across different atmospheres."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Final Density by Atmosphere
    ax = axes[0, 0]
    sns.boxplot(data=df, x='Atmosphere', y='Final_Relative_Density', ax=ax, palette='Set2')
    ax.set_ylabel('Final Relative Density', fontweight='bold')
    ax.set_xlabel('Atmosphere', fontweight='bold')
    ax.set_title('Final Density by Atmosphere', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Mean Grain Size by Atmosphere
    ax = axes[0, 1]
    sns.boxplot(data=df, x='Atmosphere', y='Mean_Grain_Size_um', ax=ax, palette='Set2')
    ax.set_ylabel('Mean Grain Size (μm)', fontweight='bold')
    ax.set_xlabel('Atmosphere', fontweight='bold')
    ax.set_title('Grain Size by Atmosphere', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Porosity by Atmosphere
    ax = axes[1, 0]
    sns.violinplot(data=df, x='Atmosphere', y='Final_Porosity_percent', ax=ax, palette='Set2')
    ax.set_ylabel('Final Porosity (%)', fontweight='bold')
    ax.set_xlabel('Atmosphere', fontweight='bold')
    ax.set_title('Porosity Distribution by Atmosphere', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Densification by Atmosphere
    ax = axes[1, 1]
    sns.boxplot(data=df, x='Atmosphere', y='Densification_Percent', ax=ax, palette='Set2')
    ax.set_ylabel('Densification (%)', fontweight='bold')
    ax.set_xlabel('Atmosphere', fontweight='bold')
    ax.set_title('Densification by Atmosphere', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('03_atmosphere_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 03_atmosphere_comparison.png")
    plt.close()

def create_process_window_analysis(df):
    """Create 2D process window maps."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Temperature-Time window for Final Density
    ax = axes[0]
    scatter = ax.scatter(df['Hold_Temperature_C'], df['Hold_Time_hours'],
                        c=df['Final_Relative_Density'], cmap='RdYlGn', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5,
                        vmin=0.55, vmax=0.80)
    ax.set_xlabel('Hold Temperature (°C)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Hold Time (hours)', fontweight='bold', fontsize=12)
    ax.set_title('Process Window: Final Relative Density', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Final Relative Density', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add contour lines
    from scipy.interpolate import griddata
    grid_x, grid_y = np.mgrid[df['Hold_Temperature_C'].min():df['Hold_Temperature_C'].max():100j,
                               df['Hold_Time_hours'].min():df['Hold_Time_hours'].max():100j]
    grid_z = griddata((df['Hold_Temperature_C'], df['Hold_Time_hours']), 
                      df['Final_Relative_Density'], (grid_x, grid_y), method='cubic')
    contours = ax.contour(grid_x, grid_y, grid_z, levels=5, colors='black', alpha=0.3, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8)
    
    # 2. Temperature-Time window for Grain Size
    ax = axes[1]
    scatter = ax.scatter(df['Hold_Temperature_C'], df['Hold_Time_hours'],
                        c=df['Mean_Grain_Size_um'], cmap='YlOrRd', 
                        s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Hold Temperature (°C)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Hold Time (hours)', fontweight='bold', fontsize=12)
    ax.set_title('Process Window: Mean Grain Size', fontweight='bold', fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Grain Size (μm)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add contour lines
    grid_z = griddata((df['Hold_Temperature_C'], df['Hold_Time_hours']), 
                      df['Mean_Grain_Size_um'], (grid_x, grid_y), method='cubic')
    contours = ax.contour(grid_x, grid_y, grid_z, levels=5, colors='black', alpha=0.3, linewidths=1)
    ax.clabel(contours, inline=True, fontsize=8)
    
    plt.tight_layout()
    plt.savefig('04_process_window_maps.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 04_process_window_maps.png")
    plt.close()

def create_distribution_plots(df):
    """Create distribution plots for key microstructure features."""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # 1. Final Relative Density Distribution
    ax = axes[0, 0]
    ax.hist(df['Final_Relative_Density'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(df['Final_Relative_Density'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {df["Final_Relative_Density"].mean():.3f}')
    ax.set_xlabel('Final Relative Density', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Final Density Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Mean Grain Size Distribution
    ax = axes[0, 1]
    ax.hist(df['Mean_Grain_Size_um'], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
    ax.axvline(df['Mean_Grain_Size_um'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {df["Mean_Grain_Size_um"].mean():.2f} μm')
    ax.set_xlabel('Mean Grain Size (μm)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Grain Size Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Final Porosity Distribution
    ax = axes[0, 2]
    ax.hist(df['Final_Porosity_percent'], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax.axvline(df['Final_Porosity_percent'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {df["Final_Porosity_percent"].mean():.2f}%')
    ax.set_xlabel('Final Porosity (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Porosity Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Hold Temperature Distribution
    ax = axes[1, 0]
    ax.hist(df['Hold_Temperature_C'], bins=25, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(df['Hold_Temperature_C'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {df["Hold_Temperature_C"].mean():.1f}°C')
    ax.set_xlabel('Hold Temperature (°C)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Temperature Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Applied Pressure Distribution
    ax = axes[1, 1]
    pressure_counts = df['Applied_Pressure_MPa'].value_counts().sort_index()
    ax.bar(pressure_counts.index, pressure_counts.values, edgecolor='black', alpha=0.7, color='purple')
    ax.set_xlabel('Applied Pressure (MPa)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Pressure Distribution', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Densification Distribution
    ax = axes[1, 2]
    ax.hist(df['Densification_Percent'], bins=30, edgecolor='black', alpha=0.7, color='pink')
    ax.axvline(df['Densification_Percent'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean = {df["Densification_Percent"].mean():.2f}%')
    ax.set_xlabel('Densification (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Densification Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('05_distributions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 05_distributions.png")
    plt.close()

def create_pairplot(df):
    """Create a pairplot for key variables."""
    
    key_vars = ['Hold_Temperature_C', 'Hold_Time_hours', 'Applied_Pressure_MPa',
                'Final_Relative_Density', 'Mean_Grain_Size_um', 'Final_Porosity_percent']
    
    # Sample if too many points (for clearer visualization)
    if len(df) > 100:
        df_sample = df.sample(n=100, random_state=42)
    else:
        df_sample = df
    
    pairplot = sns.pairplot(df_sample[key_vars + ['Atmosphere']], 
                           hue='Atmosphere', palette='Set2',
                           diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
    pairplot.fig.suptitle('Pairwise Relationships: Process Parameters & Microstructure', 
                         y=1.02, fontsize=16, fontweight='bold')
    plt.savefig('06_pairplot.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_pairplot.png")
    plt.close()

def main():
    """Main function to generate all visualizations."""
    
    print("="*80)
    print("SINTERING DATASET VISUALIZATION")
    print("="*80)
    print("\nLoading dataset...")
    
    df = load_data()
    print(f"✓ Loaded {len(df)} samples")
    
    print("\nGenerating visualizations...")
    print("-" * 80)
    
    # Generate all plots
    create_correlation_heatmap(df)
    create_parameter_vs_microstructure_plots(df)
    create_atmosphere_comparison(df)
    create_process_window_analysis(df)
    create_distribution_plots(df)
    create_pairplot(df)
    
    print("-" * 80)
    print("\n✓ All visualizations generated successfully!")
    print("\nGenerated files:")
    print("  1. 01_correlation_heatmap.png")
    print("  2. 02_parameter_microstructure_relationships.png")
    print("  3. 03_atmosphere_comparison.png")
    print("  4. 04_process_window_maps.png")
    print("  5. 05_distributions.png")
    print("  6. 06_pairplot.png")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()