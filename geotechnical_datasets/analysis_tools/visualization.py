#!/usr/bin/env python3
"""
Visualization Module for Geotechnical Datasets
Provides functions for data visualization and analysis plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class GeotechnicalVisualizer:
    """Visualization tools for geotechnical data"""
    
    def __init__(self, figsize=(12, 8)):
        """Initialize visualizer with default figure size"""
        self.figsize = figsize
        
    def plot_grain_size_distribution(self, df, sample_ids=None):
        """Plot grain size distribution curves"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if sample_ids is None:
            sample_ids = df['sample_id'].unique()[:5]  # Plot first 5 samples
        
        grain_sizes = ['d10_mm', 'd30_mm', 'd50_mm', 'd60_mm']
        percentages = [10, 30, 50, 60]
        
        for sample_id in sample_ids:
            sample_data = df[df['sample_id'] == sample_id]
            if not sample_data.empty:
                sizes = sample_data[grain_sizes].values[0]
                ax.semilogx(sizes, percentages, 'o-', label=sample_id, linewidth=2, markersize=8)
        
        ax.set_xlabel('Grain Size (mm)', fontsize=12)
        ax.set_ylabel('Percent Finer (%)', fontsize=12)
        ax.set_title('Grain Size Distribution Curves', fontsize=14, fontweight='bold')
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc='best')
        
        plt.tight_layout()
        return fig
    
    def plot_plasticity_chart(self, df):
        """Plot Casagrande plasticity chart for clay soils"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # A-line equation: PI = 0.73(LL - 20)
        ll_range = np.linspace(0, 100, 100)
        a_line = 0.73 * (ll_range - 20)
        
        # U-line equation: PI = 0.9(LL - 8)
        u_line = 0.9 * (ll_range - 8)
        
        # Plot lines
        ax.plot(ll_range, a_line, 'k--', label='A-line', linewidth=2)
        ax.plot(ll_range, u_line, 'r--', label='U-line', linewidth=2)
        
        # Plot data points
        scatter = ax.scatter(df['liquid_limit_%'], df['plasticity_index'], 
                           c=df['clay_content_%'], s=100, cmap='YlOrRd', 
                           edgecolors='black', alpha=0.7)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Clay Content (%)', fontsize=12)
        
        ax.set_xlabel('Liquid Limit (%)', fontsize=12)
        ax.set_ylabel('Plasticity Index', fontsize=12)
        ax.set_title('Plasticity Chart (Casagrande)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(df['liquid_limit_%'].max(), 100))
        ax.set_ylim(0, max(df['plasticity_index'].max(), 80))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        
        # Add soil classification regions
        ax.text(30, 10, 'ML/OL', fontsize=10, ha='center')
        ax.text(50, 35, 'MH/OH', fontsize=10, ha='center')
        ax.text(25, 20, 'CL', fontsize=10, ha='center')
        ax.text(70, 50, 'CH', fontsize=10, ha='center')
        
        plt.tight_layout()
        return fig
    
    def plot_liquefaction_assessment(self, df):
        """Plot liquefaction assessment chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CSR vs Safety Factor
        scatter1 = ax1.scatter(df['CSR'], df['safety_factor'], 
                              c=df['relative_density_%'], s=100, 
                              cmap='RdYlGn', edgecolors='black', alpha=0.7)
        ax1.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='FS=1.0')
        ax1.set_xlabel('Cyclic Stress Ratio (CSR)', fontsize=12)
        ax1.set_ylabel('Factor of Safety', fontsize=12)
        ax1.set_title('Liquefaction Safety Assessment', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Relative Density (%)', fontsize=10)
        
        # N1(60) vs CSR/CRR
        ax2.scatter(df['N1_60'], df['CSR'], label='CSR', alpha=0.6, s=80)
        ax2.scatter(df['N1_60'], df['CRR'], label='CRR', alpha=0.6, s=80)
        ax2.set_xlabel('Normalized SPT (N1)60', fontsize=12)
        ax2.set_ylabel('Stress Ratio', fontsize=12)
        ax2.set_title('SPT-based Liquefaction Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_failure_analysis(self, df):
        """Plot failure case analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Failure modes distribution
        failure_counts = df['failure_mode'].value_counts()
        ax1.bar(range(len(failure_counts)), failure_counts.values)
        ax1.set_xticks(range(len(failure_counts)))
        ax1.set_xticklabels(failure_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of Cases', fontsize=12)
        ax1.set_title('Failure Modes Distribution', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Settlement vs Ground Loss
        scatter2 = ax2.scatter(df['ground_loss_%'], df['max_settlement_mm'], 
                              c=df['depth_m'], s=100, cmap='viridis', 
                              edgecolors='black', alpha=0.7)
        ax2.set_xlabel('Ground Loss (%)', fontsize=12)
        ax2.set_ylabel('Maximum Settlement (mm)', fontsize=12)
        ax2.set_title('Settlement vs Ground Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Depth (m)', fontsize=10)
        
        # Economic loss by soil type
        soil_loss = df.groupby('soil_type')['economic_loss_million_USD'].mean()
        ax3.bar(range(len(soil_loss)), soil_loss.values)
        ax3.set_xticks(range(len(soil_loss)))
        ax3.set_xticklabels(soil_loss.index, rotation=45, ha='right')
        ax3.set_ylabel('Average Economic Loss (Million USD)', fontsize=12)
        ax3.set_title('Economic Loss by Soil Type', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Depth vs Damage Severity
        severity_map = {'Minor': 1, 'Moderate': 2, 'Severe': 3, 'Catastrophic': 4}
        df['severity_num'] = df['damage_severity'].map(severity_map)
        
        ax4.boxplot([df[df['severity_num'] == i]['depth_m'].values 
                    for i in range(1, 5)],
                   labels=['Minor', 'Moderate', 'Severe', 'Catastrophic'])
        ax4.set_xlabel('Damage Severity', fontsize=12)
        ax4.set_ylabel('Structure Depth (m)', fontsize=12)
        ax4.set_title('Depth Distribution by Damage Severity', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, df, columns=None, title="Correlation Matrix"):
        """Plot correlation matrix heatmap"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if columns is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = numeric_cols[:15]  # Limit to first 15 numeric columns
        
        corr_matrix = df[columns].corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def plot_time_series_monitoring(self, df, case_id, parameters=None):
        """Plot time series monitoring data"""
        case_data = df[df['case_id'] == case_id].copy()
        
        if parameters is None:
            parameters = ['vertical_displacement_mm', 'horizontal_displacement_mm', 
                         'pore_pressure_kPa', 'stress_MPa']
        
        n_params = len(parameters)
        fig, axes = plt.subplots(n_params, 1, figsize=(12, 3*n_params), sharex=True)
        
        if n_params == 1:
            axes = [axes]
        
        for i, param in enumerate(parameters):
            if param in case_data.columns:
                axes[i].plot(case_data['timestamp'], case_data[param], 
                           'o-', linewidth=2, markersize=6)
                axes[i].set_ylabel(param.replace('_', ' ').title(), fontsize=11)
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                x_numeric = np.arange(len(case_data))
                z = np.polyfit(x_numeric, case_data[param].values, 1)
                p = np.poly1d(z)
                axes[i].plot(case_data['timestamp'], p(x_numeric), 
                           'r--', alpha=0.5, label='Trend')
                axes[i].legend(loc='best')
        
        axes[-1].set_xlabel('Time', fontsize=12)
        axes[0].set_title(f'Monitoring Data for Case {case_id}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_spatial_distribution(self, df, parameter, lat_col='latitude', lon_col='longitude'):
        """Plot spatial distribution of a parameter"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        scatter = ax.scatter(df[lon_col], df[lat_col], 
                           c=df[parameter], s=200, cmap='YlOrRd', 
                           edgecolors='black', alpha=0.7)
        
        # Add labels for each point
        for idx, row in df.iterrows():
            if 'grid_id' in row:
                ax.annotate(row['grid_id'], (row[lon_col], row[lat_col]), 
                          fontsize=8, ha='center')
        
        cbar = plt.colorbar(scatter)
        cbar.set_label(parameter.replace('_', ' ').title(), fontsize=12)
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Spatial Distribution of {parameter.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    from data_loader import GeotechnicalDataLoader
    
    # Load data
    loader = GeotechnicalDataLoader()
    sandy_data, sandy_merged = loader.load_sandy_soils()
    clay_data, clay_merged = loader.load_clay_soils()
    case_data = loader.load_case_studies()
    spatial_data = loader.load_spatial_data()
    
    # Initialize visualizer
    viz = GeotechnicalVisualizer()
    
    # Create sample plots
    print("Generating sample visualizations...")
    
    # 1. Grain size distribution
    fig1 = viz.plot_grain_size_distribution(sandy_merged)
    plt.savefig('grain_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plasticity chart
    fig2 = viz.plot_plasticity_chart(clay_merged)
    plt.savefig('plasticity_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Liquefaction assessment
    fig3 = viz.plot_liquefaction_assessment(sandy_merged)
    plt.savefig('liquefaction_assessment.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Failure analysis
    fig4 = viz.plot_failure_analysis(case_data['failures'])
    plt.savefig('failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved successfully!")