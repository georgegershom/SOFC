#!/usr/bin/env python3
"""
Data Visualization Module for Welding Inverse Design Dataset
============================================================

This module provides comprehensive visualization capabilities for the welding
inverse design dataset, including parameter distributions, correlation analysis,
and performance metrics visualization.

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class WeldingDataVisualizer:
    """Comprehensive visualization class for welding dataset analysis."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.input_features = [col for col in dataset.columns 
                              if col not in ['weld_id', 'data_source', 'fidelity_level', 
                                           'dataset_version', 'generation_date', 'total_samples',
                                           'quality_score', 'thermal_cycling_applied', 'max_temp_celsius']]
        self.output_features = [col for col in dataset.columns 
                               if col in ['nugget_width', 'penetration_depth', 'haz_width',
                                        'tensile_shear_strength', 'peel_strength', 'contact_resistance',
                                        'thermal_cycles_to_failure', 'strength_degradation_pct',
                                        'resistance_increase_pct', 'imc_thickness', 'creep_time_to_failure']]
    
    def plot_parameter_distributions(self, save_path=None):
        """Plot distributions of input parameters by data source."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        input_cols = [col for col in self.input_features if col != 'material_combination']
        
        for i, col in enumerate(input_cols[:12]):  # Limit to 12 plots
            ax = axes[i]
            
            for source in self.dataset['data_source'].unique():
                data = self.dataset[self.dataset['data_source'] == source][col]
                ax.hist(data, alpha=0.6, label=source, bins=30)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(input_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_heatmap(self, save_path=None):
        """Plot correlation heatmap for all features."""
        # Select numeric columns only
        numeric_cols = self.dataset.select_dtypes(include=[np.number]).columns
        corr_matrix = self.dataset[numeric_cols].corr()
        
        plt.figure(figsize=(16, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_output_distributions(self, save_path=None):
        """Plot distributions of output properties."""
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, col in enumerate(self.output_features[:12]):
            ax = axes[i]
            
            for source in self.dataset['data_source'].unique():
                data = self.dataset[self.dataset['data_source'] == source][col]
                ax.hist(data, alpha=0.6, label=source, bins=30)
            
            ax.set_title(f'{col} Distribution')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(self.output_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_3d_parameter_space(self, x_col, y_col, z_col, color_col, save_path=None):
        """Plot 3D parameter space visualization."""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(self.dataset[x_col], self.dataset[y_col], self.dataset[z_col],
                           c=self.dataset[color_col], cmap='viridis', alpha=0.6)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.colorbar(scatter, label=color_col)
        plt.title(f'3D Parameter Space: {x_col} vs {y_col} vs {z_col}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, save_path=None):
        """Plot key performance metrics by data source."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = ['tensile_shear_strength', 'contact_resistance', 'thermal_cycles_to_failure',
                  'strength_degradation_pct', 'imc_thickness', 'creep_time_to_failure']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Box plot by data source
            sns.boxplot(data=self.dataset, x='data_source', y=metric, ax=ax)
            ax.set_title(f'{metric} by Data Source')
            ax.set_xlabel('Data Source')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_material_analysis(self, save_path=None):
        """Plot analysis by material combination."""
        # Map material combinations
        material_map = {0: 'Cu-Al', 1: 'Al-Al', 2: 'Cu-Steel', 3: 'Al-Steel'}
        self.dataset['material_name'] = self.dataset['material_combination'].map(material_map)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = ['tensile_shear_strength', 'contact_resistance', 'thermal_cycles_to_failure',
                  'strength_degradation_pct', 'imc_thickness', 'creep_time_to_failure']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Box plot by material combination
            sns.boxplot(data=self.dataset, x='material_name', y=metric, ax=ax)
            ax.set_title(f'{metric} by Material Combination')
            ax.set_xlabel('Material Combination')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_energy_density_analysis(self, save_path=None):
        """Plot analysis based on energy density calculations."""
        # Calculate energy density
        self.dataset['energy_density'] = (self.dataset['laser_power'] / 
                                        (self.dataset['welding_speed'] * self.dataset['material_thickness']))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        metrics = ['nugget_width', 'penetration_depth', 'tensile_shear_strength',
                  'contact_resistance', 'thermal_cycles_to_failure', 'strength_degradation_pct']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Scatter plot: energy density vs metric
            scatter = ax.scatter(self.dataset['energy_density'], self.dataset[metric],
                               c=self.dataset['material_combination'], cmap='tab10', alpha=0.6)
            ax.set_xlabel('Energy Density (W·s/mm²)')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Energy Density')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_extreme_temperature_analysis(self, save_path=None):
        """Plot extreme temperature performance analysis."""
        # Filter for extreme temperature tested samples
        extreme_data = self.dataset[self.dataset.get('thermal_cycling_applied', False) == True]
        
        if len(extreme_data) == 0:
            print("No extreme temperature data available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # Temperature cycling performance
        ax1 = axes[0]
        ax1.scatter(extreme_data['max_temp_celsius'], extreme_data['thermal_cycles_to_failure'],
                   c=extreme_data['material_combination'], cmap='tab10', alpha=0.7)
        ax1.set_xlabel('Maximum Temperature (°C)')
        ax1.set_ylabel('Cycles to Failure')
        ax1.set_title('Thermal Cycling Performance')
        ax1.grid(True, alpha=0.3)
        
        # Strength degradation
        ax2 = axes[1]
        ax2.scatter(extreme_data['max_temp_celsius'], extreme_data['strength_degradation_pct'],
                   c=extreme_data['material_combination'], cmap='tab10', alpha=0.7)
        ax2.set_xlabel('Maximum Temperature (°C)')
        ax2.set_ylabel('Strength Degradation (%)')
        ax2.set_title('Strength Degradation vs Temperature')
        ax2.grid(True, alpha=0.3)
        
        # IMC thickness
        ax3 = axes[2]
        ax3.scatter(extreme_data['max_temp_celsius'], extreme_data['imc_thickness'],
                   c=extreme_data['material_combination'], cmap='tab10', alpha=0.7)
        ax3.set_xlabel('Maximum Temperature (°C)')
        ax3.set_ylabel('IMC Thickness (µm)')
        ax3.set_title('IMC Growth vs Temperature')
        ax3.grid(True, alpha=0.3)
        
        # Creep performance
        ax4 = axes[3]
        ax4.scatter(extreme_data['max_temp_celsius'], extreme_data['creep_time_to_failure'],
                   c=extreme_data['material_combination'], cmap='tab10', alpha=0.7)
        ax4.set_xlabel('Maximum Temperature (°C)')
        ax4.set_ylabel('Creep Time to Failure (hours)')
        ax4.set_title('Creep Performance vs Temperature')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self, save_path=None):
        """Plot PCA analysis of the dataset."""
        # Prepare data for PCA
        numeric_data = self.dataset.select_dtypes(include=[np.number])
        numeric_data = numeric_data.drop(columns=['material_combination', 'thermal_cycling_applied', 'max_temp_celsius'], errors='ignore')
        
        # Standardize data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_data)
        
        # Plot PCA results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # PCA by data source
        for source in self.dataset['data_source'].unique():
            mask = self.dataset['data_source'] == source
            ax1.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       label=source, alpha=0.6)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('PCA by Data Source')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # PCA by material combination
        for material in self.dataset['material_combination'].unique():
            mask = self.dataset['material_combination'] == material
            ax2.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       label=f'Material {material}', alpha=0.6)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title('PCA by Material Combination')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, save_path='welding_dashboard.html'):
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Parameter Distributions', 'Performance Metrics',
                          'Energy Density Analysis', 'Material Analysis',
                          'Extreme Temperature Performance', 'Correlation Matrix'),
            specs=[[{"type": "histogram"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "box"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Add traces (simplified for demonstration)
        # This would be expanded with actual Plotly traces
        
        fig.update_layout(
            title="Welding Inverse Design Dataset Dashboard",
            height=1200,
            showlegend=True
        )
        
        fig.write_html(save_path)
        print(f"Interactive dashboard saved to {save_path}")
    
    def generate_comprehensive_report(self, save_dir='visualization_outputs'):
        """Generate comprehensive visualization report."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("Generating comprehensive visualization report...")
        
        # Generate all visualizations
        self.plot_parameter_distributions(f"{save_dir}/parameter_distributions.png")
        self.plot_correlation_heatmap(f"{save_dir}/correlation_heatmap.png")
        self.plot_output_distributions(f"{save_dir}/output_distributions.png")
        self.plot_performance_metrics(f"{save_dir}/performance_metrics.png")
        self.plot_material_analysis(f"{save_dir}/material_analysis.png")
        self.plot_energy_density_analysis(f"{save_dir}/energy_density_analysis.png")
        self.plot_extreme_temperature_analysis(f"{save_dir}/extreme_temperature_analysis.png")
        self.plot_pca_analysis(f"{save_dir}/pca_analysis.png")
        
        # Create interactive dashboard
        self.create_interactive_dashboard(f"{save_dir}/interactive_dashboard.html")
        
        print(f"All visualizations saved to {save_dir}/")

def main():
    """Test the visualization module."""
    # Load dataset (assuming it exists)
    try:
        dataset = pd.read_csv('welding_master_dataset.csv')
        visualizer = WeldingDataVisualizer(dataset)
        visualizer.generate_comprehensive_report()
    except FileNotFoundError:
        print("Dataset not found. Please run the main dataset generator first.")

if __name__ == "__main__":
    main()