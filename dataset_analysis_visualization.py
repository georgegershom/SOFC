#!/usr/bin/env python3
"""
Dataset Analysis and Visualization for Sintering Microstructure Data
===================================================================

This script provides comprehensive analysis and visualization of the generated
sintering microstructure dataset, helping to understand relationships between
process parameters and resulting microstructure.

Author: AI Assistant
Date: October 8, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DatasetAnalyzer:
    """
    Comprehensive analysis and visualization of sintering microstructure dataset.
    """
    
    def __init__(self, df):
        self.df = df
        self.input_features = [
            'ramp_up_rate_C_per_min', 'peak_temperature_C', 'hold_time_hours',
            'cool_down_rate_C_per_min', 'applied_pressure_MPa', 
            'initial_relative_density_percent', 'particle_size_um'
        ]
        self.output_features = [
            'final_relative_density_percent', 'porosity_percent', 
            'grain_size_mean_um', 'pore_size_mean_um'
        ]
    
    def create_correlation_heatmap(self):
        """Create correlation heatmap between key variables."""
        
        # Select numerical columns for correlation
        key_cols = self.input_features + self.output_features
        corr_data = self.df[key_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        
        sns.heatmap(corr_data, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        
        plt.title('Correlation Matrix: Process Parameters vs Microstructure Properties', 
                 fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_temperature_effects(self):
        """Analyze and plot temperature effects on microstructure."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature vs Final Density
        axes[0,0].scatter(self.df['peak_temperature_C'], 
                         self.df['final_relative_density_percent'], 
                         alpha=0.6, s=30)
        axes[0,0].set_xlabel('Peak Temperature (°C)')
        axes[0,0].set_ylabel('Final Relative Density (%)')
        axes[0,0].set_title('Temperature vs Final Density')
        
        # Add trend line
        z = np.polyfit(self.df['peak_temperature_C'], 
                      self.df['final_relative_density_percent'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(self.df['peak_temperature_C'], 
                      p(self.df['peak_temperature_C']), "r--", alpha=0.8)
        
        # Temperature vs Grain Size
        axes[0,1].scatter(self.df['peak_temperature_C'], 
                         self.df['grain_size_mean_um'], 
                         alpha=0.6, s=30, c='orange')
        axes[0,1].set_xlabel('Peak Temperature (°C)')
        axes[0,1].set_ylabel('Mean Grain Size (μm)')
        axes[0,1].set_title('Temperature vs Grain Size')
        
        # Temperature vs Porosity
        axes[1,0].scatter(self.df['peak_temperature_C'], 
                         self.df['porosity_percent'], 
                         alpha=0.6, s=30, c='green')
        axes[1,0].set_xlabel('Peak Temperature (°C)')
        axes[1,0].set_ylabel('Porosity (%)')
        axes[1,0].set_title('Temperature vs Porosity')
        
        # Temperature distribution by atmosphere
        for i, atm in enumerate(self.df['atmosphere'].unique()):
            subset = self.df[self.df['atmosphere'] == atm]
            axes[1,1].hist(subset['peak_temperature_C'], alpha=0.6, 
                          label=atm, bins=20)
        axes[1,1].set_xlabel('Peak Temperature (°C)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Temperature Distribution by Atmosphere')
        axes[1,1].legend()
        
        plt.suptitle('Temperature Effects on Sintering Outcomes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_pressure_effects(self):
        """Analyze pressure effects on microstructure."""
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Separate pressureless and pressure-assisted
        pressureless = self.df[self.df['applied_pressure_MPa'] == 0]
        pressure_assisted = self.df[self.df['applied_pressure_MPa'] > 0]
        
        # Density comparison
        axes[0].boxplot([pressureless['final_relative_density_percent'], 
                        pressure_assisted['final_relative_density_percent']], 
                       labels=['Pressureless', 'Pressure-Assisted'])
        axes[0].set_ylabel('Final Relative Density (%)')
        axes[0].set_title('Density: Pressureless vs Pressure-Assisted')
        
        # Pressure vs Density (for pressure-assisted only)
        if len(pressure_assisted) > 0:
            axes[1].scatter(pressure_assisted['applied_pressure_MPa'], 
                           pressure_assisted['final_relative_density_percent'], 
                           alpha=0.6, s=30, c='red')
            axes[1].set_xlabel('Applied Pressure (MPa)')
            axes[1].set_ylabel('Final Relative Density (%)')
            axes[1].set_title('Pressure vs Density (Pressure-Assisted Only)')
        
        # Grain size comparison
        axes[2].boxplot([pressureless['grain_size_mean_um'], 
                        pressure_assisted['grain_size_mean_um']], 
                       labels=['Pressureless', 'Pressure-Assisted'])
        axes[2].set_ylabel('Mean Grain Size (μm)')
        axes[2].set_title('Grain Size: Pressureless vs Pressure-Assisted')
        
        plt.suptitle('Pressure Effects on Sintering Outcomes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def plot_time_effects(self):
        """Analyze hold time effects."""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Hold time vs grain size
        axes[0].scatter(self.df['hold_time_hours'], 
                       self.df['grain_size_mean_um'], 
                       alpha=0.6, s=30)
        axes[0].set_xlabel('Hold Time (hours)')
        axes[0].set_ylabel('Mean Grain Size (μm)')
        axes[0].set_title('Hold Time vs Grain Size')
        axes[0].set_xscale('log')
        
        # Hold time vs density
        axes[1].scatter(self.df['hold_time_hours'], 
                       self.df['final_relative_density_percent'], 
                       alpha=0.6, s=30, c='purple')
        axes[1].set_xlabel('Hold Time (hours)')
        axes[1].set_ylabel('Final Relative Density (%)')
        axes[1].set_title('Hold Time vs Final Density')
        axes[1].set_xscale('log')
        
        plt.suptitle('Time Effects on Sintering Outcomes', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_pca_analysis(self):
        """Perform PCA analysis on the dataset."""
        
        # Prepare data for PCA
        feature_cols = self.input_features
        X = self.df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Explained variance plot
        axes[0].bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                   pca.explained_variance_ratio_)
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('PCA: Explained Variance by Component')
        
        # Cumulative explained variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        axes[0].plot(range(1, len(cumsum) + 1), cumsum, 'ro-', alpha=0.7)
        axes[0].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, 
                       label='95% Variance')
        axes[0].legend()
        
        # PCA scatter plot (first two components)
        scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                                 c=self.df['final_relative_density_percent'], 
                                 cmap='viridis', alpha=0.6, s=30)
        axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        axes[1].set_title('PCA: First Two Components (colored by final density)')
        plt.colorbar(scatter, ax=axes[1], label='Final Density (%)')
        
        plt.suptitle('Principal Component Analysis of Process Parameters', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, pca, scaler
    
    def feature_importance_analysis(self):
        """Analyze feature importance using Random Forest."""
        
        # Prepare data
        X = self.df[self.input_features]
        
        # Analyze importance for each output
        results = {}
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, target in enumerate(self.output_features):
            y = self.df[target]
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            results[target] = dict(zip(self.input_features, importance))
            
            # Plot
            sorted_idx = np.argsort(importance)
            axes[i].barh(range(len(importance)), importance[sorted_idx])
            axes[i].set_yticks(range(len(importance)))
            axes[i].set_yticklabels([self.input_features[j] for j in sorted_idx])
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'Feature Importance for {target}')
        
        plt.suptitle('Feature Importance Analysis (Random Forest)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig, results
    
    def create_process_maps(self):
        """Create process maps showing optimal regions."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Temperature vs Time colored by density
        scatter1 = axes[0,0].scatter(self.df['peak_temperature_C'], 
                                    self.df['hold_time_hours'],
                                    c=self.df['final_relative_density_percent'], 
                                    cmap='RdYlGn', s=50, alpha=0.7)
        axes[0,0].set_xlabel('Peak Temperature (°C)')
        axes[0,0].set_ylabel('Hold Time (hours)')
        axes[0,0].set_title('Process Map: Density')
        axes[0,0].set_yscale('log')
        plt.colorbar(scatter1, ax=axes[0,0], label='Final Density (%)')
        
        # Temperature vs Pressure colored by grain size
        scatter2 = axes[0,1].scatter(self.df['peak_temperature_C'], 
                                    self.df['applied_pressure_MPa'],
                                    c=self.df['grain_size_mean_um'], 
                                    cmap='plasma', s=50, alpha=0.7)
        axes[0,1].set_xlabel('Peak Temperature (°C)')
        axes[0,1].set_ylabel('Applied Pressure (MPa)')
        axes[0,1].set_title('Process Map: Grain Size')
        plt.colorbar(scatter2, ax=axes[0,1], label='Grain Size (μm)')
        
        # Particle size vs Temperature colored by porosity
        scatter3 = axes[1,0].scatter(self.df['particle_size_um'], 
                                    self.df['peak_temperature_C'],
                                    c=self.df['porosity_percent'], 
                                    cmap='RdYlBu_r', s=50, alpha=0.7)
        axes[1,0].set_xlabel('Particle Size (μm)')
        axes[1,0].set_ylabel('Peak Temperature (°C)')
        axes[1,0].set_title('Process Map: Porosity')
        axes[1,0].set_xscale('log')
        plt.colorbar(scatter3, ax=axes[1,0], label='Porosity (%)')
        
        # Initial density vs final density colored by temperature
        scatter4 = axes[1,1].scatter(self.df['initial_relative_density_percent'], 
                                    self.df['final_relative_density_percent'],
                                    c=self.df['peak_temperature_C'], 
                                    cmap='coolwarm', s=50, alpha=0.7)
        axes[1,1].set_xlabel('Initial Relative Density (%)')
        axes[1,1].set_ylabel('Final Relative Density (%)')
        axes[1,1].set_title('Densification Map')
        axes[1,1].plot([50, 100], [50, 100], 'k--', alpha=0.5, label='No change')
        axes[1,1].legend()
        plt.colorbar(scatter4, ax=axes[1,1], label='Temperature (°C)')
        
        plt.suptitle('Sintering Process Maps', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def generate_comprehensive_report(self, save_plots=True):
        """Generate comprehensive analysis report with all visualizations."""
        
        print("Generating comprehensive dataset analysis...")
        
        figures = {}
        
        # Create all visualizations
        print("  - Creating correlation heatmap...")
        figures['correlation'] = self.create_correlation_heatmap()
        
        print("  - Analyzing temperature effects...")
        figures['temperature'] = self.plot_temperature_effects()
        
        print("  - Analyzing pressure effects...")
        figures['pressure'] = self.plot_pressure_effects()
        
        print("  - Analyzing time effects...")
        figures['time'] = self.plot_time_effects()
        
        print("  - Performing PCA analysis...")
        figures['pca'], pca_model, scaler = self.create_pca_analysis()
        
        print("  - Analyzing feature importance...")
        figures['importance'], importance_results = self.feature_importance_analysis()
        
        print("  - Creating process maps...")
        figures['process_maps'] = self.create_process_maps()
        
        if save_plots:
            # Save all plots
            import os
            os.makedirs('/workspace/analysis_plots', exist_ok=True)
            
            for name, fig in figures.items():
                fig.savefig(f'/workspace/analysis_plots/{name}_analysis.png', 
                           dpi=300, bbox_inches='tight')
                print(f"    Saved: /workspace/analysis_plots/{name}_analysis.png")
        
        # Generate text report
        report_path = '/workspace/analysis_plots/analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write("SINTERING MICROSTRUCTURE DATASET ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("DATASET OVERVIEW:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total samples: {len(self.df)}\n")
            f.write(f"Input features: {len(self.input_features)}\n")
            f.write(f"Output features: {len(self.output_features)}\n\n")
            
            f.write("KEY CORRELATIONS:\n")
            f.write("-" * 20 + "\n")
            corr_matrix = self.df[self.input_features + self.output_features].corr()
            
            # Find strongest correlations with outputs
            for output in self.output_features:
                correlations = corr_matrix[output][self.input_features].abs().sort_values(ascending=False)
                f.write(f"\n{output}:\n")
                for feature, corr in correlations.head(3).items():
                    f.write(f"  - {feature}: {corr:.3f}\n")
            
            f.write(f"\nFEATURE IMPORTANCE (Random Forest):\n")
            f.write("-" * 40 + "\n")
            for target, importances in importance_results.items():
                f.write(f"\n{target}:\n")
                sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
                for feature, importance in sorted_features[:3]:
                    f.write(f"  - {feature}: {importance:.3f}\n")
        
        print(f"  - Analysis report saved: {report_path}")
        
        return figures, importance_results

def main():
    """Main analysis function."""
    
    print("Loading dataset...")
    
    # Try to load the dataset
    try:
        df = pd.read_csv('/workspace/datasets/sintering_microstructure_dataset.csv')
        print(f"Dataset loaded: {df.shape}")
    except FileNotFoundError:
        print("Dataset not found. Please run the dataset generator first.")
        return None
    
    # Create analyzer
    analyzer = DatasetAnalyzer(df)
    
    # Generate comprehensive analysis
    figures, importance_results = analyzer.generate_comprehensive_report()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nGenerated visualizations:")
    print("  - Correlation heatmap")
    print("  - Temperature effects analysis")
    print("  - Pressure effects analysis") 
    print("  - Time effects analysis")
    print("  - PCA analysis")
    print("  - Feature importance analysis")
    print("  - Process maps")
    print(f"\nAll plots saved in: /workspace/analysis_plots/")
    
    return analyzer, figures, importance_results

if __name__ == "__main__":
    analyzer, figures, importance_results = main()