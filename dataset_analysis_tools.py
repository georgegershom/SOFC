#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Tools
for Fire-Resistant Concrete Microstructural Analysis

This module provides tools for analyzing and visualizing the generated dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.interpolate import interp1d
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Comprehensive analysis tools for the fire-resistant concrete dataset."""
    
    def __init__(self, dataset_path: str):
        """Initialize with dataset path."""
        self.df = pd.read_csv(dataset_path)
        self.metadata = self.load_metadata()
        
    def load_metadata(self) -> Dict:
        """Load dataset metadata."""
        try:
            with open('/workspace/dataset_metadata.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate comprehensive summary statistics."""
        summary_stats = []
        
        for analysis_type in self.df['Analysis_Type'].unique():
            subset = self.df[self.df['Analysis_Type'] == analysis_type]
            
            for metric in subset['Quantitative_Metric'].unique():
                metric_data = subset[subset['Quantitative_Metric'] == metric]
                
                stats_dict = {
                    'Analysis_Type': analysis_type,
                    'Quantitative_Metric': metric,
                    'Count': len(metric_data),
                    'Mean': metric_data['Value'].mean(),
                    'Std_Dev': metric_data['Value'].std(),
                    'Min': metric_data['Value'].min(),
                    'Max': metric_data['Value'].max(),
                    'Median': metric_data['Value'].median(),
                    'Q25': metric_data['Value'].quantile(0.25),
                    'Q75': metric_data['Value'].quantile(0.75),
                    'Unit': metric_data['Unit'].iloc[0] if len(metric_data) > 0 else 'N/A'
                }
                summary_stats.append(stats_dict)
        
        return pd.DataFrame(summary_stats)
    
    def analyze_temperature_evolution(self) -> Dict[str, pd.DataFrame]:
        """Analyze temperature-dependent evolution of key metrics."""
        evolution_data = {}
        
        key_metrics = [
            'Total_Mass_Loss', 'Rubber_Melt_Fraction', 'Crack_Density',
            'Interface_Quality_Index', 'Total_Porosity', 'C3S_Content'
        ]
        
        for metric in key_metrics:
            metric_data = self.df[self.df['Quantitative_Metric'] == metric]
            if len(metric_data) > 0:
                # Group by temperature and rubber content
                evolution = metric_data.groupby(['Temperature_C', 'Rubber_Content_Vol_Percent']).agg({
                    'Value': ['mean', 'std', 'count']
                }).round(4)
                evolution.columns = ['Mean', 'Std_Dev', 'Count']
                evolution_data[metric] = evolution.reset_index()
        
        return evolution_data
    
    def analyze_rubber_effects(self) -> Dict[str, Any]:
        """Analyze rubber-specific effects on microstructural properties."""
        rubber_effects = {}
        
        # Compare rubber content effects at different temperatures
        for temp in [25, 200, 400, 600, 800]:
            temp_data = self.df[self.df['Temperature_C'] == temp]
            
            # Key metrics affected by rubber
            rubber_metrics = [
                'Rubber_Melt_Fraction', 'Gas_Evolution_Pore_Density',
                'Rubber_Void_Volume_Fraction', 'Total_Porosity'
            ]
            
            temp_effects = {}
            for metric in rubber_metrics:
                metric_data = temp_data[temp_data['Quantitative_Metric'] == metric]
                if len(metric_data) > 0:
                    # Correlation between rubber content and metric value
                    correlation = metric_data.groupby('Rubber_Content_Vol_Percent')['Value'].mean()
                    temp_effects[metric] = {
                        'correlation_data': correlation.to_dict(),
                        'r_squared': np.corrcoef(
                            correlation.index, correlation.values
                        )[0,1]**2 if len(correlation) > 1 else 0
                    }
            
            rubber_effects[f'T_{temp}C'] = temp_effects
        
        return rubber_effects
    
    def create_temperature_evolution_plots(self, output_dir: str = "/workspace"):
        """Create comprehensive temperature evolution plots."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Key metrics for temperature evolution
        key_metrics = {
            'SEM': ['Pore_Size_Distribution', 'Crack_Density', 'Interface_Quality_Index'],
            'XRD': ['C3S_Content', 'CH_Content', 'CSH_Content'],
            'TGA': ['Total_Mass_Loss', 'Rubber_Mass_Loss'],
            'MicroCT': ['Total_Porosity', 'Pore_Connectivity', 'Tortuosity']
        }
        
        for analysis_type, metrics in key_metrics.items():
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics):
                if i >= 4:  # Limit to 4 subplots
                    break
                    
                metric_data = self.df[
                    (self.df['Analysis_Type'] == analysis_type) & 
                    (self.df['Quantitative_Metric'] == metric)
                ]
                
                if len(metric_data) > 0:
                    # Plot for different rubber contents
                    for rubber_content in sorted(metric_data['Rubber_Content_Vol_Percent'].unique()):
                        rubber_data = metric_data[metric_data['Rubber_Content_Vol_Percent'] == rubber_content]
                        
                        # Group by temperature and calculate mean ± std
                        temp_evolution = rubber_data.groupby('Temperature_C').agg({
                            'Value': ['mean', 'std']
                        })
                        temp_evolution.columns = ['mean', 'std']
                        temp_evolution = temp_evolution.reset_index()
                        
                        # Plot with error bars
                        axes[i].errorbar(
                            temp_evolution['Temperature_C'],
                            temp_evolution['mean'],
                            yerr=temp_evolution['std'],
                            label=f'{rubber_content}% Rubber',
                            marker='o',
                            capsize=5,
                            capthick=2
                        )
                    
                    axes[i].set_xlabel('Temperature (°C)')
                    axes[i].set_ylabel(f'{metric} ({metric_data["Unit"].iloc[0]})')
                    axes[i].set_title(f'{analysis_type}: {metric}')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(len(metrics), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/temperature_evolution_{analysis_type.lower()}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_rubber_effect_plots(self, output_dir: str = "/workspace"):
        """Create plots showing rubber-specific effects."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Rubber-specific metrics
        rubber_metrics = [
            'Rubber_Melt_Fraction', 'Gas_Evolution_Pore_Density',
            'Rubber_Void_Volume_Fraction', 'Rubber_Void_Sphericity'
        ]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(rubber_metrics):
            metric_data = self.df[self.df['Quantitative_Metric'] == metric]
            
            if len(metric_data) > 0:
                # Create heatmap of rubber content vs temperature
                pivot_data = metric_data.pivot_table(
                    values='Value',
                    index='Rubber_Content_Vol_Percent',
                    columns='Temperature_C',
                    aggfunc='mean'
                )
                
                sns.heatmap(
                    pivot_data,
                    annot=True,
                    fmt='.3f',
                    cmap='viridis',
                    ax=axes[i],
                    cbar_kws={'label': f'{metric} ({metric_data["Unit"].iloc[0]})'}
                )
                
                axes[i].set_title(f'{metric} vs Rubber Content & Temperature')
                axes[i].set_xlabel('Temperature (°C)')
                axes[i].set_ylabel('Rubber Content (vol%)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/rubber_effects_heatmap.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_correlation_matrix(self, output_dir: str = "/workspace"):
        """Create correlation matrix for key quantitative metrics."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Select key metrics for correlation analysis
        key_metrics = [
            'Total_Mass_Loss', 'Rubber_Melt_Fraction', 'Crack_Density',
            'Interface_Quality_Index', 'Total_Porosity', 'Pore_Connectivity',
            'C3S_Content', 'CH_Content', 'CSH_Content'
        ]
        
        # Create pivot table for correlation analysis
        correlation_data = []
        for metric in key_metrics:
            metric_data = self.df[self.df['Quantitative_Metric'] == metric]
            if len(metric_data) > 0:
                # Average across replicates for each sample
                avg_data = metric_data.groupby(['Sample_ID', 'Temperature_C', 'Rubber_Content_Vol_Percent'])['Value'].mean().reset_index()
                avg_data['Metric'] = metric
                correlation_data.append(avg_data)
        
        if correlation_data:
            # Combine all metrics
            combined_data = pd.concat(correlation_data, ignore_index=True)
            
            # Pivot to get metrics as columns
            pivot_data = combined_data.pivot_table(
                values='Value',
                index=['Sample_ID', 'Temperature_C', 'Rubber_Content_Vol_Percent'],
                columns='Metric',
                aggfunc='mean'
            ).reset_index()
            
            # Calculate correlation matrix
            numeric_cols = [col for col in pivot_data.columns if col not in ['Sample_ID', 'Temperature_C', 'Rubber_Content_Vol_Percent']]
            corr_matrix = pivot_data[numeric_cols].corr()
            
            # Create correlation plot
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                cbar_kws={'label': 'Correlation Coefficient'}
            )
            plt.title('Correlation Matrix of Key Quantitative Metrics')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/correlation_matrix.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_phase_transformation_analysis(self) -> Dict[str, Any]:
        """Analyze phase transformations with temperature."""
        phase_analysis = {}
        
        # XRD phases
        xrd_phases = ['C3S', 'C2S', 'C3A', 'C4AF', 'CH', 'CSH', 'CASH', 'AFt', 'AFm']
        
        for phase in xrd_phases:
            phase_data = self.df[
                (self.df['Analysis_Type'] == 'XRD') & 
                (self.df['Quantitative_Metric'] == f'{phase}_Content')
            ]
            
            if len(phase_data) > 0:
                # Analyze temperature-dependent evolution
                temp_evolution = phase_data.groupby('Temperature_C').agg({
                    'Value': ['mean', 'std', 'count']
                }).round(4)
                temp_evolution.columns = ['Mean', 'Std_Dev', 'Count']
                
                # Calculate transformation temperatures
                mean_values = temp_evolution['Mean']
                if len(mean_values) > 1:
                    # Find temperature where content drops significantly
                    max_content = mean_values.max()
                    half_content_temp = None
                    for temp in sorted(mean_values.index):
                        if mean_values[temp] <= max_content * 0.5:
                            half_content_temp = temp
                            break
                    
                    phase_analysis[phase] = {
                        'temperature_evolution': temp_evolution.to_dict(),
                        'max_content': max_content,
                        'half_content_temperature': half_content_temp,
                        'decomposition_rate': self.calculate_decomposition_rate(mean_values)
                    }
        
        return phase_analysis
    
    def calculate_decomposition_rate(self, values: pd.Series) -> float:
        """Calculate average decomposition rate."""
        if len(values) < 2:
            return 0.0
        
        temps = values.index
        contents = values.values
        
        # Calculate rate of change
        rates = []
        for i in range(1, len(temps)):
            rate = (contents[i] - contents[i-1]) / (temps[i] - temps[i-1])
            rates.append(rate)
        
        return np.mean(rates) if rates else 0.0
    
    def export_analysis_results(self, output_dir: str = "/workspace"):
        """Export all analysis results."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics()
        summary_stats.to_csv(f'{output_dir}/summary_statistics.csv', index=False)
        
        # Temperature evolution analysis
        temp_evolution = self.analyze_temperature_evolution()
        with open(f'{output_dir}/temperature_evolution_analysis.json', 'w') as f:
            json.dump({k: v.to_dict() for k, v in temp_evolution.items()}, f, indent=2)
        
        # Rubber effects analysis
        rubber_effects = self.analyze_rubber_effects()
        with open(f'{output_dir}/rubber_effects_analysis.json', 'w') as f:
            json.dump(rubber_effects, f, indent=2)
        
        # Phase transformation analysis
        phase_analysis = self.generate_phase_transformation_analysis()
        with open(f'{output_dir}/phase_transformation_analysis.json', 'w') as f:
            json.dump(phase_analysis, f, indent=2)
        
        # Create visualizations
        self.create_temperature_evolution_plots(output_dir)
        self.create_rubber_effect_plots(output_dir)
        self.create_correlation_matrix(output_dir)
        
        print(f"Analysis results exported to: {output_dir}")
        return output_dir

def main():
    """Main function to run comprehensive dataset analysis."""
    print("Starting comprehensive dataset analysis...")
    
    # Initialize analyzer
    analyzer = DatasetAnalyzer('/workspace/fire_resistant_concrete_dataset.csv')
    
    # Export analysis results
    output_dir = analyzer.export_analysis_results()
    
    print("\n" + "="*60)
    print("DATASET ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analysis results exported to: {output_dir}")
    print("\nGenerated files:")
    print("  - summary_statistics.csv")
    print("  - temperature_evolution_analysis.json")
    print("  - rubber_effects_analysis.json")
    print("  - phase_transformation_analysis.json")
    print("  - temperature_evolution_*.png (visualizations)")
    print("  - rubber_effects_heatmap.png")
    print("  - correlation_matrix.png")

if __name__ == "__main__":
    main()