"""
Dataset Visualization and Analysis Tools
======================================

This module provides comprehensive visualization and analysis tools for the residual stress dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import h5py
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class DatasetVisualizer:
    """Comprehensive visualization and analysis for residual stress datasets."""
    
    def __init__(self, dataset_path=None, dataset=None):
        """Initialize with either a file path or dataset dictionary."""
        if dataset_path:
            self.load_dataset(dataset_path)
        elif dataset:
            self.dataset = dataset
            self.df = dataset['main_dataset']
        else:
            raise ValueError("Either dataset_path or dataset must be provided")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def load_dataset(self, filepath):
        """Load dataset from file."""
        if filepath.endswith('.csv'):
            self.df = pd.read_csv(filepath)
            self.dataset = {'main_dataset': self.df}
        elif filepath.endswith('.h5'):
            self.dataset = {}
            with h5py.File(filepath, 'r') as f:
                # Load main dataset
                main_data = {}
                for col in f['main_dataset'].keys():
                    main_data[col] = f['main_dataset'][col][:]
                self.df = pd.DataFrame(main_data)
                self.dataset['main_dataset'] = self.df
        else:
            raise ValueError("Unsupported file format. Use .csv or .h5")
    
    def plot_parameter_distributions(self, save_path=None):
        """Plot distributions of all input parameters."""
        
        # Identify parameter categories
        geometric_cols = [col for col in self.df.columns if any(x in col for x in ['length', 'width', 'thickness', 'density'])]
        material_cols = [col for col in self.df.columns if any(x in col for x in ['youngs_modulus', 'cte', 'poisson', 'shrinkage', 'activation'])]
        process_cols = [col for col in self.df.columns if any(x in col for x in ['temp', 'rate', 'time', 'pressure', 'atmosphere'])]
        
        categories = [
            ('Geometric Parameters', geometric_cols),
            ('Material Properties', material_cols),
            ('Process Parameters', process_cols)
        ]
        
        fig, axes = plt.subplots(len(categories), 1, figsize=(15, 5*len(categories)))
        if len(categories) == 1:
            axes = [axes]
        
        for i, (category, cols) in enumerate(categories):
            if not cols:
                continue
                
            n_cols = len(cols)
            n_rows = (n_cols + 3) // 4  # 4 plots per row
            
            fig_cat, axes_cat = plt.subplots(n_rows, 4, figsize=(20, 5*n_rows))
            fig_cat.suptitle(f'{category} Distributions', fontsize=16, fontweight='bold')
            
            if n_rows == 1:
                axes_cat = axes_cat.reshape(1, -1)
            
            for j, col in enumerate(cols):
                row, col_idx = j // 4, j % 4
                ax = axes_cat[row, col_idx]
                
                # Plot histogram with KDE
                self.df[col].hist(bins=50, alpha=0.7, ax=ax, density=True)
                self.df[col].plot.kde(ax=ax, color='red', linewidth=2)
                
                ax.set_title(col.replace('_', ' ').title(), fontsize=10)
                ax.set_xlabel('Value')
                ax.set_ylabel('Density')
                ax.grid(True, alpha=0.3)
            
            # Hide empty subplots
            for j in range(len(cols), n_rows * 4):
                row, col_idx = j // 4, j % 4
                axes_cat[row, col_idx].set_visible(False)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_{category.lower().replace(' ', '_')}_distributions.png", 
                           dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_stress_distributions(self, save_path=None):
        """Plot residual stress distributions by layer."""
        
        stress_cols = [col for col in self.df.columns if 'stress' in col and 'total' in col]
        von_mises_cols = [col for col in self.df.columns if 'von_mises' in col]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Residual Stress Distributions', fontsize=16, fontweight='bold')
        
        # Plot stress by layer
        for i, layer in enumerate(['anode', 'electrolyte', 'cathode']):
            layer_stress_col = f'{layer}_residual_stress_total'
            layer_vm_col = f'{layer}_von_mises_stress'
            
            if layer_stress_col in self.df.columns:
                ax = axes[0, 0] if i == 0 else axes[0, 1] if i == 1 else axes[1, 0]
                
                # Histogram
                self.df[layer_stress_col].hist(bins=50, alpha=0.7, ax=ax, label=f'{layer.title()} Total Stress')
                if layer_vm_col in self.df.columns:
                    self.df[layer_vm_col].hist(bins=50, alpha=0.7, ax=ax, label=f'{layer.title()} von Mises')
                
                ax.set_title(f'{layer.title()} Layer Stress Distribution')
                ax.set_xlabel('Stress (Pa)')
                ax.set_ylabel('Frequency')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Combined stress comparison
        ax = axes[1, 1]
        for layer in ['anode', 'electrolyte', 'cathode']:
            stress_col = f'{layer}_residual_stress_total'
            if stress_col in self.df.columns:
                ax.hist(self.df[stress_col], bins=50, alpha=0.6, label=f'{layer.title()}')
        
        ax.set_title('Stress Comparison Across Layers')
        ax.set_xlabel('Residual Stress (Pa)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_stress_distributions.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_matrix(self, save_path=None):
        """Plot correlation matrix of key parameters."""
        
        # Select numerical columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = self.df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('Parameter Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_matrix
    
    def plot_parameter_vs_stress(self, save_path=None):
        """Plot key parameters vs residual stress."""
        
        key_params = [
            'plate_length', 'plate_width', 'anode_thickness', 'electrolyte_thickness',
            'anode_cte', 'electrolyte_cte', 'cathode_cte',
            'max_sintering_temp', 'heating_rate', 'cooling_rate'
        ]
        
        stress_cols = [col for col in self.df.columns if 'residual_stress_total' in col]
        
        for stress_col in stress_cols:
            layer = stress_col.split('_')[0]
            
            fig, axes = plt.subplots(2, 5, figsize=(25, 10))
            fig.suptitle(f'{layer.title()} Layer: Parameter vs Residual Stress', fontsize=16, fontweight='bold')
            
            for i, param in enumerate(key_params):
                if param not in self.df.columns:
                    continue
                    
                row, col = i // 5, i % 5
                ax = axes[row, col]
                
                # Scatter plot with trend line
                ax.scatter(self.df[param], self.df[stress_col], alpha=0.6, s=10)
                
                # Add trend line
                z = np.polyfit(self.df[param], self.df[stress_col], 1)
                p = np.poly1d(z)
                ax.plot(self.df[param], p(self.df[param]), "r--", alpha=0.8)
                
                # Calculate correlation
                corr, _ = pearsonr(self.df[param], self.df[stress_col])
                
                ax.set_xlabel(param.replace('_', ' ').title())
                ax.set_ylabel('Residual Stress (Pa)')
                ax.set_title(f'r = {corr:.3f}')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}_{layer}_parameter_vs_stress.png", dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_pca_analysis(self, save_path=None):
        """Perform and plot PCA analysis."""
        
        # Select input features (exclude stress outputs and sample_id)
        feature_cols = [col for col in self.df.columns 
                       if not any(x in col for x in ['stress', 'sample_id', 'von_mises'])]
        
        X = self.df[feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot explained variance
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Explained variance ratio
        axes[0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                    pca.explained_variance_ratio_, 'bo-')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Component')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, len(cumvar) + 1), cumvar, 'ro-')
        axes[1].axhline(y=0.95, color='k', linestyle='--', alpha=0.7, label='95% Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # First two principal components
        axes[2].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, s=10)
        axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[2].set_title('First Two Principal Components')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_pca_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return pca, X_scaled, feature_cols
    
    def create_interactive_dashboard(self, save_path=None):
        """Create interactive Plotly dashboard."""
        
        # Get stress columns
        stress_cols = [col for col in self.df.columns if 'residual_stress_total' in col]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Parameter Distributions', 'Stress vs Temperature', 
                          'CTE Mismatch Effects', 'Thickness Effects'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Parameter distributions
        for i, col in enumerate(['anode_cte', 'electrolyte_cte', 'cathode_cte']):
            if col in self.df.columns:
                fig.add_trace(
                    go.Histogram(x=self.df[col], name=col.replace('_', ' ').title(),
                               opacity=0.7, nbinsx=30),
                    row=1, col=1
                )
        
        # Stress vs temperature
        if 'max_sintering_temp' in self.df.columns and stress_cols:
            for stress_col in stress_cols:
                layer = stress_col.split('_')[0]
                fig.add_trace(
                    go.Scatter(x=self.df['max_sintering_temp'], y=self.df[stress_col],
                             mode='markers', name=f'{layer.title()} Stress',
                             opacity=0.6),
                    row=1, col=2
                )
        
        # CTE mismatch effects
        if all(col in self.df.columns for col in ['anode_cte', 'electrolyte_cte']) and stress_cols:
            cte_diff = self.df['anode_cte'] - self.df['electrolyte_cte']
            fig.add_trace(
                go.Scatter(x=cte_diff, y=self.df['anode_residual_stress_total'],
                         mode='markers', name='CTE Mismatch vs Stress',
                         opacity=0.6),
                row=2, col=1
            )
        
        # Thickness effects
        if all(col in self.df.columns for col in ['anode_thickness', 'electrolyte_thickness']) and stress_cols:
            thickness_ratio = self.df['anode_thickness'] / self.df['electrolyte_thickness']
            fig.add_trace(
                go.Scatter(x=thickness_ratio, y=self.df['anode_residual_stress_total'],
                         mode='markers', name='Thickness Ratio vs Stress',
                         opacity=0.6),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Residual Stress Dataset Interactive Dashboard",
            title_x=0.5,
            height=800,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="CTE (1/K)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        
        fig.update_xaxes(title_text="Max Sintering Temperature (K)", row=1, col=2)
        fig.update_yaxes(title_text="Residual Stress (Pa)", row=1, col=2)
        
        fig.update_xaxes(title_text="CTE Difference (1/K)", row=2, col=1)
        fig.update_yaxes(title_text="Residual Stress (Pa)", row=2, col=1)
        
        fig.update_xaxes(title_text="Thickness Ratio", row=2, col=2)
        fig.update_yaxes(title_text="Residual Stress (Pa)", row=2, col=2)
        
        if save_path:
            fig.write_html(f"{save_path}_interactive_dashboard.html")
        
        fig.show()
        
        return fig
    
    def generate_summary_report(self, save_path=None):
        """Generate comprehensive summary report."""
        
        report = {
            'dataset_info': {
                'n_samples': len(self.df),
                'n_features': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum(),
            },
            'parameter_statistics': {},
            'stress_statistics': {},
            'correlations': {}
        }
        
        # Parameter statistics
        feature_cols = [col for col in self.df.columns 
                       if not any(x in col for x in ['stress', 'sample_id', 'von_mises'])]
        
        for col in feature_cols:
            if self.df[col].dtype in ['float64', 'int64']:
                report['parameter_statistics'][col] = {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'median': float(self.df[col].median())
                }
        
        # Stress statistics
        stress_cols = [col for col in self.df.columns if 'stress' in col]
        for col in stress_cols:
            if col in self.df.columns:
                report['stress_statistics'][col] = {
                    'mean': float(self.df[col].mean()),
                    'std': float(self.df[col].std()),
                    'min': float(self.df[col].min()),
                    'max': float(self.df[col].max()),
                    'median': float(self.df[col].median())
                }
        
        # Key correlations
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        corr_matrix = self.df[numerical_cols].corr()
        
        # Find strongest correlations with stress
        for stress_col in stress_cols:
            if stress_col in corr_matrix.columns:
                correlations = corr_matrix[stress_col].abs().sort_values(ascending=False)
                # Exclude self-correlation and other stress variables
                correlations = correlations[~correlations.index.str.contains('stress|sample_id')]
                report['correlations'][stress_col] = correlations.head(5).to_dict()
        
        # Save report
        if save_path:
            import json
            with open(f"{save_path}_summary_report.json", 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def plot_all_visualizations(self, save_path='dataset_analysis'):
        """Generate all visualizations and save them."""
        
        print("Generating comprehensive dataset visualizations...")
        
        print("1. Parameter distributions...")
        self.plot_parameter_distributions(save_path)
        
        print("2. Stress distributions...")
        self.plot_stress_distributions(save_path)
        
        print("3. Correlation matrix...")
        self.plot_correlation_matrix(save_path)
        
        print("4. Parameter vs stress relationships...")
        self.plot_parameter_vs_stress(save_path)
        
        print("5. PCA analysis...")
        self.plot_pca_analysis(save_path)
        
        print("6. Interactive dashboard...")
        self.create_interactive_dashboard(save_path)
        
        print("7. Summary report...")
        self.generate_summary_report(save_path)
        
        print("All visualizations completed!")


def main():
    """Main function for testing the visualizer."""
    
    # This would typically load an existing dataset
    print("Dataset visualizer module ready!")
    print("Usage:")
    print("  visualizer = DatasetVisualizer('path/to/dataset.csv')")
    print("  visualizer.plot_all_visualizations()")


if __name__ == "__main__":
    main()