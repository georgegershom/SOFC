#!/usr/bin/env python3
"""
SOFC Multi-Fidelity Dataset Analysis and Documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import json
from pathlib import Path

class SOFCDatasetAnalyzer:
    """Comprehensive analysis of the SOFC multi-fidelity dataset."""
    
    def __init__(self, dataset_dir='sofc_dataset'):
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self.metadata = {}
        self.load_datasets()
        
    def load_datasets(self):
        """Load all datasets and metadata."""
        # Load metadata
        with open(self.dataset_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load CSV datasets
        for fidelity in ['LF', 'MF', 'HF']:
            csv_path = self.dataset_dir / f'sofc_dataset_{fidelity.lower()}.csv'
            self.datasets[fidelity] = pd.read_csv(csv_path)
        
        print(f"Loaded datasets: {list(self.datasets.keys())}")
        print(f"Total samples: {self.metadata['total_samples']}")
        
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        report = []
        report.append("# Multi-Fidelity SOFC Dataset Summary Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset overview
        report.append("## Dataset Overview")
        report.append(f"- **Total Samples**: {self.metadata['total_samples']:,}")
        report.append(f"- **Fidelity Levels**: {', '.join(self.metadata['fidelity_levels'])}")
        report.append(f"- **Total Parameters**: {len(self.datasets['MF'].columns)}")
        report.append(f"- **Sampling Method**: {self.metadata['generation_info']['sampling_method']}")
        report.append(f"- **Generation Date**: {self.metadata['generation_info']['generation_date']}")
        report.append("")
        
        # Parameter categories
        report.append("## Parameter Categories")
        for category, params in self.metadata['parameter_categories'].items():
            report.append(f"### {category.replace('_', ' ').title()}")
            report.append(f"- **Count**: {len(params)} parameters")
            report.append(f"- **Parameters**: {', '.join(params)}")
            report.append("")
        
        # Statistical summary
        report.append("## Statistical Summary by Fidelity Level")
        for fidelity, df in self.datasets.items():
            report.append(f"### {fidelity} Fidelity")
            report.append(f"- **Samples**: {len(df):,}")
            report.append(f"- **Parameters**: {len(df.columns)}")
            
            # Key parameter statistics
            key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization']
            for param in key_params:
                if param in df.columns:
                    stats = df[param].describe()
                    report.append(f"- **{param}**: {stats['mean']:.3f} ± {stats['std']:.3f} (range: {stats['min']:.3f} - {stats['max']:.3f})")
            report.append("")
        
        # Data quality assessment
        report.append("## Data Quality Assessment")
        for fidelity, df in self.datasets.items():
            report.append(f"### {fidelity} Fidelity Quality Metrics")
            report.append(f"- **Missing Values**: {df.isnull().sum().sum()}")
            report.append(f"- **Duplicate Rows**: {df.duplicated().sum()}")
            report.append(f"- **Memory Usage**: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            report.append("")
        
        return "\n".join(report)
    
    def create_parameter_space_analysis(self):
        """Create comprehensive parameter space analysis."""
        print("Creating parameter space analysis...")
        
        # Create analysis directory
        analysis_dir = self.dataset_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)
        
        # 1. Parameter ranges comparison
        self._plot_parameter_ranges_comparison(analysis_dir)
        
        # 2. Multi-dimensional scaling
        self._plot_multidimensional_scaling(analysis_dir)
        
        # 3. Parameter importance analysis
        self._plot_parameter_importance(analysis_dir)
        
        # 4. Fidelity level differences
        self._plot_fidelity_differences(analysis_dir)
        
        print(f"Parameter space analysis saved to {analysis_dir}/")
    
    def _plot_parameter_ranges_comparison(self, output_dir):
        """Plot parameter ranges across fidelity levels."""
        # Select key parameters for comparison
        key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization',
                     'anode_porosity', 'cathode_porosity', 'electrolyte_thickness']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(key_params):
            if i < len(axes):
                ax = axes[i]
                
                # Collect data from all fidelity levels
                data_to_plot = []
                labels = []
                
                for fidelity, df in self.datasets.items():
                    if param in df.columns:
                        data_to_plot.append(df[param])
                        labels.append(fidelity)
                
                if data_to_plot:
                    ax.boxplot(data_to_plot, labels=labels)
                    ax.set_title(f'{param} Distribution by Fidelity')
                    ax.set_ylabel(param)
                    ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(key_params), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_ranges_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_multidimensional_scaling(self, output_dir):
        """Plot multi-dimensional scaling of parameter space."""
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # Combine all data
        all_data = []
        fidelity_labels = []
        
        for fidelity, df in self.datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                all_data.append(df[numeric_cols].values)
                fidelity_labels.extend([fidelity] * len(df))
        
        if all_data:
            X = np.vstack(all_data)
            
            # PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            X_tsne = tsne.fit_transform(X)
            
            # Plot both
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # PCA plot
            colors = ['red', 'green', 'blue']
            for i, fidelity in enumerate(['LF', 'MF', 'HF']):
                mask = np.array(fidelity_labels) == fidelity
                if np.any(mask):
                    ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                              c=colors[i], label=fidelity, alpha=0.6, s=20)
            
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax1.set_title('PCA of Parameter Space')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # t-SNE plot
            for i, fidelity in enumerate(['LF', 'MF', 'HF']):
                mask = np.array(fidelity_labels) == fidelity
                if np.any(mask):
                    ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                              c=colors[i], label=fidelity, alpha=0.6, s=20)
            
            ax2.set_xlabel('t-SNE 1')
            ax2.set_ylabel('t-SNE 2')
            ax2.set_title('t-SNE of Parameter Space')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'multidimensional_scaling.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_parameter_importance(self, output_dir):
        """Plot parameter importance based on variance."""
        # Calculate variance for each parameter across all fidelity levels
        all_params = set()
        for df in self.datasets.values():
            all_params.update(df.columns)
        
        param_variance = {}
        for param in all_params:
            if param != 'fidelity_level':
                variances = []
                for df in self.datasets.values():
                    if param in df.columns and df[param].dtype in [np.float64, np.int64]:
                        variances.append(df[param].var())
                if variances:
                    param_variance[param] = np.mean(variances)
        
        # Sort by variance
        sorted_params = sorted(param_variance.items(), key=lambda x: x[1], reverse=True)
        
        # Plot top 20 parameters
        top_params = sorted_params[:20]
        params, variances = zip(*top_params)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(params)), variances)
        plt.yticks(range(len(params)), params)
        plt.xlabel('Variance')
        plt.title('Parameter Importance (Top 20 by Variance)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_fidelity_differences(self, output_dir):
        """Plot differences between fidelity levels."""
        # Select key parameters
        key_params = ['temperature', 'current_density', 'voltage', 'fuel_utilization']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, param in enumerate(key_params):
            ax = axes[i]
            
            # Create violin plots
            data_to_plot = []
            labels = []
            
            for fidelity, df in self.datasets.items():
                if param in df.columns:
                    data_to_plot.append(df[param])
                    labels.append(fidelity)
            
            if data_to_plot:
                parts = ax.violinplot(data_to_plot, positions=range(len(labels)))
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels)
                ax.set_title(f'{param} Distribution by Fidelity Level')
                ax.set_ylabel(param)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fidelity_differences.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_usage_examples(self):
        """Create usage examples for the dataset."""
        examples = []
        examples.append("# SOFC Multi-Fidelity Dataset Usage Examples")
        examples.append("=" * 50)
        examples.append("")
        
        # Python usage example
        examples.append("## Python Usage Example")
        examples.append("```python")
        examples.append("import pandas as pd")
        examples.append("import numpy as np")
        examples.append("")
        examples.append("# Load datasets")
        examples.append("lf_data = pd.read_csv('sofc_dataset_lf.csv')")
        examples.append("mf_data = pd.read_csv('sofc_dataset_mf.csv')")
        examples.append("hf_data = pd.read_csv('sofc_dataset_hf.csv')")
        examples.append("")
        examples.append("# Combine all fidelity levels")
        examples.append("all_data = pd.concat([lf_data, mf_data, hf_data], ignore_index=True)")
        examples.append("")
        examples.append("# Filter by operating conditions")
        examples.append("high_temp_data = all_data[all_data['temperature'] > 1050]")
        examples.append("")
        examples.append("# Calculate derived metrics")
        examples.append("all_data['power_density'] = all_data['current_density'] * all_data['voltage']")
        examples.append("all_data['efficiency'] = all_data['voltage'] / 1.25")
        examples.append("```")
        examples.append("")
        
        # Machine learning example
        examples.append("## Machine Learning Example")
        examples.append("```python")
        examples.append("from sklearn.ensemble import RandomForestRegressor")
        examples.append("from sklearn.model_selection import train_test_split")
        examples.append("")
        examples.append("# Prepare features and target")
        examples.append("feature_cols = [col for col in all_data.columns if col not in ['fidelity_level']]")
        examples.append("X = all_data[feature_cols]")
        examples.append("y = all_data['efficiency']  # or any other target variable")
        examples.append("")
        examples.append("# Split data")
        examples.append("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)")
        examples.append("")
        examples.append("# Train model")
        examples.append("model = RandomForestRegressor(n_estimators=100, random_state=42)")
        examples.append("model.fit(X_train, y_train)")
        examples.append("")
        examples.append("# Evaluate")
        examples.append("score = model.score(X_test, y_test)")
        examples.append("print(f'R² Score: {score:.3f}')")
        examples.append("```")
        examples.append("")
        
        # HDF5 usage example
        examples.append("## HDF5 Usage Example")
        examples.append("```python")
        examples.append("import h5py")
        examples.append("")
        examples.append("# Load from HDF5")
        examples.append("with h5py.File('sofc_dataset.h5', 'r') as f:")
        examples.append("    # Access different fidelity levels")
        examples.append("    lf_group = f['LF']")
        examples.append("    mf_group = f['MF']")
        examples.append("    hf_group = f['HF']")
        examples.append("    ")
        examples.append("    # Access specific parameters")
        examples.append("    temperature = lf_group['temperature'][:]")
        examples.append("    current_density = lf_group['current_density'][:]")
        examples.append("```")
        examples.append("")
        
        return "\n".join(examples)

def main():
    """Main analysis function."""
    print("=== SOFC Multi-Fidelity Dataset Analysis ===")
    
    # Initialize analyzer
    analyzer = SOFCDatasetAnalyzer()
    
    # Generate summary report
    print("Generating summary report...")
    summary_report = analyzer.generate_summary_report()
    
    with open('sofc_dataset/summary_report.md', 'w') as f:
        f.write(summary_report)
    print("Summary report saved to sofc_dataset/summary_report.md")
    
    # Create parameter space analysis
    analyzer.create_parameter_space_analysis()
    
    # Create usage examples
    print("Creating usage examples...")
    usage_examples = analyzer.create_usage_examples()
    
    with open('sofc_dataset/usage_examples.md', 'w') as f:
        f.write(usage_examples)
    print("Usage examples saved to sofc_dataset/usage_examples.md")
    
    print("\n=== Analysis Complete ===")
    print("Generated files:")
    print("- sofc_dataset/summary_report.md")
    print("- sofc_dataset/usage_examples.md")
    print("- sofc_dataset/analysis/ (parameter space analysis plots)")

if __name__ == "__main__":
    main()