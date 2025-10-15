"""
Visualization tools for SOFC multi-fidelity dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import argparse
import os
from typing import Dict, List, Optional, Tuple

class SOFCDataVisualizer:
    """Visualize and analyze SOFC dataset"""
    
    def __init__(self, dataset_path: str, output_dir: str = 'visualizations'):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        self.data = self._load_dataset()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def _load_dataset(self) -> Dict:
        """Load HDF5 dataset"""
        data = {}
        with h5py.File(self.dataset_path, 'r') as f:
            # Load time vector
            data['time'] = f['time_hours'][:]
            
            # Load data for each fidelity
            for fidelity in ['LF', 'MF', 'HF']:
                if fidelity in f:
                    data[fidelity] = {}
                    
                    # Load inputs
                    inputs = {}
                    for key in f[fidelity]['inputs'].keys():
                        inputs[key] = f[fidelity]['inputs'][key][:]
                    data[fidelity]['inputs'] = pd.DataFrame(inputs)
                    
                    # Load first few samples of responses
                    data[fidelity]['responses'] = []
                    response_grp = f[fidelity]['responses']
                    
                    # Load up to 10 samples for visualization
                    for i in range(min(10, len(response_grp))):
                        sample_key = f'sample_{i:05d}'
                        if sample_key in response_grp:
                            sample_data = {}
                            
                            # Load degradation data
                            if 'degradation' in response_grp[sample_key]:
                                sample_data['degradation'] = {}
                                for key in response_grp[sample_key]['degradation'].keys():
                                    sample_data['degradation'][key] = \
                                        response_grp[sample_key]['degradation'][key][:]
                            
                            data[fidelity]['responses'].append(sample_data)
        
        return data
    
    def plot_parameter_distributions(self):
        """Plot parameter distributions for each fidelity level"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
            if fidelity in self.data:
                df = self.data[fidelity]['inputs']
                
                # Select subset of parameters to plot
                param_cols = [col for col in df.columns 
                             if col not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']]
                
                # Select up to 6 important parameters
                important_params = [
                    col for col in param_cols 
                    if any(key in col for key in ['temperature', 'current_density', 'porosity', 'thickness'])
                ][:6]
                
                if important_params:
                    df[important_params].hist(ax=axes[idx], bins=30)
                    axes[idx].set_title(f'{fidelity} Parameter Distributions', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_distributions.png'), dpi=150)
        plt.show()
    
    def plot_degradation_curves(self):
        """Plot voltage degradation curves for different fidelity levels"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Low Fidelity', 'Medium Fidelity', 'High Fidelity'),
            shared_xaxes=True
        )
        
        colors = px.colors.qualitative.Plotly
        
        for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
            if fidelity in self.data and self.data[fidelity]['responses']:
                for i, response in enumerate(self.data[fidelity]['responses'][:5]):
                    if 'degradation' in response and 'voltage' in response['degradation']:
                        fig.add_trace(
                            go.Scatter(
                                x=self.data['time'],
                                y=response['degradation']['voltage'],
                                mode='lines',
                                name=f'Sample {i}',
                                line=dict(color=colors[i % len(colors)]),
                                showlegend=(idx == 0)
                            ),
                            row=idx+1, col=1
                        )
        
        fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
        fig.update_yaxes(title_text="Voltage (V)")
        
        fig.update_layout(
            height=900,
            title_text="Voltage Degradation Across Fidelity Levels",
            hovermode='x unified'
        )
        
        fig.write_html(os.path.join(self.output_dir, 'degradation_curves.html'))
        fig.show()
    
    def plot_correlation_matrix(self):
        """Plot correlation matrix of parameters for each fidelity"""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
            if fidelity in self.data:
                df = self.data[fidelity]['inputs']
                
                # Select numeric columns
                numeric_cols = [col for col in df.columns 
                               if col not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']]
                
                # Select subset for visualization
                cols_subset = numeric_cols[:15]  # Top 15 parameters
                
                if cols_subset:
                    corr_matrix = df[cols_subset].corr()
                    
                    # Shorten column names for display
                    short_names = [col.split('.')[-1][:15] for col in cols_subset]
                    
                    sns.heatmap(corr_matrix, 
                               ax=axes[idx],
                               cmap='coolwarm',
                               center=0,
                               vmin=-1, vmax=1,
                               xticklabels=short_names,
                               yticklabels=short_names,
                               cbar_kws={'shrink': 0.8})
                    
                    axes[idx].set_title(f'{fidelity} Parameter Correlations', fontsize=12)
                    axes[idx].tick_params(axis='both', labelsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'correlation_matrices.png'), dpi=150)
        plt.show()
    
    def plot_performance_comparison(self):
        """Compare performance metrics across fidelity levels"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Power Density', 'Area Specific Resistance', 
                          'Efficiency', 'Degradation Rate'),
            specs=[[{'type': 'box'}, {'type': 'box'}],
                  [{'type': 'box'}, {'type': 'box'}]]
        )
        
        metrics_data = {
            'power_density': [],
            'ASR': [],
            'efficiency': [],
            'degradation_rate': []
        }
        
        fidelity_labels = []
        
        for fidelity in ['LF', 'MF', 'HF']:
            if fidelity in self.data and self.data[fidelity]['responses']:
                for response in self.data[fidelity]['responses']:
                    if 'degradation' in response:
                        # Get final values
                        if 'power_density' in response['degradation']:
                            metrics_data['power_density'].append(
                                response['degradation']['power_density'][-1]
                            )
                        if 'ASR' in response['degradation']:
                            metrics_data['ASR'].append(
                                response['degradation']['ASR'][-1]
                            )
                        if 'efficiency' in response['degradation']:
                            metrics_data['efficiency'].append(
                                response['degradation']['efficiency'][-1]
                            )
                        if 'degradation_rate' in response['degradation']:
                            metrics_data['degradation_rate'].append(
                                np.mean(response['degradation']['degradation_rate'])
                            )
                        
                        fidelity_labels.append(fidelity)
        
        # Create box plots
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        metric_names = ['power_density', 'ASR', 'efficiency', 'degradation_rate']
        y_titles = ['Power Density (W/cm²)', 'ASR (Ω·cm²)', 'Efficiency', 'Degradation Rate (mV/1000h)']
        
        for pos, metric, y_title in zip(positions, metric_names, y_titles):
            if metrics_data[metric]:
                for fidelity in ['LF', 'MF', 'HF']:
                    mask = [f == fidelity for f in fidelity_labels]
                    values = [v for v, m in zip(metrics_data[metric], mask) if m]
                    
                    if values:
                        fig.add_trace(
                            go.Box(
                                y=values,
                                name=fidelity,
                                boxmean=True
                            ),
                            row=pos[0], col=pos[1]
                        )
                
                fig.update_yaxes(title_text=y_title, row=pos[0], col=pos[1])
        
        fig.update_layout(
            height=800,
            title_text="Performance Metrics Comparison Across Fidelity Levels",
            showlegend=False
        )
        
        fig.write_html(os.path.join(self.output_dir, 'performance_comparison.html'))
        fig.show()
    
    def plot_parameter_sensitivity(self):
        """Plot parameter sensitivity analysis"""
        # This would typically require running sensitivity analysis
        # Here we'll show parameter importance based on variance
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, fidelity in enumerate(['LF', 'MF', 'HF']):
            if fidelity in self.data:
                df = self.data[fidelity]['inputs']
                
                # Calculate coefficient of variation for each parameter
                numeric_cols = [col for col in df.columns 
                               if col not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']]
                
                cv_values = []
                param_names = []
                
                for col in numeric_cols[:20]:  # Top 20 parameters
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if mean_val != 0:
                        cv = std_val / abs(mean_val)
                        cv_values.append(cv)
                        param_names.append(col.split('.')[-1][:20])
                
                if cv_values:
                    # Sort by CV
                    sorted_indices = np.argsort(cv_values)[::-1][:15]
                    
                    axes[idx].barh(range(len(sorted_indices)), 
                                  [cv_values[i] for i in sorted_indices])
                    axes[idx].set_yticks(range(len(sorted_indices)))
                    axes[idx].set_yticklabels([param_names[i] for i in sorted_indices])
                    axes[idx].set_xlabel('Coefficient of Variation')
                    axes[idx].set_title(f'{fidelity} Parameter Variability')
                    axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_sensitivity.png'), dpi=150)
        plt.show()
    
    def create_summary_report(self):
        """Create a summary report of the dataset"""
        report = []
        report.append("=" * 80)
        report.append("SOFC MULTI-FIDELITY DATASET SUMMARY REPORT")
        report.append("=" * 80)
        report.append("")
        
        for fidelity in ['LF', 'MF', 'HF']:
            if fidelity in self.data:
                report.append(f"\n{fidelity} FIDELITY LEVEL")
                report.append("-" * 40)
                
                df = self.data[fidelity]['inputs']
                report.append(f"Number of samples: {len(df)}")
                
                numeric_cols = [col for col in df.columns 
                               if col not in ['sample_id', 'fidelity', 'parent_fidelity', 'parent_sample_id']]
                report.append(f"Number of parameters: {len(numeric_cols)}")
                
                # Parameter statistics
                report.append("\nKey Parameter Ranges:")
                important_params = ['system.temperature', 'system.current_density', 
                                  'system.fuel_utilization']
                
                for param in important_params:
                    if param in df.columns:
                        report.append(f"  {param}:")
                        report.append(f"    Min: {df[param].min():.3f}")
                        report.append(f"    Max: {df[param].max():.3f}")
                        report.append(f"    Mean: {df[param].mean():.3f}")
                        report.append(f"    Std: {df[param].std():.3f}")
                
                # Response statistics (from available samples)
                if self.data[fidelity]['responses']:
                    report.append("\nResponse Statistics (from sample subset):")
                    
                    voltages = []
                    for response in self.data[fidelity]['responses']:
                        if 'degradation' in response and 'voltage' in response['degradation']:
                            voltages.append(response['degradation']['voltage'][-1])
                    
                    if voltages:
                        report.append(f"  Final Voltage Range: {min(voltages):.3f} - {max(voltages):.3f} V")
                        report.append(f"  Mean Final Voltage: {np.mean(voltages):.3f} V")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        # Save report
        report_path = os.path.join(self.output_dir, 'dataset_summary.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print('\n'.join(report))
        print(f"\nReport saved to: {report_path}")
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        print("1. Parameter distributions...")
        self.plot_parameter_distributions()
        
        print("2. Degradation curves...")
        self.plot_degradation_curves()
        
        print("3. Correlation matrices...")
        self.plot_correlation_matrix()
        
        print("4. Performance comparison...")
        self.plot_performance_comparison()
        
        print("5. Parameter sensitivity...")
        self.plot_parameter_sensitivity()
        
        print("6. Summary report...")
        self.create_summary_report()
        
        print(f"\nAll visualizations saved to: {self.output_dir}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Visualize SOFC dataset')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--output', type=str, default='visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = SOFCDataVisualizer(args.dataset, args.output)
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()

if __name__ == '__main__':
    main()