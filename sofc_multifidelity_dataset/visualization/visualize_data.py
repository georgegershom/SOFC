"""
Visualization Tools for SOFC Multi-Fidelity Dataset
Creates comprehensive plots and analysis of the generated data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import h5py
from typing import Dict, List, Optional, Tuple
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class SOFCDataVisualizer:
    """Visualization tools for SOFC multi-fidelity data"""
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.datasets = {}
        self.load_datasets()
        
    def load_datasets(self):
        """Load all available datasets"""
        
        # Low fidelity
        lf_path = os.path.join(self.data_dir, 'lf_dataset.csv')
        if os.path.exists(lf_path):
            self.datasets['low_fidelity'] = pd.read_csv(lf_path)
            print(f"Loaded LF dataset: {len(self.datasets['low_fidelity'])} samples")
        
        # Mid fidelity
        mf_path = os.path.join(self.data_dir, 'mf_dataset_scalars.csv')
        if os.path.exists(mf_path):
            self.datasets['mid_fidelity'] = pd.read_csv(mf_path)
            print(f"Loaded MF dataset: {len(self.datasets['mid_fidelity'])} samples")
        
        # High fidelity
        hf_path = os.path.join(self.data_dir, 'hf_dataset_scalars.csv')
        if os.path.exists(hf_path):
            self.datasets['high_fidelity'] = pd.read_csv(hf_path)
            print(f"Loaded HF dataset: {len(self.datasets['high_fidelity'])} samples")
        
        # Experimental
        exp_path = os.path.join(self.data_dir, 'experimental_dataset_summary.csv')
        if os.path.exists(exp_path):
            self.datasets['experimental'] = pd.read_csv(exp_path)
            print(f"Loaded Experimental dataset: {len(self.datasets['experimental'])} samples")
    
    def plot_fidelity_comparison(self, save_path: str = None):
        """Compare key metrics across different fidelity levels"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Multi-Fidelity Dataset Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['voltage', 'temperature_max', 'stress_max', 
                  'voltage_degradation_rate_mV_kh', 'estimated_lifetime_hours']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            
            data_to_plot = []
            labels = []
            
            for fidelity, df in self.datasets.items():
                if metric in df.columns:
                    data_to_plot.append(df[metric].dropna())
                    labels.append(fidelity.replace('_', ' ').title())
            
            if data_to_plot:
                bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
                
                # Color boxes
                colors = ['lightblue', 'lightgreen', 'coral', 'gold']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                
                ax.set_title(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Remove empty subplot
        fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved fidelity comparison plot to {save_path}")
        plt.show()
    
    def plot_correlation_matrix(self, fidelity: str = 'low_fidelity', save_path: str = None):
        """Plot correlation matrix for a specific fidelity level"""
        
        if fidelity not in self.datasets:
            print(f"Dataset {fidelity} not found")
            return
        
        df = self.datasets[fidelity]
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Create plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Correlation Matrix - {fidelity.replace("_", " ").title()}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved correlation matrix to {save_path}")
        plt.show()
    
    def plot_degradation_analysis(self, save_path: str = None):
        """Analyze degradation patterns across datasets"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Voltage Degradation vs Time',
                          'Degradation Mechanisms',
                          'Lifetime Estimation',
                          'Stress vs Degradation'),
            specs=[[{'type': 'scatter'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        colors = {'low_fidelity': 'blue', 'mid_fidelity': 'green', 
                 'high_fidelity': 'red', 'experimental': 'purple'}
        
        # Plot 1: Voltage degradation vs time
        for fidelity, df in self.datasets.items():
            if 'time' in df.columns and 'voltage' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['time'], y=df['voltage'],
                             mode='markers', name=fidelity,
                             marker=dict(color=colors[fidelity], size=5, opacity=0.6)),
                    row=1, col=1
                )
        
        # Plot 2: Degradation mechanisms
        if 'low_fidelity' in self.datasets:
            df = self.datasets['low_fidelity']
            deg_cols = [col for col in df.columns if 'degradation' in col and 'rate' not in col]
            if deg_cols:
                means = [df[col].mean() for col in deg_cols]
                fig.add_trace(
                    go.Bar(x=[col.replace('degradation_', '').replace('_', ' ') for col in deg_cols],
                          y=means, name='Average Degradation'),
                    row=1, col=2
                )
        
        # Plot 3: Lifetime estimation
        for fidelity, df in self.datasets.items():
            if 'estimated_lifetime_hours' in df.columns and 'current_density' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['current_density'], y=df['estimated_lifetime_hours'],
                             mode='markers', name=fidelity,
                             marker=dict(color=colors[fidelity], size=5, opacity=0.6)),
                    row=2, col=1
                )
        
        # Plot 4: Stress vs degradation
        for fidelity, df in self.datasets.items():
            if 'stress_max' in df.columns and 'voltage_degradation_rate_mV_kh' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df['stress_max'], y=df['voltage_degradation_rate_mV_kh'],
                             mode='markers', name=fidelity,
                             marker=dict(color=colors[fidelity], size=5, opacity=0.6)),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True,
                         title_text="SOFC Degradation Analysis",
                         title_font_size=16)
        fig.update_xaxes(title_text="Time (hours)", row=1, col=1)
        fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
        fig.update_xaxes(title_text="Current Density (A/m²)", row=2, col=1)
        fig.update_yaxes(title_text="Lifetime (hours)", row=2, col=1)
        fig.update_xaxes(title_text="Max Stress (Pa)", row=2, col=2)
        fig.update_yaxes(title_text="Degradation Rate (mV/kh)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Saved degradation analysis to {save_path}")
        fig.show()
    
    def visualize_2d_fields(self, dataset_path: str = 'data/mf_dataset.h5', 
                           sample_idx: int = 0, save_path: str = None):
        """Visualize 2D field data from mid-fidelity dataset"""
        
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found")
            return
        
        with h5py.File(dataset_path, 'r') as f:
            if 'field_data' not in f:
                print("No field data found in dataset")
                return
            
            fields = f['field_data']
            
            # Select fields to visualize
            field_names = ['temperature_2d', 'current_density_2d', 'stress_2d', 'damage_2d']
            available_fields = [name for name in field_names if name in fields]
            
            if not available_fields:
                print("No 2D fields found")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'2D Field Visualization - Sample {sample_idx}', 
                        fontsize=14, fontweight='bold')
            
            for idx, field_name in enumerate(available_fields[:4]):
                ax = axes[idx // 2, idx % 2]
                
                field_data = fields[field_name][sample_idx]
                
                im = ax.imshow(field_data, cmap='hot', aspect='auto')
                ax.set_title(field_name.replace('_', ' ').title())
                ax.set_xlabel('X Direction')
                ax.set_ylabel('Y Direction')
                plt.colorbar(im, ax=ax)
            
            # Remove empty subplots
            for idx in range(len(available_fields), 4):
                fig.delaxes(axes[idx // 2, idx % 2])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved 2D fields visualization to {save_path}")
            plt.show()
    
    def visualize_3d_fields(self, dataset_path: str = 'data/hf_dataset.h5',
                           sample_idx: int = 0, field_name: str = 'temperature',
                           save_path: str = None):
        """Visualize 3D field data from high-fidelity dataset"""
        
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found")
            return
        
        with h5py.File(dataset_path, 'r') as f:
            if 'field_data_3d' not in f:
                print("No 3D field data found")
                return
            
            sample_group = f[f'field_data_3d/sample_{sample_idx:04d}']
            
            if field_name not in sample_group:
                print(f"Field {field_name} not found")
                return
            
            field_3d = sample_group[field_name][:]
            
            # Create 3D visualization using plotly
            nx, ny, nz = field_3d.shape
            
            # Sample the field for visualization (too many points slow down rendering)
            step = max(1, nx // 20)
            x, y, z = np.mgrid[0:nx:step, 0:ny:step, 0:nz:step]
            values = field_3d[::step, ::step, ::step]
            
            fig = go.Figure(data=go.Volume(
                x=x.flatten(),
                y=y.flatten(),
                z=z.flatten(),
                value=values.flatten(),
                isomin=values.min(),
                isomax=values.max(),
                opacity=0.1,
                surface_count=15,
                colorscale='Hot'
            ))
            
            fig.update_layout(
                title=f'3D {field_name.title()} Field - Sample {sample_idx}',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z (Layers)'
                )
            )
            
            if save_path:
                fig.write_html(save_path)
                print(f"Saved 3D visualization to {save_path}")
            fig.show()
    
    def plot_experimental_measurements(self, dataset_path: str = 'data/experimental_dataset.h5',
                                      sample_idx: int = 0, save_path: str = None):
        """Visualize experimental measurements"""
        
        if not os.path.exists(dataset_path):
            print(f"Dataset {dataset_path} not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Experimental Measurements - Sample {sample_idx}', 
                    fontsize=14, fontweight='bold')
        
        with h5py.File(dataset_path, 'r') as f:
            exp_data = f['experimental_measurements']
            
            # IV Curve
            ax = axes[0, 0]
            iv_data = exp_data[f'iv_curves/sample_{sample_idx:03d}']
            ax.plot(iv_data['current_density'][:], iv_data['voltage'][:], 'b-', linewidth=2)
            ax.set_xlabel('Current Density (A/m²)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title('I-V Curve')
            ax.grid(True, alpha=0.3)
            
            # Power curve
            ax2 = ax.twinx()
            ax2.plot(iv_data['current_density'][:], iv_data['power_density'][:], 
                    'r--', linewidth=2, alpha=0.7)
            ax2.set_ylabel('Power Density (W/m²)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # EIS Nyquist plot
            ax = axes[0, 1]
            eis_data = exp_data[f'eis_spectra/sample_{sample_idx:03d}']
            ax.plot(eis_data['Z_real'][:], -eis_data['Z_imag'][:], 'go-', markersize=4)
            ax.set_xlabel('Z_real (Ω)')
            ax.set_ylabel('-Z_imag (Ω)')
            ax.set_title('EIS Nyquist Plot')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            # EIS Bode plot
            ax = axes[0, 2]
            ax.semilogx(eis_data['frequency'][:], eis_data['Z_magnitude'][:], 'b-')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('|Z| (Ω)')
            ax.set_title('EIS Bode Plot')
            ax.grid(True, alpha=0.3)
            
            ax2 = ax.twinx()
            ax2.semilogx(eis_data['frequency'][:], eis_data['Z_phase'][:], 'r--')
            ax2.set_ylabel('Phase (deg)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            
            # Thermography
            ax = axes[1, 0]
            thermo_data = exp_data[f'thermography/sample_{sample_idx:03d}']
            T_field = thermo_data['temperature_field'][:]
            im = ax.imshow(T_field, cmap='hot', aspect='auto')
            ax.set_title('IR Thermography')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.colorbar(im, ax=ax, label='Temperature (K)')
            
            # SEM Image
            ax = axes[1, 1]
            sem_data = exp_data[f'sem_images/sample_{sample_idx:03d}']
            sem_image = sem_data['image'][:]
            ax.imshow(sem_image, cmap='gray')
            ax.set_title('SEM Microstructure')
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.axis('off')
            
            # XRD Pattern
            ax = axes[1, 2]
            xrd_data = exp_data[f'xrd_patterns/sample_{sample_idx:03d}']
            ax.plot(xrd_data['two_theta'][:], xrd_data['intensity'][:], 'k-', linewidth=1)
            ax.set_xlabel('2θ (degrees)')
            ax.set_ylabel('Intensity (a.u.)')
            ax.set_title('XRD Pattern')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved experimental measurements to {save_path}")
        plt.show()
    
    def generate_summary_report(self, output_dir: str = 'reports'):
        """Generate comprehensive summary report with all visualizations"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating summary report...")
        
        # 1. Fidelity comparison
        self.plot_fidelity_comparison(
            save_path=os.path.join(output_dir, 'fidelity_comparison.png'))
        
        # 2. Correlation matrices
        for fidelity in self.datasets.keys():
            self.plot_correlation_matrix(
                fidelity, 
                save_path=os.path.join(output_dir, f'correlation_{fidelity}.png'))
        
        # 3. Degradation analysis
        self.plot_degradation_analysis(
            save_path=os.path.join(output_dir, 'degradation_analysis.html'))
        
        # 4. 2D fields (if available)
        mf_path = os.path.join(self.data_dir, 'mf_dataset.h5')
        if os.path.exists(mf_path):
            self.visualize_2d_fields(
                mf_path, sample_idx=0,
                save_path=os.path.join(output_dir, '2d_fields.png'))
        
        # 5. 3D fields (if available)
        hf_path = os.path.join(self.data_dir, 'hf_dataset.h5')
        if os.path.exists(hf_path):
            self.visualize_3d_fields(
                hf_path, sample_idx=0, field_name='temperature',
                save_path=os.path.join(output_dir, '3d_temperature.html'))
        
        # 6. Experimental measurements (if available)
        exp_path = os.path.join(self.data_dir, 'experimental_dataset.h5')
        if os.path.exists(exp_path):
            self.plot_experimental_measurements(
                exp_path, sample_idx=0,
                save_path=os.path.join(output_dir, 'experimental_measurements.png'))
        
        # Generate statistics summary
        stats_file = os.path.join(output_dir, 'dataset_statistics.txt')
        with open(stats_file, 'w') as f:
            f.write("SOFC Multi-Fidelity Dataset Statistics\n")
            f.write("=" * 50 + "\n\n")
            
            for fidelity, df in self.datasets.items():
                f.write(f"\n{fidelity.upper().replace('_', ' ')}:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Number of samples: {len(df)}\n")
                f.write(f"Number of features: {len(df.columns)}\n")
                f.write(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")
                f.write("\nFeature statistics:\n")
                f.write(df.describe().to_string())
                f.write("\n\n")
        
        print(f"Summary report generated in {output_dir}/")
        return output_dir


def visualize_all_datasets():
    """Main function to visualize all datasets"""
    
    visualizer = SOFCDataVisualizer()
    
    # Generate summary report
    report_dir = visualizer.generate_summary_report()
    
    print(f"\nVisualization complete! Reports saved in {report_dir}/")
    
    return visualizer


if __name__ == "__main__":
    visualizer = visualize_all_datasets()