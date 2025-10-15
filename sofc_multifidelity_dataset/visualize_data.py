#!/usr/bin/env python3
"""
Visualization tools for multi-fidelity SOFC dataset.

Usage:
    python visualize_data.py --data ./data --output ./results/figures
    python visualize_data.py --data ./data --fidelity low --samples 100
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
from typing import Dict, List, Optional, Tuple

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.data_utils import DatasetManager

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class DataVisualizer:
    """Visualize multi-fidelity SOFC dataset."""
    
    def __init__(self, data_path: str = "./data"):
        """Initialize visualizer."""
        self.data_path = data_path
        self.manager = DatasetManager(data_path)
        self.colors = {
            'low_fidelity': 'blue',
            'mid_fidelity': 'green', 
            'high_fidelity': 'red',
            'experimental': 'purple'
        }
    
    def plot_input_distributions(self, output_dir: str):
        """Plot distributions of input variables across fidelities."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        input_vars = [
            'temperature', 'current_density', 'fuel_utilization',
            'air_utilization', 'pressure', 'anode_thickness',
            'cathode_thickness', 'operating_time', 'thermal_cycles'
        ]
        
        for idx, var in enumerate(input_vars):
            ax = axes[idx]
            
            for fidelity in ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']:
                try:
                    data = self.manager.read_dataset(fidelity, indices=list(range(min(1000, 
                                                    self.manager.file_paths[fidelity].stat().st_size))))
                    if var in data['inputs']:
                        values = data['inputs'][var]
                        if values.ndim > 1:
                            values = values[:, 0]  # Take first component if multi-dimensional
                        
                        ax.hist(values, bins=30, alpha=0.5, label=fidelity.replace('_', ' ').title(),
                               color=self.colors[fidelity], density=True)
                except:
                    continue
            
            ax.set_xlabel(var.replace('_', ' ').title())
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Input Variable Distributions Across Fidelities', fontsize=14, y=1.02)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'input_distributions.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_iv_curves(self, output_dir: str, n_samples: int = 10):
        """Plot I-V curves from different fidelities."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        fidelities = ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']
        
        for idx, fidelity in enumerate(fidelities):
            ax = axes[idx // 2, idx % 2]
            
            try:
                data = self.manager.read_dataset(fidelity, indices=list(range(min(n_samples, 100))))
                
                if fidelity == 'experimental' and 'iv_curve' in data['outputs']:
                    # Experimental has explicit I-V curves
                    for i in range(min(n_samples, len(data['outputs']['iv_curve']))):
                        iv_data = data['outputs']['iv_curve'][i]
                        ax.plot(iv_data[:, 0] / 10000, iv_data[:, 1], 
                               alpha=0.5, color=self.colors[fidelity])
                else:
                    # Generate I-V curves from voltage and current data
                    if 'voltage' in data['outputs'] and 'current_density' in data['inputs']:
                        currents = data['inputs']['current_density'] / 10000  # Convert to A/cm²
                        voltages = data['outputs']['voltage']
                        
                        # Sort by current for proper curve
                        sort_idx = np.argsort(currents)
                        ax.scatter(currents[sort_idx], voltages[sort_idx], 
                                 alpha=0.3, color=self.colors[fidelity], s=10)
                        
                        # Add trend line
                        z = np.polyfit(currents[currents > 0], voltages[currents > 0], 2)
                        p = np.poly1d(z)
                        i_range = np.linspace(0, currents.max(), 100)
                        ax.plot(i_range, p(i_range), color=self.colors[fidelity], 
                               linewidth=2, label='Trend')
            except Exception as e:
                print(f"Could not plot I-V for {fidelity}: {e}")
                continue
            
            ax.set_xlabel('Current Density (A/cm²)')
            ax.set_ylabel('Voltage (V)')
            ax.set_title(fidelity.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1.0])
            ax.set_ylim([0.4, 1.2])
        
        plt.suptitle('I-V Characteristics Across Fidelities', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'iv_curves.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_temperature_fields(self, output_dir: str, sample_idx: int = 0):
        """Plot temperature fields for spatial fidelities."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        fidelities = ['mid_fidelity', 'high_fidelity', 'experimental']
        
        for idx, fidelity in enumerate(fidelities):
            ax = axes[idx]
            
            try:
                data = self.manager.read_dataset(fidelity, indices=[sample_idx])
                
                if fidelity == 'experimental' and 'temperature_map' in data['outputs']:
                    T_field = data['outputs']['temperature_map'][0]
                    im = ax.imshow(T_field, cmap='hot', aspect='auto')
                    plt.colorbar(im, ax=ax, label='Temperature (K)')
                    ax.set_title('Experimental (IR Thermography)')
                    
                elif 'temperature_field' in data['outputs']:
                    T_field = data['outputs']['temperature_field'][0]
                    
                    # Take a slice if 3D
                    if T_field.ndim == 3:
                        # Middle z-slice
                        T_slice = T_field[:, :, T_field.shape[2] // 2]
                    else:
                        T_slice = T_field
                    
                    im = ax.imshow(T_slice, cmap='hot', aspect='auto')
                    plt.colorbar(im, ax=ax, label='Temperature (K)')
                    ax.set_title(f'{fidelity.replace("_", " ").title()} (z=middle)')
                
                ax.set_xlabel('Y index')
                ax.set_ylabel('X index')
                
            except Exception as e:
                print(f"Could not plot temperature for {fidelity}: {e}")
                ax.text(0.5, 0.5, f'No data\n{str(e)[:30]}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(fidelity.replace('_', ' ').title())
        
        plt.suptitle(f'Temperature Fields (Sample {sample_idx})', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'temperature_fields_sample_{sample_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_stress_analysis(self, output_dir: str, sample_idx: int = 0):
        """Plot stress and damage fields."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        for fidelity_idx, fidelity in enumerate(['mid_fidelity', 'high_fidelity']):
            try:
                data = self.manager.read_dataset(fidelity, indices=[sample_idx])
                
                if 'stress_tensor' in data['outputs']:
                    stress = data['outputs']['stress_tensor'][0]
                    
                    # Von Mises stress
                    von_mises = np.sqrt(0.5 * ((stress[0] - stress[1])**2 + 
                                               (stress[1] - stress[2])**2 + 
                                               (stress[2] - stress[0])**2 + 
                                               6 * (stress[3]**2 + stress[4]**2 + stress[5]**2)))
                    
                    # Take middle slice if 3D
                    if von_mises.ndim == 3:
                        von_mises_slice = von_mises[:, :, von_mises.shape[2] // 2]
                    else:
                        von_mises_slice = von_mises
                    
                    im = axes[fidelity_idx, 0].imshow(von_mises_slice / 1e6, cmap='plasma')
                    plt.colorbar(im, ax=axes[fidelity_idx, 0], label='Von Mises Stress (MPa)')
                    axes[fidelity_idx, 0].set_title(f'{fidelity.replace("_", " ").title()}: Von Mises')
                
                if 'damage_field' in data['outputs']:
                    damage = data['outputs']['damage_field'][0]
                    
                    if damage.ndim == 3:
                        damage_slice = damage[:, :, damage.shape[2] // 2]
                    else:
                        damage_slice = damage
                    
                    im = axes[fidelity_idx, 1].imshow(damage_slice, cmap='Reds', vmin=0, vmax=1)
                    plt.colorbar(im, ax=axes[fidelity_idx, 1], label='Damage [-]')
                    axes[fidelity_idx, 1].set_title(f'{fidelity.replace("_", " ").title()}: Damage')
                
                if 'strain_tensor' in data['outputs']:
                    strain = data['outputs']['strain_tensor'][0]
                    
                    # Total strain magnitude
                    total_strain = np.sqrt(strain[0]**2 + strain[1]**2 + strain[2]**2)
                    
                    if total_strain.ndim == 3:
                        strain_slice = total_strain[:, :, total_strain.shape[2] // 2]
                    else:
                        strain_slice = total_strain
                    
                    im = axes[fidelity_idx, 2].imshow(strain_slice * 1000, cmap='viridis')
                    plt.colorbar(im, ax=axes[fidelity_idx, 2], label='Total Strain (‰)')
                    axes[fidelity_idx, 2].set_title(f'{fidelity.replace("_", " ").title()}: Strain')
                    
            except Exception as e:
                print(f"Could not plot stress analysis for {fidelity}: {e}")
                for j in range(3):
                    axes[fidelity_idx, j].text(0.5, 0.5, 'No data', 
                                              ha='center', va='center',
                                              transform=axes[fidelity_idx, j].transAxes)
        
        plt.suptitle(f'Stress and Damage Analysis (Sample {sample_idx})', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'stress_analysis_sample_{sample_idx}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_degradation_correlations(self, output_dir: str):
        """Plot degradation correlations across fidelities."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        for idx, fidelity in enumerate(['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']):
            ax = axes[idx // 2, idx % 2]
            
            try:
                data = self.manager.read_dataset(fidelity, indices=list(range(min(500, 1000))))
                
                if 'operating_time' in data['inputs']:
                    time = data['inputs']['operating_time']
                    
                    # Plot voltage vs time if available
                    if 'voltage' in data['outputs']:
                        voltage = data['outputs']['voltage']
                        
                        # Color by temperature
                        if 'temperature' in data['inputs']:
                            temp = data['inputs']['temperature']
                            sc = ax.scatter(time, voltage, c=temp, cmap='coolwarm',
                                          alpha=0.5, s=10)
                            plt.colorbar(sc, ax=ax, label='Temperature (K)')
                        else:
                            ax.scatter(time, voltage, alpha=0.5, s=10,
                                     color=self.colors[fidelity])
                        
                        ax.set_xlabel('Operating Time (hours)')
                        ax.set_ylabel('Voltage (V)')
                    
                    # Alternative: degradation rate
                    elif 'degradation_rate' in data['outputs']:
                        deg_rate = data['outputs']['degradation_rate']
                        ax.scatter(time, deg_rate, alpha=0.5, s=10,
                                 color=self.colors[fidelity])
                        ax.set_xlabel('Operating Time (hours)')
                        ax.set_ylabel('Degradation Rate (%/1000h)')
                
                ax.set_title(fidelity.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Could not plot degradation for {fidelity}: {e}")
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes)
        
        plt.suptitle('Degradation Characteristics Across Fidelities', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'degradation_correlations.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def plot_fidelity_comparison(self, output_dir: str):
        """Compare key metrics across fidelity levels."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = {
            'voltage': [],
            'power_density': [],
            'degradation_rate': []
        }
        
        fidelity_labels = []
        
        for fidelity in ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']:
            try:
                data = self.manager.read_dataset(fidelity, indices=list(range(min(100, 1000))))
                
                if 'voltage' in data['outputs']:
                    metrics['voltage'].append(data['outputs']['voltage'])
                else:
                    metrics['voltage'].append(np.array([np.nan]))
                
                if 'power_density' in data['outputs']:
                    metrics['power_density'].append(data['outputs']['power_density'])
                else:
                    metrics['power_density'].append(np.array([np.nan]))
                
                if 'degradation_rate' in data['outputs']:
                    metrics['degradation_rate'].append(data['outputs']['degradation_rate'])
                else:
                    metrics['degradation_rate'].append(np.array([np.nan]))
                
                fidelity_labels.append(fidelity.replace('_', '\n'))
                
            except:
                for key in metrics:
                    metrics[key].append(np.array([np.nan]))
                fidelity_labels.append(fidelity.replace('_', '\n'))
        
        # Box plots
        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[idx]
            
            # Filter out NaN arrays
            valid_data = []
            valid_labels = []
            for d, l in zip(metric_data, fidelity_labels):
                if not np.all(np.isnan(d)):
                    valid_data.append(d[~np.isnan(d)])
                    valid_labels.append(l)
            
            if valid_data:
                bp = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True)
                
                # Color boxes
                colors_list = ['blue', 'green', 'red', 'purple']
                for patch, color in zip(bp['boxes'], colors_list[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.5)
                
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(f'{metric_name.replace("_", " ").title()} Distribution')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Fidelity Comparison of Key Metrics', fontsize=14)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, 'fidelity_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    def generate_summary_report(self, output_dir: str):
        """Generate a summary report of the dataset."""
        report = []
        report.append("="*60)
        report.append("MULTI-FIDELITY SOFC DATASET SUMMARY REPORT")
        report.append("="*60)
        
        for fidelity in ['low_fidelity', 'mid_fidelity', 'high_fidelity', 'experimental']:
            report.append(f"\n{fidelity.upper().replace('_', ' ')}:")
            report.append("-"*40)
            
            try:
                validation = self.manager.validate_dataset(fidelity)
                
                report.append(f"  Samples: {validation['n_samples']:,}")
                report.append(f"  File size: {validation['file_size_mb']:.1f} MB")
                
                # Input statistics
                if 'inputs' in validation['datasets']:
                    report.append("\n  Input Variables:")
                    for var, stats in validation['datasets']['inputs'].items():
                        if 'mean' in stats:
                            report.append(f"    {var:25s}: mean={stats['mean']:.3f}, "
                                        f"std={stats.get('std', 0):.3f}, "
                                        f"range=[{stats.get('min', 0):.3f}, {stats.get('max', 0):.3f}]")
                
                # Output statistics
                if 'outputs' in validation['datasets']:
                    report.append("\n  Output Variables:")
                    for var, stats in validation['datasets']['outputs'].items():
                        shape_str = str(stats['shape'])
                        size_str = f"{stats['size_mb']:.1f} MB"
                        report.append(f"    {var:25s}: shape={shape_str:20s}, size={size_str}")
                
            except Exception as e:
                report.append(f"  Error reading dataset: {e}")
        
        # Save report
        report_path = os.path.join(output_dir, 'dataset_summary.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"Saved summary report: {report_path}")
        
        # Also print to console
        print('\n'.join(report))


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize multi-fidelity SOFC dataset')
    
    parser.add_argument('--data', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--output', type=str, default='./results/figures',
                       help='Output directory for figures')
    parser.add_argument('--plots', type=str, nargs='+',
                       choices=['inputs', 'iv', 'temperature', 'stress', 'degradation', 'comparison', 'all'],
                       default=['all'], help='Which plots to generate')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to visualize for detailed plots')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize visualizer
    viz = DataVisualizer(args.data)
    
    # Generate plots
    if 'all' in args.plots:
        plots_to_make = ['inputs', 'iv', 'temperature', 'stress', 'degradation', 'comparison']
    else:
        plots_to_make = args.plots
    
    print(f"\nGenerating visualizations...")
    print(f"  Data path: {args.data}")
    print(f"  Output path: {args.output}")
    print(f"  Plots: {', '.join(plots_to_make)}")
    
    if 'inputs' in plots_to_make:
        viz.plot_input_distributions(args.output)
    
    if 'iv' in plots_to_make:
        viz.plot_iv_curves(args.output, n_samples=args.samples)
    
    if 'temperature' in plots_to_make:
        for i in range(min(args.samples, 3)):
            viz.plot_temperature_fields(args.output, sample_idx=i)
    
    if 'stress' in plots_to_make:
        for i in range(min(args.samples, 3)):
            viz.plot_stress_analysis(args.output, sample_idx=i)
    
    if 'degradation' in plots_to_make:
        viz.plot_degradation_correlations(args.output)
    
    if 'comparison' in plots_to_make:
        viz.plot_fidelity_comparison(args.output)
    
    # Always generate summary report
    viz.generate_summary_report(args.output)
    
    print(f"\n✓ Visualizations complete! Saved to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()