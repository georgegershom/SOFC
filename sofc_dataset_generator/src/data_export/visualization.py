"""
Dataset Visualization Module

Provides visualization capabilities for the SOFC synthetic dataset,
including warp fields, stress fields, and parameter analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import pearsonr
from pathlib import Path


class DatasetVisualizer:
    """Visualizes SOFC dataset components"""
    
    def __init__(self, dataset: Dict):
        self.dataset = dataset
        self.n_samples = len(dataset['warp_data'])
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_dataset_overview(self, save_path: Optional[str] = None):
        """Create dataset overview visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract data for overview
        max_warps = []
        rms_warps = []
        max_von_mises = []
        max_principal = []
        cell_lengths = []
        electrolyte_thicknesses = []
        
        for i in range(self.n_samples):
            # Warp data
            warp_metrics = self.dataset['warp_data'][i]['top_warp_metrics']
            max_warps.append(warp_metrics.get('max_warp', 0))
            rms_warps.append(warp_metrics.get('rms_warp', 0))
            
            # Stress data
            stress_metrics = self.dataset['stress_data'][i]['stress_metrics']['electrolyte']
            max_von_mises.append(stress_metrics.get('max_von_mises', 0))
            max_principal.append(stress_metrics.get('max_principal_1', 0))
            
            # Parameters
            params = self.dataset['parameters'][i]
            cell_lengths.append(params.get('cell_length', 0))
            electrolyte_thicknesses.append(params.get('electrolyte_thickness', 0))
        
        # Plot 1: Warp distribution
        axes[0, 0].hist(max_warps, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Maximum Warp Distribution')
        axes[0, 0].set_xlabel('Max Warp (mm)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Plot 2: Stress distribution
        axes[0, 1].hist(max_von_mises, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Maximum Von Mises Stress Distribution')
        axes[0, 1].set_xlabel('Max Von Mises Stress (MPa)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Plot 3: Parameter distribution
        axes[0, 2].scatter(cell_lengths, electrolyte_thicknesses, alpha=0.7)
        axes[0, 2].set_title('Parameter Space')
        axes[0, 2].set_xlabel('Cell Length (mm)')
        axes[0, 2].set_ylabel('Electrolyte Thickness (μm)')
        
        # Plot 4: Warp vs Stress correlation
        axes[1, 0].scatter(max_warps, max_von_mises, alpha=0.7)
        axes[1, 0].set_title('Warp vs Stress Correlation')
        axes[1, 0].set_xlabel('Max Warp (mm)')
        axes[1, 0].set_ylabel('Max Von Mises Stress (MPa)')
        
        # Add correlation coefficient
        corr, _ = pearsonr(max_warps, max_von_mises)
        axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 5: RMS vs Max warp
        axes[1, 1].scatter(rms_warps, max_warps, alpha=0.7)
        axes[1, 1].set_title('RMS vs Max Warp')
        axes[1, 1].set_xlabel('RMS Warp (mm)')
        axes[1, 1].set_ylabel('Max Warp (mm)')
        
        # Plot 6: Principal stress distribution
        axes[1, 2].hist(max_principal, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Maximum Principal Stress Distribution')
        axes[1, 2].set_xlabel('Max Principal Stress (MPa)')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dataset overview saved to {save_path}")
        
        plt.show()
    
    def plot_warp_field(self, sample_idx: int, save_path: Optional[str] = None):
        """Plot warp field for specific sample"""
        if sample_idx >= self.n_samples:
            print(f"Sample index {sample_idx} out of range")
            return
        
        sample = self.dataset['warp_data'][sample_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Top surface height map
        height_map = sample['top_height_map']
        x_grid = sample['top_height_x']
        y_grid = sample['top_height_y']
        
        im1 = axes[0, 0].contourf(x_grid, y_grid, height_map, levels=20, cmap='viridis')
        axes[0, 0].set_title(f'Sample {sample_idx}: Top Surface Height Map')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Height (mm)')
        
        # Bottom surface height map
        bottom_height_map = sample['bottom_height_map']
        im2 = axes[0, 1].contourf(x_grid, y_grid, bottom_height_map, levels=20, cmap='viridis')
        axes[0, 1].set_title(f'Sample {sample_idx}: Bottom Surface Height Map')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[0, 1], label='Height (mm)')
        
        # 3D surface plot
        ax3d = fig.add_subplot(2, 2, 3, projection='3d')
        ax3d.plot_surface(x_grid, y_grid, height_map, cmap='viridis', alpha=0.8)
        ax3d.set_title(f'Sample {sample_idx}: 3D Top Surface')
        ax3d.set_xlabel('X (mm)')
        ax3d.set_ylabel('Y (mm)')
        ax3d.set_zlabel('Height (mm)')
        
        # Warp metrics
        metrics = sample['top_warp_metrics']
        metrics_text = f"""Warp Metrics:
Max Warp: {metrics.get('max_warp', 0):.3f} mm
RMS Warp: {metrics.get('rms_warp', 0):.3f} mm
Max Displacement: {metrics.get('max_displacement', 0):.3f} mm
Mean Displacement: {metrics.get('mean_displacement', 0):.3f} mm"""
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Warp field plot saved to {save_path}")
        
        plt.show()
    
    def plot_stress_field(self, sample_idx: int, save_path: Optional[str] = None):
        """Plot stress field for specific sample"""
        if sample_idx >= self.n_samples:
            print(f"Sample index {sample_idx} out of range")
            return
        
        sample = self.dataset['stress_data'][sample_idx]
        stress_maps = sample['electrolyte_stress_maps']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Von Mises stress map
        von_mises_map = stress_maps['von_mises_map']
        x_grid = stress_maps['x_grid']
        y_grid = stress_maps['y_grid']
        
        im1 = axes[0, 0].contourf(x_grid, y_grid, von_mises_map, levels=20, cmap='plasma')
        axes[0, 0].set_title(f'Sample {sample_idx}: Von Mises Stress')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Stress (MPa)')
        
        # Principal stress map
        principal_map = stress_maps['principal_1_map']
        im2 = axes[0, 1].contourf(x_grid, y_grid, principal_map, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title(f'Sample {sample_idx}: Principal Stress σ₁')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[0, 1], label='Stress (MPa)')
        
        # Stress distribution histogram
        von_mises_data = sample['electrolyte_von_mises']
        axes[1, 0].hist(von_mises_data, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title(f'Sample {sample_idx}: Von Mises Stress Distribution')
        axes[1, 0].set_xlabel('Stress (MPa)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Stress metrics
        metrics = sample['stress_metrics']['electrolyte']
        metrics_text = f"""Stress Metrics:
Max Von Mises: {metrics.get('max_von_mises', 0):.1f} MPa
Mean Von Mises: {metrics.get('mean_von_mises', 0):.1f} MPa
Max Principal: {metrics.get('max_principal_1', 0):.1f} MPa
Safety Factor: {metrics.get('safety_factor', 0):.2f}
Fracture Risk: {metrics.get('fracture_risk_level', 'unknown')}"""
        
        axes[1, 1].text(0.05, 0.95, metrics_text, transform=axes[1, 1].transAxes,
                       verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stress field plot saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_distributions(self, save_path: Optional[str] = None):
        """Plot parameter distributions"""
        # Convert parameters to DataFrame
        param_df = pd.DataFrame(self.dataset['parameters'])
        
        # Select key parameters for visualization
        key_params = [
            'cell_length', 'cell_width', 'electrolyte_thickness',
            'anode_thickness', 'cathode_thickness', 'sintering_temperature',
            'cooling_rate', 'assembly_pressure', 'operating_temperature'
        ]
        
        # Filter available parameters
        available_params = [p for p in key_params if p in param_df.columns]
        
        n_params = len(available_params)
        n_cols = 3
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(available_params):
            row = i // n_cols
            col = i % n_cols
            
            ax = axes[row, col]
            ax.hist(param_df[param], bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{param.replace("_", " ").title()}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            
            # Add statistics
            mean_val = param_df[param].mean()
            std_val = param_df[param].std()
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.2f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'±1σ: {std_val:.2f}')
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            ax.legend()
        
        # Hide empty subplots
        for i in range(n_params, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter distributions saved to {save_path}")
        
        plt.show()
    
    def plot_parameter_correlations(self, save_path: Optional[str] = None):
        """Plot parameter correlations"""
        # Convert parameters to DataFrame
        param_df = pd.DataFrame(self.dataset['parameters'])
        
        # Select key parameters
        key_params = [
            'cell_length', 'cell_width', 'electrolyte_thickness',
            'anode_thickness', 'cathode_thickness', 'sintering_temperature',
            'cooling_rate', 'assembly_pressure', 'operating_temperature'
        ]
        
        # Filter available parameters
        available_params = [p for p in key_params if p in param_df.columns]
        
        if len(available_params) < 2:
            print("Not enough parameters for correlation analysis")
            return
        
        # Create correlation matrix
        corr_matrix = param_df[available_params].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Set ticks and labels
        ax.set_xticks(range(len(available_params)))
        ax.set_yticks(range(len(available_params)))
        ax.set_xticklabels([p.replace('_', ' ').title() for p in available_params], rotation=45, ha='right')
        ax.set_yticklabels([p.replace('_', ' ').title() for p in available_params])
        
        # Add correlation values
        for i in range(len(available_params)):
            for j in range(len(available_params)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient')
        
        ax.set_title('Parameter Correlation Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter correlations saved to {save_path}")
        
        plt.show()
    
    def plot_warp_stress_correlation(self, save_path: Optional[str] = None):
        """Plot correlation between warp and stress metrics"""
        # Extract data
        max_warps = []
        rms_warps = []
        max_von_mises = []
        max_principal = []
        
        for i in range(self.n_samples):
            warp_metrics = self.dataset['warp_data'][i]['top_warp_metrics']
            stress_metrics = self.dataset['stress_data'][i]['stress_metrics']['electrolyte']
            
            max_warps.append(warp_metrics.get('max_warp', 0))
            rms_warps.append(warp_metrics.get('rms_warp', 0))
            max_von_mises.append(stress_metrics.get('max_von_mises', 0))
            max_principal.append(stress_metrics.get('max_principal_1', 0))
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Max warp vs Max Von Mises
        axes[0, 0].scatter(max_warps, max_von_mises, alpha=0.7)
        axes[0, 0].set_xlabel('Max Warp (mm)')
        axes[0, 0].set_ylabel('Max Von Mises Stress (MPa)')
        axes[0, 0].set_title('Max Warp vs Max Von Mises Stress')
        
        corr1, _ = pearsonr(max_warps, max_von_mises)
        axes[0, 0].text(0.05, 0.95, f'r = {corr1:.3f}', transform=axes[0, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # RMS warp vs Max Von Mises
        axes[0, 1].scatter(rms_warps, max_von_mises, alpha=0.7)
        axes[0, 1].set_xlabel('RMS Warp (mm)')
        axes[0, 1].set_ylabel('Max Von Mises Stress (MPa)')
        axes[0, 1].set_title('RMS Warp vs Max Von Mises Stress')
        
        corr2, _ = pearsonr(rms_warps, max_von_mises)
        axes[0, 1].text(0.05, 0.95, f'r = {corr2:.3f}', transform=axes[0, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Max warp vs Max Principal
        axes[1, 0].scatter(max_warps, max_principal, alpha=0.7)
        axes[1, 0].set_xlabel('Max Warp (mm)')
        axes[1, 0].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 0].set_title('Max Warp vs Max Principal Stress')
        
        corr3, _ = pearsonr(max_warps, max_principal)
        axes[1, 0].text(0.05, 0.95, f'r = {corr3:.3f}', transform=axes[1, 0].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # RMS warp vs Max Principal
        axes[1, 1].scatter(rms_warps, max_principal, alpha=0.7)
        axes[1, 1].set_xlabel('RMS Warp (mm)')
        axes[1, 1].set_ylabel('Max Principal Stress (MPa)')
        axes[1, 1].set_title('RMS Warp vs Max Principal Stress')
        
        corr4, _ = pearsonr(rms_warps, max_principal)
        axes[1, 1].text(0.05, 0.95, f'r = {corr4:.3f}', transform=axes[1, 1].transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Warp-stress correlations saved to {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    print("Dataset Visualizer Example")
    print("This module provides visualization capabilities for SOFC datasets")
    print("Use the examples/visualize_dataset.py script to see it in action")