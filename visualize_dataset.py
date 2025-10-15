"""
Visualization utilities for SOFC warp and stress dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
import json
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DatasetVisualizer:
    """Visualization tools for SOFC dataset."""
    
    def __init__(self, dataset_dir: str = "sofc_dataset"):
        self.dataset_dir = Path(dataset_dir)
        
        # Load metadata
        metadata_file = self.dataset_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
    
    def load_sample(self, sample_idx: int) -> Tuple[dict, dict, dict]:
        """Load a specific sample's data."""
        sample = self.metadata['samples'][sample_idx]
        
        # Load warp data
        warp_file = self.dataset_dir / sample['warp_file']
        warp_data = dict(np.load(warp_file))
        
        # Load stress data
        stress_file = self.dataset_dir / sample['stress_file']
        stress_data = dict(np.load(stress_file))
        
        # Parameters
        params = sample['parameters']
        
        return warp_data, stress_data, params
    
    def plot_warp_field(self, sample_idx: int, save_path: Optional[str] = None):
        """
        Visualize warp field for a sample.
        Shows top and bottom surface deformations.
        """
        warp_data, _, params = self.load_sample(sample_idx)
        
        X = warp_data['X'] * 1000  # Convert to mm
        Y = warp_data['Y'] * 1000
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Top surface
        ax = axes[0]
        im = ax.contourf(X, Y, warp_data['warp_top'], levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Top Surface Warp')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Warp (mm)')
        
        # Bottom surface
        ax = axes[1]
        im = ax.contourf(X, Y, warp_data['warp_bottom'], levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Bottom Surface Warp')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Warp (mm)')
        
        # Mean warp (3D surface plot)
        ax = axes[2]
        im = ax.contourf(X, Y, warp_data['warp_mean'], levels=20, cmap='viridis')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Mean Warp Field')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Warp (mm)')
        
        fig.suptitle(f'Sample {sample_idx}: Warp Field Visualization\n' + 
                    f'Peak Temp: {params["peak_sintering_temp"]:.0f}°C, ' +
                    f'Plate: {params["plate_length"]:.0f}x{params["plate_width"]:.0f} mm',
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved warp visualization to {save_path}")
        
        return fig
    
    def plot_stress_field(self, sample_idx: int, z_slice: Optional[int] = None,
                         save_path: Optional[str] = None):
        """
        Visualize stress field for a sample.
        Shows in-plane stress components at a specific z-slice.
        """
        warp_data, stress_data, params = self.load_sample(sample_idx)
        
        X = warp_data['X'] * 1000  # Convert to mm
        Y = warp_data['Y'] * 1000
        
        # Default to middle slice if not specified
        if z_slice is None:
            z_slice = stress_data['sigma_xx'].shape[2] // 2
        
        z_coord = stress_data['z_coords'][z_slice] * 1e6  # Convert to μm
        
        # Extract stress components at z-slice
        sigma_xx = stress_data['sigma_xx'][:, :, z_slice] / 1e6  # Convert to MPa
        sigma_yy = stress_data['sigma_yy'][:, :, z_slice] / 1e6
        sigma_xy = stress_data['sigma_xy'][:, :, z_slice] / 1e6
        
        # Compute von Mises stress
        sigma_zz = stress_data['sigma_zz'][:, :, z_slice] / 1e6
        sigma_yz = stress_data['sigma_yz'][:, :, z_slice] / 1e6
        sigma_xz = stress_data['sigma_xz'][:, :, z_slice] / 1e6
        
        sigma_vm = np.sqrt(0.5 * (
            (sigma_xx - sigma_yy)**2 +
            (sigma_yy - sigma_zz)**2 +
            (sigma_zz - sigma_xx)**2 +
            6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
        ))
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # σ_xx
        ax = axes[0, 0]
        im = ax.contourf(X, Y, sigma_xx, levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('σ_xx (MPa)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Stress (MPa)')
        
        # σ_yy
        ax = axes[0, 1]
        im = ax.contourf(X, Y, sigma_yy, levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('σ_yy (MPa)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Stress (MPa)')
        
        # σ_xy
        ax = axes[1, 0]
        im = ax.contourf(X, Y, sigma_xy, levels=20, cmap='RdBu_r')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('σ_xy (MPa)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Stress (MPa)')
        
        # Von Mises
        ax = axes[1, 1]
        im = ax.contourf(X, Y, sigma_vm, levels=20, cmap='jet')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Von Mises Stress (MPa)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='Stress (MPa)')
        
        fig.suptitle(f'Sample {sample_idx}: Stress Field at z = {z_coord:.1f} μm\n' +
                    f'Peak Temp: {params["peak_sintering_temp"]:.0f}°C, ' +
                    f'Total Thickness: {params["anode_thickness"] + params["electrolyte_thickness"] + params["cathode_thickness"]:.0f} μm',
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved stress visualization to {save_path}")
        
        return fig
    
    def plot_through_thickness_stress(self, sample_idx: int, 
                                     x_idx: Optional[int] = None,
                                     y_idx: Optional[int] = None,
                                     save_path: Optional[str] = None):
        """
        Plot through-thickness stress profile at a specific (x, y) location.
        """
        warp_data, stress_data, params = self.load_sample(sample_idx)
        
        # Default to center if not specified
        if x_idx is None:
            x_idx = warp_data['X'].shape[0] // 2
        if y_idx is None:
            y_idx = warp_data['Y'].shape[1] // 2
        
        z_coords = stress_data['z_coords'] * 1e6  # Convert to μm
        
        # Extract stress profile
        sigma_xx = stress_data['sigma_xx'][x_idx, y_idx, :] / 1e6
        sigma_yy = stress_data['sigma_yy'][x_idx, y_idx, :] / 1e6
        sigma_zz = stress_data['sigma_zz'][x_idx, y_idx, :] / 1e6
        
        # Von Mises
        sigma_xy = stress_data['sigma_xy'][x_idx, y_idx, :] / 1e6
        sigma_yz = stress_data['sigma_yz'][x_idx, y_idx, :] / 1e6
        sigma_xz = stress_data['sigma_xz'][x_idx, y_idx, :] / 1e6
        
        sigma_vm = np.sqrt(0.5 * (
            (sigma_xx - sigma_yy)**2 +
            (sigma_yy - sigma_zz)**2 +
            (sigma_zz - sigma_xx)**2 +
            6 * (sigma_xy**2 + sigma_yz**2 + sigma_xz**2)
        ))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(z_coords, sigma_xx, 'o-', label='σ_xx', linewidth=2)
        ax.plot(z_coords, sigma_yy, 's-', label='σ_yy', linewidth=2)
        ax.plot(z_coords, sigma_zz, '^-', label='σ_zz', linewidth=2)
        ax.plot(z_coords, sigma_vm, 'D-', label='Von Mises', linewidth=2, color='red')
        
        # Add layer boundaries
        t_a = params['anode_thickness']
        t_e = params['electrolyte_thickness']
        t_c = params['cathode_thickness']
        
        ax.axvline(t_a, color='k', linestyle='--', alpha=0.3, label='Anode/Electrolyte')
        ax.axvline(t_a + t_e, color='k', linestyle=':', alpha=0.3, label='Electrolyte/Cathode')
        
        # Annotate layers
        ax.text(t_a/2, ax.get_ylim()[1]*0.9, 'Anode', ha='center', fontsize=10, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.text(t_a + t_e/2, ax.get_ylim()[1]*0.9, 'Electrolyte', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.text(t_a + t_e + t_c/2, ax.get_ylim()[1]*0.9, 'Cathode', ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        
        ax.set_xlabel('Through-thickness Position (μm)', fontsize=12)
        ax.set_ylabel('Stress (MPa)', fontsize=12)
        ax.set_title(f'Sample {sample_idx}: Through-thickness Stress Profile\n' +
                    f'Location: x={warp_data["X"][x_idx, y_idx]*1000:.1f} mm, ' +
                    f'y={warp_data["Y"][x_idx, y_idx]*1000:.1f} mm',
                    fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved through-thickness stress plot to {save_path}")
        
        return fig
    
    def plot_parameter_distributions(self, save_path: Optional[str] = None):
        """Plot distributions of manufacturing parameters across the dataset."""
        # Extract all parameters
        param_names = list(self.metadata['samples'][0]['parameters'].keys())
        n_params = len(param_names)
        
        params_array = np.array([[s['parameters'][name] for name in param_names] 
                                for s in self.metadata['samples']])
        
        # Create subplots
        n_cols = 4
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
        axes = axes.flatten()
        
        for i, name in enumerate(param_names):
            ax = axes[i]
            ax.hist(params_array[:, i], bins=30, edgecolor='black', alpha=0.7)
            ax.set_xlabel(name.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(f'Manufacturing Parameter Distributions (N={len(self.metadata["samples"])})',
                    fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved parameter distributions to {save_path}")
        
        return fig
    
    def create_visualization_summary(self, n_samples: int = 5, output_dir: str = "visualizations"):
        """Create a comprehensive visualization summary for the dataset."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Creating visualization summary with {n_samples} samples...")
        print(f"Output directory: {output_path}")
        
        # Plot parameter distributions
        print("Plotting parameter distributions...")
        self.plot_parameter_distributions(save_path=output_path / "parameter_distributions.png")
        plt.close()
        
        # Plot sample visualizations
        sample_indices = np.linspace(0, len(self.metadata['samples']) - 1, n_samples, dtype=int)
        
        for i, idx in enumerate(sample_indices):
            print(f"Visualizing sample {i+1}/{n_samples} (index {idx})...")
            
            # Warp field
            self.plot_warp_field(idx, save_path=output_path / f"sample_{idx:05d}_warp.png")
            plt.close()
            
            # Stress field
            self.plot_stress_field(idx, save_path=output_path / f"sample_{idx:05d}_stress.png")
            plt.close()
            
            # Through-thickness
            self.plot_through_thickness_stress(idx, 
                                              save_path=output_path / f"sample_{idx:05d}_through_thickness.png")
            plt.close()
        
        print(f"\nVisualization summary complete! Files saved to {output_path}/")


def main():
    """Main entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize SOFC dataset')
    parser.add_argument('--dataset-dir', type=str, default='sofc_dataset',
                       help='Dataset directory (default: sofc_dataset)')
    parser.add_argument('--sample-idx', type=int, default=None,
                       help='Sample index to visualize (default: random)')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples for summary (default: 5)')
    parser.add_argument('--output-dir', type=str, default='visualizations',
                       help='Output directory for visualizations (default: visualizations)')
    parser.add_argument('--summary', action='store_true',
                       help='Create visualization summary')
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = DatasetVisualizer(dataset_dir=args.dataset_dir)
    
    if args.summary:
        # Create comprehensive summary
        visualizer.create_visualization_summary(n_samples=args.n_samples,
                                               output_dir=args.output_dir)
    else:
        # Visualize single sample
        if args.sample_idx is None:
            args.sample_idx = np.random.randint(0, len(visualizer.metadata['samples']))
        
        print(f"Visualizing sample {args.sample_idx}...")
        
        visualizer.plot_warp_field(args.sample_idx)
        visualizer.plot_stress_field(args.sample_idx)
        visualizer.plot_through_thickness_stress(args.sample_idx)
        
        plt.show()


if __name__ == "__main__":
    main()
