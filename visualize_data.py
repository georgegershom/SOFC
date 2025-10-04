#!/usr/bin/env python3
"""
Visualization Tools for Synthetic Synchrotron X-ray Data
=========================================================

This script provides visualization capabilities for the generated
synchrotron data including 3D rendering, time-series plots, and
analysis dashboards.
"""

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse


class SynchrotronDataVisualizer:
    """
    Visualization tools for synchrotron X-ray data.
    """
    
    def __init__(self, data_dir="synchrotron_data"):
        """Initialize visualizer with data directory."""
        self.data_dir = Path(data_dir)
        self.tomography_dir = self.data_dir / "tomography"
        self.diffraction_dir = self.data_dir / "diffraction"
        self.metadata_dir = self.data_dir / "metadata"
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
    
    def plot_tomography_slices(self, time_step=0, output_file=None):
        """
        Plot 2D slices through 3D tomography volume.
        
        Parameters:
        -----------
        time_step : int
            Which time step to visualize
        output_file : str, optional
            Save figure to file
        """
        tomo_file = self.tomography_dir / "tomography_4D.h5"
        
        with h5py.File(tomo_file, 'r') as f:
            volume = f['tomography'][time_step]
            time = f['time_hours'][time_step]
            voxel_size = f.attrs['voxel_size_um']
        
        # Get middle slices
        mid_z, mid_y, mid_x = [s // 2 for s in volume.shape]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Tomography Slices at t = {time:.1f} hours', fontsize=16)
        
        # XY slice (top view)
        im1 = axes[0, 0].imshow(volume[mid_z, :, :], cmap='gray', origin='lower')
        axes[0, 0].set_title('XY Slice (Top View)')
        axes[0, 0].set_xlabel(f'X [pixels, {voxel_size} μm/pixel]')
        axes[0, 0].set_ylabel(f'Y [pixels]')
        plt.colorbar(im1, ax=axes[0, 0], label='Normalized Density')
        
        # XZ slice (side view)
        im2 = axes[0, 1].imshow(volume[:, mid_y, :], cmap='gray', origin='lower')
        axes[0, 1].set_title('XZ Slice (Side View)')
        axes[0, 1].set_xlabel(f'X [pixels]')
        axes[0, 1].set_ylabel(f'Z [pixels]')
        plt.colorbar(im2, ax=axes[0, 1], label='Normalized Density')
        
        # YZ slice (front view)
        im3 = axes[1, 0].imshow(volume[:, :, mid_x], cmap='gray', origin='lower')
        axes[1, 0].set_title('YZ Slice (Front View)')
        axes[1, 0].set_xlabel(f'Y [pixels]')
        axes[1, 0].set_ylabel(f'Z [pixels]')
        plt.colorbar(im3, ax=axes[1, 0], label='Normalized Density')
        
        # Histogram
        axes[1, 1].hist(volume.ravel(), bins=100, color='blue', alpha=0.7)
        axes[1, 1].set_title('Density Distribution')
        axes[1, 1].set_xlabel('Normalized Density')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(0.3, color='red', linestyle='--', label='Void threshold')
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_file}")
        else:
            plt.show()
    
    def plot_creep_evolution(self, output_file=None):
        """
        Plot time evolution of creep damage metrics.
        
        Parameters:
        -----------
        output_file : str, optional
            Save figure to file
        """
        metrics_file = self.tomography_dir / "tomography_metrics.json"
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Creep Damage Evolution Over Time', fontsize=16, fontweight='bold')
        
        time = metrics['time_hours']
        
        # Porosity
        axes[0, 0].plot(time, metrics['porosity_percent'], 'o-', linewidth=2, 
                       markersize=8, color='#2E86AB')
        axes[0, 0].set_xlabel('Time [hours]', fontsize=11)
        axes[0, 0].set_ylabel('Porosity [%]', fontsize=11)
        axes[0, 0].set_title('Porosity Evolution', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cavity count
        axes[0, 1].plot(time, metrics['cavity_count'], 's-', linewidth=2, 
                       markersize=8, color='#A23B72')
        axes[0, 1].set_xlabel('Time [hours]', fontsize=11)
        axes[0, 1].set_ylabel('Cavity Count', fontsize=11)
        axes[0, 1].set_title('Cavity Nucleation and Growth', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Crack volume
        axes[1, 0].plot(time, metrics['crack_volume_mm3'], '^-', linewidth=2, 
                       markersize=8, color='#F18F01')
        axes[1, 0].set_xlabel('Time [hours]', fontsize=11)
        axes[1, 0].set_ylabel('Crack Volume [mm³]', fontsize=11)
        axes[1, 0].set_title('Crack Propagation', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Grain boundary integrity
        axes[1, 1].plot(time, metrics['mean_grain_boundary_integrity'], 'd-', 
                       linewidth=2, markersize=8, color='#C73E1D')
        axes[1, 1].set_xlabel('Time [hours]', fontsize=11)
        axes[1, 1].set_ylabel('Mean GB Integrity', fontsize=11)
        axes[1, 1].set_title('Grain Boundary Degradation', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_file}")
        else:
            plt.show()
    
    def plot_xrd_patterns(self, output_file=None):
        """
        Plot X-ray diffraction patterns.
        
        Parameters:
        -----------
        output_file : str, optional
            Save figure to file
        """
        pattern_file = self.diffraction_dir / "xrd_patterns.json"
        
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        two_theta = np.array(data['two_theta_deg'])
        intensity = np.array(data['intensity_counts'])
        
        ax.plot(two_theta, intensity, linewidth=1.5, color='#1B4965')
        ax.fill_between(two_theta, intensity, alpha=0.3, color='#1B4965')
        
        # Mark major phases
        phases = data['phases_detected']
        colors = {'ferrite_alpha_Fe': 'red', 'chromia_Cr2O3': 'green'}
        
        for phase_name, phase_data in phases.items():
            peaks = phase_data['peaks_deg']
            color = colors.get(phase_name, 'blue')
            label = phase_name.replace('_', ' ')
            
            for i, peak in enumerate(peaks):
                if i == 0:
                    ax.axvline(peak, color=color, linestyle='--', alpha=0.7, 
                             label=f"{label} ({phase_data['fraction']*100:.0f}%)")
                else:
                    ax.axvline(peak, color=color, linestyle='--', alpha=0.7)
        
        ax.set_xlabel('2θ [degrees]', fontsize=12)
        ax.set_ylabel('Intensity [counts]', fontsize=12)
        ax.set_title('X-ray Diffraction Pattern - Phase Identification', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_file}")
        else:
            plt.show()
    
    def plot_strain_maps(self, time_step=0, output_file=None):
        """
        Plot strain and stress distribution maps.
        
        Parameters:
        -----------
        time_step : int
            Which time step to visualize
        output_file : str, optional
            Save figure to file
        """
        strain_file = self.diffraction_dir / "strain_stress_maps.h5"
        
        with h5py.File(strain_file, 'r') as f:
            strain = f['elastic_strain'][time_step]
            stress = f['residual_stress_MPa'][time_step]
            time = f['time_hours'][time_step]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Strain and Stress Maps at t = {time:.1f} hours', 
                    fontsize=14, fontweight='bold')
        
        # Strain map
        im1 = axes[0].imshow(strain, cmap='RdYlBu_r', origin='lower')
        axes[0].set_title('Elastic Strain Distribution')
        axes[0].set_xlabel('X [pixels]')
        axes[0].set_ylabel('Y [pixels]')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Strain [ε]', rotation=270, labelpad=20)
        
        # Stress map
        im2 = axes[1].imshow(stress, cmap='plasma', origin='lower')
        axes[1].set_title('Residual Stress Distribution')
        axes[1].set_xlabel('X [pixels]')
        axes[1].set_ylabel('Y [pixels]')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_label('Stress [MPa]', rotation=270, labelpad=20)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_file}")
        else:
            plt.show()
    
    def plot_3d_volume_rendering(self, time_step=0, threshold=0.3, output_file=None):
        """
        Create 3D visualization of damage (voids and cracks).
        
        Parameters:
        -----------
        time_step : int
            Which time step to visualize
        threshold : float
            Threshold for identifying voids
        output_file : str, optional
            Save figure to file
        """
        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            print("3D plotting requires mpl_toolkits.mplot3d")
            return
        
        tomo_file = self.tomography_dir / "tomography_4D.h5"
        
        with h5py.File(tomo_file, 'r') as f:
            volume = f['tomography'][time_step]
            time = f['time_hours'][time_step]
        
        # Downsample for visualization
        volume_small = volume[::4, ::4, ::4]
        
        # Find voids
        voids = volume_small < threshold
        
        # Get coordinates of void voxels
        z, y, x = np.where(voids)
        
        # Subsample if too many points
        if len(x) > 10000:
            indices = np.random.choice(len(x), 10000, replace=False)
            x, y, z = x[indices], y[indices], z[indices]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by depth
        colors = z / z.max()
        
        scatter = ax.scatter(x, y, z, c=colors, cmap='hot', 
                           marker='.', s=1, alpha=0.6)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'3D Void Distribution at t = {time:.1f} hours', 
                    fontweight='bold')
        
        plt.colorbar(scatter, ax=ax, label='Depth', shrink=0.5)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved figure to {output_file}")
        else:
            plt.show()
    
    def create_summary_dashboard(self, output_file='dashboard.png'):
        """
        Create comprehensive visualization dashboard.
        """
        print("\nGenerating summary dashboard...")
        
        # Load all data
        metrics_file = self.tomography_dir / "tomography_metrics.json"
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        tomo_file = self.tomography_dir / "tomography_4D.h5"
        with h5py.File(tomo_file, 'r') as f:
            volume_0 = f['tomography'][0]
            volume_final = f['tomography'][-1]
            time_final = f['time_hours'][-1]
        
        # Create figure
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Title
        fig.suptitle('Synchrotron X-ray Analysis Dashboard - SOFC Creep Study', 
                    fontsize=16, fontweight='bold')
        
        # 1. Initial microstructure
        ax1 = fig.add_subplot(gs[0, 0])
        mid_z = volume_0.shape[0] // 2
        im1 = ax1.imshow(volume_0[mid_z, :, :], cmap='gray')
        ax1.set_title('Initial State (t=0)')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # 2. Final microstructure
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(volume_final[mid_z, :, :], cmap='gray')
        ax2.set_title(f'Final State (t={time_final:.0f}h)')
        ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        # 3. Difference map
        ax3 = fig.add_subplot(gs[0, 2])
        diff = volume_0[mid_z, :, :] - volume_final[mid_z, :, :]
        im3 = ax3.imshow(diff, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax3.set_title('Damage Map (Δ)')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, label='Density Loss')
        
        # 4. Porosity evolution
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(metrics['time_hours'], metrics['porosity_percent'], 
                'o-', linewidth=2, markersize=6)
        ax4.set_xlabel('Time [hours]')
        ax4.set_ylabel('Porosity [%]')
        ax4.set_title('Porosity Evolution')
        ax4.grid(True, alpha=0.3)
        
        # 5. Cavity count
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(metrics['time_hours'], metrics['cavity_count'], 
                's-', linewidth=2, markersize=6, color='#A23B72')
        ax5.set_xlabel('Time [hours]')
        ax5.set_ylabel('Cavity Count')
        ax5.set_title('Cavity Nucleation')
        ax5.grid(True, alpha=0.3)
        
        # 6. Crack volume
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.plot(metrics['time_hours'], metrics['crack_volume_mm3'], 
                '^-', linewidth=2, markersize=6, color='#F18F01')
        ax6.set_xlabel('Time [hours]')
        ax6.set_ylabel('Crack Volume [mm³]')
        ax6.set_title('Crack Propagation')
        ax6.grid(True, alpha=0.3)
        
        # 7. XRD pattern
        pattern_file = self.diffraction_dir / "xrd_patterns.json"
        with open(pattern_file, 'r') as f:
            xrd_data = json.load(f)
        
        ax7 = fig.add_subplot(gs[2, :])
        two_theta = np.array(xrd_data['two_theta_deg'])
        intensity = np.array(xrd_data['intensity_counts'])
        ax7.plot(two_theta, intensity, linewidth=1.5)
        ax7.fill_between(two_theta, intensity, alpha=0.3)
        ax7.set_xlabel('2θ [degrees]')
        ax7.set_ylabel('Intensity [counts]')
        ax7.set_title('X-ray Diffraction Pattern')
        ax7.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved dashboard to {output_file}")
        
        return output_file


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Visualize synthetic synchrotron X-ray data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="synchrotron_data",
        help="Directory containing generated data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization outputs"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize visualizer
    viz = SynchrotronDataVisualizer(data_dir=args.data_dir)
    
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    
    # Generate all visualizations
    print("\n[1/6] Creating tomography slices...")
    viz.plot_tomography_slices(time_step=0, 
                               output_file=output_dir / "tomography_initial.png")
    viz.plot_tomography_slices(time_step=-1, 
                               output_file=output_dir / "tomography_final.png")
    
    print("[2/6] Creating creep evolution plots...")
    viz.plot_creep_evolution(output_file=output_dir / "creep_evolution.png")
    
    print("[3/6] Creating XRD patterns...")
    viz.plot_xrd_patterns(output_file=output_dir / "xrd_patterns.png")
    
    print("[4/6] Creating strain/stress maps...")
    viz.plot_strain_maps(time_step=0, 
                        output_file=output_dir / "strain_maps_initial.png")
    viz.plot_strain_maps(time_step=-1, 
                        output_file=output_dir / "strain_maps_final.png")
    
    print("[5/6] Creating 3D volume rendering...")
    viz.plot_3d_volume_rendering(time_step=-1, 
                                output_file=output_dir / "3d_voids.png")
    
    print("[6/6] Creating summary dashboard...")
    viz.create_summary_dashboard(output_file=output_dir / "dashboard.png")
    
    print("\n" + "="*70)
    print("✓ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
