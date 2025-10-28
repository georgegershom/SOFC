#!/usr/bin/env python3
"""
Visualization tools for synthetic synchrotron X-ray data

This module provides comprehensive visualization capabilities for the generated
4D tomography and XRD data, including damage evolution analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import os
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from skimage import measure
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SynchrotronDataVisualizer:
    """
    Comprehensive visualization tool for synthetic synchrotron data
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize visualizer with data directory
        
        Args:
            data_directory: Path to directory containing synchrotron data
        """
        self.data_dir = Path(data_directory)
        self.metadata = self._load_metadata()
        self.analysis_metrics = self._load_analysis_metrics()
        
    def _load_metadata(self) -> Dict:
        """Load metadata from JSON file"""
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    def _load_analysis_metrics(self) -> Dict:
        """Load analysis metrics from JSON file"""
        metrics_path = self.data_dir / 'analysis_metrics.json'
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Analysis metrics file not found: {metrics_path}")
    
    def _load_tomography_data(self) -> Dict[float, np.ndarray]:
        """Load 4D tomography data from HDF5 file"""
        tomo_path = self.data_dir / 'tomography_4d.h5'
        tomography_data = {}
        
        with h5py.File(tomo_path, 'r') as f:
            for key in f.keys():
                time_str = key.replace('time_', '').replace('h', '')
                time_point = float(time_str)
                tomography_data[time_point] = f[key][:]
                
        return tomography_data
    
    def _load_xrd_data(self) -> Dict:
        """Load XRD data from HDF5 file"""
        xrd_path = self.data_dir / 'xrd_data.h5'
        xrd_data = {}
        
        with h5py.File(xrd_path, 'r') as f:
            for time_key in f.keys():
                time_str = time_key.replace('time_', '').replace('h', '')
                time_point = float(time_str)
                
                time_group = f[time_key]
                xrd_data[time_point] = {
                    'strain_map': time_group['strain_map'][:],
                    'stress_map': time_group['stress_map'][:],
                    'diffraction_patterns': {}
                }
                
                # Load diffraction patterns
                patterns_group = time_group['diffraction_patterns']
                for phase_name in patterns_group.keys():
                    phase_group = patterns_group[phase_name]
                    pattern = {}
                    for key in phase_group.keys():
                        pattern[key] = phase_group[key][:]
                    xrd_data[time_point]['diffraction_patterns'][phase_name] = pattern
        
        return xrd_data
    
    def plot_damage_evolution(self, save_path: Optional[str] = None):
        """Plot damage evolution metrics over time"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Creep Damage Evolution Analysis', fontsize=16, fontweight='bold')
        
        # Extract time points and convert to arrays for plotting
        time_points = sorted(self.analysis_metrics['porosity_evolution'].keys())
        time_array = np.array([float(t) for t in time_points])
        
        # Porosity evolution
        porosity = [self.analysis_metrics['porosity_evolution'][str(t)] for t in time_array]
        axes[0, 0].plot(time_array, porosity, 'o-', linewidth=2, markersize=8, color='red')
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Porosity Fraction')
        axes[0, 0].set_title('Porosity Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Damage evolution
        damage = [self.analysis_metrics['damage_evolution'][str(t)] for t in time_array]
        axes[0, 1].plot(time_array, damage, 'o-', linewidth=2, markersize=8, color='orange')
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Damage Parameter')
        axes[0, 1].set_title('Overall Damage Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Crack density evolution
        crack_density = [self.analysis_metrics['crack_density_evolution'][str(t)] for t in time_array]
        axes[1, 0].plot(time_array, crack_density, 'o-', linewidth=2, markersize=8, color='blue')
        axes[1, 0].set_xlabel('Time (hours)')
        axes[1, 0].set_ylabel('Crack Density (per mm³)')
        axes[1, 0].set_title('Crack Density Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Connectivity evolution (Euler characteristic)
        connectivity = [self.analysis_metrics['connectivity_evolution'][str(t)] for t in time_array]
        axes[1, 1].plot(time_array, connectivity, 'o-', linewidth=2, markersize=8, color='green')
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Euler Characteristic')
        axes[1, 1].set_title('Pore Connectivity Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Damage evolution plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_microstructure_slices(self, time_points: Optional[List[float]] = None, 
                                 slice_index: Optional[int] = None, save_path: Optional[str] = None):
        """Plot 2D slices of microstructure at different time points"""
        
        tomography_data = self._load_tomography_data()
        
        if time_points is None:
            time_points = sorted(list(tomography_data.keys()))[:6]  # First 6 time points
        
        if slice_index is None:
            slice_index = tomography_data[time_points[0]].shape[2] // 2  # Middle slice
        
        n_times = len(time_points)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten() if n_times > 1 else [axes]
        
        fig.suptitle(f'Microstructure Evolution (Z-slice {slice_index})', fontsize=16, fontweight='bold')
        
        for i, time_point in enumerate(time_points[:6]):
            if i >= len(axes):
                break
                
            structure = tomography_data[time_point]
            slice_data = structure[:, :, slice_index]
            
            # Create custom colormap for different phases
            im = axes[i].imshow(slice_data, cmap='viridis', origin='lower')
            axes[i].set_title(f'Time: {time_point:.1f} hours')
            axes[i].set_xlabel('X (voxels)')
            axes[i].set_ylabel('Y (voxels)')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i])
            cbar.set_label('Phase ID')
        
        # Hide unused subplots
        for i in range(len(time_points), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Microstructure slices plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_3d_damage_visualization(self, time_point: float, save_path: Optional[str] = None):
        """Create 3D visualization of damage at a specific time point"""
        
        tomography_data = self._load_tomography_data()
        
        if time_point not in tomography_data:
            available_times = list(tomography_data.keys())
            time_point = min(available_times, key=lambda x: abs(x - time_point))
            print(f"Requested time not available. Using closest time: {time_point}")
        
        structure = tomography_data[time_point]
        
        # Extract pores (damage) for 3D visualization
        pores = (structure == 0)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Find pore coordinates
        pore_coords = np.where(pores)
        
        # Sample points for visualization (too many points slow down rendering)
        n_points = min(10000, len(pore_coords[0]))
        if len(pore_coords[0]) > n_points:
            indices = np.random.choice(len(pore_coords[0]), n_points, replace=False)
            x = pore_coords[0][indices]
            y = pore_coords[1][indices]
            z = pore_coords[2][indices]
        else:
            x, y, z = pore_coords
        
        # Create 3D scatter plot
        scatter = ax.scatter(x, y, z, c=z, cmap='Reds', alpha=0.6, s=1)
        
        ax.set_xlabel('X (voxels)')
        ax.set_ylabel('Y (voxels)')
        ax.set_zlabel('Z (voxels)')
        ax.set_title(f'3D Damage Visualization at {time_point:.1f} hours')
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Z coordinate')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D damage visualization saved to: {save_path}")
        else:
            plt.show()
    
    def plot_xrd_patterns(self, time_points: Optional[List[float]] = None, 
                         save_path: Optional[str] = None):
        """Plot X-ray diffraction patterns evolution"""
        
        xrd_data = self._load_xrd_data()
        
        if time_points is None:
            time_points = sorted(list(xrd_data.keys()))[:4]  # First 4 time points
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        fig.suptitle('X-ray Diffraction Pattern Evolution', fontsize=16, fontweight='bold')
        
        for i, time_point in enumerate(time_points[:4]):
            if i >= len(axes):
                break
                
            patterns = xrd_data[time_point]['diffraction_patterns']
            
            for phase_name, pattern in patterns.items():
                two_theta = pattern['two_theta']
                intensity = pattern['intensity']
                
                axes[i].plot(two_theta, intensity, 'o-', label=phase_name, linewidth=2, markersize=4)
            
            axes[i].set_xlabel('2θ (degrees)')
            axes[i].set_ylabel('Intensity (counts)')
            axes[i].set_title(f'Time: {time_point:.1f} hours')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(time_points), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"XRD patterns plot saved to: {save_path}")
        else:
            plt.show()
    
    def plot_strain_stress_maps(self, time_point: float, slice_index: Optional[int] = None,
                               save_path: Optional[str] = None):
        """Plot strain and stress field maps"""
        
        xrd_data = self._load_xrd_data()
        
        if time_point not in xrd_data:
            available_times = list(xrd_data.keys())
            time_point = min(available_times, key=lambda x: abs(x - time_point))
            print(f"Requested time not available. Using closest time: {time_point}")
        
        strain_map = xrd_data[time_point]['strain_map']
        stress_map = xrd_data[time_point]['stress_map']
        
        if slice_index is None:
            slice_index = strain_map.shape[2] // 2  # Middle slice
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Strain and Stress Fields at {time_point:.1f} hours (Z-slice {slice_index})', 
                    fontsize=16, fontweight='bold')
        
        # Strain components
        strain_components = ['εxx', 'εyy', 'εzz']
        for i, (component, label) in enumerate(zip(range(3), strain_components)):
            strain_slice = strain_map[:, :, slice_index, component]
            im1 = axes[0, i].imshow(strain_slice, cmap='RdBu_r', origin='lower')
            axes[0, i].set_title(f'Strain {label}')
            axes[0, i].set_xlabel('X (voxels)')
            axes[0, i].set_ylabel('Y (voxels)')
            plt.colorbar(im1, ax=axes[0, i], label='Strain')
        
        # Stress components
        stress_components = ['σxx', 'σyy', 'σzz']
        for i, (component, label) in enumerate(zip(range(3), stress_components)):
            stress_slice = stress_map[:, :, slice_index, component]
            im2 = axes[1, i].imshow(stress_slice, cmap='plasma', origin='lower')
            axes[1, i].set_title(f'Stress {label}')
            axes[1, i].set_xlabel('X (voxels)')
            axes[1, i].set_ylabel('Y (voxels)')
            plt.colorbar(im2, ax=axes[1, i], label='Stress (MPa)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Strain/stress maps saved to: {save_path}")
        else:
            plt.show()
    
    def create_comprehensive_report(self, output_dir: Optional[str] = None):
        """Create a comprehensive visualization report"""
        
        if output_dir is None:
            output_dir = self.data_dir / 'visualization_report'
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        
        print("Creating comprehensive visualization report...")
        
        # 1. Damage evolution plot
        self.plot_damage_evolution(save_path=output_dir / 'damage_evolution.png')
        
        # 2. Microstructure slices
        self.plot_microstructure_slices(save_path=output_dir / 'microstructure_evolution.png')
        
        # 3. 3D damage visualization (final time point)
        tomography_data = self._load_tomography_data()
        final_time = max(tomography_data.keys())
        self.plot_3d_damage_visualization(final_time, save_path=output_dir / '3d_damage_final.png')
        
        # 4. XRD patterns
        self.plot_xrd_patterns(save_path=output_dir / 'xrd_patterns.png')
        
        # 5. Strain/stress maps (final time point)
        self.plot_strain_stress_maps(final_time, save_path=output_dir / 'strain_stress_maps.png')
        
        # 6. Create summary HTML report
        self._create_html_report(output_dir)
        
        print(f"Comprehensive report created in: {output_dir}")
    
    def _create_html_report(self, output_dir: Path):
        """Create an HTML summary report"""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Synchrotron Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
                .metadata {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
                .figure {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Synthetic Synchrotron X-ray Data Analysis Report</h1>
            
            <div class="metadata">
                <h2>Experiment Metadata</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Generation Date</td><td>{self.metadata['generation_timestamp']}</td></tr>
                    <tr><td>Data Type</td><td>{self.metadata['data_type']}</td></tr>
                    <tr><td>Experiment Type</td><td>{self.metadata['experiment_type']}</td></tr>
                    <tr><td>Temperature</td><td>{self.metadata['operational_parameters']['temperature']}°C</td></tr>
                    <tr><td>Mechanical Stress</td><td>{self.metadata['operational_parameters']['mechanical_stress']} MPa</td></tr>
                    <tr><td>Voxel Size</td><td>{self.metadata['voxel_size_um']} μm</td></tr>
                    <tr><td>Image Dimensions</td><td>{self.metadata['image_dimensions']}</td></tr>
                </table>
            </div>
            
            <h2>Damage Evolution Analysis</h2>
            <div class="figure">
                <img src="damage_evolution.png" alt="Damage Evolution">
                <p><em>Figure 1: Evolution of key damage parameters over time</em></p>
            </div>
            
            <h2>Microstructure Evolution</h2>
            <div class="figure">
                <img src="microstructure_evolution.png" alt="Microstructure Evolution">
                <p><em>Figure 2: 2D slices showing microstructural changes over time</em></p>
            </div>
            
            <h2>3D Damage Visualization</h2>
            <div class="figure">
                <img src="3d_damage_final.png" alt="3D Damage">
                <p><em>Figure 3: 3D visualization of damage distribution at final time point</em></p>
            </div>
            
            <h2>X-ray Diffraction Analysis</h2>
            <div class="figure">
                <img src="xrd_patterns.png" alt="XRD Patterns">
                <p><em>Figure 4: Evolution of X-ray diffraction patterns</em></p>
            </div>
            
            <h2>Strain and Stress Field Analysis</h2>
            <div class="figure">
                <img src="strain_stress_maps.png" alt="Strain Stress Maps">
                <p><em>Figure 5: Spatial distribution of strain and stress fields</em></p>
            </div>
            
            <h2>Summary</h2>
            <p>This synthetic dataset provides comprehensive 4D synchrotron X-ray data for SOFC creep 
            deformation analysis. The data includes realistic microstructural evolution, damage progression, 
            and corresponding X-ray diffraction signatures that can be used for model validation and 
            algorithm development.</p>
            
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html_content)

def main():
    """Main function for command-line usage"""
    
    parser = argparse.ArgumentParser(description='Visualize synthetic synchrotron X-ray data')
    parser.add_argument('data_directory', help='Directory containing synchrotron data')
    parser.add_argument('--output', '-o', help='Output directory for plots')
    parser.add_argument('--report', '-r', action='store_true', 
                       help='Generate comprehensive report')
    
    args = parser.parse_args()
    
    try:
        visualizer = SynchrotronDataVisualizer(args.data_directory)
        
        if args.report:
            visualizer.create_comprehensive_report(args.output)
        else:
            # Interactive mode - show individual plots
            print("Creating individual visualizations...")
            visualizer.plot_damage_evolution()
            visualizer.plot_microstructure_slices()
            
            # Get final time point for 3D and field visualizations
            tomography_data = visualizer._load_tomography_data()
            final_time = max(tomography_data.keys())
            
            visualizer.plot_3d_damage_visualization(final_time)
            visualizer.plot_xrd_patterns()
            visualizer.plot_strain_stress_maps(final_time)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())