"""
Visualization tools for synchrotron X-ray tomography and diffraction data.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
import plotly.graph_objects as go
import plotly.express as px
from scipy import ndimage
import os


class DataVisualizer:
    """Visualize synchrotron data."""
    
    def __init__(self):
        self.figures_dir = 'synchrotron_data/visualizations'
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def visualize_3d_tomography(self, h5_file, time_step=0, threshold=0.5):
        """
        Create 3D visualization of tomography data.
        
        Parameters:
        -----------
        h5_file : str
            Path to HDF5 file with tomography data
        time_step : int
            Time step to visualize (for time series data)
        threshold : float
            Threshold for isosurface extraction
        """
        print(f"Loading tomography data from {h5_file}...")
        
        with h5py.File(h5_file, 'r') as f:
            # Check if it's time series or single volume
            if f'time_{time_step:03d}' in f:
                volume = f[f'time_{time_step:03d}'][:]
                time_hours = f[f'time_{time_step:03d}'].attrs.get('time_hours', 0)
            else:
                volume = f['volume'][:]
                time_hours = 0
            
            voxel_size = f.get('volume', f[f'time_{time_step:03d}']).attrs.get('voxel_size', 1e-6)
        
        # Create orthogonal slices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # XY slice (axial)
        z_mid = volume.shape[0] // 2
        axes[0, 0].imshow(volume[z_mid, :, :], cmap='gray')
        axes[0, 0].set_title(f'XY Slice (z={z_mid})')
        axes[0, 0].set_xlabel('X [pixels]')
        axes[0, 0].set_ylabel('Y [pixels]')
        
        # XZ slice (sagittal)
        y_mid = volume.shape[1] // 2
        axes[0, 1].imshow(volume[:, y_mid, :], cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'XZ Slice (y={y_mid})')
        axes[0, 1].set_xlabel('X [pixels]')
        axes[0, 1].set_ylabel('Z [pixels]')
        
        # YZ slice (coronal)
        x_mid = volume.shape[2] // 2
        axes[0, 2].imshow(volume[:, :, x_mid], cmap='gray', aspect='auto')
        axes[0, 2].set_title(f'YZ Slice (x={x_mid})')
        axes[0, 2].set_xlabel('Y [pixels]')
        axes[0, 2].set_ylabel('Z [pixels]')
        
        # Phase histogram
        axes[1, 0].hist(volume.flatten(), bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('Gray Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Phase Distribution')
        axes[1, 0].set_yscale('log')
        
        # 3D projection (maximum intensity)
        projection = np.max(volume, axis=0)
        axes[1, 1].imshow(projection, cmap='hot')
        axes[1, 1].set_title('Maximum Intensity Projection')
        axes[1, 1].set_xlabel('X [pixels]')
        axes[1, 1].set_ylabel('Y [pixels]')
        
        # Feature statistics
        # Identify different phases
        phases = {
            'Matrix': (volume == 1).sum(),
            'Grain Boundary': (volume == 2).sum(),
            'Pores': (volume == 0).sum(),
            'Oxide': (volume == 3).sum(),
            'Cavities': (volume == 4).sum()
        }
        
        phases_filtered = {k: v for k, v in phases.items() if v > 0}
        axes[1, 2].bar(phases_filtered.keys(), phases_filtered.values())
        axes[1, 2].set_xlabel('Phase')
        axes[1, 2].set_ylabel('Voxel Count')
        axes[1, 2].set_title('Phase Fractions')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.suptitle(f'3D Tomography Visualization (t={time_hours:.1f} hours)', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'tomography_slices.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {output_file}")
        plt.close()
        
        return volume
    
    def visualize_creep_evolution(self, h5_file):
        """
        Visualize creep evolution metrics over time.
        
        Parameters:
        -----------
        h5_file : str
            Path to time series HDF5 file
        """
        print(f"Visualizing creep evolution from {h5_file}...")
        
        with h5py.File(h5_file, 'r') as f:
            # Load evolution data
            time_hours = f['evolution/time_hours'][:]
            cavity_volume = f['evolution/cavity_volume'][:]
            crack_length = f['evolution/crack_length'][:]
            strain = f['evolution/strain'][:]
            damage = f['evolution/damage'][:]
            
            # Get test conditions
            conditions = f.attrs.get('test_conditions', '{}')
            if isinstance(conditions, str):
                import json
                conditions = json.loads(conditions)
        
        # Create multi-panel plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cavity volume evolution
        axes[0, 0].plot(time_hours, cavity_volume, 'b-o', linewidth=2)
        axes[0, 0].set_xlabel('Time [hours]')
        axes[0, 0].set_ylabel('Cavity Volume [voxels]')
        axes[0, 0].set_title('Cavity Volume Evolution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Crack length evolution
        axes[0, 1].plot(time_hours, crack_length, 'r-s', linewidth=2)
        axes[0, 1].set_xlabel('Time [hours]')
        axes[0, 1].set_ylabel('Crack Length [μm]')
        axes[0, 1].set_title('Crack Length Evolution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Strain evolution
        axes[1, 0].plot(time_hours, np.array(strain) * 100, 'g-^', linewidth=2)
        axes[1, 0].set_xlabel('Time [hours]')
        axes[1, 0].set_ylabel('Creep Strain [%]')
        axes[1, 0].set_title('Creep Strain Evolution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Damage parameter evolution
        axes[1, 1].plot(time_hours, np.array(damage) * 100, 'm-d', linewidth=2)
        axes[1, 1].set_xlabel('Time [hours]')
        axes[1, 1].set_ylabel('Damage Parameter [%]')
        axes[1, 1].set_title('Damage Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add creep stages annotations
        if len(time_hours) > 3:
            # Primary creep
            axes[1, 0].axvspan(time_hours[0], time_hours[len(time_hours)//3], 
                              alpha=0.2, color='yellow', label='Primary')
            # Secondary creep
            axes[1, 0].axvspan(time_hours[len(time_hours)//3], time_hours[2*len(time_hours)//3],
                              alpha=0.2, color='green', label='Secondary')
            # Tertiary creep
            axes[1, 0].axvspan(time_hours[2*len(time_hours)//3], time_hours[-1],
                              alpha=0.2, color='red', label='Tertiary')
            axes[1, 0].legend()
        
        # Add title with test conditions
        title = 'Creep Evolution Analysis'
        if conditions:
            title += f"\nT={conditions.get('temperature', 'N/A')}°C, σ={conditions.get('stress', 'N/A')} MPa"
        plt.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'creep_evolution.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved evolution plot to {output_file}")
        plt.close()
    
    def visualize_diffraction_pattern(self, h5_file, pattern_idx=0):
        """
        Visualize X-ray diffraction pattern.
        
        Parameters:
        -----------
        h5_file : str
            Path to diffraction HDF5 file
        pattern_idx : int
            Pattern index to visualize
        """
        print(f"Visualizing diffraction pattern from {h5_file}...")
        
        with h5py.File(h5_file, 'r') as f:
            # Load pattern data
            if 'patterns' in f:
                pattern_grp = f[f'patterns/pattern_{pattern_idx:03d}']
                two_theta = pattern_grp['2theta'][:]
                intensity = pattern_grp['intensity'][:]
                
                # Get phase composition if available
                import json
                phase_comp = pattern_grp.attrs.get('phase_composition', '{}')
                if isinstance(phase_comp, str):
                    phase_comp = json.loads(phase_comp)
            else:
                print("No diffraction patterns found in file")
                return
        
        # Create figure with pattern and phase information
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        
        # Main diffraction pattern
        axes[0].plot(two_theta, intensity, 'b-', linewidth=1, label='Observed')
        axes[0].set_xlabel('2θ [degrees]')
        axes[0].set_ylabel('Intensity [a.u.]')
        axes[0].set_title('Synchrotron X-ray Diffraction Pattern')
        axes[0].grid(True, alpha=0.3)
        
        # Add peak markers for major phases
        peak_positions = {
            'Ferrite (110)': 44.7,
            'Ferrite (200)': 65.0,
            'Ferrite (211)': 82.3,
            'Austenite (111)': 43.6,
            'Austenite (200)': 50.8,
            'Cr2O3 (104)': 33.6,
            'Cr2O3 (110)': 36.2
        }
        
        for label, position in peak_positions.items():
            if position > two_theta[0] and position < two_theta[-1]:
                idx = np.argmin(np.abs(two_theta - position))
                if intensity[idx] > np.mean(intensity) + np.std(intensity):
                    axes[0].axvline(position, color='red', linestyle='--', alpha=0.5)
                    axes[0].text(position, intensity[idx], label.split()[0], 
                               rotation=90, fontsize=8, ha='right', va='bottom')
        
        # Phase composition bar chart
        if phase_comp:
            phases = list(phase_comp.keys())
            fractions = list(phase_comp.values())
            colors = ['blue', 'green', 'orange', 'red'][:len(phases)]
            
            bars = axes[1].bar(phases, fractions, color=colors, edgecolor='black')
            axes[1].set_ylabel('Phase Fraction')
            axes[1].set_title('Phase Composition')
            axes[1].set_ylim([0, 1])
            
            # Add percentage labels on bars
            for bar, fraction in zip(bars, fractions):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height/2,
                           f'{fraction*100:.1f}%', ha='center', va='center')
        
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'diffraction_pattern.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved diffraction pattern to {output_file}")
        plt.close()
    
    def visualize_strain_maps(self, h5_file, map_idx=0):
        """
        Visualize strain/stress maps.
        
        Parameters:
        -----------
        h5_file : str
            Path to strain map HDF5 file
        map_idx : int
            Map index to visualize
        """
        print(f"Visualizing strain/stress maps from {h5_file}...")
        
        with h5py.File(h5_file, 'r') as f:
            # Load strain map data
            if 'strain_maps' in f:
                map_grp = f[f'strain_maps/map_{map_idx:03d}']
                
                # Load different components
                strain_xx = map_grp['strain_xx'][:] * 1000  # Convert to millistrain
                strain_yy = map_grp['strain_yy'][:] * 1000
                stress_xx = map_grp['stress_xx'][:]
                stress_yy = map_grp['stress_yy'][:]
                von_mises = map_grp['von_mises_stress'][:]
                max_shear = map_grp['max_shear_strain'][:] * 1000
            else:
                print("No strain maps found in file")
                return
        
        # Create figure with multiple strain/stress components
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Strain εxx
        im1 = axes[0, 0].imshow(strain_xx, cmap='RdBu_r', aspect='auto')
        axes[0, 0].set_title('Strain εxx [mε]')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Strain εyy
        im2 = axes[0, 1].imshow(strain_yy, cmap='RdBu_r', aspect='auto')
        axes[0, 1].set_title('Strain εyy [mε]')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Max shear strain
        im3 = axes[0, 2].imshow(max_shear, cmap='viridis', aspect='auto')
        axes[0, 2].set_title('Max Shear Strain [mε]')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Stress σxx
        im4 = axes[1, 0].imshow(stress_xx, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Stress σxx [MPa]')
        plt.colorbar(im4, ax=axes[1, 0])
        
        # Stress σyy
        im5 = axes[1, 1].imshow(stress_yy, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Stress σyy [MPa]')
        plt.colorbar(im5, ax=axes[1, 1])
        
        # Von Mises stress
        im6 = axes[1, 2].imshow(von_mises, cmap='hot', aspect='auto')
        axes[1, 2].set_title('Von Mises Stress [MPa]')
        plt.colorbar(im6, ax=axes[1, 2])
        
        # Remove axis labels for cleaner look
        for ax in axes.flat:
            ax.set_xlabel('X [pixels]')
            ax.set_ylabel('Y [pixels]')
        
        plt.suptitle('Residual Strain/Stress Maps from X-ray Diffraction', fontsize=14)
        plt.tight_layout()
        
        output_file = os.path.join(self.figures_dir, 'strain_stress_maps.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"  Saved strain/stress maps to {output_file}")
        plt.close()
    
    def create_3d_interactive_volume(self, volume, output_file='3d_volume.html'):
        """
        Create interactive 3D visualization using plotly.
        
        Parameters:
        -----------
        volume : ndarray
            3D volume data
        output_file : str
            Output HTML file name
        """
        print("Creating interactive 3D visualization...")
        
        # Downsample for performance
        step = max(1, volume.shape[0] // 64)
        vol_small = volume[::step, ::step, ::step]
        
        # Create isosurface for each phase
        fig = go.Figure()
        
        # Add isosurface for cavities/pores
        if np.any(vol_small == 4) or np.any(vol_small == 0):
            X, Y, Z = np.mgrid[0:vol_small.shape[0], 0:vol_small.shape[1], 0:vol_small.shape[2]]
            
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=vol_small.flatten(),
                isomin=3.5,
                isomax=4.5,
                surface_count=1,
                opacity=0.6,
                colorscale='Reds',
                name='Cavities'
            ))
        
        fig.update_layout(
            title='3D Microstructure Visualization',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            )
        )
        
        output_path = os.path.join(self.figures_dir, output_file)
        fig.write_html(output_path)
        print(f"  Saved interactive 3D visualization to {output_path}")


def main():
    """Demo visualization of generated data."""
    
    print("\n=== Data Visualization Demo ===")
    
    viz = DataVisualizer()
    
    # Check if data exists
    tomo_file = 'synchrotron_data/tomography/time_series/creep_T700_S100.h5'
    diff_file = 'synchrotron_data/diffraction/time_series/insitu_T700_S100.h5'
    
    if os.path.exists(tomo_file):
        print("\n--- Visualizing Tomography Data ---")
        volume = viz.visualize_3d_tomography(tomo_file, time_step=5)
        viz.visualize_creep_evolution(tomo_file)
        viz.create_3d_interactive_volume(volume)
    
    if os.path.exists(diff_file):
        print("\n--- Visualizing Diffraction Data ---")
        viz.visualize_diffraction_pattern(diff_file, pattern_idx=5)
        viz.visualize_strain_maps(diff_file, map_idx=5)
    
    print("\n=== Visualization Complete ===")
    print(f"Figures saved in {viz.figures_dir}/")


if __name__ == '__main__':
    main()