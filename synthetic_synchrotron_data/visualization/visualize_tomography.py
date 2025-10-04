#!/usr/bin/env python3
"""
Visualization Scripts for Synthetic Synchrotron Tomography Data

This script provides visualization tools for the synthetic tomography data,
including 3D rendering, slice views, and evolution animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import sys

class TomographyVisualizer:
    def __init__(self, data_directory):
        """
        Initialize visualizer with path to tomography data

        Parameters:
        -----------
        data_directory : str
            Path to tomography data directory
        """
        self.data_dir = data_directory

        # Load data if available
        self.load_data()

    def load_data(self):
        """Load tomography data from files"""
        try:
            # Load initial data
            self.initial_attenuation = np.load(f'{self.data_dir}/initial/attenuation_map.npy')
            self.initial_porosity = np.load(f'{self.data_dir}/initial/porosity_map.npy')
            self.initial_defects = np.load(f'{self.data_dir}/initial/defect_map.npy')

            print(f"Loaded initial data: {self.initial_attenuation.shape}")

            # Check for time series data
            timeseries_dir = f'{self.data_dir}/timeseries'
            if os.path.exists(timeseries_dir):
                # Find all attenuation files
                attenuation_files = [f for f in os.listdir(timeseries_dir)
                                   if f.startswith('attenuation_step_') and f.endswith('.npy')]

                if attenuation_files:
                    self.timeseries_data = []
                    for filename in sorted(attenuation_files):
                        step_num = int(filename.split('_')[-1].split('.')[0])
                        data = np.load(f'{timeseries_dir}/{filename}')
                        self.timeseries_data.append((step_num, data))

                    print(f"Loaded {len(self.timeseries_data)} time steps")

        except FileNotFoundError as e:
            print(f"Warning: Could not load all data files: {e}")
            self.initial_attenuation = None
            self.timeseries_data = []

    def plot_orthogonal_slices(self, data=None, step=0, slice_indices=None):
        """
        Plot orthogonal slices through the 3D volume

        Parameters:
        -----------
        data : array, optional
            3D data to visualize (if None, uses initial data)
        step : int
            Time step for timeseries data
        slice_indices : tuple, optional
            (x, y, z) slice indices (if None, uses center slices)
        """
        if data is None:
            if self.timeseries_data and step < len(self.timeseries_data):
                data = self.timeseries_data[step][1]
            else:
                data = self.initial_attenuation

        if data is None:
            print("No data available for visualization")
            return

        # Default slice indices (center of volume)
        if slice_indices is None:
            center = [s // 2 for s in data.shape]
            slice_indices = tuple(center)

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # XY slice
        axes[0, 0].imshow(data[:, :, slice_indices[2]], cmap='viridis',
                         origin='lower', aspect='equal')
        axes[0, 0].set_title(f'XY Slice (Z={slice_indices[2]})')
        axes[0, 0].set_xlabel('X (voxels)')
        axes[0, 0].set_ylabel('Y (voxels)')

        # XZ slice
        axes[0, 1].imshow(data[:, slice_indices[1], :], cmap='viridis',
                         origin='lower', aspect='equal')
        axes[0, 1].set_title(f'XZ Slice (Y={slice_indices[1]})')
        axes[0, 1].set_xlabel('X (voxels)')
        axes[0, 1].set_ylabel('Z (voxels)')

        # YZ slice
        axes[1, 0].imshow(data[slice_indices[0], :, :], cmap='viridis',
                         origin='lower', aspect='equal')
        axes[1, 0].set_title(f'YZ Slice (X={slice_indices[0]})')
        axes[1, 0].set_xlabel('Y (voxels)')
        axes[1, 0].set_ylabel('Z (voxels)')

        # 3D-like representation using all three slices
        axes[1, 1].remove()  # Remove the fourth subplot

        plt.tight_layout()
        plt.show()

    def plot_porosity_evolution(self, n_steps=5):
        """
        Plot porosity evolution over time

        Parameters:
        -----------
        n_steps : int
            Number of time steps to show
        """
        if not self.timeseries_data:
            print("No timeseries data available")
            return

        fig, axes = plt.subplots(1, n_steps, figsize=(15, 3))

        for i in range(min(n_steps, len(self.timeseries_data))):
            step_num, data = self.timeseries_data[i]
            porosity_map = np.load(f'{self.data_dir}/timeseries/porosity_step_{step_num:03d}.npy')

            # Plot center slice
            center_slice = porosity_map.shape[2] // 2
            im = axes[i].imshow(porosity_map[:, :, center_slice],
                              cmap='Reds', alpha=0.7, origin='lower')
            axes[i].set_title(f'Time: {step_num * 3.125:.1f} h')
            axes[i].set_xlabel('X (voxels)')
            if i == 0:
                axes[i].set_ylabel('Y (voxels)')

        plt.colorbar(im, ax=axes, label='Porosity')
        plt.tight_layout()
        plt.show()

    def plot_defect_evolution(self, n_steps=5):
        """
        Plot defect evolution over time

        Parameters:
        -----------
        n_steps : int
            Number of time steps to show
        """
        if not self.timeseries_data:
            print("No timeseries data available")
            return

        fig, axes = plt.subplots(1, n_steps, figsize=(15, 3))

        for i in range(min(n_steps, len(self.timeseries_data))):
            step_num, data = self.timeseries_data[i]
            defect_map = np.load(f'{self.data_dir}/timeseries/defects_step_{step_num:03d}.npy')

            # Plot center slice
            center_slice = defect_map.shape[2] // 2
            im = axes[i].imshow(defect_map[:, :, center_slice],
                              cmap='Blues', alpha=0.7, origin='lower')
            axes[i].set_title(f'Time: {step_num * 3.125:.1f} h')
            axes[i].set_xlabel('X (voxels)')
            if i == 0:
                axes[i].set_ylabel('Y (voxels)')

        plt.colorbar(im, ax=axes, label='Defects')
        plt.tight_layout()
        plt.show()

    def create_animation(self, output_file='tomography_evolution.gif', fps=2):
        """
        Create animation of tomography evolution

        Parameters:
        -----------
        output_file : str
            Output animation file path
        fps : float
            Frames per second for animation
        """
        if not self.timeseries_data:
            print("No timeseries data available for animation")
            return

        fig, ax = plt.subplots(figsize=(8, 8))

        def animate(frame):
            step_num, data = self.timeseries_data[frame]

            # Get center slice
            center_slice = data.shape[2] // 2
            im = ax.imshow(data[:, :, center_slice], cmap='viridis',
                          origin='lower', animated=True)

            ax.set_title(f'Attenuation Map - Time: {step_num * 3.125:.1f} h')
            ax.set_xlabel('X (voxels)')
            ax.set_ylabel('Y (voxels)')

            return [im]

        anim = animation.FuncAnimation(fig, animate, frames=len(self.timeseries_data),
                                    interval=1000/fps, blit=True)

        # Save animation
        print(f"Saving animation to {output_file}...")
        anim.save(output_file, writer='pillow', fps=fps)
        print("Animation saved!")

        plt.show()

    def plot_3d_volume(self, data=None, threshold=0.5, step=0):
        """
        Create 3D visualization of volume data

        Parameters:
        -----------
        data : array, optional
            3D data to visualize
        threshold : float
            Threshold for isosurface
        step : int
            Time step for timeseries data
        """
        if data is None:
            if self.timeseries_data and step < len(self.timeseries_data):
                data = self.timeseries_data[step][1]
            else:
                data = self.initial_attenuation

        if data is None:
            print("No data available for 3D visualization")
            return

        try:
            from skimage import measure

            # Create isosurface
            verts, faces, _, _ = measure.marching_cubes(data, level=threshold)

            # Create 3D plot
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot surface
            ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                          triangles=faces, cmap='viridis', alpha=0.6)

            ax.set_xlabel('X (voxels)')
            ax.set_ylabel('Y (voxels)')
            ax.set_zlabel('Z (voxels)')
            ax.set_title(f'3D Isosurface (threshold={threshold})')

            # Set equal aspect ratio
            max_range = max(data.shape) / 2.0
            mid_x = data.shape[0] / 2.0
            mid_y = data.shape[1] / 2.0
            mid_z = data.shape[2] / 2.0

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            plt.show()

        except ImportError:
            print("scikit-image not available for 3D visualization")
            print("Install with: pip install scikit-image")

def main():
    """Main visualization function"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_tomography.py <data_directory>")
        print("Example: python visualize_tomography.py ../tomography/")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found")
        sys.exit(1)

    print("Synthetic Tomography Data Visualizer")
    print("=" * 40)

    visualizer = TomographyVisualizer(data_dir)

    # Interactive visualization menu
    while True:
        print("\nVisualization Options:")
        print("1. Show orthogonal slices (initial)")
        print("2. Show orthogonal slices (timeseries)")
        print("3. Show porosity evolution")
        print("4. Show defect evolution")
        print("5. Create evolution animation")
        print("6. Show 3D volume")
        print("0. Exit")

        choice = input("\nSelect option (0-6): ").strip()

        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            visualizer.plot_orthogonal_slices()
        elif choice == '2':
            step = int(input("Enter time step (0-15): ") or "0")
            visualizer.plot_orthogonal_slices(step=step)
        elif choice == '3':
            n_steps = int(input("Number of steps to show (default 5): ") or "5")
            visualizer.plot_porosity_evolution(n_steps)
        elif choice == '4':
            n_steps = int(input("Number of steps to show (default 5): ") or "5")
            visualizer.plot_defect_evolution(n_steps)
        elif choice == '5':
            output_file = input("Output file (default: tomography_evolution.gif): ").strip() or "tomography_evolution.gif"
            fps = float(input("FPS (default 2): ") or "2")
            visualizer.create_animation(output_file, fps)
        elif choice == '6':
            step = int(input("Time step (0-15, default 0): ") or "0")
            threshold = float(input("Threshold (default 0.5): ") or "0.5")
            visualizer.plot_3d_volume(step=step, threshold=threshold)
        else:
            print("Invalid choice. Please select 0-6.")

if __name__ == "__main__":
    main()