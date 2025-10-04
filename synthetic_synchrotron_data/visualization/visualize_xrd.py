#!/usr/bin/env python3
"""
Visualization Scripts for Synthetic XRD Data

This script provides visualization tools for the synthetic XRD patterns,
including peak identification, stress mapping, and time evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

class XRDVisualizer:
    def __init__(self, data_directory):
        """
        Initialize XRD visualizer

        Parameters:
        -----------
        data_directory : str
            Path to XRD data directory
        """
        self.data_dir = data_directory
        self.reference_patterns = {}
        self.spatial_data = None
        self.timeseries_data = []

        # Load data
        self.load_data()

    def load_data(self):
        """Load XRD data from files"""
        try:
            # Load reference patterns
            for filename in os.listdir(self.data_dir):
                if filename.endswith('_reference.json'):
                    phase_name = filename.replace('_reference.json', '').replace('_', ' ').title()
                    with open(f'{self.data_dir}/{filename}', 'r') as f:
                        data = json.load(f)
                        self.reference_patterns[phase_name] = data

            print(f"Loaded {len(self.reference_patterns)} reference patterns")

            # Load spatial mapping data
            spatial_file = f'{self.data_dir}/spatial_xrd_mapping.json'
            if os.path.exists(spatial_file):
                with open(spatial_file, 'r') as f:
                    self.spatial_data = json.load(f)
                print("Loaded spatial XRD mapping data")

            # Load timeseries data
            timeseries_dir = f'{self.data_dir}/xrd_timeseries'
            if os.path.exists(timeseries_dir):
                for filename in sorted(os.listdir(timeseries_dir)):
                    if filename.startswith('xrd_step_') and filename.endswith('.json'):
                        with open(f'{timeseries_dir}/{filename}', 'r') as f:
                            data = json.load(f)
                            self.timeseries_data.append(data)

                print(f"Loaded {len(self.timeseries_data)} XRD time steps")

        except Exception as e:
            print(f"Warning: Could not load all XRD data: {e}")

    def plot_reference_patterns(self):
        """
        Plot all reference XRD patterns
        """
        if not self.reference_patterns:
            print("No reference patterns available")
            return

        n_patterns = len(self.reference_patterns)
        fig, axes = plt.subplots(n_patterns, 1, figsize=(12, 4*n_patterns))

        if n_patterns == 1:
            axes = [axes]

        for i, (phase_name, data) in enumerate(self.reference_patterns.items()):
            two_theta = np.array(data['two_theta'])
            intensity = np.array(data['intensity'])

            axes[i].plot(two_theta, intensity, 'b-', linewidth=1.5)
            axes[i].set_xlabel('2θ (degrees)')
            axes[i].set_ylabel('Intensity (counts)')
            axes[i].set_title(f'{phase_name} - Reference Pattern')
            axes[i].grid(True, alpha=0.3)

            # Mark major peaks
            peak_indices = self.find_peaks(intensity)
            if len(peak_indices) > 0:
                peak_2theta = two_theta[peak_indices][:10]  # Show first 10 peaks
                peak_intensities = intensity[peak_indices][:10]

                axes[i].scatter(peak_2theta, peak_intensities, color='red', s=30, zorder=5)
                for j, (theta, inten) in enumerate(zip(peak_2theta, peak_intensities)):
                    axes[i].annotate(f'{theta:.1f}°',
                                   (theta, inten),
                                   xytext=(5, 5),
                                   textcoords='offset points',
                                   fontsize=8)

        plt.tight_layout()
        plt.show()

    def find_peaks(self, intensity, prominence=100):
        """Simple peak finding for XRD patterns"""
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensity, prominence=prominence, distance=10)
        return peaks

    def plot_stress_strain_maps(self):
        """
        Plot spatial maps of stress and strain
        """
        if not self.spatial_data:
            print("No spatial data available")
            return

        # Extract strain and stress data for first phase
        phase_names = list(self.spatial_data['xrd_map'].keys())
        phase_name = phase_names[0]

        strain_values = np.array(self.spatial_data['xrd_map'][phase_name]['strain_values'])
        stress_values = np.array(self.spatial_data['xrd_map'][phase_name]['stress_values'])

        grid_size = self.spatial_data['grid_size']
        n_points = grid_size[0] * grid_size[1]

        # Calculate average strain and stress for each point
        avg_strain = np.mean(strain_values, axis=1)  # Average over tensor components
        avg_stress = np.mean(stress_values, axis=1)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Strain map
        strain_map = avg_strain.reshape(grid_size)
        im1 = axes[0].imshow(strain_map, cmap='RdBu_r', origin='lower')
        axes[0].set_title(f'Average Strain - {phase_name}')
        axes[0].set_xlabel('X position')
        axes[0].set_ylabel('Y position')
        plt.colorbar(im1, ax=axes[0], label='Strain')

        # Stress map
        stress_map = avg_stress.reshape(grid_size)
        im2 = axes[1].imshow(stress_map, cmap='viridis', origin='lower')
        axes[1].set_title(f'Average Stress - {phase_name}')
        axes[1].set_xlabel('X position')
        axes[1].set_ylabel('Y position')
        plt.colorbar(im2, ax=axes[1], label='Stress (MPa)')

        plt.tight_layout()
        plt.show()

    def plot_peak_shift_analysis(self):
        """
        Analyze peak shifts over time (strain evolution)
        """
        if not self.timeseries_data:
            print("No timeseries data available")
            return

        # Extract peak positions for first major peak over time
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Get reference peak positions
        if self.reference_patterns:
            ref_phase = list(self.reference_patterns.keys())[0]
            ref_two_theta = np.array(self.reference_patterns[ref_phase]['two_theta'])
            ref_intensity = np.array(self.reference_patterns[ref_phase]['intensity'])
            ref_peaks = self.find_peaks(ref_intensity, prominence=200)

            if len(ref_peaks) > 0:
                reference_peak = ref_two_theta[ref_peaks[0]]
                print(f"Reference peak at {reference_peak:.2f}°")
            else:
                reference_peak = 44.0  # Default value
                print(f"Using default reference peak at {reference_peak:.2f}°")
        else:
            reference_peak = 44.0

        # Plot strain evolution
        times = [data['time'] for data in self.timeseries_data]
        strain_tensors = [data['strain_tensor'] for data in self.timeseries_data]

        # Extract different strain components
        strain_xx = [np.array(tensor)[0,0] for tensor in strain_tensors]
        strain_yy = [np.array(tensor)[1,1] for tensor in strain_tensors]
        strain_zz = [np.array(tensor)[2,2] for tensor in strain_tensors]

        axes[0, 0].plot(times, strain_xx, 'r-o', label='ε_xx', linewidth=2)
        axes[0, 0].plot(times, strain_yy, 'g-s', label='ε_yy', linewidth=2)
        axes[0, 0].plot(times, strain_zz, 'b-^', label='ε_zz', linewidth=2)
        axes[0, 0].set_xlabel('Time (hours)')
        axes[0, 0].set_ylabel('Strain')
        axes[0, 0].set_title('Strain Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot stress evolution
        stress_tensors = [data['stress_tensor'] for data in self.timeseries_data]
        stress_xx = [np.array(tensor)[0,0] for tensor in stress_tensors]
        stress_yy = [np.array(tensor)[1,1] for tensor in stress_tensors]
        stress_zz = [np.array(tensor)[2,2] for tensor in stress_tensors]

        axes[0, 1].plot(times, stress_xx, 'r-o', label='σ_xx', linewidth=2)
        axes[0, 1].plot(times, stress_yy, 'g-s', label='σ_yy', linewidth=2)
        axes[0, 1].plot(times, stress_zz, 'b-^', label='σ_zz', linewidth=2)
        axes[0, 1].set_xlabel('Time (hours)')
        axes[0, 1].set_ylabel('Stress (MPa)')
        axes[0, 1].set_title('Stress Evolution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot peak position evolution (if available)
        if self.timeseries_data and 'xrd_map' in self.timeseries_data[0]:
            peak_positions = []
            for data in self.timeseries_data:
                if data['xrd_map']:
                    phase_name = list(data['xrd_map'].keys())[0]
                    positions = data['xrd_map'][phase_name]['peak_positions']
                    if positions and len(positions) > 0:
                        # Get first peak position for first measurement point
                        first_peak = positions[0][0] if positions[0] else reference_peak
                        peak_positions.append(first_peak)
                    else:
                        peak_positions.append(reference_peak)
                else:
                    peak_positions.append(reference_peak)

            if len(peak_positions) == len(times):
                peak_shifts = np.array(peak_positions) - reference_peak
                axes[1, 0].plot(times, peak_shifts, 'k-o', linewidth=2)
                axes[1, 0].set_xlabel('Time (hours)')
                axes[1, 0].set_ylabel('Peak Shift (degrees)')
                axes[1, 0].set_title('XRD Peak Shift Evolution')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Plot creep curve (strain vs time)
        total_strain = [np.mean([strain_xx[i], strain_yy[i], strain_zz[i]]) for i in range(len(times))]
        axes[1, 1].plot(times, total_strain, 'm-o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Total Strain')
        axes[1, 1].set_title('Creep Curve')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_single_xrd_pattern(self, time_step=0, measurement_point=(0, 0)):
        """
        Plot a single XRD pattern from timeseries data

        Parameters:
        -----------
        time_step : int
            Time step to plot
        measurement_point : tuple
            (x, y) coordinates of measurement point
        """
        if not self.timeseries_data or time_step >= len(self.timeseries_data):
            print("Timeseries data not available or invalid time step")
            return

        data = self.timeseries_data[time_step]

        if not data['xrd_map']:
            print("No XRD map data available")
            return

        # Get phase name and data
        phase_name = list(data['xrd_map'].keys())[0]
        xrd_data = data['xrd_map'][phase_name]

        # Calculate linear index from 2D coordinates
        grid_size = (20, 20)  # Assuming 20x20 grid
        point_index = measurement_point[0] * grid_size[1] + measurement_point[1]

        if point_index >= len(xrd_data['peak_positions']):
            print(f"Invalid measurement point: {measurement_point}")
            return

        # Get peak positions and intensities for this point
        peak_2theta = xrd_data['peak_positions'][point_index]
        peak_intensities = xrd_data['peak_intensities'][point_index]

        # Create synthetic pattern
        two_theta = np.linspace(20, 80, 1000)  # Typical 2theta range
        intensity = np.zeros_like(two_theta)

        # Add peaks as Gaussians
        for theta, inten in zip(peak_2theta, peak_intensities):
            sigma = 0.15  # Peak width
            peak_shape = inten * np.exp(-0.5 * ((two_theta - theta) / sigma)**2)
            intensity += peak_shape

        # Plot pattern
        plt.figure(figsize=(10, 6))
        plt.plot(two_theta, intensity, 'b-', linewidth=1.5)
        plt.xlabel('2θ (degrees)')
        plt.ylabel('Intensity (counts)')
        plt.title(f'XRD Pattern - {phase_name}\nTime: {data["time"]:.1f} h, Point: {measurement_point}')
        plt.grid(True, alpha=0.3)

        # Mark peaks
        plt.scatter(peak_2theta, peak_intensities, color='red', s=40, zorder=5)
        for theta, inten in zip(peak_2theta, peak_intensities):
            plt.annotate(f'{theta:.1f}°', (theta, inten),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.tight_layout()
        plt.show()

def main():
    """Main XRD visualization function"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_xrd.py <data_directory>")
        print("Example: python visualize_xrd.py ../diffraction/")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} not found")
        sys.exit(1)

    print("Synthetic XRD Data Visualizer")
    print("=" * 30)

    visualizer = XRDVisualizer(data_dir)

    # Interactive visualization menu
    while True:
        print("\nXRD Visualization Options:")
        print("1. Show reference patterns")
        print("2. Show stress/strain maps")
        print("3. Show peak shift analysis")
        print("4. Show single XRD pattern")
        print("0. Exit")

        choice = input("\nSelect option (0-4): ").strip()

        if choice == '0':
            print("Goodbye!")
            break
        elif choice == '1':
            visualizer.plot_reference_patterns()
        elif choice == '2':
            visualizer.plot_stress_strain_maps()
        elif choice == '3':
            visualizer.plot_peak_shift_analysis()
        elif choice == '4':
            time_step = int(input("Time step (0-15): ") or "0")
            x = int(input("X coordinate (0-19): ") or "0")
            y = int(input("Y coordinate (0-19): ") or "0")
            visualizer.plot_single_xrd_pattern(time_step, (x, y))
        else:
            print("Invalid choice. Please select 0-4.")

if __name__ == "__main__":
    main()