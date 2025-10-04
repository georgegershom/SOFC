#!/usr/bin/env python3
"""
Simplified Synthetic X-ray Diffraction (XRD) Data Generator

This is a simplified version that generates essential XRD data
without complex computations that may cause memory issues.
"""

import numpy as np
import json
import os

def generate_xrd_pattern(phase_name, two_theta_range=(20, 80), n_points=1000):
    """
    Generate a simple XRD pattern for a given phase
    """
    print(f"Generating XRD pattern for {phase_name}...")

    # Define typical peaks for different phases
    phase_peaks = {
        'Crofer 22 APU': [
            {'hkl': [1, 1, 0], 'intensity': 100, 'position': 44.3},
            {'hkl': [2, 0, 0], 'intensity': 20, 'position': 64.5},
            {'hkl': [2, 1, 1], 'intensity': 30, 'position': 81.7},
        ],
        'Oxide Scale': [
            {'hkl': [1, 1, 1], 'intensity': 80, 'position': 36.8},
            {'hkl': [2, 2, 0], 'intensity': 60, 'position': 54.1},
            {'hkl': [3, 1, 1], 'intensity': 100, 'position': 83.2},
        ],
        'Ni-YSZ': [
            {'hkl': [1, 1, 1], 'intensity': 100, 'position': 44.3},
            {'hkl': [2, 0, 0], 'intensity': 40, 'position': 51.8},
            {'hkl': [2, 2, 0], 'intensity': 20, 'position': 76.4},
        ],
        'YSZ': [
            {'hkl': [1, 1, 1], 'intensity': 30, 'position': 30.2},
            {'hkl': [2, 0, 0], 'intensity': 100, 'position': 35.0},
            {'hkl': [2, 2, 0], 'intensity': 50, 'position': 50.7},
        ]
    }

    peaks = phase_peaks.get(phase_name, phase_peaks['Crofer 22 APU'])

    # Generate 2theta axis
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], n_points)

    # Initialize intensity array
    intensity = np.zeros_like(two_theta)

    # Add peaks as Gaussians
    for peak in peaks:
        position = peak['position']
        peak_intensity = peak['intensity']
        sigma = 0.2  # Peak width

        # Find closest indices
        idx = np.argmin(np.abs(two_theta - position))

        # Add Gaussian peak
        peak_shape = peak_intensity * np.exp(-0.5 * ((two_theta - position) / sigma)**2)
        intensity += peak_shape

    # Add background and noise
    background = np.random.poisson(10, len(intensity))
    noise = np.random.normal(0, 5, len(intensity))

    intensity += background + noise

    return two_theta, intensity, peaks

def generate_spatial_xrd_map(grid_size=(20, 20)):
    """
    Generate spatially resolved XRD data
    """
    print(f"Generating spatial XRD map ({grid_size[0]}x{grid_size[1]} points)...")

    # Create coordinate grid
    x_coords = np.linspace(0, 1, grid_size[0])
    y_coords = np.linspace(0, 1, grid_size[1])
    X, Y = np.meshgrid(x_coords, y_coords)

    # Generate strain and stress fields
    strain_field = np.zeros((grid_size[0], grid_size[1], 3, 3))
    stress_field = np.zeros((grid_size[0], grid_size[1], 3, 3))

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            # Add some spatial variation
            strain_field[i, j] = 0.002 * np.eye(3) + 0.001 * np.random.randn(3, 3)
            stress_field[i, j] = 50 * np.eye(3) + 10 * np.random.randn(3, 3)

    # Generate XRD data for each point
    phases = ['Crofer 22 APU', 'Oxide Scale']
    xrd_map = {}

    for phase_name in phases:
        print(f"Processing phase: {phase_name}")

        phase_data = {
            'two_theta': None,
            'reference_intensity': None,
            'peak_positions': [],
            'peak_intensities': [],
            'strain_values': [],
            'stress_values': []
        }

        # Generate reference pattern
        two_theta, ref_intensity, peaks = generate_xrd_pattern(phase_name)
        phase_data['two_theta'] = two_theta.tolist()
        phase_data['reference_intensity'] = ref_intensity.tolist()

        # Generate patterns for each spatial point
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                # Get local strain and stress
                local_strain = strain_field[i, j]
                local_stress = stress_field[i, j]

                # Apply strain/stress effects (simplified)
                shifted_intensity = ref_intensity.copy()

                # Simple strain effect: shift peaks
                strain_shift = np.trace(local_strain) / 3 * 0.1  # Convert to peak shift
                shifted_intensity = np.roll(shifted_intensity, int(strain_shift * 10))

                # Find peaks in shifted pattern
                from scipy.signal import find_peaks
                peak_indices, _ = find_peaks(shifted_intensity, prominence=50, distance=10)

                if len(peak_indices) > 0:
                    peak_2theta = two_theta[peak_indices][:5]  # First 5 peaks
                    peak_intensities = shifted_intensity[peak_indices][:5]
                else:
                    peak_2theta = [two_theta[len(two_theta)//2]]  # Default peak
                    peak_intensities = [np.max(shifted_intensity)]

                phase_data['peak_positions'].append(peak_2theta.tolist())
                phase_data['peak_intensities'].append(peak_intensities.tolist())
                phase_data['strain_values'].append(local_strain.flatten().tolist())
                phase_data['stress_values'].append(local_stress.flatten().tolist())

        xrd_map[phase_name] = phase_data

    return xrd_map, strain_field, stress_field

def save_xrd_data(data_dict, filename):
    """Save XRD data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"Saved XRD data to {filename}")

def main():
    """Main function to generate simplified XRD data"""
    print("Simplified Synthetic X-ray Diffraction Generator")
    print("=" * 50)

    output_dir = 'synthetic_synchrotron_data/diffraction'
    os.makedirs(output_dir, exist_ok=True)

    # Generate reference patterns for all phases
    print("\nGenerating reference XRD patterns...")

    phases = ['Crofer 22 APU', 'Oxide Scale', 'Ni-YSZ', 'YSZ']

    for phase_name in phases:
        two_theta, intensity, peaks = generate_xrd_pattern(phase_name)

        pattern_data = {
            'phase': phase_name,
            'two_theta': two_theta.tolist(),
            'intensity': intensity.tolist(),
            'peaks': peaks,
            'wavelength': 1.24,
            'material_type': 'interconnect' if 'Crofer' in phase_name or 'Oxide' in phase_name else 'anode'
        }

        filename = f'{output_dir}/{phase_name.lower().replace(" ", "_").replace("-", "_")}_reference.json'
        save_xrd_data(pattern_data, filename)

    # Generate spatial mapping
    print("\nGenerating spatial XRD mapping...")
    xrd_map, strain_field, stress_field = generate_spatial_xrd_map(grid_size=(20, 20))

    spatial_data = {
        'xrd_map': xrd_map,
        'strain_field': strain_field.tolist(),
        'stress_field': stress_field.tolist(),
        'grid_size': [20, 20]
    }

    save_xrd_data(spatial_data, f'{output_dir}/spatial_xrd_mapping.json')

    print("\n" + "=" * 50)
    print("XRD Data Generation Complete!")
    print("=" * 50)
    print(f"Reference patterns: {len(phases)} phases")
    print(f"Spatial mapping: {20}x{20} grid")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main()