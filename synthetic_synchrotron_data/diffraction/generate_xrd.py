#!/usr/bin/env python3
"""
Synthetic X-ray Diffraction (XRD) Data Generator for SOFC Materials

This script generates realistic synthetic XRD patterns for phase identification,
residual stress mapping, and strain analysis in SOFC materials.
"""

import numpy as np
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt

class SyntheticXRDGenerator:
    def __init__(self, material_type='interconnect', wavelength=0.124, detector_size=2048):
        """
        Initialize XRD generator

        Parameters:
        -----------
        material_type : str
            'interconnect' for metallic interconnect or 'anode' for anode support
        wavelength : float
            X-ray wavelength in Angstroms (Cu K-alpha = 1.54, synchrotron often 0.1-1.0)
        detector_size : int
            Number of pixels in detector (affects 2theta range)
        """
        self.material_type = material_type
        self.wavelength = wavelength  # Angstroms
        self.detector_size = detector_size

        # Define typical phases for each material type
        self.phases = {
            'interconnect': {
                'Crofer 22 APU': {
                    'lattice_parameter': 3.58,  # Angstroms (BCC structure)
                    'peaks': [
                        {'hkl': [1, 1, 0], 'intensity': 100, 'expected_2theta': None},
                        {'hkl': [2, 0, 0], 'intensity': 20, 'expected_2theta': None},
                        {'hkl': [2, 1, 1], 'intensity': 30, 'expected_2theta': None},
                        {'hkl': [2, 2, 0], 'intensity': 10, 'expected_2theta': None},
                        {'hkl': [3, 1, 0], 'intensity': 15, 'expected_2theta': None},
                    ],
                    'composition': 'Fe-22Cr-0.5Mn-0.3Ti (wt%)'
                },
                'Oxide Scale': {
                    'lattice_parameter': 4.18,  # Angstroms (spinel structure)
                    'peaks': [
                        {'hkl': [1, 1, 1], 'intensity': 80, 'expected_2theta': None},
                        {'hkl': [2, 2, 0], 'intensity': 60, 'expected_2theta': None},
                        {'hkl': [3, 1, 1], 'intensity': 100, 'expected_2theta': None},
                        {'hkl': [2, 2, 2], 'intensity': 20, 'expected_2theta': None},
                        {'hkl': [4, 0, 0], 'intensity': 40, 'expected_2theta': None},
                    ],
                    'composition': 'Cr2O3 + (Mn,Cr)3O4 spinel'
                }
            },
            'anode': {
                'Ni-YSZ': {
                    'lattice_parameter': 5.14,  # Angstroms (Ni FCC)
                    'peaks': [
                        {'hkl': [1, 1, 1], 'intensity': 100, 'expected_2theta': None},
                        {'hkl': [2, 0, 0], 'intensity': 40, 'expected_2theta': None},
                        {'hkl': [2, 2, 0], 'intensity': 20, 'expected_2theta': None},
                        {'hkl': [3, 1, 1], 'intensity': 25, 'expected_2theta': None},
                        {'hkl': [2, 2, 2], 'intensity': 10, 'expected_2theta': None},
                    ],
                    'composition': 'Ni-8YSZ cermet'
                },
                'YSZ': {
                    'lattice_parameter': 5.14,  # Angstroms (fluorite structure)
                    'peaks': [
                        {'hkl': [1, 1, 1], 'intensity': 30, 'expected_2theta': None},
                        {'hkl': [2, 0, 0], 'intensity': 100, 'expected_2theta': None},
                        {'hkl': [2, 2, 0], 'intensity': 50, 'expected_2theta': None},
                        {'hkl': [3, 1, 1], 'intensity': 80, 'expected_2theta': None},
                        {'hkl': [2, 2, 2], 'intensity': 20, 'expected_2theta': None},
                    ],
                    'composition': '8 mol% Y2O3-ZrO2'
                }
            }
        }

        # Calculate expected 2theta angles for all peaks
        self.calculate_2theta_angles()

    def calculate_2theta_angles(self):
        """Calculate expected 2theta angles using Bragg's law"""
        for material in self.phases.values():
            for phase_name, phase_data in material.items():
                for peak in phase_data['peaks']:
                    # Bragg's law: nλ = 2d sinθ
                    # For cubic structures: d = a / sqrt(h² + k² + l²)
                    h, k, l = peak['hkl']
                    a = phase_data['lattice_parameter']

                    d_hkl = a / np.sqrt(h**2 + k**2 + l**2)
                    theta = np.arcsin(self.wavelength / (2 * d_hkl))
                    two_theta = 2 * np.degrees(theta)

                    peak['expected_2theta'] = two_theta

    def generate_ideal_pattern(self, phase_name, scale_factor=1000):
        """
        Generate ideal XRD pattern for a given phase

        Parameters:
        -----------
        phase_name : str
            Name of the phase to simulate
        scale_factor : float
            Overall intensity scaling
        """
        # Get 2theta range
        min_2theta = min(peak['expected_2theta'] for phase in self.phases[self.material_type].values()
                        for peak in phase['peaks']) - 5
        max_2theta = max(peak['expected_2theta'] for phase in self.phases[self.material_type].values()
                        for peak in phase['peaks']) + 5

        two_theta = np.linspace(min_2theta, max_2theta, self.detector_size)
        intensity = np.zeros_like(two_theta)

        # Add peaks for the specified phase
        phase_data = self.phases[self.material_type].get(phase_name)

        if phase_data is None:
            raise ValueError(f"Phase {phase_name} not found")

        # Add diffraction peaks
        for peak in phase_data['peaks']:
            expected_2theta = peak['expected_2theta']
            peak_intensity = peak['intensity'] * scale_factor

            # Find closest index
            idx = np.argmin(np.abs(two_theta - expected_2theta))

            # Add Gaussian peak
            sigma = 0.1  # Peak width in degrees
            peak_shape = peak_intensity * np.exp(-0.5 * ((two_theta - expected_2theta) / sigma)**2)

            intensity += peak_shape

        # Add background noise
        background = np.random.poisson(10, len(intensity))
        intensity += background

        return two_theta, intensity

    def add_stress_strain_effects(self, two_theta, intensity, strain_tensor, stress_tensor):
        """
        Add effects of residual stress and strain to XRD pattern

        Parameters:
        -----------
        two_theta : array
            2theta values
        intensity : array
            Intensity values
        strain_tensor : array
            3x3 strain tensor
        stress_tensor : array
            3x3 stress tensor
        """
        # Calculate average strain and stress
        avg_strain = np.trace(strain_tensor) / 3
        avg_stress = np.trace(stress_tensor) / 3

        # Peak shifting due to strain (simplified - assumes isotropic strain)
        strain_shift = avg_strain * 180 / np.pi  # Convert to degrees

        # Peak broadening due to stress/strain
        broadening_factor = 1 + 0.1 * (avg_stress / 100)  # Stress in MPa

        # Apply effects
        shifted_intensity = np.zeros_like(intensity)

        for i, (theta, inten) in enumerate(zip(two_theta, intensity)):
            # Shift peak position
            shifted_theta = theta + strain_shift

            # Find original index
            original_idx = np.argmin(np.abs(two_theta - shifted_theta))

            if 0 <= original_idx < len(shifted_intensity):
                # Broaden peak
                sigma = 0.1 * broadening_factor
                broadened_peak = inten * np.exp(-0.5 * ((two_theta - two_theta[original_idx]) / sigma)**2)

                # Add to shifted intensity
                shift_idx = min(max(i, 0), len(shifted_intensity) - 1)
                shifted_intensity[shift_idx] += broadened_peak[shift_idx]

        return two_theta, shifted_intensity

    def generate_spatial_xrd_map(self, grid_size=(50, 50), strain_field=None, stress_field=None):
        """
        Generate spatially resolved XRD data (stress/strain mapping)

        Parameters:
        -----------
        grid_size : tuple
            Number of measurement points in x,y
        strain_field : array, optional
            3D strain field (if None, generated randomly)
        stress_field : array, optional
            3D stress field (if None, generated randomly)
        """
        # Generate coordinate grid
        x_coords = np.linspace(0, 1, grid_size[0])
        y_coords = np.linspace(0, 1, grid_size[1])
        X, Y = np.meshgrid(x_coords, y_coords)

        # Generate strain and stress fields if not provided
        if strain_field is None:
            # Create realistic strain distribution
            strain_field = np.zeros((grid_size[0], grid_size[1], 3, 3))
            for i in range(3):
                for j in range(3):
                    strain_field[:, :, i, j] = 0.001 * np.random.randn(*grid_size) + \
                                              0.002 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)

        if stress_field is None:
            # Create realistic stress distribution
            stress_field = np.zeros((grid_size[0], grid_size[1], 3, 3))
            for i in range(3):
                for j in range(3):
                    stress_field[:, :, i, j] = 20 * np.random.randn(*grid_size) + \
                                               30 * np.sin(np.pi * X) * np.sin(np.pi * Y)

        # Generate XRD pattern for each location
        xrd_map = {}

        for phase_name in self.phases[self.material_type].keys():
            print(f"Generating XRD map for phase: {phase_name}")

            xrd_data = {
                'two_theta': None,
                'peak_positions': [],
                'peak_intensities': [],
                'strain_values': [],
                'stress_values': []
            }

            # Generate reference pattern (unstrained)
            two_theta, ref_intensity = self.generate_ideal_pattern(phase_name)

            xrd_data['two_theta'] = two_theta
            xrd_data['reference_intensity'] = ref_intensity

            # Generate spatially varying patterns
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    # Get local strain and stress
                    local_strain = strain_field[i, j]
                    local_stress = stress_field[i, j]

                    # Generate pattern with strain/stress effects
                    _, strained_intensity = self.add_stress_strain_effects(
                        two_theta, ref_intensity, local_strain, local_stress
                    )

                    # Find peak positions and intensities
                    peak_indices = self.find_peaks(strained_intensity)
                    peak_2theta = two_theta[peak_indices]
                    peak_intensities = strained_intensity[peak_indices]

                    xrd_data['peak_positions'].append(peak_2theta.tolist())
                    xrd_data['peak_intensities'].append(peak_intensities.tolist())
                    xrd_data['strain_values'].append(local_strain.flatten().tolist())
                    xrd_data['stress_values'].append(local_stress.flatten().tolist())

            xrd_map[phase_name] = xrd_data

        return xrd_map, strain_field, stress_field

    def find_peaks(self, intensity, prominence=100):
        """
        Simple peak finding algorithm for XRD patterns
        """
        # Find local maxima
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(intensity, prominence=prominence, distance=10)

        return peaks

    def generate_time_evolution_xrd(self, evolution_data, output_dir='xrd_timeseries'):
        """
        Generate XRD data for each time step in the evolution

        Parameters:
        -----------
        evolution_data : list
            List of evolution data from creep simulation
        output_dir : str
            Output directory for XRD time series
        """
        os.makedirs(output_dir, exist_ok=True)

        xrd_time_series = []

        for step, data in enumerate(evolution_data):
            print(f"Generating XRD data for time step {step}")

            # Generate increasing strain/stress with time
            time_factor = data['time'] / evolution_data[-1]['time']

            # Base strain increases with time
            base_strain = 0.002 * time_factor
            strain_variation = 0.001 * np.random.randn(3, 3)

            strain_tensor = np.eye(3) * base_strain + strain_variation

            # Base stress increases with time
            base_stress = 50 * time_factor  # MPa
            stress_variation = 10 * np.random.randn(3, 3)

            stress_tensor = np.eye(3) * base_stress + stress_variation

            # Generate XRD map for this time step
            xrd_map, _, _ = self.generate_spatial_xrd_map(
                grid_size=(20, 20),
                strain_field=np.array([strain_tensor] * 20 * 20).reshape(20, 20, 3, 3),
                stress_field=np.array([stress_tensor] * 20 * 20).reshape(20, 20, 3, 3)
            )

            # Save XRD data
            step_data = {
                'time': data['time'],
                'strain_tensor': strain_tensor.tolist(),
                'stress_tensor': stress_tensor.tolist(),
                'xrd_map': xrd_map
            }

            xrd_time_series.append(step_data)

            # Save to file
            filename = f'{output_dir}/xrd_step_{step:03d}.json'
            with open(filename, 'w') as f:
                json.dump(step_data, f, indent=2)

        print(f"XRD time series saved to {output_dir}")
        return xrd_time_series

def main():
    """Main function to generate XRD data"""
    print("Synthetic X-ray Diffraction Generator for SOFC Materials")
    print("=" * 55)

    # Generate XRD data for interconnect material
    generator = SyntheticXRDGenerator(
        material_type='interconnect',
        wavelength=0.124,  # Typical synchrotron energy
        detector_size=1024
    )

    # Generate reference patterns for all phases
    output_dir = 'synthetic_synchrotron_data/diffraction'
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating reference XRD patterns...")
    for phase_name in generator.phases['interconnect'].keys():
        two_theta, intensity = generator.generate_ideal_pattern(phase_name)

        # Save reference pattern
        pattern_data = {
            'phase': phase_name,
            'two_theta': two_theta.tolist(),
            'intensity': intensity.tolist(),
            'wavelength': generator.wavelength,
            'material_type': generator.material_type
        }

        filename = f'{output_dir}/{phase_name.lower().replace(" ", "_")}_reference.json'
        with open(filename, 'w') as f:
            json.dump(pattern_data, f, indent=2)

        print(f"Saved reference pattern: {filename}")

    # Generate spatial XRD mapping
    print("\nGenerating spatial XRD mapping...")
    xrd_map, strain_field, stress_field = generator.generate_spatial_xrd_map(grid_size=(30, 30))

    # Save spatial data
    spatial_data = {
        'xrd_map': xrd_map,
        'strain_field': strain_field.tolist(),
        'stress_field': stress_field.tolist(),
        'grid_size': [30, 30]
    }

    with open(f'{output_dir}/spatial_xrd_mapping.json', 'w') as f:
        json.dump(spatial_data, f, indent=2)

    print(f"Spatial XRD mapping saved to {output_dir}/spatial_xrd_mapping.json")

    print("\nXRD data generation complete!")

if __name__ == "__main__":
    main()