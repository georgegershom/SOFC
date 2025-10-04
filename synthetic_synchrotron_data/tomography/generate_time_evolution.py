#!/usr/bin/env python3
"""
Time Evolution Simulator for Creep Deformation in SOFC Materials

This script simulates the microstructural evolution during creep deformation,
including cavitation, crack propagation, and grain boundary sliding.
"""

import numpy as np
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime

class CreepEvolutionSimulator:
    def __init__(self, initial_data_path, temperature=700, stress=50, total_time=100):
        """
        Initialize creep evolution simulator

        Parameters:
        -----------
        initial_data_path : str
            Path to initial tomography data
        temperature : float
            Operating temperature in Celsius
        stress : float
            Applied mechanical stress in MPa
        total_time : float
            Total simulation time in hours
        """
        # Load initial data
        self.grain_map = np.load(f'{initial_data_path}/grain_map.npy')
        self.porosity_map = np.load(f'{initial_data_path}/porosity_map.npy')
        self.defect_map = np.load(f'{initial_data_path}/defect_map.npy')
        self.attenuation_map = np.load(f'{initial_data_path}/attenuation_map.npy')

        self.temperature = temperature
        self.stress = stress
        self.total_time = total_time

        # Evolution parameters (physically motivated)
        self.creep_rate = self.calculate_creep_rate()
        self.cavitation_rate = 0.001  # Pores per hour per unit stress
        self.crack_growth_rate = 0.05  # microns per hour

        # Get sample dimensions
        self.sample_size = self.grain_map.shape

    def calculate_creep_rate(self):
        """
        Calculate creep rate based on temperature and stress
        (simplified Arrhenius-type relationship)
        """
        # Simplified creep rate model
        Q = 200000  # Activation energy (J/mol)
        R = 8.314   # Gas constant
        T = self.temperature + 273.15  # Kelvin

        # Stress exponent (typically 3-5 for metals)
        n = 4

        # Pre-exponential factor (fitted parameter)
        A = 1e-10

        creep_rate = A * (self.stress ** n) * np.exp(-Q / (R * T))

        return creep_rate

    def evolve_porosity(self, current_porosity, time_step):
        """
        Simulate pore nucleation and growth
        """
        # Pore nucleation (stress and temperature dependent)
        nucleation_prob = self.cavitation_rate * time_step * (self.stress / 100)

        # Random nucleation sites
        nucleation_mask = np.random.rand(*self.sample_size) < nucleation_prob

        # Add to existing porosity
        new_porosity = current_porosity | nucleation_mask

        # Pore growth (diffusion-based)
        # Pores tend to grow towards high stress regions
        stress_field = self.generate_stress_field()

        # Growth probability proportional to local stress
        growth_prob = 0.1 * time_step * (stress_field / np.max(stress_field))

        # Grow existing pores
        for i in range(3):  # Multiple iterations for realistic growth
            growth_mask = np.random.rand(*self.sample_size) < growth_prob
            new_porosity = new_porosity | (growth_mask & ndimage.binary_dilation(current_porosity))

        return new_porosity

    def generate_stress_field(self):
        """
        Generate stress concentration field (simplified)
        """
        # Base stress field
        stress_field = np.ones(self.sample_size) * self.stress

        # Add stress concentrations at grain boundaries
        grain_boundaries = self.find_grain_boundaries()
        stress_field[grain_boundaries] *= 1.5  # Stress concentration factor

        # Add stress concentrations at existing defects
        stress_field[self.defect_map] *= 2.0

        # Add some spatial variation
        noise = np.random.rand(*self.sample_size) * 0.2 + 0.9
        stress_field *= noise

        return stress_field

    def find_grain_boundaries(self):
        """
        Identify grain boundary regions
        """
        boundaries = np.zeros_like(self.grain_map, dtype=bool)

        # Simple boundary detection using dilation
        for grain_id in np.unique(self.grain_map):
            grain_mask = self.grain_map == grain_id
            dilated = ndimage.binary_dilation(grain_mask)
            boundary = dilated & ~grain_mask
            boundaries |= boundary

        return boundaries

    def evolve_cracks(self, current_defects, time_step):
        """
        Simulate crack initiation and propagation
        """
        # Crack propagation from existing defects
        new_defects = current_defects.copy()

        # Find crack tips (edges of current defects)
        crack_tips = ndimage.binary_dilation(current_defects) & ~current_defects

        # Propagate cracks along grain boundaries and stress directions
        propagation_prob = self.crack_growth_rate * time_step * (self.stress / 100)

        # Random propagation
        propagation_mask = np.random.rand(*self.sample_size) < propagation_prob

        # Preferential growth along grain boundaries
        boundaries = self.find_grain_boundaries()
        stress_field = self.generate_stress_field()

        # Combined growth criteria
        growth_preference = (boundaries * 0.6 + stress_field / np.max(stress_field) * 0.4)
        growth_mask = propagation_mask & (growth_preference > 0.3)

        # Add new crack segments
        new_defects |= growth_mask & crack_tips

        # Expand existing cracks slightly
        new_defects = ndimage.binary_dilation(new_defects, iterations=1)

        return new_defects

    def simulate_grain_sliding(self, time_step):
        """
        Simulate grain boundary sliding (simplified)
        """
        # This is a simplified model - real grain sliding would require
        # more sophisticated finite element modeling

        # For now, we'll simulate minor grain displacement
        displacement_field = np.random.rand(*self.sample_size, 3) * 0.1 * time_step

        return displacement_field

    def update_attenuation_map(self, porosity_map, defect_map, displacement_field):
        """
        Update attenuation map based on evolved microstructure
        """
        # Start with original attenuation
        new_attenuation = self.attenuation_map.copy()

        # Reduce attenuation in porous regions (cavitation)
        new_attenuation[porosity_map] *= 0.5

        # Very low attenuation in cracked regions
        new_attenuation[defect_map] *= 0.01

        # Add effects of grain displacement (slight density changes)
        # This is a simplified representation
        displacement_magnitude = np.sqrt(np.sum(displacement_field**2, axis=-1))
        new_attenuation += displacement_magnitude * 0.1

        # Add some noise
        new_attenuation += np.random.normal(0, 0.02, self.sample_size)

        return new_attenuation

    def run_simulation(self, n_time_steps=20, output_dir='timeseries'):
        """
        Run complete creep evolution simulation
        """
        print(f"Running creep evolution simulation ({n_time_steps} time steps)...")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Initialize current state
        current_porosity = self.porosity_map.copy()
        current_defects = self.defect_map.copy()

        # Time step size
        time_step = self.total_time / n_time_steps

        # Storage for evolution data
        evolution_data = []

        for step in range(n_time_steps + 1):
            print(f"Time step {step}/{n_time_steps}")

            # Generate displacement field for this step
            displacement_field = self.simulate_grain_sliding(time_step)

            # Update attenuation map
            attenuation_map = self.update_attenuation_map(current_porosity, current_defects, displacement_field)

            # Save current state
            step_data = {
                'time': step * time_step,
                'porosity_map': current_porosity.copy(),
                'defect_map': current_defects.copy(),
                'attenuation_map': attenuation_map.copy(),
                'displacement_field': displacement_field.copy()
            }

            evolution_data.append(step_data)

            # Save to files
            np.save(f'{output_dir}/porosity_step_{step:03d}.npy', current_porosity)
            np.save(f'{output_dir}/defects_step_{step:03d}.npy', current_defects)
            np.save(f'{output_dir}/attenuation_step_{step:03d}.npy', attenuation_map)

            # Evolve for next step (except on last iteration)
            if step < n_time_steps:
                current_porosity = self.evolve_porosity(current_porosity, time_step)
                current_defects = self.evolve_cracks(current_defects, time_step)

        # Save evolution summary
        summary = {
            'total_time': self.total_time,
            'n_steps': n_time_steps,
            'temperature': self.temperature,
            'stress': self.stress,
            'creep_rate': self.creep_rate,
            'final_porosity_fraction': np.mean(evolution_data[-1]['porosity_map']),
            'final_defect_fraction': np.mean(evolution_data[-1]['defect_map'])
        }

        with open(f'{output_dir}/evolution_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Simulation complete! Results saved to {output_dir}")
        print(f"Final porosity fraction: {summary['final_porosity_fraction']:.4f}")
        print(f"Final defect fraction: {summary['final_defect_fraction']:.4f}")

        return evolution_data

def main():
    """Main function to run creep evolution simulation"""
    print("Creep Evolution Simulator for SOFC Materials")
    print("=" * 45)

    # Path to initial data
    initial_data_path = 'synthetic_synchrotron_data/tomography/initial'

    if not os.path.exists(initial_data_path):
        print("Error: Initial data not found. Run generate_tomography.py first.")
        return

    # Initialize simulator
    simulator = CreepEvolutionSimulator(
        initial_data_path=initial_data_path,
        temperature=750,  # Celsius
        stress=60,        # MPa
        total_time=50     # hours
    )

    # Run simulation
    evolution_data = simulator.run_simulation(
        n_time_steps=15,
        output_dir='synthetic_synchrotron_data/tomography/timeseries'
    )

    print("\nSimulation completed successfully!")

if __name__ == "__main__":
    main()