#!/usr/bin/env python3
"""
Simplified Time Evolution for Creep Deformation

This script generates time evolution data showing creep deformation,
cavitation, and crack propagation over time.
"""

import numpy as np
import json
import os

def evolve_microstructure(initial_porosity, initial_defects, time_step, total_time):
    """
    Simple evolution of microstructure with time
    """
    # Increase porosity (cavitation) over time
    porosity_increase = 0.001 * time_step  # Linear increase

    # Add new porosity sites randomly
    new_porosity = np.random.rand(*initial_porosity.shape) < porosity_increase

    # Combine with existing porosity
    evolved_porosity = initial_porosity | new_porosity

    # Grow existing pores slightly
    from scipy.ndimage import binary_dilation
    evolved_porosity = binary_dilation(evolved_porosity, iterations=1)

    # Increase defects (crack propagation)
    defect_increase = 0.0005 * time_step

    # Add new defects randomly
    new_defects = np.random.rand(*initial_defects.shape) < defect_increase

    # Combine with existing defects
    evolved_defects = initial_defects | new_defects

    # Grow existing defects slightly
    evolved_defects = binary_dilation(evolved_defects, iterations=1)

    return evolved_porosity, evolved_defects

def generate_time_series(initial_data_dir, n_steps=10, output_dir='timeseries'):
    """
    Generate time evolution series
    """
    print(f"Generating time evolution series ({n_steps} steps)...")

    # Load initial data
    initial_porosity = np.load(f'{initial_data_dir}/porosity_map.npy')
    initial_defects = np.load(f'{initial_data_dir}/defect_map.npy')
    initial_attenuation = np.load(f'{initial_data_dir}/attenuation_map.npy')

    os.makedirs(output_dir, exist_ok=True)

    evolution_data = []

    for step in range(n_steps):
        print(f"Processing time step {step}...")

        # Calculate time
        time_hours = step * 3.125  # 3.125 hours per step for 50 total hours

        if step == 0:
            # Use initial data for step 0
            porosity_map = initial_porosity
            defect_map = initial_defects
        else:
            # Evolve microstructure
            porosity_map, defect_map = evolve_microstructure(
                evolution_data[-1]['porosity_map'],
                evolution_data[-1]['defect_map'],
                time_step=1,
                total_time=time_hours
            )

        # Update attenuation map based on evolved microstructure
        attenuation_map = initial_attenuation.copy()

        # Reduce attenuation in porous regions
        attenuation_map[porosity_map] *= 0.7

        # Very low attenuation in cracked regions
        attenuation_map[defect_map] *= 0.1

        # Add some noise
        attenuation_map += np.random.normal(0, 0.02, attenuation_map.shape)

        # Save data
        step_data = {
            'time_hours': time_hours,
            'porosity_fraction': float(np.mean(porosity_map)),
            'defect_fraction': float(np.mean(defect_map))
        }

        evolution_data.append(step_data)

        # Save arrays
        np.save(f'{output_dir}/porosity_step_{step:03d}.npy', porosity_map)
        np.save(f'{output_dir}/defects_step_{step:03d}.npy', defect_map)
        np.save(f'{output_dir}/attenuation_step_{step:03d}.npy', attenuation_map)

        # Store arrays in step data for evolution
        step_data['porosity_map'] = porosity_map
        step_data['defect_map'] = defect_map
        step_data['attenuation_map'] = attenuation_map

    # Save evolution summary
    summary = {
        'total_time_hours': n_steps * 3.125,
        'n_steps': n_steps,
        'initial_porosity': float(np.mean(initial_porosity)),
        'initial_defects': float(np.mean(initial_defects)),
        'final_porosity': evolution_data[-1]['porosity_fraction'],
        'final_defects': evolution_data[-1]['defect_fraction'],
        'porosity_growth_rate': (evolution_data[-1]['porosity_fraction'] - evolution_data[0]['porosity_fraction']) / (n_steps * 3.125),
        'defect_growth_rate': (evolution_data[-1]['defect_fraction'] - evolution_data[0]['defect_fraction']) / (n_steps * 3.125)
    }

    with open(f'{output_dir}/evolution_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Time evolution saved to {output_dir}")
    print(f"Final porosity: {summary['final_porosity']:.4f}")
    print(f"Final defects: {summary['final_defects']:.4f}")

    return evolution_data

def main():
    """Main function to generate time evolution"""
    print("Simplified Time Evolution Generator for Creep Deformation")
    print("=" * 60)

    # Path to initial data
    initial_data_dir = 'synthetic_synchrotron_data/tomography/initial'

    if not os.path.exists(initial_data_dir):
        print("Error: Initial data not found. Run generate_simple_tomography.py first.")
        return

    # Generate time series
    evolution_data = generate_time_series(
        initial_data_dir,
        n_steps=10,
        output_dir='synthetic_synchrotron_data/tomography/timeseries'
    )

    print("\n" + "=" * 60)
    print("Time Evolution Generation Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()