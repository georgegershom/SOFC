#!/usr/bin/env python3
"""
Example script demonstrating how to load and use the Atomic-Scale Simulation Dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def load_dft_data(filename):
    """Load DFT calculation data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

def load_md_trajectory(filename):
    """Load MD trajectory data from XYZ file"""
    frames = []
    current_frame = None

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# Frame'):
                # Save previous frame if exists
                if current_frame:
                    frames.append(current_frame)
                # Start new frame
                current_frame = {'header': line, 'atoms': []}
            elif line and not line.startswith('#') and not line.startswith('Lattice'):
                if current_frame is not None:
                    parts = line.split()
                    if len(parts) >= 4:  # At minimum: element, x, y, z
                        atom = {
                            'element': parts[0],
                            'position': [float(x) for x in parts[1:4]]
                        }
                        # Add forces if available (fx, fy, fz)
                        if len(parts) >= 7:
                            atom['force'] = [float(x) for x in parts[4:7]]
                        current_frame['atoms'].append(atom)

    # Don't forget the last frame
    if current_frame:
        frames.append(current_frame)

    return frames

def analyze_formation_energies(data):
    """Analyze formation energies of defects"""
    print("=== Formation Energies Analysis ===")

    # Vacancy formation energies
    vacancies = data['defects']['vacancies']
    for vac in vacancies:
        print(f"{vac['element']} vacancy in {vac['lattice_site']}: {vac['formation_energy']} eV")

    # Dislocation formation energies
    dislocations = data['defects']['dislocations']
    for disl in dislocations:
        print(f"{disl['type']}: {disl['formation_energy_per_length']} eV/Å")

    # Grain boundary energies
    boundaries = data['defects']['grain_boundaries']
    for gb in boundaries:
        print(f"{gb['type']} GB ({gb['misorientation_angle']}°): {gb['formation_energy']} J/m²")

def analyze_diffusion_barriers(data):
    """Analyze activation energy barriers for diffusion"""
    print("\n=== Diffusion Barriers Analysis ===")

    # Vacancy migration
    vacancy_migration = data['diffusion_processes']['vacancy_migration']
    for vm in vacancy_migration:
        print(f"{vm['host_element']} vacancy migration: {vm['activation_energy']} eV")

    # Grain boundary diffusion
    gb_diffusion = data['diffusion_processes']['grain_boundary_diffusion']
    for gb in gb_diffusion:
        print(f"{gb['boundary_type']} GB diffusion: E_parallel={gb['activation_energy_parallel']} eV, E_perp={gb['activation_energy_perpendicular']} eV")

def analyze_surface_energies(data):
    """Analyze surface energies"""
    print("\n=== Surface Energies Analysis ===")

    # Free surface energies
    surfaces = data['surface_energies']['free_surfaces']
    orientations = [surf['orientation'] for surf in surfaces]
    energies = [surf['surface_energy'] for surf in surfaces]

    plt.figure(figsize=(8, 5))
    plt.bar(orientations, energies)
    plt.xlabel('Surface Orientation')
    plt.ylabel('Surface Energy (J/m²)')
    plt.title('Surface Energies for Different Orientations')
    plt.savefig('surface_energies.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Cavitation energies vs radius
    cavitation = data['surface_energies']['cavitation_surfaces'][0]
    radii = [point['radius'] for point in cavitation['surface_energy_vs_radius']]
    energies = [point['energy'] for point in cavitation['surface_energy_vs_radius']]

    plt.figure(figsize=(8, 5))
    plt.plot(radii, energies, 'bo-')
    plt.xlabel('Void Radius (Å)')
    plt.ylabel('Surface Energy (J/m²)')
    plt.title('Cavitation Energy vs Void Radius')
    plt.grid(True, alpha=0.3)
    plt.savefig('cavitation_energies.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_dislocation_mobility(data):
    """Analyze dislocation mobility data"""
    print("\n=== Dislocation Mobility Analysis ===")

    # Edge dislocation mobility
    edge_data = data['dislocation_types']['edge_dislocation']['mobility_data']
    temperatures = [d['temperature'] for d in edge_data]
    velocities = [d['velocity'] for d in edge_data]

    plt.figure(figsize=(8, 5))
    plt.loglog(temperatures, velocities, 'ro-', label='Edge dislocation')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Dislocation Velocity vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('dislocation_mobility.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main analysis function"""
    print("Atomic-Scale Simulation Dataset Analysis")
    print("=" * 50)

    # Load and analyze DFT data
    formation_data = load_dft_data('dft_calculations/formation_energies_defects.json')
    analyze_formation_energies(formation_data)

    diffusion_data = load_dft_data('dft_calculations/activation_barriers_diffusion.json')
    analyze_diffusion_barriers(diffusion_data)

    surface_data = load_dft_data('dft_calculations/surface_energies.json')
    analyze_surface_energies(surface_data)

    # Load and analyze MD data
    mobility_data = load_dft_data('md_simulations/dislocation_mobility.json')
    analyze_dislocation_mobility(mobility_data)

    # Load trajectory data
    trajectory_frames = load_md_trajectory('md_simulations/grain_boundary_sliding.xyz')
    print(f"\nLoaded {len(trajectory_frames)} frames from MD trajectory")
    print(f"Frame 0 has {len(trajectory_frames[0]['atoms'])} atoms")
    print(f"Final frame time: {trajectory_frames[-1]['atoms'][0]}")

    print("\nAnalysis complete! Generated plots: surface_energies.png, cavitation_energies.png, dislocation_mobility.png")

if __name__ == "__main__":
    main()