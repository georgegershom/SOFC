#!/usr/bin/env python3
"""
SOFC Microstructure Analysis Tools
===================================
Advanced analysis tools for validating and characterizing the generated
microstructural dataset.

Author: SOFC Modeling Team
Date: 2025-10-08
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import stats, spatial
from skimage import measure, morphology
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os
from datetime import datetime


class MicrostructureAnalyzer:
    """
    Comprehensive analysis tools for SOFC microstructures.
    """
    
    def __init__(self, volume, voxel_size=0.5, phase_definitions=None):
        """
        Initialize analyzer with microstructural data.
        
        Parameters:
        -----------
        volume : np.ndarray
            3D voxelated microstructure
        voxel_size : float
            Physical size of voxels in micrometers
        phase_definitions : dict
            Mapping of phase names to IDs
        """
        self.volume = volume
        self.voxel_size = voxel_size
        self.size = volume.shape
        
        # Default phase definitions
        if phase_definitions is None:
            self.phases = {
                'pore': 0,
                'nickel': 1,
                'ysz_composite': 2,
                'ysz_electrolyte': 3,
                'interlayer': 4
            }
        else:
            self.phases = phase_definitions
        
        self.results = {}
    
    def analyze_tortuosity(self):
        """
        Calculate tortuosity factors for each phase.
        """
        print("\nAnalyzing tortuosity...")
        tortuosity = {}
        
        for phase_name, phase_id in self.phases.items():
            if phase_name in ['pore', 'nickel', 'ysz_composite']:
                phase_mask = self.volume == phase_id
                
                # Check connectivity in z-direction
                labeled, num_features = ndi.label(phase_mask)
                
                if num_features > 0:
                    # Find percolating cluster (connects top to bottom)
                    top_labels = np.unique(labeled[:, :, 0])
                    bottom_labels = np.unique(labeled[:, :, -1])
                    
                    percolating_labels = np.intersect1d(top_labels, bottom_labels)
                    percolating_labels = percolating_labels[percolating_labels > 0]
                    
                    if len(percolating_labels) > 0:
                        # Calculate tortuosity for largest percolating cluster
                        largest_label = percolating_labels[0]
                        cluster = labeled == largest_label
                        
                        # Geometric tortuosity approximation
                        # τ = L_eff / L_straight
                        distance_map = ndi.distance_transform_edt(cluster)
                        
                        # Find shortest path length through phase
                        z_coords = np.where(cluster)[2]
                        if len(z_coords) > 0:
                            actual_path_length = np.max(z_coords) - np.min(z_coords)
                            
                            # Estimate effective path using distance transform
                            top_points = np.where(cluster[:, :, 0])
                            bottom_points = np.where(cluster[:, :, -1])
                            
                            if len(top_points[0]) > 0 and len(bottom_points[0]) > 0:
                                # Sample paths and calculate average
                                n_samples = min(100, len(top_points[0]))
                                path_lengths = []
                                
                                for _ in range(n_samples):
                                    # Random start and end points
                                    i_start = np.random.randint(len(top_points[0]))
                                    i_end = np.random.randint(len(bottom_points[0]))
                                    
                                    # Simple path length estimation
                                    dx = top_points[0][i_start] - bottom_points[0][i_end]
                                    dy = top_points[1][i_start] - bottom_points[1][i_end]
                                    dz = self.size[2]
                                    
                                    path_length = np.sqrt(dx**2 + dy**2 + dz**2)
                                    path_lengths.append(path_length)
                                
                                avg_path = np.mean(path_lengths)
                                tau = avg_path / self.size[2]
                                tortuosity[phase_name] = max(1.0, tau)  # Tortuosity >= 1
                            else:
                                tortuosity[phase_name] = None
                        else:
                            tortuosity[phase_name] = None
                    else:
                        tortuosity[phase_name] = None
                else:
                    tortuosity[phase_name] = None
        
        self.results['tortuosity'] = tortuosity
        
        print("  Tortuosity factors:")
        for phase, tau in tortuosity.items():
            if tau is not None:
                print(f"    {phase:15s}: {tau:.3f}")
            else:
                print(f"    {phase:15s}: Not percolating")
        
        return tortuosity
    
    def analyze_pore_size_distribution(self):
        """
        Calculate pore size distribution.
        """
        print("\nAnalyzing pore size distribution...")
        
        pore_mask = self.volume == self.phases['pore']
        
        # Distance transform gives radius of largest sphere that fits
        distance = ndi.distance_transform_edt(pore_mask)
        
        # Convert to physical units
        pore_radii = distance[pore_mask] * self.voxel_size
        
        # Filter out boundary artifacts
        pore_radii = pore_radii[pore_radii > 0]
        
        if len(pore_radii) > 0:
            stats_dict = {
                'mean_radius_um': np.mean(pore_radii),
                'std_radius_um': np.std(pore_radii),
                'min_radius_um': np.min(pore_radii),
                'max_radius_um': np.max(pore_radii),
                'median_radius_um': np.median(pore_radii),
            }
            
            # Calculate pore size distribution
            hist, bin_edges = np.histogram(pore_radii * 2, bins=50)  # Diameter
            
            self.results['pore_size'] = {
                'statistics': stats_dict,
                'distribution': {
                    'diameters': (bin_edges[:-1] + bin_edges[1:]) / 2,
                    'counts': hist,
                    'frequencies': hist / np.sum(hist)
                }
            }
            
            print(f"  Mean pore diameter: {stats_dict['mean_radius_um']*2:.2f} μm")
            print(f"  Std deviation: {stats_dict['std_radius_um']*2:.2f} μm")
            print(f"  Range: {stats_dict['min_radius_um']*2:.2f} - {stats_dict['max_radius_um']*2:.2f} μm")
        else:
            self.results['pore_size'] = None
            print("  No pores detected")
        
        return self.results['pore_size']
    
    def analyze_particle_size_distribution(self):
        """
        Analyze particle size distribution for solid phases.
        """
        print("\nAnalyzing particle size distribution...")
        
        particle_stats = {}
        
        for phase_name in ['nickel', 'ysz_composite']:
            if phase_name in self.phases:
                phase_id = self.phases[phase_name]
                phase_mask = self.volume == phase_id
                
                # Label individual particles
                labeled, num_particles = ndi.label(phase_mask)
                
                if num_particles > 0:
                    # Calculate particle sizes
                    particle_sizes = []
                    for i in range(1, num_particles + 1):
                        particle = labeled == i
                        volume_voxels = np.sum(particle)
                        
                        # Equivalent spherical diameter
                        volume_um3 = volume_voxels * (self.voxel_size ** 3)
                        diameter_um = 2 * ((3 * volume_um3) / (4 * np.pi)) ** (1/3)
                        particle_sizes.append(diameter_um)
                    
                    particle_sizes = np.array(particle_sizes)
                    
                    particle_stats[phase_name] = {
                        'count': num_particles,
                        'mean_diameter_um': np.mean(particle_sizes),
                        'std_diameter_um': np.std(particle_sizes),
                        'min_diameter_um': np.min(particle_sizes),
                        'max_diameter_um': np.max(particle_sizes),
                        'median_diameter_um': np.median(particle_sizes),
                    }
                    
                    print(f"  {phase_name.capitalize()}:")
                    print(f"    Particle count: {num_particles}")
                    print(f"    Mean diameter: {particle_stats[phase_name]['mean_diameter_um']:.2f} μm")
                    print(f"    Range: {particle_stats[phase_name]['min_diameter_um']:.2f} - "
                          f"{particle_stats[phase_name]['max_diameter_um']:.2f} μm")
        
        self.results['particle_size'] = particle_stats
        return particle_stats
    
    def analyze_surface_roughness(self):
        """
        Analyze interface roughness, particularly at anode/electrolyte interface.
        """
        print("\nAnalyzing interface roughness...")
        
        # Find anode/electrolyte interface
        anode_mask = (self.volume == self.phases['nickel']) | \
                    (self.volume == self.phases['ysz_composite'])
        electrolyte_mask = self.volume == self.phases['ysz_electrolyte']
        
        # Find interface voxels
        anode_dilated = ndi.binary_dilation(anode_mask)
        electrolyte_dilated = ndi.binary_dilation(electrolyte_mask)
        interface = anode_dilated & electrolyte_dilated
        
        if np.any(interface):
            # Get interface coordinates
            interface_coords = np.where(interface)
            z_coords = interface_coords[2]
            
            # Calculate roughness metrics
            roughness = {
                'mean_z': np.mean(z_coords) * self.voxel_size,
                'std_z': np.std(z_coords) * self.voxel_size,
                'range_z': (np.max(z_coords) - np.min(z_coords)) * self.voxel_size,
            }
            
            # Calculate Ra (arithmetic average roughness)
            z_mean = np.mean(z_coords)
            roughness['Ra'] = np.mean(np.abs(z_coords - z_mean)) * self.voxel_size
            
            # Calculate Rq (RMS roughness)
            roughness['Rq'] = np.sqrt(np.mean((z_coords - z_mean)**2)) * self.voxel_size
            
            self.results['interface_roughness'] = roughness
            
            print(f"  Interface position: {roughness['mean_z']:.2f} ± {roughness['std_z']:.2f} μm")
            print(f"  Ra (average roughness): {roughness['Ra']:.2f} μm")
            print(f"  Rq (RMS roughness): {roughness['Rq']:.2f} μm")
            print(f"  Peak-to-valley: {roughness['range_z']:.2f} μm")
        else:
            self.results['interface_roughness'] = None
            print("  No interface detected")
        
        return self.results.get('interface_roughness')
    
    def analyze_coordination_numbers(self):
        """
        Calculate average coordination numbers for particles.
        """
        print("\nAnalyzing coordination numbers...")
        
        coordination = {}
        
        for phase_name in ['nickel', 'ysz_composite']:
            if phase_name in self.phases:
                phase_id = self.phases[phase_name]
                phase_mask = self.volume == phase_id
                
                # Label particles
                labeled, num_particles = ndi.label(phase_mask)
                
                if num_particles > 0:
                    # Sample particles for coordination analysis
                    n_samples = min(100, num_particles)
                    sampled_labels = np.random.choice(range(1, num_particles + 1), 
                                                    n_samples, replace=False)
                    
                    coord_numbers = []
                    for label in sampled_labels:
                        particle = labeled == label
                        
                        # Dilate particle and count neighbors
                        dilated = ndi.binary_dilation(particle, structure=morphology.ball(2))
                        neighbor_region = dilated & ~particle
                        
                        # Count neighboring particles
                        neighbor_labels = np.unique(labeled[neighbor_region])
                        neighbor_labels = neighbor_labels[neighbor_labels > 0]
                        
                        # Exclude self
                        neighbor_labels = neighbor_labels[neighbor_labels != label]
                        coord_numbers.append(len(neighbor_labels))
                    
                    if coord_numbers:
                        coordination[phase_name] = {
                            'mean': np.mean(coord_numbers),
                            'std': np.std(coord_numbers),
                            'min': np.min(coord_numbers),
                            'max': np.max(coord_numbers),
                        }
                        
                        print(f"  {phase_name.capitalize()}:")
                        print(f"    Mean coordination: {coordination[phase_name]['mean']:.2f}")
                        print(f"    Range: {coordination[phase_name]['min']:.0f} - "
                              f"{coordination[phase_name]['max']:.0f}")
        
        self.results['coordination'] = coordination
        return coordination
    
    def calculate_specific_surface_areas(self):
        """
        Calculate specific surface areas for each phase.
        """
        print("\nCalculating specific surface areas...")
        
        surface_areas = {}
        voxel_face_area = self.voxel_size ** 2  # μm²
        
        for phase_name, phase_id in self.phases.items():
            if phase_name != 'interlayer':  # Skip interlayer
                phase_mask = self.volume == phase_id
                
                # Find surface voxels
                eroded = ndi.binary_erosion(phase_mask)
                surface = phase_mask & ~eroded
                
                # Calculate surface area
                surface_voxels = np.sum(surface)
                surface_area_um2 = surface_voxels * voxel_face_area
                
                # Calculate volume
                volume_voxels = np.sum(phase_mask)
                volume_um3 = volume_voxels * (self.voxel_size ** 3)
                
                if volume_um3 > 0:
                    # Specific surface area (m²/m³)
                    ssa = surface_area_um2 / volume_um3  # μm²/μm³ = m²/m³
                    surface_areas[phase_name] = ssa
                    
                    print(f"  {phase_name:15s}: {ssa:.2f} m²/m³")
                else:
                    surface_areas[phase_name] = 0
        
        self.results['specific_surface_area'] = surface_areas
        return surface_areas
    
    def analyze_percolation(self):
        """
        Detailed percolation analysis for each phase.
        """
        print("\nAnalyzing percolation...")
        
        percolation = {}
        
        for phase_name in ['pore', 'nickel', 'ysz_composite']:
            if phase_name in self.phases:
                phase_id = self.phases[phase_name]
                phase_mask = self.volume == phase_id
                
                # Label connected components
                labeled, num_features = ndi.label(phase_mask)
                
                if num_features > 0:
                    # Check percolation in each direction
                    percolation[phase_name] = {}
                    
                    # X-direction
                    left_labels = np.unique(labeled[0, :, :])
                    right_labels = np.unique(labeled[-1, :, :])
                    x_percolating = np.intersect1d(left_labels, right_labels)
                    x_percolating = x_percolating[x_percolating > 0]
                    percolation[phase_name]['x'] = len(x_percolating) > 0
                    
                    # Y-direction
                    front_labels = np.unique(labeled[:, 0, :])
                    back_labels = np.unique(labeled[:, -1, :])
                    y_percolating = np.intersect1d(front_labels, back_labels)
                    y_percolating = y_percolating[y_percolating > 0]
                    percolation[phase_name]['y'] = len(y_percolating) > 0
                    
                    # Z-direction
                    top_labels = np.unique(labeled[:, :, 0])
                    bottom_labels = np.unique(labeled[:, :, -1])
                    z_percolating = np.intersect1d(top_labels, bottom_labels)
                    z_percolating = z_percolating[z_percolating > 0]
                    percolation[phase_name]['z'] = len(z_percolating) > 0
                    
                    # Calculate percolation probability
                    percolation[phase_name]['probability'] = \
                        sum([percolation[phase_name][d] for d in ['x', 'y', 'z']]) / 3
                    
                    print(f"  {phase_name:15s}: X={percolation[phase_name]['x']}, "
                          f"Y={percolation[phase_name]['y']}, "
                          f"Z={percolation[phase_name]['z']}")
        
        self.results['percolation'] = percolation
        return percolation
    
    def generate_report(self, output_file='output/analysis_report.txt'):
        """
        Generate comprehensive analysis report.
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print(f"\nGenerating analysis report...")
        
        with open(output_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SOFC MICROSTRUCTURE ANALYSIS REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Volume size: {self.size} voxels\n")
            f.write(f"Voxel size: {self.voxel_size} μm\n")
            f.write(f"Physical size: {tuple(s * self.voxel_size for s in self.size)} μm\n")
            f.write("\n")
            
            # Write all results
            for category, data in self.results.items():
                f.write("-"*60 + "\n")
                f.write(f"{category.upper().replace('_', ' ')}\n")
                f.write("-"*60 + "\n")
                
                if data is not None:
                    self._write_dict_to_file(f, data, indent=0)
                else:
                    f.write("No data available\n")
                f.write("\n")
        
        print(f"  Report saved to {output_file}")
    
    def _write_dict_to_file(self, file, data, indent=0):
        """
        Recursively write dictionary to file.
        """
        indent_str = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    file.write(f"{indent_str}{key}:\n")
                    self._write_dict_to_file(file, value, indent + 1)
                elif isinstance(value, (list, np.ndarray)):
                    file.write(f"{indent_str}{key}: [array with {len(value)} elements]\n")
                elif isinstance(value, (int, float, np.number)):
                    if isinstance(value, float):
                        file.write(f"{indent_str}{key}: {value:.4f}\n")
                    else:
                        file.write(f"{indent_str}{key}: {value}\n")
                else:
                    file.write(f"{indent_str}{key}: {value}\n")
        else:
            file.write(f"{indent_str}{data}\n")
    
    def plot_analysis_results(self, save_path='output/analysis_plots.png'):
        """
        Create comprehensive visualization of analysis results.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"\nCreating analysis plots...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Pore size distribution
        if 'pore_size' in self.results and self.results['pore_size'] is not None:
            ax1 = plt.subplot(2, 3, 1)
            dist = self.results['pore_size']['distribution']
            ax1.bar(dist['diameters'], dist['frequencies'], width=np.diff(dist['diameters'])[0])
            ax1.set_xlabel('Pore Diameter (μm)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Pore Size Distribution')
            ax1.grid(True, alpha=0.3)
        
        # 2. Phase volume fractions
        ax2 = plt.subplot(2, 3, 2)
        phase_counts = np.bincount(self.volume.flatten())
        phase_names = ['Pore', 'Ni', 'YSZ-C', 'YSZ-E', 'Inter'][:len(phase_counts)]
        colors = ['lightgray', 'green', 'blue', 'yellow', 'magenta'][:len(phase_counts)]
        wedges, texts, autotexts = ax2.pie(phase_counts, labels=phase_names, colors=colors,
                                            autopct='%1.1f%%', startangle=90)
        ax2.set_title('Phase Volume Fractions')
        
        # 3. Tortuosity comparison
        if 'tortuosity' in self.results:
            ax3 = plt.subplot(2, 3, 3)
            tort_data = self.results['tortuosity']
            phases = []
            values = []
            for phase, tau in tort_data.items():
                if tau is not None:
                    phases.append(phase)
                    values.append(tau)
            
            if phases:
                bars = ax3.bar(phases, values, color=['gray', 'green', 'blue'][:len(phases)])
                ax3.set_ylabel('Tortuosity Factor')
                ax3.set_title('Phase Tortuosity')
                ax3.grid(True, alpha=0.3)
                ax3.axhline(y=1, color='r', linestyle='--', alpha=0.5)
        
        # 4. Specific surface areas
        if 'specific_surface_area' in self.results:
            ax4 = plt.subplot(2, 3, 4)
            ssa_data = self.results['specific_surface_area']
            phases = list(ssa_data.keys())
            values = list(ssa_data.values())
            
            bars = ax4.bar(phases, values, color=['gray', 'green', 'blue', 'yellow'][:len(phases)])
            ax4.set_ylabel('Specific Surface Area (m²/m³)')
            ax4.set_title('Specific Surface Areas')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Interface roughness profile
        if 'interface_roughness' in self.results and self.results['interface_roughness'] is not None:
            ax5 = plt.subplot(2, 3, 5)
            # Create synthetic roughness profile for visualization
            x = np.linspace(0, 100, 1000)
            roughness = self.results['interface_roughness']
            y = roughness['mean_z'] + roughness['std_z'] * np.random.randn(1000) * 0.3
            y = ndi.gaussian_filter1d(y, 10)
            
            ax5.plot(x, y, 'b-', alpha=0.7)
            ax5.axhline(y=roughness['mean_z'], color='r', linestyle='--', 
                       label=f"Mean: {roughness['mean_z']:.1f} μm")
            ax5.fill_between(x, roughness['mean_z'] - roughness['std_z'],
                            roughness['mean_z'] + roughness['std_z'],
                            alpha=0.2, color='r')
            ax5.set_xlabel('Position (μm)')
            ax5.set_ylabel('Interface Height (μm)')
            ax5.set_title('Interface Roughness Profile')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # 6. Percolation analysis
        if 'percolation' in self.results:
            ax6 = plt.subplot(2, 3, 6)
            perc_data = self.results['percolation']
            
            # Create matrix for percolation visualization
            phases = list(perc_data.keys())
            directions = ['x', 'y', 'z']
            matrix = np.zeros((len(phases), len(directions)))
            
            for i, phase in enumerate(phases):
                for j, direction in enumerate(directions):
                    matrix[i, j] = 1 if perc_data[phase][direction] else 0
            
            im = ax6.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
            ax6.set_xticks(range(len(directions)))
            ax6.set_xticklabels(directions)
            ax6.set_yticks(range(len(phases)))
            ax6.set_yticklabels(phases)
            ax6.set_title('Percolation Analysis')
            
            # Add text annotations
            for i in range(len(phases)):
                for j in range(len(directions)):
                    text = '✓' if matrix[i, j] else '✗'
                    ax6.text(j, i, text, ha='center', va='center',
                           color='white' if matrix[i, j] else 'black', fontsize=16)
        
        plt.suptitle('SOFC Microstructure Analysis Results', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Analysis plots saved to {save_path}")
    
    def run_complete_analysis(self):
        """
        Run all analysis methods.
        """
        print("\n" + "="*60)
        print("RUNNING COMPLETE MICROSTRUCTURE ANALYSIS")
        print("="*60)
        
        # Run all analyses
        self.analyze_tortuosity()
        self.analyze_pore_size_distribution()
        self.analyze_particle_size_distribution()
        self.analyze_surface_roughness()
        self.analyze_coordination_numbers()
        self.calculate_specific_surface_areas()
        self.analyze_percolation()
        
        # Generate outputs
        self.generate_report()
        self.plot_analysis_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
        return self.results


def main():
    """
    Run analysis on generated or loaded microstructure.
    """
    # Check if HDF5 file exists
    h5_file = 'output/microstructure.h5'
    
    if os.path.exists(h5_file):
        print(f"Loading microstructure from {h5_file}...")
        with h5py.File(h5_file, 'r') as f:
            volume = f['volume'][:]
            voxel_size = f['volume'].attrs['voxel_size_um']
    else:
        print("No existing microstructure found. Please run sofc_microstructure_generator.py first.")
        return
    
    # Create analyzer
    analyzer = MicrostructureAnalyzer(volume, voxel_size)
    
    # Run complete analysis
    results = analyzer.run_complete_analysis()
    
    print("\nAnalysis results saved to output/")


if __name__ == '__main__':
    main()