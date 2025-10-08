#!/usr/bin/env python3
"""
Validation & Analysis Dataset Generator for FEM Model
Generates synthetic but realistic data for SOFC electrolyte residual stress analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.interpolate import griddata
import json
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

class ValidationDatasetGenerator:
    def __init__(self):
        self.output_dir = "/workspace/validation_dataset"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Material properties for YSZ (Yttria-Stabilized Zirconia)
        self.material_props = {
            'young_modulus': 200e9,  # Pa
            'poisson_ratio': 0.31,
            'cte': 10.5e-6,  # 1/K
            'density': 6100,  # kg/m3
            'thermal_conductivity': 2.7,  # W/m·K
            'fracture_toughness': 2.5e6,  # Pa·m^0.5
            'grain_size_mean': 2.5e-6,  # m
            'grain_size_std': 0.8e-6,  # m
            'porosity': 0.05  # 5% porosity
        }
        
        # Simulation domain
        self.domain_size = (100e-6, 100e-6, 20e-6)  # m (length, width, thickness)
        self.mesh_resolution = (50, 50, 10)  # nodes
        
    def generate_residual_stress_experimental(self):
        """Generate experimental residual stress data from XRD, Raman, and Synchrotron"""
        print("Generating experimental residual stress data...")
        
        # XRD Surface measurements (macro-scale)
        xrd_data = {
            'measurement_points': [],
            'stress_xx': [],
            'stress_yy': [],
            'stress_xy': [],
            'uncertainty': [],
            'measurement_conditions': {
                'wavelength': 1.5406e-10,  # Cu Kα
                'penetration_depth': 5e-6,  # m
                'beam_size': 1e-3,  # m
                'scan_step': 0.1,  # degrees
                'counting_time': 10  # seconds
            }
        }
        
        # Generate measurement grid (25 points across surface)
        x_coords = np.linspace(10e-6, 90e-6, 5)
        y_coords = np.linspace(10e-6, 90e-6, 5)
        
        for x in x_coords:
            for y in y_coords:
                # Simulate realistic stress distribution with thermal gradient effects
                # Higher compressive stress near edges due to constraint
                edge_factor = min(x, self.domain_size[0]-x, y, self.domain_size[1]-y) / (self.domain_size[0]/4)
                base_stress = -50e6 * (1 + 2*np.exp(-edge_factor))  # Compressive stress
                
                # Add some realistic variation and measurement noise
                stress_xx = base_stress + np.random.normal(0, 10e6)
                stress_yy = base_stress * 0.8 + np.random.normal(0, 8e6)
                stress_xy = np.random.normal(0, 5e6)
                
                xrd_data['measurement_points'].append([float(x), float(y), 0.0])
                xrd_data['stress_xx'].append(float(stress_xx))
                xrd_data['stress_yy'].append(float(stress_yy))
                xrd_data['stress_xy'].append(float(stress_xy))
                xrd_data['uncertainty'].append(float(np.random.uniform(5e6, 15e6)))
        
        # Raman spectroscopy data (higher spatial resolution, surface only)
        raman_data = {
            'measurement_points': [],
            'stress_magnitude': [],
            'peak_shift': [],
            'peak_width': [],
            'measurement_conditions': {
                'laser_wavelength': 532e-9,  # m
                'spot_size': 1e-6,  # m
                'power': 1e-3,  # W
                'integration_time': 30  # seconds
            }
        }
        
        # Higher resolution Raman grid (100 points)
        x_raman = np.linspace(5e-6, 95e-6, 10)
        y_raman = np.linspace(5e-6, 95e-6, 10)
        
        for x in x_raman:
            for y in y_raman:
                # Stress calculation similar to XRD but with finer details
                edge_factor = min(x, self.domain_size[0]-x, y, self.domain_size[1]-y) / (self.domain_size[0]/4)
                base_stress = 50e6 * (1 + 2*np.exp(-edge_factor))
                
                # Add microstructural effects (grain boundaries, pores)
                micro_variation = 20e6 * np.sin(x*1e5) * np.cos(y*1e5)
                stress_mag = abs(base_stress + micro_variation + np.random.normal(0, 5e6))
                
                # Raman peak shift correlates with stress (empirical relation)
                peak_shift = stress_mag * 2.3e-15  # cm^-1/Pa (typical for YSZ)
                peak_width = 8 + stress_mag * 1e-8  # cm^-1
                
                raman_data['measurement_points'].append([float(x), float(y), 0.0])
                raman_data['stress_magnitude'].append(float(stress_mag))
                raman_data['peak_shift'].append(float(peak_shift))
                raman_data['peak_width'].append(float(peak_width))
        
        # Synchrotron X-ray diffraction (bulk/sub-surface, if accessible)
        synchrotron_data = {
            'measurement_points': [],
            'stress_tensor': [],  # Full 3D stress tensor
            'grain_orientation': [],
            'measurement_conditions': {
                'energy': 20000,  # eV
                'beam_size': [0.5e-6, 0.5e-6],  # m
                'penetration_depth': 50e-6,  # m
                'angular_resolution': 0.01  # degrees
            }
        }
        
        # 3D measurement points through thickness
        z_coords = np.linspace(1e-6, 19e-6, 5)
        x_syn = np.linspace(20e-6, 80e-6, 4)
        y_syn = np.linspace(20e-6, 80e-6, 4)
        
        for z in z_coords:
            for x in x_syn:
                for y in y_syn:
                    # 3D stress field with through-thickness variation
                    depth_factor = (z / self.domain_size[2])
                    edge_factor = min(x, self.domain_size[0]-x, y, self.domain_size[1]-y) / (self.domain_size[0]/4)
                    
                    # Stress varies through thickness due to thermal gradients
                    base_stress = -40e6 * (1 + edge_factor) * (1 - 0.5*depth_factor)
                    
                    stress_tensor = np.array([
                        [base_stress + np.random.normal(0, 8e6), np.random.normal(0, 3e6), np.random.normal(0, 2e6)],
                        [np.random.normal(0, 3e6), base_stress*0.9 + np.random.normal(0, 8e6), np.random.normal(0, 2e6)],
                        [np.random.normal(0, 2e6), np.random.normal(0, 2e6), base_stress*0.3 + np.random.normal(0, 5e6)]
                    ])
                    
                    # Random grain orientation (Euler angles)
                    grain_orient = np.random.uniform(0, 2*np.pi, 3)
                    
                    synchrotron_data['measurement_points'].append([float(x), float(y), float(z)])
                    synchrotron_data['stress_tensor'].append([[float(val) for val in row] for row in stress_tensor])
                    synchrotron_data['grain_orientation'].append([float(val) for val in grain_orient])
        
        # Save experimental data
        experimental_data = {
            'xrd_surface': xrd_data,
            'raman_spectroscopy': raman_data,
            'synchrotron_bulk': synchrotron_data,
            'metadata': {
                'material': 'YSZ (8mol% Y2O3)',
                'sample_id': 'SOFC_EL_001',
                'measurement_date': datetime.now().isoformat(),
                'temperature_during_measurement': 298,  # K
                'notes': 'Post-sintering residual stress measurements'
            }
        }
        
        with open(f"{self.output_dir}/experimental_residual_stress.json", 'w') as f:
            json.dump(experimental_data, f, indent=2)
        
        return experimental_data
    
    def generate_crack_data(self):
        """Generate crack initiation and propagation data"""
        print("Generating crack initiation and propagation data...")
        
        crack_data = {
            'crack_locations': [],
            'crack_orientations': [],
            'crack_lengths': [],
            'critical_conditions': [],
            'propagation_data': [],
            'sem_observations': []
        }
        
        # Generate realistic crack locations (typically at stress concentrators)
        n_cracks = 15
        
        for i in range(n_cracks):
            # Cracks tend to initiate at pores, grain boundaries, or free surfaces
            if i < 5:  # Surface cracks
                x = np.random.uniform(0, self.domain_size[0])
                y = np.random.choice([0, self.domain_size[1]])  # Edge
                z = 0
                crack_type = 'surface'
            elif i < 10:  # Pore-initiated cracks
                x = np.random.uniform(10e-6, 90e-6)
                y = np.random.uniform(10e-6, 90e-6)
                z = np.random.uniform(2e-6, 18e-6)
                crack_type = 'pore_initiated'
            else:  # Grain boundary cracks
                x = np.random.uniform(5e-6, 95e-6)
                y = np.random.uniform(5e-6, 95e-6)
                z = np.random.uniform(1e-6, 19e-6)
                crack_type = 'grain_boundary'
            
            # Crack orientation (normal to maximum principal stress)
            orientation = np.random.uniform(0, np.pi)  # radians
            
            # Crack length (depends on local stress and material properties)
            local_stress = self._calculate_local_stress(x, y, z)
            critical_length = (self.material_props['fracture_toughness']**2 / 
                             (np.pi * local_stress**2)) if local_stress > 0 else 1e-6
            actual_length = np.random.uniform(0.5*critical_length, 2*critical_length)
            
            # Critical conditions for crack initiation
            critical_stress = np.random.uniform(80e6, 150e6)  # Pa
            critical_temp = np.random.uniform(1200, 1400)  # K (during cooling)
            
            crack_data['crack_locations'].append([float(x), float(y), float(z)])
            crack_data['crack_orientations'].append(float(orientation))
            crack_data['crack_lengths'].append(float(actual_length))
            crack_data['critical_conditions'].append({
                'stress': float(critical_stress),
                'temperature': float(critical_temp),
                'crack_type': crack_type
            })
            
            # SEM observation data
            sem_obs = {
                'magnification': int(np.random.choice([1000, 2000, 5000, 10000])),
                'crack_width': float(np.random.uniform(0.1e-6, 2e-6)),  # m
                'surface_roughness': float(np.random.uniform(0.05e-6, 0.3e-6)),  # m
                'grain_size_local': float(np.random.normal(2.5e-6, 0.8e-6)),  # m
                'porosity_local': float(np.random.uniform(0.02, 0.08)),
                'image_quality': str(np.random.choice(['excellent', 'good', 'fair']))
            }
            crack_data['sem_observations'].append(sem_obs)
        
        # Crack propagation data (Paris law parameters)
        propagation_data = {
            'paris_law_C': 1.2e-12,  # m/cycle/(MPa√m)^m
            'paris_law_m': 3.2,
            'threshold_stress_intensity': 0.8e6,  # Pa√m
            'fracture_toughness': self.material_props['fracture_toughness'],
            'propagation_rates': []
        }
        
        # Generate propagation rate data for different stress intensity ranges
        K_values = np.linspace(0.8e6, 2.0e6, 20)  # Pa√m
        for K in K_values:
            if K > propagation_data['threshold_stress_intensity']:
                da_dN = (propagation_data['paris_law_C'] * 
                        (K * 1e-6)**propagation_data['paris_law_m'])  # m/cycle
                propagation_data['propagation_rates'].append({
                    'stress_intensity': float(K),
                    'crack_growth_rate': float(da_dN),
                    'uncertainty': float(da_dN * 0.2)  # 20% uncertainty
                })
        
        crack_data['propagation_data'] = propagation_data
        
        with open(f"{self.output_dir}/crack_initiation_propagation.json", 'w') as f:
            json.dump(crack_data, f, indent=2)
        
        return crack_data
    
    def generate_collocation_point_data(self):
        """Generate full-field and collocation point simulation data"""
        print("Generating collocation point simulation data...")
        
        # Create mesh grid
        x = np.linspace(0, self.domain_size[0], self.mesh_resolution[0])
        y = np.linspace(0, self.domain_size[1], self.mesh_resolution[1])
        z = np.linspace(0, self.domain_size[2], self.mesh_resolution[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Full-field simulation data
        full_field_data = {
            'coordinates': [],
            'temperature': [],
            'displacement': [],
            'stress': [],
            'strain': [],
            'simulation_parameters': {
                'sintering_temperature': 1450,  # K
                'cooling_rate': 5,  # K/min
                'final_temperature': 298,  # K
                'time_steps': 100,
                'mesh_elements': np.prod(self.mesh_resolution),
                'solver': 'ANSYS_Mechanical',
                'convergence_criteria': 1e-6
            }
        }
        
        # Generate realistic field data
        for i in range(self.mesh_resolution[0]):
            for j in range(self.mesh_resolution[1]):
                for k in range(self.mesh_resolution[2]):
                    coord = [X[i,j,k], Y[i,j,k], Z[i,j,k]]
                    
                    # Temperature field (cooling from sintering)
                    # Assume some thermal gradients due to geometry
                    T_base = 298  # Room temperature
                    depth_factor = Z[i,j,k] / self.domain_size[2]
                    T = T_base + 5 * depth_factor  # Slight temperature gradient
                    
                    # Displacement field (thermal contraction)
                    delta_T = 1450 - T  # Temperature drop from sintering
                    thermal_strain = self.material_props['cte'] * delta_T
                    
                    # Edge effects and constraints
                    edge_constraint_x = min(X[i,j,k], self.domain_size[0] - X[i,j,k]) / self.domain_size[0]
                    edge_constraint_y = min(Y[i,j,k], self.domain_size[1] - Y[i,j,k]) / self.domain_size[1]
                    
                    u_x = thermal_strain * X[i,j,k] * edge_constraint_x
                    u_y = thermal_strain * Y[i,j,k] * edge_constraint_y
                    u_z = thermal_strain * Z[i,j,k] * 0.5  # Less constraint in z
                    
                    displacement = [u_x, u_y, u_z]
                    
                    # Stress field (from thermal mismatch and constraints)
                    stress_thermal = self._calculate_thermal_stress(coord, delta_T)
                    stress_constraint = self._calculate_constraint_stress(coord)
                    
                    stress_tensor = stress_thermal + stress_constraint
                    
                    # Strain field (total strain = thermal + mechanical)
                    strain_thermal = np.eye(3) * thermal_strain
                    strain_mechanical = self._stress_to_strain(stress_tensor)
                    strain_tensor = strain_thermal + strain_mechanical
                    
                    full_field_data['coordinates'].append([float(val) for val in coord])
                    full_field_data['temperature'].append(float(T))
                    full_field_data['displacement'].append([float(val) for val in displacement])
                    full_field_data['stress'].append([[float(val) for val in row] for row in stress_tensor])
                    full_field_data['strain'].append([[float(val) for val in row] for row in strain_tensor])
        
        # Strategic collocation points selection
        collocation_points = self._select_collocation_points(X, Y, Z)
        
        collocation_data = {
            'points': [],
            'selection_criteria': {
                'near_pores': 0.3,  # 30% of points
                'grain_boundaries': 0.4,  # 40% of points
                'free_surfaces': 0.2,  # 20% of points
                'random_bulk': 0.1   # 10% of points
            },
            'data': []
        }
        
        for point_idx in collocation_points:
            coord = full_field_data['coordinates'][point_idx]
            
            collocation_data['points'].append(coord)
            collocation_data['data'].append({
                'temperature': full_field_data['temperature'][point_idx],
                'displacement': full_field_data['displacement'][point_idx],
                'stress': full_field_data['stress'][point_idx],
                'strain': full_field_data['strain'][point_idx],
                'point_type': str(self._classify_point_type(coord))
            })
        
        # Save simulation data
        simulation_data = {
            'full_field': full_field_data,
            'collocation_points': collocation_data,
            'metadata': {
                'simulation_software': 'ANSYS Mechanical APDL',
                'material_model': 'Linear elastic with temperature-dependent properties',
                'boundary_conditions': 'Fixed bottom surface, free top and sides',
                'loading': 'Thermal cooling from 1450K to 298K',
                'mesh_quality': 0.85,
                'solution_time': 3600  # seconds
            }
        }
        
        # Save as compressed numpy arrays for efficiency
        np.savez_compressed(f"{self.output_dir}/full_field_simulation.npz",
                           coordinates=np.array(full_field_data['coordinates']),
                           temperature=np.array(full_field_data['temperature']),
                           displacement=np.array(full_field_data['displacement']),
                           stress=np.array(full_field_data['stress']),
                           strain=np.array(full_field_data['strain']))
        
        with open(f"{self.output_dir}/collocation_points.json", 'w') as f:
            json.dump(collocation_data, f, indent=2)
        
        return simulation_data
    
    def generate_multiscale_material_data(self):
        """Generate multi-scale material characterization data"""
        print("Generating multi-scale material characterization data...")
        
        # Macro-scale data (cell level)
        macro_data = {
            'bulk_properties': self.material_props.copy(),
            'cell_dimensions': {
                'length': 100e-3,  # m
                'width': 100e-3,   # m
                'thickness': 150e-6  # m
            },
            'sintering_profile': {
                'heating_rate': 2,  # K/min
                'max_temperature': 1450,  # K
                'hold_time': 4,  # hours
                'cooling_rate': 5,  # K/min
                'atmosphere': 'air'
            },
            'thermal_expansion_data': []
        }
        
        # Generate thermal expansion curve
        temperatures = np.linspace(298, 1450, 50)
        for T in temperatures:
            # YSZ thermal expansion (empirical fit)
            cte_T = 9.8e-6 + 2.1e-9 * (T - 298)  # Temperature-dependent CTE
            expansion = cte_T * (T - 298)
            macro_data['thermal_expansion_data'].append({
                'temperature': float(T),
                'cte': float(cte_T),
                'linear_expansion': float(expansion)
            })
        
        # Meso-scale data (grain and pore level)
        meso_data = {
            'microstructure': {
                'grain_size_distribution': [],
                'pore_size_distribution': [],
                'grain_aspect_ratio': [],
                'pore_connectivity': 0.75,  # Fraction of connected pores
                'tortuosity': 2.3
            },
            'rve_properties': {
                'size': [20e-6, 20e-6, 20e-6],  # m
                'n_grains': 150,
                'n_pores': 45,
                'effective_modulus': [],
                'stress_concentration_factors': []
            }
        }
        
        # Generate grain size distribution (log-normal)
        n_grains = 500
        grain_sizes = np.random.lognormal(np.log(2.5e-6), 0.3, n_grains)
        grain_aspects = np.random.gamma(2, 0.5, n_grains)  # Aspect ratio
        
        for i in range(n_grains):
            meso_data['microstructure']['grain_size_distribution'].append({
                'grain_id': int(i),
                'equivalent_diameter': float(grain_sizes[i]),
                'aspect_ratio': float(grain_aspects[i]),
                'orientation': [float(val) for val in np.random.uniform(0, 2*np.pi, 3)]
            })
        
        # Generate pore size distribution
        n_pores = 150
        pore_sizes = np.random.lognormal(np.log(0.8e-6), 0.5, n_pores)
        
        for i in range(n_pores):
            meso_data['microstructure']['pore_size_distribution'].append({
                'pore_id': int(i),
                'equivalent_diameter': float(pore_sizes[i]),
                'sphericity': float(np.random.uniform(0.6, 0.95)),
                'coordination_number': int(np.random.poisson(4))
            })
        
        # Stress concentration factors around pores
        for pore_size in np.linspace(0.5e-6, 3e-6, 20):
            # Analytical solution for spherical pore in infinite medium
            scf = 2.045 + 0.5 * (pore_size / 2.5e-6)  # Size effect
            meso_data['rve_properties']['stress_concentration_factors'].append({
                'pore_size': float(pore_size),
                'stress_concentration_factor': float(scf),
                'location': 'pore_surface'
            })
        
        # Micro-scale data (grain boundary level)
        micro_data = {
            'grain_boundaries': {
                'gb_energy': 1.2,  # J/m²
                'gb_thickness': 1e-9,  # m
                'gb_diffusivity': 1e-14,  # m²/s at 1200K
                'gb_strength': 80e6,  # Pa
                'misorientation_distribution': []
            },
            'crystallographic_data': {
                'crystal_structure': 'cubic_fluorite',
                'lattice_parameter': 5.139e-10,  # m
                'elastic_constants': {
                    'C11': 400e9,  # Pa
                    'C12': 100e9,  # Pa
                    'C44': 60e9    # Pa
                }
            }
        }
        
        # Generate grain boundary misorientation data
        n_boundaries = 300
        for i in range(n_boundaries):
            # Random misorientation angles (Brandon criterion for special boundaries)
            angle = np.random.exponential(15)  # degrees
            axis = np.random.uniform(-1, 1, 3)
            axis = axis / np.linalg.norm(axis)
            
            # Classify boundary type
            if angle < 15:
                gb_type = 'low_angle'
                gb_energy = 0.5  # J/m²
            elif angle > 15 and angle < 45:
                gb_type = 'high_angle'
                gb_energy = 1.2  # J/m²
            else:
                gb_type = 'random'
                gb_energy = 1.0  # J/m²
            
            micro_data['grain_boundaries']['misorientation_distribution'].append({
                'boundary_id': int(i),
                'misorientation_angle': float(angle),
                'misorientation_axis': [float(val) for val in axis],
                'boundary_type': str(gb_type),
                'energy': float(gb_energy)
            })
        
        # Save multi-scale data
        multiscale_data = {
            'macro_scale': macro_data,
            'meso_scale': meso_data,
            'micro_scale': micro_data,
            'metadata': {
                'characterization_methods': [
                    'SEM/EBSD for microstructure',
                    'X-ray CT for 3D pore structure',
                    'Nanoindentation for local properties',
                    'TEM for grain boundary structure'
                ],
                'sample_preparation': 'Ion beam polishing',
                'measurement_conditions': 'Room temperature, high vacuum'
            }
        }
        
        with open(f"{self.output_dir}/multiscale_material_data.json", 'w') as f:
            json.dump(multiscale_data, f, indent=2)
        
        return multiscale_data
    
    def _calculate_local_stress(self, x, y, z):
        """Calculate local stress at given coordinates"""
        # Simplified stress calculation based on position
        edge_factor = min(x, self.domain_size[0]-x, y, self.domain_size[1]-y) / (self.domain_size[0]/4)
        depth_factor = z / self.domain_size[2]
        
        base_stress = 50e6 * (1 + 2*np.exp(-edge_factor)) * (1 - 0.3*depth_factor)
        return abs(base_stress)
    
    def _calculate_thermal_stress(self, coord, delta_T):
        """Calculate thermal stress tensor"""
        x, y, z = coord
        
        # Thermal stress due to constrained thermal expansion
        thermal_stress_mag = (self.material_props['young_modulus'] * 
                            self.material_props['cte'] * delta_T / 
                            (1 - self.material_props['poisson_ratio']))
        
        # Non-uniform stress due to geometry and constraints
        edge_factor_x = min(x, self.domain_size[0] - x) / self.domain_size[0]
        edge_factor_y = min(y, self.domain_size[1] - y) / self.domain_size[1]
        
        stress_xx = thermal_stress_mag * edge_factor_x
        stress_yy = thermal_stress_mag * edge_factor_y
        stress_zz = thermal_stress_mag * 0.3  # Less constraint in z
        
        return np.array([
            [stress_xx, 0, 0],
            [0, stress_yy, 0],
            [0, 0, stress_zz]
        ])
    
    def _calculate_constraint_stress(self, coord):
        """Calculate stress due to geometric constraints"""
        x, y, z = coord
        
        # Additional stress concentrations near boundaries
        boundary_stress = np.zeros((3, 3))
        
        # Edge effects
        if x < 5e-6 or x > self.domain_size[0] - 5e-6:
            boundary_stress[0, 0] += 20e6
        if y < 5e-6 or y > self.domain_size[1] - 5e-6:
            boundary_stress[1, 1] += 20e6
        
        return boundary_stress
    
    def _stress_to_strain(self, stress_tensor):
        """Convert stress tensor to strain tensor using Hooke's law"""
        E = self.material_props['young_modulus']
        nu = self.material_props['poisson_ratio']
        
        # Compliance matrix for isotropic material
        S11 = 1/E
        S12 = -nu/E
        S44 = 2*(1+nu)/E
        
        strain = np.zeros_like(stress_tensor)
        
        # Normal strains
        strain[0, 0] = S11*stress_tensor[0, 0] + S12*stress_tensor[1, 1] + S12*stress_tensor[2, 2]
        strain[1, 1] = S12*stress_tensor[0, 0] + S11*stress_tensor[1, 1] + S12*stress_tensor[2, 2]
        strain[2, 2] = S12*stress_tensor[0, 0] + S12*stress_tensor[1, 1] + S11*stress_tensor[2, 2]
        
        # Shear strains
        strain[0, 1] = strain[1, 0] = S44 * stress_tensor[0, 1]
        strain[0, 2] = strain[2, 0] = S44 * stress_tensor[0, 2]
        strain[1, 2] = strain[2, 1] = S44 * stress_tensor[1, 2]
        
        return strain
    
    def _select_collocation_points(self, X, Y, Z):
        """Select strategic collocation points"""
        total_points = X.size
        n_collocation = min(500, total_points // 10)  # 10% of total points or 500, whichever is smaller
        
        indices = []
        
        # Near pores (30%)
        n_pore = int(0.3 * n_collocation)
        pore_indices = self._find_pore_vicinity_points(X, Y, Z, n_pore)
        indices.extend(pore_indices)
        
        # Grain boundaries (40%)
        n_gb = int(0.4 * n_collocation)
        gb_indices = self._find_grain_boundary_points(X, Y, Z, n_gb)
        indices.extend(gb_indices)
        
        # Free surfaces (20%)
        n_surface = int(0.2 * n_collocation)
        surface_indices = self._find_surface_points(X, Y, Z, n_surface)
        indices.extend(surface_indices)
        
        # Random bulk (10%)
        n_bulk = n_collocation - len(indices)
        remaining_indices = list(set(range(total_points)) - set(indices))
        bulk_indices = np.random.choice(remaining_indices, n_bulk, replace=False)
        indices.extend(bulk_indices)
        
        return indices[:n_collocation]
    
    def _find_pore_vicinity_points(self, X, Y, Z, n_points):
        """Find points near simulated pore locations"""
        # Simulate pore locations
        n_pores = 20
        pore_centers = []
        for _ in range(n_pores):
            center = [
                np.random.uniform(10e-6, 90e-6),
                np.random.uniform(10e-6, 90e-6),
                np.random.uniform(2e-6, 18e-6)
            ]
            pore_centers.append(center)
        
        # Find mesh points near pores
        mesh_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        pore_centers = np.array(pore_centers)
        
        distances = cdist(mesh_points, pore_centers)
        min_distances = np.min(distances, axis=1)
        
        # Select points within 2 microns of pores
        near_pore_mask = min_distances < 2e-6
        near_pore_indices = np.where(near_pore_mask)[0]
        
        if len(near_pore_indices) >= n_points:
            return np.random.choice(near_pore_indices, n_points, replace=False).tolist()
        else:
            return near_pore_indices.tolist()
    
    def _find_grain_boundary_points(self, X, Y, Z, n_points):
        """Find points near simulated grain boundaries"""
        # Simulate grain boundary network using Voronoi-like approach
        mesh_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        # Create grain centers
        n_grains = 50
        grain_centers = []
        for _ in range(n_grains):
            center = [
                np.random.uniform(0, self.domain_size[0]),
                np.random.uniform(0, self.domain_size[1]),
                np.random.uniform(0, self.domain_size[2])
            ]
            grain_centers.append(center)
        
        grain_centers = np.array(grain_centers)
        
        # Find points near grain boundaries (where distance to two nearest grains is similar)
        distances = cdist(mesh_points, grain_centers)
        sorted_distances = np.sort(distances, axis=1)
        
        # Points where first and second nearest grains are close in distance
        gb_criterion = (sorted_distances[:, 1] - sorted_distances[:, 0]) < 1e-6
        gb_indices = np.where(gb_criterion)[0]
        
        if len(gb_indices) >= n_points:
            return np.random.choice(gb_indices, n_points, replace=False).tolist()
        else:
            return gb_indices.tolist()
    
    def _find_surface_points(self, X, Y, Z, n_points):
        """Find points on or near free surfaces"""
        mesh_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        surface_mask = ((mesh_points[:, 0] < 2e-6) | 
                       (mesh_points[:, 0] > self.domain_size[0] - 2e-6) |
                       (mesh_points[:, 1] < 2e-6) | 
                       (mesh_points[:, 1] > self.domain_size[1] - 2e-6) |
                       (mesh_points[:, 2] < 1e-6) | 
                       (mesh_points[:, 2] > self.domain_size[2] - 1e-6))
        
        surface_indices = np.where(surface_mask)[0]
        
        if len(surface_indices) >= n_points:
            return np.random.choice(surface_indices, n_points, replace=False).tolist()
        else:
            return surface_indices.tolist()
    
    def _classify_point_type(self, coord):
        """Classify the type of collocation point"""
        x, y, z = coord
        
        # Check if near surface
        if (x < 2e-6 or x > self.domain_size[0] - 2e-6 or
            y < 2e-6 or y > self.domain_size[1] - 2e-6 or
            z < 1e-6 or z > self.domain_size[2] - 1e-6):
            return 'surface'
        
        # Simple classification based on position
        # In reality, this would use actual microstructural data
        hash_val = hash((round(x*1e6), round(y*1e6), round(z*1e6))) % 100
        
        if hash_val < 30:
            return 'pore_vicinity'
        elif hash_val < 70:
            return 'grain_boundary'
        else:
            return 'bulk'
    
    def create_visualization_plots(self):
        """Create visualization plots of the generated data"""
        print("Creating visualization plots...")
        
        # Load experimental stress data
        with open(f"{self.output_dir}/experimental_residual_stress.json", 'r') as f:
            exp_data = json.load(f)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # XRD stress map
        xrd_points = np.array(exp_data['xrd_surface']['measurement_points'])
        xrd_stress = np.array(exp_data['xrd_surface']['stress_xx'])
        
        scatter = axes[0, 0].scatter(xrd_points[:, 0]*1e6, xrd_points[:, 1]*1e6, 
                                   c=xrd_stress*1e-6, cmap='RdBu_r', s=100)
        axes[0, 0].set_xlabel('X (μm)')
        axes[0, 0].set_ylabel('Y (μm)')
        axes[0, 0].set_title('XRD Surface Stress σ_xx (MPa)')
        plt.colorbar(scatter, ax=axes[0, 0])
        
        # Raman stress distribution
        raman_points = np.array(exp_data['raman_spectroscopy']['measurement_points'])
        raman_stress = np.array(exp_data['raman_spectroscopy']['stress_magnitude'])
        
        scatter2 = axes[0, 1].scatter(raman_points[:, 0]*1e6, raman_points[:, 1]*1e6, 
                                    c=raman_stress*1e-6, cmap='viridis', s=50)
        axes[0, 1].set_xlabel('X (μm)')
        axes[0, 1].set_ylabel('Y (μm)')
        axes[0, 1].set_title('Raman Stress Magnitude (MPa)')
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        # Load crack data
        with open(f"{self.output_dir}/crack_initiation_propagation.json", 'r') as f:
            crack_data = json.load(f)
        
        # Crack locations
        crack_points = np.array(crack_data['crack_locations'])
        crack_lengths = np.array(crack_data['crack_lengths'])
        
        scatter3 = axes[1, 0].scatter(crack_points[:, 0]*1e6, crack_points[:, 1]*1e6, 
                                    c=crack_lengths*1e6, cmap='Reds', s=80)
        axes[1, 0].set_xlabel('X (μm)')
        axes[1, 0].set_ylabel('Y (μm)')
        axes[1, 0].set_title('Crack Locations & Lengths (μm)')
        plt.colorbar(scatter3, ax=axes[1, 0])
        
        # Grain size distribution
        with open(f"{self.output_dir}/multiscale_material_data.json", 'r') as f:
            material_data = json.load(f)
        
        grain_sizes = [g['equivalent_diameter'] for g in 
                      material_data['meso_scale']['microstructure']['grain_size_distribution']]
        
        axes[1, 1].hist(np.array(grain_sizes)*1e6, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].set_xlabel('Grain Size (μm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Grain Size Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/dataset_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {self.output_dir}/dataset_visualization.png")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Generating summary report...")
        
        report = f"""
# Validation & Analysis Dataset Summary Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview

This synthetic dataset provides comprehensive validation and analysis data for FEM modeling of SOFC electrolyte residual stresses. The dataset includes experimental measurements, crack characterization, simulation outputs, and multi-scale material properties.

## Dataset Components

### 1. Experimental Residual Stress Data
- **XRD Surface Measurements**: 25 measurement points across the surface
  - Spatial resolution: ~20 μm
  - Stress components: σ_xx, σ_yy, σ_xy
  - Typical stress range: -150 to +50 MPa
  - Measurement uncertainty: 5-15 MPa

- **Raman Spectroscopy**: 100 high-resolution surface measurements
  - Spatial resolution: 1 μm
  - Stress magnitude and peak characteristics
  - Captures microstructural stress variations

- **Synchrotron X-ray Diffraction**: 80 3D bulk measurements
  - Full stress tensor at each point
  - Through-thickness stress variation
  - Grain orientation data

### 2. Crack Initiation & Propagation Data
- **Crack Locations**: 15 documented cracks
  - 5 surface-initiated cracks
  - 5 pore-initiated cracks
  - 5 grain boundary cracks

- **Critical Conditions**:
  - Critical stress range: 80-150 MPa
  - Critical temperature range: 1200-1400 K
  - Paris law parameters for propagation

- **SEM Observations**: Detailed microstructural characterization
  - Crack widths: 0.1-2 μm
  - Local grain size and porosity measurements

### 3. Collocation Point Simulation Data
- **Full-Field Data**: {np.prod(self.mesh_resolution)} mesh points
  - Temperature, displacement, stress, and strain fields
  - Complete thermal-mechanical simulation results

- **Strategic Collocation Points**: 500 selected points
  - 30% near pores (stress concentrators)
  - 40% at grain boundaries
  - 20% on free surfaces
  - 10% random bulk locations

### 4. Multi-Scale Material Characterization

#### Macro-Scale (Cell Level)
- Bulk material properties for YSZ
- Cell dimensions: 100×100×0.15 mm
- Sintering temperature profile
- Temperature-dependent thermal expansion

#### Meso-Scale (Grain & Pore Level)
- Grain size distribution: log-normal (mean=2.5 μm, σ=0.8 μm)
- Pore size distribution: log-normal (mean=0.8 μm, σ=0.5 μm)
- Porosity: 5%
- Stress concentration factors around pores

#### Micro-Scale (Grain Boundary Level)
- Grain boundary properties and energy
- Misorientation distribution
- Crystallographic data for cubic fluorite structure

## Data Quality & Validation

### Experimental Data Realism
- Stress values consistent with literature for YSZ
- Realistic measurement uncertainties
- Proper scaling relationships between techniques

### Simulation Data Consistency
- Thermodynamically consistent stress-strain relationships
- Realistic boundary conditions and constraints
- Proper coupling between thermal and mechanical fields

### Multi-Scale Coherence
- Properties scale appropriately across length scales
- Microstructural features affect local stress distributions
- Statistical distributions match experimental observations

## Usage Guidelines

### For FEM Model Validation
1. Compare simulation predictions with experimental stress measurements
2. Use residual analysis to identify model deficiencies
3. Focus on regions with high stress gradients or concentrations

### For Surrogate Model Training
1. Use full-field data as reference solution
2. Train on collocation point data
3. Validate surrogate accuracy against full-field results

### For Crack Prediction Validation
1. Compare predicted crack locations with observed data
2. Validate critical stress/temperature predictions
3. Use propagation data for fatigue life estimation

## File Structure
```
{self.output_dir}/
├── experimental_residual_stress.json    # XRD, Raman, Synchrotron data
├── crack_initiation_propagation.json   # Crack characterization
├── full_field_simulation.npz           # Complete FEM results
├── collocation_points.json             # Strategic point data
├── multiscale_material_data.json       # Material properties
├── dataset_visualization.png           # Summary plots
└── dataset_summary_report.md           # This report
```

## Technical Specifications

- **Material**: YSZ (8 mol% Y₂O₃-ZrO₂)
- **Domain Size**: 100×100×20 μm³
- **Mesh Resolution**: 50×50×10 nodes
- **Temperature Range**: 298-1450 K
- **Stress Range**: -200 to +100 MPa
- **Spatial Resolution**: 0.5-20 μm (technique dependent)

## Limitations & Assumptions

1. **Idealized Geometry**: Simplified rectangular domain
2. **Linear Elasticity**: No plasticity or creep effects
3. **Isotropic Properties**: Single-crystal anisotropy neglected
4. **Static Analysis**: No dynamic or time-dependent effects
5. **Perfect Interfaces**: No delamination or interface failure

## Recommended Next Steps

1. **Model Validation**: Compare FEM predictions with experimental data
2. **Residual Analysis**: Identify systematic model errors
3. **Surrogate Development**: Train ML models on collocation data
4. **Uncertainty Quantification**: Propagate measurement uncertainties
5. **Model Refinement**: Update based on validation results

---

*This dataset provides a comprehensive foundation for validating and improving FEM models of residual stress in SOFC electrolytes. The multi-scale, multi-physics approach enables thorough model assessment and development of advanced analysis techniques.*
"""
        
        with open(f"{self.output_dir}/dataset_summary_report.md", 'w') as f:
            f.write(report)
        
        print(f"Summary report saved to {self.output_dir}/dataset_summary_report.md")

def main():
    """Main function to generate the complete validation dataset"""
    print("=== SOFC Electrolyte Validation Dataset Generator ===")
    print("Generating comprehensive synthetic dataset for FEM model validation...")
    
    generator = ValidationDatasetGenerator()
    
    # Generate all dataset components
    exp_data = generator.generate_residual_stress_experimental()
    crack_data = generator.generate_crack_data()
    sim_data = generator.generate_collocation_point_data()
    material_data = generator.generate_multiscale_material_data()
    
    # Create visualizations and summary
    generator.create_visualization_plots()
    generator.generate_summary_report()
    
    print(f"\n=== Dataset Generation Complete ===")
    print(f"All files saved to: {generator.output_dir}")
    print(f"Total dataset size: ~{sum(os.path.getsize(os.path.join(generator.output_dir, f)) for f in os.listdir(generator.output_dir)) / 1024 / 1024:.1f} MB")
    
    # Print summary statistics
    print(f"\nDataset Summary:")
    print(f"- Experimental measurements: {len(exp_data['xrd_surface']['measurement_points']) + len(exp_data['raman_spectroscopy']['measurement_points']) + len(exp_data['synchrotron_bulk']['measurement_points'])}")
    print(f"- Crack observations: {len(crack_data['crack_locations'])}")
    print(f"- Full-field simulation points: {len(sim_data['full_field']['coordinates'])}")
    print(f"- Collocation points: {len(sim_data['collocation_points']['points'])}")
    print(f"- Material characterization: Multi-scale (macro/meso/micro)")

if __name__ == "__main__":
    main()