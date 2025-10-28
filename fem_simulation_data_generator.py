"""
Multi-Physics FEM Simulation Dataset Generator
Generates realistic numerical simulation data mimicking COMSOL/ABAQUS outputs
for multi-physics analysis including thermal, mechanical, and electrochemical domains.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class FEMSimulationDataGenerator:
    """Generate comprehensive FEM simulation datasets"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_mesh_data(self, nx=50, ny=50, nz=20, refinement_factor=2.0):
        """
        Generate mesh data with element size, type, and interface refinement
        
        Parameters:
        -----------
        nx, ny, nz : int
            Number of elements in x, y, z directions
        refinement_factor : float
            Refinement factor at interfaces (higher = finer mesh)
        """
        print("Generating mesh data...")
        
        # Create structured mesh
        x = np.linspace(0, 10, nx)  # mm
        y = np.linspace(0, 10, ny)  # mm
        z = np.linspace(0, 5, nz)   # mm
        
        X, Y, Z = np.meshgrid(x, y, z)
        
        # Flatten for node coordinates
        nodes = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        n_nodes = len(nodes)
        
        # Generate elements (hexahedral elements)
        elements = []
        element_types = []
        element_sizes = []
        
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    # Node indices for hexahedral element
                    n0 = i*ny*nz + j*nz + k
                    n1 = (i+1)*ny*nz + j*nz + k
                    n2 = (i+1)*ny*nz + (j+1)*nz + k
                    n3 = i*ny*nz + (j+1)*nz + k
                    n4 = i*ny*nz + j*nz + (k+1)
                    n5 = (i+1)*ny*nz + j*nz + (k+1)
                    n6 = (i+1)*ny*nz + (j+1)*nz + (k+1)
                    n7 = i*ny*nz + (j+1)*nz + (k+1)
                    
                    elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
                    
                    # Determine element type based on location
                    if k < nz/2:
                        element_types.append("C3D8")  # Linear brick
                    else:
                        element_types.append("C3D8R")  # Reduced integration brick
                    
                    # Calculate element size (with interface refinement)
                    interface_zone = (2 < z[k] < 3)  # Interface at z=2.5mm
                    if interface_zone:
                        size = 0.1 / refinement_factor  # Finer mesh
                    else:
                        size = 0.2  # Coarser mesh
                    element_sizes.append(size)
        
        elements = np.array(elements)
        n_elements = len(elements)
        
        mesh_data = {
            'nodes': nodes,
            'n_nodes': n_nodes,
            'elements': elements,
            'n_elements': n_elements,
            'element_types': element_types,
            'element_sizes': np.array(element_sizes),
            'dimensions': {'nx': nx, 'ny': ny, 'nz': nz},
            'domain_size': [x.max(), y.max(), z.max()],
            'refinement_factor': refinement_factor
        }
        
        print(f"  Generated {n_nodes} nodes and {n_elements} elements")
        return mesh_data
    
    def generate_boundary_conditions(self, mesh_data):
        """Generate boundary conditions for temperature, displacement, and voltage"""
        print("Generating boundary conditions...")
        
        nodes = mesh_data['nodes']
        n_nodes = mesh_data['n_nodes']
        
        # Temperature boundary conditions
        temp_bc = {
            'fixed_nodes': [],
            'fixed_temperatures': []
        }
        
        # Bottom surface: fixed temperature
        bottom_nodes = np.where(nodes[:, 2] < 0.1)[0]
        temp_bc['fixed_nodes'].extend(bottom_nodes.tolist())
        temp_bc['fixed_temperatures'].extend([25.0] * len(bottom_nodes))  # °C
        
        # Top surface: prescribed temperature (will vary with time)
        top_nodes = np.where(nodes[:, 2] > mesh_data['domain_size'][2] - 0.1)[0]
        temp_bc['top_surface_nodes'] = top_nodes.tolist()
        temp_bc['top_surface_temp_function'] = "time-dependent (see thermal profiles)"
        
        # Displacement boundary conditions
        disp_bc = {
            'fixed_nodes': [],
            'fixed_dof': [],  # 0=x, 1=y, 2=z
            'prescribed_displacement': []
        }
        
        # Fix bottom surface in z-direction
        for node in bottom_nodes:
            disp_bc['fixed_nodes'].append(int(node))
            disp_bc['fixed_dof'].append(2)  # z-direction
            disp_bc['prescribed_displacement'].append(0.0)
        
        # Symmetry conditions on side faces
        left_nodes = np.where(nodes[:, 0] < 0.1)[0]
        for node in left_nodes[:10]:  # Sample nodes
            disp_bc['fixed_nodes'].append(int(node))
            disp_bc['fixed_dof'].append(0)  # x-direction
            disp_bc['prescribed_displacement'].append(0.0)
        
        # Voltage boundary conditions (for electrochemical)
        voltage_bc = {
            'cathode_nodes': [],
            'cathode_voltage': 4.2,  # V
            'anode_nodes': [],
            'anode_voltage': 0.0,  # V (ground)
        }
        
        # Cathode at top
        voltage_bc['cathode_nodes'] = top_nodes[:100].tolist()
        # Anode at bottom
        voltage_bc['anode_nodes'] = bottom_nodes[:100].tolist()
        
        bc_data = {
            'temperature': temp_bc,
            'displacement': disp_bc,
            'voltage': voltage_bc
        }
        
        print(f"  Generated BCs: {len(temp_bc['fixed_nodes'])} temp nodes, "
              f"{len(disp_bc['fixed_nodes'])} disp nodes")
        
        return bc_data
    
    def generate_material_models(self):
        """Generate material models with various constitutive behaviors"""
        print("Generating material models...")
        
        materials = {
            'cathode_material': {
                'name': 'NMC (LiNi0.8Mn0.1Co0.1O2)',
                'elastic': {
                    'youngs_modulus': 150e9,  # Pa
                    'poissons_ratio': 0.3,
                    'density': 4700  # kg/m³
                },
                'plastic': {
                    'yield_stress': 150e6,  # Pa
                    'hardening_modulus': 2e9,  # Pa
                    'hardening_exponent': 0.2
                },
                'thermal': {
                    'thermal_expansion': 12e-6,  # 1/K
                    'thermal_conductivity': 2.0,  # W/m·K
                    'specific_heat': 700  # J/kg·K
                },
                'electrochemical': {
                    'diffusion_coefficient': 1e-14,  # m²/s
                    'ionic_conductivity': 0.1,  # S/m
                    'max_concentration': 51765  # mol/m³
                }
            },
            'anode_material': {
                'name': 'Graphite',
                'elastic': {
                    'youngs_modulus': 15e9,  # Pa
                    'poissons_ratio': 0.3,
                    'density': 2260  # kg/m³
                },
                'plastic': {
                    'yield_stress': 50e6,  # Pa
                    'hardening_modulus': 1e9,  # Pa
                    'hardening_exponent': 0.15
                },
                'thermal': {
                    'thermal_expansion': 8e-6,  # 1/K
                    'thermal_conductivity': 1.5,  # W/m·K
                    'specific_heat': 1200  # J/kg·K
                },
                'electrochemical': {
                    'diffusion_coefficient': 3.9e-14,  # m²/s
                    'ionic_conductivity': 0.05,  # S/m
                    'max_concentration': 30555  # mol/m³
                }
            },
            'separator': {
                'name': 'Polymer Separator',
                'elastic': {
                    'youngs_modulus': 0.5e9,  # Pa
                    'poissons_ratio': 0.4,
                    'density': 1200  # kg/m³
                },
                'thermal': {
                    'thermal_expansion': 50e-6,  # 1/K
                    'thermal_conductivity': 0.3,  # W/m·K
                    'specific_heat': 1200  # J/kg·K
                },
                'electrochemical': {
                    'ionic_conductivity': 1.0,  # S/m
                    'porosity': 0.4
                }
            },
            'creep_model': {
                'power_law_creep': {
                    'coefficient_A': 1e-10,  # 1/(Pa^n·s)
                    'stress_exponent_n': 5.0,
                    'activation_energy_Q': 120e3,  # J/mol
                    'gas_constant_R': 8.314  # J/mol·K
                }
            }
        }
        
        print(f"  Generated {len(materials)} material models")
        return materials
    
    def generate_transient_thermal_profiles(self, duration=3600, dt=10, 
                                           heating_rates=[1, 5, 10], 
                                           cooling_rates=[1, 5, 10]):
        """
        Generate transient thermal profiles with various heating/cooling rates
        
        Parameters:
        -----------
        duration : float
            Total simulation time (seconds)
        dt : float
            Time step (seconds)
        heating_rates : list
            Heating rates in °C/min
        cooling_rates : list
            Cooling rates in °C/min
        """
        print("Generating transient thermal profiles...")
        
        time = np.arange(0, duration, dt)
        n_steps = len(time)
        
        profiles = {}
        
        for h_rate in heating_rates:
            for c_rate in cooling_rates:
                profile_name = f"heat_{h_rate}C_cool_{c_rate}C"
                
                T_base = 25.0  # °C
                T_max = 60.0   # °C
                
                # Create heating and cooling cycle
                t_heat = (T_max - T_base) / (h_rate / 60.0)  # Convert to °C/s
                t_cool = (T_max - T_base) / (c_rate / 60.0)
                
                temperature = np.zeros(n_steps)
                
                for i, t in enumerate(time):
                    cycle_time = t % (t_heat + t_cool)
                    if cycle_time < t_heat:
                        # Heating phase
                        temperature[i] = T_base + (h_rate / 60.0) * cycle_time
                    else:
                        # Cooling phase
                        temperature[i] = T_max - (c_rate / 60.0) * (cycle_time - t_heat)
                    
                    # Add some noise
                    temperature[i] += np.random.normal(0, 0.1)
                
                profiles[profile_name] = {
                    'time': time,
                    'temperature': temperature,
                    'heating_rate': h_rate,
                    'cooling_rate': c_rate,
                    'max_temp': T_max,
                    'base_temp': T_base
                }
        
        print(f"  Generated {len(profiles)} thermal profiles")
        return profiles
    
    def generate_stress_distributions(self, mesh_data, thermal_profile, time_idx):
        """Generate stress distributions (von Mises, principal, interfacial shear)"""
        
        nodes = mesh_data['nodes']
        n_nodes = mesh_data['n_nodes']
        
        # Get temperature at this time step
        T = thermal_profile['temperature'][time_idx]
        T_ref = thermal_profile['base_temp']
        dT = T - T_ref
        
        # Generate stress field with spatial variation
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        
        # Thermal stress component
        alpha = 12e-6  # thermal expansion coefficient
        E = 150e9  # Young's modulus
        nu = 0.3  # Poisson's ratio
        
        thermal_stress = E * alpha * dT / (1 - 2*nu)
        
        # Add spatial variation based on geometry
        stress_magnitude = thermal_stress * (1 + 0.3*np.sin(2*np.pi*x/10) * np.cos(2*np.pi*y/10))
        
        # Interface effect (stress concentration at z=2.5)
        interface_factor = 1 + 2*np.exp(-((z - 2.5)**2) / 0.5)
        stress_magnitude *= interface_factor
        
        # Add boundary layer effects
        edge_factor = 1 + 0.5*np.exp(-x/2) + 0.5*np.exp(-(10-x)/2)
        stress_magnitude *= edge_factor
        
        # Principal stresses
        sigma_1 = stress_magnitude * (1.0 + 0.2*np.random.randn(n_nodes))
        sigma_2 = stress_magnitude * (0.7 + 0.2*np.random.randn(n_nodes))
        sigma_3 = stress_magnitude * (0.4 + 0.2*np.random.randn(n_nodes))
        
        # von Mises stress
        von_mises = np.sqrt(0.5 * ((sigma_1 - sigma_2)**2 + 
                                    (sigma_2 - sigma_3)**2 + 
                                    (sigma_3 - sigma_1)**2))
        
        # Interfacial shear stress (highest at interface)
        shear_stress = 0.577 * stress_magnitude * np.exp(-((z - 2.5)**2) / 0.3)
        shear_stress += 0.1 * stress_magnitude * np.random.randn(n_nodes)
        
        stress_data = {
            'von_mises': von_mises,
            'principal_1': sigma_1,
            'principal_2': sigma_2,
            'principal_3': sigma_3,
            'shear_stress': shear_stress,
            'hydrostatic': (sigma_1 + sigma_2 + sigma_3) / 3
        }
        
        return stress_data
    
    def generate_strain_fields(self, mesh_data, stress_data, material, time_step, dt):
        """Generate strain fields (elastic, plastic, creep, thermal)"""
        
        n_nodes = mesh_data['n_nodes']
        nodes = mesh_data['nodes']
        z = nodes[:, 2]
        
        E = material['elastic']['youngs_modulus']
        nu = material['elastic']['poissons_ratio']
        alpha = material['thermal']['thermal_expansion']
        
        # Elastic strain (from Hooke's law)
        elastic_strain = stress_data['von_mises'] / E
        
        # Plastic strain (based on von Mises plasticity)
        sigma_y = material['plastic']['yield_stress']
        sigma_vm = stress_data['von_mises']
        
        plastic_strain = np.zeros(n_nodes)
        plastic_mask = sigma_vm > sigma_y
        plastic_strain[plastic_mask] = (sigma_vm[plastic_mask] - sigma_y) / \
                                       (material['plastic']['hardening_modulus'])
        
        # Accumulated plastic strain over time
        plastic_strain *= (time_step * 0.01)  # Accumulation factor
        
        # Creep strain (power law creep)
        T = 298 + 20 * np.sin(time_step * 0.01)  # Temperature variation
        creep_coef = 1e-10
        n_exp = 5.0
        Q = 120e3
        R = 8.314
        
        creep_rate = creep_coef * (sigma_vm ** n_exp) * np.exp(-Q / (R * T))
        creep_strain = creep_rate * dt * time_step
        
        # Thermal strain
        dT = 20 * np.sin(time_step * 0.01)
        thermal_strain = alpha * dT * np.ones(n_nodes)
        
        # Total strain
        total_strain = elastic_strain + plastic_strain + creep_strain + thermal_strain
        
        strain_data = {
            'elastic': elastic_strain,
            'plastic': plastic_strain,
            'creep': creep_strain,
            'thermal': thermal_strain,
            'total': total_strain,
            'equivalent_plastic': plastic_strain  # Simplified
        }
        
        return strain_data
    
    def generate_damage_evolution(self, mesh_data, stress_data, strain_data, 
                                  time_step, n_steps):
        """Generate damage variable (D) evolution over time"""
        
        n_nodes = mesh_data['n_nodes']
        nodes = mesh_data['nodes']
        
        # Initialize damage
        if time_step == 0:
            damage = np.zeros(n_nodes)
        else:
            # Damage evolution based on stress and plastic strain
            sigma_vm = stress_data['von_mises']
            eps_pl = strain_data['plastic']
            
            # Damage threshold
            sigma_threshold = 50e6  # Pa
            eps_threshold = 0.001
            
            # Damage growth rate
            damage_rate = np.zeros(n_nodes)
            
            # Only grow damage where stress/strain exceed threshold
            damage_mask = (sigma_vm > sigma_threshold) | (eps_pl > eps_threshold)
            
            if np.any(damage_mask):
                damage_rate[damage_mask] = 0.001 * (sigma_vm[damage_mask] / 1e8) * \
                                           (1 + eps_pl[damage_mask] * 100)
            
            # Accumulate damage
            damage = time_step / n_steps * damage_rate
            
            # Add interface damage (higher at z=2.5)
            z = nodes[:, 2]
            interface_damage = 0.02 * time_step / n_steps * np.exp(-((z - 2.5)**2) / 0.5)
            damage += interface_damage
            
            # Clip damage between 0 and 1
            damage = np.clip(damage, 0, 0.99)
        
        return damage
    
    def generate_temperature_voltage_distributions(self, mesh_data, thermal_profile, 
                                                   time_idx):
        """Generate temperature and voltage spatial distributions"""
        
        nodes = mesh_data['nodes']
        n_nodes = mesh_data['n_nodes']
        x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
        
        # Temperature distribution (3D heat conduction)
        T_top = thermal_profile['temperature'][time_idx]
        T_bottom = thermal_profile['base_temp']
        
        # Linear gradient + boundary effects
        T_distribution = T_bottom + (T_top - T_bottom) * (z / mesh_data['domain_size'][2])
        
        # Add lateral heat dissipation effects
        x_center, y_center = 5.0, 5.0
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2)
        T_distribution -= 2.0 * (r / 7.0)  # Cooler at edges
        
        # Add small random fluctuations
        T_distribution += np.random.normal(0, 0.5, n_nodes)
        
        # Voltage distribution (electrochemical potential)
        V_cathode = 4.2  # V
        V_anode = 0.0    # V
        
        # Linear drop across cell + overpotential effects
        V_distribution = V_anode + (V_cathode - V_anode) * (z / mesh_data['domain_size'][2])
        
        # Add concentration overpotential (higher in center)
        overpotential = 0.1 * np.exp(-((z - 2.5)**2) / 1.0) * (1 - r/10)
        V_distribution += overpotential
        
        # Add ohmic drop
        V_distribution -= 0.05 * (1 - z / mesh_data['domain_size'][2])
        
        distributions = {
            'temperature': T_distribution,
            'voltage': V_distribution,
            'temperature_gradient': np.gradient(T_distribution)[0] if len(T_distribution) > 1 else np.zeros_like(T_distribution),
            'current_density': np.abs(np.gradient(V_distribution)[0]) if len(V_distribution) > 1 else np.zeros_like(V_distribution)
        }
        
        return distributions
    
    def generate_delamination_crack_predictions(self, mesh_data, stress_data, 
                                                strain_data, damage, time_step):
        """Generate delamination and crack initiation predictions"""
        
        nodes = mesh_data['nodes']
        n_nodes = mesh_data['n_nodes']
        z = nodes[:, 2]
        
        # Delamination risk (highest at interface z=2.5)
        interface_distance = np.abs(z - 2.5)
        shear_stress = stress_data['shear_stress']
        
        # Mode II fracture criterion
        critical_shear = 30e6  # Pa
        delamination_risk = shear_stress / critical_shear
        
        # Enhanced at interface
        interface_factor = np.exp(-interface_distance / 0.3)
        delamination_risk *= (1 + 2*interface_factor)
        
        # Delamination initiation (boolean)
        delamination_initiated = (delamination_risk > 1.0) & (interface_distance < 0.5)
        
        # Crack initiation based on damage and stress
        sigma_vm = stress_data['von_mises']
        critical_stress = 100e6  # Pa
        
        crack_initiation_risk = (sigma_vm / critical_stress) * (damage / 0.5)
        crack_initiated = (crack_initiation_risk > 1.0) & (damage > 0.3)
        
        # Crack propagation direction (based on principal stress)
        crack_angle = np.arctan2(stress_data['principal_2'], stress_data['principal_1'])
        
        predictions = {
            'delamination_risk': delamination_risk,
            'delamination_initiated': delamination_initiated,
            'delamination_area': np.sum(delamination_initiated) * 0.2**2,  # mm²
            'crack_risk': crack_initiation_risk,
            'crack_initiated': crack_initiated,
            'crack_count': np.sum(crack_initiated),
            'crack_propagation_angle': crack_angle,
            'damage_variable': damage
        }
        
        return predictions
    
    def generate_complete_dataset(self, output_dir='fem_simulation_data'):
        """Generate complete multi-physics FEM simulation dataset"""
        
        print("\n" + "="*70)
        print("MULTI-PHYSICS FEM SIMULATION DATA GENERATOR")
        print("="*70 + "\n")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Generate mesh data
        mesh_data = self.generate_mesh_data(nx=30, ny=30, nz=15, refinement_factor=2.5)
        
        # 2. Generate boundary conditions
        bc_data = self.generate_boundary_conditions(mesh_data)
        
        # 3. Generate material models
        materials = self.generate_material_models()
        
        # 4. Generate thermal profiles
        thermal_profiles = self.generate_transient_thermal_profiles(
            duration=3600, dt=10, 
            heating_rates=[1, 5, 10], 
            cooling_rates=[1, 5, 10]
        )
        
        # Select one profile for time-dependent simulation
        selected_profile = thermal_profiles['heat_5C_cool_5C']
        time = selected_profile['time']
        n_steps = len(time)
        
        # 5. Generate time-dependent output data
        print("\nGenerating time-dependent output data...")
        
        # Sample time steps for full output
        sample_indices = np.linspace(0, n_steps-1, 20, dtype=int)
        
        output_data = {
            'time_steps': [],
            'stress_distributions': [],
            'strain_fields': [],
            'damage_evolution': [],
            'temperature_voltage_distributions': [],
            'delamination_crack_predictions': []
        }
        
        for idx, t_idx in enumerate(sample_indices):
            print(f"  Processing time step {idx+1}/{len(sample_indices)} "
                  f"(t = {time[t_idx]:.1f} s)...")
            
            # Generate stress distributions
            stress_data = self.generate_stress_distributions(
                mesh_data, selected_profile, t_idx
            )
            
            # Generate strain fields
            strain_data = self.generate_strain_fields(
                mesh_data, stress_data, materials['cathode_material'], idx, 10
            )
            
            # Generate damage evolution
            damage = self.generate_damage_evolution(
                mesh_data, stress_data, strain_data, idx, len(sample_indices)
            )
            
            # Generate temperature and voltage distributions
            temp_volt_dist = self.generate_temperature_voltage_distributions(
                mesh_data, selected_profile, t_idx
            )
            
            # Generate delamination and crack predictions
            delam_crack = self.generate_delamination_crack_predictions(
                mesh_data, stress_data, strain_data, damage, idx
            )
            
            # Store data
            output_data['time_steps'].append(time[t_idx])
            output_data['stress_distributions'].append(stress_data)
            output_data['strain_fields'].append(strain_data)
            output_data['damage_evolution'].append(damage)
            output_data['temperature_voltage_distributions'].append(temp_volt_dist)
            output_data['delamination_crack_predictions'].append(delam_crack)
        
        # Package complete dataset
        complete_dataset = {
            'metadata': {
                'generated_at': self.timestamp,
                'software': 'FEM Simulation Data Generator (COMSOL/ABAQUS equivalent)',
                'version': '1.0',
                'physics': ['thermal', 'mechanical', 'electrochemical'],
                'analysis_type': 'transient',
                'n_time_steps': len(sample_indices),
                'total_duration': time[-1]
            },
            'input_parameters': {
                'mesh': mesh_data,
                'boundary_conditions': bc_data,
                'materials': materials,
                'thermal_profiles': {k: {'heating_rate': v['heating_rate'],
                                        'cooling_rate': v['cooling_rate'],
                                        'max_temp': v['max_temp']}
                                    for k, v in thermal_profiles.items()}
            },
            'output_data': output_data
        }
        
        # Save dataset
        print("\nSaving dataset...")
        self.save_dataset(complete_dataset, output_path)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        self.create_visualizations(complete_dataset, output_path)
        
        print("\n" + "="*70)
        print("DATASET GENERATION COMPLETE!")
        print(f"Output directory: {output_path.absolute()}")
        print("="*70 + "\n")
        
        return complete_dataset
    
    def save_dataset(self, dataset, output_path):
        """Save dataset in multiple formats"""
        
        # Helper function to convert numpy types to native Python types
        def convert_to_native(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            return obj
        
        # Save metadata and input parameters as JSON
        json_data = {
            'metadata': dataset['metadata'],
            'input_parameters': {
                'mesh': {
                    'n_nodes': int(dataset['input_parameters']['mesh']['n_nodes']),
                    'n_elements': int(dataset['input_parameters']['mesh']['n_elements']),
                    'dimensions': convert_to_native(dataset['input_parameters']['mesh']['dimensions']),
                    'domain_size': convert_to_native(dataset['input_parameters']['mesh']['domain_size']),
                    'refinement_factor': float(dataset['input_parameters']['mesh']['refinement_factor'])
                },
                'boundary_conditions': {
                    'temperature': {
                        'n_fixed_nodes': len(dataset['input_parameters']['boundary_conditions']['temperature']['fixed_nodes']),
                        'n_top_surface_nodes': len(dataset['input_parameters']['boundary_conditions']['temperature']['top_surface_nodes'])
                    },
                    'displacement': {
                        'n_fixed_nodes': len(dataset['input_parameters']['boundary_conditions']['displacement']['fixed_nodes'])
                    },
                    'voltage': {
                        'cathode_voltage': float(dataset['input_parameters']['boundary_conditions']['voltage']['cathode_voltage']),
                        'anode_voltage': float(dataset['input_parameters']['boundary_conditions']['voltage']['anode_voltage'])
                    }
                },
                'materials': convert_to_native(dataset['input_parameters']['materials']),
                'thermal_profiles': convert_to_native(dataset['input_parameters']['thermal_profiles'])
            }
        }
        
        with open(output_path / 'dataset_metadata.json', 'w') as f:
            json.dump(json_data, f, indent=2, cls=NumpyEncoder)
        
        # Save mesh data as CSV
        mesh = dataset['input_parameters']['mesh']
        nodes_df = pd.DataFrame(mesh['nodes'], columns=['x', 'y', 'z'])
        nodes_df.to_csv(output_path / 'mesh_nodes.csv', index_label='node_id')
        
        elements_df = pd.DataFrame(mesh['elements'])
        elements_df['element_type'] = mesh['element_types']
        elements_df['element_size'] = mesh['element_sizes']
        elements_df.to_csv(output_path / 'mesh_elements.csv', index_label='element_id')
        
        # Save output data for each time step
        output_dir = output_path / 'time_series_output'
        output_dir.mkdir(exist_ok=True)
        
        for i, t in enumerate(dataset['output_data']['time_steps']):
            step_data = pd.DataFrame({
                'node_id': range(mesh['n_nodes']),
                'x': mesh['nodes'][:, 0],
                'y': mesh['nodes'][:, 1],
                'z': mesh['nodes'][:, 2],
                'time': t,
                'von_mises_stress': dataset['output_data']['stress_distributions'][i]['von_mises'],
                'principal_stress_1': dataset['output_data']['stress_distributions'][i]['principal_1'],
                'principal_stress_2': dataset['output_data']['stress_distributions'][i]['principal_2'],
                'principal_stress_3': dataset['output_data']['stress_distributions'][i]['principal_3'],
                'shear_stress': dataset['output_data']['stress_distributions'][i]['shear_stress'],
                'elastic_strain': dataset['output_data']['strain_fields'][i]['elastic'],
                'plastic_strain': dataset['output_data']['strain_fields'][i]['plastic'],
                'creep_strain': dataset['output_data']['strain_fields'][i]['creep'],
                'thermal_strain': dataset['output_data']['strain_fields'][i]['thermal'],
                'total_strain': dataset['output_data']['strain_fields'][i]['total'],
                'damage': dataset['output_data']['damage_evolution'][i],
                'temperature': dataset['output_data']['temperature_voltage_distributions'][i]['temperature'],
                'voltage': dataset['output_data']['temperature_voltage_distributions'][i]['voltage'],
                'delamination_risk': dataset['output_data']['delamination_crack_predictions'][i]['delamination_risk'],
                'crack_risk': dataset['output_data']['delamination_crack_predictions'][i]['crack_risk'],
            })
            step_data.to_csv(output_dir / f'output_t{i:03d}_time_{t:.1f}s.csv', index=False)
        
        # Save summary statistics
        summary_data = []
        for i, t in enumerate(dataset['output_data']['time_steps']):
            summary_data.append({
                'time': t,
                'max_von_mises': np.max(dataset['output_data']['stress_distributions'][i]['von_mises']),
                'avg_von_mises': np.mean(dataset['output_data']['stress_distributions'][i]['von_mises']),
                'max_plastic_strain': np.max(dataset['output_data']['strain_fields'][i]['plastic']),
                'max_damage': np.max(dataset['output_data']['damage_evolution'][i]),
                'avg_damage': np.mean(dataset['output_data']['damage_evolution'][i]),
                'max_temperature': np.max(dataset['output_data']['temperature_voltage_distributions'][i]['temperature']),
                'delamination_area': dataset['output_data']['delamination_crack_predictions'][i]['delamination_area'],
                'crack_count': dataset['output_data']['delamination_crack_predictions'][i]['crack_count']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / 'simulation_summary.csv', index=False)
        
        print(f"  Saved {len(dataset['output_data']['time_steps'])} time steps")
        print(f"  Saved to: {output_path.absolute()}")
    
    def create_visualizations(self, dataset, output_path):
        """Create visualization plots"""
        
        vis_dir = output_path / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        # Extract data
        time_steps = dataset['output_data']['time_steps']
        
        # 1. Damage evolution over time
        fig, ax = plt.subplots(figsize=(10, 6))
        max_damage = [np.max(d) for d in dataset['output_data']['damage_evolution']]
        avg_damage = [np.mean(d) for d in dataset['output_data']['damage_evolution']]
        ax.plot(time_steps, max_damage, 'r-', linewidth=2, label='Max Damage')
        ax.plot(time_steps, avg_damage, 'b--', linewidth=2, label='Avg Damage')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Damage Variable D', fontsize=12)
        ax.set_title('Damage Evolution Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / 'damage_evolution.png', dpi=300)
        plt.close()
        
        # 2. Stress evolution
        fig, ax = plt.subplots(figsize=(10, 6))
        max_stress = [np.max(s['von_mises'])/1e6 for s in dataset['output_data']['stress_distributions']]
        avg_stress = [np.mean(s['von_mises'])/1e6 for s in dataset['output_data']['stress_distributions']]
        ax.plot(time_steps, max_stress, 'r-', linewidth=2, label='Max von Mises')
        ax.plot(time_steps, avg_stress, 'b--', linewidth=2, label='Avg von Mises')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Stress (MPa)', fontsize=12)
        ax.set_title('von Mises Stress Evolution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(vis_dir / 'stress_evolution.png', dpi=300)
        plt.close()
        
        # 3. Delamination and crack tracking
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        delam_area = [p['delamination_area'] for p in dataset['output_data']['delamination_crack_predictions']]
        ax1.plot(time_steps, delam_area, 'g-', linewidth=2)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Delamination Area (mm²)', fontsize=12)
        ax1.set_title('Delamination Growth', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        crack_count = [p['crack_count'] for p in dataset['output_data']['delamination_crack_predictions']]
        ax2.plot(time_steps, crack_count, 'm-', linewidth=2)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Crack Count', fontsize=12)
        ax2.set_title('Crack Initiation', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'failure_mechanisms.png', dpi=300)
        plt.close()
        
        # 4. Temperature distribution (last time step)
        fig, ax = plt.subplots(figsize=(10, 8))
        mesh = dataset['input_parameters']['mesh']
        nodes = mesh['nodes']
        temp_dist = dataset['output_data']['temperature_voltage_distributions'][-1]['temperature']
        
        scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=temp_dist, 
                           cmap='hot', s=10, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Temperature (°C)', fontsize=12)
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(f'Temperature Distribution at t = {time_steps[-1]:.1f}s', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(vis_dir / 'temperature_distribution.png', dpi=300)
        plt.close()
        
        # 5. Voltage distribution (last time step)
        fig, ax = plt.subplots(figsize=(10, 8))
        volt_dist = dataset['output_data']['temperature_voltage_distributions'][-1]['voltage']
        
        scatter = ax.scatter(nodes[:, 0], nodes[:, 1], c=volt_dist, 
                           cmap='viridis', s=10, alpha=0.6)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Voltage (V)', fontsize=12)
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(f'Voltage Distribution at t = {time_steps[-1]:.1f}s', 
                    fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(vis_dir / 'voltage_distribution.png', dpi=300)
        plt.close()
        
        print(f"  Created 5 visualization plots in {vis_dir}")


def main():
    """Main execution function"""
    
    # Initialize generator
    generator = FEMSimulationDataGenerator(seed=42)
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset(output_dir='fem_simulation_data')
    
    print("\nDataset Summary:")
    print(f"  Total nodes: {dataset['input_parameters']['mesh']['n_nodes']}")
    print(f"  Total elements: {dataset['input_parameters']['mesh']['n_elements']}")
    print(f"  Time steps: {len(dataset['output_data']['time_steps'])}")
    print(f"  Final max damage: {np.max(dataset['output_data']['damage_evolution'][-1]):.4f}")
    print(f"  Final delamination area: {dataset['output_data']['delamination_crack_predictions'][-1]['delamination_area']:.2f} mm²")
    print(f"  Final crack count: {dataset['output_data']['delamination_crack_predictions'][-1]['crack_count']}")
    

if __name__ == "__main__":
    main()
