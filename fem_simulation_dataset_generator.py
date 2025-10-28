#!/usr/bin/env python3
"""
Multi-Physics FEM Simulation Dataset Generator
Generates synthetic numerical simulation data for COMSOL/ABAQUS-like models
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.spatial.distance import cdist
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FEMDatasetGenerator:
    def __init__(self, seed=42):
        """Initialize the FEM dataset generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_mesh_data(self, n_elements=5000, n_nodes=8000):
        """Generate mesh data with element size, type, and interface refinement"""
        print("🔧 Generating mesh data...")
        
        # Node coordinates (3D)
        nodes = {
            'node_id': np.arange(1, n_nodes + 1),
            'x': np.random.uniform(-50, 50, n_nodes),
            'y': np.random.uniform(-25, 25, n_nodes),
            'z': np.random.uniform(0, 10, n_nodes)
        }
        
        # Element data
        element_types = ['TETRA4', 'TETRA10', 'HEX8', 'HEX20', 'WEDGE6']
        elements = {
            'element_id': np.arange(1, n_elements + 1),
            'element_type': np.random.choice(element_types, n_elements),
            'element_size': np.random.lognormal(mean=0.5, sigma=0.3, size=n_elements),
            'material_id': np.random.randint(1, 6, n_elements),
            'interface_refinement': np.random.choice([1, 2, 3, 4], n_elements, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        # Interface elements (special refinement)
        n_interface = int(n_elements * 0.15)  # 15% interface elements
        interface_elements = np.random.choice(n_elements, n_interface, replace=False)
        elements['interface_refinement'][interface_elements] = np.random.choice([3, 4, 5], n_interface, p=[0.5, 0.3, 0.2])
        
        mesh_data = {
            'nodes': pd.DataFrame(nodes),
            'elements': pd.DataFrame(elements),
            'mesh_quality': {
                'min_element_size': float(np.min(elements['element_size'])),
                'max_element_size': float(np.max(elements['element_size'])),
                'avg_element_size': float(np.mean(elements['element_size'])),
                'aspect_ratio_range': [1.2, 8.5],
                'skewness_max': 0.85
            }
        }
        
        return mesh_data
    
    def generate_boundary_conditions(self, n_nodes=8000, n_time_steps=100):
        """Generate boundary conditions for temperature, displacement, and voltage"""
        print("🌡️ Generating boundary conditions...")
        
        time_array = np.linspace(0, 3600, n_time_steps)  # 1 hour simulation
        
        # Temperature boundary conditions
        temp_bc_nodes = np.random.choice(n_nodes, int(n_nodes * 0.1), replace=False)
        temp_bc = {
            'node_ids': temp_bc_nodes.tolist(),
            'type': 'prescribed_temperature',
            'values': {
                'time': time_array.tolist(),
                'temperature': (25 + 200 * np.sin(2 * np.pi * time_array / 1800) * 
                               np.exp(-time_array / 2000)).tolist()
            }
        }
        
        # Displacement boundary conditions
        disp_bc_nodes = np.random.choice(n_nodes, int(n_nodes * 0.05), replace=False)
        displacement_bc = {
            'node_ids': disp_bc_nodes.tolist(),
            'type': 'prescribed_displacement',
            'values': {
                'time': time_array.tolist(),
                'ux': (0.001 * np.sin(2 * np.pi * time_array / 600)).tolist(),
                'uy': (0.0005 * np.cos(2 * np.pi * time_array / 800)).tolist(),
                'uz': np.zeros(n_time_steps).tolist()
            }
        }
        
        # Voltage boundary conditions (for electrochemical coupling)
        voltage_bc_nodes = np.random.choice(n_nodes, int(n_nodes * 0.08), replace=False)
        voltage_bc = {
            'node_ids': voltage_bc_nodes.tolist(),
            'type': 'prescribed_voltage',
            'values': {
                'time': time_array.tolist(),
                'voltage': (3.7 + 0.8 * np.sin(2 * np.pi * time_array / 1200) * 
                           np.exp(-time_array / 3000)).tolist()
            }
        }
        
        # Heat flux boundary conditions
        flux_bc_nodes = np.random.choice(n_nodes, int(n_nodes * 0.12), replace=False)
        heat_flux_bc = {
            'node_ids': flux_bc_nodes.tolist(),
            'type': 'heat_flux',
            'values': {
                'time': time_array.tolist(),
                'flux': (1000 + 500 * np.random.normal(0, 0.1, n_time_steps)).tolist()
            }
        }
        
        boundary_conditions = {
            'temperature': temp_bc,
            'displacement': displacement_bc,
            'voltage': voltage_bc,
            'heat_flux': heat_flux_bc,
            'time_parameters': {
                'total_time': float(time_array[-1]),
                'time_step': float(time_array[1] - time_array[0]),
                'n_steps': n_time_steps
            }
        }
        
        return boundary_conditions
    
    def generate_material_models(self):
        """Generate material model parameters for different physics"""
        print("🔬 Generating material models...")
        
        materials = {}
        
        # Material 1: Aluminum (electrode)
        materials['aluminum'] = {
            'id': 1,
            'name': 'Aluminum',
            'elastic': {
                'youngs_modulus': 70e9,  # Pa
                'poissons_ratio': 0.33,
                'density': 2700  # kg/m³
            },
            'plastic': {
                'yield_strength': 276e6,  # Pa
                'hardening_modulus': 2e9,
                'hardening_exponent': 0.15
            },
            'thermal': {
                'conductivity': 237,  # W/m·K
                'specific_heat': 900,  # J/kg·K
                'expansion_coefficient': 23.1e-6,  # 1/K
                'melting_point': 933.15  # K
            },
            'electrochemical': {
                'electrical_conductivity': 3.77e7,  # S/m
                'diffusion_coefficient': 1e-14,  # m²/s
                'exchange_current_density': 0.1  # A/m²
            }
        }
        
        # Material 2: Copper (current collector)
        materials['copper'] = {
            'id': 2,
            'name': 'Copper',
            'elastic': {
                'youngs_modulus': 110e9,
                'poissons_ratio': 0.34,
                'density': 8960
            },
            'plastic': {
                'yield_strength': 210e6,
                'hardening_modulus': 1.5e9,
                'hardening_exponent': 0.12
            },
            'thermal': {
                'conductivity': 401,
                'specific_heat': 385,
                'expansion_coefficient': 16.5e-6,
                'melting_point': 1357.77
            },
            'electrochemical': {
                'electrical_conductivity': 5.96e7,
                'diffusion_coefficient': 5e-15,
                'exchange_current_density': 0.05
            }
        }
        
        # Material 3: Polymer separator
        materials['polymer'] = {
            'id': 3,
            'name': 'Polymer_Separator',
            'elastic': {
                'youngs_modulus': 2e9,
                'poissons_ratio': 0.4,
                'density': 1200
            },
            'creep': {
                'creep_coefficient': 1e-20,  # 1/Pa·s
                'stress_exponent': 4.5,
                'activation_energy': 180e3,  # J/mol
                'temperature_dependence': True
            },
            'thermal': {
                'conductivity': 0.2,
                'specific_heat': 1500,
                'expansion_coefficient': 80e-6,
                'glass_transition': 373.15
            },
            'electrochemical': {
                'electrical_conductivity': 1e-8,
                'ionic_conductivity': 1e-3,
                'diffusion_coefficient': 1e-12
            }
        }
        
        # Material 4: Ceramic coating
        materials['ceramic'] = {
            'id': 4,
            'name': 'Ceramic_Coating',
            'elastic': {
                'youngs_modulus': 380e9,
                'poissons_ratio': 0.22,
                'density': 3950
            },
            'thermal': {
                'conductivity': 25,
                'specific_heat': 750,
                'expansion_coefficient': 8.5e-6,
                'melting_point': 2323.15
            },
            'damage': {
                'critical_stress': 150e6,
                'fracture_energy': 25,  # J/m²
                'damage_evolution_law': 'exponential'
            }
        }
        
        # Material 5: Interface material
        materials['interface'] = {
            'id': 5,
            'name': 'Interface_Layer',
            'cohesive': {
                'normal_strength': 50e6,  # Pa
                'shear_strength': 35e6,   # Pa
                'normal_stiffness': 1e12, # Pa/m
                'shear_stiffness': 5e11,  # Pa/m
                'fracture_energy_I': 100, # J/m²
                'fracture_energy_II': 250 # J/m²
            },
            'thermal': {
                'conductivity': 1.5,
                'resistance': 1e-6  # m²·K/W
            }
        }
        
        return materials
    
    def generate_thermal_profiles(self, duration=3600, n_profiles=10):
        """Generate transient thermal profiles with specified heating/cooling rates"""
        print("🔥 Generating thermal profiles...")
        
        time_array = np.linspace(0, duration, 1000)
        profiles = {}
        
        for i in range(n_profiles):
            # Random heating/cooling rates between 1-10°C/min
            heating_rate = np.random.uniform(1, 10) / 60  # °C/s
            cooling_rate = np.random.uniform(-10, -1) / 60  # °C/s
            
            # Create profile with heating, hold, and cooling phases
            t1 = duration * 0.3  # heating phase
            t2 = duration * 0.7  # hold phase
            t3 = duration       # cooling phase
            
            temperature = np.zeros_like(time_array)
            base_temp = 25  # °C
            max_temp = base_temp + heating_rate * t1 * 60
            
            for j, t in enumerate(time_array):
                if t <= t1:
                    # Heating phase
                    temperature[j] = base_temp + heating_rate * t
                elif t <= t2:
                    # Hold phase with small fluctuations
                    temperature[j] = max_temp + np.random.normal(0, 2)
                else:
                    # Cooling phase
                    temperature[j] = max_temp + cooling_rate * (t - t2)
            
            # Add realistic noise
            temperature += np.random.normal(0, 0.5, len(temperature))
            
            profiles[f'profile_{i+1}'] = {
                'time': time_array.tolist(),
                'temperature': temperature.tolist(),
                'heating_rate': float(heating_rate * 60),  # °C/min
                'cooling_rate': float(cooling_rate * 60),  # °C/min
                'max_temperature': float(np.max(temperature)),
                'min_temperature': float(np.min(temperature))
            }
        
        return profiles
    
    def generate_stress_distributions(self, n_elements=5000, n_time_steps=100):
        """Generate stress distribution outputs"""
        print("💪 Generating stress distributions...")
        
        time_array = np.linspace(0, 3600, n_time_steps)
        element_ids = np.arange(1, n_elements + 1)
        
        stress_data = {}
        
        # Von Mises stress
        base_stress = np.random.lognormal(mean=15, sigma=1.5, size=n_elements)  # MPa
        von_mises = np.zeros((n_elements, n_time_steps))
        
        for i in range(n_time_steps):
            # Time-dependent stress evolution
            time_factor = 1 + 0.5 * np.sin(2 * np.pi * time_array[i] / 1800)
            thermal_factor = 1 + 0.3 * np.exp(-time_array[i] / 2000)
            noise = np.random.normal(1, 0.1, n_elements)
            
            von_mises[:, i] = base_stress * time_factor * thermal_factor * noise
        
        stress_data['von_mises'] = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'values': von_mises.tolist(),
            'units': 'MPa',
            'max_value': float(np.max(von_mises)),
            'min_value': float(np.min(von_mises))
        }
        
        # Principal stresses
        principal_1 = von_mises * np.random.uniform(1.1, 1.5, (n_elements, 1))
        principal_2 = von_mises * np.random.uniform(0.3, 0.8, (n_elements, 1))
        principal_3 = von_mises * np.random.uniform(-0.5, 0.2, (n_elements, 1))
        
        stress_data['principal'] = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'sigma_1': principal_1.tolist(),
            'sigma_2': principal_2.tolist(),
            'sigma_3': principal_3.tolist(),
            'units': 'MPa'
        }
        
        # Interfacial shear stress (for interface elements only)
        n_interface = int(n_elements * 0.15)
        interface_elements = np.random.choice(element_ids, n_interface, replace=False)
        shear_stress = np.random.lognormal(mean=2, sigma=0.8, size=(n_interface, n_time_steps))
        
        stress_data['interfacial_shear'] = {
            'element_ids': interface_elements.tolist(),
            'time': time_array.tolist(),
            'values': shear_stress.tolist(),
            'units': 'MPa',
            'critical_value': 25.0  # MPa
        }
        
        return stress_data
    
    def generate_strain_fields(self, n_elements=5000, n_time_steps=100):
        """Generate strain field data"""
        print("📏 Generating strain fields...")
        
        time_array = np.linspace(0, 3600, n_time_steps)
        element_ids = np.arange(1, n_elements + 1)
        
        strain_data = {}
        
        # Elastic strain
        elastic_strain = np.random.normal(0, 0.001, (n_elements, n_time_steps))
        for i in range(n_time_steps):
            elastic_strain[:, i] += 0.002 * np.sin(2 * np.pi * time_array[i] / 1200)
        
        strain_data['elastic'] = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'values': elastic_strain.tolist(),
            'units': 'dimensionless'
        }
        
        # Plastic strain (accumulated)
        plastic_strain = np.zeros((n_elements, n_time_steps))
        for i in range(1, n_time_steps):
            # Plastic strain accumulation
            increment = np.maximum(0, np.random.normal(0, 1e-6, n_elements))
            plastic_strain[:, i] = plastic_strain[:, i-1] + increment
        
        strain_data['plastic'] = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'values': plastic_strain.tolist(),
            'units': 'dimensionless'
        }
        
        # Creep strain (for polymer elements)
        polymer_elements = np.random.choice(element_ids, int(n_elements * 0.2), replace=False)
        creep_strain = np.zeros((len(polymer_elements), n_time_steps))
        
        for i in range(1, n_time_steps):
            dt = time_array[i] - time_array[i-1]
            creep_rate = 1e-8 * np.exp(-5000 / (298 + 50 * np.sin(2 * np.pi * time_array[i] / 1800)))
            creep_strain[:, i] = creep_strain[:, i-1] + creep_rate * dt
        
        strain_data['creep'] = {
            'element_ids': polymer_elements.tolist(),
            'time': time_array.tolist(),
            'values': creep_strain.tolist(),
            'units': 'dimensionless'
        }
        
        # Thermal strain
        thermal_expansion = 23e-6  # 1/K for aluminum
        temp_variation = 50 * np.sin(2 * np.pi * time_array / 1800)  # Temperature variation
        thermal_strain = thermal_expansion * temp_variation
        thermal_strain_field = np.tile(thermal_strain, (n_elements, 1))
        
        # Add spatial variation
        spatial_factor = np.random.normal(1, 0.2, (n_elements, 1))
        thermal_strain_field *= spatial_factor
        
        strain_data['thermal'] = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'values': thermal_strain_field.tolist(),
            'units': 'dimensionless',
            'expansion_coefficient': thermal_expansion
        }
        
        return strain_data
    
    def generate_damage_evolution(self, n_elements=5000, n_time_steps=100):
        """Generate damage variable (D) evolution over time"""
        print("💥 Generating damage evolution...")
        
        time_array = np.linspace(0, 3600, n_time_steps)
        element_ids = np.arange(1, n_elements + 1)
        
        # Initialize damage variable (0 = no damage, 1 = complete failure)
        damage = np.zeros((n_elements, n_time_steps))
        
        # Damage initiation threshold (random for each element)
        damage_threshold = np.random.lognormal(mean=3, sigma=0.5, size=n_elements)
        
        # Damage evolution parameters
        damage_rate = np.random.lognormal(mean=-8, sigma=1, size=n_elements)  # Very slow damage
        
        # Simulate stress-driven damage
        for i in range(1, n_time_steps):
            # Equivalent stress (simplified)
            equiv_stress = 20 + 30 * np.sin(2 * np.pi * time_array[i] / 1200) * np.random.lognormal(0, 0.3, n_elements)
            
            # Damage initiation
            initiated = (equiv_stress > damage_threshold) & (damage[:, i-1] == 0)
            damage[initiated, i] = 0.01  # Initial damage
            
            # Damage evolution for already damaged elements
            damaged = damage[:, i-1] > 0
            if np.any(damaged):
                dt = time_array[i] - time_array[i-1]
                stress_factor = np.maximum(1, equiv_stress[damaged] / damage_threshold[damaged])
                damage_increment = damage_rate[damaged] * stress_factor * dt
                damage[damaged, i] = np.minimum(1.0, damage[damaged, i-1] + damage_increment)
        
        # Identify critical elements (high damage)
        critical_elements = element_ids[np.max(damage, axis=1) > 0.8]
        
        damage_data = {
            'element_ids': element_ids.tolist(),
            'time': time_array.tolist(),
            'damage_variable': damage.tolist(),
            'critical_elements': critical_elements.tolist(),
            'damage_statistics': {
                'max_damage': float(np.max(damage)),
                'avg_damage': float(np.mean(damage[:, -1])),
                'damaged_elements_count': int(np.sum(damage[:, -1] > 0)),
                'failed_elements_count': int(np.sum(damage[:, -1] >= 1.0))
            }
        }
        
        return damage_data
    
    def generate_field_distributions(self, n_nodes=8000, n_time_steps=100):
        """Generate temperature and voltage distributions"""
        print("🌡️⚡ Generating field distributions...")
        
        time_array = np.linspace(0, 3600, n_time_steps)
        node_ids = np.arange(1, n_nodes + 1)
        
        # Generate node coordinates for spatial correlation
        x = np.random.uniform(-50, 50, n_nodes)
        y = np.random.uniform(-25, 25, n_nodes)
        z = np.random.uniform(0, 10, n_nodes)
        
        field_data = {}
        
        # Temperature distribution
        temperature = np.zeros((n_nodes, n_time_steps))
        base_temp = 25  # °C
        
        for i, t in enumerate(time_array):
            # Spatial temperature variation
            temp_gradient_x = 2 * x / 50  # 2°C gradient across x
            temp_gradient_z = 5 * z / 10  # 5°C gradient across z
            
            # Temporal variation
            temp_oscillation = 20 * np.sin(2 * np.pi * t / 1800)
            temp_decay = 15 * np.exp(-t / 2000)
            
            # Combine effects
            temperature[:, i] = (base_temp + temp_gradient_x + temp_gradient_z + 
                               temp_oscillation + temp_decay + 
                               np.random.normal(0, 1, n_nodes))
        
        field_data['temperature'] = {
            'node_ids': node_ids.tolist(),
            'coordinates': {
                'x': x.tolist(),
                'y': y.tolist(),
                'z': z.tolist()
            },
            'time': time_array.tolist(),
            'values': temperature.tolist(),
            'units': 'Celsius',
            'statistics': {
                'max_temp': float(np.max(temperature)),
                'min_temp': float(np.min(temperature)),
                'avg_temp': float(np.mean(temperature))
            }
        }
        
        # Voltage distribution
        voltage = np.zeros((n_nodes, n_time_steps))
        base_voltage = 3.7  # V
        
        for i, t in enumerate(time_array):
            # Spatial voltage variation (electrochemical gradients)
            voltage_gradient_x = 0.1 * np.sin(np.pi * x / 50)
            voltage_gradient_y = 0.05 * np.cos(np.pi * y / 25)
            
            # Temporal variation (charge/discharge cycles)
            voltage_cycle = 0.8 * np.sin(2 * np.pi * t / 1200)
            voltage_drift = -0.3 * t / 3600  # Voltage fade
            
            # Temperature coupling
            temp_coupling = 0.002 * (temperature[:, i] - 25)
            
            voltage[:, i] = (base_voltage + voltage_gradient_x + voltage_gradient_y + 
                           voltage_cycle + voltage_drift + temp_coupling + 
                           np.random.normal(0, 0.01, n_nodes))
        
        field_data['voltage'] = {
            'node_ids': node_ids.tolist(),
            'coordinates': {
                'x': x.tolist(),
                'y': y.tolist(),
                'z': z.tolist()
            },
            'time': time_array.tolist(),
            'values': voltage.tolist(),
            'units': 'Volts',
            'statistics': {
                'max_voltage': float(np.max(voltage)),
                'min_voltage': float(np.min(voltage)),
                'avg_voltage': float(np.mean(voltage))
            }
        }
        
        return field_data
    
    def generate_failure_predictions(self, n_elements=5000, n_time_steps=100):
        """Generate delamination and crack initiation predictions"""
        print("🔍 Generating failure predictions...")
        
        time_array = np.linspace(0, 3600, n_time_steps)
        element_ids = np.arange(1, n_elements + 1)
        
        failure_data = {}
        
        # Delamination prediction
        interface_elements = np.random.choice(element_ids, int(n_elements * 0.15), replace=False)
        n_interface = len(interface_elements)
        
        # Delamination criteria (Mode I and Mode II)
        mode_I_strength = 50e6  # Pa
        mode_II_strength = 35e6  # Pa
        
        delamination_risk = np.zeros((n_interface, n_time_steps))
        delamination_initiated = np.zeros((n_interface, n_time_steps), dtype=bool)
        
        for i, t in enumerate(time_array):
            # Normal and shear stresses at interfaces
            normal_stress = np.random.lognormal(mean=15, sigma=0.8, size=n_interface) * 1e6
            shear_stress = np.random.lognormal(mean=12, sigma=0.6, size=n_interface) * 1e6
            
            # Time-dependent stress amplification
            stress_factor = 1 + 0.5 * np.sin(2 * np.pi * t / 1200) + 0.2 * t / 3600
            normal_stress *= stress_factor
            shear_stress *= stress_factor
            
            # Mixed-mode delamination criterion
            mode_I_ratio = normal_stress / mode_I_strength
            mode_II_ratio = shear_stress / mode_II_strength
            
            # Quadratic interaction criterion
            delamination_risk[:, i] = mode_I_ratio**2 + mode_II_ratio**2
            
            # Delamination initiation
            if i > 0:
                newly_initiated = (delamination_risk[:, i] > 1.0) & ~delamination_initiated[:, i-1]
                delamination_initiated[:, i] = delamination_initiated[:, i-1] | newly_initiated
            else:
                delamination_initiated[:, i] = delamination_risk[:, i] > 1.0
        
        failure_data['delamination'] = {
            'interface_elements': interface_elements.tolist(),
            'time': time_array.tolist(),
            'risk_factor': delamination_risk.tolist(),
            'initiated': delamination_initiated.tolist(),
            'criteria': {
                'mode_I_strength': mode_I_strength,
                'mode_II_strength': mode_II_strength,
                'interaction_law': 'quadratic'
            },
            'statistics': {
                'max_risk': float(np.max(delamination_risk)),
                'initiated_count': int(np.sum(delamination_initiated[:, -1])),
                'initiation_time': []
            }
        }
        
        # Find initiation times
        for elem_idx in range(n_interface):
            initiation_indices = np.where(delamination_initiated[elem_idx, :])[0]
            if len(initiation_indices) > 0:
                initiation_time = time_array[initiation_indices[0]]
                failure_data['delamination']['statistics']['initiation_time'].append({
                    'element_id': int(interface_elements[elem_idx]),
                    'time': float(initiation_time)
                })
        
        # Crack initiation prediction
        # Select elements prone to cracking (high stress concentration)
        crack_prone_elements = np.random.choice(element_ids, int(n_elements * 0.1), replace=False)
        n_crack_prone = len(crack_prone_elements)
        
        # Fatigue crack initiation model
        stress_amplitude = np.random.lognormal(mean=4, sigma=0.5, size=n_crack_prone)  # Log scale
        stress_amplitude = 10**stress_amplitude  # Convert to Pa
        
        # Paris law parameters
        crack_threshold = 5e6  # Pa (threshold stress)
        fatigue_exponent = 3.5
        
        crack_initiation_cycles = np.zeros(n_crack_prone)
        crack_initiated = np.zeros((n_crack_prone, n_time_steps), dtype=bool)
        
        # Estimate cycles from time (assuming 1 Hz loading)
        cycles = time_array  # 1 cycle per second
        
        for elem_idx in range(n_crack_prone):
            if stress_amplitude[elem_idx] > crack_threshold:
                # Simplified crack initiation life
                delta_stress = stress_amplitude[elem_idx] - crack_threshold
                N_initiation = 1e6 / (delta_stress / 1e6)**fatigue_exponent
                crack_initiation_cycles[elem_idx] = N_initiation
                
                # Find when crack initiates
                initiation_indices = cycles >= N_initiation
                crack_initiated[elem_idx, initiation_indices] = True
        
        failure_data['crack_initiation'] = {
            'elements': crack_prone_elements.tolist(),
            'time': time_array.tolist(),
            'cycles': cycles.tolist(),
            'stress_amplitude': stress_amplitude.tolist(),
            'initiated': crack_initiated.tolist(),
            'initiation_cycles': crack_initiation_cycles.tolist(),
            'parameters': {
                'threshold_stress': crack_threshold,
                'fatigue_exponent': fatigue_exponent,
                'loading_frequency': 1.0  # Hz
            },
            'statistics': {
                'initiated_count': int(np.sum(crack_initiated[:, -1])),
                'avg_initiation_cycles': float(np.mean(crack_initiation_cycles[crack_initiation_cycles > 0]))
            }
        }
        
        return failure_data
    
    def save_dataset(self, dataset, output_dir="/workspace/fem_dataset"):
        """Save the complete dataset to files"""
        print(f"💾 Saving dataset to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each component
        for key, data in dataset.items():
            if key == 'mesh_data':
                # Save mesh data as CSV and JSON
                data['nodes'].to_csv(f"{output_dir}/nodes.csv", index=False)
                data['elements'].to_csv(f"{output_dir}/elements.csv", index=False)
                
                with open(f"{output_dir}/mesh_quality.json", 'w') as f:
                    json.dump(data['mesh_quality'], f, indent=2)
                    
            elif isinstance(data, dict):
                with open(f"{output_dir}/{key}.json", 'w') as f:
                    json.dump(data, f, indent=2)
        
        # Create summary report
        self.create_summary_report(dataset, output_dir)
        
        print(f"✅ Dataset saved successfully to {output_dir}")
        return output_dir
    
    def create_summary_report(self, dataset, output_dir):
        """Create a summary report of the generated dataset"""
        report = f"""
# FEM Simulation Dataset Summary
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
This synthetic dataset simulates multi-physics FEM analysis results typical of COMSOL/ABAQUS simulations.

## Input Parameters

### Mesh Data
- Total nodes: {len(dataset['mesh_data']['nodes'])}
- Total elements: {len(dataset['mesh_data']['elements'])}
- Element types: {', '.join(dataset['mesh_data']['elements']['element_type'].unique())}
- Element size range: {dataset['mesh_data']['mesh_quality']['min_element_size']:.2e} - {dataset['mesh_data']['mesh_quality']['max_element_size']:.2e}

### Boundary Conditions
- Temperature BC nodes: {len(dataset['boundary_conditions']['temperature']['node_ids'])}
- Displacement BC nodes: {len(dataset['boundary_conditions']['displacement']['node_ids'])}
- Voltage BC nodes: {len(dataset['boundary_conditions']['voltage']['node_ids'])}
- Heat flux BC nodes: {len(dataset['boundary_conditions']['heat_flux']['node_ids'])}
- Simulation time: {dataset['boundary_conditions']['time_parameters']['total_time']} seconds

### Material Models
- Number of materials: {len(dataset['material_models'])}
- Materials: {', '.join(dataset['material_models'].keys())}

### Thermal Profiles
- Number of profiles: {len(dataset['thermal_profiles'])}
- Heating rates: 1-10°C/min
- Cooling rates: 1-10°C/min

## Output Data

### Stress Distributions
- Von Mises stress range: {dataset['stress_distributions']['von_mises']['min_value']:.1f} - {dataset['stress_distributions']['von_mises']['max_value']:.1f} MPa
- Principal stresses: σ₁, σ₂, σ₃
- Interface elements with shear stress: {len(dataset['stress_distributions']['interfacial_shear']['element_ids'])}

### Strain Fields
- Elastic strain: All elements
- Plastic strain: All elements (accumulated)
- Creep strain: {len(dataset['strain_fields']['creep']['element_ids'])} polymer elements
- Thermal strain: All elements

### Damage Evolution
- Elements with damage: {dataset['damage_evolution']['damage_statistics']['damaged_elements_count']}
- Failed elements: {dataset['damage_evolution']['damage_statistics']['failed_elements_count']}
- Maximum damage: {dataset['damage_evolution']['damage_statistics']['max_damage']:.3f}

### Field Distributions
- Temperature range: {dataset['field_distributions']['temperature']['statistics']['min_temp']:.1f} - {dataset['field_distributions']['temperature']['statistics']['max_temp']:.1f} °C
- Voltage range: {dataset['field_distributions']['voltage']['statistics']['min_voltage']:.2f} - {dataset['field_distributions']['voltage']['statistics']['max_voltage']:.2f} V

### Failure Predictions
- Delamination initiated: {dataset['failure_predictions']['delamination']['statistics']['initiated_count']} interface elements
- Crack initiation: {dataset['failure_predictions']['crack_initiation']['statistics']['initiated_count']} elements

## File Structure
```
fem_dataset/
├── nodes.csv                    # Node coordinates
├── elements.csv                 # Element connectivity and properties
├── mesh_quality.json           # Mesh quality metrics
├── boundary_conditions.json    # All boundary conditions
├── material_models.json        # Material property definitions
├── thermal_profiles.json       # Transient thermal loading
├── stress_distributions.json   # Stress field results
├── strain_fields.json          # Strain field results
├── damage_evolution.json       # Damage variable evolution
├── field_distributions.json    # Temperature and voltage fields
├── failure_predictions.json    # Delamination and crack predictions
└── summary_report.md           # This summary
```

## Usage Notes
- All stress values are in Pa (Pascals) unless otherwise specified
- Time arrays are in seconds
- Temperature values are in Celsius
- Voltage values are in Volts
- Damage variable ranges from 0 (no damage) to 1 (complete failure)
- Element and node IDs start from 1

## Data Validation
This synthetic dataset includes realistic:
- Material property ranges
- Stress-strain relationships
- Thermal coupling effects
- Damage evolution patterns
- Failure mode interactions

The data can be used for:
- Machine learning model training
- Algorithm validation
- Visualization development
- Educational purposes
"""
        
        with open(f"{output_dir}/summary_report.md", 'w') as f:
            f.write(report)
    
    def generate_complete_dataset(self):
        """Generate the complete FEM simulation dataset"""
        print("🚀 Starting FEM dataset generation...")
        print("=" * 60)
        
        dataset = {}
        
        # Generate all components
        dataset['mesh_data'] = self.generate_mesh_data()
        dataset['boundary_conditions'] = self.generate_boundary_conditions()
        dataset['material_models'] = self.generate_material_models()
        dataset['thermal_profiles'] = self.generate_thermal_profiles()
        dataset['stress_distributions'] = self.generate_stress_distributions()
        dataset['strain_fields'] = self.generate_strain_fields()
        dataset['damage_evolution'] = self.generate_damage_evolution()
        dataset['field_distributions'] = self.generate_field_distributions()
        dataset['failure_predictions'] = self.generate_failure_predictions()
        
        # Save dataset
        output_dir = self.save_dataset(dataset)
        
        print("=" * 60)
        print("🎉 FEM dataset generation completed successfully!")
        print(f"📁 Output directory: {output_dir}")
        
        return dataset, output_dir

def main():
    """Main function to generate the FEM dataset"""
    generator = FEMDatasetGenerator(seed=42)
    dataset, output_dir = generator.generate_complete_dataset()
    
    print("\n📊 Dataset Statistics:")
    print(f"   • Nodes: {len(dataset['mesh_data']['nodes']):,}")
    print(f"   • Elements: {len(dataset['mesh_data']['elements']):,}")
    print(f"   • Materials: {len(dataset['material_models'])}")
    print(f"   • Time steps: {len(dataset['boundary_conditions']['time_parameters'])}")
    print(f"   • Thermal profiles: {len(dataset['thermal_profiles'])}")
    
    return dataset, output_dir

if __name__ == "__main__":
    dataset, output_dir = main()