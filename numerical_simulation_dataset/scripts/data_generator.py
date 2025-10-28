"""
Multi-physics FEM Simulation Data Generator
Generates realistic numerical simulation data for battery cells
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import h5py
import json

@dataclass
class MeshParameters:
    """Mesh configuration parameters"""
    element_size: float  # mm
    element_type: str  # 'hex8', 'tet4', 'tet10', 'hex20'
    interface_refinement: float  # refinement factor at interfaces
    num_elements: int
    num_nodes: int
    
class MaterialModel:
    """Advanced material model with temperature and rate dependency"""
    
    def __init__(self):
        # Baseline properties at 25°C
        self.E_ref = 70000  # Young's modulus (MPa)
        self.nu = 0.33  # Poisson's ratio
        self.alpha = 23e-6  # Thermal expansion (1/°C)
        self.k_thermal = 200  # Thermal conductivity (W/m·K)
        self.cp = 900  # Specific heat (J/kg·K)
        self.rho = 2700  # Density (kg/m³)
        
        # Plastic parameters
        self.yield_strength = 250  # MPa
        self.hardening_modulus = 1000  # MPa
        
        # Creep parameters (Norton law)
        self.A_creep = 1e-20  # Creep coefficient
        self.n_creep = 3.5  # Creep exponent
        self.Q_creep = 150000  # Activation energy (J/mol)
        
        # Electrochemical
        self.conductivity = 1e-3  # S/m
        self.diffusivity = 1e-10  # m²/s
        
    def get_temperature_dependent_E(self, T):
        """Temperature-dependent Young's modulus"""
        return self.E_ref * (1 - 0.0004 * (T - 25))
    
    def get_creep_rate(self, stress, T):
        """Norton-Bailey creep law"""
        R = 8.314  # Gas constant
        T_kelvin = T + 273.15
        return self.A_creep * (stress ** self.n_creep) * np.exp(-self.Q_creep / (R * T_kelvin))

class SimulationDataGenerator:
    """Generate realistic FEM simulation data"""
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        self.material = MaterialModel()
        
    def generate_mesh_data(self, refinement_level: int = 2) -> MeshParameters:
        """Generate mesh parameters based on refinement level"""
        base_size = 1.0  # mm
        element_sizes = {
            1: base_size * 2,    # Coarse
            2: base_size,        # Medium
            3: base_size * 0.5,  # Fine
            4: base_size * 0.25  # Very fine
        }
        
        element_types = ['hex8', 'hex20', 'tet4', 'tet10']
        
        mesh = MeshParameters(
            element_size=element_sizes.get(refinement_level, base_size),
            element_type=np.random.choice(element_types),
            interface_refinement=np.random.uniform(2, 5),
            num_elements=int(10000 * (4 ** (refinement_level - 1))),
            num_nodes=int(12000 * (4 ** (refinement_level - 1)))
        )
        return mesh
    
    def generate_thermal_profile(self, 
                                heating_rate: float,
                                T_initial: float = 25,
                                T_max: float = 85,
                                hold_time: float = 3600,
                                cooling_rate: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate transient thermal profile"""
        if cooling_rate is None:
            cooling_rate = heating_rate * 0.5
            
        # Time points
        t_heat = (T_max - T_initial) / (heating_rate / 60)  # Convert to seconds
        t_hold = hold_time
        t_cool = (T_max - T_initial) / (cooling_rate / 60)
        
        # Create time array
        t1 = np.linspace(0, t_heat, 50)
        t2 = np.linspace(t_heat, t_heat + t_hold, 30)
        t3 = np.linspace(t_heat + t_hold, t_heat + t_hold + t_cool, 50)
        
        # Create temperature array
        T1 = T_initial + (heating_rate / 60) * t1
        T2 = np.ones_like(t2) * T_max
        T3 = T_max - (cooling_rate / 60) * (t3 - (t_heat + t_hold))
        
        time = np.concatenate([t1, t2, t3])
        temperature = np.concatenate([T1, T2, T3])
        
        # Add noise
        temperature += np.random.normal(0, 0.5, len(temperature))
        
        return time, temperature
    
    def generate_stress_field(self,
                            nx: int = 50,
                            ny: int = 50,
                            nz: int = 20,
                            load_type: str = 'thermal',
                            time_steps: int = 100) -> Dict:
        """Generate 3D stress field data"""
        
        # Create spatial grid
        x = np.linspace(0, 100, nx)  # mm
        y = np.linspace(0, 100, ny)  # mm
        z = np.linspace(0, 10, nz)   # mm
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        stress_data = {}
        
        for t in range(time_steps):
            # Time-dependent loading factor
            load_factor = np.sin(np.pi * t / time_steps) + 0.5
            
            if load_type == 'thermal':
                # Thermal stress pattern
                T_field = 25 + 60 * load_factor * np.exp(-((X-50)**2 + (Y-50)**2) / 1000)
                thermal_strain = self.material.alpha * (T_field - 25)
                
                # Von Mises stress (simplified)
                sigma_vm = self.material.get_temperature_dependent_E(T_field.mean()) * thermal_strain
                sigma_vm = np.abs(sigma_vm) * (1 + 0.3 * np.random.randn(*X.shape))
                
                # Principal stresses
                sigma_1 = sigma_vm * 1.2 * (1 + 0.1 * np.sin(X/10))
                sigma_2 = sigma_vm * 0.8 * (1 + 0.1 * np.cos(Y/10))
                sigma_3 = sigma_vm * 0.4 * (1 - 0.1 * Z/10)
                
            elif load_type == 'mechanical':
                # Mechanical loading pattern
                sigma_vm = 100 * load_factor * (1 + 0.5 * Z/10)
                sigma_vm *= (1 + 0.2 * np.sin(X/20) * np.cos(Y/20))
                
                sigma_1 = sigma_vm * 1.5
                sigma_2 = sigma_vm * 0.5
                sigma_3 = sigma_vm * 0.2
                
            else:  # coupled
                # Combined loading
                T_field = 25 + 40 * load_factor
                thermal_stress = self.material.E_ref * self.material.alpha * (T_field - 25)
                mechanical_stress = 50 * load_factor * (1 + Z/10)
                
                sigma_vm = np.sqrt(thermal_stress**2 + mechanical_stress**2)
                sigma_1 = sigma_vm * 1.3
                sigma_2 = sigma_vm * 0.7
                sigma_3 = sigma_vm * 0.3
            
            # Add spatial smoothing for realism
            sigma_vm = gaussian_filter(sigma_vm, sigma=1.5)
            sigma_1 = gaussian_filter(sigma_1, sigma=1.5)
            sigma_2 = gaussian_filter(sigma_2, sigma=1.5)
            sigma_3 = gaussian_filter(sigma_3, sigma=1.5)
            
            # Interfacial shear stress
            tau_xy = 0.3 * sigma_vm * np.sin(X/15) * np.cos(Y/15)
            tau_xz = 0.2 * sigma_vm * np.exp(-Z/5)
            tau_yz = 0.25 * sigma_vm * np.sin(Y/20)
            
            stress_data[f't_{t}'] = {
                'von_mises': sigma_vm,
                'sigma_1': sigma_1,
                'sigma_2': sigma_2,
                'sigma_3': sigma_3,
                'tau_xy': tau_xy,
                'tau_xz': tau_xz,
                'tau_yz': tau_yz,
                'coordinates': {'x': X, 'y': Y, 'z': Z}
            }
        
        return stress_data
    
    def generate_strain_field(self, stress_data: Dict) -> Dict:
        """Generate strain fields from stress data"""
        strain_data = {}
        
        for time_key, stress in stress_data.items():
            E = self.material.E_ref
            nu = self.material.nu
            
            # Elastic strains (simplified Hooke's law)
            eps_elastic_1 = (stress['sigma_1'] - nu * (stress['sigma_2'] + stress['sigma_3'])) / E
            eps_elastic_2 = (stress['sigma_2'] - nu * (stress['sigma_1'] + stress['sigma_3'])) / E
            eps_elastic_3 = (stress['sigma_3'] - nu * (stress['sigma_1'] + stress['sigma_2'])) / E
            
            # Plastic strains (simplified J2 plasticity)
            sigma_vm = stress['von_mises']
            eps_plastic = np.zeros_like(sigma_vm)
            yield_mask = sigma_vm > self.material.yield_strength
            eps_plastic[yield_mask] = (sigma_vm[yield_mask] - self.material.yield_strength) / self.material.hardening_modulus
            
            # Creep strains (time-dependent)
            time_factor = float(time_key.split('_')[1]) / 100  # Normalized time
            eps_creep = self.material.get_creep_rate(sigma_vm.mean(), 60) * time_factor * 3600
            eps_creep = eps_creep * np.ones_like(sigma_vm) * (1 + 0.2 * np.random.randn(*sigma_vm.shape))
            
            # Thermal strains
            T_variation = 25 + 35 * time_factor  # Simplified temperature
            eps_thermal = self.material.alpha * (T_variation - 25) * np.ones_like(sigma_vm)
            
            # Total strain
            eps_total = np.sqrt(eps_elastic_1**2 + eps_elastic_2**2 + eps_elastic_3**2) + eps_plastic + eps_creep + np.abs(eps_thermal)
            
            strain_data[time_key] = {
                'elastic_1': eps_elastic_1,
                'elastic_2': eps_elastic_2,
                'elastic_3': eps_elastic_3,
                'plastic': eps_plastic,
                'creep': eps_creep,
                'thermal': eps_thermal,
                'total': eps_total
            }
        
        return strain_data
    
    def generate_damage_evolution(self, 
                                stress_data: Dict,
                                damage_model: str = 'lemaitre') -> Dict:
        """Generate damage variable evolution"""
        damage_data = {}
        D_prev = None
        
        for idx, (time_key, stress) in enumerate(stress_data.items()):
            sigma_vm = stress['von_mises']
            
            if damage_model == 'lemaitre':
                # Lemaitre damage model
                S = 1.0  # Damage strength parameter (MPa)
                s = 1.0  # Damage exponent
                eps_threshold = 0.001  # Damage threshold strain
                
                # Simplified damage rate
                eps_eq = sigma_vm / self.material.E_ref  # Equivalent strain
                
                if D_prev is None:
                    D = np.zeros_like(sigma_vm)
                else:
                    D = D_prev.copy()
                
                # Update damage where strain exceeds threshold
                damage_mask = eps_eq > eps_threshold
                if np.any(damage_mask):
                    Y = 0.5 * sigma_vm**2 / (self.material.E_ref * (1 - D)**2)  # Energy release rate
                    dD = (Y / S)**s * 0.01  # Damage increment
                    D[damage_mask] = np.minimum(D[damage_mask] + dD[damage_mask], 0.99)
                
                # Add some spatial correlation
                D = gaussian_filter(D, sigma=1.0)
                
            elif damage_model == 'cohesive':
                # Cohesive zone model for interfaces
                if D_prev is None:
                    D = np.zeros_like(sigma_vm)
                else:
                    D = D_prev.copy()
                
                # Interface damage based on shear stress
                tau_max = 50  # Maximum interface strength (MPa)
                tau = np.sqrt(stress['tau_xy']**2 + stress['tau_xz']**2 + stress['tau_yz']**2)
                
                D_interface = np.minimum(tau / tau_max, 1.0)
                D = np.maximum(D, D_interface * 0.8)
                
            else:  # fatigue
                # Fatigue damage accumulation
                if D_prev is None:
                    D = np.zeros_like(sigma_vm)
                else:
                    D = D_prev.copy()
                
                # Miner's rule simplified
                N_f = 1e6 * (250 / np.maximum(sigma_vm, 1))**3  # S-N curve
                dD = 1.0 / N_f
                D = np.minimum(D + dD, 0.99)
            
            damage_data[time_key] = {
                'damage': D,
                'damage_rate': dD if 'dD' in locals() else np.zeros_like(D),
                'critical_elements': np.sum(D > 0.9),
                'mean_damage': np.mean(D),
                'max_damage': np.max(D)
            }
            
            D_prev = D
        
        return damage_data
    
    def generate_temperature_distribution(self,
                                        nx: int = 50,
                                        ny: int = 50,
                                        nz: int = 20,
                                        time_steps: int = 100) -> Dict:
        """Generate 3D temperature field with heat source"""
        
        x = np.linspace(0, 100, nx)
        y = np.linspace(0, 100, ny)
        z = np.linspace(0, 10, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        temp_data = {}
        T_prev = 25 * np.ones((nx, ny, nz))  # Initial temperature
        
        for t in range(time_steps):
            # Heat source (e.g., from electrical heating)
            heat_source = 5 * np.exp(-((X-50)**2 + (Y-50)**2 + (Z-5)**2) / 500)
            heat_source *= (1 + 0.5 * np.sin(2 * np.pi * t / time_steps))
            
            # Simple heat diffusion
            dt = 1.0  # Time step
            alpha = self.material.k_thermal / (self.material.rho * self.material.cp)
            
            # Finite difference (simplified)
            T_new = T_prev + dt * alpha * 0.1 * gaussian_filter(T_prev, sigma=1.5)
            T_new += dt * heat_source / (self.material.rho * self.material.cp)
            
            # Boundary conditions
            T_new[0, :, :] = 25  # Fixed temperature at x=0
            T_new[-1, :, :] = 25  # Fixed temperature at x=L
            T_new[:, :, 0] = 30  # Slight heating at bottom
            
            # Add convection cooling at top surface
            h_conv = 10  # W/m²K
            T_ambient = 25
            T_new[:, :, -1] -= dt * h_conv * (T_new[:, :, -1] - T_ambient) / (self.material.rho * self.material.cp * 0.1)
            
            temp_data[f't_{t}'] = {
                'temperature': T_new.copy(),
                'heat_flux_x': -self.material.k_thermal * np.gradient(T_new, axis=0),
                'heat_flux_y': -self.material.k_thermal * np.gradient(T_new, axis=1),
                'heat_flux_z': -self.material.k_thermal * np.gradient(T_new, axis=2),
                'max_temp': np.max(T_new),
                'min_temp': np.min(T_new),
                'mean_temp': np.mean(T_new)
            }
            
            T_prev = T_new
        
        return temp_data
    
    def generate_voltage_distribution(self,
                                    nx: int = 50,
                                    ny: int = 50,
                                    nz: int = 20,
                                    time_steps: int = 100) -> Dict:
        """Generate voltage/electric potential distribution"""
        
        x = np.linspace(0, 100, nx)
        y = np.linspace(0, 100, ny)
        z = np.linspace(0, 10, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        voltage_data = {}
        
        for t in range(time_steps):
            # Time-dependent voltage profile (charging/discharging)
            V_applied = 4.2 * np.sin(np.pi * t / time_steps) + 3.7
            
            # Voltage distribution (simplified Laplace equation solution)
            V = V_applied * (1 - Z / 10)  # Linear drop through thickness
            
            # Add spatial variation
            V += 0.1 * np.sin(X / 20) * np.cos(Y / 20) * (1 - Z / 10)
            
            # Current density (Ohm's law)
            J_x = -self.material.conductivity * np.gradient(V, axis=0)
            J_y = -self.material.conductivity * np.gradient(V, axis=1)
            J_z = -self.material.conductivity * np.gradient(V, axis=2)
            
            # Joule heating
            joule_heat = (J_x**2 + J_y**2 + J_z**2) / self.material.conductivity
            
            voltage_data[f't_{t}'] = {
                'voltage': V,
                'current_density_x': J_x,
                'current_density_y': J_y,
                'current_density_z': J_z,
                'joule_heating': joule_heat,
                'total_current': np.sum(np.sqrt(J_x**2 + J_y**2 + J_z**2)),
                'max_voltage': np.max(V),
                'min_voltage': np.min(V)
            }
        
        return voltage_data
    
    def generate_failure_predictions(self,
                                    damage_data: Dict,
                                    stress_data: Dict) -> Dict:
        """Generate delamination and crack predictions"""
        
        failure_data = {}
        
        for time_key in damage_data.keys():
            D = damage_data[time_key]['damage']
            sigma_vm = stress_data[time_key]['von_mises']
            tau_xz = stress_data[time_key]['tau_xz']
            
            # Delamination criterion (interface failure)
            G_c = 0.5  # Critical energy release rate (N/mm)
            a = 1.0  # Crack length (mm)
            K_I = sigma_vm.mean() * np.sqrt(np.pi * a)  # Mode I stress intensity
            G_I = K_I**2 / self.material.E_ref  # Energy release rate
            
            delamination_risk = np.minimum(G_I / G_c, 1.0) * np.ones_like(D)
            
            # Crack initiation sites (where damage is high)
            crack_probability = D * (sigma_vm / self.material.yield_strength)
            crack_sites = crack_probability > 0.8
            
            # Crack propagation direction (perpendicular to max principal stress)
            crack_angle = np.arctan2(stress_data[time_key]['sigma_2'], 
                                    stress_data[time_key]['sigma_1'])
            
            failure_data[time_key] = {
                'delamination_risk': delamination_risk,
                'crack_probability': crack_probability,
                'crack_initiation_sites': crack_sites,
                'crack_angle': crack_angle,
                'num_crack_sites': np.sum(crack_sites),
                'max_delamination_risk': np.max(delamination_risk),
                'failure_index': np.maximum(delamination_risk, crack_probability)
            }
        
        return failure_data

def save_simulation_data(data: Dict, filepath: str, format: str = 'hdf5'):
    """Save simulation data to file"""
    
    if format == 'hdf5':
        with h5py.File(filepath, 'w') as f:
            def save_dict_to_hdf5(group, data_dict):
                for key, value in data_dict.items():
                    if isinstance(value, dict):
                        subgroup = group.create_group(key)
                        save_dict_to_hdf5(subgroup, value)
                    elif isinstance(value, np.ndarray):
                        group.create_dataset(key, data=value, compression='gzip')
                    else:
                        group.attrs[key] = value
            
            save_dict_to_hdf5(f, data)
            
    elif format == 'npz':
        # Flatten nested dictionaries for npz format
        flat_data = {}
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                new_key = f"{prefix}_{key}" if prefix else key
                if isinstance(value, dict):
                    flatten_dict(value, new_key)
                elif isinstance(value, np.ndarray):
                    flat_data[new_key] = value
                    
        flatten_dict(data)
        np.savez_compressed(filepath, **flat_data)

def generate_simulation_metadata(sim_id: str, params: Dict) -> Dict:
    """Generate metadata for simulation"""
    
    metadata = {
        'simulation_id': sim_id,
        'timestamp': pd.Timestamp.now().isoformat(),
        'software': 'COMSOL/ABAQUS equivalent',
        'solver': 'Nonlinear transient coupled',
        'convergence_tolerance': 1e-6,
        'time_stepping': 'Adaptive',
        'parameters': params,
        'units': {
            'stress': 'MPa',
            'strain': 'mm/mm',
            'temperature': 'Celsius',
            'voltage': 'V',
            'time': 'seconds',
            'length': 'mm',
            'damage': 'dimensionless (0-1)'
        }
    }
    
    return metadata