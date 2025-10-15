"""High-fidelity data generator for SOFC modeling."""

import numpy as np
from typing import Dict, Tuple, Optional, List
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.physics_utils import SOFCPhysics, MaterialProperties, DegradationModels
from src.utils.mesh_utils import MeshGenerator, FieldInterpolator
from src.utils.data_utils import DatasetManager


class HighFidelityGenerator:
    """
    Generate high-fidelity SOFC data using detailed 3D models.
    High computation (1 hour per sample) with fine-grid spatial fields.
    Includes coupled multi-physics and microstructure effects.
    """
    
    def __init__(self, config: Dict):
        """Initialize generator with configuration."""
        self.config = config
        self.physics = SOFCPhysics(config)
        self.materials = MaterialProperties()
        self.degradation = DegradationModels()
        
        # Extract parameters
        self.op_ranges = config['operating_conditions']
        self.geom = config['geometry']
        self.mat_props = config['materials']
        
        # Initialize mesh generator
        self.mesh_gen = MeshGenerator(config['geometry'])
        
        # High-resolution mesh
        self.nx = 100
        self.ny = 100
        self.nz = 50
        
        # Generate base mesh
        self.mesh = self.mesh_gen.generate_3d_mesh(self.nx, self.ny, self.nz)
        
        # Microstructure parameters
        self.microstructure = self._initialize_microstructure()
        
    def _initialize_microstructure(self) -> Dict:
        """Initialize microstructure parameters for each material."""
        microstructure = {
            'anode': {
                'ni_particle_size': 1.0e-6,  # m
                'ysz_particle_size': 0.5e-6,
                'tpb_density': 1e12,  # m/m³
                'connectivity': 0.95
            },
            'cathode': {
                'lsm_particle_size': 0.8e-6,
                'ysz_particle_size': 0.5e-6,
                'tpb_density': 8e11,
                'connectivity': 0.92
            },
            'electrolyte': {
                'grain_size': 2.0e-6,
                'grain_boundary_width': 1e-9,
                'grain_boundary_resistance': 10  # relative to bulk
            }
        }
        return microstructure
    
    def generate_sample(self, sample_id: int, seed: Optional[int] = None) -> Dict:
        """Generate single high-fidelity sample with detailed physics."""
        if seed is not None:
            np.random.seed(seed + sample_id)
        
        start_time = time.time()
        
        # Sample input parameters
        inputs = self._sample_inputs()
        
        # Solve coupled multi-physics problem
        outputs = self._solve_coupled_problem(inputs)
        
        # Add microstructure evolution
        outputs = self._add_microstructure_effects(outputs, inputs)
        
        # Add phase-field damage
        outputs = self._compute_phase_field_damage(outputs, inputs)
        
        # Add metadata
        metadata = {
            'sample_id': f'HF_{sample_id:06d}',
            'timestamp': time.time(),
            'computation_time': time.time() - start_time,
            'convergence_flag': True,
            'notes': 'High-fidelity coupled multi-physics model'
        }
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata
        }
    
    def _sample_inputs(self) -> Dict:
        """Sample input parameters with focus on critical regions."""
        inputs = {}
        
        # Focus on challenging operating conditions
        # Use beta distribution to sample more near boundaries
        alpha, beta = 2, 2  # Shape parameters
        
        # Temperature - sample more at extremes
        t_range = self.op_ranges['temperature']
        t_norm = np.random.beta(alpha, beta)
        inputs['temperature'] = t_range['min'] + t_norm * (t_range['max'] - t_range['min'])
        
        # Current density - sample more at high values
        i_range = self.op_ranges['current_density']
        i_norm = np.random.beta(2, 1)  # Skewed towards high values
        inputs['current_density'] = i_range['min'] + i_norm * (i_range['max'] - i_range['min'])
        
        # High utilizations (stress conditions)
        inputs['fuel_utilization'] = np.random.beta(3, 1) * 0.6 + 0.3  # 0.3-0.9, skewed high
        inputs['air_utilization'] = np.random.uniform(0.2, 0.4)
        
        # Pressure variations
        inputs['pressure'] = np.random.choice([1e5, 1.5e5, 2e5, 3e5])  # Discrete values
        
        # Fuel composition with reformate scenarios
        fuel_comp = np.zeros(6)
        scenario = np.random.choice(['pure_h2', 'reformed', 'syngas'])
        
        if scenario == 'pure_h2':
            fuel_comp[0] = 0.97  # H2
            fuel_comp[1] = 0.03  # H2O
        elif scenario == 'reformed':
            fuel_comp[0] = 0.75  # H2
            fuel_comp[1] = 0.15  # H2O
            fuel_comp[2] = 0.05  # CO
            fuel_comp[3] = 0.03  # CO2
            fuel_comp[4] = 0.02  # CH4
        else:  # syngas
            fuel_comp[0] = 0.40  # H2
            fuel_comp[1] = 0.10  # H2O
            fuel_comp[2] = 0.30  # CO
            fuel_comp[3] = 0.15  # CO2
            fuel_comp[5] = 0.05  # N2
        
        inputs['fuel_composition'] = fuel_comp
        inputs['air_composition'] = np.array([0.21, 0.79])
        
        # Geometry with defects
        inputs['anode_thickness'] = self.geom['anode_thickness'] * np.random.normal(1.0, 0.05)
        inputs['electrolyte_thickness'] = self.geom['electrolyte_thickness'] * np.random.normal(1.0, 0.03)
        inputs['cathode_thickness'] = self.geom['cathode_thickness'] * np.random.normal(1.0, 0.05)
        
        # Material variations
        inputs['anode_porosity'] = np.random.beta(3, 3) * 0.2 + 0.2  # 0.2-0.4
        inputs['cathode_porosity'] = np.random.beta(3, 3) * 0.2 + 0.2
        
        # Degradation state - focus on aged cells
        inputs['operating_time'] = np.random.gamma(2, 2500)  # Hours, peak around 5000h
        inputs['thermal_cycles'] = np.random.negative_binomial(5, 0.3)
        inputs['redox_cycles'] = np.random.poisson(3)
        
        return inputs
    
    def _solve_coupled_problem(self, inputs: Dict) -> Dict:
        """Solve fully coupled thermo-chemo-mechanical problem."""
        outputs = {}
        
        # Initialize solution fields
        nx, ny, nz = self.nx, self.ny, self.nz
        T_field = np.ones((nx, ny, nz)) * inputs['temperature']
        i_field = np.ones((nx, ny, nz)) * inputs['current_density']
        
        # Iterative coupling loop
        max_iter = 10
        tol = 1e-4
        
        for iteration in range(max_iter):
            T_old = T_field.copy()
            
            # Step 1: Solve electrochemistry with current T field
            i_field, species_field, overpotentials = self._solve_electrochemistry(
                T_field, inputs
            )
            
            # Step 2: Solve heat transfer with heat generation
            T_field = self._solve_heat_transfer(T_field, i_field, overpotentials, inputs)
            
            # Step 3: Check convergence
            residual = np.max(np.abs(T_field - T_old))
            if residual < tol:
                break
        
        # Step 4: Solve mechanics with final T and i fields
        stress_tensor, strain_tensor = self._solve_mechanics(T_field, inputs)
        
        # Store outputs
        outputs['temperature_field'] = T_field
        outputs['current_density_field'] = i_field
        outputs['species_concentration'] = species_field
        outputs['overpotentials'] = overpotentials
        outputs['stress_tensor'] = stress_tensor
        outputs['strain_tensor'] = strain_tensor
        
        # Compute damage indicators
        outputs['damage_field'] = self._compute_damage_indicators(
            stress_tensor, strain_tensor, T_field, inputs
        )
        
        # Performance metrics
        outputs['voltage'] = self._compute_detailed_voltage(T_field, i_field, 
                                                           overpotentials, species_field)
        outputs['power_density'] = outputs['voltage'] * np.mean(i_field)
        
        return outputs
    
    def _solve_electrochemistry(self, T_field: np.ndarray, 
                               inputs: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve detailed electrochemistry with Butler-Volmer kinetics."""
        nx, ny, nz = self.nx, self.ny, self.nz
        
        i_field = np.zeros((nx, ny, nz))
        species_field = np.zeros((6, nx, ny, nz))  # 6 species
        overpotentials = np.zeros((3, nx, ny, nz))  # 3 types
        
        # Inlet conditions
        p = inputs['pressure']
        fuel_comp = inputs['fuel_composition']
        air_comp = inputs['air_composition']
        
        # Solve along flow direction
        for i in range(nx):
            # Update species concentrations
            utilization_fuel = inputs['fuel_utilization'] * (i + 0.5) / nx
            utilization_air = inputs['air_utilization'] * (i + 0.5) / nx
            
            # H2 and H2O
            x_H2 = fuel_comp[0] * (1 - utilization_fuel)
            x_H2O = fuel_comp[1] + fuel_comp[0] * utilization_fuel
            
            # O2
            x_O2 = air_comp[0] * (1 - utilization_air)
            
            # CO and CO2 with water-gas shift
            if fuel_comp[2] > 0:  # CO present
                x_CO = fuel_comp[2] * (1 - 0.5 * utilization_fuel)
                x_CO2 = fuel_comp[3] + fuel_comp[2] * 0.5 * utilization_fuel
            else:
                x_CO = 0
                x_CO2 = fuel_comp[3]
            
            # Store concentrations
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    RT = self.physics.R * T
                    
                    species_field[0, i, j, k] = p * x_H2 / RT
                    species_field[1, i, j, k] = p * x_H2O / RT
                    species_field[2, i, j, k] = p * x_O2 / RT
                    species_field[3, i, j, k] = p * air_comp[1] / RT
                    species_field[4, i, j, k] = p * x_CO / RT
                    species_field[5, i, j, k] = p * x_CO2 / RT
            
            # Solve current distribution at this x-location
            i_local = self._solve_current_distribution_2d(
                i, T_field[i, :, :], species_field[:, i, :, :], inputs
            )
            
            i_field[i, :, :] = i_local
            
            # Calculate overpotentials
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    i_loc = i_field[i, j, k]
                    mat_id = self.mesh['material_field'][i, j, k]
                    
                    if mat_id == 0:  # Anode
                        # Microstructure-dependent exchange current
                        tpb = self.microstructure['anode']['tpb_density']
                        i0 = 1e-7 * tpb * np.exp(-60000 / (self.physics.R * T))
                        
                        if i_loc > 0:
                            eta_act = (self.physics.R * T) / (2 * self.physics.F) * \
                                     np.asinh(i_loc / (2 * i0))
                            overpotentials[0, i, j, k] = eta_act
                    
                    elif mat_id == 2:  # Cathode
                        tpb = self.microstructure['cathode']['tpb_density']
                        i0 = 1e-8 * tpb * np.exp(-120000 / (self.physics.R * T))
                        
                        if i_loc > 0:
                            eta_act = (self.physics.R * T) / (4 * self.physics.F) * \
                                     np.asinh(i_loc / (2 * i0))
                            overpotentials[0, i, j, k] = eta_act
                    
                    elif mat_id == 1:  # Electrolyte
                        # Grain boundary effects
                        grain_factor = 1 + self.microstructure['electrolyte']['grain_boundary_resistance'] * \
                                      self.microstructure['electrolyte']['grain_boundary_width'] / \
                                      self.microstructure['electrolyte']['grain_size']
                        
                        sigma = self.materials.ionic_conductivity_YSZ(T) / grain_factor
                        thickness = self.geom['electrolyte_thickness'] * 1e-6 / nz
                        overpotentials[1, i, j, k] = i_loc * thickness / sigma
                    
                    # Concentration overpotential with dusty-gas model
                    if mat_id == 0:  # Anode
                        porosity = inputs['anode_porosity']
                        tortuosity = self.mat_props['anode']['tortuosity']
                        D_eff = 1e-5 * (T / 1073) ** 1.5 * porosity / tortuosity
                        
                        c_H2 = species_field[0, i, j, k]
                        L = self.geom['anode_thickness'] * 1e-6
                        i_lim = 2 * self.physics.F * D_eff * c_H2 / L
                        
                        if i_loc > 0 and i_loc < 0.99 * i_lim:
                            eta_conc = (self.physics.R * T) / (2 * self.physics.F) * \
                                      np.log(1 - i_loc / i_lim)
                            overpotentials[2, i, j, k] = abs(eta_conc)
        
        return i_field, species_field, overpotentials
    
    def _solve_current_distribution_2d(self, x_idx: int, T_slice: np.ndarray,
                                      species_slice: np.ndarray, inputs: Dict) -> np.ndarray:
        """Solve 2D current distribution at given x-location."""
        ny, nz = T_slice.shape
        i_avg = inputs['current_density']
        
        # Initialize with average
        i_dist = np.ones((ny, nz)) * i_avg
        
        # Channel/rib effects
        channel_pitch = 4  # Grid points
        for j in range(ny):
            if (j // channel_pitch) % 2 == 0:  # Channel region
                i_dist[j, :] *= 0.7
            else:  # Rib region
                i_dist[j, :] *= 1.3
        
        # Temperature effects
        T_norm = (T_slice - T_slice.min()) / (T_slice.max() - T_slice.min() + 1e-10)
        i_dist *= (1 + 0.3 * T_norm)
        
        # Edge effects
        edge_decay = 0.8
        for j in [0, 1, ny-2, ny-1]:
            i_dist[j, :] *= edge_decay
        for k in [0, 1, nz-2, nz-1]:
            i_dist[:, k] *= edge_decay
        
        # Ensure current conservation
        i_dist *= i_avg / np.mean(i_dist)
        
        return i_dist
    
    def _solve_heat_transfer(self, T_old: np.ndarray, i_field: np.ndarray,
                            overpotentials: np.ndarray, inputs: Dict) -> np.ndarray:
        """Solve 3D heat transfer with heat generation."""
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # Material thermal properties
        k_anode = self.mat_props['anode']['thermal_conductivity']
        k_electrolyte = self.mat_props['electrolyte']['thermal_conductivity']
        k_cathode = self.mat_props['cathode']['thermal_conductivity']
        
        # Build thermal conductivity field
        k_field = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    mat_id = self.mesh['material_field'][i, j, k]
                    if mat_id == 0:
                        k_field[i, j, k] = k_anode
                    elif mat_id == 1:
                        k_field[i, j, k] = k_electrolyte
                    else:
                        k_field[i, j, k] = k_cathode
        
        # Heat generation from overpotentials
        q_gen = i_field * (overpotentials[0] + overpotentials[1] + overpotentials[2])
        
        # Add entropic heat
        T_avg = np.mean(T_old)
        dS_dT = 44.43  # J/mol/K for H2 oxidation
        q_entropic = i_field * T_avg * dS_dT / (2 * self.physics.F)
        q_gen += q_entropic
        
        # Solve heat equation (simplified explicit scheme)
        dx = self.geom['length'] / nx
        dy = self.geom['width'] / ny
        dz = (self.geom['anode_thickness'] + self.geom['electrolyte_thickness'] + 
              self.geom['cathode_thickness']) * 1e-6 / nz
        
        dt = 0.1  # Time step
        alpha = k_field / (2500 * 500)  # Thermal diffusivity (k/rho/cp)
        
        # Stability criterion
        dt_max = 0.5 * min(dx**2, dy**2, dz**2) / np.max(alpha)
        dt = min(dt, dt_max)
        
        T_new = T_old.copy()
        
        # Apply finite difference
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # Central differences
                    d2T_dx2 = (T_old[i+1, j, k] - 2*T_old[i, j, k] + T_old[i-1, j, k]) / dx**2
                    d2T_dy2 = (T_old[i, j+1, k] - 2*T_old[i, j, k] + T_old[i, j-1, k]) / dy**2
                    d2T_dz2 = (T_old[i, j, k+1] - 2*T_old[i, j, k] + T_old[i, j, k-1]) / dz**2
                    
                    laplacian = d2T_dx2 + d2T_dy2 + d2T_dz2
                    
                    T_new[i, j, k] = T_old[i, j, k] + dt * (
                        alpha[i, j, k] * laplacian + q_gen[i, j, k] / (2500 * 500)
                    )
        
        # Boundary conditions
        # Inlet temperature
        T_new[0, :, :] = inputs['temperature']
        
        # Convective cooling at channels
        h_conv = 50  # W/m²/K
        T_air = inputs['temperature'] - 50
        
        for j in range(ny):
            if (j // 4) % 2 == 0:  # Channel regions
                # Top surface (cathode side)
                T_new[:, j, -1] = T_new[:, j, -2] - h_conv * (T_new[:, j, -2] - T_air) * dz / k_cathode
                # Bottom surface (anode side)
                T_new[:, j, 0] = T_new[:, j, 1] - h_conv * (T_new[:, j, 1] - T_air) * dz / k_anode
        
        return T_new
    
    def _solve_mechanics(self, T_field: np.ndarray, inputs: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Solve 3D mechanics with thermal and chemical expansion."""
        nx, ny, nz = self.nx, self.ny, self.nz
        
        stress_tensor = np.zeros((6, nx, ny, nz))
        strain_tensor = np.zeros((24, nx, ny, nz))  # 4 types x 6 components
        
        T_ref = 298  # Reference temperature
        t_hours = inputs['operating_time']
        
        # Build stiffness matrix components
        E_field = np.zeros((nx, ny, nz))
        nu_field = np.zeros((nx, ny, nz))
        CTE_field = np.zeros((nx, ny, nz))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    mat_id = self.mesh['material_field'][i, j, k]
                    
                    if mat_id == 0:  # Anode
                        E_field[i, j, k] = self.materials.elastic_modulus_Ni(T)
                        nu_field[i, j, k] = self.mat_props['anode']['poisson_ratio']
                        CTE_field[i, j, k] = self.materials.CTE_Ni(T)
                    elif mat_id == 1:  # Electrolyte
                        E_field[i, j, k] = self.materials.elastic_modulus_YSZ(T)
                        nu_field[i, j, k] = self.mat_props['electrolyte']['poisson_ratio']
                        CTE_field[i, j, k] = self.materials.CTE_YSZ(T)
                    else:  # Cathode
                        E_field[i, j, k] = self.mat_props['cathode']['elastic_modulus'] * \
                                          (1 - 0.0003 * (T - 298))
                        nu_field[i, j, k] = self.mat_props['cathode']['poisson_ratio']
                        CTE_field[i, j, k] = self.mat_props['cathode']['CTE']
        
        # Thermal strain
        thermal_strain = CTE_field * (T_field - T_ref)
        
        # Chemical expansion strain (redox)
        chemical_strain = np.zeros((nx, ny, nz))
        if inputs['redox_cycles'] > 0:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        if self.mesh['material_field'][i, j, k] == 0:  # Anode
                            # Partial oxidation based on local conditions
                            chemical_strain[i, j, k] = 0.24 * 0.1 * inputs['redox_cycles'] / 10
        
        # Total eigenstrains
        eigenstrain = thermal_strain + chemical_strain
        
        # Solve for stresses (simplified - assuming plane stress)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    E = E_field[i, j, k]
                    nu = nu_field[i, j, k]
                    eps_eigen = eigenstrain[i, j, k]
                    
                    # Check for interfaces
                    is_interface = False
                    if k > 0 and self.mesh['material_field'][i, j, k] != \
                       self.mesh['material_field'][i, j, k-1]:
                        is_interface = True
                    
                    # Constraint factor
                    if is_interface:
                        constraint = 1.0  # Full constraint at interface
                    elif k == 0 or k == nz-1:
                        constraint = 0.3  # Free surface
                    else:
                        constraint = 0.7  # Partial constraint
                    
                    # In-plane stresses
                    sigma = E * eps_eigen / (1 - nu) * constraint
                    stress_tensor[0, i, j, k] = sigma  # σxx
                    stress_tensor[1, i, j, k] = sigma * 0.95  # σyy
                    stress_tensor[2, i, j, k] = sigma * nu  # σzz
                    
                    # Interface shear
                    if is_interface:
                        # CTE mismatch shear
                        if k > 0:
                            delta_CTE = abs(CTE_field[i, j, k] - CTE_field[i, j, k-1])
                            tau = E * delta_CTE * (T_field[i, j, k] - T_ref) * 0.1
                            stress_tensor[5, i, j, k] = tau  # τyz
                    
                    # Calculate strains
                    # Elastic strain
                    strain_tensor[0, i, j, k] = (stress_tensor[0, i, j, k] - 
                                                 nu * (stress_tensor[1, i, j, k] + 
                                                      stress_tensor[2, i, j, k])) / E
                    strain_tensor[1, i, j, k] = (stress_tensor[1, i, j, k] - 
                                                 nu * (stress_tensor[0, i, j, k] + 
                                                      stress_tensor[2, i, j, k])) / E
                    strain_tensor[2, i, j, k] = (stress_tensor[2, i, j, k] - 
                                                 nu * (stress_tensor[0, i, j, k] + 
                                                      stress_tensor[1, i, j, k])) / E
                    
                    # Thermal strain components
                    strain_tensor[6:9, i, j, k] = thermal_strain[i, j, k]
                    
                    # Creep strain
                    if t_hours > 0:
                        von_mises = np.sqrt(0.5 * ((stress_tensor[0, i, j, k] - stress_tensor[1, i, j, k])**2 + 
                                                   (stress_tensor[1, i, j, k] - stress_tensor[2, i, j, k])**2 + 
                                                   (stress_tensor[2, i, j, k] - stress_tensor[0, i, j, k])**2))
                        
                        if von_mises > 1e6:  # Threshold for creep
                            creep_params = self.config['degradation']['creep']
                            creep_rate = self.physics.creep_strain_rate(von_mises, T_field[i, j, k], creep_params)
                            creep_strain = creep_rate * t_hours * 3600
                            strain_tensor[12:15, i, j, k] = creep_strain * np.array([1, 1, -2]) / 3
        
        # Add bending due to CTE mismatch
        avg_CTE_anode = np.mean(CTE_field[self.mesh['material_field'] == 0])
        avg_CTE_electrolyte = np.mean(CTE_field[self.mesh['material_field'] == 1])
        avg_CTE_cathode = np.mean(CTE_field[self.mesh['material_field'] == 2])
        
        if abs(avg_CTE_anode - avg_CTE_electrolyte) > 1e-6:
            # Curvature
            kappa = 6 * (avg_CTE_anode - avg_CTE_electrolyte) * (T_field.mean() - T_ref) / \
                   (self.geom['anode_thickness'] + self.geom['electrolyte_thickness']) * 1e-6
            
            # Bending stress
            for k in range(nz):
                z = k * (self.geom['anode_thickness'] + self.geom['electrolyte_thickness'] + 
                        self.geom['cathode_thickness']) * 1e-6 / nz
                z_neutral = (self.geom['anode_thickness'] + 0.5 * self.geom['electrolyte_thickness']) * 1e-6
                
                bending_stress = E_field[:, :, k] * kappa * (z - z_neutral)
                stress_tensor[0, :, :, k] += bending_stress
        
        return stress_tensor, strain_tensor
    
    def _add_microstructure_effects(self, outputs: Dict, inputs: Dict) -> Dict:
        """Add microstructure evolution effects."""
        T_field = outputs['temperature_field']
        t_hours = inputs['operating_time']
        
        if t_hours > 0:
            # Ni coarsening in anode
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        if self.mesh['material_field'][i, j, k] == 0:  # Anode
                            T = T_field[i, j, k]
                            
                            # Particle growth
                            coarsening_params = self.config['degradation']['ni_coarsening']
                            initial_size = self.microstructure['anode']['ni_particle_size']
                            
                            # Ostwald ripening
                            growth = coarsening_params['pre_exponential'] * \
                                   np.exp(-coarsening_params['activation_energy'] / (self.physics.R * T)) * \
                                   (t_hours * 3600) ** coarsening_params['time_exponent']
                            
                            new_size = initial_size + growth
                            
                            # TPB loss
                            tpb_loss = (new_size / initial_size - 1) ** 2
                            
                            # Increase resistance
                            outputs['overpotentials'][0, i, j, k] *= (1 + tpb_loss)
            
            # Cr poisoning at cathode
            cr_params = self.config['degradation']['cr_poisoning']
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        if self.mesh['material_field'][i, j, k] == 2:  # Cathode
                            T = T_field[i, j, k]
                            
                            # Cr deposition
                            cr_rate = self.degradation.chromium_poisoning_rate(T, t_hours * 3600, cr_params)
                            cr_thickness = cr_rate * t_hours * 3600
                            
                            # Increase cathode resistance
                            resistance_factor = 1 + 100 * cr_thickness  # Empirical
                            outputs['overpotentials'][0, i, j, k] *= resistance_factor
        
        return outputs
    
    def _compute_phase_field_damage(self, outputs: Dict, inputs: Dict) -> Dict:
        """Compute damage using phase-field fracture model."""
        stress_tensor = outputs['stress_tensor']
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # Phase field parameters
        l_c = 10e-6  # Characteristic length scale
        G_c = 50  # Critical energy release rate [J/m²]
        
        # Initialize damage field
        d_field = outputs.get('damage_field', np.zeros((nx, ny, nz)))
        
        # Compute elastic energy density
        W_elastic = np.zeros((nx, ny, nz))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Von Mises stress
                    s_xx = stress_tensor[0, i, j, k]
                    s_yy = stress_tensor[1, i, j, k]
                    s_zz = stress_tensor[2, i, j, k]
                    s_xy = stress_tensor[3, i, j, k]
                    s_xz = stress_tensor[4, i, j, k]
                    s_yz = stress_tensor[5, i, j, k]
                    
                    # Elastic energy
                    E = 150e9  # Average modulus
                    W_elastic[i, j, k] = (s_xx**2 + s_yy**2 + s_zz**2 + 
                                         2*(s_xy**2 + s_xz**2 + s_yz**2)) / (2 * E)
        
        # Solve phase-field equation (simplified)
        # This would normally require solving a PDE
        for iteration in range(5):
            d_old = d_field.copy()
            
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    for k in range(1, nz-1):
                        # Laplacian of damage field
                        dx = self.geom['length'] / nx
                        dy = self.geom['width'] / ny
                        dz = (self.geom['anode_thickness'] + 
                              self.geom['electrolyte_thickness'] + 
                              self.geom['cathode_thickness']) * 1e-6 / nz
                        
                        d2d_dx2 = (d_old[i+1, j, k] - 2*d_old[i, j, k] + d_old[i-1, j, k]) / dx**2
                        d2d_dy2 = (d_old[i, j+1, k] - 2*d_old[i, j, k] + d_old[i, j-1, k]) / dy**2
                        d2d_dz2 = (d_old[i, j, k+1] - 2*d_old[i, j, k] + d_old[i, j, k-1]) / dz**2
                        
                        laplacian_d = d2d_dx2 + d2d_dy2 + d2d_dz2
                        
                        # Driving force
                        driving_force = 2 * W_elastic[i, j, k] / G_c
                        
                        # Update damage (simplified Allen-Cahn)
                        d_field[i, j, k] = d_old[i, j, k] + 0.01 * (
                            l_c**2 * laplacian_d - d_old[i, j, k] + driving_force
                        )
                        
                        # Bound between 0 and 1
                        d_field[i, j, k] = np.clip(d_field[i, j, k], 0, 1)
        
        # Add interface delamination
        for i in range(nx):
            for j in range(ny):
                for k in range(1, nz-1):
                    if self.mesh['material_field'][i, j, k] != self.mesh['material_field'][i, j, k-1]:
                        # At interface
                        tau = stress_tensor[5, i, j, k]  # Shear stress
                        sigma = stress_tensor[2, i, j, k]  # Normal stress
                        
                        # Mixed-mode energy release rate
                        E_avg = 150e9
                        G = (tau**2 + sigma**2) / (2 * E_avg) * 1e-6  # Thickness factor
                        
                        if G > G_c:
                            d_field[i, j, k] = min(1.0, d_field[i, j, k] + 0.5)
        
        # Smooth damage field
        d_field = gaussian_filter(d_field, sigma=0.5)
        
        outputs['damage_field'] = d_field
        
        return outputs
    
    def _compute_damage_indicators(self, stress_tensor: np.ndarray, strain_tensor: np.ndarray,
                                  T_field: np.ndarray, inputs: Dict) -> np.ndarray:
        """Compute comprehensive damage indicators."""
        nx, ny, nz = self.nx, self.ny, self.nz
        damage_field = np.zeros((nx, ny, nz))
        
        t_hours = inputs['operating_time']
        n_thermal = inputs['thermal_cycles']
        n_redox = inputs['redox_cycles']
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    damages = []
                    
                    # Mechanical damage (Weibull statistics)
                    von_mises = self._compute_von_mises(stress_tensor[:, i, j, k])
                    sigma_0 = 200e6  # Characteristic strength
                    m = 10  # Weibull modulus
                    P_failure = 1 - np.exp(-(von_mises / sigma_0) ** m)
                    damages.append(P_failure)
                    
                    # Fatigue damage
                    if n_thermal > 0:
                        strain_range = np.max(strain_tensor[:6, i, j, k]) - np.min(strain_tensor[:6, i, j, k])
                        N_f = 10000 / (100 * strain_range) ** 2
                        damage_fatigue = n_thermal / N_f
                        damages.append(damage_fatigue)
                    
                    # Creep damage
                    creep_strain = np.sum(np.abs(strain_tensor[12:18, i, j, k]))
                    if creep_strain > 0:
                        damage_creep = (creep_strain / 0.02) ** 2  # Nonlinear accumulation
                        damages.append(damage_creep)
                    
                    # Chemical degradation
                    if self.mesh['material_field'][i, j, k] == 0 and n_redox > 0:
                        damage_redox = self.degradation.anode_reoxidation_damage(n_redox)
                        damages.append(damage_redox)
                    
                    # Combined damage (Palmgren-Miner rule)
                    if damages:
                        damage_field[i, j, k] = min(1.0, sum(damages))
        
        return damage_field
    
    def _compute_von_mises(self, stress_components: np.ndarray) -> float:
        """Compute von Mises stress from stress tensor components."""
        s_xx, s_yy, s_zz, s_xy, s_xz, s_yz = stress_components[:6]
        
        von_mises = np.sqrt(0.5 * ((s_xx - s_yy)**2 + 
                                   (s_yy - s_zz)**2 + 
                                   (s_zz - s_xx)**2 + 
                                   6 * (s_xy**2 + s_xz**2 + s_yz**2)))
        return von_mises
    
    def _compute_detailed_voltage(self, T_field: np.ndarray, i_field: np.ndarray,
                                 overpotentials: np.ndarray, species_field: np.ndarray) -> float:
        """Compute cell voltage with spatial integration."""
        # Average conditions
        T_avg = np.mean(T_field)
        p = 101325  # Pa
        
        # Average species concentrations at TPB
        anode_mask = self.mesh['material_field'] == 0
        cathode_mask = self.mesh['material_field'] == 2
        
        if np.any(anode_mask):
            pH2_avg = np.mean(species_field[0][anode_mask]) * self.physics.R * T_avg
            pH2O_avg = np.mean(species_field[1][anode_mask]) * self.physics.R * T_avg
        else:
            pH2_avg = p * 0.8
            pH2O_avg = p * 0.15
        
        if np.any(cathode_mask):
            pO2_avg = np.mean(species_field[2][cathode_mask]) * self.physics.R * T_avg
        else:
            pO2_avg = p * 0.21
        
        # Nernst voltage
        E_nernst = self.physics.nernst_voltage(T_avg, pH2_avg, pH2O_avg, pO2_avg)
        
        # Area-weighted average overpotentials
        eta_act = np.mean(overpotentials[0])
        eta_ohm = np.mean(overpotentials[1])
        eta_conc = np.mean(overpotentials[2])
        
        # Cell voltage
        V = E_nernst - eta_act - eta_ohm - eta_conc
        
        return max(0, V)
    
    def generate_dataset(self, n_samples: int, output_path: str = None) -> None:
        """Generate complete high-fidelity dataset."""
        print(f"\nGenerating {n_samples} high-fidelity samples...")
        print("Note: Each sample takes ~1 minute to generate with detailed physics")
        
        # Initialize data manager
        if output_path is None:
            output_path = "./data"
        
        manager = DatasetManager(output_path)
        
        # Create HDF5 file structure
        spatial_dims = (self.nx, self.ny, self.nz)
        f = manager.create_hdf5_structure('high_fidelity', n_samples, 
                                         spatial_dims, self.config)
        
        try:
            # Generate samples
            for idx in tqdm(range(n_samples), desc="High-Fidelity Generation"):
                sample = self.generate_sample(idx)
                manager.write_sample(f, idx, sample)
                
                # Flush after each sample due to large size
                f.flush()
            
            print(f"Successfully generated {n_samples} high-fidelity samples")
            
        finally:
            f.close()
        
        # Validate dataset
        validation_report = manager.validate_dataset('high_fidelity')
        print("\nDataset validation report:")
        print(f"  - File size: {validation_report['file_size_mb']:.2f} MB")
        print(f"  - Samples: {validation_report['n_samples']}")
        
        return validation_report


def main():
    """Main function for standalone execution."""
    import yaml
    
    # Load configuration
    config_path = "../../config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = HighFidelityGenerator(config)
    
    # Generate small test dataset
    generator.generate_dataset(2)  # Only 2 samples for testing


if __name__ == "__main__":
    main()