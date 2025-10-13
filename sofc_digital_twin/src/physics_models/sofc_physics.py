"""
SOFC Multi-Physics Models for Digital Twin Dataset Generation
Implements coupled electrochemical, thermal, and structural physics
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional
import warnings

class SOFCPhysicsModel:
    """
    Multi-physics SOFC model implementing:
    1. Electrochemical model (Butler-Volmer kinetics)
    2. Thermal model (heat conduction + generation)
    3. Structural model (linear elasticity with thermal expansion)
    """
    
    def __init__(self, geometry: Dict, material_props: Dict):
        self.geometry = geometry
        self.material_props = material_props
        self.setup_mesh()
        self.setup_material_properties()
        
    def setup_mesh(self):
        """Create 3D structured mesh"""
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        Lx, Ly, Lz = self.geometry['length'], self.geometry['width'], self.geometry['height']
        
        # Create coordinate arrays
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny) 
        self.z = np.linspace(0, Lz, nz)
        
        # Create 3D meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        
        # Grid spacing
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.dz = Lz / (nz - 1)
        
        # Total number of nodes
        self.n_nodes = nx * ny * nz
        
    def setup_material_properties(self):
        """Initialize material properties with spatial variation"""
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        
        # Initialize property arrays
        self.porosity = np.full((nx, ny, nz), self.material_props['electrode_porosity'])
        self.tortuosity = np.full((nx, ny, nz), self.material_props['electrode_tortuosity'])
        self.conductivity = np.full((nx, ny, nz), self.material_props['anode_conductivity'])
        
        # Add spatial variation and degradation
        if 'porosity_degradation' in self.material_props:
            # Simulate porosity loss near current collectors
            degradation = self.material_props['porosity_degradation']
            z_normalized = self.Z / self.geometry['height']
            self.porosity *= (1 - degradation * z_normalized)
            
        # Thermal properties
        self.thermal_conductivity = 2.0  # W/m/K
        self.density = 6000.0  # kg/m³
        self.specific_heat = 400.0  # J/kg/K
        
        # Mechanical properties
        self.youngs_modulus = 200e9  # Pa
        self.poisson_ratio = 0.3
        self.thermal_expansion = 12e-6  # 1/K
        
    def solve_electrochemical(self, operating_conditions: Dict) -> Dict:
        """
        Solve electrochemical model using Butler-Volmer kinetics
        Returns current density and overpotential distributions
        """
        current_density_target = operating_conditions['current_density']
        fuel_util = operating_conditions['fuel_utilization']
        
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        
        # Initialize fields
        potential = np.zeros((nx, ny, nz))
        current_density = np.zeros((nx, ny, nz))
        overpotential = np.zeros((nx, ny, nz))
        
        # Species concentrations (simplified)
        h2_conc = np.full((nx, ny, nz), operating_conditions['fuel_composition']['h2_percentage'])
        h2o_conc = np.full((nx, ny, nz), operating_conditions['fuel_composition']['h2o_percentage'])
        o2_conc = np.full((nx, ny, nz), 0.21)  # Air composition
        
        # Butler-Volmer parameters
        i0_anode = 1000.0  # A/m² exchange current density
        i0_cathode = 100.0  # A/m² exchange current density
        alpha_a = 0.5  # charge transfer coefficient
        alpha_c = 0.5
        
        # Temperature-dependent parameters
        T_ref = 1073.15  # K (800°C)
        F = 96485.0  # C/mol Faraday constant
        R = 8.314   # J/mol/K Gas constant
        
        # Solve for potential distribution (simplified Laplace equation)
        # ∇²φ = -i/σ
        A = self._build_laplacian_matrix()
        
        # Apply boundary conditions (current density at boundaries)
        rhs = np.zeros(self.n_nodes)
        
        # Top boundary: applied current density
        for i in range(nx):
            for j in range(ny):
                idx = self._get_node_index(i, j, nz-1)
                rhs[idx] = -current_density_target / self.conductivity[i, j, nz-1]
                
        # Bottom boundary: ground (φ = 0)
        for i in range(nx):
            for j in range(ny):
                idx = self._get_node_index(i, j, 0)
                A[idx, :] = 0
                A[idx, idx] = 1
                rhs[idx] = 0
                
        # Solve linear system
        phi_vec = spsolve(A, rhs)
        potential = phi_vec.reshape((nx, ny, nz))
        
        # Calculate current density from potential gradient
        grad_phi_x, grad_phi_y, grad_phi_z = np.gradient(potential, self.dx, self.dy, self.dz)
        current_density = -self.conductivity * np.sqrt(grad_phi_x**2 + grad_phi_y**2 + grad_phi_z**2)
        
        # Calculate overpotential using Butler-Volmer
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    i_local = abs(current_density[i, j, k])
                    if i_local > 0:
                        # Anode overpotential
                        eta_a = (R * T_ref / (alpha_a * F)) * np.log(i_local / i0_anode)
                        # Cathode overpotential  
                        eta_c = (R * T_ref / (alpha_c * F)) * np.log(i_local / i0_cathode)
                        overpotential[i, j, k] = eta_a + eta_c
        
        # Update species concentrations based on consumption
        consumption_rate = current_density / (2 * F)  # mol/m²/s
        h2_conc *= (1 - fuel_util * consumption_rate / np.max(consumption_rate))
        
        return {
            'potential': potential,
            'current_density': current_density,
            'overpotential': overpotential,
            'h2_concentration': h2_conc,
            'h2o_concentration': h2o_conc,
            'o2_concentration': o2_conc
        }
        
    def solve_thermal(self, electrochemical_results: Dict, operating_conditions: Dict) -> Dict:
        """
        Solve thermal model with heat generation from electrochemical reactions
        """
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        
        # Heat generation from overpotential losses
        current_density = electrochemical_results['current_density']
        overpotential = electrochemical_results['overpotential']
        heat_generation = current_density * overpotential  # W/m³
        
        # Initialize temperature field
        T_inlet_fuel = operating_conditions['inlet_fuel_temperature'] + 273.15  # K
        T_inlet_air = operating_conditions['inlet_air_temperature'] + 273.15   # K
        
        temperature = np.full((nx, ny, nz), (T_inlet_fuel + T_inlet_air) / 2)
        
        # Solve steady-state heat conduction with generation
        # ∇·(k∇T) + q = 0
        A = self._build_thermal_matrix()
        
        # Right-hand side with heat generation
        rhs = heat_generation.flatten()
        
        # Apply boundary conditions
        # Inlet temperatures at boundaries
        for i in range(nx):
            for j in range(ny):
                # Fuel inlet (bottom)
                idx = self._get_node_index(i, j, 0)
                A[idx, :] = 0
                A[idx, idx] = 1
                rhs[idx] = T_inlet_fuel
                
                # Air inlet (top) 
                idx = self._get_node_index(i, j, nz-1)
                A[idx, :] = 0
                A[idx, idx] = 1
                rhs[idx] = T_inlet_air
        
        # Solve
        T_vec = spsolve(A, rhs)
        temperature = T_vec.reshape((nx, ny, nz))
        
        # Calculate heat flux
        grad_T_x, grad_T_y, grad_T_z = np.gradient(temperature, self.dx, self.dy, self.dz)
        heat_flux_x = -self.thermal_conductivity * grad_T_x
        heat_flux_y = -self.thermal_conductivity * grad_T_y
        heat_flux_z = -self.thermal_conductivity * grad_T_z
        
        return {
            'temperature': temperature,
            'heat_generation': heat_generation,
            'heat_flux_x': heat_flux_x,
            'heat_flux_y': heat_flux_y,
            'heat_flux_z': heat_flux_z
        }
        
    def solve_structural(self, thermal_results: Dict) -> Dict:
        """
        Solve structural mechanics with thermal expansion
        """
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        temperature = thermal_results['temperature']
        T_ref = 298.15  # K (25°C reference)
        
        # Thermal strain
        thermal_strain = self.thermal_expansion * (temperature - T_ref)
        
        # Initialize displacement fields
        displacement_x = np.zeros((nx, ny, nz))
        displacement_y = np.zeros((nx, ny, nz))
        displacement_z = np.zeros((nx, ny, nz))
        
        # Simplified structural analysis - assume plane stress
        # Calculate displacements from thermal expansion
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Thermal expansion displacement
                    displacement_x[i, j, k] = thermal_strain[i, j, k] * self.x[i]
                    displacement_y[i, j, k] = thermal_strain[i, j, k] * self.y[j]
                    displacement_z[i, j, k] = thermal_strain[i, j, k] * self.z[k]
        
        # Calculate strain tensor
        du_dx, du_dy, du_dz = np.gradient(displacement_x, self.dx, self.dy, self.dz)
        dv_dx, dv_dy, dv_dz = np.gradient(displacement_y, self.dx, self.dy, self.dz)
        dw_dx, dw_dy, dw_dz = np.gradient(displacement_z, self.dx, self.dy, self.dz)
        
        strain_xx = du_dx
        strain_yy = dv_dy
        strain_zz = dw_dz
        strain_xy = 0.5 * (du_dy + dv_dx)
        strain_yz = 0.5 * (dv_dz + dw_dy)
        strain_zx = 0.5 * (dw_dx + du_dz)
        
        # Calculate stress tensor using Hooke's law
        E = self.youngs_modulus
        nu = self.poisson_ratio
        
        # Lame parameters
        lambda_lame = E * nu / ((1 + nu) * (1 - 2*nu))
        mu = E / (2 * (1 + nu))
        
        # Stress components
        strain_vol = strain_xx + strain_yy + strain_zz
        stress_xx = lambda_lame * strain_vol + 2 * mu * strain_xx
        stress_yy = lambda_lame * strain_vol + 2 * mu * strain_yy
        stress_zz = lambda_lame * strain_vol + 2 * mu * strain_zz
        stress_xy = 2 * mu * strain_xy
        stress_yz = 2 * mu * strain_yz
        stress_zx = 2 * mu * strain_zx
        
        # Von Mises stress
        von_mises_stress = np.sqrt(0.5 * (
            (stress_xx - stress_yy)**2 + 
            (stress_yy - stress_zz)**2 + 
            (stress_zz - stress_xx)**2 + 
            6 * (stress_xy**2 + stress_yz**2 + stress_zx**2)
        ))
        
        # Strain energy density
        strain_energy_density = 0.5 * (
            stress_xx * strain_xx + stress_yy * strain_yy + stress_zz * strain_zz +
            2 * (stress_xy * strain_xy + stress_yz * strain_yz + stress_zx * strain_zx)
        )
        
        return {
            'displacement_x': displacement_x,
            'displacement_y': displacement_y,
            'displacement_z': displacement_z,
            'strain_xx': strain_xx,
            'strain_yy': strain_yy,
            'strain_zz': strain_zz,
            'strain_xy': strain_xy,
            'strain_yz': strain_yz,
            'strain_zx': strain_zx,
            'stress_xx': stress_xx,
            'stress_yy': stress_yy,
            'stress_zz': stress_zz,
            'stress_xy': stress_xy,
            'stress_yz': stress_yz,
            'stress_zx': stress_zx,
            'von_mises_stress': von_mises_stress,
            'strain_energy_density': strain_energy_density
        }
        
    def _build_laplacian_matrix(self) -> sp.csr_matrix:
        """Build finite difference Laplacian matrix for 3D grid"""
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        n = nx * ny * nz
        
        # Create sparse matrix
        row_ind = []
        col_ind = []
        data = []
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    idx = self._get_node_index(i, j, k)
                    
                    # Diagonal term
                    diag_val = -2 * (1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2)
                    row_ind.append(idx)
                    col_ind.append(idx)
                    data.append(diag_val)
                    
                    # x-direction neighbors
                    if i > 0:
                        neighbor_idx = self._get_node_index(i-1, j, k)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dx**2)
                    if i < nx-1:
                        neighbor_idx = self._get_node_index(i+1, j, k)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dx**2)
                    
                    # y-direction neighbors
                    if j > 0:
                        neighbor_idx = self._get_node_index(i, j-1, k)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dy**2)
                    if j < ny-1:
                        neighbor_idx = self._get_node_index(i, j+1, k)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dy**2)
                    
                    # z-direction neighbors
                    if k > 0:
                        neighbor_idx = self._get_node_index(i, j, k-1)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dz**2)
                    if k < nz-1:
                        neighbor_idx = self._get_node_index(i, j, k+1)
                        row_ind.append(idx)
                        col_ind.append(neighbor_idx)
                        data.append(1/self.dz**2)
        
        return sp.csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    
    def _build_thermal_matrix(self) -> sp.csr_matrix:
        """Build thermal conduction matrix"""
        # For now, use same structure as Laplacian but with thermal conductivity
        A = self._build_laplacian_matrix()
        A *= -self.thermal_conductivity
        return A
    
    def _get_node_index(self, i: int, j: int, k: int) -> int:
        """Convert 3D indices to linear index"""
        nx, ny, nz = self.geometry['nx'], self.geometry['ny'], self.geometry['nz']
        return i * ny * nz + j * nz + k
        
    def solve_coupled_physics(self, operating_conditions: Dict) -> Dict:
        """
        Solve fully coupled multi-physics problem
        """
        # Solve electrochemical model
        electro_results = self.solve_electrochemical(operating_conditions)
        
        # Solve thermal model with electrochemical heat generation
        thermal_results = self.solve_thermal(electro_results, operating_conditions)
        
        # Solve structural model with thermal expansion
        structural_results = self.solve_structural(thermal_results)
        
        # Combine all results
        all_results = {}
        all_results.update(electro_results)
        all_results.update(thermal_results)
        all_results.update(structural_results)
        
        # Add derived quantities
        all_results['cell_voltage'] = np.mean(electro_results['potential'])
        all_results['max_temperature'] = np.max(thermal_results['temperature'])
        all_results['max_von_mises_stress'] = np.max(structural_results['von_mises_stress'])
        
        return all_results