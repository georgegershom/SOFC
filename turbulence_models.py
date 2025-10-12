#!/usr/bin/env python3
"""
Turbulence Models for Stratified Flow Simulation
Implements various turbulence models for CFD simulation data generation
"""

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

class TurbulenceModels:
    """
    Collection of turbulence models for stratified flow simulation
    """
    
    def __init__(self, grid_spacing, fluid_properties):
        """
        Initialize turbulence models
        
        Parameters:
        -----------
        grid_spacing : tuple
            Grid spacing (dx, dy, dz)
        fluid_properties : dict
            Fluid properties dictionary
        """
        self.dx, self.dy, self.dz = grid_spacing
        self.rho1 = fluid_properties['density_1']
        self.rho2 = fluid_properties['density_2']
        self.mu1 = fluid_properties['viscosity_1']
        self.mu2 = fluid_properties['viscosity_2']
    
    def k_epsilon_model(self, velocity_field, density_field, vof_field):
        """
        Standard k-epsilon turbulence model
        
        Parameters:
        -----------
        velocity_field : array
            3D velocity field
        density_field : array
            3D density field
        vof_field : array
            Volume of fluid field
            
        Returns:
        --------
        k : array
            Turbulent kinetic energy
        epsilon : array
            Turbulent dissipation rate
        mu_t : array
            Turbulent viscosity
        """
        # Calculate velocity gradients
        grad_u = self._calculate_velocity_gradients(velocity_field)
        
        # Calculate strain rate tensor
        S_ij = self._calculate_strain_rate_tensor(grad_u)
        
        # Calculate turbulent kinetic energy production
        P_k = self._calculate_production_rate(S_ij, density_field)
        
        # Initialize k and epsilon
        k = np.zeros_like(density_field)
        epsilon = np.zeros_like(density_field)
        
        # Calculate k and epsilon using simplified algebraic relations
        u, v, w = velocity_field
        
        for i in range(density_field.shape[0]):
            for j in range(density_field.shape[1]):
                for k_idx in range(density_field.shape[2]):
                    # Mixing length model for k
                    mixing_length = 0.07 * min(self.dx, self.dy, self.dz)
                    velocity_magnitude = np.sqrt(u[i, j, k_idx]**2 + v[i, j, k_idx]**2 + w[i, j, k_idx]**2)
                    
                    k[i, j, k_idx] = 1.5 * (0.1 * velocity_magnitude)**2
                    
                    # Dissipation rate
                    epsilon[i, j, k_idx] = k[i, j, k_idx]**1.5 / mixing_length
        
        # Calculate turbulent viscosity
        mu_t = self._calculate_turbulent_viscosity(k, epsilon, density_field)
        
        return k, epsilon, mu_t
    
    def k_omega_sst_model(self, velocity_field, density_field, vof_field):
        """
        k-omega SST turbulence model
        
        Parameters:
        -----------
        velocity_field : array
            3D velocity field
        density_field : array
            3D density field
        vof_field : array
            Volume of fluid field
            
        Returns:
        --------
        k : array
            Turbulent kinetic energy
        omega : array
            Specific dissipation rate
        mu_t : array
            Turbulent viscosity
        """
        # Calculate velocity gradients
        grad_u = self._calculate_velocity_gradients(velocity_field)
        
        # Calculate strain rate tensor
        S_ij = self._calculate_strain_rate_tensor(grad_u)
        
        # Initialize k and omega
        k = np.zeros_like(density_field)
        omega = np.zeros_like(density_field)
        
        # Calculate k and omega using SST model
        u, v, w = velocity_field
        
        for i in range(density_field.shape[0]):
            for j in range(density_field.shape[1]):
                for k_idx in range(density_field.shape[2]):
                    # Turbulent kinetic energy
                    velocity_magnitude = np.sqrt(u[i, j, k_idx]**2 + v[i, j, k_idx]**2 + w[i, j, k_idx]**2)
                    k[i, j, k_idx] = 1.5 * (0.1 * velocity_magnitude)**2
                    
                    # Specific dissipation rate
                    mixing_length = 0.07 * min(self.dx, self.dy, self.dz)
                    omega[i, j, k_idx] = np.sqrt(k[i, j, k_idx]) / mixing_length
        
        # Calculate turbulent viscosity with SST blending
        mu_t = self._calculate_sst_turbulent_viscosity(k, omega, density_field, grad_u)
        
        return k, omega, mu_t
    
    def les_model(self, velocity_field, density_field, vof_field, filter_width):
        """
        Large Eddy Simulation (LES) model
        
        Parameters:
        -----------
        velocity_field : array
            3D velocity field
        density_field : array
            3D density field
        vof_field : array
            Volume of fluid field
        filter_width : float
            LES filter width
            
        Returns:
        --------
        k_sgs : array
            Subgrid-scale kinetic energy
        mu_sgs : array
            Subgrid-scale viscosity
        """
        # Calculate velocity gradients
        grad_u = self._calculate_velocity_gradients(velocity_field)
        
        # Calculate strain rate tensor
        S_ij = self._calculate_strain_rate_tensor(grad_u)
        
        # Initialize subgrid-scale quantities
        k_sgs = np.zeros_like(density_field)
        mu_sgs = np.zeros_like(density_field)
        
        # Smagorinsky model
        Cs = 0.1  # Smagorinsky constant
        
        for i in range(density_field.shape[0]):
            for j in range(density_field.shape[1]):
                for k_idx in range(density_field.shape[2]):
                    # Calculate strain rate magnitude
                    S_magnitude = np.sqrt(2 * np.sum(S_ij[i, j, k_idx]**2))
                    
                    # Subgrid-scale viscosity
                    mu_sgs[i, j, k_idx] = density_field[i, j, k_idx] * (Cs * filter_width)**2 * S_magnitude
                    
                    # Subgrid-scale kinetic energy
                    k_sgs[i, j, k_idx] = (Cs * filter_width * S_magnitude)**2
        
        return k_sgs, mu_sgs
    
    def stratified_turbulence_model(self, velocity_field, density_field, vof_field, 
                                  interface_position, stratification_parameter):
        """
        Turbulence model accounting for stratification effects
        
        Parameters:
        -----------
        velocity_field : array
            3D velocity field
        density_field : array
            3D density field
        vof_field : array
            Volume of fluid field
        interface_position : float
            Position of density interface
        stratification_parameter : float
            Stratification strength parameter
            
        Returns:
        --------
        k : array
            Turbulent kinetic energy
        epsilon : array
            Turbulent dissipation rate
        mu_t : array
            Turbulent viscosity
        """
        # Calculate base k-epsilon
        k, epsilon, mu_t = self.k_epsilon_model(velocity_field, density_field, vof_field)
        
        # Apply stratification corrections
        u, v, w = velocity_field
        
        for i in range(density_field.shape[0]):
            for j in range(density_field.shape[1]):
                for k_idx in range(density_field.shape[2]):
                    # Distance from interface
                    distance_from_interface = abs(self.dy * j - interface_position)
                    
                    # Stratification correction factor
                    Ri = self._calculate_richardson_number((u, v, w), density_field, i, j, k_idx)
                    
                    if Ri > 0.25:  # Stable stratification
                        # Suppress turbulence near interface
                        correction_factor = np.exp(-stratification_parameter * distance_from_interface)
                        k[i, j, k_idx] *= correction_factor
                        epsilon[i, j, k_idx] *= correction_factor
                        mu_t[i, j, k_idx] *= correction_factor
                    elif Ri < -0.25:  # Unstable stratification
                        # Enhance turbulence
                        correction_factor = 1.0 + stratification_parameter * abs(Ri)
                        k[i, j, k_idx] *= correction_factor
                        epsilon[i, j, k_idx] *= correction_factor
                        mu_t[i, j, k_idx] *= correction_factor
        
        return k, epsilon, mu_t
    
    def _calculate_velocity_gradients(self, velocity_field):
        """
        Calculate velocity gradients using finite differences
        """
        u, v, w = velocity_field
        
        # Calculate gradients using central differences
        du_dx = np.gradient(u, self.dx, axis=0)
        du_dy = np.gradient(u, self.dy, axis=1)
        du_dz = np.gradient(u, self.dz, axis=2)
        
        dv_dx = np.gradient(v, self.dx, axis=0)
        dv_dy = np.gradient(v, self.dy, axis=1)
        dv_dz = np.gradient(v, self.dz, axis=2)
        
        dw_dx = np.gradient(w, self.dx, axis=0)
        dw_dy = np.gradient(w, self.dy, axis=1)
        dw_dz = np.gradient(w, self.dz, axis=2)
        
        # Return gradient tensor
        grad_tensor = np.zeros((u.shape[0], u.shape[1], u.shape[2], 3, 3))
        
        for i in range(u.shape[0]):
            for j in range(u.shape[1]):
                for k in range(u.shape[2]):
                    grad_tensor[i, j, k] = np.array([
                        [du_dx[i, j, k], du_dy[i, j, k], du_dz[i, j, k]],
                        [dv_dx[i, j, k], dv_dy[i, j, k], dv_dz[i, j, k]],
                        [dw_dx[i, j, k], dw_dy[i, j, k], dw_dz[i, j, k]]
                    ])
        
        return grad_tensor
    
    def _calculate_strain_rate_tensor(self, grad_u):
        """
        Calculate strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        """
        S_ij = np.zeros_like(grad_u)
        
        for i in range(grad_u.shape[0]):
            for j in range(grad_u.shape[1]):
                for k in range(grad_u.shape[2]):
                    grad_matrix = grad_u[i, j, k]
                    S_ij[i, j, k] = 0.5 * (grad_matrix + grad_matrix.T)
        
        return S_ij
    
    def _calculate_production_rate(self, S_ij, density_field):
        """
        Calculate turbulent kinetic energy production rate
        P_k = 2 * mu_t * S_ij * S_ij
        """
        P_k = np.zeros_like(density_field)
        
        for i in range(S_ij.shape[0]):
            for j in range(S_ij.shape[1]):
                for k in range(S_ij.shape[2]):
                    S_matrix = S_ij[i, j, k]
                    P_k[i, j, k] = 2 * np.sum(S_matrix**2)
        
        return P_k
    
    def _calculate_turbulent_viscosity(self, k, epsilon, density_field):
        """
        Calculate turbulent viscosity using k-epsilon model
        mu_t = rho * C_mu * k^2 / epsilon
        """
        C_mu = 0.09  # k-epsilon model constant
        
        # Avoid division by zero
        epsilon_safe = np.maximum(epsilon, 1e-10)
        
        mu_t = density_field * C_mu * k**2 / epsilon_safe
        
        return mu_t
    
    def _calculate_sst_turbulent_viscosity(self, k, omega, density_field, grad_u):
        """
        Calculate turbulent viscosity using SST model
        """
        C_mu = 0.09
        
        # Avoid division by zero
        omega_safe = np.maximum(omega, 1e-10)
        
        mu_t = density_field * C_mu * k / omega_safe
        
        return mu_t
    
    def _calculate_richardson_number(self, velocity_field, density_field, i, j, k_idx):
        """
        Calculate Richardson number for stratification analysis
        Ri = -g/rho * drho/dy / (du/dy)^2
        """
        g = 9.81  # gravitational acceleration
        
        # Calculate density gradient
        if j > 0 and j < density_field.shape[1] - 1:
            drho_dy = (density_field[i, j+1, k_idx] - density_field[i, j-1, k_idx]) / (2 * self.dy)
        else:
            drho_dy = 0.0
        
        # Calculate velocity gradient
        u = velocity_field[0]  # x-component
        if j > 0 and j < u.shape[1] - 1:
            du_dy = (u[i, j+1, k_idx] - u[i, j-1, k_idx]) / (2 * self.dy)
        else:
            du_dy = 1e-6  # Avoid division by zero
        
        # Richardson number
        Ri = -g / density_field[i, j, k_idx] * drho_dy / (du_dy**2)
        
        return Ri

def generate_turbulence_data():
    """
    Generate comprehensive turbulence data for the thesis
    """
    # Grid parameters
    nx, ny, nz = 100, 50, 25
    dx, dy, dz = 0.1, 0.04, 0.04
    
    # Fluid properties
    fluid_props = {
        'density_1': 1000.0,
        'density_2': 1.2,
        'viscosity_1': 1e-3,
        'viscosity_2': 1.8e-5
    }
    
    # Initialize turbulence models
    turbulence = TurbulenceModels((dx, dy, dz), fluid_props)
    
    # Generate synthetic velocity field
    x = np.linspace(0, nx*dx, nx)
    y = np.linspace(0, ny*dy, ny)
    z = np.linspace(0, nz*dz, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Create stratified velocity field
    u = np.zeros((nx, ny, nz))
    v = np.zeros((nx, ny, nz))
    w = np.zeros((nx, ny, nz))
    
    interface_y = 0.5 * ny * dy
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                y_pos = Y[i, j, k]
                if y_pos < interface_y:
                    u[i, j, k] = 0.1  # Water velocity
                else:
                    u[i, j, k] = 5.0  # Air velocity
                
                # Add random fluctuations
                u[i, j, k] += 0.05 * np.random.normal(0, 1)
                v[i, j, k] = 0.01 * np.random.normal(0, 1)
                w[i, j, k] = 0.01 * np.random.normal(0, 1)
    
    velocity_field = (u, v, w)
    
    # Create density field
    density_field = np.zeros((nx, ny, nz))
    vof_field = np.zeros((nx, ny, nz))
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                y_pos = Y[i, j, k]
                if y_pos < interface_y:
                    density_field[i, j, k] = fluid_props['density_1']
                    vof_field[i, j, k] = 1.0
                else:
                    density_field[i, j, k] = fluid_props['density_2']
                    vof_field[i, j, k] = 0.0
    
    # Generate turbulence data using different models
    k_eps_k, k_eps_eps, k_eps_mu_t = turbulence.k_epsilon_model(
        velocity_field, density_field, vof_field)
    
    k_omega_k, k_omega_omega, k_omega_mu_t = turbulence.k_omega_sst_model(
        velocity_field, density_field, vof_field)
    
    les_k, les_mu_t = turbulence.les_model(
        velocity_field, density_field, vof_field, filter_width=0.1)
    
    stratified_k, stratified_eps, stratified_mu_t = turbulence.stratified_turbulence_model(
        velocity_field, density_field, vof_field, interface_y, stratification_parameter=1.0)
    
    return {
        'coordinates': (X, Y, Z),
        'velocity_field': velocity_field,
        'density_field': density_field,
        'vof_field': vof_field,
        'k_epsilon': {'k': k_eps_k, 'epsilon': k_eps_eps, 'mu_t': k_eps_mu_t},
        'k_omega_sst': {'k': k_omega_k, 'omega': k_omega_omega, 'mu_t': k_omega_mu_t},
        'les': {'k_sgs': les_k, 'mu_sgs': les_mu_t},
        'stratified': {'k': stratified_k, 'epsilon': stratified_eps, 'mu_t': stratified_mu_t}
    }

if __name__ == "__main__":
    # Generate turbulence data
    data = generate_turbulence_data()
    
    # Save data
    np.savez('turbulence_data.npz', **data)
    print("Turbulence data generated and saved to turbulence_data.npz")