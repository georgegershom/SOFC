"""
Thermal simulation module for SOFC systems.
Implements energy conservation, heat transfer, and thermal management.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional
import logging

class ThermalSimulator:
    """
    Thermal simulator for SOFC systems.
    
    Simulates:
    - Energy conservation (heat equation)
    - Conduction, convection, and radiation
    - Heat generation from electrochemical reactions
    - Thermal stress calculation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize thermal simulator.
        
        Args:
            config: Configuration dictionary with thermal properties and boundary conditions
        """
        self.config = config
        self.logger = logging.getLogger('ThermalSimulator')
        
        # Physical constants
        self.sigma = 5.67e-8  # Stefan-Boltzmann constant (W/m²K⁴)
        
        # Default thermal properties
        self.thermal_props = {
            'anode': {
                'thermal_conductivity': 2.0,  # W/mK
                'density': 3000,  # kg/m³
                'specific_heat': 500,  # J/kgK
                'emissivity': 0.8,
            },
            'electrolyte': {
                'thermal_conductivity': 2.5,  # W/mK
                'density': 6000,  # kg/m³
                'specific_heat': 400,  # J/kgK
                'emissivity': 0.9,
            },
            'cathode': {
                'thermal_conductivity': 1.5,  # W/mK
                'density': 3000,  # kg/m³
                'specific_heat': 500,  # J/kgK
                'emissivity': 0.8,
            },
            'interconnect': {
                'thermal_conductivity': 20.0,  # W/mK
                'density': 8000,  # kg/m³
                'specific_heat': 500,  # J/kgK
                'emissivity': 0.6,
            }
        }
        
        # Update with config values
        self.thermal_props.update(config.get('thermal_properties', {}))
    
    def solve_energy_conservation(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        current_density_field: np.ndarray,
        heat_generation: np.ndarray,
        boundary_conditions: Dict,
        time_step: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve energy conservation equation: ρcp(∂T/∂t) = ∇·(k∇T) + Q
        
        Args:
            mesh: 3D mesh coordinates (Nx, Ny, Nz, 3)
            temperature_field: Current temperature field (Nx, Ny, Nz)
            current_density_field: Current density field (Nx, Ny, Nz, 3)
            heat_generation: Heat generation rate (Nx, Ny, Nz)
            boundary_conditions: Boundary condition specifications
            time_step: Time step for transient analysis (s)
            
        Returns:
            Tuple of (new_temperature_field, heat_flux_field)
        """
        Nx, Ny, Nz = mesh.shape[:3]
        
        # Calculate thermal properties
        thermal_conductivity = self._calculate_thermal_conductivity(mesh, temperature_field)
        density = self._calculate_density(mesh)
        specific_heat = self._calculate_specific_heat(mesh)
        
        # Calculate heat generation from electrochemical reactions
        electrochemical_heat = self._calculate_electrochemical_heat(
            current_density_field, temperature_field
        )
        
        # Total heat generation
        total_heat_generation = heat_generation + electrochemical_heat
        
        # Build thermal stiffness matrix
        K = self._build_thermal_stiffness_matrix(mesh, thermal_conductivity)
        
        # Build mass matrix for transient term
        M = self._build_mass_matrix(mesh, density, specific_heat)
        
        # Apply boundary conditions
        b = self._apply_thermal_boundary_conditions(
            mesh, temperature_field, boundary_conditions, total_heat_generation
        )
        
        # Solve thermal equation: (M/dt + K)T = M/dt * T_old + b
        dt = time_step
        A = M / dt + K
        rhs = M.dot(temperature_field.flatten()) / dt + b
        
        new_temperature_flat = spsolve(A, rhs)
        new_temperature_field = new_temperature_flat.reshape((Nx, Ny, Nz))
        
        # Calculate heat flux
        heat_flux_field = self._calculate_heat_flux(
            mesh, new_temperature_field, thermal_conductivity
        )
        
        return new_temperature_field, heat_flux_field
    
    def calculate_thermal_stress(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        reference_temperature: float = 298.15
    ) -> np.ndarray:
        """
        Calculate thermal stress from temperature field.
        
        Args:
            mesh: 3D mesh coordinates
            temperature_field: Temperature field (K)
            reference_temperature: Reference temperature for thermal expansion (K)
            
        Returns:
            Thermal stress tensor (Nx, Ny, Nz, 6) - [σxx, σyy, σzz, σxy, σyz, σzx]
        """
        Nx, Ny, Nz = mesh.shape[:3]
        thermal_stress = np.zeros((Nx, Ny, Nz, 6))
        
        # Calculate temperature difference
        delta_T = temperature_field - reference_temperature
        
        # Get material properties
        youngs_modulus = self._calculate_youngs_modulus(mesh)
        poisson_ratio = self._calculate_poisson_ratio(mesh)
        thermal_expansion = self._calculate_thermal_expansion_coefficient(mesh)
        
        # Calculate thermal strain
        thermal_strain = thermal_expansion * delta_T
        
        # Calculate thermal stress using Hooke's law
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    E = youngs_modulus[i, j, k]
                    nu = poisson_ratio[i, j, k]
                    alpha = thermal_expansion[i, j, k]
                    
                    # Thermal strain components (isotropic)
                    epsilon_thermal = alpha * delta_T[i, j, k]
                    
                    # Stress components (plane stress assumption)
                    factor = E / (1 - nu**2)
                    thermal_stress[i, j, k, 0] = factor * (1 + nu) * epsilon_thermal  # σxx
                    thermal_stress[i, j, k, 1] = factor * (1 + nu) * epsilon_thermal  # σyy
                    thermal_stress[i, j, k, 2] = 0  # σzz (plane stress)
                    thermal_stress[i, j, k, 3] = 0  # σxy
                    thermal_stress[i, j, k, 4] = 0  # σyz
                    thermal_stress[i, j, k, 5] = 0  # σzx
        
        return thermal_stress
    
    def calculate_convection_heat_transfer(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        fluid_temperature: float,
        convection_coefficient: float
    ) -> np.ndarray:
        """
        Calculate convection heat transfer at boundaries.
        
        Args:
            mesh: 3D mesh coordinates
            temperature_field: Temperature field (K)
            fluid_temperature: Fluid temperature (K)
            convection_coefficient: Convection coefficient (W/m²K)
            
        Returns:
            Convection heat flux (Nx, Ny, Nz)
        """
        # Simplified convection calculation
        heat_flux = convection_coefficient * (temperature_field - fluid_temperature)
        return heat_flux
    
    def calculate_radiation_heat_transfer(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        ambient_temperature: float = 298.15
    ) -> np.ndarray:
        """
        Calculate radiation heat transfer.
        
        Args:
            mesh: 3D mesh coordinates
            temperature_field: Temperature field (K)
            ambient_temperature: Ambient temperature (K)
            
        Returns:
            Radiation heat flux (Nx, Ny, Nz)
        """
        # Get emissivity
        emissivity = self._calculate_emissivity(mesh)
        
        # Stefan-Boltzmann law
        heat_flux = emissivity * self.sigma * (
            temperature_field**4 - ambient_temperature**4
        )
        
        return heat_flux
    
    def _calculate_thermal_conductivity(self, mesh: np.ndarray, temperature_field: np.ndarray) -> np.ndarray:
        """Calculate temperature-dependent thermal conductivity."""
        Nx, Ny, Nz = mesh.shape[:3]
        thermal_conductivity = np.zeros((Nx, Ny, Nz))
        
        # Determine material regions based on z-coordinate
        z_coords = mesh[:, :, :, 2]
        
        # Anode region (z < 0.2)
        anode_mask = z_coords < 0.2
        thermal_conductivity[anode_mask] = self.thermal_props['anode']['thermal_conductivity']
        
        # Electrolyte region (0.2 <= z < 0.8)
        electrolyte_mask = (z_coords >= 0.2) & (z_coords < 0.8)
        thermal_conductivity[electrolyte_mask] = self.thermal_props['electrolyte']['thermal_conductivity']
        
        # Cathode region (0.8 <= z < 1.0)
        cathode_mask = (z_coords >= 0.8) & (z_coords < 1.0)
        thermal_conductivity[cathode_mask] = self.thermal_props['cathode']['thermal_conductivity']
        
        # Interconnect region (z >= 1.0)
        interconnect_mask = z_coords >= 1.0
        thermal_conductivity[interconnect_mask] = self.thermal_props['interconnect']['thermal_conductivity']
        
        return thermal_conductivity
    
    def _calculate_density(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate material density."""
        Nx, Ny, Nz = mesh.shape[:3]
        density = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Anode region
        anode_mask = z_coords < 0.2
        density[anode_mask] = self.thermal_props['anode']['density']
        
        # Electrolyte region
        electrolyte_mask = (z_coords >= 0.2) & (z_coords < 0.8)
        density[electrolyte_mask] = self.thermal_props['electrolyte']['density']
        
        # Cathode region
        cathode_mask = (z_coords >= 0.8) & (z_coords < 1.0)
        density[cathode_mask] = self.thermal_props['cathode']['density']
        
        # Interconnect region
        interconnect_mask = z_coords >= 1.0
        density[interconnect_mask] = self.thermal_props['interconnect']['density']
        
        return density
    
    def _calculate_specific_heat(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate material specific heat."""
        Nx, Ny, Nz = mesh.shape[:3]
        specific_heat = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Anode region
        anode_mask = z_coords < 0.2
        specific_heat[anode_mask] = self.thermal_props['anode']['specific_heat']
        
        # Electrolyte region
        electrolyte_mask = (z_coords >= 0.2) & (z_coords < 0.8)
        specific_heat[electrolyte_mask] = self.thermal_props['electrolyte']['specific_heat']
        
        # Cathode region
        cathode_mask = (z_coords >= 0.8) & (z_coords < 1.0)
        specific_heat[cathode_mask] = self.thermal_props['cathode']['specific_heat']
        
        # Interconnect region
        interconnect_mask = z_coords >= 1.0
        specific_heat[interconnect_mask] = self.thermal_props['interconnect']['specific_heat']
        
        return specific_heat
    
    def _calculate_electrochemical_heat(
        self, current_density_field: np.ndarray, temperature_field: np.ndarray
    ) -> np.ndarray:
        """Calculate heat generation from electrochemical reactions."""
        # Heat generation from overpotential losses
        # Simplified: Q = I²R + η_act * I
        current_magnitude = np.linalg.norm(current_density_field, axis=3)
        
        # Ohmic heating (simplified)
        ohmic_heat = current_magnitude**2 * 1e-6  # Simplified resistance
        
        # Activation heating (simplified)
        activation_heat = current_magnitude * 0.1  # Simplified overpotential
        
        total_heat = ohmic_heat + activation_heat
        return total_heat
    
    def _build_thermal_stiffness_matrix(self, mesh: np.ndarray, thermal_conductivity: np.ndarray) -> csr_matrix:
        """Build stiffness matrix for thermal equation."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        
        # Simplified finite difference approach
        row_indices = []
        col_indices = []
        data = []
        
        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                for k in range(1, Nz-1):
                    node_idx = i * Ny * Nz + j * Nz + k
                    
                    # Central node
                    row_indices.append(node_idx)
                    col_indices.append(node_idx)
                    data.append(-6 * thermal_conductivity[i, j, k])
                    
                    # Neighboring nodes
                    neighbors = [
                        (i-1, j, k), (i+1, j, k),
                        (i, j-1, k), (i, j+1, k),
                        (i, j, k-1), (i, j, k+1)
                    ]
                    
                    for ni, nj, nk in neighbors:
                        if 0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz:
                            neighbor_idx = ni * Ny * Nz + nj * Nz + nk
                            row_indices.append(node_idx)
                            col_indices.append(neighbor_idx)
                            data.append(thermal_conductivity[i, j, k])
        
        return csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    
    def _build_mass_matrix(self, mesh: np.ndarray, density: np.ndarray, specific_heat: np.ndarray) -> csr_matrix:
        """Build mass matrix for transient thermal analysis."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        
        # Lumped mass matrix
        mass_data = (density * specific_heat).flatten()
        row_indices = np.arange(n_nodes)
        col_indices = np.arange(n_nodes)
        
        return csr_matrix((mass_data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    
    def _apply_thermal_boundary_conditions(
        self, mesh: np.ndarray, temperature_field: np.ndarray, 
        boundary_conditions: Dict, heat_generation: np.ndarray
    ) -> np.ndarray:
        """Apply thermal boundary conditions."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        b = heat_generation.flatten()
        
        # Apply fixed temperature boundary conditions
        if 'fixed_temperature' in boundary_conditions:
            for boundary, temp in boundary_conditions['fixed_temperature'].items():
                if boundary == 'inlet':
                    # Inlet nodes (x = 0)
                    for j in range(Ny):
                        for k in range(Nz):
                            node_idx = j * Nz + k
                            b[node_idx] = temp
        
        return b
    
    def _calculate_heat_flux(
        self, mesh: np.ndarray, temperature_field: np.ndarray, thermal_conductivity: np.ndarray
    ) -> np.ndarray:
        """Calculate heat flux field from temperature gradient."""
        Nx, Ny, Nz = mesh.shape[:3]
        heat_flux = np.zeros((Nx, Ny, Nz, 3))
        
        # Calculate gradients using finite differences
        dx = mesh[1, 0, 0, 0] - mesh[0, 0, 0, 0]
        dy = mesh[0, 1, 0, 1] - mesh[0, 0, 0, 1]
        dz = mesh[0, 0, 1, 2] - mesh[0, 0, 0, 2]
        
        # X-component
        heat_flux[1:-1, :, :, 0] = -thermal_conductivity[1:-1, :, :] * (
            temperature_field[2:, :, :] - temperature_field[:-2, :, :]
        ) / (2 * dx)
        
        # Y-component
        heat_flux[:, 1:-1, :, 1] = -thermal_conductivity[:, 1:-1, :] * (
            temperature_field[:, 2:, :] - temperature_field[:, :-2, :]
        ) / (2 * dy)
        
        # Z-component
        heat_flux[:, :, 1:-1, 2] = -thermal_conductivity[:, :, 1:-1] * (
            temperature_field[:, :, 2:] - temperature_field[:, :, :-2]
        ) / (2 * dz)
        
        return heat_flux
    
    def _calculate_youngs_modulus(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate Young's modulus for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        youngs_modulus = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material properties (simplified)
        youngs_modulus[z_coords < 0.2] = 200e9  # Anode
        youngs_modulus[(z_coords >= 0.2) & (z_coords < 0.8)] = 200e9  # Electrolyte
        youngs_modulus[(z_coords >= 0.8) & (z_coords < 1.0)] = 200e9  # Cathode
        youngs_modulus[z_coords >= 1.0] = 200e9  # Interconnect
        
        return youngs_modulus
    
    def _calculate_poisson_ratio(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate Poisson's ratio for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        poisson_ratio = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material properties (simplified)
        poisson_ratio[z_coords < 0.2] = 0.3  # Anode
        poisson_ratio[(z_coords >= 0.2) & (z_coords < 0.8)] = 0.3  # Electrolyte
        poisson_ratio[(z_coords >= 0.8) & (z_coords < 1.0)] = 0.3  # Cathode
        poisson_ratio[z_coords >= 1.0] = 0.3  # Interconnect
        
        return poisson_ratio
    
    def _calculate_thermal_expansion_coefficient(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate thermal expansion coefficient for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        thermal_expansion = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material properties (simplified)
        thermal_expansion[z_coords < 0.2] = 12e-6  # Anode
        thermal_expansion[(z_coords >= 0.2) & (z_coords < 0.8)] = 10e-6  # Electrolyte
        thermal_expansion[(z_coords >= 0.8) & (z_coords < 1.0)] = 12e-6  # Cathode
        thermal_expansion[z_coords >= 1.0] = 12e-6  # Interconnect
        
        return thermal_expansion
    
    def _calculate_emissivity(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate emissivity for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        emissivity = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material properties
        emissivity[z_coords < 0.2] = self.thermal_props['anode']['emissivity']
        emissivity[(z_coords >= 0.2) & (z_coords < 0.8)] = self.thermal_props['electrolyte']['emissivity']
        emissivity[(z_coords >= 0.8) & (z_coords < 1.0)] = self.thermal_props['cathode']['emissivity']
        emissivity[z_coords >= 1.0] = self.thermal_props['interconnect']['emissivity']
        
        return emissivity