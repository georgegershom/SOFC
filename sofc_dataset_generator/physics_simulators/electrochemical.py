"""
Electrochemical simulation module for SOFC systems.
Implements charge conservation, species transport, and electrochemical reactions.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional
import logging

class ElectrochemicalSimulator:
    """
    Electrochemical simulator for SOFC systems.
    
    Simulates:
    - Charge conservation (Ohm's law)
    - Species transport (Fick's law)
    - Electrochemical reactions (Butler-Volmer kinetics)
    - Nernst potential calculation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize electrochemical simulator.
        
        Args:
            config: Configuration dictionary with material properties and operating conditions
        """
        self.config = config
        self.logger = logging.getLogger('ElectrochemicalSimulator')
        
        # Physical constants
        self.F = 96485.3329  # Faraday constant (C/mol)
        self.R = 8.314462618  # Gas constant (J/mol/K)
        
        # Default material properties
        self.material_props = {
            'anode': {
                'ionic_conductivity': 1.0,  # S/m
                'electronic_conductivity': 1e6,  # S/m
                'exchange_current_density': 1000,  # A/m²
                'activation_energy': 100000,  # J/mol
            },
            'electrolyte': {
                'ionic_conductivity': 0.1,  # S/m
                'electronic_conductivity': 1e-10,  # S/m
                'thickness': 10e-6,  # m
            },
            'cathode': {
                'ionic_conductivity': 1.0,  # S/m
                'electronic_conductivity': 1e6,  # S/m
                'exchange_current_density': 100,  # A/m²
                'activation_energy': 120000,  # J/mol
            }
        }
        
        # Update with config values
        self.material_props.update(config.get('material_properties', {}))
    
    def solve_charge_conservation(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        current_density: float,
        boundary_conditions: Dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve charge conservation equation: ∇·(σ∇φ) = 0
        
        Args:
            mesh: 3D mesh coordinates (Nx, Ny, Nz, 3)
            temperature_field: Temperature field (Nx, Ny, Nz)
            current_density: Applied current density (A/m²)
            boundary_conditions: Boundary condition specifications
            
        Returns:
            Tuple of (potential_field, current_density_field)
        """
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        
        # Initialize potential field
        potential_field = np.zeros((Nx, Ny, Nz))
        current_density_field = np.zeros((Nx, Ny, Nz, 3))  # 3D current density vector
        
        # Calculate conductivity based on temperature and material
        conductivity = self._calculate_conductivity(mesh, temperature_field)
        
        # Build stiffness matrix for charge conservation
        K = self._build_charge_stiffness_matrix(mesh, conductivity)
        
        # Apply boundary conditions
        b = np.zeros(n_nodes)
        
        # Anode boundary (fuel side) - fixed potential
        anode_nodes = self._get_boundary_nodes(mesh, 'anode')
        for node in anode_nodes:
            i, j, k = node
            node_idx = i * Ny * Nz + j * Nz + k
            K[node_idx, :] = 0
            K[node_idx, node_idx] = 1
            b[node_idx] = 0  # Reference potential
        
        # Cathode boundary (air side) - current density
        cathode_nodes = self._get_boundary_nodes(mesh, 'cathode')
        for node in cathode_nodes:
            i, j, k = node
            node_idx = i * Ny * Nz + j * Nz + k
            b[node_idx] = current_density
        
        # Solve linear system
        potential_flat = spsolve(K, b)
        potential_field = potential_flat.reshape((Nx, Ny, Nz))
        
        # Calculate current density field
        current_density_field = self._calculate_current_density(
            mesh, potential_field, conductivity
        )
        
        return potential_field, current_density_field
    
    def solve_species_transport(
        self,
        mesh: np.ndarray,
        temperature_field: np.ndarray,
        pressure_field: np.ndarray,
        velocity_field: np.ndarray,
        species: str
    ) -> np.ndarray:
        """
        Solve species transport equation: ∇·(D∇c) + ∇·(vc) = 0
        
        Args:
            mesh: 3D mesh coordinates
            temperature_field: Temperature field
            pressure_field: Pressure field
            velocity_field: Velocity field (Nx, Ny, Nz, 3)
            species: Species name ('H2', 'H2O', 'O2', 'N2')
            
        Returns:
            Species concentration field
        """
        Nx, Ny, Nz = mesh.shape[:3]
        
        # Species properties
        species_props = self._get_species_properties(species)
        
        # Calculate diffusivity
        diffusivity = self._calculate_diffusivity(
            temperature_field, pressure_field, species_props
        )
        
        # Build transport matrix
        K = self._build_transport_matrix(mesh, diffusivity, velocity_field)
        
        # Apply boundary conditions
        b = self._apply_species_boundary_conditions(mesh, species)
        
        # Solve system
        concentration_flat = spsolve(K, b)
        concentration_field = concentration_flat.reshape((Nx, Ny, Nz))
        
        return concentration_field
    
    def calculate_nernst_potential(
        self,
        temperature_field: np.ndarray,
        h2_concentration: np.ndarray,
        h2o_concentration: np.ndarray,
        o2_concentration: np.ndarray,
        pressure: float
    ) -> np.ndarray:
        """
        Calculate Nernst potential: E = E° + (RT/2F) * ln(pH2 * pO2^0.5 / pH2O)
        
        Args:
            temperature_field: Temperature field (K)
            h2_concentration: H2 concentration field (mol/m³)
            h2o_concentration: H2O concentration field (mol/m³)
            o2_concentration: O2 concentration field (mol/m³)
            pressure: Operating pressure (Pa)
            
        Returns:
            Nernst potential field (V)
        """
        # Standard potential at reference temperature
        E0 = 1.229  # V at 298.15 K
        
        # Partial pressures
        pH2 = h2_concentration * self.R * temperature_field / pressure
        pH2O = h2o_concentration * self.R * temperature_field / pressure
        pO2 = o2_concentration * self.R * temperature_field / pressure
        
        # Nernst equation
        nernst_potential = E0 + (self.R * temperature_field / (2 * self.F)) * np.log(
            pH2 * np.sqrt(pO2) / pH2O
        )
        
        return nernst_potential
    
    def calculate_activation_overpotential(
        self,
        current_density: np.ndarray,
        temperature_field: np.ndarray,
        electrode: str
    ) -> np.ndarray:
        """
        Calculate activation overpotential using Butler-Volmer equation.
        
        Args:
            current_density: Current density field (A/m²)
            temperature_field: Temperature field (K)
            electrode: Electrode type ('anode' or 'cathode')
            
        Returns:
            Activation overpotential field (V)
        """
        props = self.material_props[electrode]
        i0 = props['exchange_current_density']
        Ea = props['activation_energy']
        
        # Temperature-dependent exchange current density
        i0_T = i0 * np.exp(-Ea / (self.R * temperature_field))
        
        # Butler-Volmer equation (simplified)
        alpha = 0.5  # Transfer coefficient
        overpotential = (self.R * temperature_field / (alpha * self.F)) * np.arcsinh(
            current_density / (2 * i0_T)
        )
        
        return overpotential
    
    def calculate_ohmic_overpotential(
        self,
        current_density: np.ndarray,
        conductivity: np.ndarray,
        thickness: float
    ) -> np.ndarray:
        """
        Calculate ohmic overpotential: η_ohm = i * R_ohm
        
        Args:
            current_density: Current density field (A/m²)
            conductivity: Conductivity field (S/m)
            thickness: Electrode thickness (m)
            
        Returns:
            Ohmic overpotential field (V)
        """
        resistance = thickness / conductivity
        ohmic_overpotential = current_density * resistance
        
        return ohmic_overpotential
    
    def _calculate_conductivity(self, mesh: np.ndarray, temperature_field: np.ndarray) -> np.ndarray:
        """Calculate temperature-dependent conductivity."""
        Nx, Ny, Nz = mesh.shape[:3]
        conductivity = np.zeros((Nx, Ny, Nz))
        
        # Determine material regions based on z-coordinate
        z_coords = mesh[:, :, :, 2]
        
        # Anode region (z < 0.2)
        anode_mask = z_coords < 0.2
        conductivity[anode_mask] = self.material_props['anode']['ionic_conductivity'] * np.exp(
            -1000 / temperature_field[anode_mask]
        )
        
        # Electrolyte region (0.2 <= z < 0.8)
        electrolyte_mask = (z_coords >= 0.2) & (z_coords < 0.8)
        conductivity[electrolyte_mask] = self.material_props['electrolyte']['ionic_conductivity'] * np.exp(
            -1000 / temperature_field[electrolyte_mask]
        )
        
        # Cathode region (z >= 0.8)
        cathode_mask = z_coords >= 0.8
        conductivity[cathode_mask] = self.material_props['cathode']['ionic_conductivity'] * np.exp(
            -1000 / temperature_field[cathode_mask]
        )
        
        return conductivity
    
    def _build_charge_stiffness_matrix(self, mesh: np.ndarray, conductivity: np.ndarray) -> csr_matrix:
        """Build stiffness matrix for charge conservation equation."""
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
                    data.append(-6 * conductivity[i, j, k])
                    
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
                            data.append(conductivity[i, j, k])
        
        return csr_matrix((data, (row_indices, col_indices)), shape=(n_nodes, n_nodes))
    
    def _calculate_current_density(
        self, mesh: np.ndarray, potential_field: np.ndarray, conductivity: np.ndarray
    ) -> np.ndarray:
        """Calculate current density field from potential gradient."""
        Nx, Ny, Nz = mesh.shape[:3]
        current_density = np.zeros((Nx, Ny, Nz, 3))
        
        # Calculate gradients using finite differences
        dx = mesh[1, 0, 0, 0] - mesh[0, 0, 0, 0]
        dy = mesh[0, 1, 0, 1] - mesh[0, 0, 0, 1]
        dz = mesh[0, 0, 1, 2] - mesh[0, 0, 0, 2]
        
        # X-component
        current_density[1:-1, :, :, 0] = -conductivity[1:-1, :, :] * (
            potential_field[2:, :, :] - potential_field[:-2, :, :]
        ) / (2 * dx)
        
        # Y-component
        current_density[:, 1:-1, :, 1] = -conductivity[:, 1:-1, :] * (
            potential_field[:, 2:, :] - potential_field[:, :-2, :]
        ) / (2 * dy)
        
        # Z-component
        current_density[:, :, 1:-1, 2] = -conductivity[:, :, 1:-1] * (
            potential_field[:, :, 2:] - potential_field[:, :, :-2]
        ) / (2 * dz)
        
        return current_density
    
    def _get_boundary_nodes(self, mesh: np.ndarray, boundary: str) -> List[Tuple[int, int, int]]:
        """Get boundary node indices."""
        Nx, Ny, Nz = mesh.shape[:3]
        nodes = []
        
        if boundary == 'anode':
            # Bottom surface (z = 0)
            for i in range(Nx):
                for j in range(Ny):
                    nodes.append((i, j, 0))
        elif boundary == 'cathode':
            # Top surface (z = Nz-1)
            for i in range(Nx):
                for j in range(Ny):
                    nodes.append((i, j, Nz-1))
        
        return nodes
    
    def _get_species_properties(self, species: str) -> Dict:
        """Get species transport properties."""
        properties = {
            'H2': {'molecular_weight': 2.016, 'diameter': 2.89e-10},
            'H2O': {'molecular_weight': 18.015, 'diameter': 2.65e-10},
            'O2': {'molecular_weight': 31.999, 'diameter': 3.46e-10},
            'N2': {'molecular_weight': 28.014, 'diameter': 3.70e-10}
        }
        return properties.get(species, {})
    
    def _calculate_diffusivity(
        self, temperature_field: np.ndarray, pressure_field: np.ndarray, species_props: Dict
    ) -> np.ndarray:
        """Calculate species diffusivity using Chapman-Enskog theory."""
        # Simplified temperature dependence
        diffusivity = 1e-5 * (temperature_field / 1000) ** 1.75
        return diffusivity
    
    def _build_transport_matrix(
        self, mesh: np.ndarray, diffusivity: np.ndarray, velocity_field: np.ndarray
    ) -> csr_matrix:
        """Build transport matrix for species conservation."""
        # Simplified implementation
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        
        # For now, return identity matrix (simplified)
        return csr_matrix(np.eye(n_nodes))
    
    def _apply_species_boundary_conditions(self, mesh: np.ndarray, species: str) -> np.ndarray:
        """Apply species boundary conditions."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_nodes = Nx * Ny * Nz
        b = np.zeros(n_nodes)
        
        # Set inlet concentrations
        if species == 'H2':
            b[0] = 1.0  # Inlet H2 concentration
        elif species == 'H2O':
            b[0] = 0.0  # Inlet H2O concentration
        elif species == 'O2':
            b[-1] = 0.21  # Inlet O2 concentration
        
        return b