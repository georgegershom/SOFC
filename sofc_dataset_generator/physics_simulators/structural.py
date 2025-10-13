"""
Structural simulation module for SOFC systems.
Implements stress analysis, strain calculation, and failure prediction.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional
import logging

class StructuralSimulator:
    """
    Structural simulator for SOFC systems.
    
    Simulates:
    - Linear elasticity (stress-strain relationships)
    - Thermal stress from temperature gradients
    - Mechanical stress from external loads
    - Failure prediction and crack propagation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize structural simulator.
        
        Args:
            config: Configuration dictionary with material properties and loading conditions
        """
        self.config = config
        self.logger = logging.getLogger('StructuralSimulator')
        
        # Default material properties
        self.material_props = {
            'anode': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_expansion': 12e-6,  # 1/K
                'density': 3000,  # kg/m³
                'yield_strength': 100e6,  # Pa
                'fracture_toughness': 2.0,  # MPa√m
            },
            'electrolyte': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_expansion': 10e-6,  # 1/K
                'density': 6000,  # kg/m³
                'yield_strength': 200e6,  # Pa
                'fracture_toughness': 1.5,  # MPa√m
            },
            'cathode': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_expansion': 12e-6,  # 1/K
                'density': 3000,  # kg/m³
                'yield_strength': 100e6,  # Pa
                'fracture_toughness': 2.0,  # MPa√m
            },
            'interconnect': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'thermal_expansion': 12e-6,  # 1/K
                'density': 8000,  # kg/m³
                'yield_strength': 300e6,  # Pa
                'fracture_toughness': 50.0,  # MPa√m
            }
        }
        
        # Update with config values
        self.material_props.update(config.get('material_properties', {}))
    
    def solve_linear_elasticity(
        self,
        mesh: np.ndarray,
        displacement_field: np.ndarray,
        thermal_stress: np.ndarray,
        mechanical_loads: Dict,
        boundary_conditions: Dict
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve linear elasticity equations: ∇·σ = 0
        
        Args:
            mesh: 3D mesh coordinates (Nx, Ny, Nz, 3)
            displacement_field: Current displacement field (Nx, Ny, Nz, 3)
            thermal_stress: Thermal stress tensor (Nx, Ny, Nz, 6)
            mechanical_loads: Applied mechanical loads
            boundary_conditions: Boundary condition specifications
            
        Returns:
            Tuple of (new_displacement_field, stress_tensor, strain_tensor)
        """
        Nx, Ny, Nz = mesh.shape[:3]
        
        # Calculate material properties
        youngs_modulus = self._calculate_youngs_modulus(mesh)
        poisson_ratio = self._calculate_poisson_ratio(mesh)
        
        # Build stiffness matrix
        K = self._build_structural_stiffness_matrix(mesh, youngs_modulus, poisson_ratio)
        
        # Apply boundary conditions and loads
        b = self._apply_structural_boundary_conditions(
            mesh, displacement_field, thermal_stress, mechanical_loads, boundary_conditions
        )
        
        # Solve linear system
        displacement_flat = spsolve(K, b)
        displacement_field = displacement_flat.reshape((Nx, Ny, Nz, 3))
        
        # Calculate strain tensor
        strain_tensor = self._calculate_strain_tensor(mesh, displacement_field)
        
        # Calculate stress tensor
        stress_tensor = self._calculate_stress_tensor(
            mesh, strain_tensor, thermal_stress, youngs_modulus, poisson_ratio
        )
        
        return displacement_field, stress_tensor, strain_tensor
    
    def calculate_von_mises_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """
        Calculate von Mises stress from stress tensor.
        
        Args:
            stress_tensor: Stress tensor (Nx, Ny, Nz, 6) - [σxx, σyy, σzz, σxy, σyz, σzx]
            
        Returns:
            von Mises stress field (Nx, Ny, Nz)
        """
        # Extract stress components
        sxx = stress_tensor[:, :, :, 0]
        syy = stress_tensor[:, :, :, 1]
        szz = stress_tensor[:, :, :, 2]
        sxy = stress_tensor[:, :, :, 3]
        syz = stress_tensor[:, :, :, 4]
        szx = stress_tensor[:, :, :, 5]
        
        # von Mises stress formula
        von_mises = np.sqrt(
            0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2) +
            3 * (sxy**2 + syz**2 + szx**2)
        )
        
        return von_mises
    
    def calculate_principal_stresses(self, stress_tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate principal stresses from stress tensor.
        
        Args:
            stress_tensor: Stress tensor (Nx, Ny, Nz, 6)
            
        Returns:
            Tuple of (σ1, σ2, σ3) - principal stresses
        """
        Nx, Ny, Nz = stress_tensor.shape[:3]
        sigma1 = np.zeros((Nx, Ny, Nz))
        sigma2 = np.zeros((Nx, Ny, Nz))
        sigma3 = np.zeros((Nx, Ny, Nz))
        
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    # Build stress matrix
                    stress_matrix = np.array([
                        [stress_tensor[i, j, k, 0], stress_tensor[i, j, k, 3], stress_tensor[i, j, k, 5]],
                        [stress_tensor[i, j, k, 3], stress_tensor[i, j, k, 1], stress_tensor[i, j, k, 4]],
                        [stress_tensor[i, j, k, 5], stress_tensor[i, j, k, 4], stress_tensor[i, j, k, 2]]
                    ])
                    
                    # Calculate eigenvalues (principal stresses)
                    eigenvalues = np.linalg.eigvals(stress_matrix)
                    eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
                    
                    sigma1[i, j, k] = eigenvalues[0]
                    sigma2[i, j, k] = eigenvalues[1]
                    sigma3[i, j, k] = eigenvalues[2]
        
        return sigma1, sigma2, sigma3
    
    def calculate_stress_intensity_factors(
        self,
        mesh: np.ndarray,
        stress_tensor: np.ndarray,
        crack_tips: List[Tuple[int, int, int]]
    ) -> Dict[str, np.ndarray]:
        """
        Calculate stress intensity factors at crack tips.
        
        Args:
            mesh: 3D mesh coordinates
            stress_tensor: Stress tensor field
            crack_tips: List of crack tip coordinates
            
        Returns:
            Dictionary with KI, KII, KIII stress intensity factors
        """
        stress_intensity_factors = {
            'KI': np.zeros(len(crack_tips)),
            'KII': np.zeros(len(crack_tips)),
            'KIII': np.zeros(len(crack_tips))
        }
        
        for i, (x, y, z) in enumerate(crack_tips):
            # Simplified calculation - in practice, this would use more sophisticated methods
            # like J-integral or displacement correlation methods
            
            # Mode I (opening mode) - normal stress perpendicular to crack plane
            stress_intensity_factors['KI'][i] = stress_tensor[x, y, z, 2] * np.sqrt(np.pi * 1e-6)
            
            # Mode II (sliding mode) - shear stress in crack plane
            stress_intensity_factors['KII'][i] = stress_tensor[x, y, z, 3] * np.sqrt(np.pi * 1e-6)
            
            # Mode III (tearing mode) - out-of-plane shear stress
            stress_intensity_factors['KIII'][i] = stress_tensor[x, y, z, 4] * np.sqrt(np.pi * 1e-6)
        
        return stress_intensity_factors
    
    def calculate_strain_energy_release_rate(
        self,
        stress_tensor: np.ndarray,
        strain_tensor: np.ndarray,
        crack_tips: List[Tuple[int, int, int]]
    ) -> np.ndarray:
        """
        Calculate strain energy release rate G at crack tips.
        
        Args:
            stress_tensor: Stress tensor field
            strain_tensor: Strain tensor field
            crack_tips: List of crack tip coordinates
            
        Returns:
            Strain energy release rate G (J/m²)
        """
        G = np.zeros(len(crack_tips))
        
        for i, (x, y, z) in enumerate(crack_tips):
            # Calculate strain energy density
            stress_vector = stress_tensor[x, y, z, :]
            strain_vector = strain_tensor[x, y, z, :]
            
            # Strain energy density = 0.5 * σ:ε
            strain_energy_density = 0.5 * np.dot(stress_vector, strain_vector)
            
            # Simplified G calculation (in practice, would use J-integral)
            G[i] = strain_energy_density * 1e-6  # Convert to J/m²
        
        return G
    
    def predict_failure(
        self,
        mesh: np.ndarray,
        stress_tensor: np.ndarray,
        strain_tensor: np.ndarray,
        crack_tips: List[Tuple[int, int, int]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict failure using various failure criteria.
        
        Args:
            mesh: 3D mesh coordinates
            stress_tensor: Stress tensor field
            strain_tensor: Strain tensor field
            crack_tips: List of crack tip coordinates
            
        Returns:
            Dictionary with failure predictions
        """
        Nx, Ny, Nz = mesh.shape[:3]
        
        # Calculate failure metrics
        von_mises = self.calculate_von_mises_stress(stress_tensor)
        sigma1, sigma2, sigma3 = self.calculate_principal_stresses(stress_tensor)
        
        # Get material properties
        yield_strength = self._calculate_yield_strength(mesh)
        fracture_toughness = self._calculate_fracture_toughness(mesh)
        
        # Failure criteria
        failure_predictions = {
            'von_mises_failure': von_mises > yield_strength,
            'max_principal_stress': sigma1 > yield_strength,
            'tresca_criterion': (sigma1 - sigma3) > yield_strength,
            'failure_index': von_mises / yield_strength,
            'safety_factor': yield_strength / von_mises
        }
        
        # Fracture mechanics criteria
        if crack_tips:
            stress_intensity = self.calculate_stress_intensity_factors(mesh, stress_tensor, crack_tips)
            strain_energy_release = self.calculate_strain_energy_release_rate(
                stress_tensor, strain_tensor, crack_tips
            )
            
            # Check for crack propagation
            KI_max = np.max(stress_intensity['KI'])
            KIC_min = np.min(fracture_toughness)
            
            failure_predictions.update({
                'crack_propagation': KI_max > KIC_min,
                'max_stress_intensity': KI_max,
                'critical_stress_intensity': KIC_min,
                'strain_energy_release_rate': strain_energy_release
            })
        
        return failure_predictions
    
    def _calculate_youngs_modulus(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate Young's modulus for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        youngs_modulus = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material regions
        youngs_modulus[z_coords < 0.2] = self.material_props['anode']['youngs_modulus']
        youngs_modulus[(z_coords >= 0.2) & (z_coords < 0.8)] = self.material_props['electrolyte']['youngs_modulus']
        youngs_modulus[(z_coords >= 0.8) & (z_coords < 1.0)] = self.material_props['cathode']['youngs_modulus']
        youngs_modulus[z_coords >= 1.0] = self.material_props['interconnect']['youngs_modulus']
        
        return youngs_modulus
    
    def _calculate_poisson_ratio(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate Poisson's ratio for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        poisson_ratio = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material regions
        poisson_ratio[z_coords < 0.2] = self.material_props['anode']['poisson_ratio']
        poisson_ratio[(z_coords >= 0.2) & (z_coords < 0.8)] = self.material_props['electrolyte']['poisson_ratio']
        poisson_ratio[(z_coords >= 0.8) & (z_coords < 1.0)] = self.material_props['cathode']['poisson_ratio']
        poisson_ratio[z_coords >= 1.0] = self.material_props['interconnect']['poisson_ratio']
        
        return poisson_ratio
    
    def _calculate_yield_strength(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate yield strength for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        yield_strength = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material regions
        yield_strength[z_coords < 0.2] = self.material_props['anode']['yield_strength']
        yield_strength[(z_coords >= 0.2) & (z_coords < 0.8)] = self.material_props['electrolyte']['yield_strength']
        yield_strength[(z_coords >= 0.8) & (z_coords < 1.0)] = self.material_props['cathode']['yield_strength']
        yield_strength[z_coords >= 1.0] = self.material_props['interconnect']['yield_strength']
        
        return yield_strength
    
    def _calculate_fracture_toughness(self, mesh: np.ndarray) -> np.ndarray:
        """Calculate fracture toughness for different materials."""
        Nx, Ny, Nz = mesh.shape[:3]
        fracture_toughness = np.zeros((Nx, Ny, Nz))
        
        z_coords = mesh[:, :, :, 2]
        
        # Material regions
        fracture_toughness[z_coords < 0.2] = self.material_props['anode']['fracture_toughness']
        fracture_toughness[(z_coords >= 0.2) & (z_coords < 0.8)] = self.material_props['electrolyte']['fracture_toughness']
        fracture_toughness[(z_coords >= 0.8) & (z_coords < 1.0)] = self.material_props['cathode']['fracture_toughness']
        fracture_toughness[z_coords >= 1.0] = self.material_props['interconnect']['fracture_toughness']
        
        return fracture_toughness
    
    def _build_structural_stiffness_matrix(
        self, mesh: np.ndarray, youngs_modulus: np.ndarray, poisson_ratio: np.ndarray
    ) -> csr_matrix:
        """Build structural stiffness matrix for linear elasticity."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_dof = Nx * Ny * Nz * 3  # 3 DOF per node (ux, uy, uz)
        
        # Simplified implementation - in practice, this would be much more complex
        # involving proper finite element assembly
        
        # For now, return a simplified diagonal matrix
        diagonal = np.ones(n_dof)
        row_indices = np.arange(n_dof)
        col_indices = np.arange(n_dof)
        
        return csr_matrix((diagonal, (row_indices, col_indices)), shape=(n_dof, n_dof))
    
    def _apply_structural_boundary_conditions(
        self, mesh: np.ndarray, displacement_field: np.ndarray, 
        thermal_stress: np.ndarray, mechanical_loads: Dict, boundary_conditions: Dict
    ) -> np.ndarray:
        """Apply structural boundary conditions and loads."""
        Nx, Ny, Nz = mesh.shape[:3]
        n_dof = Nx * Ny * Nz * 3
        b = np.zeros(n_dof)
        
        # Apply thermal stress as initial stress
        thermal_force = thermal_stress.flatten()
        b[:len(thermal_force)] = thermal_force
        
        # Apply mechanical loads
        if 'pressure' in mechanical_loads:
            pressure = mechanical_loads['pressure']
            # Apply pressure to appropriate surfaces
            # (simplified implementation)
            pass
        
        return b
    
    def _calculate_strain_tensor(self, mesh: np.ndarray, displacement_field: np.ndarray) -> np.ndarray:
        """Calculate strain tensor from displacement field."""
        Nx, Ny, Nz = mesh.shape[:3]
        strain_tensor = np.zeros((Nx, Ny, Nz, 6))  # [εxx, εyy, εzz, εxy, εyz, εzx]
        
        # Calculate strain using finite differences
        dx = mesh[1, 0, 0, 0] - mesh[0, 0, 0, 0]
        dy = mesh[0, 1, 0, 1] - mesh[0, 0, 0, 1]
        dz = mesh[0, 0, 1, 2] - mesh[0, 0, 0, 2]
        
        # Normal strains
        strain_tensor[1:-1, :, :, 0] = (displacement_field[2:, :, :, 0] - displacement_field[:-2, :, :, 0]) / (2 * dx)
        strain_tensor[:, 1:-1, :, 1] = (displacement_field[:, 2:, :, 1] - displacement_field[:, :-2, :, 1]) / (2 * dy)
        strain_tensor[:, :, 1:-1, 2] = (displacement_field[:, :, 2:, 2] - displacement_field[:, :, :-2, 2]) / (2 * dz)
        
        # Shear strains
        strain_tensor[1:-1, 1:-1, :, 3] = 0.5 * (
            (displacement_field[2:, 1:-1, :, 1] - displacement_field[:-2, 1:-1, :, 1]) / (2 * dx) +
            (displacement_field[1:-1, 2:, :, 0] - displacement_field[1:-1, :-2, :, 0]) / (2 * dy)
        )
        
        strain_tensor[:, 1:-1, 1:-1, 4] = 0.5 * (
            (displacement_field[:, 2:, 1:-1, 2] - displacement_field[:, :-2, 1:-1, 2]) / (2 * dy) +
            (displacement_field[:, 1:-1, 2:, 1] - displacement_field[:, 1:-1, :-2, 1]) / (2 * dz)
        )
        
        strain_tensor[1:-1, :, 1:-1, 5] = 0.5 * (
            (displacement_field[2:, :, 1:-1, 2] - displacement_field[:-2, :, 1:-1, 2]) / (2 * dx) +
            (displacement_field[1:-1, :, 2:, 0] - displacement_field[1:-1, :, :-2, 0]) / (2 * dz)
        )
        
        return strain_tensor
    
    def _calculate_stress_tensor(
        self, mesh: np.ndarray, strain_tensor: np.ndarray, thermal_stress: np.ndarray,
        youngs_modulus: np.ndarray, poisson_ratio: np.ndarray
    ) -> np.ndarray:
        """Calculate stress tensor from strain tensor using Hooke's law."""
        Nx, Ny, Nz = mesh.shape[:3]
        stress_tensor = np.zeros((Nx, Ny, Nz, 6))
        
        # Hooke's law for isotropic materials
        for i in range(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    E = youngs_modulus[i, j, k]
                    nu = poisson_ratio[i, j, k]
                    
                    # Strain components
                    exx = strain_tensor[i, j, k, 0]
                    eyy = strain_tensor[i, j, k, 1]
                    ezz = strain_tensor[i, j, k, 2]
                    exy = strain_tensor[i, j, k, 3]
                    eyz = strain_tensor[i, j, k, 4]
                    ezx = strain_tensor[i, j, k, 5]
                    
                    # Stress components (plane stress assumption)
                    factor = E / (1 - nu**2)
                    stress_tensor[i, j, k, 0] = factor * (exx + nu * eyy)  # σxx
                    stress_tensor[i, j, k, 1] = factor * (eyy + nu * exx)  # σyy
                    stress_tensor[i, j, k, 2] = 0  # σzz (plane stress)
                    stress_tensor[i, j, k, 3] = E * exy / (2 * (1 + nu))  # σxy
                    stress_tensor[i, j, k, 4] = E * eyz / (2 * (1 + nu))  # σyz
                    stress_tensor[i, j, k, 5] = E * ezx / (2 * (1 + nu))  # σzx
        
        # Add thermal stress
        stress_tensor += thermal_stress
        
        return stress_tensor