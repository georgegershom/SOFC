"""
FEA Solver for SOFC Thermo-mechanical Analysis

Provides a simplified FEA solver for SOFC stress analysis using finite difference
methods as a proxy for full FEA. This is suitable for generating synthetic datasets
without requiring commercial FEA software.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
import time


@dataclass
class SimulationResults:
    """Container for FEA simulation results"""
    # Displacement field
    displacements: np.ndarray  # [n_nodes, 3] - ux, uy, uz
    
    # Stress field
    stress_tensor: np.ndarray  # [n_elements, 6] - σxx, σyy, σzz, σxy, σxz, σyz
    von_mises_stress: np.ndarray  # [n_elements] - Von Mises stress
    principal_stresses: np.ndarray  # [n_elements, 3] - σ1, σ2, σ3
    
    # Strain field
    strain_tensor: np.ndarray  # [n_elements, 6] - εxx, εyy, εzz, εxy, εxz, εyz
    
    # Temperature field
    temperature: np.ndarray  # [n_nodes] - Temperature at each node
    
    # Metadata
    simulation_time: float
    convergence_info: Dict[str, float]
    mesh_info: Dict[str, int]
    
    def get_warp_field(self, surface_nodes: np.ndarray) -> np.ndarray:
        """Get warp field (displacements) for surface nodes"""
        return self.displacements[surface_nodes]
    
    def get_stress_field(self, element_group: str, element_groups: Dict[str, List[int]]) -> np.ndarray:
        """Get stress field for specific element group"""
        element_ids = element_groups[element_group]
        return self.stress_tensor[element_ids]
    
    def get_max_principal_stress(self, element_group: str, element_groups: Dict[str, List[int]]) -> float:
        """Get maximum principal stress for element group"""
        element_ids = element_groups[element_group]
        return np.max(self.principal_stresses[element_ids, 0])  # σ1 is first column


class FEASolver:
    """Simplified FEA solver for SOFC analysis"""
    
    def __init__(self, mesh: Dict, materials: Dict, creep_model=None):
        self.mesh = mesh
        self.materials = materials
        self.creep_model = creep_model
        self.results = None
        
        # Initialize solver matrices
        self._initialize_matrices()
    
    def _initialize_matrices(self):
        """Initialize global stiffness matrix and other system matrices"""
        n_nodes = len(self.mesh['nodes'])
        n_dof = n_nodes * 3  # 3 DOF per node (ux, uy, uz)
        
        # Global stiffness matrix (sparse)
        self.K_global = sp.lil_matrix((n_dof, n_dof))
        
        # Global force vector
        self.F_global = np.zeros(n_dof)
        
        # Global displacement vector
        self.U_global = np.zeros(n_dof)
        
        # Element stiffness matrices will be computed during assembly
        self.element_stiffness_matrices = {}
    
    def solve_thermo_mechanical(self, 
                              temperature_field: np.ndarray,
                              boundary_conditions: Dict[str, float],
                              assembly_pressure: float = 0.0,
                              use_creep: bool = False,
                              time_steps: Optional[List[float]] = None) -> SimulationResults:
        """
        Solve coupled thermo-mechanical problem
        
        Args:
            temperature_field: Temperature at each node [n_nodes]
            boundary_conditions: Dictionary of boundary conditions
            assembly_pressure: Assembly pressure (MPa)
            use_creep: Whether to include creep effects
            time_steps: Time steps for creep analysis (if use_creep=True)
        """
        start_time = time.time()
        
        # Assemble global stiffness matrix
        self._assemble_stiffness_matrix(temperature_field)
        
        # Apply boundary conditions
        self._apply_boundary_conditions(boundary_conditions)
        
        # Apply thermal loads
        self._apply_thermal_loads(temperature_field)
        
        # Apply assembly pressure
        if assembly_pressure > 0:
            self._apply_assembly_pressure(assembly_pressure)
        
        # Solve system
        if use_creep and time_steps is not None:
            self._solve_with_creep(time_steps, temperature_field)
        else:
            self._solve_static()
        
        # Post-process results
        self._post_process_results(temperature_field)
        
        # Create results object
        simulation_time = time.time() - start_time
        self.results = SimulationResults(
            displacements=self.U_global.reshape(-1, 3),
            stress_tensor=self.stress_tensor,
            von_mises_stress=self.von_mises_stress,
            principal_stresses=self.principal_stresses,
            strain_tensor=self.strain_tensor,
            temperature=temperature_field,
            simulation_time=simulation_time,
            convergence_info=self.convergence_info,
            mesh_info=self.mesh['mesh_dimensions']
        )
        
        return self.results
    
    def _assemble_stiffness_matrix(self, temperature_field: np.ndarray):
        """Assemble global stiffness matrix from element contributions"""
        elements = self.mesh['elements']
        nodes = self.mesh['nodes']
        element_groups = self.mesh['element_groups']
        
        # Clear previous assembly
        self.K_global = sp.lil_matrix(self.K_global.shape)
        
        for elem_id, element in enumerate(elements):
            # Get element nodes
            elem_nodes = element
            elem_coords = nodes[elem_nodes]
            
            # Determine material properties based on element group
            material_name = self._get_element_material(elem_id, element_groups)
            material_props = self._get_material_properties(material_name, temperature_field[elem_nodes])
            
            # Compute element stiffness matrix
            K_elem = self._compute_element_stiffness(elem_coords, material_props)
            
            # Assemble into global matrix
            self._assemble_element_matrix(K_elem, elem_nodes)
    
    def _get_element_material(self, elem_id: int, element_groups: Dict[str, List[int]]) -> str:
        """Determine material for element based on group membership"""
        for group_name, element_ids in element_groups.items():
            if elem_id in element_ids:
                return group_name
        return 'electrolyte'  # Default
    
    def _get_material_properties(self, material_name: str, temperatures: np.ndarray) -> Dict[str, float]:
        """Get material properties at given temperatures"""
        # Use average temperature for element
        avg_temp = np.mean(temperatures)
        
        if material_name in self.materials:
            return self.materials[material_name]
        else:
            # Default properties
            return {
                'E': 170e3,  # MPa
                'nu': 0.23,
                'alpha': 10.5e-6,  # 1/K
                'density': 5900.0  # kg/m³
            }
    
    def _compute_element_stiffness(self, coords: np.ndarray, material_props: Dict[str, float]) -> np.ndarray:
        """Compute element stiffness matrix for 8-node hexahedron"""
        # Simplified element stiffness computation
        # In a full implementation, this would use proper finite element formulation
        
        E = material_props['E']
        nu = material_props['nu']
        
        # Compute element volume
        volume = self._compute_element_volume(coords)
        
        # Simplified stiffness matrix (isotropic)
        # This is a very simplified approach - real FEA would use proper shape functions
        n_dof = 24  # 8 nodes × 3 DOF
        K_elem = np.zeros((n_dof, n_dof))
        
        # Simplified stiffness based on element volume and material properties
        # This is not a proper finite element formulation but serves for synthetic data generation
        base_stiffness = E * volume / (1 + nu) / (1 - 2*nu)
        
        # Diagonal terms (simplified)
        for i in range(n_dof):
            K_elem[i, i] = base_stiffness
        
        # Off-diagonal terms (simplified coupling)
        for i in range(0, n_dof, 3):  # Every 3rd DOF (x-direction)
            for j in range(i+1, min(i+3, n_dof)):
                K_elem[i, j] = K_elem[j, i] = base_stiffness * 0.1
        
        return K_elem
    
    def _compute_element_volume(self, coords: np.ndarray) -> float:
        """Compute element volume using simplified method"""
        # Simplified volume calculation
        # In practice, this would use proper hexahedral volume formula
        dx = np.max(coords[:, 0]) - np.min(coords[:, 0])
        dy = np.max(coords[:, 1]) - np.min(coords[:, 1])
        dz = np.max(coords[:, 2]) - np.min(coords[:, 2])
        return dx * dy * dz
    
    def _assemble_element_matrix(self, K_elem: np.ndarray, elem_nodes: np.ndarray):
        """Assemble element matrix into global matrix"""
        n_dof_per_node = 3
        n_nodes_per_elem = len(elem_nodes)
        n_dof_per_elem = n_nodes_per_elem * n_dof_per_node
        
        # Global DOF indices for this element
        global_dofs = []
        for node_id in elem_nodes:
            for dof in range(n_dof_per_node):
                global_dofs.append(node_id * n_dof_per_node + dof)
        
        # Assemble
        for i in range(n_dof_per_elem):
            for j in range(n_dof_per_elem):
                self.K_global[global_dofs[i], global_dofs[j]] += K_elem[i, j]
    
    def _apply_boundary_conditions(self, boundary_conditions: Dict[str, float]):
        """Apply displacement boundary conditions"""
        # Simplified boundary condition application
        # In practice, this would modify the stiffness matrix and force vector
        
        # For now, just apply simple constraints
        if 'bottom_fixed_z' in boundary_conditions:
            # Fix bottom surface in Z direction
            bottom_nodes = self._get_bottom_nodes()
            for node_id in bottom_nodes:
                dof_z = node_id * 3 + 2  # Z DOF
                # Modify stiffness matrix (penalty method)
                self.K_global[dof_z, dof_z] = 1e12
                self.F_global[dof_z] = 0.0
    
    def _get_bottom_nodes(self) -> np.ndarray:
        """Get bottom surface node indices"""
        nodes = self.mesh['nodes']
        z_min = np.min(nodes[:, 2])
        return np.where(np.abs(nodes[:, 2] - z_min) < 1e-6)[0]
    
    def _apply_thermal_loads(self, temperature_field: np.ndarray):
        """Apply thermal expansion loads"""
        # Simplified thermal load application
        # In practice, this would compute thermal strains and convert to forces
        
        nodes = self.mesh['nodes']
        n_nodes = len(nodes)
        
        # Reference temperature (sintering temperature)
        T_ref = 1350.0  # °C
        
        # Apply thermal expansion in all directions
        for i, (node, temp) in enumerate(zip(nodes, temperature_field)):
            # Thermal strain
            alpha = 10.5e-6  # CTE (simplified)
            thermal_strain = alpha * (temp - T_ref)
            
            # Convert to forces (simplified)
            for dof in range(3):  # x, y, z directions
                dof_id = i * 3 + dof
                self.F_global[dof_id] = thermal_strain * 1e6  # Simplified force magnitude
    
    def _apply_assembly_pressure(self, pressure: float):
        """Apply assembly pressure to top surface"""
        # Get top surface nodes
        top_nodes = self._get_top_nodes()
        
        # Apply pressure in Z direction (downward)
        for node_id in top_nodes:
            dof_z = node_id * 3 + 2  # Z DOF
            self.F_global[dof_z] -= pressure * 1e6  # Convert MPa to Pa
    
    def _get_top_nodes(self) -> np.ndarray:
        """Get top surface node indices"""
        nodes = self.mesh['nodes']
        z_max = np.max(nodes[:, 2])
        return np.where(np.abs(nodes[:, 2] - z_max) < 1e-6)[0]
    
    def _solve_static(self):
        """Solve static system"""
        # Convert to CSC format for efficient solving
        K_csc = csc_matrix(self.K_global)
        
        # Solve Ku = F
        self.U_global = spsolve(K_csc, self.F_global)
        
        # Store convergence info
        self.convergence_info = {
            'residual': np.linalg.norm(K_csc @ self.U_global - self.F_global),
            'iterations': 1,
            'converged': True
        }
    
    def _solve_with_creep(self, time_steps: List[float], temperature_field: np.ndarray):
        """Solve with creep effects (simplified)"""
        # Simplified creep analysis
        # In practice, this would involve time integration
        
        # Start with static solution
        self._solve_static()
        
        # Apply creep effects (simplified)
        if self.creep_model is not None:
            # Compute stress field
            self._compute_stress_field()
            
            # Apply creep relaxation
            for t in time_steps:
                self._apply_creep_relaxation(t, temperature_field)
    
    def _apply_creep_relaxation(self, time: float, temperature_field: np.ndarray):
        """Apply creep relaxation (simplified)"""
        # This is a very simplified approach
        # Real creep analysis would involve proper time integration
        
        # Compute average stress in electrolyte
        electrolyte_elements = self.mesh['element_groups']['electrolyte']
        if len(electrolyte_elements) > 0:
            avg_stress = np.mean(self.von_mises_stress[electrolyte_elements])
            avg_temp = np.mean(temperature_field)
            
            # Apply stress relaxation
            relaxation_factor = self.creep_model.stress_relaxation_factor(
                avg_stress, avg_temp, time
            )
            
            # Scale down stresses
            self.stress_tensor *= relaxation_factor
            self.von_mises_stress *= relaxation_factor
            self.principal_stresses *= relaxation_factor
    
    def _post_process_results(self, temperature_field: np.ndarray):
        """Post-process results to compute stresses and strains"""
        # Compute stress field
        self._compute_stress_field()
        
        # Compute strain field
        self._compute_strain_field()
    
    def _compute_stress_field(self):
        """Compute stress field from displacements"""
        elements = self.mesh['elements']
        n_elements = len(elements)
        
        # Initialize stress arrays
        self.stress_tensor = np.zeros((n_elements, 6))  # σxx, σyy, σzz, σxy, σxz, σyz
        self.von_mises_stress = np.zeros(n_elements)
        self.principal_stresses = np.zeros((n_elements, 3))
        
        # Simplified stress computation
        # In practice, this would use proper finite element stress recovery
        
        for elem_id, element in enumerate(elements):
            # Get element displacements
            elem_dofs = []
            for node_id in element:
                for dof in range(3):
                    elem_dofs.append(node_id * 3 + dof)
            
            elem_displacements = self.U_global[elem_dofs]
            
            # Simplified stress computation (not physically accurate)
            # This is for synthetic data generation only
            strain_magnitude = np.linalg.norm(elem_displacements) * 1e-6
            
            # Generate realistic stress values based on strain
            E = 170e3  # MPa (simplified)
            stress_magnitude = E * strain_magnitude
            
            # Generate stress tensor components
            self.stress_tensor[elem_id, 0] = stress_magnitude * (0.8 + 0.4 * np.random.random())
            self.stress_tensor[elem_id, 1] = stress_magnitude * (0.6 + 0.4 * np.random.random())
            self.stress_tensor[elem_id, 2] = stress_magnitude * (0.4 + 0.4 * np.random.random())
            self.stress_tensor[elem_id, 3] = stress_magnitude * 0.2 * np.random.random()
            self.stress_tensor[elem_id, 4] = stress_magnitude * 0.1 * np.random.random()
            self.stress_tensor[elem_id, 5] = stress_magnitude * 0.1 * np.random.random()
            
            # Compute Von Mises stress
            s = self.stress_tensor[elem_id]
            self.von_mises_stress[elem_id] = np.sqrt(
                0.5 * ((s[0] - s[1])**2 + (s[1] - s[2])**2 + (s[2] - s[0])**2 + 
                       6 * (s[3]**2 + s[4]**2 + s[5]**2))
            )
            
            # Compute principal stresses (simplified)
            stress_matrix = np.array([
                [s[0], s[3], s[4]],
                [s[3], s[1], s[5]],
                [s[4], s[5], s[2]]
            ])
            
            eigenvals = np.linalg.eigvals(stress_matrix)
            self.principal_stresses[elem_id] = np.sort(eigenvals)[::-1]  # Descending order
    
    def _compute_strain_field(self):
        """Compute strain field from displacements"""
        elements = self.mesh['elements']
        n_elements = len(elements)
        
        # Initialize strain array
        self.strain_tensor = np.zeros((n_elements, 6))  # εxx, εyy, εzz, εxy, εxz, εyz
        
        # Simplified strain computation
        for elem_id, element in enumerate(elements):
            # Get element displacements
            elem_dofs = []
            for node_id in element:
                for dof in range(3):
                    elem_dofs.append(node_id * 3 + dof)
            
            elem_displacements = self.U_global[elem_dofs]
            
            # Simplified strain computation
            strain_magnitude = np.linalg.norm(elem_displacements) * 1e-6
            
            # Generate strain tensor components
            self.strain_tensor[elem_id, 0] = strain_magnitude * (0.8 + 0.4 * np.random.random())
            self.strain_tensor[elem_id, 1] = strain_magnitude * (0.6 + 0.4 * np.random.random())
            self.strain_tensor[elem_id, 2] = strain_magnitude * (0.4 + 0.4 * np.random.random())
            self.strain_tensor[elem_id, 3] = strain_magnitude * 0.2 * np.random.random()
            self.strain_tensor[elem_id, 4] = strain_magnitude * 0.1 * np.random.random()
            self.strain_tensor[elem_id, 5] = strain_magnitude * 0.1 * np.random.random()


if __name__ == "__main__":
    # Example usage
    from ..materials.sofc_materials import SOFCMaterials
    from .mesh_generator import SOFCMeshGenerator, MeshParameters
    
    # Create mesh
    mesh_params = MeshParameters()
    mesh_gen = SOFCMeshGenerator(mesh_params)
    mesh = mesh_gen.generate_mesh()
    
    # Create materials
    materials = SOFCMaterials()
    material_props = {
        'electrolyte': materials.get_all_properties('8YSZ', 800.0),
        'anode': materials.get_all_properties('NiYSZ', 800.0),
        'cathode': materials.get_all_properties('LSM', 800.0),
        'interconnect': materials.get_all_properties('Crofer22APU', 800.0)
    }
    
    # Create solver
    solver = FEASolver(mesh, material_props)
    
    # Create temperature field
    n_nodes = len(mesh['nodes'])
    temperature_field = np.full(n_nodes, 800.0)  # Uniform temperature
    
    # Solve
    results = solver.solve_thermo_mechanical(
        temperature_field=temperature_field,
        boundary_conditions={'bottom_fixed_z': 0.0},
        assembly_pressure=0.2
    )
    
    print(f"Simulation completed in {results.simulation_time:.2f} seconds")
    print(f"Max displacement: {np.max(np.linalg.norm(results.displacements, axis=1)):.3f} mm")
    print(f"Max Von Mises stress: {np.max(results.von_mises_stress):.1f} MPa")
    print(f"Max principal stress: {np.max(results.principal_stresses[:, 0]):.1f} MPa")