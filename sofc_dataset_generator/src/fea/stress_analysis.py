"""
Stress Analysis Module

Analyzes stress fields in SOFC components to extract residual stress data
for use as target labels in ML models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter


@dataclass
class StressField:
    """Container for stress field data"""
    # Element coordinates (centroids)
    element_coords: np.ndarray  # [n_elements, 3] - x, y, z coordinates
    
    # Stress tensor components
    stress_tensor: np.ndarray  # [n_elements, 6] - σxx, σyy, σzz, σxy, σxz, σyz
    
    # Derived stress measures
    von_mises_stress: np.ndarray  # [n_elements] - Von Mises stress
    principal_stresses: np.ndarray  # [n_elements, 3] - σ1, σ2, σ3
    max_shear_stress: np.ndarray  # [n_elements] - Maximum shear stress
    
    # Stress invariants
    hydrostatic_stress: np.ndarray  # [n_elements] - Hydrostatic stress
    deviatoric_stress: np.ndarray  # [n_elements] - Deviatoric stress magnitude
    
    # Element group information
    element_groups: Dict[str, List[int]]  # Element indices for each group
    
    def get_stress_tensor(self, element_group: Optional[str] = None) -> np.ndarray:
        """Get stress tensor for specific element group or all elements"""
        if element_group is None:
            return self.stress_tensor
        else:
            element_ids = self.element_groups[element_group]
            return self.stress_tensor[element_ids]
    
    def get_von_mises_stress(self, element_group: Optional[str] = None) -> np.ndarray:
        """Get Von Mises stress for specific element group or all elements"""
        if element_group is None:
            return self.von_mises_stress
        else:
            element_ids = self.element_groups[element_group]
            return self.von_mises_stress[element_ids]
    
    def get_principal_stresses(self, element_group: Optional[str] = None) -> np.ndarray:
        """Get principal stresses for specific element group or all elements"""
        if element_group is None:
            return self.principal_stresses
        else:
            element_ids = self.element_groups[element_group]
            return self.principal_stresses[element_ids]
    
    def get_max_principal_stress(self, element_group: Optional[str] = None) -> float:
        """Get maximum principal stress for element group"""
        principal_stresses = self.get_principal_stresses(element_group)
        return np.max(principal_stresses[:, 0])  # σ1 is first column
    
    def get_stress_volume_average(self, element_group: Optional[str] = None) -> Dict[str, float]:
        """Get volume-averaged stress measures"""
        if element_group is None:
            stress_tensor = self.stress_tensor
            von_mises = self.von_mises_stress
            principal = self.principal_stresses
        else:
            element_ids = self.element_groups[element_group]
            stress_tensor = self.stress_tensor[element_ids]
            von_mises = self.von_mises_stress[element_ids]
            principal = self.principal_stresses[element_ids]
        
        return {
            'mean_von_mises': np.mean(von_mises),
            'max_von_mises': np.max(von_mises),
            'std_von_mises': np.std(von_mises),
            'mean_principal_1': np.mean(principal[:, 0]),
            'max_principal_1': np.max(principal[:, 0]),
            'mean_principal_2': np.mean(principal[:, 1]),
            'mean_principal_3': np.mean(principal[:, 2]),
            'mean_hydrostatic': np.mean(self.hydrostatic_stress[element_ids] if element_group else self.hydrostatic_stress)
        }


class StressAnalyzer:
    """Analyzes stress fields in SOFC components"""
    
    def __init__(self, mesh: Dict, simulation_results):
        self.mesh = mesh
        self.results = simulation_results
        self.stress_fields = {}
    
    def analyze_stress_field(self, element_group: Optional[str] = None) -> StressField:
        """
        Analyze stress field for specific element group or all elements
        
        Args:
            element_group: Element group to analyze ('electrolyte', 'anode', 'cathode', 'interconnect')
        """
        # Get element coordinates
        element_coords = self._get_element_coordinates()
        
        # Get stress data
        stress_tensor = self.results.stress_tensor
        von_mises_stress = self.results.von_mises_stress
        principal_stresses = self.results.principal_stresses
        
        # Filter by element group if specified
        if element_group is not None:
            element_ids = self.mesh['element_groups'][element_group]
            element_coords = element_coords[element_ids]
            stress_tensor = stress_tensor[element_ids]
            von_mises_stress = von_mises_stress[element_ids]
            principal_stresses = principal_stresses[element_ids]
            element_groups = {element_group: list(range(len(element_ids)))}
        else:
            element_groups = self.mesh['element_groups']
        
        # Compute additional stress measures
        max_shear_stress = self._compute_max_shear_stress(principal_stresses)
        hydrostatic_stress = self._compute_hydrostatic_stress(stress_tensor)
        deviatoric_stress = self._compute_deviatoric_stress(stress_tensor)
        
        # Create stress field
        stress_field = StressField(
            element_coords=element_coords,
            stress_tensor=stress_tensor,
            von_mises_stress=von_mises_stress,
            principal_stresses=principal_stresses,
            max_shear_stress=max_shear_stress,
            hydrostatic_stress=hydrostatic_stress,
            deviatoric_stress=deviatoric_stress,
            element_groups=element_groups
        )
        
        # Store stress field
        key = element_group if element_group else 'all'
        self.stress_fields[key] = stress_field
        
        return stress_field
    
    def _get_element_coordinates(self) -> np.ndarray:
        """Get element centroid coordinates"""
        elements = self.mesh['elements']
        nodes = self.mesh['nodes']
        
        element_coords = np.zeros((len(elements), 3))
        
        for i, element in enumerate(elements):
            # Get element node coordinates
            elem_nodes = nodes[element]
            # Compute centroid
            element_coords[i] = np.mean(elem_nodes, axis=0)
        
        return element_coords
    
    def _compute_max_shear_stress(self, principal_stresses: np.ndarray) -> np.ndarray:
        """Compute maximum shear stress from principal stresses"""
        # τ_max = (σ1 - σ3) / 2
        return (principal_stresses[:, 0] - principal_stresses[:, 2]) / 2.0
    
    def _compute_hydrostatic_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute hydrostatic stress (mean normal stress)"""
        # σ_h = (σxx + σyy + σzz) / 3
        return (stress_tensor[:, 0] + stress_tensor[:, 1] + stress_tensor[:, 2]) / 3.0
    
    def _compute_deviatoric_stress(self, stress_tensor: np.ndarray) -> np.ndarray:
        """Compute deviatoric stress magnitude"""
        # Compute hydrostatic stress
        hydrostatic = self._compute_hydrostatic_stress(stress_tensor)
        
        # Compute deviatoric stress tensor
        deviatoric_tensor = stress_tensor.copy()
        deviatoric_tensor[:, 0] -= hydrostatic  # σxx - σ_h
        deviatoric_tensor[:, 1] -= hydrostatic  # σyy - σ_h
        deviatoric_tensor[:, 2] -= hydrostatic  # σzz - σ_h
        
        # Compute deviatoric stress magnitude (second invariant)
        s = deviatoric_tensor
        deviatoric_magnitude = np.sqrt(
            0.5 * ((s[:, 0] - s[:, 1])**2 + (s[:, 1] - s[:, 2])**2 + (s[:, 2] - s[:, 0])**2 + 
                   6 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2))
        )
        
        return deviatoric_magnitude
    
    def analyze_stress_distribution(self, element_group: str = 'electrolyte') -> Dict[str, float]:
        """Analyze stress distribution for specific element group"""
        if element_group not in self.stress_fields:
            self.analyze_stress_field(element_group)
        
        stress_field = self.stress_fields[element_group]
        
        # Basic statistics
        stats = stress_field.get_stress_volume_average()
        
        # Additional analysis
        von_mises = stress_field.get_von_mises_stress()
        principal = stress_field.get_principal_stresses()
        
        # Stress concentration analysis
        stats.update({
            'stress_concentration_factor': np.max(von_mises) / np.mean(von_mises),
            'stress_ratio_max_mean': np.max(principal[:, 0]) / np.mean(principal[:, 0]),
            'tensile_compressive_ratio': np.sum(principal[:, 0] > 0) / len(principal[:, 0])
        })
        
        # Fracture risk assessment
        stats.update(self._assess_fracture_risk(stress_field))
        
        return stats
    
    def _assess_fracture_risk(self, stress_field: StressField) -> Dict[str, float]:
        """Assess fracture risk based on stress field"""
        # Material strength (8YSZ)
        flexural_strength = 165.0  # MPa
        
        # Get maximum principal stress
        max_principal = stress_field.get_max_principal_stress()
        
        # Safety factor
        safety_factor = flexural_strength / max_principal if max_principal > 0 else float('inf')
        
        # Fracture probability (simplified Weibull model)
        # P_f = 1 - exp(-(σ/σ0)^m) where σ0 and m are material parameters
        sigma_0 = 100.0  # Characteristic strength (MPa)
        m = 10.0  # Weibull modulus
        
        fracture_probability = 1.0 - np.exp(-(max_principal / sigma_0)**m)
        
        return {
            'max_principal_stress': max_principal,
            'safety_factor': safety_factor,
            'fracture_probability': fracture_probability,
            'fracture_risk_level': 'high' if safety_factor < 1.2 else 'medium' if safety_factor < 1.5 else 'low'
        }
    
    def generate_stress_maps(self, element_group: str = 'electrolyte', 
                           resolution: Tuple[int, int] = (100, 100)) -> Dict[str, np.ndarray]:
        """Generate 2D stress maps for visualization"""
        if element_group not in self.stress_fields:
            self.analyze_stress_field(element_group)
        
        stress_field = self.stress_fields[element_group]
        
        # Get element coordinates and stress data
        coords = stress_field.element_coords
        von_mises = stress_field.get_von_mises_stress()
        principal_1 = stress_field.get_principal_stresses()[:, 0]
        
        # Create regular grid
        x_min, x_max = np.min(coords[:, 0]), np.max(coords[:, 0])
        y_min, y_max = np.min(coords[:, 1]), np.max(coords[:, 1])
        
        x_grid = np.linspace(x_min, x_max, resolution[0])
        y_grid = np.linspace(y_min, y_max, resolution[1])
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate stress fields
        points = coords[:, :2]  # x, y coordinates
        
        von_mises_map = griddata(points, von_mises, (X_grid, Y_grid), method='cubic', fill_value=0.0)
        principal_1_map = griddata(points, principal_1, (X_grid, Y_grid), method='cubic', fill_value=0.0)
        
        return {
            'von_mises_map': von_mises_map,
            'principal_1_map': principal_1_map,
            'x_grid': x_grid,
            'y_grid': y_grid,
            'X_grid': X_grid,
            'Y_grid': Y_grid
        }
    
    def visualize_stress_field(self, element_group: str = 'electrolyte', 
                             save_path: Optional[str] = None):
        """Visualize stress field"""
        if element_group not in self.stress_fields:
            self.analyze_stress_field(element_group)
        
        stress_field = self.stress_fields[element_group]
        
        # Generate stress maps
        stress_maps = self.generate_stress_maps(element_group)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Von Mises stress map
        im1 = axes[0, 0].contourf(stress_maps['X_grid'], stress_maps['Y_grid'], 
                                 stress_maps['von_mises_map'], levels=20, cmap='plasma')
        axes[0, 0].set_title(f'{element_group.capitalize()} Von Mises Stress')
        axes[0, 0].set_xlabel('X (mm)')
        axes[0, 0].set_ylabel('Y (mm)')
        plt.colorbar(im1, ax=axes[0, 0], label='Stress (MPa)')
        
        # Principal stress map
        im2 = axes[0, 1].contourf(stress_maps['X_grid'], stress_maps['Y_grid'], 
                                 stress_maps['principal_1_map'], levels=20, cmap='RdBu_r')
        axes[0, 1].set_title(f'{element_group.capitalize()} Principal Stress σ₁')
        axes[0, 1].set_xlabel('X (mm)')
        axes[0, 1].set_ylabel('Y (mm)')
        plt.colorbar(im2, ax=axes[0, 1], label='Stress (MPa)')
        
        # Stress distribution histogram
        axes[1, 0].hist(stress_field.get_von_mises_stress(), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Von Mises Stress Distribution')
        axes[1, 0].set_xlabel('Stress (MPa)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Principal stress scatter
        principal = stress_field.get_principal_stresses()
        scatter = axes[1, 1].scatter(principal[:, 1], principal[:, 2], 
                                   c=principal[:, 0], cmap='plasma', s=20)
        axes[1, 1].set_title('Principal Stress Space (σ₂ vs σ₃)')
        axes[1, 1].set_xlabel('σ₂ (MPa)')
        axes[1, 1].set_ylabel('σ₃ (MPa)')
        plt.colorbar(scatter, ax=axes[1, 1], label='σ₁ (MPa)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stress visualization saved to {save_path}")
        
        plt.show()
    
    def export_stress_data(self, element_group: str = 'electrolyte', 
                          filename: str = 'stress_data.npz'):
        """Export stress data to file"""
        if element_group not in self.stress_fields:
            self.analyze_stress_field(element_group)
        
        stress_field = self.stress_fields[element_group]
        
        # Prepare data
        data = {
            'element_coords': stress_field.element_coords,
            'stress_tensor': stress_field.stress_tensor,
            'von_mises_stress': stress_field.von_mises_stress,
            'principal_stresses': stress_field.principal_stresses,
            'max_shear_stress': stress_field.max_shear_stress,
            'hydrostatic_stress': stress_field.hydrostatic_stress,
            'deviatoric_stress': stress_field.deviatoric_stress
        }
        
        # Add stress maps
        stress_maps = self.generate_stress_maps(element_group)
        data.update({
            'von_mises_map': stress_maps['von_mises_map'],
            'principal_1_map': stress_maps['principal_1_map'],
            'x_grid': stress_maps['x_grid'],
            'y_grid': stress_maps['y_grid']
        })
        
        # Save
        np.savez(filename, **data)
        print(f"Stress data exported to {filename}")


if __name__ == "__main__":
    # Example usage
    from .mesh_generator import SOFCMeshGenerator, MeshParameters
    from .fea_solver import FEASolver
    from ..materials.sofc_materials import SOFCMaterials
    
    # Create mesh and run simulation
    mesh_params = MeshParameters()
    mesh_gen = SOFCMeshGenerator(mesh_params)
    mesh = mesh_gen.generate_mesh()
    
    materials = SOFCMaterials()
    material_props = {
        'electrolyte': materials.get_all_properties('8YSZ', 800.0),
        'anode': materials.get_all_properties('NiYSZ', 800.0),
        'cathode': materials.get_all_properties('LSM', 800.0),
        'interconnect': materials.get_all_properties('Crofer22APU', 800.0)
    }
    
    solver = FEASolver(mesh, material_props)
    n_nodes = len(mesh['nodes'])
    temperature_field = np.full(n_nodes, 800.0)
    
    results = solver.solve_thermo_mechanical(
        temperature_field=temperature_field,
        boundary_conditions={'bottom_fixed_z': 0.0},
        assembly_pressure=0.2
    )
    
    # Analyze stress
    stress_analyzer = StressAnalyzer(mesh, results)
    
    # Analyze electrolyte stress
    electrolyte_stress = stress_analyzer.analyze_stress_field('electrolyte')
    print(f"Electrolyte max Von Mises stress: {np.max(electrolyte_stress.von_mises_stress):.1f} MPa")
    print(f"Electrolyte max principal stress: {electrolyte_stress.get_max_principal_stress():.1f} MPa")
    
    # Analyze stress distribution
    stats = stress_analyzer.analyze_stress_distribution('electrolyte')
    print("\nStress statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.3f}")
    
    # Visualize
    stress_analyzer.visualize_stress_field('electrolyte', 'stress_analysis.png')