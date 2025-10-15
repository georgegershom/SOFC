"""
Finite Element Analysis (FEA) Solver for SOFC Thermo-Mechanical Simulation

This module implements a coupled thermo-mechanical FEA solver for simulating
the SOFC manufacturing process including sintering, cooling, and residual stress development.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

# FEA and mesh libraries
try:
    import dolfin as df
    import fenics as fe
    FENICS_AVAILABLE = True
except ImportError:
    FENICS_AVAILABLE = False
    print("Warning: FEniCS not available. Using simplified FEA implementation.")

import meshio
import pyvista as pv
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

from ..materials.material_models import SOFCMaterialModel
from ..geometry.mesh_generator import SOFCMeshGenerator


@dataclass
class FEAConfiguration:
    """Configuration for FEA simulation."""
    # Mesh parameters
    mesh_resolution: float = 0.5  # mm
    element_order: int = 1
    
    # Time stepping
    n_time_steps: int = 100
    time_step_adaptive: bool = True
    max_time_step: float = 10.0  # minutes
    min_time_step: float = 0.1   # minutes
    
    # Solver parameters
    solver_type: str = "direct"  # direct, iterative
    tolerance: float = 1e-6
    max_iterations: int = 1000
    
    # Thermal analysis
    enable_thermal: bool = True
    thermal_bc_type: str = "prescribed_temperature"  # prescribed_temperature, convection
    
    # Mechanical analysis
    enable_mechanical: bool = True
    mechanical_bc_type: str = "free_standing"  # free_standing, supported
    
    # Coupling
    coupling_type: str = "sequential"  # sequential, monolithic
    coupling_iterations: int = 5
    
    # Output
    output_frequency: int = 10  # Every N time steps
    save_intermediate: bool = False


@dataclass
class SimulationResults:
    """Container for FEA simulation results."""
    # Mesh and geometry
    mesh: Any = None
    coordinates: np.ndarray = None
    connectivity: np.ndarray = None
    
    # Field results
    temperature_history: List[np.ndarray] = field(default_factory=list)
    displacement_history: List[np.ndarray] = field(default_factory=list)
    stress_history: List[np.ndarray] = field(default_factory=list)
    strain_history: List[np.ndarray] = field(default_factory=list)
    
    # Final results
    final_temperature: np.ndarray = None
    final_displacement: np.ndarray = None
    final_stress: np.ndarray = None
    final_strain: np.ndarray = None
    
    # Derived quantities
    warp_field: np.ndarray = None
    residual_stress_field: np.ndarray = None
    
    # Metadata
    time_points: np.ndarray = None
    simulation_time: float = 0.0
    convergence_history: List[float] = field(default_factory=list)


class SOFCFEASolver:
    """Finite Element Analysis solver for SOFC manufacturing simulation."""
    
    def __init__(self, config: FEAConfiguration):
        """Initialize FEA solver with configuration."""
        self.config = config
        self.material_model = None
        self.mesh_generator = None
        self.results = SimulationResults()
        
        # Check FEniCS availability
        if not FENICS_AVAILABLE and self.config.solver_type == "fenics":
            print("Warning: FEniCS not available, falling back to simplified solver")
            self.config.solver_type = "simplified"
    
    def setup_problem(self, doe_parameters: Dict[str, Any], 
                     material_model: SOFCMaterialModel,
                     mesh_generator: SOFCMeshGenerator):
        """Setup the FEA problem with DOE parameters."""
        self.doe_parameters = doe_parameters
        self.material_model = material_model
        self.mesh_generator = mesh_generator
        
        # Generate mesh
        self.mesh = mesh_generator.generate_mesh(doe_parameters)
        self.results.mesh = self.mesh
        self.results.coordinates = self.mesh.points
        self.results.connectivity = self.mesh.cells[0].data
        
        # Setup thermal profile
        self._setup_thermal_profile()
        
        # Initialize field variables
        self._initialize_fields()
    
    def _setup_thermal_profile(self):
        """Setup temperature profile for the simulation."""
        # Extract thermal parameters
        peak_temp = self.doe_parameters.get('thermal.peak_temperature', 1400.0)  # Celsius
        heating_rate = self.doe_parameters.get('thermal.heating_rate', 2.0)  # C/min
        cooling_rate = self.doe_parameters.get('thermal.cooling_rate', 1.0)   # C/min
        dwell_time = self.doe_parameters.get('thermal.dwell_time', 120.0)     # minutes
        
        room_temp = 25.0  # Celsius
        
        # Create time-temperature profile
        heating_time = (peak_temp - room_temp) / heating_rate
        cooling_time = (peak_temp - room_temp) / cooling_rate
        total_time = heating_time + dwell_time + cooling_time
        
        # Time points
        time_heating = np.linspace(0, heating_time, int(heating_time / self.config.max_time_step) + 1)
        time_dwell = np.linspace(heating_time, heating_time + dwell_time, 
                                int(dwell_time / self.config.max_time_step) + 1)
        time_cooling = np.linspace(heating_time + dwell_time, total_time,
                                  int(cooling_time / self.config.max_time_step) + 1)
        
        # Temperature points
        temp_heating = room_temp + heating_rate * time_heating
        temp_dwell = np.full_like(time_dwell, peak_temp)
        temp_cooling = peak_temp - cooling_rate * (time_cooling - heating_time - dwell_time)
        
        # Combine profiles
        self.time_profile = np.concatenate([time_heating, time_dwell[1:], time_cooling[1:]])
        self.temperature_profile = np.concatenate([temp_heating, temp_dwell[1:], temp_cooling[1:]])
        
        # Create interpolation function
        self.temperature_function = interp1d(self.time_profile, self.temperature_profile, 
                                           kind='linear', bounds_error=False, fill_value='extrapolate')
        
        self.results.time_points = self.time_profile
    
    def _initialize_fields(self):
        """Initialize field variables."""
        n_nodes = len(self.results.coordinates)
        n_elements = len(self.results.connectivity)
        
        # Initialize temperature field
        self.temperature = np.full(n_nodes, 25.0)  # Room temperature
        
        # Initialize displacement field (3D)
        self.displacement = np.zeros((n_nodes, 3))
        
        # Initialize stress and strain fields (6 components: xx, yy, zz, xy, xz, yz)
        self.stress = np.zeros((n_elements, 6))
        self.strain = np.zeros((n_elements, 6))
    
    def solve(self) -> SimulationResults:
        """Solve the coupled thermo-mechanical problem."""
        print("Starting FEA simulation...")
        start_time = time.time()
        
        # Time stepping loop
        for i, current_time in enumerate(self.time_profile):
            print(f"Time step {i+1}/{len(self.time_profile)}: t = {current_time:.2f} min")
            
            # Update temperature boundary conditions
            current_temp = self.temperature_function(current_time)
            
            if self.config.enable_thermal:
                self._solve_thermal_step(current_time, current_temp)
            
            if self.config.enable_mechanical:
                self._solve_mechanical_step(current_time)
            
            # Store results
            if i % self.config.output_frequency == 0 or i == len(self.time_profile) - 1:
                self._store_results(current_time)
        
        # Finalize results
        self._finalize_results()
        
        self.results.simulation_time = time.time() - start_time
        print(f"Simulation completed in {self.results.simulation_time:.2f} seconds")
        
        return self.results
    
    def _solve_thermal_step(self, current_time: float, prescribed_temp: float):
        """Solve thermal analysis for current time step."""
        if FENICS_AVAILABLE and self.config.solver_type == "fenics":
            self._solve_thermal_fenics(current_time, prescribed_temp)
        else:
            self._solve_thermal_simplified(current_time, prescribed_temp)
    
    def _solve_thermal_fenics(self, current_time: float, prescribed_temp: float):
        """Solve thermal problem using FEniCS."""
        # This would implement the full FEniCS thermal solver
        # For now, use simplified approach
        self._solve_thermal_simplified(current_time, prescribed_temp)
    
    def _solve_thermal_simplified(self, current_time: float, prescribed_temp: float):
        """Simplified thermal analysis assuming uniform temperature."""
        # For simplified analysis, assume uniform temperature distribution
        # In reality, this would solve the heat equation with proper boundary conditions
        self.temperature[:] = prescribed_temp
    
    def _solve_mechanical_step(self, current_time: float):
        """Solve mechanical analysis for current time step."""
        if FENICS_AVAILABLE and self.config.solver_type == "fenics":
            self._solve_mechanical_fenics(current_time)
        else:
            self._solve_mechanical_simplified(current_time)
    
    def _solve_mechanical_fenics(self, current_time: float):
        """Solve mechanical problem using FEniCS."""
        # This would implement the full FEniCS mechanical solver
        # For now, use simplified approach
        self._solve_mechanical_simplified(current_time)
    
    def _solve_mechanical_simplified(self, current_time: float):
        """Simplified mechanical analysis using analytical approximations."""
        # Get material properties at current temperature
        avg_temp = np.mean(self.temperature)
        
        # Get material properties for each layer
        properties = self.material_model.get_properties_at_temperature(
            avg_temp, self.doe_parameters
        )
        
        # Calculate thermal strain
        thermal_strain = self._calculate_thermal_strain(avg_temp)
        
        # Calculate shrinkage strain (simplified)
        shrinkage_strain = self._calculate_shrinkage_strain(current_time)
        
        # Total strain
        total_strain = thermal_strain + shrinkage_strain
        
        # Calculate stress using simplified constitutive model
        self._calculate_stress_from_strain(total_strain, properties)
        
        # Calculate displacement from strain (simplified)
        self._calculate_displacement_from_strain(total_strain)
    
    def _calculate_thermal_strain(self, temperature: float) -> np.ndarray:
        """Calculate thermal strain based on temperature change."""
        reference_temp = 25.0  # Room temperature
        temp_change = temperature - reference_temp
        
        # Get thermal expansion coefficients for each layer
        n_elements = len(self.results.connectivity)
        thermal_strain = np.zeros((n_elements, 6))
        
        # Simplified: assume isotropic thermal expansion
        alpha = 10e-6  # Typical thermal expansion coefficient for ceramics (1/K)
        thermal_strain_value = alpha * temp_change
        
        # Apply to normal strain components
        thermal_strain[:, 0] = thermal_strain_value  # xx
        thermal_strain[:, 1] = thermal_strain_value  # yy
        thermal_strain[:, 2] = thermal_strain_value  # zz
        
        return thermal_strain
    
    def _calculate_shrinkage_strain(self, current_time: float) -> np.ndarray:
        """Calculate shrinkage strain during sintering."""
        n_elements = len(self.results.connectivity)
        shrinkage_strain = np.zeros((n_elements, 6))
        
        # Get shrinkage parameters
        differential_shrinkage = self.doe_parameters.get('manufacturing.differential_shrinkage.anode_electrolyte', 1.0)
        
        # Simplified shrinkage model based on temperature and time
        current_temp = np.mean(self.temperature)
        if current_temp > 1000.0:  # Sintering temperature threshold
            # Calculate shrinkage based on temperature and time
            max_shrinkage = 0.15  # 15% maximum shrinkage
            temp_factor = (current_temp - 1000.0) / 400.0  # Normalized temperature
            time_factor = np.tanh(current_time / 60.0)  # Time-dependent factor
            
            shrinkage = max_shrinkage * temp_factor * time_factor * differential_shrinkage
            
            # Apply to normal strain components (compressive)
            shrinkage_strain[:, 0] = -shrinkage  # xx
            shrinkage_strain[:, 1] = -shrinkage  # yy
            shrinkage_strain[:, 2] = -shrinkage  # zz
        
        return shrinkage_strain
    
    def _calculate_stress_from_strain(self, total_strain: np.ndarray, properties: Dict[str, Any]):
        """Calculate stress from total strain using constitutive model."""
        # Simplified elastic constitutive model
        E = properties.get('elastic_modulus', 200e9)  # Pa
        nu = properties.get('poisson_ratio', 0.3)
        
        # Elastic stiffness matrix (plane stress)
        D = E / (1 - nu**2) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        
        # Calculate stress (simplified 2D)
        n_elements = total_strain.shape[0]
        for i in range(n_elements):
            strain_vec = total_strain[i, :3]  # xx, yy, xy components
            stress_vec = D @ strain_vec
            self.stress[i, :3] = stress_vec
        
        # Store strain
        self.strain = total_strain.copy()
    
    def _calculate_displacement_from_strain(self, total_strain: np.ndarray):
        """Calculate displacement field from strain (simplified)."""
        # This is a very simplified approach
        # In reality, this would require solving the equilibrium equations
        
        coordinates = self.results.coordinates
        n_nodes = coordinates.shape[0]
        
        # Calculate average strain
        avg_strain = np.mean(total_strain, axis=0)
        
        # Apply strain to coordinates to get displacement
        center = np.mean(coordinates, axis=0)
        
        for i in range(n_nodes):
            rel_pos = coordinates[i] - center
            
            # Apply strain to relative position
            self.displacement[i, 0] = avg_strain[0] * rel_pos[0]  # xx strain
            self.displacement[i, 1] = avg_strain[1] * rel_pos[1]  # yy strain
            self.displacement[i, 2] = avg_strain[2] * rel_pos[2]  # zz strain
    
    def _store_results(self, current_time: float):
        """Store current field results."""
        self.results.temperature_history.append(self.temperature.copy())
        self.results.displacement_history.append(self.displacement.copy())
        self.results.stress_history.append(self.stress.copy())
        self.results.strain_history.append(self.strain.copy())
    
    def _finalize_results(self):
        """Finalize simulation results and extract key quantities."""
        # Store final fields
        self.results.final_temperature = self.temperature.copy()
        self.results.final_displacement = self.displacement.copy()
        self.results.final_stress = self.stress.copy()
        self.results.final_strain = self.strain.copy()
        
        # Extract warp field (deformed coordinates)
        self.results.warp_field = self.results.coordinates + self.results.final_displacement
        
        # Extract residual stress field (full 3D stress tensor)
        self.results.residual_stress_field = self.results.final_stress.copy()
    
    def export_results(self, output_path: Union[str, Path], format: str = "vtk"):
        """Export simulation results to file."""
        output_path = Path(output_path)
        
        if format.lower() == "vtk":
            self._export_vtk(output_path)
        elif format.lower() == "hdf5":
            self._export_hdf5(output_path)
        elif format.lower() == "json":
            self._export_json(output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_vtk(self, output_path: Path):
        """Export results to VTK format."""
        # Create PyVista mesh
        mesh = pv.UnstructuredGrid(self.results.connectivity, 
                                  np.full(len(self.results.connectivity), 10),  # Cell type (tetrahedron)
                                  self.results.warp_field)
        
        # Add field data
        mesh.point_data["Temperature"] = self.results.final_temperature
        mesh.point_data["Displacement"] = self.results.final_displacement
        mesh.cell_data["Stress"] = self.results.final_stress
        mesh.cell_data["Strain"] = self.results.final_strain
        
        # Save mesh
        mesh.save(str(output_path.with_suffix('.vtk')))
    
    def _export_hdf5(self, output_path: Path):
        """Export results to HDF5 format."""
        import h5py
        
        with h5py.File(output_path.with_suffix('.h5'), 'w') as f:
            # Mesh data
            f.create_dataset('coordinates', data=self.results.coordinates)
            f.create_dataset('connectivity', data=self.results.connectivity)
            f.create_dataset('warp_field', data=self.results.warp_field)
            
            # Field data
            f.create_dataset('final_temperature', data=self.results.final_temperature)
            f.create_dataset('final_displacement', data=self.results.final_displacement)
            f.create_dataset('final_stress', data=self.results.final_stress)
            f.create_dataset('final_strain', data=self.results.final_strain)
            
            # Time history (if available)
            if self.results.time_points is not None:
                f.create_dataset('time_points', data=self.results.time_points)
    
    def _export_json(self, output_path: Path):
        """Export metadata to JSON format."""
        metadata = {
            'simulation_time': self.results.simulation_time,
            'n_nodes': len(self.results.coordinates),
            'n_elements': len(self.results.connectivity),
            'doe_parameters': self.doe_parameters,
            'config': {
                'mesh_resolution': self.config.mesh_resolution,
                'n_time_steps': self.config.n_time_steps,
                'solver_type': self.config.solver_type
            }
        }
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)


def create_fea_solver(config: Optional[FEAConfiguration] = None) -> SOFCFEASolver:
    """Factory function to create FEA solver."""
    if config is None:
        config = FEAConfiguration()
    return SOFCFEASolver(config)


if __name__ == "__main__":
    # Test FEA solver creation
    config = FEAConfiguration(mesh_resolution=1.0, n_time_steps=50)
    solver = create_fea_solver(config)
    print(f"Created FEA solver with config: {config}")