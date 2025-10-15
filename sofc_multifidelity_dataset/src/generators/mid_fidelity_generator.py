"""Mid-fidelity data generator for SOFC modeling."""

import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import time
from scipy.ndimage import gaussian_filter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.physics_utils import SOFCPhysics, MaterialProperties, DegradationModels
from src.utils.mesh_utils import MeshGenerator, FieldInterpolator
from src.utils.data_utils import DatasetManager


class MidFidelityGenerator:
    """
    Generate mid-fidelity SOFC data using 2D/3D coarse grid models.
    Moderate computation (10s per sample) with spatial field outputs.
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
        
        # Define mesh resolution for mid-fidelity
        self.nx = 20
        self.ny = 20
        self.nz = 10
        
        # Generate base mesh
        self.mesh = self.mesh_gen.generate_3d_mesh(self.nx, self.ny, self.nz)
        
    def generate_sample(self, sample_id: int, seed: Optional[int] = None) -> Dict:
        """Generate single mid-fidelity sample with spatial fields."""
        if seed is not None:
            np.random.seed(seed + sample_id)
        
        start_time = time.time()
        
        # Sample input parameters (similar to low-fidelity)
        inputs = self._sample_inputs()
        
        # Compute spatially resolved outputs
        outputs = self._compute_spatial_outputs(inputs)
        
        # Add metadata
        metadata = {
            'sample_id': f'MF_{sample_id:06d}',
            'timestamp': time.time(),
            'computation_time': time.time() - start_time,
            'convergence_flag': True,
            'notes': 'Mid-fidelity 3D coarse grid model'
        }
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata
        }
    
    def _sample_inputs(self) -> Dict:
        """Sample input parameters (reuse from low-fidelity)."""
        inputs = {}
        
        # Operating conditions
        inputs['temperature'] = np.random.uniform(
            self.op_ranges['temperature']['min'],
            self.op_ranges['temperature']['max']
        )
        
        inputs['current_density'] = np.random.uniform(
            self.op_ranges['current_density']['min'],
            self.op_ranges['current_density']['max']
        )
        
        inputs['fuel_utilization'] = np.random.uniform(
            self.op_ranges['fuel_utilization']['min'],
            self.op_ranges['fuel_utilization']['max']
        )
        
        inputs['air_utilization'] = np.random.uniform(
            self.op_ranges['air_utilization']['min'],
            self.op_ranges['air_utilization']['max']
        )
        
        inputs['pressure'] = np.random.uniform(
            self.op_ranges['pressure']['min'],
            self.op_ranges['pressure']['max']
        )
        
        # Fuel composition
        fuel_comp = np.zeros(6)
        h2_range = self.op_ranges['inlet_compositions']['fuel']['H2']
        fuel_comp[0] = np.random.uniform(h2_range[0], h2_range[1])
        fuel_comp[1] = np.random.uniform(0.03, 0.3)
        remaining = 1.0 - fuel_comp[0] - fuel_comp[1]
        fuel_comp[2:] = np.random.dirichlet(np.ones(4)) * remaining
        inputs['fuel_composition'] = fuel_comp
        
        # Air composition
        inputs['air_composition'] = np.array([0.21, 0.79])
        
        # Geometry variations
        inputs['anode_thickness'] = self.geom['anode_thickness'] * np.random.uniform(0.9, 1.1)
        inputs['electrolyte_thickness'] = self.geom['electrolyte_thickness'] * np.random.uniform(0.9, 1.1)
        inputs['cathode_thickness'] = self.geom['cathode_thickness'] * np.random.uniform(0.9, 1.1)
        
        # Material properties
        inputs['anode_porosity'] = self.mat_props['anode']['porosity'] * np.random.uniform(0.9, 1.1)
        inputs['cathode_porosity'] = self.mat_props['cathode']['porosity'] * np.random.uniform(0.9, 1.1)
        
        # Degradation state
        inputs['operating_time'] = np.random.exponential(5000)
        inputs['thermal_cycles'] = np.random.poisson(10)
        inputs['redox_cycles'] = np.random.poisson(2)
        
        return inputs
    
    def _compute_spatial_outputs(self, inputs: Dict) -> Dict:
        """Compute spatially resolved outputs on coarse grid."""
        outputs = {}
        
        T_inlet = inputs['temperature']
        i_avg = inputs['current_density']
        p = inputs['pressure']
        
        # Initialize fields
        nx, ny, nz = self.nx, self.ny, self.nz
        
        # Temperature field with spatial variations
        T_field = self._compute_temperature_field(T_inlet, i_avg)
        outputs['temperature_field'] = T_field
        
        # Current density field
        i_field = self._compute_current_density_field(i_avg, T_field)
        outputs['current_density_field'] = i_field
        
        # Species concentration fields (6 species)
        species_field = self._compute_species_fields(inputs, T_field, i_field)
        outputs['species_concentration'] = species_field
        
        # Overpotentials (activation, ohmic, concentration)
        overpotentials = self._compute_overpotentials(T_field, i_field, species_field)
        outputs['overpotentials'] = overpotentials
        
        # Stress tensor fields
        stress_tensor = self._compute_stress_fields(T_field, inputs)
        outputs['stress_tensor'] = stress_tensor
        
        # Strain tensor fields (elastic, plastic, creep, thermal)
        strain_tensor = self._compute_strain_fields(stress_tensor, T_field, inputs)
        outputs['strain_tensor'] = strain_tensor
        
        # Damage field
        damage_field = self._compute_damage_field(stress_tensor, strain_tensor, inputs)
        outputs['damage_field'] = damage_field
        
        # Integrated performance metrics
        outputs['voltage'] = self._compute_cell_voltage(T_field, i_field, overpotentials)
        outputs['power_density'] = outputs['voltage'] * i_avg
        
        return outputs
    
    def _compute_temperature_field(self, T_inlet: float, i_avg: float) -> np.ndarray:
        """Compute 3D temperature field with heat generation."""
        nx, ny, nz = self.nx, self.ny, self.nz
        T_field = np.ones((nx, ny, nz)) * T_inlet
        
        # Add flow direction temperature gradient
        for i in range(nx):
            flow_factor = i / nx
            T_field[i, :, :] += 50 * flow_factor  # Temperature rise along flow
        
        # Add heat generation from electrochemical reactions
        # Higher temperature in active layers
        material_field = self.mesh['material_field']
        
        for k in range(nz):
            if material_field[0, 0, k] in [0, 2]:  # Anode or cathode
                heat_gen = i_avg * 0.2 / 1e5  # Simplified heat generation
                T_field[:, :, k] += heat_gen * 100
        
        # Add channel cooling effects
        for j in range(ny):
            if j % 4 < 2:  # Channel regions
                T_field[:, j, :] *= 0.98  # Slightly cooler in channels
        
        # Smooth the field
        T_field = gaussian_filter(T_field, sigma=0.5)
        
        return T_field
    
    def _compute_current_density_field(self, i_avg: float, T_field: np.ndarray) -> np.ndarray:
        """Compute current density distribution."""
        nx, ny, nz = self.nx, self.ny, self.nz
        i_field = np.ones((nx, ny, nz)) * i_avg
        
        # Current focusing under ribs
        for j in range(ny):
            if j % 4 >= 2:  # Rib regions
                i_field[:, j, :] *= 1.2  # Higher current under ribs
            else:  # Channel regions
                i_field[:, j, :] *= 0.8  # Lower current under channels
        
        # Temperature dependence
        T_norm = (T_field - T_field.min()) / (T_field.max() - T_field.min() + 1e-10)
        i_field *= (1 + 0.2 * T_norm)  # Higher current in hotter regions
        
        # Ensure current conservation
        i_field *= i_avg / np.mean(i_field)
        
        return i_field
    
    def _compute_species_fields(self, inputs: Dict, T_field: np.ndarray, 
                               i_field: np.ndarray) -> np.ndarray:
        """Compute species concentration fields."""
        nx, ny, nz = self.nx, self.ny, self.nz
        n_species = 6  # H2, H2O, O2, N2, CO, CO2
        species_field = np.zeros((n_species, nx, ny, nz))
        
        # Initial concentrations from inlet
        p = inputs['pressure']
        fuel_comp = inputs['fuel_composition']
        air_comp = inputs['air_composition']
        
        # H2 concentration (depletes along flow)
        for i in range(nx):
            utilization = inputs['fuel_utilization'] * (i + 1) / nx
            c_H2 = p * fuel_comp[0] * (1 - utilization) / (8.314 * T_field[i, :, :])
            species_field[0, i, :, :] = c_H2
        
        # H2O concentration (increases along flow)
        for i in range(nx):
            utilization = inputs['fuel_utilization'] * (i + 1) / nx
            c_H2O = p * (fuel_comp[1] + fuel_comp[0] * utilization) / (8.314 * T_field[i, :, :])
            species_field[1, i, :, :] = c_H2O
        
        # O2 concentration (depletes along flow)
        for i in range(nx):
            utilization = inputs['air_utilization'] * (i + 1) / nx
            c_O2 = p * air_comp[0] * (1 - utilization) / (8.314 * T_field[i, :, :])
            species_field[2, i, :, :] = c_O2
        
        # N2 concentration (constant)
        c_N2 = p * air_comp[1] / (8.314 * T_field)
        species_field[3, :, :, :] = c_N2
        
        # CO and CO2 (if present)
        if fuel_comp[2] > 0:
            species_field[4, :, :, :] = p * fuel_comp[2] / (8.314 * T_field)
        if fuel_comp[3] > 0:
            species_field[5, :, :, :] = p * fuel_comp[3] / (8.314 * T_field)
        
        return species_field
    
    def _compute_overpotentials(self, T_field: np.ndarray, i_field: np.ndarray,
                               species_field: np.ndarray) -> np.ndarray:
        """Compute overpotential fields."""
        nx, ny, nz = self.nx, self.ny, self.nz
        overpotentials = np.zeros((3, nx, ny, nz))  # activation, ohmic, concentration
        
        # Activation overpotential
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    i_local = i_field[i, j, k]
                    
                    if self.mesh['material_field'][i, j, k] == 0:  # Anode
                        i0 = 1000 * np.exp(-60000 / (8.314 * T))
                        if i_local > 0:
                            eta = (8.314 * T) / (2 * 96485) * np.log(i_local / i0)
                            overpotentials[0, i, j, k] = eta
                    
                    elif self.mesh['material_field'][i, j, k] == 2:  # Cathode
                        i0 = 100 * np.exp(-120000 / (8.314 * T))
                        if i_local > 0:
                            eta = (8.314 * T) / (2 * 96485) * np.log(i_local / i0)
                            overpotentials[0, i, j, k] = eta
        
        # Ohmic overpotential
        for k in range(nz):
            if self.mesh['material_field'][0, 0, k] == 1:  # Electrolyte
                sigma = self.materials.ionic_conductivity_YSZ(T_field[:, :, k])
                thickness = self.geom['electrolyte_thickness'] * 1e-6 / nz
                overpotentials[1, :, :, k] = i_field[:, :, k] * thickness / sigma
        
        # Concentration overpotential
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    i_local = i_field[i, j, k]
                    
                    # Limiting current based on local species concentration
                    if self.mesh['material_field'][i, j, k] == 0:  # Anode
                        c_H2 = species_field[0, i, j, k]
                        D_eff = 1e-5 * (T / 1073) ** 1.5
                        L = self.geom['anode_thickness'] * 1e-6
                        i_lim = 2 * 96485 * D_eff * c_H2 / L
                        
                        if i_local < 0.95 * i_lim:
                            eta = (8.314 * T) / (2 * 96485) * np.log(1 - i_local / i_lim)
                            overpotentials[2, i, j, k] = eta
        
        return overpotentials
    
    def _compute_stress_fields(self, T_field: np.ndarray, inputs: Dict) -> np.ndarray:
        """Compute stress tensor fields."""
        nx, ny, nz = self.nx, self.ny, self.nz
        stress_tensor = np.zeros((6, nx, ny, nz))  # xx, yy, zz, xy, xz, yz
        
        T_ref = 298  # Reference temperature
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    mat_id = self.mesh['material_field'][i, j, k]
                    
                    # Get material properties
                    if mat_id == 0:  # Anode
                        E = self.materials.elastic_modulus_Ni(T)
                        CTE = self.materials.CTE_Ni(T)
                        nu = self.mat_props['anode']['poisson_ratio']
                    elif mat_id == 1:  # Electrolyte
                        E = self.materials.elastic_modulus_YSZ(T)
                        CTE = self.materials.CTE_YSZ(T)
                        nu = self.mat_props['electrolyte']['poisson_ratio']
                    else:  # Cathode
                        E = self.mat_props['cathode']['elastic_modulus']
                        CTE = self.mat_props['cathode']['CTE']
                        nu = self.mat_props['cathode']['poisson_ratio']
                    
                    # Thermal stress (simplified)
                    thermal_strain = CTE * (T - T_ref)
                    
                    # Constraint factor for multilayer
                    if k > 0 and k < nz - 1:
                        # Interface constraint
                        constraint = 0.8
                    else:
                        constraint = 0.5
                    
                    # Principal stresses
                    sigma_thermal = E * thermal_strain / (1 - nu) * constraint
                    stress_tensor[0, i, j, k] = sigma_thermal  # σxx
                    stress_tensor[1, i, j, k] = sigma_thermal * 0.9  # σyy
                    stress_tensor[2, i, j, k] = sigma_thermal * 0.3  # σzz
                    
                    # Shear stresses near interfaces
                    if k > 0 and self.mesh['material_field'][i, j, k] != self.mesh['material_field'][i, j, k-1]:
                        stress_tensor[5, i, j, k] = sigma_thermal * 0.1  # τyz
        
        # Add bending stresses for non-uniform temperature
        T_gradient = np.gradient(T_field, axis=0)[0]
        bending_stress = 1e7 * T_gradient  # Simplified bending
        stress_tensor[0, :, :, :] += bending_stress
        
        return stress_tensor
    
    def _compute_strain_fields(self, stress_tensor: np.ndarray, T_field: np.ndarray,
                              inputs: Dict) -> np.ndarray:
        """Compute strain tensor fields."""
        nx, ny, nz = self.nx, self.ny, self.nz
        # 4 types x 6 components = 24 total
        strain_tensor = np.zeros((24, nx, ny, nz))
        
        T_ref = 298
        t_hours = inputs['operating_time']
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    T = T_field[i, j, k]
                    mat_id = self.mesh['material_field'][i, j, k]
                    
                    # Get material properties
                    if mat_id == 0:  # Anode
                        E = self.materials.elastic_modulus_Ni(T)
                        CTE = self.materials.CTE_Ni(T)
                        nu = self.mat_props['anode']['poisson_ratio']
                    elif mat_id == 1:  # Electrolyte
                        E = self.materials.elastic_modulus_YSZ(T)
                        CTE = self.materials.CTE_YSZ(T)
                        nu = self.mat_props['electrolyte']['poisson_ratio']
                    else:  # Cathode
                        E = self.mat_props['cathode']['elastic_modulus']
                        CTE = self.mat_props['cathode']['CTE']
                        nu = self.mat_props['cathode']['poisson_ratio']
                    
                    # Elastic strain (first 6 components)
                    sigma_xx = stress_tensor[0, i, j, k]
                    sigma_yy = stress_tensor[1, i, j, k]
                    sigma_zz = stress_tensor[2, i, j, k]
                    
                    strain_tensor[0, i, j, k] = (sigma_xx - nu * (sigma_yy + sigma_zz)) / E
                    strain_tensor[1, i, j, k] = (sigma_yy - nu * (sigma_xx + sigma_zz)) / E
                    strain_tensor[2, i, j, k] = (sigma_zz - nu * (sigma_xx + sigma_yy)) / E
                    
                    # Shear strains
                    G = E / (2 * (1 + nu))
                    strain_tensor[3, i, j, k] = stress_tensor[3, i, j, k] / G
                    strain_tensor[4, i, j, k] = stress_tensor[4, i, j, k] / G
                    strain_tensor[5, i, j, k] = stress_tensor[5, i, j, k] / G
                    
                    # Thermal strain (components 6-11)
                    thermal_strain = CTE * (T - T_ref)
                    strain_tensor[6:9, i, j, k] = thermal_strain
                    
                    # Creep strain (components 12-17)
                    if t_hours > 0:
                        von_mises = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                                   (sigma_yy - sigma_zz)**2 + 
                                                   (sigma_zz - sigma_xx)**2))
                        if von_mises > 0:
                            creep_params = self.config['degradation']['creep']
                            creep_rate = self.physics.creep_strain_rate(von_mises, T, creep_params)
                            creep_strain = creep_rate * t_hours * 3600
                            strain_tensor[12:15, i, j, k] = creep_strain
                    
                    # Plastic strain (components 18-23) - simplified
                    yield_stress = 100e6  # Pa
                    if von_mises > yield_stress:
                        plastic_strain = (von_mises - yield_stress) / E * 0.1
                        strain_tensor[18:21, i, j, k] = plastic_strain
        
        return strain_tensor
    
    def _compute_damage_field(self, stress_tensor: np.ndarray, strain_tensor: np.ndarray,
                             inputs: Dict) -> np.ndarray:
        """Compute damage field based on stress/strain history."""
        nx, ny, nz = self.nx, self.ny, self.nz
        damage_field = np.zeros((nx, ny, nz))
        
        t_hours = inputs['operating_time']
        n_thermal = inputs['thermal_cycles']
        n_redox = inputs['redox_cycles']
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # Von Mises stress
                    sigma_xx = stress_tensor[0, i, j, k]
                    sigma_yy = stress_tensor[1, i, j, k]
                    sigma_zz = stress_tensor[2, i, j, k]
                    tau_xy = stress_tensor[3, i, j, k]
                    tau_xz = stress_tensor[4, i, j, k]
                    tau_yz = stress_tensor[5, i, j, k]
                    
                    von_mises = np.sqrt(0.5 * ((sigma_xx - sigma_yy)**2 + 
                                               (sigma_yy - sigma_zz)**2 + 
                                               (sigma_zz - sigma_xx)**2 + 
                                               6 * (tau_xy**2 + tau_xz**2 + tau_yz**2)))
                    
                    # Fatigue damage (Coffin-Manson)
                    total_strain = np.sum(np.abs(strain_tensor[:6, i, j, k]))
                    if n_thermal > 0:
                        N_f = 1000 / (total_strain * 100) ** 2  # Cycles to failure
                        damage_thermal = n_thermal / N_f
                    else:
                        damage_thermal = 0
                    
                    # Creep damage (Larson-Miller)
                    creep_strain = np.sum(strain_tensor[12:18, i, j, k])
                    if creep_strain > 0:
                        damage_creep = creep_strain / 0.02  # 2% creep strain limit
                    else:
                        damage_creep = 0
                    
                    # Redox damage
                    if n_redox > 0 and self.mesh['material_field'][i, j, k] == 0:  # Anode
                        damage_redox = self.degradation.anode_reoxidation_damage(n_redox)
                    else:
                        damage_redox = 0
                    
                    # Interface delamination risk
                    if k > 0 and self.mesh['material_field'][i, j, k] != self.mesh['material_field'][i, j, k-1]:
                        # At interface
                        tau_yz_val = stress_tensor[5, i, j, k]  # Get shear stress
                        G = tau_yz_val**2 / (2 * 100e9)  # Energy release rate
                        G_c = self.config['degradation']['interface']['critical_energy_release_rate']
                        damage_interface = G / G_c
                    else:
                        damage_interface = 0
                    
                    # Combined damage (linear summation)
                    damage_field[i, j, k] = min(1.0, damage_thermal + damage_creep + 
                                               damage_redox + damage_interface)
        
        # Smooth damage field
        damage_field = gaussian_filter(damage_field, sigma=0.3)
        
        return damage_field
    
    def _compute_cell_voltage(self, T_field: np.ndarray, i_field: np.ndarray,
                             overpotentials: np.ndarray) -> float:
        """Compute average cell voltage from field data."""
        T_avg = np.mean(T_field)
        
        # Average partial pressures (simplified)
        p = 101325  # Pa
        pH2 = p * 0.8
        pH2O = p * 0.15
        pO2 = p * 0.21
        
        # Nernst voltage
        E_nernst = self.physics.nernst_voltage(T_avg, pH2, pH2O, pO2)
        
        # Average overpotentials
        eta_act = np.mean(overpotentials[0, :, :, :])
        eta_ohm = np.mean(overpotentials[1, :, :, :])
        eta_conc = np.mean(overpotentials[2, :, :, :])
        
        # Cell voltage
        V = E_nernst - eta_act - eta_ohm - abs(eta_conc)
        
        return max(0, V)
    
    def generate_dataset(self, n_samples: int, output_path: str = None) -> None:
        """Generate complete mid-fidelity dataset."""
        print(f"\nGenerating {n_samples} mid-fidelity samples...")
        
        # Initialize data manager
        if output_path is None:
            output_path = "./data"
        
        manager = DatasetManager(output_path)
        
        # Create HDF5 file structure
        spatial_dims = (self.nx, self.ny, self.nz)
        f = manager.create_hdf5_structure('mid_fidelity', n_samples, 
                                         spatial_dims, self.config)
        
        try:
            # Generate samples
            for idx in tqdm(range(n_samples), desc="Mid-Fidelity Generation"):
                sample = self.generate_sample(idx)
                manager.write_sample(f, idx, sample)
                
                # Flush periodically
                if idx % 10 == 0:
                    f.flush()
            
            print(f"Successfully generated {n_samples} mid-fidelity samples")
            
        finally:
            f.close()
        
        # Validate dataset
        validation_report = manager.validate_dataset('mid_fidelity')
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
    generator = MidFidelityGenerator(config)
    
    # Generate small test dataset
    generator.generate_dataset(10)


if __name__ == "__main__":
    main()