"""Low-fidelity data generator for SOFC modeling."""

import numpy as np
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.utils.physics_utils import SOFCPhysics, MaterialProperties, DegradationModels
from src.utils.data_utils import DatasetManager


class LowFidelityGenerator:
    """
    Generate low-fidelity SOFC data using lumped parameter models.
    Fast computation (0.1s per sample) with 0D/1D averaged outputs.
    """
    
    def __init__(self, config: Dict):
        """Initialize generator with configuration."""
        self.config = config
        self.physics = SOFCPhysics(config)
        self.materials = MaterialProperties()
        self.degradation = DegradationModels()
        
        # Extract operating ranges
        self.op_ranges = config['operating_conditions']
        self.geom = config['geometry']
        self.mat_props = config['materials']
        
    def generate_sample(self, sample_id: int, seed: Optional[int] = None) -> Dict:
        """Generate single low-fidelity sample."""
        if seed is not None:
            np.random.seed(seed + sample_id)
        
        start_time = time.time()
        
        # Sample input parameters
        inputs = self._sample_inputs()
        
        # Compute outputs
        outputs = self._compute_outputs(inputs)
        
        # Add metadata
        metadata = {
            'sample_id': f'LF_{sample_id:06d}',
            'timestamp': time.time(),
            'computation_time': time.time() - start_time,
            'convergence_flag': True,
            'notes': 'Low-fidelity lumped model'
        }
        
        return {
            'inputs': inputs,
            'outputs': outputs,
            'metadata': metadata
        }
    
    def _sample_inputs(self) -> Dict:
        """Sample input parameters from ranges."""
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
        
        # Fuel composition (H2, H2O, CO, CO2, CH4, N2)
        fuel_comp = np.zeros(6)
        h2_range = self.op_ranges['inlet_compositions']['fuel']['H2']
        fuel_comp[0] = np.random.uniform(h2_range[0], h2_range[1])  # H2
        
        h2o_range = self.op_ranges['inlet_compositions']['fuel']['H2O']
        fuel_comp[1] = np.random.uniform(h2o_range[0], h2o_range[1])  # H2O
        
        # Normalize to sum to 1
        remaining = 1.0 - fuel_comp[0] - fuel_comp[1]
        fuel_comp[2] = np.random.uniform(0, min(0.2, remaining))  # CO
        fuel_comp[3] = np.random.uniform(0, min(0.1, remaining - fuel_comp[2]))  # CO2
        fuel_comp[4] = np.random.uniform(0, min(0.05, remaining - fuel_comp[2] - fuel_comp[3]))  # CH4
        fuel_comp[5] = remaining - fuel_comp[2] - fuel_comp[3] - fuel_comp[4]  # N2
        
        inputs['fuel_composition'] = fuel_comp
        
        # Air composition (O2, N2)
        inputs['air_composition'] = np.array([0.21, 0.79])
        
        # Geometry variations (±10% from nominal)
        inputs['anode_thickness'] = self.geom['anode_thickness'] * np.random.uniform(0.9, 1.1)
        inputs['electrolyte_thickness'] = self.geom['electrolyte_thickness'] * np.random.uniform(0.9, 1.1)
        inputs['cathode_thickness'] = self.geom['cathode_thickness'] * np.random.uniform(0.9, 1.1)
        
        # Material property variations
        inputs['anode_porosity'] = self.mat_props['anode']['porosity'] * np.random.uniform(0.9, 1.1)
        inputs['cathode_porosity'] = self.mat_props['cathode']['porosity'] * np.random.uniform(0.9, 1.1)
        
        # Degradation state
        inputs['operating_time'] = np.random.exponential(5000)  # hours
        inputs['thermal_cycles'] = np.random.poisson(10)
        inputs['redox_cycles'] = np.random.poisson(2)
        
        return inputs
    
    def _compute_outputs(self, inputs: Dict) -> Dict:
        """Compute low-fidelity outputs using lumped models."""
        outputs = {}
        
        T = inputs['temperature']
        i = inputs['current_density']
        p = inputs['pressure']
        
        # Partial pressures
        pH2 = p * inputs['fuel_composition'][0]
        pH2O = p * inputs['fuel_composition'][1]
        pO2 = p * inputs['air_composition'][0]
        
        # Nernst voltage
        E_nernst = self.physics.nernst_voltage(T, pH2, pH2O, pO2)
        
        # Overpotentials (simplified models)
        # Activation overpotential
        i0_anode = 1000 * np.exp(-60000 / (self.physics.R * T))  # A/m²
        i0_cathode = 100 * np.exp(-120000 / (self.physics.R * T))  # A/m²
        
        if i > 0:
            eta_act_anode = (self.physics.R * T) / (2 * self.physics.F) * np.log(i / i0_anode)
            eta_act_cathode = (self.physics.R * T) / (2 * self.physics.F) * np.log(i / i0_cathode)
        else:
            eta_act_anode = 0
            eta_act_cathode = 0
        
        eta_act = eta_act_anode + eta_act_cathode
        
        # Ohmic overpotential
        thickness_total = (inputs['anode_thickness'] + 
                          inputs['electrolyte_thickness'] + 
                          inputs['cathode_thickness']) * 1e-6  # Convert to m
        
        sigma_eff = self.materials.ionic_conductivity_YSZ(T)
        R_ohm = thickness_total / sigma_eff
        eta_ohm = i * R_ohm
        
        # Concentration overpotential
        D_eff = 1e-5 * (T / 1073) ** 1.5  # Effective diffusivity
        L_diff = inputs['anode_thickness'] * 1e-6
        i_lim = 2 * self.physics.F * D_eff * (pH2 / (self.physics.R * T)) / L_diff
        
        if i < i_lim * 0.95:
            eta_conc = self.physics.concentration_overpotential(i, i_lim, T)
        else:
            eta_conc = 0.2  # Cap concentration overpotential
        
        # Cell voltage
        V = E_nernst - eta_act - eta_ohm - abs(eta_conc)
        outputs['voltage'] = max(0, V)
        
        # Power density
        outputs['power_density'] = outputs['voltage'] * i
        
        # Average temperature (with heat generation)
        E_tn = 1.25  # Thermoneutral voltage
        q_gen = i * (E_tn - outputs['voltage'])
        delta_T = q_gen / (1000 * 50)  # Simplified heat transfer
        outputs['avg_temperature'] = T + delta_T
        
        # Average current density (same as input for lumped model)
        outputs['avg_current_density'] = i
        
        # Average stress (simplified thermal stress)
        T_ref = 298  # Reference temperature
        CTE_mismatch = abs(self.mat_props['anode']['CTE'] - 
                          self.mat_props['electrolyte']['CTE'])
        E_eff = 150e9  # Effective modulus
        nu = 0.3
        
        outputs['avg_stress'] = self.physics.thermal_stress(T, T_ref, CTE_mismatch, E_eff, nu)
        
        # Add creep contribution
        if outputs['avg_stress'] > 0:
            creep_params = self.config['degradation']['creep']
            creep_rate = self.physics.creep_strain_rate(outputs['avg_stress'], T, creep_params)
            creep_strain = creep_rate * inputs['operating_time'] * 3600  # Convert to seconds
            outputs['avg_stress'] *= (1 - creep_strain)  # Stress relaxation
        
        # Degradation rate
        deg_params = {
            'A': 0.1,
            'E_act': 50000,
            'current_exp': 0.5,
            'time_exp': 0.25
        }
        
        outputs['degradation_rate'] = self.degradation.voltage_degradation_rate(
            inputs['operating_time'], T, i, deg_params
        )
        
        # Add effects of thermal cycling
        if inputs['thermal_cycles'] > 0:
            cycle_damage = 0.001 * inputs['thermal_cycles']
            outputs['degradation_rate'] += cycle_damage
        
        # Add effects of redox cycling
        if inputs['redox_cycles'] > 0:
            redox_damage = self.degradation.anode_reoxidation_damage(inputs['redox_cycles'])
            outputs['degradation_rate'] += redox_damage * 10  # Scale to %/1000h
        
        return outputs
    
    def generate_dataset(self, n_samples: int, output_path: str = None) -> None:
        """Generate complete low-fidelity dataset."""
        print(f"\nGenerating {n_samples} low-fidelity samples...")
        
        # Initialize data manager
        if output_path is None:
            output_path = "./data"
        
        manager = DatasetManager(output_path)
        
        # Create HDF5 file structure
        spatial_dims = (1,)  # 1D for low fidelity
        f = manager.create_hdf5_structure('low_fidelity', n_samples, 
                                         spatial_dims, self.config)
        
        try:
            # Generate samples with progress bar
            for idx in tqdm(range(n_samples), desc="Low-Fidelity Generation"):
                sample = self.generate_sample(idx)
                manager.write_sample(f, idx, sample)
                
                # Flush periodically
                if idx % 100 == 0:
                    f.flush()
            
            print(f"Successfully generated {n_samples} low-fidelity samples")
            
        finally:
            f.close()
        
        # Validate dataset
        validation_report = manager.validate_dataset('low_fidelity')
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
    generator = LowFidelityGenerator(config)
    
    # Generate dataset
    n_samples = config['dataset']['low_fidelity']['n_samples']
    generator.generate_dataset(min(n_samples, 1000))  # Generate subset for testing


if __name__ == "__main__":
    main()