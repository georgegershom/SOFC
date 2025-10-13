"""
Dataset 1 Generator: High-Fidelity Physics-Based Simulation Data
Generates comprehensive multi-physics SOFC simulation data using parameter sweeps
"""

import numpy as np
import h5py
import yaml
import os
from typing import Dict, List, Tuple
from itertools import product
import time
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from physics_models.sofc_physics import SOFCPhysicsModel

class Dataset1Generator:
    """
    Generates Dataset 1: High-Fidelity Physics-Based Simulation Data
    
    This class performs parameter sweeps over operating conditions, material properties,
    and degradation states to generate comprehensive training data for the physics-informed
    machine learning model.
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration file"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset1_config = self.config['dataset1']
        self.simulation_config = self.config['simulation']
        self.output_config = self.config['output']
        
        # Create output directory
        self.output_dir = "datasets/dataset1_physics_simulation"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_parameter_combinations(self) -> List[Dict]:
        """
        Generate all parameter combinations using Latin Hypercube Sampling or full factorial
        """
        print("Generating parameter combinations...")
        
        # Operating conditions
        op_conditions = self.dataset1_config['operating_conditions']
        material_params = self.dataset1_config['material_parameters']
        degradation_params = self.dataset1_config['degradation_parameters']
        
        # Generate parameter ranges
        param_ranges = {}
        
        # Operating conditions
        for param, config in op_conditions.items():
            if param == 'fuel_composition':
                for sub_param, sub_config in config.items():
                    param_ranges[f"fuel_composition_{sub_param}"] = np.linspace(
                        sub_config['min'], sub_config['max'], sub_config['samples']
                    )
            else:
                param_ranges[param] = np.linspace(
                    config['min'], config['max'], config['samples']
                )
        
        # Material parameters
        for param, config in material_params.items():
            param_ranges[param] = np.linspace(
                config['min'], config['max'], config['samples']
            )
            
        # Degradation parameters
        for param, config in degradation_params.items():
            param_ranges[param] = np.linspace(
                config['min'], config['max'], config['samples']
            )
        
        # Use Latin Hypercube Sampling for efficient parameter space coverage
        n_samples = 1000  # Reduced for demonstration, increase for full dataset
        param_combinations = self._latin_hypercube_sampling(param_ranges, n_samples)
        
        print(f"Generated {len(param_combinations)} parameter combinations")
        return param_combinations
    
    def _latin_hypercube_sampling(self, param_ranges: Dict, n_samples: int) -> List[Dict]:
        """Generate Latin Hypercube samples for efficient parameter space coverage"""
        param_names = list(param_ranges.keys())
        n_params = len(param_names)
        
        # Generate LHS samples
        samples = np.random.rand(n_samples, n_params)
        
        # Apply LHS permutation
        for i in range(n_params):
            samples[:, i] = (np.random.permutation(n_samples) + samples[:, i]) / n_samples
        
        # Scale to parameter ranges
        param_combinations = []
        for sample in samples:
            combination = {}
            for i, param_name in enumerate(param_names):
                param_range = param_ranges[param_name]
                value = param_range[int(sample[i] * len(param_range))]
                combination[param_name] = value
            param_combinations.append(combination)
            
        return param_combinations
    
    def _convert_to_operating_conditions(self, params: Dict) -> Tuple[Dict, Dict]:
        """Convert flat parameter dictionary to structured format"""
        operating_conditions = {
            'current_density': params['current_density'],
            'fuel_utilization': params['fuel_utilization'],
            'air_utilization': params['air_utilization'],
            'inlet_fuel_temperature': params['inlet_fuel_temperature'],
            'inlet_air_temperature': params['inlet_air_temperature'],
            'fuel_composition': {
                'h2_percentage': params['fuel_composition_h2_percentage'],
                'h2o_percentage': params['fuel_composition_h2o_percentage'],
                'co_percentage': params['fuel_composition_co_percentage'],
                'ch4_percentage': params['fuel_composition_ch4_percentage']
            }
        }
        
        material_properties = {
            'electrode_porosity': params['electrode_porosity'],
            'electrode_tortuosity': params['electrode_tortuosity'],
            'anode_conductivity': params['anode_conductivity'],
            'cathode_conductivity': params['cathode_conductivity'],
            'electrolyte_thickness': params['electrolyte_thickness'],
            'electrode_thickness': params['electrode_thickness'],
            'initial_crack_length': params['initial_crack_length'],
            'porosity_degradation': params['porosity_degradation']
        }
        
        return operating_conditions, material_properties
    
    def run_single_simulation(self, params: Dict, sim_id: int) -> Dict:
        """Run a single multi-physics simulation"""
        
        # Convert parameters
        operating_conditions, material_properties = self._convert_to_operating_conditions(params)
        
        # Create geometry configuration
        geometry = {
            'nx': self.simulation_config['mesh']['nx'],
            'ny': self.simulation_config['mesh']['ny'], 
            'nz': self.simulation_config['mesh']['nz'],
            'length': self.simulation_config['geometry']['length'],
            'width': self.simulation_config['geometry']['width'],
            'height': self.simulation_config['geometry']['height']
        }
        
        # Initialize physics model
        physics_model = SOFCPhysicsModel(geometry, material_properties)
        
        # Solve coupled physics
        results = physics_model.solve_coupled_physics(operating_conditions)
        
        # Add metadata
        results['simulation_id'] = sim_id
        results['parameters'] = params
        results['operating_conditions'] = operating_conditions
        results['material_properties'] = material_properties
        results['geometry'] = geometry
        
        return results
    
    def save_simulation_data(self, results: Dict, sim_id: int):
        """Save simulation results to HDF5 format"""
        filename = os.path.join(self.output_dir, f"simulation_{sim_id:06d}.h5")
        
        with h5py.File(filename, 'w') as f:
            # Create groups
            fields_group = f.create_group('fields')
            params_group = f.create_group('parameters')
            metadata_group = f.create_group('metadata')
            
            # Save field data
            fields_to_save = self.output_config['fields_to_save']
            for field_name in fields_to_save:
                if field_name in results:
                    fields_group.create_dataset(
                        field_name, 
                        data=results[field_name],
                        compression='gzip' if self.output_config['compression'] else None
                    )
            
            # Save parameters
            for key, value in results['parameters'].items():
                params_group.attrs[key] = value
                
            # Save operating conditions
            op_group = params_group.create_group('operating_conditions')
            for key, value in results['operating_conditions'].items():
                if isinstance(value, dict):
                    sub_group = op_group.create_group(key)
                    for sub_key, sub_value in value.items():
                        sub_group.attrs[sub_key] = sub_value
                else:
                    op_group.attrs[key] = value
            
            # Save material properties
            mat_group = params_group.create_group('material_properties')
            for key, value in results['material_properties'].items():
                mat_group.attrs[key] = value
            
            # Save metadata
            metadata_group.attrs['simulation_id'] = results['simulation_id']
            metadata_group.attrs['timestamp'] = time.time()
            
            # Save derived quantities
            derived_group = f.create_group('derived_quantities')
            derived_group.attrs['cell_voltage'] = results['cell_voltage']
            derived_group.attrs['max_temperature'] = results['max_temperature']
            derived_group.attrs['max_von_mises_stress'] = results['max_von_mises_stress']
            
            # Save geometry
            geom_group = metadata_group.create_group('geometry')
            for key, value in results['geometry'].items():
                geom_group.attrs[key] = value
    
    def generate_dataset(self):
        """Generate the complete Dataset 1"""
        print("Starting Dataset 1 generation...")
        print("="*50)
        
        # Generate parameter combinations
        param_combinations = self.generate_parameter_combinations()
        
        # Create summary file
        summary_data = {
            'simulations': [],
            'statistics': {},
            'failed_simulations': []
        }
        
        # Run simulations
        print(f"Running {len(param_combinations)} simulations...")
        successful_sims = 0
        failed_sims = 0
        
        for sim_id, params in enumerate(tqdm(param_combinations, desc="Simulations")):
            try:
                # Run simulation
                results = self.run_single_simulation(params, sim_id)
                
                # Save results
                self.save_simulation_data(results, sim_id)
                
                # Update summary
                summary_data['simulations'].append({
                    'simulation_id': sim_id,
                    'cell_voltage': float(results['cell_voltage']),
                    'max_temperature': float(results['max_temperature']),
                    'max_von_mises_stress': float(results['max_von_mises_stress']),
                    'parameters': params
                })
                
                successful_sims += 1
                
            except Exception as e:
                print(f"Simulation {sim_id} failed: {str(e)}")
                summary_data['failed_simulations'].append({
                    'simulation_id': sim_id,
                    'error': str(e),
                    'parameters': params
                })
                failed_sims += 1
                continue
        
        # Save summary
        summary_data['statistics'] = {
            'total_simulations': len(param_combinations),
            'successful_simulations': successful_sims,
            'failed_simulations': failed_sims,
            'success_rate': successful_sims / len(param_combinations)
        }
        
        summary_file = os.path.join(self.output_dir, 'dataset_summary.yaml')
        with open(summary_file, 'w') as f:
            yaml.dump(summary_data, f, default_flow_style=False)
        
        print(f"\nDataset 1 generation completed!")
        print(f"Successful simulations: {successful_sims}")
        print(f"Failed simulations: {failed_sims}")
        print(f"Success rate: {successful_sims/len(param_combinations)*100:.1f}%")
        print(f"Data saved to: {self.output_dir}")
        
        return summary_data

def main():
    """Main function to generate Dataset 1"""
    config_path = "config/simulation_config.yaml"
    
    # Create generator
    generator = Dataset1Generator(config_path)
    
    # Generate dataset
    summary = generator.generate_dataset()
    
    print("\nDataset 1 generation summary:")
    print(f"Total simulations: {summary['statistics']['total_simulations']}")
    print(f"Successful: {summary['statistics']['successful_simulations']}")
    print(f"Failed: {summary['statistics']['failed_simulations']}")

if __name__ == "__main__":
    main()