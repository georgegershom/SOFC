"""
Comprehensive Dataset Generator for Residual Stress Prediction in Multi-Layer Ceramics
=========================================================================

This module generates a comprehensive dataset of process parameters, material properties,
and geometric variations for training ML models to predict residual stress in multi-layer
ceramic structures (e.g., SOFC cells).

The dataset includes:
1. Geometric parameters (dimensions, thicknesses, densities)
2. Material properties (temperature-dependent)
3. Process parameters (sintering profiles, atmospheres)
4. Simulated residual stress responses

Author: AI Assistant
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate
from scipy.stats import norm, uniform, lognorm
import h5py
import json
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ResidualStressDatasetGenerator:
    """
    Main class for generating comprehensive residual stress datasets.
    """
    
    def __init__(self, random_seed=42):
        """Initialize the dataset generator."""
        np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Initialize parameter ranges and distributions
        self._initialize_parameter_ranges()
        
    def _initialize_parameter_ranges(self):
        """Define realistic parameter ranges for ceramic materials."""
        
        # Geometric Parameters
        self.geometric_params = {
            'plate_length': {'min': 50e-3, 'max': 200e-3, 'units': 'm'},  # 50-200mm
            'plate_width': {'min': 50e-3, 'max': 200e-3, 'units': 'm'},   # 50-200mm
            'anode_thickness': {'min': 300e-6, 'max': 1500e-6, 'units': 'm'},  # 300-1500μm
            'electrolyte_thickness': {'min': 5e-6, 'max': 50e-6, 'units': 'm'},  # 5-50μm
            'cathode_thickness': {'min': 20e-6, 'max': 100e-6, 'units': 'm'},   # 20-100μm
            'green_density_anode': {'min': 0.45, 'max': 0.65, 'units': 'fraction'},
            'green_density_electrolyte': {'min': 0.50, 'max': 0.70, 'units': 'fraction'},
            'green_density_cathode': {'min': 0.40, 'max': 0.60, 'units': 'fraction'},
        }
        
        # Material Properties (at room temperature, with temperature dependence)
        self.material_params = {
            'anode': {
                'youngs_modulus_rt': {'min': 50e9, 'max': 150e9, 'units': 'Pa'},  # Ni-YSZ
                'cte': {'min': 11e-6, 'max': 14e-6, 'units': '1/K'},
                'poisson_ratio': {'min': 0.25, 'max': 0.35, 'units': 'dimensionless'},
                'sintering_shrinkage': {'min': 0.15, 'max': 0.25, 'units': 'fraction'},
                'shrinkage_onset_temp': {'min': 1100, 'max': 1300, 'units': 'K'},
                'creep_activation_energy': {'min': 300e3, 'max': 500e3, 'units': 'J/mol'},
            },
            'electrolyte': {
                'youngs_modulus_rt': {'min': 180e9, 'max': 220e9, 'units': 'Pa'},  # YSZ
                'cte': {'min': 10e-6, 'max': 11e-6, 'units': '1/K'},
                'poisson_ratio': {'min': 0.28, 'max': 0.32, 'units': 'dimensionless'},
                'sintering_shrinkage': {'min': 0.18, 'max': 0.28, 'units': 'fraction'},
                'shrinkage_onset_temp': {'min': 1200, 'max': 1400, 'units': 'K'},
                'creep_activation_energy': {'min': 400e3, 'max': 600e3, 'units': 'J/mol'},
            },
            'cathode': {
                'youngs_modulus_rt': {'min': 80e9, 'max': 120e9, 'units': 'Pa'},  # LSM-YSZ
                'cte': {'min': 11.5e-6, 'max': 13e-6, 'units': '1/K'},
                'poisson_ratio': {'min': 0.26, 'max': 0.34, 'units': 'dimensionless'},
                'sintering_shrinkage': {'min': 0.12, 'max': 0.22, 'units': 'fraction'},
                'shrinkage_onset_temp': {'min': 1000, 'max': 1200, 'units': 'K'},
                'creep_activation_energy': {'min': 250e3, 'max': 450e3, 'units': 'J/mol'},
            }
        }
        
        # Process Parameters
        self.process_params = {
            'max_sintering_temp': {'min': 1573, 'max': 1773, 'units': 'K'},  # 1300-1500°C
            'heating_rate': {'min': 1, 'max': 10, 'units': 'K/min'},
            'cooling_rate': {'min': 1, 'max': 5, 'units': 'K/min'},
            'hold_time': {'min': 1, 'max': 8, 'units': 'hours'},
            'atmosphere_oxygen_partial_pressure': {'min': 1e-20, 'max': 0.21, 'units': 'atm'},
        }
        
    def generate_geometric_parameters(self, n_samples):
        """Generate geometric parameter variations."""
        
        geometric_data = {}
        
        for param, ranges in self.geometric_params.items():
            if 'density' in param:
                # Use beta distribution for densities (bounded between 0 and 1)
                alpha, beta = 2, 2  # Symmetric beta distribution
                samples = np.random.beta(alpha, beta, n_samples)
                samples = ranges['min'] + samples * (ranges['max'] - ranges['min'])
            else:
                # Use log-normal distribution for dimensions (realistic manufacturing variation)
                mean_val = (ranges['min'] + ranges['max']) / 2
                std_val = (ranges['max'] - ranges['min']) / 6  # 99.7% within range
                samples = np.random.lognormal(
                    np.log(mean_val), 
                    std_val / mean_val, 
                    n_samples
                )
                samples = np.clip(samples, ranges['min'], ranges['max'])
            
            geometric_data[param] = samples
            
        return pd.DataFrame(geometric_data)
    
    def generate_material_properties(self, n_samples):
        """Generate material property variations with temperature dependence."""
        
        material_data = {}
        
        for layer in ['anode', 'electrolyte', 'cathode']:
            layer_params = self.material_params[layer]
            
            for param, ranges in layer_params.items():
                param_name = f"{layer}_{param}"
                
                if 'youngs_modulus' in param:
                    # Young's modulus typically follows log-normal distribution
                    mean_val = (ranges['min'] + ranges['max']) / 2
                    std_val = (ranges['max'] - ranges['min']) / 6
                    samples = np.random.lognormal(
                        np.log(mean_val), 
                        std_val / mean_val, 
                        n_samples
                    )
                elif 'cte' in param:
                    # CTE follows normal distribution
                    mean_val = (ranges['min'] + ranges['max']) / 2
                    std_val = (ranges['max'] - ranges['min']) / 6
                    samples = np.random.normal(mean_val, std_val, n_samples)
                elif 'poisson' in param:
                    # Poisson's ratio bounded between 0 and 0.5
                    samples = np.random.uniform(ranges['min'], ranges['max'], n_samples)
                elif 'activation_energy' in param:
                    # Activation energy log-normal
                    mean_val = (ranges['min'] + ranges['max']) / 2
                    std_val = (ranges['max'] - ranges['min']) / 6
                    samples = np.random.lognormal(
                        np.log(mean_val), 
                        std_val / mean_val, 
                        n_samples
                    )
                else:
                    # Default uniform distribution
                    samples = np.random.uniform(ranges['min'], ranges['max'], n_samples)
                
                samples = np.clip(samples, ranges['min'], ranges['max'])
                material_data[param_name] = samples
        
        return pd.DataFrame(material_data)
    
    def generate_process_parameters(self, n_samples):
        """Generate process parameter variations."""
        
        process_data = {}
        
        for param, ranges in self.process_params.items():
            if 'temp' in param:
                # Temperature follows normal distribution
                mean_val = (ranges['min'] + ranges['max']) / 2
                std_val = (ranges['max'] - ranges['min']) / 6
                samples = np.random.normal(mean_val, std_val, n_samples)
            elif 'rate' in param:
                # Rates follow log-normal distribution
                mean_val = (ranges['min'] + ranges['max']) / 2
                std_val = (ranges['max'] - ranges['min']) / 6
                samples = np.random.lognormal(
                    np.log(mean_val), 
                    std_val / mean_val, 
                    n_samples
                )
            elif 'pressure' in param:
                # Oxygen partial pressure log-uniform
                samples = np.random.uniform(
                    np.log10(ranges['min']), 
                    np.log10(ranges['max']), 
                    n_samples
                )
                samples = 10 ** samples
            else:
                # Default uniform distribution
                samples = np.random.uniform(ranges['min'], ranges['max'], n_samples)
            
            samples = np.clip(samples, ranges['min'], ranges['max'])
            process_data[param] = samples
        
        return pd.DataFrame(process_data)
    
    def generate_temperature_profiles(self, n_samples):
        """Generate realistic sintering temperature profiles."""
        
        profiles = []
        
        for i in range(n_samples):
            # Base parameters
            max_temp = np.random.uniform(1573, 1773)  # K
            heating_rate = np.random.uniform(1, 10)    # K/min
            cooling_rate = np.random.uniform(1, 5)     # K/min
            hold_time = np.random.uniform(1, 8)        # hours
            
            # Create temperature profile
            room_temp = 298  # K
            
            # Heating phase
            heating_time = (max_temp - room_temp) / heating_rate  # minutes
            
            # Total time points
            total_time = heating_time + hold_time * 60 + (max_temp - room_temp) / cooling_rate
            
            # Time array (in minutes)
            time_points = np.linspace(0, total_time, 100)
            
            # Temperature array
            temp_profile = np.zeros_like(time_points)
            
            for j, t in enumerate(time_points):
                if t <= heating_time:
                    # Heating phase
                    temp_profile[j] = room_temp + heating_rate * t
                elif t <= heating_time + hold_time * 60:
                    # Hold phase
                    temp_profile[j] = max_temp
                else:
                    # Cooling phase
                    cooling_time = t - heating_time - hold_time * 60
                    temp_profile[j] = max_temp - cooling_rate * cooling_time
            
            profiles.append({
                'sample_id': i,
                'time_minutes': time_points.tolist(),
                'temperature_K': temp_profile.tolist(),
                'max_temp': max_temp,
                'heating_rate': heating_rate,
                'cooling_rate': cooling_rate,
                'hold_time': hold_time
            })
        
        return profiles
    
    def calculate_temperature_dependent_properties(self, base_properties, temperature_profiles):
        """Calculate temperature-dependent material properties."""
        
        temp_dependent_props = []
        
        for i, profile in enumerate(temperature_profiles):
            temps = np.array(profile['temperature_K'])
            
            # Get base properties for this sample
            sample_props = {}
            for col in base_properties.columns:
                if i < len(base_properties):
                    sample_props[col] = base_properties.iloc[i][col]
            
            # Calculate temperature dependence
            temp_props = {}
            
            for layer in ['anode', 'electrolyte', 'cathode']:
                # Young's modulus temperature dependence (decreases with temperature)
                E_rt = sample_props.get(f'{layer}_youngs_modulus_rt', 100e9)
                E_temp = E_rt * (1 - 0.0003 * (temps - 298))  # Typical ceramic behavior
                temp_props[f'{layer}_youngs_modulus_temp'] = E_temp.tolist()
                
                # CTE (slight temperature dependence)
                cte_rt = sample_props.get(f'{layer}_cte', 12e-6)
                cte_temp = cte_rt * (1 + 0.0001 * (temps - 298))
                temp_props[f'{layer}_cte_temp'] = cte_temp.tolist()
                
                # Poisson's ratio (slight increase with temperature)
                nu_rt = sample_props.get(f'{layer}_poisson_ratio', 0.3)
                nu_temp = nu_rt * (1 + 0.0001 * (temps - 298))
                temp_props[f'{layer}_poisson_ratio_temp'] = nu_temp.tolist()
            
            temp_props['sample_id'] = i
            temp_props['temperature_K'] = temps.tolist()
            temp_dependent_props.append(temp_props)
        
        return temp_dependent_props
    
    def simulate_residual_stress(self, geometric_params, material_params, process_params, temperature_profiles):
        """
        Simulate residual stress using simplified analytical models.
        This is a placeholder for actual FEA simulation.
        """
        
        n_samples = len(geometric_params)
        stress_results = []
        
        print("Simulating residual stress for each sample...")
        
        for i in tqdm(range(n_samples)):
            # Get parameters for this sample
            geom = geometric_params.iloc[i]
            mat = material_params.iloc[i]
            proc = process_params.iloc[i]
            temp_profile = temperature_profiles[i]
            
            # Simplified stress calculation based on CTE mismatch and constrained sintering
            
            # Layer properties
            layers = ['anode', 'electrolyte', 'cathode']
            thicknesses = [geom['anode_thickness'], geom['electrolyte_thickness'], geom['cathode_thickness']]
            
            # CTE mismatch stress (dominant mechanism)
            delta_T = temp_profile['max_temp'] - 298  # Cooling from max temp
            
            cte_mismatch_stress = {}
            for j, layer in enumerate(layers):
                cte = mat[f'{layer}_cte']
                E = mat[f'{layer}_youngs_modulus_rt']
                nu = mat[f'{layer}_poisson_ratio']
                
                # Reference CTE (electrolyte)
                cte_ref = mat['electrolyte_cte']
                
                # Biaxial stress due to CTE mismatch
                stress_cte = E / (1 - nu) * (cte - cte_ref) * delta_T
                cte_mismatch_stress[f'{layer}_stress_cte'] = stress_cte
            
            # Sintering mismatch stress
            sintering_stress = {}
            for j, layer in enumerate(layers):
                shrinkage = mat[f'{layer}_sintering_shrinkage']
                E = mat[f'{layer}_youngs_modulus_rt']
                nu = mat[f'{layer}_poisson_ratio']
                
                # Reference shrinkage (electrolyte)
                shrinkage_ref = mat['electrolyte_sintering_shrinkage']
                
                # Stress due to differential shrinkage
                stress_shrinkage = E / (1 - nu) * (shrinkage - shrinkage_ref)
                sintering_stress[f'{layer}_stress_sintering'] = stress_shrinkage
            
            # Geometric effects (aspect ratio, thickness ratios)
            aspect_ratio = geom['plate_length'] / geom['plate_width']
            total_thickness = sum(thicknesses)
            
            # Thickness ratio effects
            thickness_ratios = [t / total_thickness for t in thicknesses]
            
            # Combined stress (simplified superposition)
            total_stress = {}
            for layer in layers:
                stress_total = (cte_mismatch_stress[f'{layer}_stress_cte'] + 
                               sintering_stress[f'{layer}_stress_sintering'])
                
                # Geometric amplification factor
                geom_factor = 1 + 0.1 * abs(aspect_ratio - 1)  # Non-square amplifies stress
                
                # Thickness effect
                layer_idx = layers.index(layer)
                thickness_factor = 1 + 0.2 * abs(thickness_ratios[layer_idx] - 1/3)
                
                stress_total *= geom_factor * thickness_factor
                total_stress[f'{layer}_residual_stress_total'] = stress_total
            
            # Add process-dependent stress relaxation
            max_temp = proc['max_sintering_temp']
            hold_time = proc['hold_time']
            
            # Stress relaxation factor (higher temp and longer time reduce stress)
            relaxation_factor = np.exp(-0.001 * (max_temp - 1573) * hold_time)
            
            for layer in layers:
                total_stress[f'{layer}_residual_stress_total'] *= relaxation_factor
            
            # Calculate von Mises equivalent stress
            for layer in layers:
                sigma = total_stress[f'{layer}_residual_stress_total']
                # Assuming biaxial stress state
                von_mises = abs(sigma) * np.sqrt(1 - 0.5 + 1)  # sqrt(sigma_x^2 - sigma_x*sigma_y + sigma_y^2)
                total_stress[f'{layer}_von_mises_stress'] = von_mises
            
            # Add sample ID
            total_stress['sample_id'] = i
            
            # Add some realistic noise
            for key in total_stress:
                if key != 'sample_id':
                    noise_factor = 1 + np.random.normal(0, 0.05)  # 5% noise
                    total_stress[key] *= noise_factor
            
            stress_results.append(total_stress)
        
        return pd.DataFrame(stress_results)
    
    def generate_comprehensive_dataset(self, n_samples=1000):
        """Generate the complete dataset."""
        
        print(f"Generating comprehensive residual stress dataset with {n_samples} samples...")
        
        # Generate all parameter variations
        print("1. Generating geometric parameters...")
        geometric_data = self.generate_geometric_parameters(n_samples)
        
        print("2. Generating material properties...")
        material_data = self.generate_material_properties(n_samples)
        
        print("3. Generating process parameters...")
        process_data = self.generate_process_parameters(n_samples)
        
        print("4. Generating temperature profiles...")
        temperature_profiles = self.generate_temperature_profiles(n_samples)
        
        print("5. Calculating temperature-dependent properties...")
        temp_dependent_props = self.calculate_temperature_dependent_properties(
            material_data, temperature_profiles
        )
        
        print("6. Simulating residual stress...")
        stress_results = self.simulate_residual_stress(
            geometric_data, material_data, process_data, temperature_profiles
        )
        
        # Combine all data
        print("7. Combining datasets...")
        combined_data = pd.concat([
            geometric_data.reset_index(drop=True),
            material_data.reset_index(drop=True),
            process_data.reset_index(drop=True),
            stress_results.reset_index(drop=True)
        ], axis=1)
        
        # Add sample IDs
        combined_data['sample_id'] = range(n_samples)
        
        # Store additional data
        dataset = {
            'main_dataset': combined_data,
            'temperature_profiles': temperature_profiles,
            'temperature_dependent_properties': temp_dependent_props,
            'parameter_ranges': {
                'geometric': self.geometric_params,
                'material': self.material_params,
                'process': self.process_params
            },
            'metadata': {
                'n_samples': n_samples,
                'random_seed': self.random_seed,
                'generation_date': '2025-10-15',
                'description': 'Comprehensive dataset for residual stress prediction in multi-layer ceramics'
            }
        }
        
        print(f"Dataset generation complete! Generated {n_samples} samples with {len(combined_data.columns)} features.")
        
        return dataset
    
    def save_dataset(self, dataset, base_filename='residual_stress_dataset'):
        """Save dataset in multiple formats."""
        
        print("Saving dataset in multiple formats...")
        
        # Save main dataset as CSV
        csv_filename = f"{base_filename}.csv"
        dataset['main_dataset'].to_csv(csv_filename, index=False)
        print(f"Saved CSV: {csv_filename}")
        
        # Save as Excel with multiple sheets
        excel_filename = f"{base_filename}.xlsx"
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            dataset['main_dataset'].to_excel(writer, sheet_name='Main_Dataset', index=False)
            
            # Parameter ranges summary
            param_summary = []
            for category, params in dataset['parameter_ranges'].items():
                if category == 'material':
                    for layer, layer_params in params.items():
                        for param, ranges in layer_params.items():
                            param_summary.append({
                                'Category': f'{category}_{layer}',
                                'Parameter': param,
                                'Min': ranges['min'],
                                'Max': ranges['max'],
                                'Units': ranges['units']
                            })
                else:
                    for param, ranges in params.items():
                        param_summary.append({
                            'Category': category,
                            'Parameter': param,
                            'Min': ranges['min'],
                            'Max': ranges['max'],
                            'Units': ranges['units']
                        })
            
            pd.DataFrame(param_summary).to_excel(writer, sheet_name='Parameter_Ranges', index=False)
        
        print(f"Saved Excel: {excel_filename}")
        
        # Save as HDF5 for efficient storage
        hdf5_filename = f"{base_filename}.h5"
        with h5py.File(hdf5_filename, 'w') as f:
            # Main dataset
            main_grp = f.create_group('main_dataset')
            for col in dataset['main_dataset'].columns:
                main_grp.create_dataset(col, data=dataset['main_dataset'][col].values)
            
            # Temperature profiles
            temp_grp = f.create_group('temperature_profiles')
            for i, profile in enumerate(dataset['temperature_profiles']):
                sample_grp = temp_grp.create_group(f'sample_{i}')
                for key, value in profile.items():
                    if isinstance(value, list):
                        sample_grp.create_dataset(key, data=np.array(value))
                    else:
                        sample_grp.attrs[key] = value
            
            # Metadata
            meta_grp = f.create_group('metadata')
            for key, value in dataset['metadata'].items():
                meta_grp.attrs[key] = str(value)
        
        print(f"Saved HDF5: {hdf5_filename}")
        
        # Save metadata and parameter ranges as JSON
        json_filename = f"{base_filename}_metadata.json"
        json_data = {
            'parameter_ranges': dataset['parameter_ranges'],
            'metadata': dataset['metadata']
        }
        
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"Saved JSON metadata: {json_filename}")
        
        return {
            'csv': csv_filename,
            'excel': excel_filename,
            'hdf5': hdf5_filename,
            'json': json_filename
        }


def main():
    """Main function to generate and save the dataset."""
    
    # Initialize generator
    generator = ResidualStressDatasetGenerator(random_seed=42)
    
    # Generate dataset with different sample sizes
    sample_sizes = [1000, 5000, 10000]
    
    for n_samples in sample_sizes:
        print(f"\n{'='*60}")
        print(f"Generating dataset with {n_samples} samples")
        print(f"{'='*60}")
        
        # Generate dataset
        dataset = generator.generate_comprehensive_dataset(n_samples)
        
        # Save dataset
        filenames = generator.save_dataset(dataset, f'residual_stress_dataset_{n_samples}')
        
        print(f"\nDataset with {n_samples} samples saved as:")
        for format_type, filename in filenames.items():
            print(f"  {format_type.upper()}: {filename}")


if __name__ == "__main__":
    main()