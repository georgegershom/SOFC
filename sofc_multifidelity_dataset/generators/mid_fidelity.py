"""
Mid-Fidelity SOFC Model and Dataset Generator
2D/3D FEM-like models with spatial resolution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import h5py
import json
from datetime import datetime
import sys
import os
from scipy import ndimage, interpolate

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sofc_physics import (
    SOFCOperatingConditions, ElectrochemicalModel,
    ThermalModel, MechanicalModel, DegradationModel
)


class MidFidelitySOFCModel:
    """Mid-fidelity 2D/3D SOFC model with spatial resolution"""
    
    def __init__(self, config: dict):
        self.config = config
        self.electrochem = ElectrochemicalModel(config)
        self.thermal = ThermalModel(config)
        self.mechanical = MechanicalModel(config)
        self.degradation = DegradationModel(config)
        self.mf_config = config['dataset']['mid_fidelity']
        self.mesh_res = self.mf_config['mesh_resolution']
        
    def simulate_electrochemistry_2D(self, conditions: SOFCOperatingConditions) -> Dict[str, np.ndarray]:
        """2D electrochemical simulation"""
        nx, ny = self.mesh_res
        
        # Initialize fields
        current_density = np.zeros((nx, ny))
        overpotentials = {
            'activation_anode': np.zeros((nx, ny)),
            'activation_cathode': np.zeros((nx, ny)),
            'ohmic': np.zeros((nx, ny)),
            'concentration': np.zeros((nx, ny))
        }
        
        # Create spatial variations
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Current density distribution (higher near inlet and ribs)
        base_current = conditions.current_density
        
        # Rib pattern (assuming co-flow)
        rib_pattern = np.zeros((nx, ny))
        rib_width = 5  # grid points
        channel_width = 5
        
        for i in range(0, nx, rib_width + channel_width):
            rib_pattern[i:i+rib_width, :] = 1.0
            
        # Current enhancement under ribs
        current_enhancement = 1 + 0.3 * rib_pattern
        
        # Flow direction effects (decreasing along flow)
        flow_factor = 1 - 0.2 * Y
        
        # Combined current distribution
        current_density = base_current * current_enhancement * flow_factor
        
        # Add noise and smoothing
        noise = np.random.normal(0, 0.05, (nx, ny))
        current_density *= (1 + noise)
        current_density = ndimage.gaussian_filter(current_density, sigma=1)
        
        # Calculate local overpotentials
        T = conditions.temperature
        
        for i in range(nx):
            for j in range(ny):
                local_i = current_density[i, j]
                
                # Local temperature variation
                local_T = T + 20 * (1 - flow_factor[i, j])
                
                # Calculate overpotentials
                overpotentials['activation_anode'][i, j] = \
                    self.electrochem.activation_overpotential(local_i, local_T, 'anode')
                overpotentials['activation_cathode'][i, j] = \
                    self.electrochem.activation_overpotential(local_i, local_T, 'cathode')
                overpotentials['ohmic'][i, j] = \
                    self.electrochem.ohmic_overpotential(local_i, local_T)
                
                # Concentration overpotential increases along flow
                i_lim = self.electrochem.limiting_current_density(
                    local_T, conditions.pressure, 0.97 * flow_factor[i, j]
                )
                if local_i < i_lim:
                    overpotentials['concentration'][i, j] = \
                        self.electrochem.concentration_overpotential(local_i, local_T, i_lim)
                else:
                    overpotentials['concentration'][i, j] = 0.1  # High but not infinite
        
        return {
            'current_density': current_density,
            'overpotentials': overpotentials
        }
    
    def simulate_species_2D(self, conditions: SOFCOperatingConditions,
                           current_density: np.ndarray) -> Dict[str, np.ndarray]:
        """2D species concentration simulation"""
        nx, ny = self.mesh_res
        
        # Initialize species fields
        species = {
            'H2': np.zeros((nx, ny)),
            'H2O': np.zeros((nx, ny)),
            'O2': np.zeros((nx, ny))
        }
        
        # Inlet compositions
        y_H2_in = conditions.inlet_fuel_composition.get('H2', 0.97)
        y_H2O_in = conditions.inlet_fuel_composition.get('H2O', 0.03)
        y_O2_in = conditions.inlet_air_composition.get('O2', 0.21)
        
        # Flow direction (y-axis)
        for j in range(ny):
            # Consumption along flow based on current
            local_consumption = np.cumsum(current_density[:, :j+1], axis=1) / (2 * 96485)
            
            # H2 depletion
            depletion_factor = j / ny
            species['H2'][:, j] = y_H2_in * (1 - conditions.fuel_utilization * depletion_factor)
            
            # H2O production
            species['H2O'][:, j] = y_H2O_in + y_H2_in * conditions.fuel_utilization * depletion_factor
            
            # O2 depletion
            species['O2'][:, j] = y_O2_in * (1 - conditions.air_utilization * depletion_factor)
        
        # Add diffusion effects
        for sp in species:
            species[sp] = ndimage.gaussian_filter(species[sp], sigma=2)
            
        return species
    
    def simulate(self, conditions: SOFCOperatingConditions, cycles: int = 0) -> Dict:
        """Run mid-fidelity simulation"""
        
        # 2D Electrochemical simulation
        electro_2d = self.simulate_electrochemistry_2D(conditions)
        current_density_2d = electro_2d['current_density']
        overpotentials_2d = electro_2d['overpotentials']
        
        # Average voltage
        V_cell, overpotentials_avg = self.electrochem.cell_voltage(conditions)
        
        # 2D Thermal simulation
        T_field_2d = self.thermal.temperature_distribution_2D(
            conditions, V_cell, overpotentials_avg['E_nernst'], self.mesh_res
        )
        
        # 2D Species simulation
        species_2d = self.simulate_species_2D(conditions, current_density_2d)
        
        # 2D Mechanical simulation
        stress_field_2d = self.mechanical.thermal_stress(T_field_2d)
        von_mises_2d = self.mechanical.von_mises_stress(stress_field_2d)
        
        # Damage field
        damage_2d = self.mechanical.damage_accumulation(
            stress_field_2d, T_field_2d, cycles, conditions.time
        )
        
        # Calculate degradation metrics
        deg_metrics = self.degradation.total_degradation(
            conditions, overpotentials_avg, stress_field_2d, cycles
        )
        
        # Compile results
        results = {
            # Scalar outputs (same as LF)
            'temperature_inlet': conditions.temperature,
            'pressure': conditions.pressure,
            'current_density_avg': conditions.current_density,
            'fuel_utilization': conditions.fuel_utilization,
            'air_utilization': conditions.air_utilization,
            'time': conditions.time,
            'cycles': cycles,
            'voltage': V_cell,
            'power_density': V_cell * conditions.current_density,
            
            # 2D Fields (flattened for storage)
            'temperature_field_2d': T_field_2d.flatten(),
            'current_density_field_2d': current_density_2d.flatten(),
            'von_mises_stress_2d': von_mises_2d.flatten(),
            'damage_field_2d': damage_2d.flatten(),
            'H2_concentration_2d': species_2d['H2'].flatten(),
            'H2O_concentration_2d': species_2d['H2O'].flatten(),
            'O2_concentration_2d': species_2d['O2'].flatten(),
            
            # Field statistics
            'temperature_max': np.max(T_field_2d),
            'temperature_min': np.min(T_field_2d),
            'temperature_std': np.std(T_field_2d),
            'stress_max': np.max(von_mises_2d),
            'stress_mean': np.mean(von_mises_2d),
            'damage_max': np.max(damage_2d),
            
            # Degradation outputs
            'voltage_degradation_rate_mV_kh': deg_metrics['voltage_degradation_rate'],
            'estimated_lifetime_hours': deg_metrics['estimated_lifetime']
        }
        
        # Store field shapes for reconstruction
        results['field_shape'] = self.mesh_res
        
        return results


class MidFidelityDatasetGenerator:
    """Generate mid-fidelity SOFC dataset with spatial fields"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = MidFidelitySOFCModel(config)
        self.mf_config = config['dataset']['mid_fidelity']
        
    def select_conditions_from_lf(self, lf_dataset_path: str, n_samples: int) -> List[Tuple[SOFCOperatingConditions, int]]:
        """Select interesting conditions from LF dataset"""
        
        # Load LF dataset
        if os.path.exists(lf_dataset_path):
            df_lf = pd.read_csv(lf_dataset_path.replace('.h5', '.csv'))
        else:
            print("LF dataset not found, generating random conditions")
            return self.generate_random_conditions(n_samples)
        
        # Select interesting points (high stress, near failure, etc.)
        # Sort by interesting metrics
        df_lf['interest_score'] = (
            df_lf['stress_maximum'] / df_lf['stress_maximum'].max() +
            df_lf['temperature_gradient'] / df_lf['temperature_gradient'].max() +
            (1 - df_lf['voltage'] / df_lf['voltage'].max()) +
            df_lf['degradation_crack_density']
        )
        
        # Select top samples and some random ones
        n_interesting = int(0.7 * n_samples)
        n_random = n_samples - n_interesting
        
        df_interesting = df_lf.nlargest(n_interesting, 'interest_score')
        df_random = df_lf.sample(n=min(n_random, len(df_lf)))
        
        df_selected = pd.concat([df_interesting, df_random]).drop_duplicates()
        
        # Convert to operating conditions
        conditions_list = []
        
        for _, row in df_selected.iterrows():
            # Reconstruct fuel composition (simplified)
            fuel_comp = {'H2': 0.97, 'H2O': 0.03}
            air_comp = {'O2': 0.21, 'N2': 0.79}
            
            cond = SOFCOperatingConditions(
                temperature=row['temperature_inlet'],
                pressure=row['pressure'],
                current_density=row['current_density'],
                fuel_utilization=row['fuel_utilization'],
                air_utilization=row['air_utilization'],
                inlet_fuel_composition=fuel_comp,
                inlet_air_composition=air_comp,
                time=row['time']
            )
            
            conditions_list.append((cond, int(row['cycles'])))
            
        return conditions_list[:n_samples]
    
    def generate_random_conditions(self, n_samples: int) -> List[Tuple[SOFCOperatingConditions, int]]:
        """Generate random operating conditions if LF dataset not available"""
        from generators.low_fidelity import LowFidelityDatasetGenerator
        
        lf_gen = LowFidelityDatasetGenerator(self.config)
        conditions = lf_gen.sample_operating_conditions(n_samples)
        return lf_gen.add_cycling_conditions(conditions)
    
    def generate_dataset(self, n_samples: int = None, 
                        lf_dataset_path: str = 'data/lf_dataset.h5') -> Tuple[pd.DataFrame, Dict]:
        """Generate complete mid-fidelity dataset"""
        
        if n_samples is None:
            n_samples = self.mf_config['n_samples']
            
        print(f"Generating {n_samples} mid-fidelity samples...")
        
        # Select conditions from LF dataset or generate new ones
        conditions_with_cycles = self.select_conditions_from_lf(lf_dataset_path, n_samples)
        
        # Run simulations
        results_list = []
        fields_data = {
            'temperature_2d': [],
            'current_density_2d': [],
            'stress_2d': [],
            'damage_2d': [],
            'species_H2_2d': [],
            'species_H2O_2d': [],
            'species_O2_2d': []
        }
        
        for cond, cycles in tqdm(conditions_with_cycles[:n_samples], desc="MF Simulations"):
            try:
                result = self.model.simulate(cond, cycles)
                
                # Separate scalar and field data
                scalar_result = {k: v for k, v in result.items() 
                               if not k.endswith('_2d') and k != 'field_shape'}
                results_list.append(scalar_result)
                
                # Store field data separately
                nx, ny = result['field_shape']
                fields_data['temperature_2d'].append(
                    result['temperature_field_2d'].reshape(nx, ny))
                fields_data['current_density_2d'].append(
                    result['current_density_field_2d'].reshape(nx, ny))
                fields_data['stress_2d'].append(
                    result['von_mises_stress_2d'].reshape(nx, ny))
                fields_data['damage_2d'].append(
                    result['damage_field_2d'].reshape(nx, ny))
                fields_data['species_H2_2d'].append(
                    result['H2_concentration_2d'].reshape(nx, ny))
                fields_data['species_H2O_2d'].append(
                    result['H2O_concentration_2d'].reshape(nx, ny))
                fields_data['species_O2_2d'].append(
                    result['O2_concentration_2d'].reshape(nx, ny))
                
            except Exception as e:
                print(f"Simulation failed: {e}")
                continue
        
        # Convert to DataFrame (scalar data)
        df = pd.DataFrame(results_list)
        
        # Add metadata
        metadata = {
            'fidelity': 'mid',
            'model_type': self.mf_config['model_type'],
            'mesh_resolution': self.mf_config['mesh_resolution'],
            'n_samples': len(df),
            'generation_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        return df, fields_data, metadata
    
    def save_dataset(self, df: pd.DataFrame, fields_data: Dict, metadata: Dict,
                     filename: str = "mf_dataset.h5"):
        """Save dataset with 2D fields to HDF5 file"""
        
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save scalar DataFrame data
            scalar_group = f.create_group('scalar_data')
            for col in df.columns:
                scalar_group.create_dataset(col, data=df[col].values, 
                                          compression='gzip')
            
            # Save 2D field data
            fields_group = f.create_group('field_data')
            for field_name, field_list in fields_data.items():
                if field_list:
                    field_array = np.array(field_list)
                    fields_group.create_dataset(field_name, data=field_array,
                                              compression='gzip')
            
            # Save metadata
            f.attrs['metadata'] = json.dumps(metadata, default=str)
            f.attrs['scalar_columns'] = list(df.columns)
            f.attrs['field_names'] = list(fields_data.keys())
            
        print(f"Dataset saved to {filepath}")
        
        # Save scalar data as CSV for inspection
        csv_path = filepath.replace('.h5', '_scalars.csv')
        df.to_csv(csv_path, index=False)
        print(f"Scalar data CSV saved to {csv_path}")
        
        return filepath


def generate_mf_dataset(config_path: str = 'config.yaml', 
                       n_samples: int = None,
                       lf_dataset_path: str = 'data/lf_dataset.h5'):
    """Main function to generate mid-fidelity dataset"""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = MidFidelityDatasetGenerator(config)
    
    # Generate dataset
    df, fields_data, metadata = generator.generate_dataset(n_samples, lf_dataset_path)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nScalar feature ranges:")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    print("\n2D Field data shapes:")
    for field_name, field_list in fields_data.items():
        if field_list:
            print(f"  {field_name}: {len(field_list)} samples of shape {field_list[0].shape}")
    
    # Save dataset
    filepath = generator.save_dataset(df, fields_data, metadata)
    
    return df, fields_data, filepath


if __name__ == "__main__":
    # Generate mid-fidelity dataset
    df, fields, filepath = generate_mf_dataset(n_samples=100)  # Smaller sample for testing
    print(f"\nDataset generation complete! File saved at: {filepath}")