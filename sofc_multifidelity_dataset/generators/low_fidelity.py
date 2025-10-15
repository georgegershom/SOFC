"""
Low-Fidelity SOFC Model and Dataset Generator
Fast 1D lumped parameter models for large-scale dataset generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm
import h5py
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sofc_physics import (
    SOFCOperatingConditions, ElectrochemicalModel, 
    ThermalModel, MechanicalModel, DegradationModel
)


class LowFidelitySOFCModel:
    """Low-fidelity (1D lumped) SOFC model for fast computation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.electrochem = ElectrochemicalModel(config)
        self.thermal = ThermalModel(config)
        self.mechanical = MechanicalModel(config)
        self.degradation = DegradationModel(config)
        
    def simulate(self, conditions: SOFCOperatingConditions, 
                cycles: int = 0) -> Dict:
        """Run low-fidelity simulation"""
        
        # Electrochemical calculation
        V_cell, overpotentials = self.electrochem.cell_voltage(conditions)
        
        # Thermal calculation (1D)
        T_profile = self.thermal.temperature_distribution_1D(
            conditions, V_cell, overpotentials['E_nernst']
        )
        T_avg = np.mean(T_profile)
        T_max = np.max(T_profile)
        T_gradient = T_max - np.min(T_profile)
        
        # Mechanical calculation (simplified)
        sigma_avg = self.mechanical.thermal_stress(T_profile, T_ref=298.15)
        sigma_max = np.max(np.abs(sigma_avg))
        
        # Degradation calculation
        deg_metrics = self.degradation.total_degradation(
            conditions, overpotentials, sigma_avg, cycles
        )
        
        # Compile results
        results = {
            # Operating conditions (inputs)
            'temperature_inlet': conditions.temperature,
            'pressure': conditions.pressure,
            'current_density': conditions.current_density,
            'fuel_utilization': conditions.fuel_utilization,
            'air_utilization': conditions.air_utilization,
            'time': conditions.time,
            'cycles': cycles,
            
            # Electrochemical outputs
            'voltage': V_cell,
            'power_density': overpotentials['power_density'],
            'efficiency': V_cell / overpotentials['E_nernst'] if overpotentials['E_nernst'] > 0 else 0,
            'nernst_voltage': overpotentials['E_nernst'],
            'overpotential_activation_anode': overpotentials['eta_act_anode'],
            'overpotential_activation_cathode': overpotentials['eta_act_cathode'],
            'overpotential_ohmic': overpotentials['eta_ohmic'],
            'overpotential_concentration': overpotentials['eta_concentration'],
            
            # Thermal outputs
            'temperature_average': T_avg,
            'temperature_maximum': T_max,
            'temperature_gradient': T_gradient,
            
            # Mechanical outputs
            'stress_average': float(sigma_avg[0]) if sigma_avg.size > 0 else 0,
            'stress_maximum': sigma_max,
            
            # Degradation outputs
            'degradation_ni_coarsening': deg_metrics['nickel_coarsening'],
            'degradation_cr_poisoning': deg_metrics['chromium_poisoning'],
            'degradation_crack_density': deg_metrics['crack_density'],
            'degradation_delamination_risk': deg_metrics['delamination_risk'],
            'voltage_degradation_rate_mV_kh': deg_metrics['voltage_degradation_rate'],
            'estimated_lifetime_hours': deg_metrics['estimated_lifetime']
        }
        
        return results


class LowFidelityDatasetGenerator:
    """Generate large-scale low-fidelity SOFC dataset"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model = LowFidelitySOFCModel(config)
        self.lf_config = config['dataset']['low_fidelity']
        
    def sample_operating_conditions(self, n_samples: int) -> List[SOFCOperatingConditions]:
        """Generate random operating conditions"""
        op_ranges = self.config['operating_conditions']
        
        conditions_list = []
        
        # Latin Hypercube Sampling for better coverage
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=6)
        samples = sampler.random(n=n_samples)
        
        for i in range(n_samples):
            # Scale samples to operating ranges
            T = op_ranges['temperature']['min'] + \
                samples[i, 0] * (op_ranges['temperature']['max'] - op_ranges['temperature']['min'])
            
            p = op_ranges['pressure']['min'] + \
                samples[i, 1] * (op_ranges['pressure']['max'] - op_ranges['pressure']['min'])
            
            i_density = op_ranges['current_density']['min'] + \
                       samples[i, 2] * (op_ranges['current_density']['max'] - op_ranges['current_density']['min'])
            
            Uf = op_ranges['fuel_utilization']['min'] + \
                 samples[i, 3] * (op_ranges['fuel_utilization']['max'] - op_ranges['fuel_utilization']['min'])
            
            Ua = op_ranges['air_utilization']['min'] + \
                 samples[i, 4] * (op_ranges['air_utilization']['max'] - op_ranges['air_utilization']['min'])
            
            # Time (operating hours)
            time = samples[i, 5] * 50000  # Up to 50,000 hours
            
            # Fuel composition variations
            H2_frac = 0.7 + np.random.uniform(0, 0.27)
            H2O_frac = 0.03 + np.random.uniform(0, 0.27)
            # Normalize
            total = H2_frac + H2O_frac
            H2_frac /= total
            H2O_frac /= total
            
            fuel_comp = {'H2': H2_frac, 'H2O': H2O_frac}
            air_comp = {'O2': 0.21, 'N2': 0.79}
            
            conditions = SOFCOperatingConditions(
                temperature=T,
                pressure=p,
                current_density=i_density,
                fuel_utilization=Uf,
                air_utilization=Ua,
                inlet_fuel_composition=fuel_comp,
                inlet_air_composition=air_comp,
                time=time
            )
            
            conditions_list.append(conditions)
            
        return conditions_list
    
    def add_cycling_conditions(self, conditions_list: List[SOFCOperatingConditions]) -> List[Tuple[SOFCOperatingConditions, int]]:
        """Add cycling information to conditions"""
        conditions_with_cycles = []
        
        for cond in conditions_list:
            # Random number of cycles based on operating time
            if cond.time < 1000:
                cycles = np.random.poisson(10)
            elif cond.time < 10000:
                cycles = np.random.poisson(100)
            else:
                cycles = np.random.poisson(500)
                
            conditions_with_cycles.append((cond, cycles))
            
        return conditions_with_cycles
    
    def generate_dataset(self, n_samples: int = None) -> Tuple[pd.DataFrame, Dict]:
        """Generate complete low-fidelity dataset"""
        
        if n_samples is None:
            n_samples = self.lf_config['n_samples']
            
        print(f"Generating {n_samples} low-fidelity samples...")
        
        # Sample operating conditions
        conditions_list = self.sample_operating_conditions(n_samples)
        conditions_with_cycles = self.add_cycling_conditions(conditions_list)
        
        # Run simulations
        results_list = []
        
        for cond, cycles in tqdm(conditions_with_cycles, desc="LF Simulations"):
            try:
                result = self.model.simulate(cond, cycles)
                results_list.append(result)
            except Exception as e:
                print(f"Simulation failed: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(results_list)
        
        # Add metadata
        metadata = {
            'fidelity': 'low',
            'model_type': self.lf_config['model_type'],
            'n_samples': len(df),
            'generation_date': datetime.now().isoformat(),
            'config': self.config
        }
        
        return df, metadata
    
    def save_dataset(self, df: pd.DataFrame, metadata: Dict, 
                     filename: str = "lf_dataset.h5"):
        """Save dataset to HDF5 file"""
        
        filepath = os.path.join('data', filename)
        os.makedirs('data', exist_ok=True)
        
        with h5py.File(filepath, 'w') as f:
            # Save DataFrame as datasets
            for col in df.columns:
                f.create_dataset(f'data/{col}', data=df[col].values, 
                                compression='gzip')
            
            # Save metadata
            f.attrs['metadata'] = json.dumps(metadata, default=str)
            
            # Save column names
            f.attrs['columns'] = list(df.columns)
            
        print(f"Dataset saved to {filepath}")
        
        # Also save as CSV for easy inspection
        csv_path = filepath.replace('.h5', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV version saved to {csv_path}")
        
        return filepath


def generate_lf_dataset(config_path: str = 'config.yaml', n_samples: int = None):
    """Main function to generate low-fidelity dataset"""
    
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create generator
    generator = LowFidelityDatasetGenerator(config)
    
    # Generate dataset
    df, metadata = generator.generate_dataset(n_samples)
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total samples: {len(df)}")
    print("\nFeature ranges:")
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            print(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # Save dataset
    filepath = generator.save_dataset(df, metadata)
    
    return df, filepath


if __name__ == "__main__":
    # Generate low-fidelity dataset
    df, filepath = generate_lf_dataset(n_samples=1000)  # Start with smaller sample for testing
    print(f"\nDataset generation complete! File saved at: {filepath}")