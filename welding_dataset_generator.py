"""
Welding Inverse Design Dataset Generator
For PhD Research on Extreme-Temperature Performance of Laser Welds
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, uniform, truncnorm
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

class WeldingDatasetGenerator:
    """
    Generates synthetic welding data for inverse design ML models.
    Simulates the relationship between welding parameters and performance outcomes.
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.material_properties = self._init_material_properties()
        self.gas_properties = self._init_gas_properties()
        
    def _init_material_properties(self) -> Dict:
        """Initialize material properties database"""
        return {
            'Cu-Al': {
                'melting_point_diff': 548,  # °C difference
                'thermal_conductivity_ratio': 1.67,
                'imc_growth_rate': 2.5e-3,  # μm/hour at 120°C
                'base_strength': 250,  # MPa
                'base_resistance': 12  # μΩ
            },
            'Al-Al': {
                'melting_point_diff': 0,
                'thermal_conductivity_ratio': 1.0,
                'imc_growth_rate': 0,
                'base_strength': 180,
                'base_resistance': 8
            },
            'Al-Steel': {
                'melting_point_diff': 870,
                'thermal_conductivity_ratio': 0.15,
                'imc_growth_rate': 4.2e-3,
                'base_strength': 320,
                'base_resistance': 25
            },
            'Cu-Cu': {
                'melting_point_diff': 0,
                'thermal_conductivity_ratio': 1.0,
                'imc_growth_rate': 0,
                'base_strength': 350,
                'base_resistance': 5
            }
        }
    
    def _init_gas_properties(self) -> Dict:
        """Initialize shielding gas properties"""
        return {
            'Argon': {'ionization_potential': 15.76, 'thermal_conductivity': 0.0179},
            'Nitrogen': {'ionization_potential': 14.53, 'thermal_conductivity': 0.0260},
            'Helium': {'ionization_potential': 24.59, 'thermal_conductivity': 0.1520},
            'CO2': {'ionization_potential': 13.77, 'thermal_conductivity': 0.0166}
        }
    
    def generate_input_parameters(self, n_samples: int, tier: str = 'experimental') -> pd.DataFrame:
        """
        Generate input parameters (X) for welding experiments.
        
        Parameters:
        -----------
        n_samples: Number of samples to generate
        tier: 'experimental', 'simulation', or 'literature'
        
        Returns:
        --------
        DataFrame with input parameters
        """
        
        # Define parameter ranges based on tier
        if tier == 'experimental':
            # More conservative ranges for real experiments
            param_ranges = {
                'laser_power': (500, 3000),  # Watts
                'welding_speed': (10, 100),  # mm/s
                'pulse_frequency': (0, 1000),  # Hz (0 = continuous)
                'pulse_duration': (0, 20),  # ms
                'beam_focus_position': (-2, 2),  # mm
                'beam_spot_size': (100, 600),  # μm
                'clamping_pressure': (0.1, 5.0),  # MPa
                'gas_flow_rate': (10, 30),  # L/min
                'sheet_thickness': (0.2, 3.0),  # mm
                'overlap_distance': (1, 10)  # mm
            }
        elif tier == 'simulation':
            # Wider ranges for simulations
            param_ranges = {
                'laser_power': (300, 5000),
                'welding_speed': (5, 200),
                'pulse_frequency': (0, 2000),
                'pulse_duration': (0, 50),
                'beam_focus_position': (-5, 5),
                'beam_spot_size': (50, 1000),
                'clamping_pressure': (0, 10),
                'gas_flow_rate': (5, 50),
                'sheet_thickness': (0.1, 5.0),
                'overlap_distance': (0.5, 20)
            }
        else:  # literature
            # Mixed ranges from various sources
            param_ranges = {
                'laser_power': (400, 4000),
                'welding_speed': (8, 150),
                'pulse_frequency': (0, 1500),
                'pulse_duration': (0, 30),
                'beam_focus_position': (-3, 3),
                'beam_spot_size': (80, 800),
                'clamping_pressure': (0.05, 7),
                'gas_flow_rate': (8, 40),
                'sheet_thickness': (0.15, 4.0),
                'overlap_distance': (0.8, 15)
            }
        
        # Generate base parameters
        data = {}
        for param, (min_val, max_val) in param_ranges.items():
            if tier == 'experimental':
                # Use Latin Hypercube Sampling for better coverage
                data[param] = self._latin_hypercube_sample(min_val, max_val, n_samples)
            else:
                # Use uniform random sampling for simulations
                data[param] = np.random.uniform(min_val, max_val, n_samples)
        
        # Add categorical parameters
        data['material_combination'] = np.random.choice(
            list(self.material_properties.keys()), 
            n_samples
        )
        data['shield_gas_type'] = np.random.choice(
            list(self.gas_properties.keys()), 
            n_samples
        )
        data['joint_type'] = np.random.choice(
            ['Lap', 'Butt', 'T-joint', 'Edge'], 
            n_samples,
            p=[0.5, 0.3, 0.15, 0.05]  # Lap joints most common in batteries
        )
        
        # Calculate derived parameters
        df = pd.DataFrame(data)
        df['heat_input'] = (df['laser_power'] / df['welding_speed']) / 1000  # kJ/mm
        df['power_density'] = df['laser_power'] / (np.pi * (df['beam_spot_size']/2000)**2)  # W/mm²
        
        return df
    
    def _latin_hypercube_sample(self, min_val: float, max_val: float, n: int) -> np.ndarray:
        """Generate Latin Hypercube Samples for better parameter space coverage"""
        intervals = np.linspace(min_val, max_val, n + 1)
        samples = []
        for i in range(n):
            samples.append(np.random.uniform(intervals[i], intervals[i + 1]))
        np.random.shuffle(samples)
        return np.array(samples)
    
    def simulate_weld_outputs(self, inputs: pd.DataFrame, tier: str = 'experimental') -> pd.DataFrame:
        """
        Simulate output parameters (Y) based on inputs using physics-based models.
        
        Parameters:
        -----------
        inputs: DataFrame with input parameters
        tier: Data tier affecting noise and fidelity
        
        Returns:
        --------
        DataFrame with output parameters
        """
        
        outputs = pd.DataFrame()
        n = len(inputs)
        
        # Add noise based on tier
        noise_factor = {'experimental': 0.05, 'simulation': 0.02, 'literature': 0.08}[tier]
        
        for idx, row in inputs.iterrows():
            # Get material properties
            mat_props = self.material_properties[row['material_combination']]
            
            # --- Weld Morphology Calculations ---
            
            # Nugget width (empirical model based on heat input and spot size)
            base_width = 0.8 * row['beam_spot_size']/1000 + 0.3 * np.sqrt(row['heat_input'])
            width_factor = 1.0 + 0.1 * np.log(row['power_density']/1000)
            outputs.loc[idx, 'nugget_width'] = base_width * width_factor * (1 + np.random.normal(0, noise_factor))
            
            # Penetration depth (function of power density and material)
            penetration_factor = 1.0 - 0.2 * mat_props['thermal_conductivity_ratio']
            base_penetration = 0.5 * row['sheet_thickness'] * (row['power_density']/5000)**0.6
            outputs.loc[idx, 'penetration_depth'] = min(
                base_penetration * penetration_factor * (1 + np.random.normal(0, noise_factor)),
                row['sheet_thickness'] * 0.95  # Can't exceed sheet thickness
            )
            
            # HAZ width
            outputs.loc[idx, 'haz_width'] = outputs.loc[idx, 'nugget_width'] * 1.5 * (1 + np.random.normal(0, noise_factor * 0.5))
            
            # --- Defects Prediction ---
            
            # Porosity probability (affected by gas flow and speed)
            porosity_risk = 0.1 * (1 - row['gas_flow_rate']/30) + 0.2 * (row['welding_speed']/100)
            outputs.loc[idx, 'has_porosity'] = int(np.random.random() < porosity_risk)
            outputs.loc[idx, 'porosity_size'] = np.random.exponential(50) if outputs.loc[idx, 'has_porosity'] else 0  # μm
            
            # Crack probability (affected by material mismatch and cooling rate)
            crack_risk = 0.05 * mat_props['melting_point_diff']/500 + 0.1 * (row['welding_speed']/50)
            outputs.loc[idx, 'has_cracks'] = int(np.random.random() < crack_risk)
            
            # Spatter (affected by power density and gas)
            spatter_level = min(5, 0.5 * (row['power_density']/10000)**2)
            outputs.loc[idx, 'spatter_count'] = np.random.poisson(spatter_level)
            
            # --- Mechanical Properties ---
            
            # Tensile shear strength (N)
            geometry_factor = outputs.loc[idx, 'nugget_width'] * row['sheet_thickness']
            defect_reduction = 1.0 - 0.3 * outputs.loc[idx, 'has_porosity'] - 0.5 * outputs.loc[idx, 'has_cracks']
            base_strength = mat_props['base_strength'] * geometry_factor * 10  # Convert to N
            
            # Modify based on penetration quality
            penetration_quality = outputs.loc[idx, 'penetration_depth'] / row['sheet_thickness']
            strength_factor = 0.5 + 0.5 * min(1.0, penetration_quality * 1.5)
            
            outputs.loc[idx, 'tensile_strength'] = base_strength * strength_factor * defect_reduction * (1 + np.random.normal(0, noise_factor))
            
            # Peel strength (typically 30-50% of tensile)
            outputs.loc[idx, 'peel_strength'] = outputs.loc[idx, 'tensile_strength'] * np.random.uniform(0.3, 0.5)
            
            # --- Electrical Properties ---
            
            # Contact resistance (μΩ)
            base_resistance = mat_props['base_resistance']
            resistance_factor = 1.0 + 0.5 * outputs.loc[idx, 'has_porosity'] + 0.2 * (1 - penetration_quality)
            outputs.loc[idx, 'contact_resistance'] = base_resistance * resistance_factor * (1 + np.random.normal(0, noise_factor * 0.5))
            
            # --- Extreme Temperature Performance ---
            
            # Thermal cycling degradation
            imc_growth = mat_props['imc_growth_rate']
            if imc_growth > 0:  # Dissimilar metals
                # IMC thickness after aging (μm)
                aging_time = 500  # hours at 120°C equivalent
                outputs.loc[idx, 'imc_thickness_initial'] = np.random.uniform(0.5, 2.0)
                outputs.loc[idx, 'imc_thickness_post_aging'] = (
                    outputs.loc[idx, 'imc_thickness_initial'] + 
                    imc_growth * aging_time * np.random.uniform(0.8, 1.2)
                )
            else:
                outputs.loc[idx, 'imc_thickness_initial'] = 0
                outputs.loc[idx, 'imc_thickness_post_aging'] = 0
            
            # Cycles to failure (thermal cycling)
            base_cycles = 1000
            imc_factor = 1.0 - min(0.8, outputs.loc[idx, 'imc_thickness_post_aging'] / 10)
            defect_factor = 1.0 - 0.3 * outputs.loc[idx, 'has_cracks'] - 0.2 * outputs.loc[idx, 'has_porosity']
            outputs.loc[idx, 'cycles_to_failure'] = int(
                base_cycles * imc_factor * defect_factor * 
                np.random.lognormal(0, 0.3)
            )
            
            # Strength degradation after thermal cycling (%)
            degradation_base = 5 + 10 * (outputs.loc[idx, 'imc_thickness_post_aging'] / 5)
            outputs.loc[idx, 'strength_degradation_percent'] = min(
                50, 
                degradation_base * (1 + np.random.normal(0, 0.1))
            )
            
            # Resistance increase after thermal cycling (%)
            resistance_increase = 10 + 15 * (outputs.loc[idx, 'imc_thickness_post_aging'] / 5)
            outputs.loc[idx, 'resistance_increase_percent'] = min(
                100,
                resistance_increase * (1 + np.random.normal(0, 0.1))
            )
            
            # Creep test results (hours to failure at 100°C, 50% UTS)
            creep_resistance = 100 * strength_factor * defect_factor
            outputs.loc[idx, 'creep_time_to_failure'] = creep_resistance * np.random.lognormal(0, 0.2)
            
            # Grain size evolution (μm)
            outputs.loc[idx, 'grain_size_initial'] = np.random.uniform(5, 20)
            outputs.loc[idx, 'grain_size_post_aging'] = outputs.loc[idx, 'grain_size_initial'] * np.random.uniform(1.2, 2.0)
        
        return outputs
    
    def generate_tier1_experimental_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate high-fidelity experimental data"""
        inputs = self.generate_input_parameters(n_samples, tier='experimental')
        outputs = self.simulate_weld_outputs(inputs, tier='experimental')
        
        # Combine inputs and outputs
        data = pd.concat([inputs, outputs], axis=1)
        
        # Add metadata
        data['weld_id'] = [f'W-{str(i+1).zfill(4)}' for i in range(n_samples)]
        data['data_source'] = 'Experimental'
        data['data_tier'] = 1
        data['timestamp'] = datetime.now().isoformat()
        
        # Add measurement uncertainty
        for col in outputs.columns:
            if outputs[col].dtype in [np.float64, np.int64]:
                data[f'{col}_std'] = np.abs(data[col] * np.random.uniform(0.02, 0.05, n_samples))
        
        return data
    
    def generate_tier2_simulation_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate computational simulation data"""
        inputs = self.generate_input_parameters(n_samples, tier='simulation')
        outputs = self.simulate_weld_outputs(inputs, tier='simulation')
        
        # Add additional simulation-specific outputs
        for idx in range(len(outputs)):
            # Temperature field data
            outputs.loc[idx, 'peak_temperature'] = 1500 + 1000 * (inputs.loc[idx, 'power_density']/10000)
            outputs.loc[idx, 'cooling_rate'] = 50 + 100 * inputs.loc[idx, 'welding_speed']/50
            
            # Residual stress (MPa)
            outputs.loc[idx, 'residual_stress_longitudinal'] = np.random.uniform(50, 200)
            outputs.loc[idx, 'residual_stress_transverse'] = np.random.uniform(30, 150)
            
            # Strain fields
            outputs.loc[idx, 'max_plastic_strain'] = np.random.uniform(0.001, 0.05)
            outputs.loc[idx, 'max_elastic_strain'] = np.random.uniform(0.0001, 0.005)
        
        # Combine inputs and outputs
        data = pd.concat([inputs, outputs], axis=1)
        
        # Add metadata
        data['weld_id'] = [f'S-{str(i+1).zfill(5)}' for i in range(n_samples)]
        data['data_source'] = 'Simulation'
        data['data_tier'] = 2
        data['timestamp'] = datetime.now().isoformat()
        data['simulation_software'] = np.random.choice(['ANSYS', 'COMSOL', 'Abaqus'], n_samples)
        data['mesh_quality'] = np.random.choice(['Fine', 'Medium', 'Coarse'], n_samples, p=[0.3, 0.5, 0.2])
        
        return data
    
    def generate_tier3_literature_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate literature-based augmented data"""
        inputs = self.generate_input_parameters(n_samples, tier='literature')
        outputs = self.simulate_weld_outputs(inputs, tier='literature')
        
        # Combine inputs and outputs
        data = pd.concat([inputs, outputs], axis=1)
        
        # Add metadata
        data['weld_id'] = [f'L-{str(i+1).zfill(4)}' for i in range(n_samples)]
        data['data_source'] = 'Literature'
        data['data_tier'] = 3
        data['timestamp'] = datetime.now().isoformat()
        
        # Add literature-specific metadata
        sources = ['DOI:10.1016/j.jmatprotec.2023', 'DOI:10.1115/1.4054987', 
                  'DOI:10.1007/s00170-022', 'Conference:AWS2023', 'TechReport:ORNL-2023']
        data['reference'] = np.random.choice(sources, n_samples)
        data['year'] = np.random.choice(range(2018, 2024), n_samples)
        
        # Some literature data might have missing values
        missing_prob = 0.1
        for col in outputs.columns:
            mask = np.random.random(n_samples) < missing_prob
            data.loc[mask, col] = np.nan
        
        return data
    
    def generate_complete_dataset(self, 
                                 tier1_samples: int = 500,
                                 tier2_samples: int = 10000,
                                 tier3_samples: int = 1000) -> pd.DataFrame:
        """Generate complete multi-tier dataset"""
        
        print("Generating Tier 1: Experimental Data...")
        tier1 = self.generate_tier1_experimental_data(tier1_samples)
        
        print("Generating Tier 2: Simulation Data...")
        tier2 = self.generate_tier2_simulation_data(tier2_samples)
        
        print("Generating Tier 3: Literature Data...")
        tier3 = self.generate_tier3_literature_data(tier3_samples)
        
        # Combine all tiers
        complete_dataset = pd.concat([tier1, tier2, tier3], ignore_index=True)
        
        # Add global unique ID
        complete_dataset['global_id'] = [f'GW-{str(i+1).zfill(6)}' for i in range(len(complete_dataset))]
        
        # Calculate quality metrics
        complete_dataset['quality_score'] = self._calculate_quality_score(complete_dataset)
        
        print(f"Complete dataset generated: {len(complete_dataset)} samples")
        
        return complete_dataset
    
    def _calculate_quality_score(self, data: pd.DataFrame) -> np.ndarray:
        """Calculate quality score for each weld based on multiple factors"""
        scores = np.zeros(len(data))
        
        for idx in range(len(data)):
            # Base score from tensile strength
            strength_score = min(1.0, data.loc[idx, 'tensile_strength'] / 3000) if not pd.isna(data.loc[idx, 'tensile_strength']) else 0.5
            
            # Resistance score (lower is better)
            resistance_score = max(0, 1.0 - data.loc[idx, 'contact_resistance'] / 50) if not pd.isna(data.loc[idx, 'contact_resistance']) else 0.5
            
            # Defect penalty
            defect_penalty = (1.0 - 0.3 * data.loc[idx, 'has_cracks'] - 0.2 * data.loc[idx, 'has_porosity']) if 'has_cracks' in data.columns else 1.0
            
            # Thermal performance
            thermal_score = min(1.0, data.loc[idx, 'cycles_to_failure'] / 1000) if not pd.isna(data.loc[idx, 'cycles_to_failure']) else 0.5
            
            # Combined score
            scores[idx] = (strength_score * 0.3 + resistance_score * 0.3 + 
                          defect_penalty * 0.2 + thermal_score * 0.2)
        
        return scores
    
    def save_dataset(self, dataset: pd.DataFrame, base_path: str = 'welding_dataset'):
        """Save dataset in multiple formats"""
        os.makedirs(base_path, exist_ok=True)
        
        # Save as CSV
        dataset.to_csv(f'{base_path}/complete_dataset.csv', index=False)
        
        # Save as Parquet (more efficient)
        dataset.to_parquet(f'{base_path}/complete_dataset.parquet', index=False)
        
        # Save metadata
        metadata = {
            'total_samples': len(dataset),
            'tier1_samples': len(dataset[dataset['data_tier'] == 1]),
            'tier2_samples': len(dataset[dataset['data_tier'] == 2]),
            'tier3_samples': len(dataset[dataset['data_tier'] == 3]),
            'features': {
                'input': [col for col in dataset.columns if col in [
                    'laser_power', 'welding_speed', 'pulse_frequency', 'pulse_duration',
                    'beam_focus_position', 'beam_spot_size', 'clamping_pressure',
                    'gas_flow_rate', 'sheet_thickness', 'overlap_distance',
                    'material_combination', 'shield_gas_type', 'joint_type'
                ]],
                'output': [col for col in dataset.columns if col in [
                    'nugget_width', 'penetration_depth', 'haz_width', 'has_porosity',
                    'porosity_size', 'has_cracks', 'spatter_count', 'tensile_strength',
                    'peel_strength', 'contact_resistance', 'imc_thickness_initial',
                    'imc_thickness_post_aging', 'cycles_to_failure',
                    'strength_degradation_percent', 'resistance_increase_percent',
                    'creep_time_to_failure', 'grain_size_initial', 'grain_size_post_aging'
                ]]
            },
            'generation_date': datetime.now().isoformat(),
            'version': '1.0.0'
        }
        
        with open(f'{base_path}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {base_path}/")
        
        # Generate summary statistics
        summary_stats = dataset.describe()
        summary_stats.to_csv(f'{base_path}/summary_statistics.csv')
        
        return metadata


if __name__ == "__main__":
    # Initialize generator
    generator = WeldingDatasetGenerator(seed=42)
    
    # Generate complete dataset
    print("Starting dataset generation...")
    dataset = generator.generate_complete_dataset(
        tier1_samples=500,
        tier2_samples=10000,
        tier3_samples=1000
    )
    
    # Save dataset
    generator.save_dataset(dataset, 'welding_dataset')
    
    print("\nDataset generation complete!")
    print(f"Total samples: {len(dataset)}")
    print(f"Features: {len(dataset.columns)}")
    print("\nDataset saved to welding_dataset/")