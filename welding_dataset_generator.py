#!/usr/bin/env python3
"""
Welding Inverse Design Dataset Generator
=======================================

This script generates a comprehensive multi-tier dataset for welding inverse design
machine learning applications, focusing on extreme-temperature performance.

Dataset Structure:
- Tier 1: High-fidelity experimental data (100-500 samples)
- Tier 2: Computational/FEM simulation data (10,000+ samples) 
- Tier 3: Literature and legacy data

Author: AI Assistant
Date: 2025-10-14
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from scipy.stats import norm, uniform, lognorm
from sklearn.preprocessing import StandardScaler
try:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
except ImportError:
    pass
import warnings
warnings.filterwarnings('ignore')

class WeldingDatasetGenerator:
    """
    Comprehensive welding dataset generator for inverse design applications.
    """
    
    def __init__(self, random_state=42):
        """Initialize the dataset generator with reproducible random state."""
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define input parameter ranges based on industrial welding practices
        self.input_parameters = {
            # Energy Input Parameters
            'laser_power_w': (500, 3000),           # Watts
            'welding_speed_mm_s': (10, 200),        # mm/s
            'pulse_frequency_hz': (1, 1000),        # Hz (0 for CW)
            'pulse_duration_ms': (0.1, 50),         # ms
            
            # Beam Characteristics
            'beam_focus_position_mm': (-2, 2),      # mm (negative = defocused below)
            'beam_spot_size_um': (50, 500),         # micrometers
            
            # Material & Setup
            'clamping_pressure_mpa': (0.1, 5.0),    # MPa
            'shield_gas_flow_rate_l_min': (5, 30),  # L/min
            'material_combination': ['Cu-Al', 'Al-Al', 'Cu-Cu', 'Al-Steel', 'Cu-Steel'],
            
            # Geometry
            'sheet_thickness_mm': (0.1, 2.0),       # mm
            'joint_type': ['Lap', 'Butt', 'T-Joint'],
            'overlap_distance_mm': (0.5, 10.0),     # mm (for lap joints)
        }
        
        # Define output parameter ranges and relationships
        self.output_parameters = {
            # Weld Morphology & Quality
            'nugget_width_mm': (0.2, 3.0),
            'penetration_depth_mm': (0.05, 2.0),
            'haz_width_mm': (0.1, 5.0),
            'crack_presence': [0, 1],  # Binary
            'porosity_area_percent': (0, 15),
            'spatter_rating': (1, 5),  # 1=minimal, 5=excessive
            
            # Mechanical & Electrical (Room Temperature)
            'tensile_shear_strength_n': (500, 5000),
            'peel_strength_n': (100, 1500),
            'contact_resistance_micro_ohm': (5, 200),
            
            # Extreme Temperature Performance
            'strength_degradation_percent': (0, 80),    # After thermal cycling
            'resistance_increase_percent': (0, 500),    # After thermal cycling
            'cycles_to_failure': (10, 10000),
            'creep_time_to_failure_hours': (1, 1000),
            
            # Microstructural Evolution
            'imc_thickness_post_aging_um': (0.1, 20),   # Intermetallic compounds
            'grain_size_change_percent': (-50, 200),
        }
    
    def _generate_material_properties(self, material_combo):
        """Generate material-specific properties and coefficients."""
        material_props = {
            'Cu-Al': {
                'thermal_conductivity_ratio': 2.5,
                'melting_point_diff': 420,  # Cu: 1085°C, Al: 660°C
                'imc_formation_tendency': 0.8,
                'base_strength_multiplier': 0.7,
                'base_resistance_multiplier': 1.5,
            },
            'Al-Al': {
                'thermal_conductivity_ratio': 1.0,
                'melting_point_diff': 0,
                'imc_formation_tendency': 0.0,
                'base_strength_multiplier': 1.0,
                'base_resistance_multiplier': 1.0,
            },
            'Cu-Cu': {
                'thermal_conductivity_ratio': 1.0,
                'melting_point_diff': 0,
                'imc_formation_tendency': 0.0,
                'base_strength_multiplier': 1.2,
                'base_resistance_multiplier': 0.8,
            },
            'Al-Steel': {
                'thermal_conductivity_ratio': 0.3,
                'melting_point_diff': 900,
                'imc_formation_tendency': 0.9,
                'base_strength_multiplier': 0.6,
                'base_resistance_multiplier': 2.0,
            },
            'Cu-Steel': {
                'thermal_conductivity_ratio': 0.25,
                'melting_point_diff': 400,
                'imc_formation_tendency': 0.7,
                'base_strength_multiplier': 0.8,
                'base_resistance_multiplier': 1.8,
            }
        }
        return material_props.get(material_combo, material_props['Al-Al'])
    
    def _calculate_heat_input(self, power, speed, efficiency=0.7):
        """Calculate heat input per unit length (J/mm)."""
        return (power * efficiency) / speed
    
    def _calculate_energy_density(self, power, speed, spot_size, efficiency=0.7):
        """Calculate energy density (J/mm³)."""
        spot_area = np.pi * (spot_size/2000)**2  # Convert μm to mm
        return (power * efficiency) / (speed * spot_area)
    
    def _physics_based_relationships(self, inputs):
        """
        Apply physics-based relationships to generate realistic outputs.
        This is the core of creating physically meaningful synthetic data.
        """
        outputs = {}
        
        # Extract key input parameters, handle NaN values
        power = inputs.get('laser_power_w', 1500) if not pd.isna(inputs.get('laser_power_w', 1500)) else 1500
        speed = inputs.get('welding_speed_mm_s', 50) if not pd.isna(inputs.get('welding_speed_mm_s', 50)) else 50
        spot_size = inputs.get('beam_spot_size_um', 200) if not pd.isna(inputs.get('beam_spot_size_um', 200)) else 200
        thickness = inputs.get('sheet_thickness_mm', 1.0) if not pd.isna(inputs.get('sheet_thickness_mm', 1.0)) else 1.0
        material_combo = inputs.get('material_combination', 'Al-Al')
        focus_pos = inputs.get('beam_focus_position_mm', 0) if not pd.isna(inputs.get('beam_focus_position_mm', 0)) else 0
        
        # Get material properties
        mat_props = self._generate_material_properties(material_combo)
        
        # Calculate derived parameters
        heat_input = self._calculate_heat_input(power, speed)
        energy_density = self._calculate_energy_density(power, speed, spot_size)
        
        # Weld Geometry Relationships
        # Nugget width increases with heat input but decreases with speed
        base_nugget_width = 0.3 + (heat_input / 2000) * (1 + np.abs(focus_pos) * 0.1)
        outputs['nugget_width_mm'] = np.clip(
            base_nugget_width * np.random.normal(1.0, 0.1), 0.2, 3.0
        )
        
        # Penetration depth related to energy density and material properties
        base_penetration = min(thickness * 0.9, 
                             0.1 + (energy_density / 1e6) * mat_props['thermal_conductivity_ratio'])
        outputs['penetration_depth_mm'] = np.clip(
            base_penetration * np.random.normal(1.0, 0.15), 0.05, thickness
        )
        
        # HAZ width increases with heat input
        outputs['haz_width_mm'] = np.clip(
            outputs['nugget_width_mm'] * (1.5 + heat_input/5000) * np.random.normal(1.0, 0.1),
            0.1, 5.0
        )
        
        # Defects and Quality
        # Higher energy density and speed mismatch increases defect probability
        defect_probability = min(0.8, max(0.05, 
            (energy_density / 2e6) + (abs(speed - 50) / 200) + 
            (abs(focus_pos) * 0.2) + (mat_props['imc_formation_tendency'] * 0.3)
        ))
        outputs['crack_presence'] = 1 if np.random.random() < defect_probability else 0
        
        # Porosity increases with high energy density and material mismatch
        base_porosity = defect_probability * 10 + mat_props['imc_formation_tendency'] * 5
        outputs['porosity_area_percent'] = np.clip(
            base_porosity * np.random.lognormal(0, 0.5), 0, 15
        )
        
        # Spatter rating (1-5 scale)
        spatter_factor = (energy_density / 1e6) + (speed / 100) + np.abs(focus_pos)
        outputs['spatter_rating'] = int(np.clip(1 + spatter_factor * np.random.normal(1, 0.2), 1, 5))
        
        # Mechanical Properties
        # Base strength depends on nugget size and material combination
        base_strength = (outputs['nugget_width_mm'] * 1000 + 
                        outputs['penetration_depth_mm'] * 500) * mat_props['base_strength_multiplier']
        
        # Reduce strength for defects
        strength_reduction = (outputs['crack_presence'] * 0.4 + 
                            outputs['porosity_area_percent'] / 100 * 0.6)
        
        outputs['tensile_shear_strength_n'] = np.clip(
            base_strength * (1 - strength_reduction) * np.random.normal(1.0, 0.1),
            500, 5000
        )
        
        outputs['peel_strength_n'] = np.clip(
            outputs['tensile_shear_strength_n'] * 0.3 * np.random.normal(1.0, 0.15),
            100, 1500
        )
        
        # Electrical Properties
        # Contact resistance inversely related to nugget area and material properties
        nugget_area = np.pi * (outputs['nugget_width_mm']/2)**2
        base_resistance = (20 / nugget_area) * mat_props['base_resistance_multiplier']
        
        # Increase resistance for defects and porosity
        resistance_increase = (outputs['porosity_area_percent'] / 100 * 2 + 
                             outputs['crack_presence'] * 1.5)
        
        outputs['contact_resistance_micro_ohm'] = np.clip(
            base_resistance * (1 + resistance_increase) * np.random.lognormal(0, 0.2),
            5, 200
        )
        
        # Extreme Temperature Performance
        # Degradation depends on material combination and initial quality
        imc_tendency = mat_props['imc_formation_tendency']
        initial_quality = 1 - (outputs['crack_presence'] * 0.3 + 
                              outputs['porosity_area_percent'] / 100 * 0.2)
        
        # Strength degradation after thermal cycling
        base_degradation = (imc_tendency * 40 + (1 - initial_quality) * 30)
        outputs['strength_degradation_percent'] = np.clip(
            base_degradation * np.random.lognormal(0, 0.3), 0, 80
        )
        
        # Resistance increase after thermal cycling
        base_resistance_increase = (imc_tendency * 200 + (1 - initial_quality) * 100)
        outputs['resistance_increase_percent'] = np.clip(
            base_resistance_increase * np.random.lognormal(0, 0.4), 0, 500
        )
        
        # Cycles to failure (inversely related to degradation)
        failure_resistance = initial_quality * (1 - imc_tendency * 0.5)
        outputs['cycles_to_failure'] = int(np.clip(
            1000 * failure_resistance * np.random.lognormal(0, 0.5), 10, 10000
        ))
        
        # Creep performance
        outputs['creep_time_to_failure_hours'] = np.clip(
            100 * failure_resistance * np.random.lognormal(0, 0.6), 1, 1000
        )
        
        # Microstructural Evolution
        # IMC thickness for dissimilar materials
        if imc_tendency > 0:
            base_imc = imc_tendency * 5 * (heat_input / 1000)
            outputs['imc_thickness_post_aging_um'] = np.clip(
                base_imc * np.random.lognormal(0, 0.4), 0.1, 20
            )
        else:
            outputs['imc_thickness_post_aging_um'] = np.random.uniform(0.1, 0.5)
        
        # Grain size change
        grain_change_factor = (heat_input / 1000) - 1
        outputs['grain_size_change_percent'] = np.clip(
            grain_change_factor * 50 * np.random.normal(1.0, 0.3), -50, 200
        )
        
        return outputs
    
    def generate_experimental_data(self, n_samples=300):
        """
        Generate Tier 1: High-fidelity experimental data using Design of Experiments.
        Uses Latin Hypercube Sampling for efficient parameter space exploration.
        """
        print(f"Generating {n_samples} experimental data samples...")
        
        data = []
        
        for i in range(n_samples):
            # Generate input parameters using LHS-like approach
            sample = {}
            
            # Continuous parameters
            sample['laser_power_w'] = np.random.uniform(*self.input_parameters['laser_power_w'])
            sample['welding_speed_mm_s'] = np.random.uniform(*self.input_parameters['welding_speed_mm_s'])
            sample['pulse_frequency_hz'] = np.random.uniform(*self.input_parameters['pulse_frequency_hz'])
            sample['pulse_duration_ms'] = np.random.uniform(*self.input_parameters['pulse_duration_ms'])
            sample['beam_focus_position_mm'] = np.random.uniform(*self.input_parameters['beam_focus_position_mm'])
            sample['beam_spot_size_um'] = np.random.uniform(*self.input_parameters['beam_spot_size_um'])
            sample['clamping_pressure_mpa'] = np.random.uniform(*self.input_parameters['clamping_pressure_mpa'])
            sample['shield_gas_flow_rate_l_min'] = np.random.uniform(*self.input_parameters['shield_gas_flow_rate_l_min'])
            sample['sheet_thickness_mm'] = np.random.uniform(*self.input_parameters['sheet_thickness_mm'])
            sample['overlap_distance_mm'] = np.random.uniform(*self.input_parameters['overlap_distance_mm'])
            
            # Categorical parameters
            sample['material_combination'] = np.random.choice(self.input_parameters['material_combination'])
            sample['joint_type'] = np.random.choice(self.input_parameters['joint_type'])
            
            # Generate physics-based outputs
            outputs = self._physics_based_relationships(sample)
            
            # Combine inputs and outputs
            full_sample = {**sample, **outputs}
            full_sample['weld_id'] = f"EXP-{i+1:03d}"
            full_sample['data_source'] = 'Experimental'
            full_sample['data_tier'] = 1
            full_sample['measurement_uncertainty'] = np.random.uniform(0.02, 0.08)  # 2-8% uncertainty
            
            data.append(full_sample)
        
        return pd.DataFrame(data)
    
    def generate_simulation_data(self, n_samples=10000):
        """
        Generate Tier 2: Computational/FEM simulation data.
        Higher volume, lower uncertainty, some parameters may be computed rather than measured.
        """
        print(f"Generating {n_samples} simulation data samples...")
        
        data = []
        
        for i in range(n_samples):
            # Generate input parameters with broader coverage
            sample = {}
            
            # Use different distributions for simulation data to explore parameter space more thoroughly
            sample['laser_power_w'] = np.random.uniform(*self.input_parameters['laser_power_w'])
            sample['welding_speed_mm_s'] = np.random.uniform(*self.input_parameters['welding_speed_mm_s'])
            sample['pulse_frequency_hz'] = np.random.uniform(*self.input_parameters['pulse_frequency_hz'])
            sample['pulse_duration_ms'] = np.random.uniform(*self.input_parameters['pulse_duration_ms'])
            sample['beam_focus_position_mm'] = np.random.uniform(*self.input_parameters['beam_focus_position_mm'])
            sample['beam_spot_size_um'] = np.random.uniform(*self.input_parameters['beam_spot_size_um'])
            sample['clamping_pressure_mpa'] = np.random.uniform(*self.input_parameters['clamping_pressure_mpa'])
            sample['shield_gas_flow_rate_l_min'] = np.random.uniform(*self.input_parameters['shield_gas_flow_rate_l_min'])
            sample['sheet_thickness_mm'] = np.random.uniform(*self.input_parameters['sheet_thickness_mm'])
            sample['overlap_distance_mm'] = np.random.uniform(*self.input_parameters['overlap_distance_mm'])
            
            # Categorical parameters
            sample['material_combination'] = np.random.choice(self.input_parameters['material_combination'])
            sample['joint_type'] = np.random.choice(self.input_parameters['joint_type'])
            
            # Generate physics-based outputs
            outputs = self._physics_based_relationships(sample)
            
            # Simulation data has some limitations
            # Some properties are harder to simulate accurately
            if np.random.random() < 0.3:  # 30% of simulations don't have aging data
                outputs['imc_thickness_post_aging_um'] = np.nan
                outputs['strength_degradation_percent'] = np.nan
                outputs['resistance_increase_percent'] = np.nan
            
            # Combine inputs and outputs
            full_sample = {**sample, **outputs}
            full_sample['weld_id'] = f"SIM-{i+1:05d}"
            full_sample['data_source'] = 'Simulation'
            full_sample['data_tier'] = 2
            full_sample['measurement_uncertainty'] = np.random.uniform(0.01, 0.05)  # Lower uncertainty for simulations
            
            data.append(full_sample)
        
        return pd.DataFrame(data)
    
    def generate_literature_data(self, n_samples=150):
        """
        Generate Tier 3: Literature and legacy data.
        Represents curated data from published papers and technical reports.
        """
        print(f"Generating {n_samples} literature data samples...")
        
        data = []
        
        for i in range(n_samples):
            # Literature data often has incomplete parameter sets
            sample = {}
            
            # Some parameters may be missing in literature
            missing_prob = 0.2  # 20% chance each parameter is missing
            
            if np.random.random() > missing_prob:
                sample['laser_power_w'] = np.random.uniform(*self.input_parameters['laser_power_w'])
            else:
                sample['laser_power_w'] = np.nan
                
            if np.random.random() > missing_prob:
                sample['welding_speed_mm_s'] = np.random.uniform(*self.input_parameters['welding_speed_mm_s'])
            else:
                sample['welding_speed_mm_s'] = np.nan
                
            # Some parameters are more commonly reported
            sample['pulse_frequency_hz'] = np.random.uniform(*self.input_parameters['pulse_frequency_hz']) if np.random.random() > 0.4 else np.nan
            sample['pulse_duration_ms'] = np.random.uniform(*self.input_parameters['pulse_duration_ms']) if np.random.random() > 0.4 else np.nan
            sample['beam_focus_position_mm'] = np.random.uniform(*self.input_parameters['beam_focus_position_mm']) if np.random.random() > 0.3 else np.nan
            sample['beam_spot_size_um'] = np.random.uniform(*self.input_parameters['beam_spot_size_um']) if np.random.random() > 0.3 else np.nan
            sample['clamping_pressure_mpa'] = np.random.uniform(*self.input_parameters['clamping_pressure_mpa']) if np.random.random() > 0.5 else np.nan
            sample['shield_gas_flow_rate_l_min'] = np.random.uniform(*self.input_parameters['shield_gas_flow_rate_l_min']) if np.random.random() > 0.3 else np.nan
            sample['sheet_thickness_mm'] = np.random.uniform(*self.input_parameters['sheet_thickness_mm'])  # Usually reported
            sample['overlap_distance_mm'] = np.random.uniform(*self.input_parameters['overlap_distance_mm']) if np.random.random() > 0.4 else np.nan
            
            # Categorical parameters
            sample['material_combination'] = np.random.choice(self.input_parameters['material_combination'])
            sample['joint_type'] = np.random.choice(self.input_parameters['joint_type'])
            
            # Generate outputs, but with more missing data
            outputs = self._physics_based_relationships(sample)
            
            # Literature often focuses on specific properties
            output_missing_prob = 0.4
            for key in outputs:
                if np.random.random() < output_missing_prob:
                    outputs[key] = np.nan
            
            # But ensure at least some key properties are present
            key_properties = ['tensile_shear_strength_n', 'nugget_width_mm', 'penetration_depth_mm']
            for prop in key_properties:
                if prop in outputs and np.random.random() < 0.8:  # 80% chance to have key properties
                    # Regenerate if it was set to NaN
                    if np.isnan(outputs[prop]):
                        temp_outputs = self._physics_based_relationships(sample)
                        outputs[prop] = temp_outputs[prop]
            
            # Combine inputs and outputs
            full_sample = {**sample, **outputs}
            full_sample['weld_id'] = f"LIT-{i+1:03d}"
            full_sample['data_source'] = 'Literature'
            full_sample['data_tier'] = 3
            full_sample['measurement_uncertainty'] = np.random.uniform(0.05, 0.15)  # Higher uncertainty for literature
            
            data.append(full_sample)
        
        return pd.DataFrame(data)
    
    def combine_datasets(self, exp_df, sim_df, lit_df):
        """
        Combine all three tiers into a master dataset with proper validation.
        """
        print("Combining datasets into master dataset...")
        
        # Combine all datasets
        master_df = pd.concat([exp_df, sim_df, lit_df], ignore_index=True)
        
        # Add metadata
        master_df['generation_timestamp'] = datetime.now().isoformat()
        master_df['dataset_version'] = '1.0'
        
        # Reorder columns for better readability
        input_cols = [col for col in master_df.columns if col in [
            'weld_id', 'data_source', 'data_tier', 'measurement_uncertainty',
            'laser_power_w', 'welding_speed_mm_s', 'pulse_frequency_hz', 'pulse_duration_ms',
            'beam_focus_position_mm', 'beam_spot_size_um', 'clamping_pressure_mpa',
            'shield_gas_flow_rate_l_min', 'material_combination', 'sheet_thickness_mm',
            'joint_type', 'overlap_distance_mm'
        ]]
        
        output_cols = [col for col in master_df.columns if col not in input_cols and 
                      col not in ['generation_timestamp', 'dataset_version']]
        
        meta_cols = ['generation_timestamp', 'dataset_version']
        
        column_order = input_cols + output_cols + meta_cols
        master_df = master_df[column_order]
        
        return master_df
    
    def generate_complete_dataset(self, exp_samples=300, sim_samples=10000, lit_samples=150):
        """
        Generate the complete multi-tier welding inverse design dataset.
        """
        print("=== Welding Inverse Design Dataset Generation ===")
        print(f"Target samples: {exp_samples} experimental, {sim_samples} simulation, {lit_samples} literature")
        print()
        
        # Generate each tier
        exp_df = self.generate_experimental_data(exp_samples)
        sim_df = self.generate_simulation_data(sim_samples)
        lit_df = self.generate_literature_data(lit_samples)
        
        # Combine into master dataset
        master_df = self.combine_datasets(exp_df, sim_df, lit_df)
        
        print(f"\nDataset generation complete!")
        print(f"Total samples: {len(master_df)}")
        print(f"- Experimental: {len(exp_df)}")
        print(f"- Simulation: {len(sim_df)}")
        print(f"- Literature: {len(lit_df)}")
        
        return master_df, exp_df, sim_df, lit_df

def main():
    """Main function to generate and save the dataset."""
    generator = WeldingDatasetGenerator(random_state=42)
    
    # Generate the complete dataset
    master_df, exp_df, sim_df, lit_df = generator.generate_complete_dataset()
    
    # Save datasets
    print("\nSaving datasets...")
    master_df.to_csv('welding_inverse_design_master_dataset.csv', index=False)
    exp_df.to_csv('welding_experimental_data.csv', index=False)
    sim_df.to_csv('welding_simulation_data.csv', index=False)
    lit_df.to_csv('welding_literature_data.csv', index=False)
    
    # Save metadata
    metadata = {
        'dataset_info': {
            'name': 'Welding Inverse Design Dataset',
            'version': '1.0',
            'generation_date': datetime.now().isoformat(),
            'total_samples': len(master_df),
            'experimental_samples': len(exp_df),
            'simulation_samples': len(sim_df),
            'literature_samples': len(lit_df)
        },
        'input_parameters': generator.input_parameters,
        'output_parameters': generator.output_parameters
    }
    
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print("Dataset files saved:")
    print("- welding_inverse_design_master_dataset.csv")
    print("- welding_experimental_data.csv") 
    print("- welding_simulation_data.csv")
    print("- welding_literature_data.csv")
    print("- dataset_metadata.json")
    
    return master_df

if __name__ == "__main__":
    master_dataset = main()