"""
Welding Inverse Design Dataset Generator
Generates synthetic datasets for laser welding with extreme-temperature performance data
Includes Tier 1 (Experimental) and Tier 2 (Computational) data
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
import json
from datetime import datetime
import os

# Set random seed for reproducibility
np.random.seed(42)

class WeldingDataGenerator:
    """
    Generates synthetic welding data with physics-based relationships
    between input parameters and output performance metrics
    """
    
    def __init__(self):
        # Material combinations
        self.materials = ['Cu-Al', 'Al-Al', 'Al-Steel', 'Cu-Cu']
        self.joint_types = ['Lap', 'Butt']
        self.shield_gases = ['Argon', 'Nitrogen', 'Helium', 'Argon-Helium']
        
    def truncated_normal(self, mean, std, low, high, size=1):
        """Generate truncated normal distribution"""
        a, b = (low - mean) / std, (high - mean) / std
        return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    
    def generate_input_parameters(self, n_samples, data_source='Experimental'):
        """Generate input parameters (X) for welding process"""
        
        # Energy Input Parameters
        laser_power = self.truncated_normal(1500, 300, 800, 2500, n_samples)  # Watts
        welding_speed = self.truncated_normal(50, 15, 10, 120, n_samples)  # mm/s
        pulse_frequency = self.truncated_normal(50, 20, 10, 200, n_samples)  # Hz
        pulse_duration = self.truncated_normal(5, 2, 1, 15, n_samples)  # ms
        
        # Beam Characteristics
        beam_focus_position = self.truncated_normal(0, 0.5, -2, 2, n_samples)  # mm
        beam_spot_size = self.truncated_normal(200, 50, 50, 400, n_samples)  # µm
        
        # Material & Setup
        clamping_pressure = self.truncated_normal(50, 15, 10, 100, n_samples)  # kPa
        shield_gas_flow = self.truncated_normal(15, 5, 5, 30, n_samples)  # L/min
        material_combination = np.random.choice(self.materials, n_samples)
        shield_gas_type = np.random.choice(self.shield_gases, n_samples)
        joint_type = np.random.choice(self.joint_types, n_samples)
        
        # Geometry
        sheet_thickness = self.truncated_normal(1.5, 0.5, 0.5, 3.5, n_samples)  # mm
        overlap_distance = self.truncated_normal(2.5, 0.8, 1, 5, n_samples)  # mm
        
        # Heat input calculation (derived parameter)
        heat_input = (laser_power * 60) / (welding_speed * 1000)  # J/mm
        
        # Energy density
        energy_density = laser_power / (beam_spot_size * welding_speed)  # J/mm³
        
        data = {
            'Laser_Power_W': laser_power,
            'Welding_Speed_mm_s': welding_speed,
            'Pulse_Frequency_Hz': pulse_frequency,
            'Pulse_Duration_ms': pulse_duration,
            'Beam_Focus_Position_mm': beam_focus_position,
            'Beam_Spot_Size_um': beam_spot_size,
            'Clamping_Pressure_kPa': clamping_pressure,
            'Shield_Gas_Flow_L_min': shield_gas_flow,
            'Material_Combination': material_combination,
            'Shield_Gas_Type': shield_gas_type,
            'Joint_Type': joint_type,
            'Sheet_Thickness_mm': sheet_thickness,
            'Overlap_Distance_mm': overlap_distance,
            'Heat_Input_J_mm': heat_input,
            'Energy_Density': energy_density,
            'Data_Source': [data_source] * n_samples
        }
        
        return pd.DataFrame(data)
    
    def calculate_weld_morphology(self, inputs_df):
        """
        Calculate weld morphology based on input parameters
        Using physics-based correlations
        """
        
        # Extract key parameters
        power = inputs_df['Laser_Power_W'].values
        speed = inputs_df['Welding_Speed_mm_s'].values
        heat_input = inputs_df['Heat_Input_J_mm'].values
        focus = inputs_df['Beam_Focus_Position_mm'].values
        spot_size = inputs_df['Beam_Spot_Size_um'].values
        thickness = inputs_df['Sheet_Thickness_mm'].values
        
        # Nugget width (increases with heat input, decreases with speed)
        base_width = 0.5 + 0.0008 * power - 0.01 * speed + 0.15 * heat_input
        nugget_width = np.maximum(0.3, base_width + np.random.normal(0, 0.1, len(power)))
        
        # Penetration depth (affected by focus position and energy density)
        base_depth = 0.3 + 0.0005 * power - 0.005 * speed - 0.1 * np.abs(focus)
        penetration_depth = np.clip(base_depth + np.random.normal(0, 0.08, len(power)), 0.1, thickness * 0.95)
        
        # Heat affected zone (HAZ) width
        haz_width = nugget_width * 1.5 + 0.002 * heat_input + np.random.normal(0, 0.15, len(power))
        haz_width = np.maximum(nugget_width + 0.2, haz_width)
        
        # Defect probabilities (lower quality with extreme parameters)
        energy_ratio = inputs_df['Energy_Density'].values / 30.0
        defect_probability = 1 / (1 + np.exp(-3 * (energy_ratio - 1)))
        
        has_cracks = (np.random.random(len(power)) < defect_probability * 0.3).astype(int)
        has_porosity = (np.random.random(len(power)) < defect_probability * 0.4).astype(int)
        has_undercut = (np.random.random(len(power)) < defect_probability * 0.25).astype(int)
        has_expulsion = (np.random.random(len(power)) < defect_probability * 0.2).astype(int)
        
        # Porosity size (if present)
        porosity_size = np.where(has_porosity, 
                                  np.random.uniform(10, 100, len(power)), 
                                  0)  # µm
        
        # Weld spatter rating (0-10 scale)
        spatter_rating = np.clip(
            5 + 0.002 * (power - 1500) + 2 * (speed - 50) / 50 + np.random.normal(0, 1, len(power)),
            0, 10
        )
        
        return {
            'Nugget_Width_mm': nugget_width,
            'Penetration_Depth_mm': penetration_depth,
            'HAZ_Width_mm': haz_width,
            'Has_Cracks': has_cracks,
            'Has_Porosity': has_porosity,
            'Porosity_Size_um': porosity_size,
            'Has_Undercut': has_undercut,
            'Has_Expulsion': has_expulsion,
            'Spatter_Rating': spatter_rating
        }
    
    def calculate_mechanical_properties(self, inputs_df, morphology_df):
        """
        Calculate mechanical and electrical properties at room temperature
        """
        
        # Extract parameters
        nugget_width = morphology_df['Nugget_Width_mm'].values
        penetration = morphology_df['Penetration_Depth_mm'].values
        has_defects = (morphology_df['Has_Cracks'].values + 
                      morphology_df['Has_Porosity'].values + 
                      morphology_df['Has_Undercut'].values).clip(0, 1)
        
        material = inputs_df['Material_Combination'].values
        thickness = inputs_df['Sheet_Thickness_mm'].values
        
        # Base strength depends on weld area and material
        material_strength_map = {'Cu-Al': 2800, 'Al-Al': 3500, 'Al-Steel': 3200, 'Cu-Cu': 3000}
        base_strength = np.array([material_strength_map[m] for m in material])
        
        # Tensile shear strength (N)
        weld_area = nugget_width * penetration
        strength_factor = 1 - 0.3 * has_defects
        tensile_strength = base_strength * weld_area * strength_factor + np.random.normal(0, 200, len(nugget_width))
        tensile_strength = np.maximum(500, tensile_strength)
        
        # Peel strength (N) - typically lower than tensile
        peel_strength = tensile_strength * 0.6 + np.random.normal(0, 150, len(nugget_width))
        peel_strength = np.maximum(300, peel_strength)
        
        # Contact resistance (µΩ) - lower is better
        # Dissimilar metals have higher resistance due to IMC
        material_resistance_map = {'Cu-Al': 25, 'Al-Al': 12, 'Al-Steel': 35, 'Cu-Cu': 8}
        base_resistance = np.array([material_resistance_map[m] for m in material])
        
        resistance_factor = 1 + 0.5 * has_defects + 0.2 * (1 / nugget_width)
        contact_resistance = base_resistance * resistance_factor + np.random.normal(0, 3, len(nugget_width))
        contact_resistance = np.maximum(5, contact_resistance)
        
        return {
            'Tensile_Shear_Strength_N': tensile_strength,
            'Peel_Strength_N': peel_strength,
            'Contact_Resistance_uOhm': contact_resistance
        }
    
    def calculate_extreme_temp_performance(self, inputs_df, morphology_df, mechanical_df):
        """
        Calculate extreme-temperature performance metrics
        This is the core output for inverse design
        """
        
        material = inputs_df['Material_Combination'].values
        heat_input = inputs_df['Heat_Input_J_mm'].values
        nugget_width = morphology_df['Nugget_Width_mm'].values
        penetration = morphology_df['Penetration_Depth_mm'].values
        has_defects = (morphology_df['Has_Cracks'].values + 
                      morphology_df['Has_Porosity'].values).clip(0, 1)
        
        initial_strength = mechanical_df['Tensile_Shear_Strength_N'].values
        initial_resistance = mechanical_df['Contact_Resistance_uOhm'].values
        
        # Thermal cycling performance (-40°C to +85°C, 1000 cycles)
        # Quality welds degrade less
        base_degradation = 0.15 + 0.05 * has_defects + 0.02 * (heat_input - 90) / 50
        strength_degradation_pct = np.clip(base_degradation + np.random.normal(0, 0.05, len(material)), 0, 0.5) * 100
        
        resistance_increase_pct = np.clip(
            (0.25 + 0.15 * has_defects + 0.03 * (heat_input - 90) / 50) + np.random.normal(0, 0.08, len(material)),
            0, 1.0
        ) * 100
        
        # Cycles to failure (for thermal cycling)
        quality_factor = 1 - 0.4 * has_defects
        material_cycle_map = {'Cu-Al': 800, 'Al-Al': 1200, 'Al-Steel': 700, 'Cu-Cu': 1000}
        base_cycles = np.array([material_cycle_map[m] for m in material])
        
        cycles_to_failure = (base_cycles * quality_factor * (nugget_width / 1.2) * (penetration / 0.8) +
                            np.random.normal(0, 100, len(material)))
        cycles_to_failure = np.maximum(200, cycles_to_failure).astype(int)
        
        # High-temperature stability
        # Creep test: time to failure at 100°C with 50% UTS load (hours)
        creep_time_hours = np.clip(
            500 * quality_factor * (1 - 0.3 * has_defects) + np.random.normal(0, 80, len(material)),
            50, 2000
        )
        
        # Static aging: property retention after 500 hours at 120°C
        aging_strength_retention_pct = np.clip(
            (0.75 + 0.15 * quality_factor - 0.1 * has_defects) + np.random.normal(0, 0.05, len(material)),
            0.5, 0.95
        ) * 100
        
        aging_resistance_increase_pct = np.clip(
            (0.35 + 0.2 * has_defects) + np.random.normal(0, 0.08, len(material)),
            0, 0.8
        ) * 100
        
        # Microstructural evolution
        # IMC thickness after aging (µm) - critical for dissimilar metals
        material_imc_map = {'Cu-Al': 3.5, 'Al-Al': 0.0, 'Al-Steel': 2.8, 'Cu-Cu': 0.0}
        base_imc = np.array([material_imc_map[m] for m in material])
        
        imc_growth_factor = 1 + 0.03 * heat_input + 0.5 * has_defects
        imc_thickness_post_aging = base_imc * imc_growth_factor + np.random.normal(0, 0.3, len(material))
        imc_thickness_post_aging = np.maximum(0, imc_thickness_post_aging)
        
        # Grain size change (%) - positive means coarsening
        grain_size_change_pct = np.clip(
            15 + 5 * heat_input / 90 + np.random.normal(0, 5, len(material)),
            0, 50
        )
        
        # Overall quality score (0-100)
        quality_score = 100 * (
            0.3 * (cycles_to_failure / 1500) +
            0.25 * (1 - strength_degradation_pct / 100) +
            0.25 * (1 - resistance_increase_pct / 100) +
            0.2 * (aging_strength_retention_pct / 100)
        )
        quality_score = np.clip(quality_score, 0, 100)
        
        return {
            'Thermal_Cycling_Strength_Degradation_pct': strength_degradation_pct,
            'Thermal_Cycling_Resistance_Increase_pct': resistance_increase_pct,
            'Cycles_to_Failure': cycles_to_failure,
            'Creep_Time_to_Failure_hours': creep_time_hours,
            'Static_Aging_Strength_Retention_pct': aging_strength_retention_pct,
            'Static_Aging_Resistance_Increase_pct': aging_resistance_increase_pct,
            'IMC_Thickness_Post_Aging_um': imc_thickness_post_aging,
            'Grain_Size_Change_pct': grain_size_change_pct,
            'Overall_Quality_Score': quality_score
        }
    
    def generate_dataset(self, n_samples, data_source='Experimental'):
        """
        Generate complete dataset with all input and output parameters
        """
        print(f"Generating {data_source} dataset with {n_samples} samples...")
        
        # Generate inputs
        inputs_df = self.generate_input_parameters(n_samples, data_source)
        
        # Calculate outputs
        morphology = self.calculate_weld_morphology(inputs_df)
        morphology_df = pd.DataFrame(morphology)
        
        mechanical = self.calculate_mechanical_properties(inputs_df, morphology_df)
        mechanical_df = pd.DataFrame(mechanical)
        
        extreme_temp = self.calculate_extreme_temp_performance(inputs_df, morphology_df, mechanical_df)
        extreme_temp_df = pd.DataFrame(extreme_temp)
        
        # Combine all dataframes
        result_df = pd.concat([inputs_df, morphology_df, mechanical_df, extreme_temp_df], axis=1)
        
        # Add weld ID
        prefix = 'W' if data_source == 'Experimental' else 'S'
        result_df.insert(0, 'Weld_ID', [f'{prefix}-{i+1:05d}' for i in range(n_samples)])
        
        # Add timestamp
        result_df['Timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add measurement uncertainty for experimental data
        if data_source == 'Experimental':
            result_df['Measurement_Uncertainty_pct'] = np.random.uniform(1, 5, n_samples)
            result_df['Replicate_Count'] = np.random.randint(3, 6, n_samples)
        
        print(f"Generated {len(result_df)} samples with {len(result_df.columns)} features")
        return result_df
    
    def add_simulation_noise(self, df):
        """
        Add characteristic simulation noise/bias to computational data
        Simulations are typically more uniform and slightly optimistic
        """
        df = df.copy()
        
        # Simulations typically don't capture all defects
        df['Has_Cracks'] = (df['Has_Cracks'] * 0.7).astype(int)
        df['Has_Porosity'] = (df['Has_Porosity'] * 0.6).astype(int)
        
        # Simulations slightly overestimate performance
        df['Tensile_Shear_Strength_N'] *= 1.05
        df['Cycles_to_Failure'] = (df['Cycles_to_Failure'] * 1.1).astype(int)
        df['Overall_Quality_Score'] = np.clip(df['Overall_Quality_Score'] * 1.08, 0, 100)
        
        # Add simulation convergence quality indicator
        df['Simulation_Convergence_Quality'] = np.random.uniform(0.85, 0.99, len(df))
        
        return df


def generate_all_datasets(output_dir='welding_datasets'):
    """
    Main function to generate all datasets
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    generator = WeldingDataGenerator()
    
    # Generate Tier 1: Experimental Data (500 samples)
    print("\n" + "="*80)
    print("TIER 1: EXPERIMENTAL DATA GENERATION")
    print("="*80)
    experimental_df = generator.generate_dataset(500, 'Experimental')
    
    exp_file = os.path.join(output_dir, 'tier1_experimental_data.csv')
    experimental_df.to_csv(exp_file, index=False)
    print(f"✓ Saved to: {exp_file}")
    
    # Generate Tier 2: Computational Data (10,000 samples)
    print("\n" + "="*80)
    print("TIER 2: COMPUTATIONAL/SIMULATION DATA GENERATION")
    print("="*80)
    computational_df = generator.generate_dataset(10000, 'Computational')
    computational_df = generator.add_simulation_noise(computational_df)
    
    comp_file = os.path.join(output_dir, 'tier2_computational_data.csv')
    computational_df.to_csv(comp_file, index=False)
    print(f"✓ Saved to: {comp_file}")
    
    # Generate Combined Master Dataset
    print("\n" + "="*80)
    print("MASTER DATASET GENERATION")
    print("="*80)
    master_df = pd.concat([experimental_df, computational_df], ignore_index=True)
    
    master_file = os.path.join(output_dir, 'master_dataset.csv')
    master_df.to_csv(master_file, index=False)
    print(f"✓ Combined dataset: {len(master_df)} total samples")
    print(f"✓ Saved to: {master_file}")
    
    # Generate dataset statistics
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)
    
    stats = {
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(master_df),
        'experimental_samples': len(experimental_df),
        'computational_samples': len(computational_df),
        'n_features': len(master_df.columns),
        'input_parameters': [
            'Laser_Power_W', 'Welding_Speed_mm_s', 'Pulse_Frequency_Hz',
            'Pulse_Duration_ms', 'Beam_Focus_Position_mm', 'Beam_Spot_Size_um',
            'Clamping_Pressure_kPa', 'Shield_Gas_Flow_L_min',
            'Material_Combination', 'Shield_Gas_Type', 'Joint_Type',
            'Sheet_Thickness_mm', 'Overlap_Distance_mm'
        ],
        'output_parameters': {
            'weld_morphology': [
                'Nugget_Width_mm', 'Penetration_Depth_mm', 'HAZ_Width_mm',
                'Has_Cracks', 'Has_Porosity', 'Has_Undercut', 'Has_Expulsion'
            ],
            'mechanical_properties': [
                'Tensile_Shear_Strength_N', 'Peel_Strength_N',
                'Contact_Resistance_uOhm'
            ],
            'extreme_temperature_performance': [
                'Thermal_Cycling_Strength_Degradation_pct',
                'Thermal_Cycling_Resistance_Increase_pct',
                'Cycles_to_Failure', 'Creep_Time_to_Failure_hours',
                'Static_Aging_Strength_Retention_pct',
                'Static_Aging_Resistance_Increase_pct',
                'IMC_Thickness_Post_Aging_um', 'Grain_Size_Change_pct',
                'Overall_Quality_Score'
            ]
        },
        'material_combinations': master_df['Material_Combination'].value_counts().to_dict(),
        'data_source_distribution': master_df['Data_Source'].value_counts().to_dict(),
        'summary_statistics': {
            'avg_tensile_strength_N': float(master_df['Tensile_Shear_Strength_N'].mean()),
            'avg_contact_resistance_uOhm': float(master_df['Contact_Resistance_uOhm'].mean()),
            'avg_cycles_to_failure': float(master_df['Cycles_to_Failure'].mean()),
            'avg_quality_score': float(master_df['Overall_Quality_Score'].mean()),
            'defect_rate_pct': float((master_df['Has_Cracks'] | master_df['Has_Porosity']).mean() * 100)
        }
    }
    
    stats_file = os.path.join(output_dir, 'dataset_metadata.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved metadata to: {stats_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("GENERATION COMPLETE!")
    print("="*80)
    print(f"Total Samples: {stats['total_samples']}")
    print(f"  - Experimental: {stats['experimental_samples']}")
    print(f"  - Computational: {stats['computational_samples']}")
    print(f"\nFeatures: {stats['n_features']}")
    print(f"Input Parameters: {len(stats['input_parameters'])}")
    print(f"Output Parameters: {sum(len(v) for v in stats['output_parameters'].values())}")
    print(f"\nAverage Quality Score: {stats['summary_statistics']['avg_quality_score']:.2f}")
    print(f"Average Cycles to Failure: {stats['summary_statistics']['avg_cycles_to_failure']:.0f}")
    print(f"Defect Rate: {stats['summary_statistics']['defect_rate_pct']:.1f}%")
    
    return master_df, experimental_df, computational_df, stats


if __name__ == '__main__':
    master_df, exp_df, comp_df, stats = generate_all_datasets()
