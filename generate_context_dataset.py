"""
Context Dataset Generator for Residual Stress Prediction in Multi-layer Ceramic Structures
This script generates a comprehensive DOE (Design of Experiments) dataset including:
- Geometric parameters
- Temperature-dependent material properties
- Process parameters (sintering profiles)
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
import json
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)


class ContextDatasetGenerator:
    """Generate comprehensive context dataset for residual stress prediction"""
    
    def __init__(self, n_samples=1000):
        self.n_samples = n_samples
        self.samples = []
        
        # Define parameter ranges based on typical SOFC and ceramic materials
        self.param_ranges = {
            # Geometric Parameters
            'plate_length_mm': (50.0, 150.0),
            'plate_width_mm': (50.0, 150.0),
            'anode_thickness_um': (300.0, 1000.0),
            'electrolyte_thickness_um': (5.0, 50.0),
            'cathode_thickness_um': (20.0, 100.0),
            'green_density_fraction': (0.45, 0.65),  # Fraction of theoretical density
            'green_shrinkage_factor': (1.15, 1.35),  # Green dimensions multiplier
            
            # Material Properties - Anode (Ni-YSZ)
            'anode_youngs_modulus_25C_GPa': (40.0, 80.0),
            'anode_youngs_modulus_1000C_GPa': (20.0, 50.0),
            'anode_CTE_25C_ppm_K': (10.5, 13.5),
            'anode_CTE_1000C_ppm_K': (12.0, 15.0),
            'anode_poisson_ratio': (0.25, 0.35),
            'anode_sintering_onset_C': (1100.0, 1250.0),
            'anode_max_shrinkage_rate_per_min': (0.001, 0.01),
            'anode_total_shrinkage_fraction': (0.15, 0.25),
            'anode_creep_activation_energy_kJ_mol': (300.0, 450.0),
            'anode_creep_stress_exponent': (1.0, 3.0),
            
            # Material Properties - Electrolyte (YSZ/GDC)
            'electrolyte_youngs_modulus_25C_GPa': (180.0, 220.0),
            'electrolyte_youngs_modulus_1000C_GPa': (120.0, 160.0),
            'electrolyte_CTE_25C_ppm_K': (9.5, 11.5),
            'electrolyte_CTE_1000C_ppm_K': (10.5, 12.5),
            'electrolyte_poisson_ratio': (0.28, 0.33),
            'electrolyte_sintering_onset_C': (1200.0, 1350.0),
            'electrolyte_max_shrinkage_rate_per_min': (0.0005, 0.005),
            'electrolyte_total_shrinkage_fraction': (0.12, 0.20),
            'electrolyte_creep_activation_energy_kJ_mol': (450.0, 600.0),
            'electrolyte_creep_stress_exponent': (1.0, 2.5),
            
            # Material Properties - Cathode (LSM/LSCF)
            'cathode_youngs_modulus_25C_GPa': (50.0, 100.0),
            'cathode_youngs_modulus_1000C_GPa': (30.0, 70.0),
            'cathode_CTE_25C_ppm_K': (11.0, 14.0),
            'cathode_CTE_1000C_ppm_K': (12.5, 16.0),
            'cathode_poisson_ratio': (0.28, 0.35),
            'cathode_sintering_onset_C': (1000.0, 1200.0),
            'cathode_max_shrinkage_rate_per_min': (0.001, 0.008),
            'cathode_total_shrinkage_fraction': (0.10, 0.22),
            'cathode_creep_activation_energy_kJ_mol': (280.0, 400.0),
            'cathode_creep_stress_exponent': (1.5, 3.5),
            
            # Process Parameters - Sintering Profile
            'heating_ramp_rate_C_per_min': (1.0, 10.0),
            'sintering_peak_temp_C': (1300.0, 1500.0),
            'sintering_hold_time_min': (60.0, 300.0),
            'cooling_ramp_rate_C_per_min': (1.0, 8.0),
            'intermediate_hold_temp_C': (800.0, 1100.0),
            'intermediate_hold_time_min': (0.0, 120.0),
            
            # Atmosphere parameters
            'oxygen_partial_pressure_atm': (0.001, 0.21),
            'humidity_percent': (0.0, 5.0),
        }
        
        self.atmosphere_types = ['Air', 'Argon', 'Nitrogen', 'Reducing (H2/N2)', 'Vacuum']
    
    def generate_lhs_samples(self):
        """Generate Latin Hypercube Samples for better space-filling"""
        param_names = list(self.param_ranges.keys())
        n_params = len(param_names)
        
        # Create LHS sampler
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        lhs_samples = sampler.random(n=self.n_samples)
        
        # Scale samples to parameter ranges
        samples_dict = {}
        for i, param_name in enumerate(param_names):
            lower, upper = self.param_ranges[param_name]
            samples_dict[param_name] = lhs_samples[:, i] * (upper - lower) + lower
        
        return pd.DataFrame(samples_dict)
    
    def add_atmosphere_types(self, df):
        """Add categorical atmosphere types"""
        # Randomly assign atmosphere types
        df['atmosphere_type'] = np.random.choice(
            self.atmosphere_types, 
            size=len(df),
            p=[0.5, 0.15, 0.10, 0.20, 0.05]  # Weighted probabilities
        )
        return df
    
    def calculate_derived_parameters(self, df):
        """Calculate derived parameters and consistency checks"""
        # Total green thickness
        df['total_green_thickness_um'] = (
            df['anode_thickness_um'] * df['green_shrinkage_factor'] +
            df['electrolyte_thickness_um'] * df['green_shrinkage_factor'] +
            df['cathode_thickness_um'] * df['green_shrinkage_factor']
        )
        
        # Total sintered thickness
        df['total_sintered_thickness_um'] = (
            df['anode_thickness_um'] +
            df['electrolyte_thickness_um'] +
            df['cathode_thickness_um']
        )
        
        # CTE mismatch parameters (critical for residual stress)
        df['CTE_mismatch_anode_electrolyte_25C'] = (
            df['anode_CTE_25C_ppm_K'] - df['electrolyte_CTE_25C_ppm_K']
        )
        df['CTE_mismatch_cathode_electrolyte_25C'] = (
            df['cathode_CTE_25C_ppm_K'] - df['electrolyte_CTE_25C_ppm_K']
        )
        df['CTE_mismatch_anode_cathode_25C'] = (
            df['anode_CTE_25C_ppm_K'] - df['cathode_CTE_25C_ppm_K']
        )
        
        # Average CTE mismatch magnitude
        df['avg_CTE_mismatch_magnitude'] = (
            np.abs(df['CTE_mismatch_anode_electrolyte_25C']) +
            np.abs(df['CTE_mismatch_cathode_electrolyte_25C'])
        ) / 2.0
        
        # Stiffness ratios (important for stress distribution)
        df['stiffness_ratio_electrolyte_anode'] = (
            df['electrolyte_youngs_modulus_25C_GPa'] / df['anode_youngs_modulus_25C_GPa']
        )
        df['stiffness_ratio_electrolyte_cathode'] = (
            df['electrolyte_youngs_modulus_25C_GPa'] / df['cathode_youngs_modulus_25C_GPa']
        )
        
        # Thickness ratios
        df['thickness_ratio_anode_electrolyte'] = (
            df['anode_thickness_um'] / df['electrolyte_thickness_um']
        )
        df['thickness_ratio_cathode_electrolyte'] = (
            df['cathode_thickness_um'] / df['electrolyte_thickness_um']
        )
        
        # Total sintering time
        df['total_process_time_min'] = (
            df['sintering_peak_temp_C'] / df['heating_ramp_rate_C_per_min'] +
            df['sintering_hold_time_min'] +
            df['intermediate_hold_time_min'] +
            df['sintering_peak_temp_C'] / df['cooling_ramp_rate_C_per_min']
        )
        
        # Cooling/heating rate ratio
        df['cooling_heating_rate_ratio'] = (
            df['cooling_ramp_rate_C_per_min'] / df['heating_ramp_rate_C_per_min']
        )
        
        return df
    
    def generate_temperature_profile(self, row):
        """Generate detailed temperature-time profile for sintering"""
        profile = {
            'time_points_min': [],
            'temperature_C': [],
            'segment_names': []
        }
        
        time = 0.0
        temp = 25.0  # Start at room temperature
        
        # Initial room temperature
        profile['time_points_min'].append(time)
        profile['temperature_C'].append(temp)
        profile['segment_names'].append('Initial')
        
        # Heating ramp to intermediate hold (if exists)
        if row['intermediate_hold_time_min'] > 0:
            time += (row['intermediate_hold_temp_C'] - 25.0) / row['heating_ramp_rate_C_per_min']
            profile['time_points_min'].append(time)
            profile['temperature_C'].append(row['intermediate_hold_temp_C'])
            profile['segment_names'].append('Heating_to_intermediate')
            
            # Intermediate hold
            time += row['intermediate_hold_time_min']
            profile['time_points_min'].append(time)
            profile['temperature_C'].append(row['intermediate_hold_temp_C'])
            profile['segment_names'].append('Intermediate_hold')
            
            # Continue heating to peak
            time += (row['sintering_peak_temp_C'] - row['intermediate_hold_temp_C']) / row['heating_ramp_rate_C_per_min']
            profile['time_points_min'].append(time)
            profile['temperature_C'].append(row['sintering_peak_temp_C'])
            profile['segment_names'].append('Heating_to_peak')
        else:
            # Direct heating to peak
            time += (row['sintering_peak_temp_C'] - 25.0) / row['heating_ramp_rate_C_per_min']
            profile['time_points_min'].append(time)
            profile['temperature_C'].append(row['sintering_peak_temp_C'])
            profile['segment_names'].append('Heating_to_peak')
        
        # Peak hold
        time += row['sintering_hold_time_min']
        profile['time_points_min'].append(time)
        profile['temperature_C'].append(row['sintering_peak_temp_C'])
        profile['segment_names'].append('Peak_hold')
        
        # Cooling to room temperature
        time += (row['sintering_peak_temp_C'] - 25.0) / row['cooling_ramp_rate_C_per_min']
        profile['time_points_min'].append(time)
        profile['temperature_C'].append(25.0)
        profile['segment_names'].append('Cooling')
        
        return json.dumps(profile)
    
    def add_temperature_profiles(self, df):
        """Add temperature profile for each sample"""
        df['temperature_profile_json'] = df.apply(
            lambda row: self.generate_temperature_profile(row),
            axis=1
        )
        return df
    
    def add_sample_ids(self, df):
        """Add unique sample identifiers"""
        df.insert(0, 'sample_id', [f'SAMPLE_{i:06d}' for i in range(len(df))])
        df.insert(1, 'generation_timestamp', datetime.now().isoformat())
        return df
    
    def add_quality_flags(self, df):
        """Add flags for potentially problematic parameter combinations"""
        # Flag extreme CTE mismatches
        df['flag_extreme_CTE_mismatch'] = (
            df['avg_CTE_mismatch_magnitude'] > 3.5
        ).astype(int)
        
        # Flag very thin electrolytes
        df['flag_thin_electrolyte'] = (
            df['electrolyte_thickness_um'] < 10.0
        ).astype(int)
        
        # Flag very fast cooling rates
        df['flag_fast_cooling'] = (
            df['cooling_ramp_rate_C_per_min'] > 6.0
        ).astype(int)
        
        # Flag asymmetric structures
        df['flag_asymmetric_structure'] = (
            np.abs(df['thickness_ratio_anode_electrolyte'] - 
                   df['thickness_ratio_cathode_electrolyte']) > 15.0
        ).astype(int)
        
        return df
    
    def generate(self):
        """Main generation method"""
        print("Generating Latin Hypercube Samples...")
        df = self.generate_lhs_samples()
        
        print("Adding atmosphere types...")
        df = self.add_atmosphere_types(df)
        
        print("Calculating derived parameters...")
        df = self.calculate_derived_parameters(df)
        
        print("Generating temperature profiles...")
        df = self.add_temperature_profiles(df)
        
        print("Adding sample IDs and quality flags...")
        df = self.add_sample_ids(df)
        df = self.add_quality_flags(df)
        
        return df
    
    def save_dataset(self, df, base_filename='context_dataset'):
        """Save dataset in multiple formats with metadata"""
        
        # Save main CSV
        csv_filename = f'{base_filename}.csv'
        df.to_csv(csv_filename, index=False)
        print(f"Saved: {csv_filename}")
        
        # Save Excel with multiple sheets
        excel_filename = f'{base_filename}.xlsx'
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Full_Dataset', index=False)
            
            # Summary statistics
            summary = df.describe()
            summary.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Parameter ranges
            ranges_df = pd.DataFrame([
                {'Parameter': k, 'Min': v[0], 'Max': v[1]}
                for k, v in self.param_ranges.items()
            ])
            ranges_df.to_excel(writer, sheet_name='Parameter_Ranges', index=False)
            
        print(f"Saved: {excel_filename}")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'n_samples': self.n_samples,
            'n_parameters': len(df.columns),
            'parameter_ranges': self.param_ranges,
            'atmosphere_types': self.atmosphere_types,
            'sampling_method': 'Latin Hypercube Sampling',
            'description': 'Context dataset for residual stress prediction in multi-layer ceramic structures',
            'column_descriptions': self.get_column_descriptions()
        }
        
        metadata_filename = f'{base_filename}_metadata.json'
        with open(metadata_filename, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"Saved: {metadata_filename}")
        
        # Generate data dictionary
        self.generate_data_dictionary(df, f'{base_filename}_data_dictionary.txt')
        
        return csv_filename, excel_filename, metadata_filename
    
    def get_column_descriptions(self):
        """Get descriptions for all columns"""
        descriptions = {
            # Geometric
            'plate_length_mm': 'Length of ceramic plate in millimeters',
            'plate_width_mm': 'Width of ceramic plate in millimeters',
            'anode_thickness_um': 'Final sintered anode thickness in micrometers',
            'electrolyte_thickness_um': 'Final sintered electrolyte thickness in micrometers',
            'cathode_thickness_um': 'Final sintered cathode thickness in micrometers',
            'green_density_fraction': 'Green body density as fraction of theoretical density',
            'green_shrinkage_factor': 'Multiplier for green dimensions relative to sintered',
            
            # Material properties descriptions
            'anode_youngs_modulus_25C_GPa': "Anode Young's modulus at 25°C in GPa",
            'anode_youngs_modulus_1000C_GPa': "Anode Young's modulus at 1000°C in GPa",
            'anode_CTE_25C_ppm_K': 'Anode coefficient of thermal expansion at 25°C in ppm/K',
            'anode_CTE_1000C_ppm_K': 'Anode coefficient of thermal expansion at 1000°C in ppm/K',
            
            # Process parameters
            'heating_ramp_rate_C_per_min': 'Heating rate during sintering in °C/min',
            'sintering_peak_temp_C': 'Maximum sintering temperature in °C',
            'sintering_hold_time_min': 'Time held at peak temperature in minutes',
            'cooling_ramp_rate_C_per_min': 'Cooling rate after sintering in °C/min',
            
            # Derived parameters
            'CTE_mismatch_anode_electrolyte_25C': 'CTE difference between anode and electrolyte at 25°C',
            'avg_CTE_mismatch_magnitude': 'Average absolute CTE mismatch magnitude',
            'stiffness_ratio_electrolyte_anode': 'Ratio of electrolyte to anode stiffness',
            'thickness_ratio_anode_electrolyte': 'Ratio of anode to electrolyte thickness',
        }
        return descriptions
    
    def generate_data_dictionary(self, df, filename):
        """Generate human-readable data dictionary"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CONTEXT DATASET - DATA DICTIONARY\n")
            f.write("Residual Stress Prediction in Multi-layer Ceramic Structures\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of Samples: {len(df)}\n")
            f.write(f"Number of Features: {len(df.columns)}\n")
            f.write(f"Sampling Method: Latin Hypercube Sampling (LHS)\n\n")
            
            f.write("="*80 + "\n")
            f.write("PARAMETER CATEGORIES\n")
            f.write("="*80 + "\n\n")
            
            categories = {
                'Geometric Parameters': [
                    'plate_length_mm', 'plate_width_mm', 'anode_thickness_um',
                    'electrolyte_thickness_um', 'cathode_thickness_um',
                    'green_density_fraction', 'green_shrinkage_factor'
                ],
                'Anode Material Properties': [
                    col for col in df.columns if col.startswith('anode_')
                ],
                'Electrolyte Material Properties': [
                    col for col in df.columns if col.startswith('electrolyte_')
                ],
                'Cathode Material Properties': [
                    col for col in df.columns if col.startswith('cathode_')
                ],
                'Process Parameters': [
                    'heating_ramp_rate_C_per_min', 'sintering_peak_temp_C',
                    'sintering_hold_time_min', 'cooling_ramp_rate_C_per_min',
                    'intermediate_hold_temp_C', 'intermediate_hold_time_min',
                    'oxygen_partial_pressure_atm', 'humidity_percent',
                    'atmosphere_type', 'temperature_profile_json'
                ],
                'Derived Parameters': [
                    col for col in df.columns if col.startswith('CTE_mismatch') or
                    col.startswith('stiffness_ratio') or col.startswith('thickness_ratio') or
                    col.startswith('total_') or col.startswith('avg_') or
                    col.startswith('cooling_heating')
                ],
                'Quality Flags': [
                    col for col in df.columns if col.startswith('flag_')
                ]
            }
            
            for category, columns in categories.items():
                f.write(f"\n{category}:\n")
                f.write("-" * 80 + "\n")
                for col in columns:
                    if col in df.columns:
                        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                            f.write(f"  {col}:\n")
                            f.write(f"    Range: [{df[col].min():.4f}, {df[col].max():.4f}]\n")
                            f.write(f"    Mean: {df[col].mean():.4f}, Std: {df[col].std():.4f}\n")
                        else:
                            f.write(f"  {col}: (categorical/text)\n")
                            if col == 'atmosphere_type':
                                f.write(f"    Values: {', '.join(df[col].unique())}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("KEY RELATIONSHIPS FOR RESIDUAL STRESS\n")
            f.write("="*80 + "\n\n")
            f.write("1. CTE Mismatch: Primary driver of residual stress during cooling\n")
            f.write("2. Stiffness Ratios: Determines stress distribution between layers\n")
            f.write("3. Thickness Ratios: Affects constraint and bending moments\n")
            f.write("4. Sintering Profiles: Temperature history affects microstructure and stress\n")
            f.write("5. Creep Parameters: High-temperature stress relaxation\n\n")
            
        print(f"Saved: {filename}")


def main():
    """Main execution function"""
    print("="*80)
    print("CONTEXT DATASET GENERATOR")
    print("Residual Stress Prediction in Multi-layer Ceramic Structures")
    print("="*80)
    print()
    
    # Generate dataset with different sizes
    sizes = {
        'small': 100,
        'medium': 1000,
        'large': 5000,
        'xlarge': 10000
    }
    
    for size_name, n_samples in sizes.items():
        print(f"\n{'='*80}")
        print(f"Generating {size_name.upper()} dataset ({n_samples} samples)...")
        print(f"{'='*80}\n")
        
        generator = ContextDatasetGenerator(n_samples=n_samples)
        df = generator.generate()
        
        print(f"\nDataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Save dataset
        base_filename = f'context_dataset_{size_name}'
        generator.save_dataset(df, base_filename)
        
        print(f"\n{size_name.upper()} dataset generation complete!")
        print(f"Files generated:")
        print(f"  - {base_filename}.csv")
        print(f"  - {base_filename}.xlsx")
        print(f"  - {base_filename}_metadata.json")
        print(f"  - {base_filename}_data_dictionary.txt")
    
    print("\n" + "="*80)
    print("ALL DATASETS GENERATED SUCCESSFULLY!")
    print("="*80)
    
    # Generate a quick summary report
    print("\n\nQUICK SUMMARY OF XLARGE DATASET:")
    print("="*80)
    generator = ContextDatasetGenerator(n_samples=10000)
    df = generator.generate()
    
    print("\nKey Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"\n  CTE Mismatch Range: [{df['avg_CTE_mismatch_magnitude'].min():.3f}, {df['avg_CTE_mismatch_magnitude'].max():.3f}] ppm/K")
    print(f"  Sintering Temp Range: [{df['sintering_peak_temp_C'].min():.1f}, {df['sintering_peak_temp_C'].max():.1f}] °C")
    print(f"  Total Thickness Range: [{df['total_sintered_thickness_um'].min():.1f}, {df['total_sintered_thickness_um'].max():.1f}] μm")
    print(f"\n  Atmosphere Distribution:")
    for atm, count in df['atmosphere_type'].value_counts().items():
        print(f"    {atm}: {count} ({count/len(df)*100:.1f}%)")
    
    print("\n  Quality Flags:")
    print(f"    Extreme CTE mismatch: {df['flag_extreme_CTE_mismatch'].sum()} samples")
    print(f"    Thin electrolyte: {df['flag_thin_electrolyte'].sum()} samples")
    print(f"    Fast cooling: {df['flag_fast_cooling'].sum()} samples")
    print(f"    Asymmetric structure: {df['flag_asymmetric_structure'].sum()} samples")
    
    print("\n" + "="*80)
    print("Dataset generation complete! Ready for ML model training.")
    print("="*80)


if __name__ == "__main__":
    main()
