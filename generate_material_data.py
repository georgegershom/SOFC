#!/usr/bin/env python3
"""
Material Property Dataset Generator for SOFC Components
Generates synthetic but realistic material property data with variations
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime

class SOFCMaterialDataGenerator:
    """Generate synthetic material property data for SOFC components"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.materials = ['YSZ-8mol', 'YSZ-3mol', 'Ni', 'NiO', 'Ni-YSZ', 
                         'LSM', 'LSCF', 'LSC', 'GDC', 'Crofer22APU']
        
    def generate_elastic_properties(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate elastic property data with realistic variations"""
        
        data = []
        
        # Base properties (E in GPa, nu dimensionless)
        base_props = {
            'YSZ-8mol': {'E_mean': 210, 'E_std': 15, 'nu_mean': 0.31, 'nu_std': 0.02},
            'YSZ-3mol': {'E_mean': 205, 'E_std': 12, 'nu_mean': 0.30, 'nu_std': 0.02},
            'Ni': {'E_mean': 200, 'E_std': 10, 'nu_mean': 0.31, 'nu_std': 0.01},
            'NiO': {'E_mean': 210, 'E_std': 15, 'nu_mean': 0.29, 'nu_std': 0.02},
            'Ni-YSZ': {'E_mean': 110, 'E_std': 20, 'nu_mean': 0.30, 'nu_std': 0.02},
            'LSM': {'E_mean': 120, 'E_std': 15, 'nu_mean': 0.33, 'nu_std': 0.02},
            'LSCF': {'E_mean': 55, 'E_std': 10, 'nu_mean': 0.35, 'nu_std': 0.02},
            'LSC': {'E_mean': 45, 'E_std': 8, 'nu_mean': 0.36, 'nu_std': 0.02},
            'GDC': {'E_mean': 215, 'E_std': 20, 'nu_mean': 0.30, 'nu_std': 0.02},
            'Crofer22APU': {'E_mean': 220, 'E_std': 10, 'nu_mean': 0.28, 'nu_std': 0.01}
        }
        
        for material, props in base_props.items():
            for i in range(n_samples // len(base_props)):
                # Generate temperature
                temp = np.random.uniform(25, 1000)
                
                # Generate porosity
                if material in ['Ni-YSZ', 'LSM', 'LSCF', 'LSC']:
                    porosity = np.random.beta(2, 5) * 0.5  # 0-50% porosity, skewed to lower values
                else:
                    porosity = np.random.beta(2, 8) * 0.3  # 0-30% porosity for dense materials
                
                # Generate grain size (microns)
                grain_size = np.random.lognormal(0, 0.5) * 2
                
                # Processing method
                methods = ['sintered', 'tape_cast', 'screen_printed', 'plasma_sprayed', 'co_fired']
                processing = np.random.choice(methods)
                
                # Calculate temperature effect on modulus
                temp_factor = 1 - (temp - 25) / 2000  # Linear decrease with temperature
                
                # Calculate porosity effect (exponential model)
                porosity_factor = (1 - 1.9 * porosity + 0.9 * porosity**2)
                
                # Processing effect
                process_factors = {
                    'sintered': 1.0,
                    'tape_cast': 0.92,
                    'screen_printed': 0.88,
                    'plasma_sprayed': 0.85,
                    'co_fired': 0.95
                }
                process_factor = process_factors[processing]
                
                # Generate Young's modulus with all effects
                E = props['E_mean'] * temp_factor * porosity_factor * process_factor
                E += np.random.normal(0, props['E_std'])
                E = max(E, 10)  # Ensure positive
                
                # Generate Poisson's ratio (less sensitive to processing)
                nu = props['nu_mean'] + np.random.normal(0, props['nu_std'])
                nu = np.clip(nu, 0.15, 0.45)  # Physical bounds
                
                # Calculate derived properties
                G = E / (2 * (1 + nu))  # Shear modulus
                K = E / (3 * (1 - 2*nu))  # Bulk modulus
                
                data.append({
                    'sample_id': f"{material}_{i:04d}",
                    'material': material,
                    'temperature_C': temp,
                    'porosity': porosity,
                    'grain_size_um': grain_size,
                    'processing_method': processing,
                    'youngs_modulus_GPa': E,
                    'poissons_ratio': nu,
                    'shear_modulus_GPa': G,
                    'bulk_modulus_GPa': K,
                    'measurement_method': np.random.choice(['nanoindentation', 'RUS', 'impulse_excitation', 'four_point_bend']),
                    'test_atmosphere': np.random.choice(['air', 'H2_3H2O', 'N2', 'Ar'])
                })
        
        return pd.DataFrame(data)
    
    def generate_fracture_properties(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate fracture property data"""
        
        data = []
        
        # Base fracture properties
        base_props = {
            'YSZ-8mol': {'KIC_mean': 2.2, 'KIC_std': 0.3, 'GC_mean': 23, 'GC_std': 5},
            'YSZ-3mol': {'KIC_mean': 4.5, 'KIC_std': 0.5, 'GC_mean': 98, 'GC_std': 15},
            'Ni': {'KIC_mean': 80, 'KIC_std': 10, 'GC_mean': 32000, 'GC_std': 5000},
            'NiO': {'KIC_mean': 1.8, 'KIC_std': 0.3, 'GC_mean': 15, 'GC_std': 4},
            'Ni-YSZ': {'KIC_mean': 3.8, 'KIC_std': 0.8, 'GC_mean': 140, 'GC_std': 30},
            'LSM': {'KIC_mean': 1.5, 'KIC_std': 0.3, 'GC_mean': 19, 'GC_std': 5},
            'LSCF': {'KIC_mean': 1.0, 'KIC_std': 0.2, 'GC_mean': 18, 'GC_std': 4},
            'LSC': {'KIC_mean': 0.8, 'KIC_std': 0.15, 'GC_mean': 14, 'GC_std': 3},
            'GDC': {'KIC_mean': 1.4, 'KIC_std': 0.2, 'GC_mean': 9, 'GC_std': 2},
            'Crofer22APU': {'KIC_mean': 50, 'KIC_std': 8, 'GC_mean': 11400, 'GC_std': 2000}
        }
        
        for material, props in base_props.items():
            for i in range(n_samples // len(base_props)):
                temp = np.random.uniform(25, 1000)
                
                # Notch type and geometry
                notch_type = np.random.choice(['SENB', 'chevron', 'DCB', 'CT'])
                notch_depth_ratio = np.random.uniform(0.3, 0.5)
                
                # Loading rate effect
                loading_rate = 10**np.random.uniform(-3, 2)  # 0.001 to 100 MPa·m^0.5/s
                
                # Environment effect
                environment = np.random.choice(['dry_air', 'humid_air', 'H2_atmosphere', 'vacuum'])
                env_factors = {'dry_air': 1.0, 'humid_air': 0.9, 'H2_atmosphere': 0.85, 'vacuum': 1.05}
                
                # Temperature effect on toughness
                temp_factor = 1 - (temp - 25) / 3000
                
                # Generate fracture toughness
                KIC = props['KIC_mean'] * temp_factor * env_factors[environment]
                KIC += np.random.normal(0, props['KIC_std'])
                KIC = max(KIC, 0.1)
                
                # Generate critical energy release rate
                GC = props['GC_mean'] * temp_factor * env_factors[environment]
                GC += np.random.normal(0, props['GC_std'])
                GC = max(GC, 1)
                
                # R-curve behavior for some materials
                if material in ['YSZ-3mol', 'Ni-YSZ']:
                    crack_extension = np.random.uniform(0, 2)  # mm
                    R_curve_slope = np.random.uniform(0.5, 2)
                    KIC_R = KIC + R_curve_slope * np.sqrt(crack_extension)
                else:
                    crack_extension = 0
                    KIC_R = KIC
                
                # Weibull parameters for brittle materials
                if material not in ['Ni', 'Crofer22APU']:
                    weibull_modulus = np.random.uniform(8, 20)
                    characteristic_strength = np.random.normal(250, 50)
                else:
                    weibull_modulus = np.nan
                    characteristic_strength = np.nan
                
                data.append({
                    'sample_id': f"{material}_frac_{i:04d}",
                    'material': material,
                    'temperature_C': temp,
                    'fracture_toughness_MPam05': KIC,
                    'fracture_toughness_Rcurve_MPam05': KIC_R,
                    'critical_energy_release_rate_Jm2': GC,
                    'notch_type': notch_type,
                    'notch_depth_ratio': notch_depth_ratio,
                    'loading_rate_MPam05s': loading_rate,
                    'test_environment': environment,
                    'crack_extension_mm': crack_extension,
                    'weibull_modulus': weibull_modulus,
                    'characteristic_strength_MPa': characteristic_strength
                })
        
        return pd.DataFrame(data)
    
    def generate_interface_properties(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate interface fracture properties"""
        
        data = []
        
        interfaces = [
            ('Ni', 'YSZ', 1.2, 0.3),
            ('Ni-YSZ', 'YSZ', 1.8, 0.4),
            ('LSM', 'YSZ', 1.3, 0.35),
            ('LSCF', 'GDC', 1.8, 0.5),
            ('LSC', 'YSZ', 1.0, 0.25),
            ('GDC', 'YSZ', 2.2, 0.4)
        ]
        
        for mat1, mat2, base_toughness, base_std in interfaces:
            for i in range(n_samples // len(interfaces)):
                temp = np.random.uniform(25, 1000)
                
                # Interface processing
                processing = np.random.choice(['co_sintered', 'screen_printed', 'plasma_sprayed', 'sputtered'])
                process_factors = {
                    'co_sintered': 1.0,
                    'screen_printed': 0.8,
                    'plasma_sprayed': 0.6,
                    'sputtered': 0.9
                }
                
                # Thermal cycles effect
                thermal_cycles = np.random.randint(0, 1000)
                cycle_degradation = np.exp(-thermal_cycles / 500)  # Exponential degradation
                
                # Redox cycles for Ni-containing interfaces
                if 'Ni' in mat1 or 'Ni' in mat2:
                    redox_cycles = np.random.randint(0, 100)
                    redox_degradation = np.exp(-redox_cycles / 50)
                else:
                    redox_cycles = 0
                    redox_degradation = 1.0
                
                # Mode mixity
                mode_I_fraction = np.random.uniform(0.3, 1.0)
                mode_II_fraction = 1 - mode_I_fraction
                
                # Calculate interface toughness
                interface_toughness = base_toughness * process_factors[processing] * cycle_degradation * redox_degradation
                interface_toughness += np.random.normal(0, base_std)
                interface_toughness = max(interface_toughness, 0.1)
                
                # Adhesion energy
                adhesion_energy = (interface_toughness ** 2) * 1000 / 200  # Approximate relation
                adhesion_energy += np.random.normal(0, 0.5)
                
                # Interface roughness
                roughness_Ra = np.random.lognormal(-1, 0.5)  # microns
                
                # Residual stress
                CTE_mismatch = np.random.uniform(-3, 3)  # 10^-6/K
                residual_stress = CTE_mismatch * 200 * (800 - temp) / 1000  # MPa
                
                data.append({
                    'sample_id': f"{mat1}_{mat2}_interface_{i:04d}",
                    'material_1': mat1,
                    'material_2': mat2,
                    'temperature_C': temp,
                    'interface_toughness_MPam05': interface_toughness,
                    'adhesion_energy_Jm2': adhesion_energy,
                    'mode_I_fraction': mode_I_fraction,
                    'mode_II_fraction': mode_II_fraction,
                    'processing_method': processing,
                    'thermal_cycles': thermal_cycles,
                    'redox_cycles': redox_cycles,
                    'interface_roughness_Ra_um': roughness_Ra,
                    'residual_stress_MPa': residual_stress,
                    'CTE_mismatch_ppm_K': CTE_mismatch
                })
        
        return pd.DataFrame(data)
    
    def generate_thermal_properties(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate thermal expansion data"""
        
        data = []
        
        # Base CTE values (10^-6/K)
        base_cte = {
            'YSZ-8mol': {'mean': 10.5, 'std': 0.3},
            'YSZ-3mol': {'mean': 10.0, 'std': 0.3},
            'Ni': {'mean': 13.3, 'std': 0.5},
            'NiO': {'mean': 14.1, 'std': 0.4},
            'Ni-YSZ': {'mean': 12.0, 'std': 0.6},
            'LSM': {'mean': 11.8, 'std': 0.4},
            'LSCF': {'mean': 15.4, 'std': 0.8},
            'LSC': {'mean': 20.5, 'std': 1.0},
            'GDC': {'mean': 12.5, 'std': 0.4},
            'Crofer22APU': {'mean': 12.3, 'std': 0.3}
        }
        
        for material, props in base_cte.items():
            for i in range(n_samples // len(base_cte)):
                # Temperature range
                T_start = np.random.uniform(25, 400)
                T_end = T_start + np.random.uniform(200, 600)
                T_mean = (T_start + T_end) / 2
                
                # CTE increases with temperature for most materials
                temp_factor = 1 + (T_mean - 500) / 5000
                
                # Anisotropy for non-cubic materials
                if material in ['YSZ-3mol', 'LSM']:
                    anisotropy_factor = np.random.uniform(0.95, 1.05)
                    direction = np.random.choice(['a-axis', 'c-axis', 'polycrystal'])
                else:
                    anisotropy_factor = 1.0
                    direction = 'isotropic'
                
                # Oxygen nonstoichiometry effect for perovskites
                if material in ['LSCF', 'LSC', 'GDC']:
                    pO2 = 10**np.random.uniform(-20, 0)  # atm
                    nonstoich_factor = 1 + 0.1 * np.log10(pO2/0.21)
                else:
                    pO2 = 0.21
                    nonstoich_factor = 1.0
                
                # Generate CTE
                CTE = props['mean'] * temp_factor * anisotropy_factor * nonstoich_factor
                CTE += np.random.normal(0, props['std'])
                
                # Thermal conductivity (W/m·K)
                thermal_cond_base = {
                    'YSZ-8mol': 2.7, 'YSZ-3mol': 2.5, 'Ni': 90, 'NiO': 12,
                    'Ni-YSZ': 6, 'LSM': 2, 'LSCF': 1.5, 'LSC': 1.2,
                    'GDC': 2.8, 'Crofer22APU': 26
                }
                k = thermal_cond_base[material] * np.exp(-(T_mean - 25) / 1000)
                k *= np.random.uniform(0.8, 1.2)
                
                # Specific heat (J/g·K)
                cp = np.random.uniform(0.4, 0.8) if material not in ['Ni', 'Crofer22APU'] else np.random.uniform(0.4, 0.5)
                
                data.append({
                    'sample_id': f"{material}_thermal_{i:04d}",
                    'material': material,
                    'T_start_C': T_start,
                    'T_end_C': T_end,
                    'T_mean_C': T_mean,
                    'CTE_ppm_K': CTE,
                    'measurement_direction': direction,
                    'pO2_atm': pO2,
                    'thermal_conductivity_W_mK': k,
                    'specific_heat_J_gK': cp,
                    'measurement_method': np.random.choice(['dilatometry', 'TMA', 'XRD_high_temp']),
                    'heating_rate_K_min': np.random.choice([2, 5, 10])
                })
        
        return pd.DataFrame(data)
    
    def generate_chemical_expansion(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate chemical expansion data"""
        
        data = []
        
        # Materials with significant chemical expansion
        chem_exp_materials = {
            'Ni': {'oxidation_strain': 0.207, 'mechanism': 'Ni->NiO'},
            'NiO': {'reduction_strain': -0.171, 'mechanism': 'NiO->Ni'},
            'LSCF': {'coefficient': 0.032, 'mechanism': 'oxygen_vacancy'},
            'LSC': {'coefficient': 0.045, 'mechanism': 'oxygen_vacancy'},
            'GDC': {'coefficient': 0.088, 'mechanism': 'Ce4+->Ce3+'},
            'Ni-YSZ': {'redox_strain': 0.002, 'mechanism': 'Ni_redox'}
        }
        
        for material, props in chem_exp_materials.items():
            for i in range(n_samples // len(chem_exp_materials)):
                temp = np.random.uniform(600, 1000)
                
                if 'Ni' in material and material != 'Ni-YSZ':
                    # Ni/NiO redox
                    pO2_initial = 10**np.random.uniform(-20, -5) if material == 'Ni' else 0.21
                    pO2_final = 0.21 if material == 'Ni' else 10**np.random.uniform(-20, -5)
                    
                    # Kinetics
                    time_hrs = np.random.uniform(0.1, 100)
                    conversion = 1 - np.exp(-time_hrs / 10)  # Simple kinetic model
                    
                    if material == 'Ni':
                        strain = props['oxidation_strain'] * conversion
                        volume_change = 0.695 * conversion
                    else:
                        strain = props['reduction_strain'] * conversion
                        volume_change = -0.41 * conversion
                    
                    delta_nonstoich = np.nan
                    
                elif material in ['LSCF', 'LSC', 'GDC']:
                    # Oxygen nonstoichiometry
                    pO2_initial = 0.21
                    pO2_final = 10**np.random.uniform(-20, 0)
                    
                    # Calculate nonstoichiometry change
                    delta_nonstoich = props['coefficient'] * np.log10(pO2_initial/pO2_final) / 20
                    strain = props['coefficient'] * delta_nonstoich
                    volume_change = strain * 3  # Isotropic expansion
                    
                    time_hrs = np.random.uniform(0.01, 10)
                    conversion = 1 - np.exp(-time_hrs / 0.5)  # Faster kinetics
                    
                else:  # Ni-YSZ
                    # Redox cycling
                    cycle_number = np.random.randint(1, 200)
                    strain = props['redox_strain'] * np.log(cycle_number + 1)
                    volume_change = strain * 3
                    pO2_initial = 0.21
                    pO2_final = 10**np.random.uniform(-20, -5)
                    time_hrs = np.random.uniform(1, 20)
                    conversion = 1.0
                    delta_nonstoich = np.nan
                
                # Microstructure effect
                grain_size = np.random.lognormal(0, 0.5) * 2
                porosity = np.random.beta(2, 5) * 0.4
                
                # Stress generation
                constrained_stress = strain * 100 * np.random.uniform(0.5, 1.0)  # GPa to MPa
                
                data.append({
                    'sample_id': f"{material}_chemexp_{i:04d}",
                    'material': material,
                    'temperature_C': temp,
                    'pO2_initial_atm': pO2_initial,
                    'pO2_final_atm': pO2_final,
                    'linear_strain': strain,
                    'volume_change': volume_change,
                    'delta_nonstoichiometry': delta_nonstoich,
                    'time_hours': time_hrs,
                    'conversion_fraction': conversion,
                    'grain_size_um': grain_size,
                    'porosity': porosity,
                    'constrained_stress_MPa': constrained_stress,
                    'mechanism': props.get('mechanism', 'unknown')
                })
        
        return pd.DataFrame(data)
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate all datasets"""
        
        print("Generating elastic properties dataset...")
        elastic_df = self.generate_elastic_properties(1000)
        
        print("Generating fracture properties dataset...")
        fracture_df = self.generate_fracture_properties(1000)
        
        print("Generating interface properties dataset...")
        interface_df = self.generate_interface_properties(500)
        
        print("Generating thermal properties dataset...")
        thermal_df = self.generate_thermal_properties(1000)
        
        print("Generating chemical expansion dataset...")
        chemexp_df = self.generate_chemical_expansion(500)
        
        return {
            'elastic_properties': elastic_df,
            'fracture_properties': fracture_df,
            'interface_properties': interface_df,
            'thermal_properties': thermal_df,
            'chemical_expansion': chemexp_df
        }
    
    def add_noise_and_outliers(self, df: pd.DataFrame, noise_level: float = 0.05, 
                               outlier_fraction: float = 0.02) -> pd.DataFrame:
        """Add realistic noise and outliers to numeric columns"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if 'id' not in col.lower():
                # Add Gaussian noise
                noise = np.random.normal(0, df[col].std() * noise_level, len(df))
                df[col] = df[col] + noise
                
                # Add outliers
                n_outliers = int(len(df) * outlier_fraction)
                outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
                outlier_factor = np.random.choice([-1, 1], n_outliers) * np.random.uniform(2, 4, n_outliers)
                df.loc[outlier_indices, col] = df.loc[outlier_indices, col] * outlier_factor
        
        return df
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], format: str = 'both'):
        """Save datasets to CSV and/or Excel formats"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for name, df in datasets.items():
            if format in ['csv', 'both']:
                filename = f"sofc_material_{name}_{timestamp}.csv"
                df.to_csv(filename, index=False)
                print(f"Saved {filename}")
            
            if format in ['excel', 'both']:
                filename = f"sofc_material_{name}_{timestamp}.xlsx"
                df.to_excel(filename, index=False, engine='openpyxl')
                print(f"Saved {filename}")
    
    def generate_summary_statistics(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate summary statistics for all datasets"""
        
        summary_data = []
        
        for name, df in datasets.items():
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if 'id' not in col.lower():
                    summary_data.append({
                        'dataset': name,
                        'property': col,
                        'count': len(df[col].dropna()),
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        '25%': df[col].quantile(0.25),
                        'median': df[col].median(),
                        '75%': df[col].quantile(0.75),
                        'max': df[col].max(),
                        'skewness': df[col].skew(),
                        'kurtosis': df[col].kurtosis()
                    })
        
        return pd.DataFrame(summary_data)


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("SOFC Material Property Dataset Generator")
    print("=" * 80)
    
    # Initialize generator
    generator = SOFCMaterialDataGenerator(seed=42)
    
    # Generate all datasets
    datasets = generator.generate_complete_dataset()
    
    # Add realistic noise and outliers
    print("\nAdding measurement noise and outliers...")
    for name in datasets:
        datasets[name] = generator.add_noise_and_outliers(datasets[name])
    
    # Save datasets
    print("\nSaving datasets...")
    generator.save_datasets(datasets, format='csv')
    
    # Generate and save summary statistics
    print("\nGenerating summary statistics...")
    summary_df = generator.generate_summary_statistics(datasets)
    summary_df.to_csv("sofc_material_summary_statistics.csv", index=False)
    print("Saved summary statistics")
    
    # Print dataset info
    print("\n" + "=" * 80)
    print("Dataset Summary:")
    print("=" * 80)
    
    total_samples = 0
    for name, df in datasets.items():
        n_samples = len(df)
        n_features = len(df.columns)
        total_samples += n_samples
        print(f"\n{name}:")
        print(f"  - Samples: {n_samples}")
        print(f"  - Features: {n_features}")
        print(f"  - Columns: {', '.join(df.columns[:5])}...")
    
    print(f"\nTotal samples generated: {total_samples}")
    print("\nDataset generation complete!")
    
    return datasets


if __name__ == "__main__":
    datasets = main()