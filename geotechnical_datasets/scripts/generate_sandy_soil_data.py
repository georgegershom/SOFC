#!/usr/bin/env python3
"""
Generate comprehensive sandy soil properties dataset for geotechnical research
Focus: Underground structure failure mechanisms in sandy soils
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_sandy_soil_properties(n_samples=500):
    """Generate sandy soil basic properties dataset"""
    
    # Sand content (% by weight) - typically 60-95% for sandy soils
    sand_content = np.random.normal(78, 12, n_samples)
    sand_content = np.clip(sand_content, 60, 95)
    
    # Silt content (complementary to sand and clay)
    silt_content = np.random.normal(15, 8, n_samples)
    silt_content = np.clip(silt_content, 3, 30)
    
    # Clay content (remaining percentage)
    clay_content = 100 - sand_content - silt_content
    clay_content = np.clip(clay_content, 2, 20)
    
    # Normalize to ensure sum = 100%
    total = sand_content + silt_content + clay_content
    sand_content = sand_content / total * 100
    silt_content = silt_content / total * 100
    clay_content = clay_content / total * 100
    
    # Grain size distribution parameters
    d10 = np.random.lognormal(np.log(0.12), 0.4, n_samples)  # mm
    d30 = d10 * np.random.uniform(2.5, 4.5, n_samples)
    d60 = d30 * np.random.uniform(1.8, 3.2, n_samples)
    
    # Uniformity coefficient and coefficient of curvature
    cu = d60 / d10
    cc = (d30**2) / (d10 * d60)
    
    # Relative density (%)
    relative_density = np.random.normal(65, 20, n_samples)
    relative_density = np.clip(relative_density, 15, 95)
    
    # Void ratio (correlated with relative density)
    e_max = np.random.normal(0.85, 0.15, n_samples)
    e_min = np.random.normal(0.45, 0.10, n_samples)
    void_ratio = e_max - (relative_density/100) * (e_max - e_min)
    
    # Dry unit weight (kN/m³)
    specific_gravity = np.random.normal(2.65, 0.05, n_samples)
    dry_unit_weight = (specific_gravity * 9.81) / (1 + void_ratio)
    
    # Porosity
    porosity = void_ratio / (1 + void_ratio)
    
    # Permeability (m/s) - log-normal distribution
    permeability = np.random.lognormal(np.log(1e-4), 1.2, n_samples)
    
    # Sample locations (synthetic coordinates)
    latitude = np.random.uniform(35.0, 45.0, n_samples)
    longitude = np.random.uniform(-120.0, -100.0, n_samples)
    depth = np.random.uniform(1.0, 30.0, n_samples)
    
    # Sample IDs and dates
    sample_ids = [f"SAND_{i+1:04d}" for i in range(n_samples)]
    base_date = datetime(2020, 1, 1)
    sample_dates = [base_date + timedelta(days=np.random.randint(0, 1460)) for _ in range(n_samples)]
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'sample_date': sample_dates,
        'latitude': latitude,
        'longitude': longitude,
        'depth_m': depth,
        'sand_content_pct': sand_content,
        'silt_content_pct': silt_content,
        'clay_content_pct': clay_content,
        'd10_mm': d10,
        'd30_mm': d30,
        'd60_mm': d60,
        'uniformity_coefficient': cu,
        'coefficient_curvature': cc,
        'relative_density_pct': relative_density,
        'void_ratio': void_ratio,
        'porosity': porosity,
        'specific_gravity': specific_gravity,
        'dry_unit_weight_kN_m3': dry_unit_weight,
        'permeability_m_s': permeability
    })

def generate_sandy_soil_mechanical_properties(basic_props_df):
    """Generate mechanical properties correlated with basic properties"""
    
    n_samples = len(basic_props_df)
    
    # Friction angle (degrees) - correlated with relative density and grain size
    friction_angle_base = 28 + 0.15 * basic_props_df['relative_density_pct']
    friction_angle_base += np.log10(basic_props_df['d50_mm']) * 2  # d50 approximated as geometric mean
    friction_angle = friction_angle_base + np.random.normal(0, 2, n_samples)
    friction_angle = np.clip(friction_angle, 25, 45)
    
    # Cohesion (kPa) - typically low for sandy soils
    cohesion = np.random.exponential(2, n_samples)
    cohesion = np.clip(cohesion, 0, 10)
    
    # Peak friction angle (slightly higher than critical state)
    peak_friction_angle = friction_angle + np.random.uniform(1, 5, n_samples)
    
    # Dilatancy angle
    dilatancy_angle = np.maximum(0, friction_angle - 30 + np.random.normal(0, 2, n_samples))
    
    # Young's modulus (MPa) - correlated with relative density
    youngs_modulus = 10 + 0.8 * basic_props_df['relative_density_pct'] + np.random.normal(0, 10, n_samples)
    youngs_modulus = np.clip(youngs_modulus, 15, 100)
    
    # Poisson's ratio
    poissons_ratio = np.random.normal(0.3, 0.05, n_samples)
    poissons_ratio = np.clip(poissons_ratio, 0.2, 0.4)
    
    # Shear modulus (MPa)
    shear_modulus = youngs_modulus / (2 * (1 + poissons_ratio))
    
    # Bulk modulus (MPa)
    bulk_modulus = youngs_modulus / (3 * (1 - 2 * poissons_ratio))
    
    # Critical state friction angle
    critical_friction_angle = friction_angle - np.random.uniform(1, 3, n_samples)
    
    return pd.DataFrame({
        'sample_id': basic_props_df['sample_id'],
        'friction_angle_deg': friction_angle,
        'peak_friction_angle_deg': peak_friction_angle,
        'critical_friction_angle_deg': critical_friction_angle,
        'cohesion_kPa': cohesion,
        'dilatancy_angle_deg': dilatancy_angle,
        'youngs_modulus_MPa': youngs_modulus,
        'shear_modulus_MPa': shear_modulus,
        'bulk_modulus_MPa': bulk_modulus,
        'poissons_ratio': poissons_ratio
    })

def generate_liquefaction_data(basic_props_df):
    """Generate liquefaction potential and pore pressure data"""
    
    n_samples = len(basic_props_df)
    
    # Cyclic resistance ratio (CRR) - function of relative density and fines content
    fines_content = basic_props_df['silt_content_pct'] + basic_props_df['clay_content_pct']
    
    # Base CRR calculation (simplified Seed-Idriss approach)
    crr_base = 0.065 + 0.008 * basic_props_df['relative_density_pct'] / 100
    
    # Fines content correction
    fc_correction = np.where(fines_content <= 5, 0, 
                           np.where(fines_content <= 35, 
                                  0.004 * (fines_content - 5), 0.12))
    
    cyclic_resistance_ratio = crr_base + fc_correction + np.random.normal(0, 0.02, n_samples)
    cyclic_resistance_ratio = np.clip(cyclic_resistance_ratio, 0.05, 0.5)
    
    # Cyclic stress ratio (CSR) for different earthquake magnitudes
    csr_m6 = np.random.uniform(0.1, 0.4, n_samples)
    csr_m7 = csr_m6 * np.random.uniform(1.2, 1.8, n_samples)
    csr_m8 = csr_m7 * np.random.uniform(1.1, 1.5, n_samples)
    
    # Factor of safety against liquefaction
    fs_m6 = cyclic_resistance_ratio / csr_m6
    fs_m7 = cyclic_resistance_ratio / csr_m7
    fs_m8 = cyclic_resistance_ratio / csr_m8
    
    # Liquefaction potential index
    lpi = np.where(fs_m7 < 1, 
                   (2 - fs_m7) * basic_props_df['depth_m'] * 0.5,
                   0)
    
    # Pore pressure parameters
    initial_pore_pressure = 9.81 * basic_props_df['depth_m']  # Hydrostatic
    
    # Excess pore pressure ratio during cyclic loading
    excess_pore_pressure_ratio = np.random.beta(2, 3, n_samples)
    
    # B-parameter (Skempton's pore pressure parameter)
    b_parameter = 0.95 + np.random.normal(0, 0.03, n_samples)
    b_parameter = np.clip(b_parameter, 0.85, 1.0)
    
    # Coefficient of earth pressure at rest
    k0 = 1 - np.sin(np.radians(basic_props_df['friction_angle_deg']))
    
    return pd.DataFrame({
        'sample_id': basic_props_df['sample_id'],
        'cyclic_resistance_ratio': cyclic_resistance_ratio,
        'csr_magnitude_6': csr_m6,
        'csr_magnitude_7': csr_m7,
        'csr_magnitude_8': csr_m8,
        'factor_safety_m6': fs_m6,
        'factor_safety_m7': fs_m7,
        'factor_safety_m8': fs_m8,
        'liquefaction_potential_index': lpi,
        'initial_pore_pressure_kPa': initial_pore_pressure,
        'excess_pore_pressure_ratio': excess_pore_pressure_ratio,
        'b_parameter': b_parameter,
        'k0_coefficient': k0
    })

def main():
    """Generate all sandy soil datasets"""
    
    print("Generating sandy soil properties datasets...")
    
    # Generate basic properties
    basic_props = generate_sandy_soil_properties(500)
    
    # Add d50 for mechanical properties calculation
    basic_props['d50_mm'] = np.sqrt(basic_props['d10_mm'] * basic_props['d60_mm'])
    
    # Generate mechanical properties
    mechanical_props = generate_sandy_soil_mechanical_properties(basic_props)
    
    # Add friction angle to basic props for liquefaction calculations
    basic_props = basic_props.merge(mechanical_props[['sample_id', 'friction_angle_deg']], on='sample_id')
    
    # Generate liquefaction data
    liquefaction_data = generate_liquefaction_data(basic_props)
    
    # Save datasets
    os.makedirs('geotechnical_datasets/sandy_soils', exist_ok=True)
    
    basic_props.to_csv('geotechnical_datasets/sandy_soils/sandy_soil_basic_properties.csv', index=False)
    mechanical_props.to_csv('geotechnical_datasets/sandy_soils/sandy_soil_mechanical_properties.csv', index=False)
    liquefaction_data.to_csv('geotechnical_datasets/sandy_soils/sandy_soil_liquefaction_data.csv', index=False)
    
    # Create combined dataset
    combined_data = basic_props.merge(mechanical_props, on='sample_id').merge(liquefaction_data, on='sample_id')
    combined_data.to_csv('geotechnical_datasets/sandy_soils/sandy_soil_complete_dataset.csv', index=False)
    
    print(f"Generated {len(basic_props)} sandy soil samples")
    print("Files created:")
    print("- sandy_soil_basic_properties.csv")
    print("- sandy_soil_mechanical_properties.csv")
    print("- sandy_soil_liquefaction_data.csv")
    print("- sandy_soil_complete_dataset.csv")
    
    return combined_data

if __name__ == "__main__":
    sandy_data = main()