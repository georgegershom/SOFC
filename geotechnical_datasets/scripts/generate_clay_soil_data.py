#!/usr/bin/env python3
"""
Generate comprehensive clay soil properties dataset for geotechnical research
Focus: Underground structure failure mechanisms in clay soils
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

def generate_clay_soil_basic_properties(n_samples=400):
    """Generate clay soil basic properties dataset"""
    
    # Clay content (% by weight) - typically 30-80% for clay soils
    clay_content = np.random.normal(55, 15, n_samples)
    clay_content = np.clip(clay_content, 30, 85)
    
    # Silt content
    silt_content = np.random.normal(25, 12, n_samples)
    silt_content = np.clip(silt_content, 10, 50)
    
    # Sand content (remaining percentage)
    sand_content = 100 - clay_content - silt_content
    sand_content = np.clip(sand_content, 5, 40)
    
    # Normalize to ensure sum = 100%
    total = sand_content + silt_content + clay_content
    sand_content = sand_content / total * 100
    silt_content = silt_content / total * 100
    clay_content = clay_content / total * 100
    
    # Atterberg limits
    liquid_limit = 25 + 0.8 * clay_content + np.random.normal(0, 8, n_samples)
    liquid_limit = np.clip(liquid_limit, 25, 120)
    
    plastic_limit = 15 + 0.3 * clay_content + np.random.normal(0, 5, n_samples)
    plastic_limit = np.clip(plastic_limit, 12, 45)
    
    # Ensure PL < LL
    plastic_limit = np.minimum(plastic_limit, liquid_limit - 5)
    
    plasticity_index = liquid_limit - plastic_limit
    
    # Activity (PI/clay fraction)
    activity = plasticity_index / (clay_content / 100)
    
    # Natural water content (%)
    natural_water_content = plastic_limit + 0.3 * plasticity_index + np.random.normal(0, 5, n_samples)
    natural_water_content = np.clip(natural_water_content, 15, 80)
    
    # Liquidity index
    liquidity_index = (natural_water_content - plastic_limit) / plasticity_index
    
    # Specific gravity
    specific_gravity = np.random.normal(2.72, 0.08, n_samples)
    specific_gravity = np.clip(specific_gravity, 2.60, 2.85)
    
    # Void ratio (higher for clays)
    void_ratio = np.random.normal(0.9, 0.3, n_samples)
    void_ratio = np.clip(void_ratio, 0.4, 2.0)
    
    # Unit weights
    dry_unit_weight = (specific_gravity * 9.81) / (1 + void_ratio)
    saturated_unit_weight = ((specific_gravity + void_ratio) * 9.81) / (1 + void_ratio)
    
    # Porosity
    porosity = void_ratio / (1 + void_ratio)
    
    # Permeability (m/s) - much lower for clays
    permeability = np.random.lognormal(np.log(1e-9), 2.0, n_samples)
    permeability = np.clip(permeability, 1e-12, 1e-6)
    
    # Preconsolidation stress (kPa)
    preconsolidation_stress = np.random.lognormal(np.log(150), 0.8, n_samples)
    preconsolidation_stress = np.clip(preconsolidation_stress, 50, 1000)
    
    # Overconsolidation ratio
    effective_stress = 20 * np.random.uniform(1, 15, n_samples)  # Approximate current effective stress
    ocr = preconsolidation_stress / effective_stress
    
    # Sample locations (synthetic coordinates)
    latitude = np.random.uniform(30.0, 50.0, n_samples)
    longitude = np.random.uniform(-125.0, -70.0, n_samples)
    depth = np.random.uniform(2.0, 50.0, n_samples)
    
    # Sample IDs and dates
    sample_ids = [f"CLAY_{i+1:04d}" for i in range(n_samples)]
    base_date = datetime(2020, 1, 1)
    sample_dates = [base_date + timedelta(days=np.random.randint(0, 1460)) for _ in range(n_samples)]
    
    return pd.DataFrame({
        'sample_id': sample_ids,
        'sample_date': sample_dates,
        'latitude': latitude,
        'longitude': longitude,
        'depth_m': depth,
        'clay_content_pct': clay_content,
        'silt_content_pct': silt_content,
        'sand_content_pct': sand_content,
        'liquid_limit_pct': liquid_limit,
        'plastic_limit_pct': plastic_limit,
        'plasticity_index': plasticity_index,
        'activity': activity,
        'natural_water_content_pct': natural_water_content,
        'liquidity_index': liquidity_index,
        'specific_gravity': specific_gravity,
        'void_ratio': void_ratio,
        'porosity': porosity,
        'dry_unit_weight_kN_m3': dry_unit_weight,
        'saturated_unit_weight_kN_m3': saturated_unit_weight,
        'permeability_m_s': permeability,
        'preconsolidation_stress_kPa': preconsolidation_stress,
        'ocr': ocr
    })

def generate_clay_mineralogy(basic_props_df):
    """Generate clay mineralogy data"""
    
    n_samples = len(basic_props_df)
    
    # Clay mineral types (percentages of clay fraction)
    # Smectite (montmorillonite) - high plasticity
    smectite_pct = np.random.beta(2, 5, n_samples) * 60
    
    # Illite - moderate plasticity
    illite_pct = np.random.beta(3, 2, n_samples) * 50
    
    # Kaolinite - low plasticity
    kaolinite_pct = np.random.beta(2, 3, n_samples) * 40
    
    # Chlorite - low to moderate plasticity
    chlorite_pct = np.random.beta(4, 6, n_samples) * 20
    
    # Normalize to 100%
    total_clay_minerals = smectite_pct + illite_pct + kaolinite_pct + chlorite_pct
    smectite_pct = smectite_pct / total_clay_minerals * 100
    illite_pct = illite_pct / total_clay_minerals * 100
    kaolinite_pct = kaolinite_pct / total_clay_minerals * 100
    chlorite_pct = chlorite_pct / total_clay_minerals * 100
    
    # Dominant clay mineral
    clay_minerals = np.array([smectite_pct, illite_pct, kaolinite_pct, chlorite_pct]).T
    mineral_names = ['Smectite', 'Illite', 'Kaolinite', 'Chlorite']
    dominant_mineral = [mineral_names[np.argmax(row)] for row in clay_minerals]
    
    # Cation exchange capacity (meq/100g) - related to clay mineralogy
    cec_base = (smectite_pct * 0.8 + illite_pct * 0.25 + kaolinite_pct * 0.05 + chlorite_pct * 0.15)
    cec = cec_base + np.random.normal(0, 5, n_samples)
    cec = np.clip(cec, 5, 100)
    
    # Specific surface area (m²/g)
    ssa_base = (smectite_pct * 6.0 + illite_pct * 0.8 + kaolinite_pct * 0.15 + chlorite_pct * 0.3)
    specific_surface_area = ssa_base + np.random.normal(0, 20, n_samples)
    specific_surface_area = np.clip(specific_surface_area, 10, 600)
    
    # Swelling potential
    swelling_potential = np.where(smectite_pct > 40, 'High',
                         np.where(smectite_pct > 20, 'Medium', 'Low'))
    
    # Free swell index (%)
    free_swell_index = smectite_pct * 2 + np.random.normal(0, 10, n_samples)
    free_swell_index = np.clip(free_swell_index, 0, 150)
    
    return pd.DataFrame({
        'sample_id': basic_props_df['sample_id'],
        'smectite_pct': smectite_pct,
        'illite_pct': illite_pct,
        'kaolinite_pct': kaolinite_pct,
        'chlorite_pct': chlorite_pct,
        'dominant_clay_mineral': dominant_mineral,
        'cation_exchange_capacity_meq_100g': cec,
        'specific_surface_area_m2_g': specific_surface_area,
        'swelling_potential': swelling_potential,
        'free_swell_index_pct': free_swell_index
    })

def generate_clay_mechanical_properties(basic_props_df, mineralogy_df):
    """Generate mechanical properties for clay soils"""
    
    n_samples = len(basic_props_df)
    
    # Undrained shear strength (kPa) - correlated with plasticity and OCR
    su_base = 20 + 0.3 * basic_props_df['plasticity_index'] + 15 * np.log(basic_props_df['ocr'])
    undrained_shear_strength = su_base + np.random.normal(0, 15, n_samples)
    undrained_shear_strength = np.clip(undrained_shear_strength, 10, 200)
    
    # Sensitivity (ratio of undisturbed to remolded strength)
    sensitivity_base = 2 + 0.1 * basic_props_df['liquidity_index']
    sensitivity = sensitivity_base + np.random.lognormal(0, 0.3, n_samples)
    sensitivity = np.clip(sensitivity, 1.5, 50)
    
    # Effective stress parameters
    # Effective friction angle (degrees)
    friction_angle_eff = 20 + 15 / (1 + 0.1 * basic_props_df['plasticity_index']) + np.random.normal(0, 3, n_samples)
    friction_angle_eff = np.clip(friction_angle_eff, 15, 35)
    
    # Effective cohesion (kPa)
    cohesion_eff = np.random.exponential(5, n_samples)
    cohesion_eff = np.clip(cohesion_eff, 0, 25)
    
    # Compression index
    compression_index = 0.007 * (basic_props_df['liquid_limit_pct'] - 10) + np.random.normal(0, 0.05, n_samples)
    compression_index = np.clip(compression_index, 0.05, 0.8)
    
    # Recompression index
    recompression_index = compression_index / np.random.uniform(5, 15, n_samples)
    
    # Coefficient of consolidation (m²/year)
    cv = np.random.lognormal(np.log(0.5), 1.0, n_samples)
    cv = np.clip(cv, 0.01, 10)
    
    # Young's modulus (MPa) - much lower for clays
    youngs_modulus = 5 + 100 / (1 + 0.1 * basic_props_df['plasticity_index']) + np.random.normal(0, 10, n_samples)
    youngs_modulus = np.clip(youngs_modulus, 2, 50)
    
    # Poisson's ratio (higher for clays)
    poissons_ratio = np.random.normal(0.4, 0.08, n_samples)
    poissons_ratio = np.clip(poissons_ratio, 0.25, 0.5)
    
    # Shear modulus
    shear_modulus = youngs_modulus / (2 * (1 + poissons_ratio))
    
    # Critical state line parameters
    lambda_parameter = compression_index / (2.3 * (1 + basic_props_df['void_ratio']))
    kappa_parameter = recompression_index / (2.3 * (1 + basic_props_df['void_ratio']))
    
    # Pore pressure parameters
    a_parameter = 0.3 + 0.7 * basic_props_df['plasticity_index'] / 100 + np.random.normal(0, 0.1, n_samples)
    a_parameter = np.clip(a_parameter, 0.2, 1.2)
    
    b_parameter = 0.95 + np.random.normal(0, 0.03, n_samples)
    b_parameter = np.clip(b_parameter, 0.85, 1.0)
    
    return pd.DataFrame({
        'sample_id': basic_props_df['sample_id'],
        'undrained_shear_strength_kPa': undrained_shear_strength,
        'sensitivity': sensitivity,
        'effective_friction_angle_deg': friction_angle_eff,
        'effective_cohesion_kPa': cohesion_eff,
        'compression_index': compression_index,
        'recompression_index': recompression_index,
        'coefficient_consolidation_m2_year': cv,
        'youngs_modulus_MPa': youngs_modulus,
        'shear_modulus_MPa': shear_modulus,
        'poissons_ratio': poissons_ratio,
        'lambda_parameter': lambda_parameter,
        'kappa_parameter': kappa_parameter,
        'a_parameter': a_parameter,
        'b_parameter': b_parameter
    })

def generate_clay_failure_susceptibility(basic_props_df, mechanical_df):
    """Generate failure susceptibility parameters for clay soils"""
    
    n_samples = len(basic_props_df)
    
    # Slope stability factor of safety
    slope_angle = np.random.uniform(10, 45, n_samples)  # degrees
    
    # Simplified Bishop method factor of safety
    factor_safety_slope = (mechanical_df['undrained_shear_strength_kPa'] / 
                          (basic_props_df['saturated_unit_weight_kN_m3'] * 
                           basic_props_df['depth_m'] * np.sin(np.radians(slope_angle))))
    
    # Progressive failure index
    progressive_failure_index = mechanical_df['sensitivity'] / factor_safety_slope
    
    # Creep susceptibility (based on plasticity and OCR)
    creep_susceptibility = np.where(
        (basic_props_df['plasticity_index'] > 30) & (basic_props_df['ocr'] > 2),
        'High',
        np.where(basic_props_df['plasticity_index'] > 20, 'Medium', 'Low')
    )
    
    # Swelling pressure (kPa) for expansive clays
    swelling_pressure = np.where(
        basic_props_df['plasticity_index'] > 35,
        50 + 2 * basic_props_df['plasticity_index'] + np.random.normal(0, 20, n_samples),
        np.random.exponential(10, n_samples)
    )
    swelling_pressure = np.clip(swelling_pressure, 0, 500)
    
    # Shrinkage limit (%)
    shrinkage_limit = basic_props_df['plastic_limit_pct'] - 0.3 * basic_props_df['plasticity_index'] + np.random.normal(0, 3, n_samples)
    shrinkage_limit = np.clip(shrinkage_limit, 5, 25)
    
    # Linear shrinkage (%)
    linear_shrinkage = 0.001 * basic_props_df['plasticity_index']**1.5 + np.random.normal(0, 0.5, n_samples)
    linear_shrinkage = np.clip(linear_shrinkage, 0, 15)
    
    # Potential for piping/erosion
    piping_potential = np.where(
        (basic_props_df['plasticity_index'] < 15) & (mechanical_df['sensitivity'] > 4),
        'High',
        np.where(basic_props_df['plasticity_index'] < 25, 'Medium', 'Low')
    )
    
    return pd.DataFrame({
        'sample_id': basic_props_df['sample_id'],
        'slope_angle_deg': slope_angle,
        'factor_safety_slope': factor_safety_slope,
        'progressive_failure_index': progressive_failure_index,
        'creep_susceptibility': creep_susceptibility,
        'swelling_pressure_kPa': swelling_pressure,
        'shrinkage_limit_pct': shrinkage_limit,
        'linear_shrinkage_pct': linear_shrinkage,
        'piping_potential': piping_potential
    })

def main():
    """Generate all clay soil datasets"""
    
    print("Generating clay soil properties datasets...")
    
    # Generate basic properties
    basic_props = generate_clay_soil_basic_properties(400)
    
    # Generate mineralogy data
    mineralogy_data = generate_clay_mineralogy(basic_props)
    
    # Generate mechanical properties
    mechanical_props = generate_clay_mechanical_properties(basic_props, mineralogy_data)
    
    # Generate failure susceptibility data
    failure_data = generate_clay_failure_susceptibility(basic_props, mechanical_props)
    
    # Save datasets
    os.makedirs('geotechnical_datasets/clay_soils', exist_ok=True)
    
    basic_props.to_csv('geotechnical_datasets/clay_soils/clay_soil_basic_properties.csv', index=False)
    mineralogy_data.to_csv('geotechnical_datasets/clay_soils/clay_soil_mineralogy.csv', index=False)
    mechanical_props.to_csv('geotechnical_datasets/clay_soils/clay_soil_mechanical_properties.csv', index=False)
    failure_data.to_csv('geotechnical_datasets/clay_soils/clay_soil_failure_susceptibility.csv', index=False)
    
    # Create combined dataset
    combined_data = (basic_props
                    .merge(mineralogy_data, on='sample_id')
                    .merge(mechanical_props, on='sample_id')
                    .merge(failure_data, on='sample_id'))
    
    combined_data.to_csv('geotechnical_datasets/clay_soils/clay_soil_complete_dataset.csv', index=False)
    
    print(f"Generated {len(basic_props)} clay soil samples")
    print("Files created:")
    print("- clay_soil_basic_properties.csv")
    print("- clay_soil_mineralogy.csv")
    print("- clay_soil_mechanical_properties.csv")
    print("- clay_soil_failure_susceptibility.csv")
    print("- clay_soil_complete_dataset.csv")
    
    return combined_data

if __name__ == "__main__":
    clay_data = main()