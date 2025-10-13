#!/usr/bin/env python3
"""
Geotechnical Dataset Generator for PhD Thesis
Generates synthetic datasets for sandy and clay soils with realistic properties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, lognorm, beta, gamma
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class GeotechnicalDataGenerator:
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed for reproducibility"""
        np.random.seed(seed)
        self.seed = seed
        
    def generate_sandy_soil_dataset(self, n_samples=1000):
        """
        Generate comprehensive sandy soil dataset with realistic geotechnical properties
        """
        print("Generating sandy soil dataset...")
        
        # Basic soil properties
        data = {}
        
        # Sand content (typically 80-100% for sandy soils)
        data['sand_content_pct'] = np.random.beta(8, 2, n_samples) * 20 + 80
        
        # Grain size distribution parameters
        # D10, D30, D50, D60 (mm) - following typical sand distributions
        data['D10_mm'] = np.random.lognormal(mean=0.5, sigma=0.3, size=n_samples)
        data['D30_mm'] = np.random.lognormal(mean=0.8, sigma=0.3, size=n_samples)
        data['D50_mm'] = np.random.lognormal(mean=1.0, sigma=0.3, size=n_samples)
        data['D60_mm'] = np.random.lognormal(mean=1.1, sigma=0.3, size=n_samples)
        
        # Uniformity coefficient
        data['Cu'] = data['D60_mm'] / data['D10_mm']
        
        # Coefficient of curvature
        data['Cc'] = (data['D30_mm'] ** 2) / (data['D60_mm'] * data['D10_mm'])
        
        # Relative density (0-100%)
        data['relative_density_pct'] = np.random.beta(2, 2, n_samples) * 100
        
        # Unit weight (kN/m³)
        data['unit_weight_kN_m3'] = np.random.normal(18.5, 1.5, n_samples)
        
        # Void ratio
        data['void_ratio'] = np.random.normal(0.6, 0.15, n_samples)
        
        # Mechanical properties
        # Friction angle (degrees) - depends on relative density and grain size
        base_friction = 30 + (data['relative_density_pct'] / 100) * 10
        data['friction_angle_deg'] = np.random.normal(base_friction, 3, n_samples)
        data['friction_angle_deg'] = np.clip(data['friction_angle_deg'], 25, 45)
        
        # Cohesion (kPa) - typically very low for sands
        data['cohesion_kPa'] = np.random.exponential(2, n_samples)
        data['cohesion_kPa'] = np.clip(data['cohesion_kPa'], 0, 10)
        
        # Liquefaction potential parameters
        # Standard Penetration Test (SPT) N-value
        data['SPT_N60'] = np.random.poisson(15 + data['relative_density_pct'] / 4, n_samples)
        data['SPT_N60'] = np.clip(data['SPT_N60'], 5, 50)
        
        # Cyclic resistance ratio (CRR) - empirical relationship
        data['CRR_7_5'] = 0.1 * (data['SPT_N60'] / 20) ** 0.5
        
        # Pore water pressure parameters
        data['initial_pore_pressure_kPa'] = np.random.uniform(0, 200, n_samples)
        data['excess_pore_pressure_kPa'] = np.random.exponential(50, n_samples)
        
        # Permeability (m/s)
        data['permeability_m_s'] = 10 ** np.random.normal(-3, 1, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata
        df['soil_type'] = 'Sandy Soil'
        df['data_source'] = 'Synthetic Generated'
        df['generation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def generate_clay_soil_dataset(self, n_samples=1000):
        """
        Generate comprehensive clay soil dataset with realistic geotechnical properties
        """
        print("Generating clay soil dataset...")
        
        data = {}
        
        # Basic soil properties
        # Liquid limit (%)
        data['liquid_limit_pct'] = np.random.lognormal(mean=3.5, sigma=0.5, size=n_samples)
        data['liquid_limit_pct'] = np.clip(data['liquid_limit_pct'], 20, 150)
        
        # Plastic limit (%)
        data['plastic_limit_pct'] = data['liquid_limit_pct'] * np.random.beta(3, 2, n_samples)
        
        # Plasticity index
        data['plasticity_index'] = data['liquid_limit_pct'] - data['plastic_limit_pct']
        
        # Preconsolidation stress (kPa)
        data['preconsolidation_stress_kPa'] = np.random.lognormal(mean=4, sigma=1, size=n_samples)
        data['preconsolidation_stress_kPa'] = np.clip(data['preconsolidation_stress_kPa'], 50, 2000)
        
        # Overconsolidation ratio
        data['OCR'] = np.random.lognormal(mean=0.5, sigma=0.8, size=n_samples)
        data['OCR'] = np.clip(data['OCR'], 1, 20)
        
        # Unit weight (kN/m³)
        data['unit_weight_kN_m3'] = np.random.normal(17.0, 2.0, n_samples)
        
        # Void ratio
        data['void_ratio'] = np.random.normal(0.8, 0.3, n_samples)
        
        # Mechanical properties
        # Undrained shear strength (kPa)
        base_strength = 20 + (data['preconsolidation_stress_kPa'] / 100) * 5
        data['undrained_shear_strength_kPa'] = np.random.normal(base_strength, base_strength * 0.3, n_samples)
        data['undrained_shear_strength_kPa'] = np.clip(data['undrained_shear_strength_kPa'], 5, 500)
        
        # Sensitivity
        data['sensitivity'] = np.random.lognormal(mean=1, sigma=0.5, size=n_samples)
        data['sensitivity'] = np.clip(data['sensitivity'], 1, 20)
        
        # Pore pressure ratio
        data['pore_pressure_ratio'] = np.random.beta(2, 3, n_samples)
        
        # Mineralogical data
        # Clay mineral content (%)
        data['smectite_pct'] = np.random.beta(2, 3, n_samples) * 100
        data['illite_pct'] = np.random.beta(3, 2, n_samples) * 100
        data['kaolinite_pct'] = np.random.beta(2, 3, n_samples) * 100
        data['chlorite_pct'] = np.random.beta(1, 4, n_samples) * 100
        
        # Normalize mineral percentages
        total_minerals = data['smectite_pct'] + data['illite_pct'] + data['kaolinite_pct'] + data['chlorite_pct']
        for mineral in ['smectite_pct', 'illite_pct', 'kaolinite_pct', 'chlorite_pct']:
            data[mineral] = (data[mineral] / total_minerals) * 100
        
        # Atterberg limits classification
        data['soil_classification'] = ['CL' if pi > 7 else 'CH' if pi < 4 else 'ML' for pi in data['plasticity_index']]
        
        # Permeability (m/s) - much lower than sands
        data['permeability_m_s'] = 10 ** np.random.normal(-8, 1, n_samples)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata
        df['soil_type'] = 'Clay Soil'
        df['data_source'] = 'Synthetic Generated'
        df['generation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def generate_failure_case_studies(self, n_cases=100):
        """
        Generate synthetic failure case study data
        """
        print("Generating failure case study dataset...")
        
        data = {}
        
        # Case study identifiers
        data['case_id'] = [f'CASE_{i:03d}' for i in range(1, n_cases + 1)]
        
        # Failure types
        failure_types = ['Liquefaction', 'Landslide', 'Settlement', 'Bearing Capacity', 'Slope Instability']
        data['failure_type'] = np.random.choice(failure_types, n_cases, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Soil conditions at failure
        data['soil_type'] = np.random.choice(['Sandy', 'Clayey', 'Mixed'], n_cases, p=[0.4, 0.4, 0.2])
        
        # Failure depth (m)
        data['failure_depth_m'] = np.random.lognormal(mean=2, sigma=1, size=n_cases)
        
        # Failure dimensions
        data['failure_length_m'] = np.random.lognormal(mean=3, sigma=1, size=n_cases)
        data['failure_width_m'] = np.random.lognormal(mean=2, sigma=0.8, size=n_cases)
        
        # Triggering factors
        data['earthquake_magnitude'] = np.random.normal(6.5, 1.5, n_cases)
        data['earthquake_magnitude'] = np.clip(data['earthquake_magnitude'], 4.0, 9.0)
        
        data['rainfall_intensity_mm_h'] = np.random.exponential(20, n_cases)
        
        # Structural response
        data['max_displacement_mm'] = np.random.lognormal(mean=3, sigma=1, size=n_cases)
        data['settlement_mm'] = np.random.lognormal(mean=2.5, sigma=1, size=n_cases)
        
        # Pore pressure measurements
        data['max_pore_pressure_kPa'] = np.random.lognormal(mean=4, sigma=1, size=n_cases)
        data['pore_pressure_ratio_at_failure'] = np.random.beta(2, 2, n_cases)
        
        # Damage assessment
        damage_levels = ['Minor', 'Moderate', 'Severe', 'Catastrophic']
        data['damage_level'] = np.random.choice(damage_levels, n_cases, p=[0.2, 0.4, 0.3, 0.1])
        
        # Economic impact (USD)
        data['economic_loss_usd'] = np.random.lognormal(mean=12, sigma=2, size=n_cases)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata
        df['data_source'] = 'Synthetic Case Studies'
        df['generation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df
    
    def generate_underground_structure_data(self, n_structures=200):
        """
        Generate underground structure data for analysis
        """
        print("Generating underground structure dataset...")
        
        data = {}
        
        # Structure identifiers
        data['structure_id'] = [f'STRUCT_{i:03d}' for i in range(1, n_structures + 1)]
        
        # Structure types
        structure_types = ['Tunnel', 'Basement', 'Underground Storage', 'Mining Shaft', 'Subway Station']
        data['structure_type'] = np.random.choice(structure_types, n_structures, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        
        # Geometric properties
        data['depth_m'] = np.random.lognormal(mean=2.5, sigma=1, size=n_structures)
        data['length_m'] = np.random.lognormal(mean=4, sigma=1, size=n_structures)
        data['width_m'] = np.random.lognormal(mean=3, sigma=0.8, size=n_structures)
        data['height_m'] = np.random.lognormal(mean=2.5, sigma=0.8, size=n_structures)
        
        # Construction parameters
        data['construction_year'] = np.random.randint(1950, 2023, n_structures)
        data['construction_duration_months'] = np.random.poisson(24, n_structures)
        
        # Soil conditions
        data['surrounding_soil_type'] = np.random.choice(['Sandy', 'Clayey', 'Mixed', 'Rock'], n_structures, p=[0.3, 0.3, 0.25, 0.15])
        
        # Loading conditions
        data['overburden_pressure_kPa'] = data['depth_m'] * np.random.normal(20, 5, n_structures)
        data['live_load_kPa'] = np.random.exponential(10, n_structures)
        
        # Performance indicators
        data['max_crack_width_mm'] = np.random.exponential(2, n_structures)
        data['leakage_rate_L_per_day'] = np.random.exponential(50, n_structures)
        data['maintenance_frequency_per_year'] = np.random.poisson(2, n_structures)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add metadata
        df['data_source'] = 'Synthetic Underground Structures'
        df['generation_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return df

def main():
    """Main function to generate all datasets"""
    print("Starting Geotechnical Dataset Generation...")
    print("=" * 50)
    
    # Initialize generator
    generator = GeotechnicalDataGenerator(seed=42)
    
    # Generate datasets
    sandy_soils = generator.generate_sandy_soil_dataset(n_samples=1000)
    clay_soils = generator.generate_clay_soil_dataset(n_samples=1000)
    failure_cases = generator.generate_failure_case_studies(n_cases=100)
    underground_structures = generator.generate_underground_structure_data(n_structures=200)
    
    # Save datasets
    print("\nSaving datasets...")
    sandy_soils.to_csv('sandy_soils/sandy_soil_properties.csv', index=False)
    clay_soils.to_csv('clay_soils/clay_soil_properties.csv', index=False)
    failure_cases.to_csv('case_studies/failure_case_studies.csv', index=False)
    underground_structures.to_csv('case_studies/underground_structures.csv', index=False)
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    summary_stats = {
        'sandy_soils': {
            'count': len(sandy_soils),
            'columns': list(sandy_soils.columns),
            'numeric_summary': sandy_soils.describe().to_dict()
        },
        'clay_soils': {
            'count': len(clay_soils),
            'columns': list(clay_soils.columns),
            'numeric_summary': clay_soils.describe().to_dict()
        },
        'failure_cases': {
            'count': len(failure_cases),
            'columns': list(failure_cases.columns),
            'failure_types': failure_cases['failure_type'].value_counts().to_dict()
        },
        'underground_structures': {
            'count': len(underground_structures),
            'columns': list(underground_structures.columns),
            'structure_types': underground_structures['structure_type'].value_counts().to_dict()
        }
    }
    
    with open('dataset_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print("\nDataset generation completed!")
    print(f"Generated {len(sandy_soils)} sandy soil samples")
    print(f"Generated {len(clay_soils)} clay soil samples")
    print(f"Generated {len(failure_cases)} failure case studies")
    print(f"Generated {len(underground_structures)} underground structure records")
    
    return sandy_soils, clay_soils, failure_cases, underground_structures

if __name__ == "__main__":
    main()