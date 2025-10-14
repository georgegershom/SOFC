"""
Main script to generate the complete integrated building retrofit dataset
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import random
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generators.iot_sensor_generator import IoTSensorDataGenerator, generate_iot_dataset_for_buildings
from src.generators.building_attributes_generator import BuildingAttributesGenerator
from src.generators.energy_performance_generator import EnergyPerformanceGenerator
from src.generators.lca_generator import LCADataGenerator


def generate_complete_dataset(n_buildings: int = 100, 
                             output_dir: str = '../data/raw/',
                             seed: int = 42):
    """Generate complete integrated dataset for building retrofits"""
    
    print("=" * 80)
    print("BUILDING RETROFIT DATASET GENERATOR")
    print("For PhD Research on AI- and IoT-driven Building Retrofit Optimization")
    print("=" * 80)
    print()
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    for subdir in ['iot_sensors', 'building_attributes', 'energy_performance', 'lca']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    # Initialize generators
    print("Initializing data generators...")
    building_gen = BuildingAttributesGenerator(seed)
    energy_gen = EnergyPerformanceGenerator(seed)
    lca_gen = LCADataGenerator(seed)
    
    # Generate building IDs
    building_ids = [f"BLD_{str(i+1).zfill(5)}" for i in range(n_buildings)]
    
    print(f"\nGenerating data for {n_buildings} buildings...")
    print("-" * 80)
    
    # 1. Generate Building Attributes
    print("\n[1/5] Generating building attributes and fabric data...")
    buildings_df = building_gen.generate_building_dataset(n_buildings)
    buildings_df.to_csv(os.path.join(output_dir, 'building_attributes', 'buildings.csv'), index=False)
    buildings_df.to_parquet(os.path.join(output_dir, 'building_attributes', 'buildings.parquet'), index=False)
    print(f"✓ Generated attributes for {len(buildings_df)} buildings")
    
    # 2. Generate IoT Sensor Data (sample for 10 buildings due to size)
    print("\n[2/5] Generating IoT sensor time-series data...")
    sample_buildings = random.sample(building_ids, min(10, n_buildings))
    iot_data = generate_iot_dataset_for_buildings(
        sample_buildings,
        start_date='2023-01-01',
        end_date='2024-01-01',
        sampling_rate_minutes=60  # Hourly data to reduce size
    )
    
    for data_type, df in iot_data.items():
        df.to_csv(os.path.join(output_dir, 'iot_sensors', f'{data_type}_data.csv'), index=False)
        df.to_parquet(os.path.join(output_dir, 'iot_sensors', f'{data_type}_data.parquet'), index=False)
        print(f"✓ Generated {len(df):,} {data_type} sensor records")
    
    # 3. Generate Energy Performance Data
    print("\n[3/5] Generating energy performance and historical consumption data...")
    
    all_historical = []
    all_ratings = []
    all_retrofits = []
    all_benchmarks = []
    
    for idx, row in tqdm(buildings_df.iterrows(), total=len(buildings_df), desc="Processing buildings"):
        building_id = row['building_id']
        
        # Historical consumption
        historical = energy_gen.generate_historical_consumption(
            building_id=building_id,
            construction_year=row['construction_year'],
            floor_area=row['gross_floor_area_m2'],
            building_type=row['building_type'],
            n_years=5
        )
        all_historical.append(historical)
        
        # Energy ratings
        ratings = energy_gen.generate_efficiency_ratings(
            building_id=building_id,
            construction_year=row['construction_year'],
            current_rating=row['energy_rating'],
            n_assessments=3
        )
        all_ratings.append(ratings)
        
        # Retrofit savings (for 30% of buildings)
        if random.random() < 0.3:
            retrofit_measures = random.sample([
                'wall_insulation', 'roof_insulation', 'window_upgrade',
                'hvac_upgrade', 'lighting_led', 'solar_panels',
                'heat_pump', 'building_controls'
            ], random.randint(2, 5))
            
            retrofit_data = energy_gen.generate_retrofit_savings(
                building_id=building_id,
                pre_retrofit_consumption=historical['total_kwh'].sum(),
                retrofit_measures=retrofit_measures,
                retrofit_date='2023-01-01'
            )
            all_retrofits.append(retrofit_data)
        
        # Benchmarking
        benchmark = energy_gen.generate_benchmarking_data(
            building_id=building_id,
            building_type=row['building_type'],
            floor_area=row['gross_floor_area_m2'],
            energy_rating=row['energy_rating']
        )
        all_benchmarks.append(benchmark)
    
    # Combine and save energy performance data
    historical_df = pd.concat(all_historical, ignore_index=True)
    ratings_df = pd.concat(all_ratings, ignore_index=True)
    retrofits_df = pd.concat(all_retrofits, ignore_index=True) if all_retrofits else pd.DataFrame()
    benchmarks_df = pd.DataFrame(all_benchmarks)
    
    historical_df.to_csv(os.path.join(output_dir, 'energy_performance', 'historical_consumption.csv'), index=False)
    ratings_df.to_csv(os.path.join(output_dir, 'energy_performance', 'energy_ratings.csv'), index=False)
    if not retrofits_df.empty:
        retrofits_df.to_csv(os.path.join(output_dir, 'energy_performance', 'retrofit_savings.csv'), index=False)
    benchmarks_df.to_csv(os.path.join(output_dir, 'energy_performance', 'benchmarking.csv'), index=False)
    
    print(f"✓ Generated {len(historical_df):,} historical consumption records")
    print(f"✓ Generated {len(ratings_df):,} energy rating assessments")
    if not retrofits_df.empty:
        print(f"✓ Generated {len(retrofits_df):,} retrofit savings records")
    print(f"✓ Generated {len(benchmarks_df):,} benchmarking records")
    
    # 4. Generate LCA Data
    print("\n[4/5] Generating Lifecycle Assessment (LCA) data...")
    
    all_building_lca = []
    all_retrofit_lca = []
    all_lifecycle_impacts = []
    all_carbon_offsets = []
    
    sample_for_lca = random.sample(range(len(buildings_df)), min(50, n_buildings))
    
    for idx in tqdm(sample_for_lca, desc="Generating LCA data"):
        row = buildings_df.iloc[idx]
        building_id = row['building_id']
        
        # Building materials LCA
        materials = {
            'structure': row['material_structure'],
            'facade': row['material_facade'],
            'insulation': row['material_insulation'],
            'windows': row['material_windows'],
            'roof': row['material_roof']
        }
        
        building_lca = lca_gen.generate_building_lca(
            building_id=building_id,
            floor_area=row['gross_floor_area_m2'],
            materials=materials
        )
        all_building_lca.append(building_lca)
        
        # Lifecycle impacts
        lifecycle_impacts = lca_gen.generate_lifecycle_impacts(
            building_id=building_id,
            building_lca=building_lca,
            lifespan_years=50
        )
        all_lifecycle_impacts.append(lifecycle_impacts)
        
        # Retrofit LCA (for buildings with retrofit potential)
        if row['retrofit_potential'] in ['high', 'medium']:
            retrofit_measures = random.sample([
                'wall_insulation', 'roof_insulation', 'window_upgrade',
                'solar_panels', 'heat_pump'
            ], random.randint(2, 4))
            
            retrofit_lca = lca_gen.generate_retrofit_lca(
                building_id=building_id,
                retrofit_measures=retrofit_measures,
                floor_area=row['gross_floor_area_m2']
            )
            all_retrofit_lca.append(retrofit_lca)
            
            # Carbon offset potential
            carbon_offset = lca_gen.generate_carbon_offset_potential(
                building_id=building_id,
                retrofit_measures=retrofit_measures,
                floor_area=row['gross_floor_area_m2']
            )
            all_carbon_offsets.append(carbon_offset)
    
    # Combine and save LCA data
    building_lca_df = pd.concat(all_building_lca, ignore_index=True) if all_building_lca else pd.DataFrame()
    retrofit_lca_df = pd.concat(all_retrofit_lca, ignore_index=True) if all_retrofit_lca else pd.DataFrame()
    lifecycle_impacts_df = pd.DataFrame(all_lifecycle_impacts)
    carbon_offsets_df = pd.DataFrame(all_carbon_offsets)
    
    if not building_lca_df.empty:
        building_lca_df.to_csv(os.path.join(output_dir, 'lca', 'building_materials_lca.csv'), index=False)
    if not retrofit_lca_df.empty:
        retrofit_lca_df.to_csv(os.path.join(output_dir, 'lca', 'retrofit_materials_lca.csv'), index=False)
    lifecycle_impacts_df.to_csv(os.path.join(output_dir, 'lca', 'lifecycle_impacts.csv'), index=False)
    carbon_offsets_df.to_csv(os.path.join(output_dir, 'lca', 'carbon_offset_potential.csv'), index=False)
    
    print(f"✓ Generated {len(building_lca_df):,} building LCA records")
    if not retrofit_lca_df.empty:
        print(f"✓ Generated {len(retrofit_lca_df):,} retrofit LCA records")
    print(f"✓ Generated {len(lifecycle_impacts_df):,} lifecycle impact assessments")
    print(f"✓ Generated {len(carbon_offsets_df):,} carbon offset calculations")
    
    # 5. Create Integrated Dataset
    print("\n[5/5] Creating integrated dataset...")
    
    # Merge key metrics for integrated analysis
    integrated_df = buildings_df.copy()
    
    # Add aggregated energy performance
    energy_summary = historical_df.groupby('building_id').agg({
        'total_kwh': ['mean', 'sum'],
        'eui_kwh_m2': 'mean',
        'carbon_emissions_kg_co2': 'sum'
    }).reset_index()
    energy_summary.columns = ['building_id', 'avg_annual_kwh', 'total_kwh', 'avg_eui', 'total_carbon_kg']
    integrated_df = integrated_df.merge(energy_summary, on='building_id', how='left')
    
    # Add latest energy rating
    latest_ratings = ratings_df.sort_values('assessment_date').groupby('building_id').last().reset_index()
    integrated_df = integrated_df.merge(
        latest_ratings[['building_id', 'primary_energy_kwh_m2_yr', 'co2_emissions_kg_m2_yr']],
        on='building_id', how='left'
    )
    
    # Add benchmarking percentile
    integrated_df = integrated_df.merge(
        benchmarks_df[['building_id', 'your_percentile', 'performance_class']],
        on='building_id', how='left'
    )
    
    # Add lifecycle carbon if available
    if not lifecycle_impacts_df.empty:
        integrated_df = integrated_df.merge(
            lifecycle_impacts_df[['building_id', 'net_lifecycle_carbon_kg_co2']],
            on='building_id', how='left'
        )
    
    # Save integrated dataset
    integrated_df.to_csv(os.path.join(output_dir, 'integrated_building_data.csv'), index=False)
    integrated_df.to_parquet(os.path.join(output_dir, 'integrated_building_data.parquet'), index=False)
    
    print(f"✓ Created integrated dataset with {len(integrated_df)} buildings and {len(integrated_df.columns)} features")
    
    # Generate metadata
    metadata = {
        'generation_date': datetime.now().isoformat(),
        'n_buildings': n_buildings,
        'data_categories': {
            'building_attributes': len(buildings_df.columns),
            'iot_sensors': {k: len(v.columns) for k, v in iot_data.items()},
            'energy_performance': {
                'historical': len(historical_df.columns),
                'ratings': len(ratings_df.columns),
                'retrofits': len(retrofits_df.columns) if not retrofits_df.empty else 0,
                'benchmarks': len(benchmarks_df.columns)
            },
            'lca': {
                'building_materials': len(building_lca_df.columns) if not building_lca_df.empty else 0,
                'retrofit_materials': len(retrofit_lca_df.columns) if not retrofit_lca_df.empty else 0,
                'lifecycle_impacts': len(lifecycle_impacts_df.columns),
                'carbon_offsets': len(carbon_offsets_df.columns)
            }
        },
        'total_records': {
            'buildings': len(buildings_df),
            'iot_records': sum(len(df) for df in iot_data.values()),
            'energy_records': len(historical_df) + len(ratings_df) + len(retrofits_df),
            'lca_records': len(building_lca_df) + len(retrofit_lca_df)
        },
        'file_formats': ['CSV', 'Parquet'],
        'random_seed': seed
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Summary Statistics:")
    print(f"   • Total buildings: {n_buildings}")
    print(f"   • Total data points: {metadata['total_records']}")
    print(f"   • Data categories: {len(metadata['data_categories'])}")
    print(f"   • Output location: {os.path.abspath(output_dir)}")
    print("\n🎓 Ready for PhD research on AI-driven building retrofit optimization!")
    print("=" * 80)
    
    return integrated_df


if __name__ == "__main__":
    # Generate the dataset
    integrated_data = generate_complete_dataset(
        n_buildings=100,  # Generate 100 buildings
        output_dir='../data/raw/',
        seed=42
    )