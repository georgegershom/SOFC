#!/usr/bin/env python3
"""
Data integration pipeline for building retrofit dataset.
Merges IoT sensor data, building attributes, energy performance, and LCA data
into a comprehensive, analysis-ready dataset.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DataIntegrator:
    def __init__(self, data_dir: str = '../raw_data'):
        self.data_dir = data_dir
        self.processed_dir = '../processed_data'
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw data files."""
        
        print("Loading raw data files...")
        
        data = {}
        
        # Load IoT sensor data
        iot_dir = f"{self.data_dir}/iot_sensors"
        if os.path.exists(iot_dir):
            for file in os.listdir(iot_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(iot_dir, file)
                    df = pd.read_csv(filepath)
                    data[f"iot_{file.replace('.csv', '')}"] = df
                    print(f"Loaded IoT data: {file} ({len(df)} records)")
        
        # Load building attributes
        attr_file = f"{self.data_dir}/building_attributes/building_attributes.csv"
        if os.path.exists(attr_file):
            data['building_attributes'] = pd.read_csv(attr_file)
            print(f"Loaded building attributes: {len(data['building_attributes'])} buildings")
        
        # Load energy performance data
        energy_dir = f"{self.data_dir}/energy_performance"
        if os.path.exists(energy_dir):
            for file in os.listdir(energy_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(energy_dir, file)
                    df = pd.read_csv(filepath)
                    data[f"energy_{file.replace('.csv', '')}"] = df
                    print(f"Loaded energy data: {file} ({len(df)} records)")
        
        # Load LCA data
        lca_dir = f"{self.data_dir}/lca_data"
        if os.path.exists(lca_dir):
            for file in os.listdir(lca_dir):
                if file.endswith('.csv'):
                    filepath = os.path.join(lca_dir, file)
                    df = pd.read_csv(filepath)
                    data[f"lca_{file.replace('.csv', '')}"] = df
                    print(f"Loaded LCA data: {file} ({len(df)} records)")
        
        return data
    
    def create_building_master_table(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create master building table with all static attributes."""
        
        print("Creating building master table...")
        
        # Start with building attributes
        if 'building_attributes' not in data:
            raise ValueError("Building attributes data not found")
        
        master_df = data['building_attributes'].copy()
        
        # Add energy efficiency ratings
        if 'energy_efficiency_ratings' in data:
            efficiency_df = data['energy_efficiency_ratings']
            master_df = master_df.merge(efficiency_df[['building_id', 'eu_rating', 'energy_star_score', 
                                                     'leed_certified', 'leed_level']], 
                                      on='building_id', how='left')
        
        # Add retrofit information
        if 'energy_retrofit_data' in data:
            retrofit_df = data['energy_retrofit_data']
            master_df = master_df.merge(retrofit_df[['building_id', 'has_retrofit', 'retrofit_year', 
                                                   'retrofit_type', 'retrofit_cost_eur', 
                                                   'energy_savings_percent']], 
                                      on='building_id', how='left')
        
        # Add LCA data
        if 'lca_building_lca_database' in data:
            lca_df = data['lca_building_lca_database']
            lca_cols = ['total_gwp_kg_co2e', 'total_energy_mj', 'total_water_l', 
                       'gwp_per_m2_kg_co2e', 'renewable_energy_percent', 'recycled_materials_percent']
            master_df = master_df.merge(lca_df[['building_id'] + lca_cols], 
                                      on='building_id', how='left')
        
        return master_df
    
    def create_time_series_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create integrated time series dataset with IoT and energy data."""
        
        print("Creating time series dataset...")
        
        # Get all IoT data
        iot_data = []
        for key, df in data.items():
            if key.startswith('iot_') and 'timestamp' in df.columns:
                iot_data.append(df)
        
        if not iot_data:
            print("No IoT data found")
            return pd.DataFrame()
        
        # Combine all IoT data
        combined_iot = pd.concat(iot_data, ignore_index=True)
        
        # Pivot to get all metrics in columns
        time_series_cols = ['building_id', 'timestamp']
        metric_cols = [col for col in combined_iot.columns if col not in time_series_cols]
        
        # Create wide format
        time_series_df = combined_iot.pivot_table(
            index=['building_id', 'timestamp'], 
            columns=[col for col in combined_iot.columns if col not in time_series_cols and col != 'building_type'],
            values=[col for col in combined_iot.columns if col not in time_series_cols and col != 'building_type'],
            aggfunc='first'
        ).reset_index()
        
        # Flatten column names
        time_series_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in time_series_df.columns]
        
        return time_series_df
    
    def create_energy_analysis_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create dataset for energy analysis with historical consumption and building attributes."""
        
        print("Creating energy analysis dataset...")
        
        # Get building master table
        if 'building_attributes' not in data:
            raise ValueError("Building attributes data not found")
        
        master_df = data['building_attributes'].copy()
        
        # Add historical consumption data
        if 'energy_historical_consumption' in data:
            hist_df = data['energy_historical_consumption']
            
            # Get latest year's data
            latest_year = hist_df['year'].max()
            latest_consumption = hist_df[hist_df['year'] == latest_year]
            
            # Merge with master table
            energy_cols = ['building_id', 'total_consumption_kwh', 'energy_intensity_kwh_m2',
                          'electricity_kwh', 'natural_gas_kwh', 'oil_kwh', 'district_heating_kwh', 'renewable_kwh']
            master_df = master_df.merge(latest_consumption[energy_cols], on='building_id', how='left')
        
        # Add efficiency ratings
        if 'energy_efficiency_ratings' in data:
            efficiency_df = data['energy_efficiency_ratings']
            master_df = master_df.merge(efficiency_df[['building_id', 'eu_rating', 'energy_star_score']], 
                                      on='building_id', how='left')
        
        return master_df
    
    def create_retrofit_analysis_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create dataset for retrofit analysis with all relevant factors."""
        
        print("Creating retrofit analysis dataset...")
        
        # Start with building master table
        if 'building_attributes' not in data:
            raise ValueError("Building attributes data not found")
        
        retrofit_df = data['building_attributes'].copy()
        
        # Add energy performance data
        if 'energy_historical_consumption' in data:
            hist_df = data['energy_historical_consumption']
            latest_year = hist_df['year'].max()
            latest_consumption = hist_df[hist_df['year'] == latest_year]
            retrofit_df = retrofit_df.merge(latest_consumption[['building_id', 'energy_intensity_kwh_m2']], 
                                          on='building_id', how='left')
        
        # Add retrofit scenarios
        if 'lca_retrofit_lca_scenarios' in data:
            retrofit_scenarios = data['lca_retrofit_lca_scenarios']
            
            # Get comprehensive retrofit scenario
            comprehensive_retrofit = retrofit_scenarios[retrofit_scenarios['retrofit_scenario'] == 'comprehensive_retrofit']
            retrofit_df = retrofit_df.merge(comprehensive_retrofit[['building_id', 'gwp_reduction_percent', 
                                                                  'energy_reduction_percent', 'retrofit_cost_eur_m2',
                                                                  'payback_period_years']], 
                                          on='building_id', how='left')
        
        # Add LCA data
        if 'lca_building_lca_database' in data:
            lca_df = data['lca_building_lca_database']
            retrofit_df = retrofit_df.merge(lca_df[['building_id', 'gwp_per_m2_kg_co2e', 'renewable_energy_percent']], 
                                          on='building_id', how='left')
        
        return retrofit_df
    
    def create_ml_ready_dataset(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create machine learning ready dataset with features and targets."""
        
        print("Creating ML-ready dataset...")
        
        # Start with building master table
        if 'building_attributes' not in data:
            raise ValueError("Building attributes data not found")
        
        ml_df = data['building_attributes'].copy()
        
        # Add energy performance as target variables
        if 'energy_historical_consumption' in data:
            hist_df = data['energy_historical_consumption']
            latest_year = hist_df['year'].max()
            latest_consumption = hist_df[hist_df['year'] == latest_year]
            ml_df = ml_df.merge(latest_consumption[['building_id', 'energy_intensity_kwh_m2']], 
                               on='building_id', how='left')
        
        # Add efficiency ratings
        if 'energy_efficiency_ratings' in data:
            efficiency_df = data['energy_efficiency_ratings']
            ml_df = ml_df.merge(efficiency_df[['building_id', 'eu_rating', 'energy_star_score']], 
                               on='building_id', how='left')
        
        # Create feature engineering
        ml_df = self.engineer_features(ml_df)
        
        return ml_df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for machine learning."""
        
        print("Engineering features...")
        
        # Age-related features
        df['age_category'] = pd.cut(df['building_age_years'], 
                                  bins=[0, 20, 40, 60, 100], 
                                  labels=['new', 'modern', 'old', 'historic'])
        
        # Size categories
        df['size_category'] = pd.cut(df['floor_area_m2'], 
                                   bins=[0, 100, 500, 1000, 5000, float('inf')], 
                                   labels=['small', 'medium', 'large', 'very_large', 'mega'])
        
        # Thermal performance score
        df['thermal_score'] = (df['r_value_wall_m2k_w'] + df['r_value_roof_m2k_w'] + 
                             df['r_value_floor_m2k_w']) / 3
        
        # Window performance
        df['window_performance'] = df['window_to_wall_ratio'] * df['r_value_window_m2k_w']
        
        # Construction quality score
        quality_scores = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
        df['quality_score'] = df['construction_quality'].map(quality_scores)
        
        # Insulation score
        df['insulation_score'] = df['insulation_present'].astype(int) * df['thermal_score']
        
        # Energy efficiency categories
        if 'energy_intensity_kwh_m2' in df.columns:
            df['energy_efficiency_category'] = pd.cut(df['energy_intensity_kwh_m2'], 
                                                    bins=[0, 100, 200, 300, 500, float('inf')], 
                                                    labels=['very_efficient', 'efficient', 'average', 'inefficient', 'very_inefficient'])
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """Validate data quality and completeness."""
        
        print("Validating data quality...")
        
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for critical missing values
        critical_columns = ['building_id', 'building_type', 'construction_year', 'floor_area_m2']
        critical_missing = {col: df[col].isnull().sum() for col in critical_columns if col in df.columns}
        validation_results['critical_missing'] = critical_missing
        
        return validation_results
    
    def save_integrated_data(self, datasets: Dict[str, pd.DataFrame], validation_results: Dict):
        """Save integrated datasets and validation results."""
        
        print("Saving integrated datasets...")
        
        # Create output directories
        os.makedirs(f"{self.processed_dir}/integrated", exist_ok=True)
        os.makedirs(f"{self.processed_dir}/validated", exist_ok=True)
        os.makedirs(f"{self.processed_dir}/analysis_ready", exist_ok=True)
        
        # Save datasets
        for name, df in datasets.items():
            if not df.empty:
                # Save to integrated folder
                df.to_csv(f"{self.processed_dir}/integrated/{name}.csv", index=False)
                
                # Save to analysis_ready folder
                df.to_csv(f"{self.processed_dir}/analysis_ready/{name}.csv", index=False)
                
                print(f"Saved {name}: {len(df)} records")
        
        # Save validation results
        with open(f"{self.processed_dir}/validated/validation_results.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        # Create dataset summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_datasets': len(datasets),
            'dataset_info': {name: {'records': len(df), 'columns': len(df.columns)} for name, df in datasets.items()},
            'validation_results': validation_results
        }
        
        with open(f"{self.processed_dir}/dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Dataset integration complete!")
        print(f"Generated {len(datasets)} integrated datasets")
        print(f"Validation results saved to {self.processed_dir}/validated/")

def main():
    """Main function to run data integration pipeline."""
    
    print("Starting data integration pipeline...")
    
    # Initialize integrator
    integrator = DataIntegrator()
    
    # Load all raw data
    raw_data = integrator.load_all_data()
    
    if not raw_data:
        print("No raw data found. Please run the data generation scripts first.")
        return
    
    # Create integrated datasets
    datasets = {}
    
    # Building master table
    try:
        datasets['building_master'] = integrator.create_building_master_table(raw_data)
    except Exception as e:
        print(f"Error creating building master table: {e}")
    
    # Time series dataset
    try:
        datasets['time_series'] = integrator.create_time_series_dataset(raw_data)
    except Exception as e:
        print(f"Error creating time series dataset: {e}")
    
    # Energy analysis dataset
    try:
        datasets['energy_analysis'] = integrator.create_energy_analysis_dataset(raw_data)
    except Exception as e:
        print(f"Error creating energy analysis dataset: {e}")
    
    # Retrofit analysis dataset
    try:
        datasets['retrofit_analysis'] = integrator.create_retrofit_analysis_dataset(raw_data)
    except Exception as e:
        print(f"Error creating retrofit analysis dataset: {e}")
    
    # ML-ready dataset
    try:
        datasets['ml_ready'] = integrator.create_ml_ready_dataset(raw_data)
    except Exception as e:
        print(f"Error creating ML-ready dataset: {e}")
    
    # Validate data quality
    validation_results = {}
    for name, df in datasets.items():
        if not df.empty:
            validation_results[name] = integrator.validate_data_quality(df)
    
    # Save integrated data
    integrator.save_integrated_data(datasets, validation_results)
    
    print("\nData integration pipeline completed successfully!")

if __name__ == "__main__":
    main()