"""
Data Integration Pipeline
Provides utilities for merging and integrating multi-source building data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import warnings

class BuildingDataIntegrator:
    """
    Integrates multiple data sources for building retrofit analysis
    """
    
    def __init__(self, data_dir: str = '../../data/raw/'):
        self.data_dir = data_dir
        self.integrated_data = None
        
    def load_building_attributes(self) -> pd.DataFrame:
        """Load building attributes data"""
        try:
            return pd.read_parquet(f'{self.data_dir}/building_attributes/buildings.parquet')
        except:
            return pd.read_csv(f'{self.data_dir}/building_attributes/buildings.csv')
    
    def load_iot_data(self, data_type: str = 'energy', 
                     aggregate: bool = True) -> pd.DataFrame:
        """Load and optionally aggregate IoT sensor data"""
        try:
            df = pd.read_parquet(f'{self.data_dir}/iot_sensors/{data_type}_data.parquet')
        except:
            df = pd.read_csv(f'{self.data_dir}/iot_sensors/{data_type}_data.csv')
        
        if aggregate and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            if data_type == 'energy':
                # Aggregate to daily level
                agg_dict = {col: 'sum' for col in df.columns 
                           if 'consumption' in col or 'kwh' in col}
                agg_dict.update({'building_id': 'first'})
                
                df = df.set_index('timestamp').groupby([
                    pd.Grouper(freq='D'), 'building_id'
                ]).agg(agg_dict).reset_index()
                
            elif data_type == 'ieq':
                # Aggregate to daily averages
                agg_dict = {
                    'temperature_c': 'mean',
                    'relative_humidity_pct': 'mean',
                    'co2_ppm': 'mean',
                    'tvoc_ugm3': 'mean',
                    'pm25_ugm3': 'mean'
                }
                
                df = df.set_index('timestamp').groupby([
                    pd.Grouper(freq='D'), 'building_id'
                ]).agg(agg_dict).reset_index()
        
        return df
    
    def load_energy_performance(self) -> Dict[str, pd.DataFrame]:
        """Load all energy performance data"""
        energy_data = {}
        
        files = [
            'historical_consumption',
            'energy_ratings',
            'retrofit_savings',
            'benchmarking'
        ]
        
        for file in files:
            try:
                try:
                    energy_data[file] = pd.read_parquet(
                        f'{self.data_dir}/energy_performance/{file}.parquet')
                except:
                    energy_data[file] = pd.read_csv(
                        f'{self.data_dir}/energy_performance/{file}.csv')
            except FileNotFoundError:
                warnings.warn(f"File {file} not found in energy_performance directory")
                energy_data[file] = pd.DataFrame()
        
        return energy_data
    
    def load_lca_data(self) -> Dict[str, pd.DataFrame]:
        """Load all LCA data"""
        lca_data = {}
        
        files = [
            'building_materials_lca',
            'retrofit_materials_lca',
            'lifecycle_impacts',
            'carbon_offset_potential'
        ]
        
        for file in files:
            try:
                try:
                    lca_data[file] = pd.read_parquet(f'{self.data_dir}/lca/{file}.parquet')
                except:
                    lca_data[file] = pd.read_csv(f'{self.data_dir}/lca/{file}.csv')
            except FileNotFoundError:
                warnings.warn(f"File {file} not found in lca directory")
                lca_data[file] = pd.DataFrame()
        
        return lca_data
    
    def create_time_series_features(self, df: pd.DataFrame, 
                                   date_col: str = 'timestamp') -> pd.DataFrame:
        """Create time-based features from datetime column"""
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col])
            df['year'] = df[date_col].dt.year
            df['month'] = df[date_col].dt.month
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['quarter'] = df[date_col].dt.quarter
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['season'] = df['month'].apply(self._get_season)
        
        return df
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def integrate_for_ml(self, include_iot: bool = True) -> pd.DataFrame:
        """
        Create integrated dataset optimized for machine learning
        """
        print("Loading building attributes...")
        buildings = self.load_building_attributes()
        
        # Convert categorical variables to numeric
        categorical_columns = buildings.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in ['building_id', 'building_name', 'address']:
                # Create dummy variables
                dummies = pd.get_dummies(buildings[col], prefix=col)
                buildings = pd.concat([buildings, dummies], axis=1)
        
        print("Loading energy performance data...")
        energy_data = self.load_energy_performance()
        
        # Aggregate historical consumption
        if not energy_data['historical_consumption'].empty:
            energy_agg = energy_data['historical_consumption'].groupby('building_id').agg({
                'total_kwh': ['mean', 'std', 'min', 'max'],
                'electricity_kwh': 'mean',
                'gas_kwh': 'mean',
                'eui_kwh_m2': 'mean',
                'carbon_emissions_kg_co2': 'sum'
            }).reset_index()
            
            # Flatten column names
            energy_agg.columns = ['building_id'] + [
                f'energy_{col[0]}_{col[1]}' for col in energy_agg.columns[1:]
            ]
            
            buildings = buildings.merge(energy_agg, on='building_id', how='left')
        
        # Add latest energy rating
        if not energy_data['energy_ratings'].empty:
            latest_rating = energy_data['energy_ratings'].sort_values(
                'assessment_date').groupby('building_id').last().reset_index()
            
            rating_cols = ['primary_energy_kwh_m2_yr', 'co2_emissions_kg_m2_yr', 
                          'renewable_energy_pct']
            buildings = buildings.merge(
                latest_rating[['building_id'] + rating_cols],
                on='building_id', how='left'
            )
        
        # Add benchmarking data
        if not energy_data['benchmarking'].empty:
            buildings = buildings.merge(
                energy_data['benchmarking'][['building_id', 'your_percentile', 
                                            'potential_savings_pct']],
                on='building_id', how='left'
            )
        
        print("Loading LCA data...")
        lca_data = self.load_lca_data()
        
        # Add lifecycle impacts
        if not lca_data['lifecycle_impacts'].empty:
            buildings = buildings.merge(
                lca_data['lifecycle_impacts'][['building_id', 
                                              'net_lifecycle_carbon_kg_co2',
                                              'carbon_per_m2_per_year_kg_co2']],
                on='building_id', how='left'
            )
        
        if include_iot:
            print("Loading and aggregating IoT data...")
            # Load and aggregate IoT data
            try:
                energy_iot = self.load_iot_data('energy', aggregate=True)
                if not energy_iot.empty:
                    # Calculate IoT-based metrics
                    iot_metrics = energy_iot.groupby('building_id').agg({
                        'total_consumption_kw': ['mean', 'std'],
                        'hvac_consumption_kw': 'mean',
                        'lighting_consumption_kw': 'mean'
                    }).reset_index()
                    
                    iot_metrics.columns = ['building_id'] + [
                        f'iot_{col[0]}_{col[1]}' for col in iot_metrics.columns[1:]
                    ]
                    
                    buildings = buildings.merge(iot_metrics, on='building_id', how='left')
            except Exception as e:
                print(f"Could not load IoT data: {e}")
        
        # Handle missing values
        numeric_columns = buildings.select_dtypes(include=[np.number]).columns
        buildings[numeric_columns] = buildings[numeric_columns].fillna(
            buildings[numeric_columns].median())
        
        print(f"Integrated dataset created with {len(buildings)} buildings and "
              f"{len(buildings.columns)} features")
        
        self.integrated_data = buildings
        return buildings
    
    def create_retrofit_recommendation_dataset(self) -> pd.DataFrame:
        """
        Create dataset specifically for retrofit recommendation models
        """
        buildings = self.load_building_attributes()
        energy_data = self.load_energy_performance()
        
        # Focus on buildings with retrofit data
        if not energy_data['retrofit_savings'].empty:
            retrofit_buildings = energy_data['retrofit_savings']['building_id'].unique()
            
            # Filter to buildings with retrofit data
            retrofit_df = buildings[buildings['building_id'].isin(retrofit_buildings)].copy()
            
            # Add retrofit measures and savings
            retrofit_summary = energy_data['retrofit_savings'].groupby('building_id').agg({
                'savings_percentage': 'mean',
                'energy_savings_kwh': 'sum',
                'cost_savings_eur': 'sum',
                'carbon_savings_kg_co2': 'sum',
                'retrofit_measures': 'first'
            }).reset_index()
            
            retrofit_df = retrofit_df.merge(retrofit_summary, on='building_id')
            
            # Create binary columns for each retrofit measure
            measures = ['wall_insulation', 'roof_insulation', 'window_upgrade',
                       'hvac_upgrade', 'lighting_led', 'solar_panels',
                       'heat_pump', 'building_controls']
            
            for measure in measures:
                retrofit_df[f'retrofit_{measure}'] = retrofit_df['retrofit_measures'].str.contains(
                    measure).astype(int)
            
            return retrofit_df
        else:
            warnings.warn("No retrofit data available")
            return pd.DataFrame()
    
    def create_time_series_dataset(self, building_ids: Optional[List[str]] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Create time series dataset for forecasting models
        """
        # Load energy IoT data
        energy_ts = self.load_iot_data('energy', aggregate=False)
        
        if building_ids:
            energy_ts = energy_ts[energy_ts['building_id'].isin(building_ids)]
        
        if start_date:
            energy_ts = energy_ts[energy_ts['timestamp'] >= start_date]
        
        if end_date:
            energy_ts = energy_ts[energy_ts['timestamp'] <= end_date]
        
        # Add time features
        energy_ts = self.create_time_series_features(energy_ts, 'timestamp')
        
        # Add weather data if available
        try:
            weather = self.load_iot_data('weather', aggregate=False)
            weather['timestamp'] = pd.to_datetime(weather['timestamp'])
            
            # Merge on nearest timestamp
            energy_ts = pd.merge_asof(
                energy_ts.sort_values('timestamp'),
                weather[['timestamp', 'outdoor_temperature_c', 'outdoor_humidity_pct',
                        'solar_radiation_wm2']].sort_values('timestamp'),
                on='timestamp',
                direction='nearest'
            )
        except:
            pass
        
        # Add building attributes
        buildings = self.load_building_attributes()
        key_attributes = ['building_id', 'building_type', 'gross_floor_area_m2',
                         'construction_year', 'energy_rating']
        
        energy_ts = energy_ts.merge(
            buildings[key_attributes],
            on='building_id',
            how='left'
        )
        
        return energy_ts
    
    def export_for_research(self, output_path: str = '../../data/processed/'):
        """
        Export integrated datasets in various formats for research use
        """
        import os
        os.makedirs(output_path, exist_ok=True)
        
        print("Creating research-ready datasets...")
        
        # 1. Main integrated dataset
        integrated = self.integrate_for_ml()
        integrated.to_csv(f'{output_path}/integrated_ml_dataset.csv', index=False)
        integrated.to_parquet(f'{output_path}/integrated_ml_dataset.parquet', index=False)
        
        # 2. Retrofit recommendation dataset
        retrofit = self.create_retrofit_recommendation_dataset()
        if not retrofit.empty:
            retrofit.to_csv(f'{output_path}/retrofit_recommendations.csv', index=False)
        
        # 3. Time series dataset (sample)
        ts_data = self.create_time_series_dataset()
        if not ts_data.empty:
            ts_data.to_csv(f'{output_path}/time_series_sample.csv', index=False)
        
        # 4. Create data dictionary
        data_dict = {
            'integrated_ml_dataset': {
                'description': 'Complete integrated dataset for ML models',
                'shape': integrated.shape,
                'features': list(integrated.columns),
                'numeric_features': list(integrated.select_dtypes(include=[np.number]).columns),
                'categorical_features': list(integrated.select_dtypes(exclude=[np.number]).columns)
            },
            'retrofit_recommendations': {
                'description': 'Dataset for retrofit recommendation models',
                'shape': retrofit.shape if not retrofit.empty else (0, 0),
                'features': list(retrofit.columns) if not retrofit.empty else []
            },
            'time_series_sample': {
                'description': 'Time series data for energy forecasting',
                'shape': ts_data.shape if not ts_data.empty else (0, 0),
                'features': list(ts_data.columns) if not ts_data.empty else []
            }
        }
        
        with open(f'{output_path}/data_dictionary.json', 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        print(f"✓ Exported {len(data_dict)} datasets to {output_path}")
        
        return data_dict


def main():
    """Main function to run the integration pipeline"""
    integrator = BuildingDataIntegrator()
    integrator.export_for_research()


if __name__ == "__main__":
    main()