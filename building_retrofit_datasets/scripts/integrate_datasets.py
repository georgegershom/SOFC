"""
Data Integration Pipeline for Building Retrofit Research
Merges all datasets and creates comprehensive analytical views.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

class DataIntegrationPipeline:
    """Integrate multiple datasets for comprehensive analysis."""
    
    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.datasets = {}
    
    def load_datasets(self):
        """Load all generated datasets."""
        print("Loading datasets...")
        
        try:
            self.datasets['building_attributes'] = pd.read_csv(f'{self.data_dir}/building_attributes.csv')
            print(f"  ✓ Building attributes: {len(self.datasets['building_attributes'])} records")
        except FileNotFoundError:
            print("  ✗ Building attributes not found")
        
        try:
            self.datasets['iot_sensor'] = pd.read_parquet(f'{self.data_dir}/iot_sensor_data.parquet')
            print(f"  ✓ IoT sensor data: {len(self.datasets['iot_sensor']):,} records")
        except FileNotFoundError:
            print("  ✗ IoT sensor data not found")
        
        try:
            self.datasets['energy_historical'] = pd.read_csv(f'{self.data_dir}/energy_performance_historical.csv')
            print(f"  ✓ Energy historical: {len(self.datasets['energy_historical']):,} records")
        except FileNotFoundError:
            print("  ✗ Energy historical not found")
        
        try:
            self.datasets['retrofit_scenarios'] = pd.read_csv(f'{self.data_dir}/retrofit_scenarios.csv')
            print(f"  ✓ Retrofit scenarios: {len(self.datasets['retrofit_scenarios'])} records")
        except FileNotFoundError:
            print("  ✗ Retrofit scenarios not found")
        
        try:
            self.datasets['lca_building'] = pd.read_csv(f'{self.data_dir}/lca_building_baseline.csv')
            print(f"  ✓ LCA building baseline: {len(self.datasets['lca_building'])} records")
        except FileNotFoundError:
            print("  ✗ LCA building baseline not found")
        
        try:
            self.datasets['lca_retrofit'] = pd.read_csv(f'{self.data_dir}/lca_retrofit_measures.csv')
            print(f"  ✓ LCA retrofit measures: {len(self.datasets['lca_retrofit'])} records")
        except FileNotFoundError:
            print("  ✗ LCA retrofit measures not found")
        
        try:
            self.datasets['carbon_payback'] = pd.read_csv(f'{self.data_dir}/lca_carbon_payback.csv')
            print(f"  ✓ Carbon payback: {len(self.datasets['carbon_payback'])} records")
        except FileNotFoundError:
            print("  ✗ Carbon payback not found")
    
    def create_building_master(self):
        """Create comprehensive building master dataset."""
        print("\nCreating building master dataset...")
        
        # Start with building attributes
        master = self.datasets['building_attributes'].copy()
        
        # Add LCA baseline data
        if 'lca_building' in self.datasets:
            lca_cols = ['building_id', 'total_embodied_carbon_kgco2eq', 'gwp_per_m2_kgco2eq',
                       'total_embodied_energy_mj', 'embodied_energy_per_m2_mj']
            lca_subset = self.datasets['lca_building'][lca_cols]
            master = master.merge(lca_subset, on='building_id', how='left')
        
        # Add energy statistics (average from historical data)
        if 'energy_historical' in self.datasets:
            energy_stats = self.datasets['energy_historical'].groupby('building_id').agg({
                'total_consumption_kwh': ['mean', 'std', 'min', 'max'],
                'carbon_emissions_kgco2': 'mean',
                'cost_eur': 'mean'
            }).reset_index()
            
            energy_stats.columns = ['building_id', 'avg_annual_consumption_kwh', 
                                   'std_annual_consumption_kwh', 'min_annual_consumption_kwh',
                                   'max_annual_consumption_kwh', 'avg_annual_carbon_kgco2',
                                   'avg_annual_cost_eur']
            
            master = master.merge(energy_stats, on='building_id', how='left')
        
        # Add best retrofit scenario info
        if 'retrofit_scenarios' in self.datasets:
            best_retrofit = self.datasets['retrofit_scenarios'].loc[
                self.datasets['retrofit_scenarios'].groupby('building_id')['roi_percent'].idxmax()
            ][['building_id', 'scenario_name', 'total_cost_eur', 'annual_energy_saving_kwh',
               'total_energy_saving_pct', 'roi_percent', 'simple_payback_years']].rename(columns={
                'scenario_name': 'best_retrofit_scenario',
                'total_cost_eur': 'best_retrofit_cost_eur',
                'annual_energy_saving_kwh': 'best_retrofit_savings_kwh',
                'total_energy_saving_pct': 'best_retrofit_energy_pct',
                'roi_percent': 'best_retrofit_roi_pct',
                'simple_payback_years': 'best_retrofit_payback_years'
            })
            
            master = master.merge(best_retrofit, on='building_id', how='left')
        
        return master
    
    def create_integrated_retrofit_analysis(self):
        """Create integrated retrofit analysis combining energy, cost, and LCA."""
        print("Creating integrated retrofit analysis...")
        
        if 'retrofit_scenarios' not in self.datasets:
            print("  Retrofit scenarios not available")
            return None
        
        retrofit = self.datasets['retrofit_scenarios'].copy()
        
        # Add building attributes
        if 'building_attributes' in self.datasets:
            building_cols = ['building_id', 'building_type', 'total_floor_area_m2', 
                           'epc_rating', 'construction_year', 'building_age_years']
            retrofit = retrofit.merge(
                self.datasets['building_attributes'][building_cols],
                on='building_id',
                how='left'
            )
        
        # Add LCA data for retrofit measures
        if 'lca_retrofit' in self.datasets:
            lca_summary = self.datasets['lca_retrofit'].groupby(['building_id', 'scenario_name']).agg({
                'gwp_total_kgco2eq': 'sum',
                'embodied_energy_total_mj': 'sum'
            }).reset_index()
            
            lca_summary.columns = ['building_id', 'scenario_name', 
                                   'retrofit_embodied_carbon_kgco2eq',
                                   'retrofit_embodied_energy_mj']
            
            retrofit = retrofit.merge(lca_summary, 
                                     on=['building_id', 'scenario_name'],
                                     how='left')
        
        # Add carbon payback data
        if 'carbon_payback' in self.datasets:
            payback_summary = self.datasets['carbon_payback'].groupby(['building_id', 'scenario_name']).agg({
                'carbon_payback_years': 'mean',
                'lifetime_carbon_benefit_kgco2eq': 'sum',
                'benefit_to_impact_ratio': 'mean'
            }).reset_index()
            
            retrofit = retrofit.merge(payback_summary,
                                     on=['building_id', 'scenario_name'],
                                     how='left')
        
        # Calculate comprehensive metrics
        retrofit['total_lifetime_carbon_savings'] = (
            retrofit['annual_carbon_saving_kgco2'] * 25  # 25 year lifetime
        ) - retrofit.get('retrofit_embodied_carbon_kgco2eq', 0)
        
        retrofit['cost_per_kgco2_saved'] = (
            retrofit['total_cost_eur'] / 
            retrofit['annual_carbon_saving_kgco2']
        ).round(2)
        
        retrofit['cost_per_kwh_saved'] = (
            retrofit['total_cost_eur'] / 
            retrofit['annual_energy_saving_kwh']
        ).round(2)
        
        return retrofit
    
    def create_time_series_aggregates(self):
        """Create aggregated time series data for analysis."""
        print("Creating time series aggregates...")
        
        if 'iot_sensor' not in self.datasets:
            print("  IoT sensor data not available")
            return None, None, None
        
        iot = self.datasets['iot_sensor'].copy()
        iot['timestamp'] = pd.to_datetime(iot['timestamp'])
        iot['date'] = iot['timestamp'].dt.date
        iot['hour'] = iot['timestamp'].dt.hour
        iot['month'] = iot['timestamp'].dt.month
        iot['day_of_week'] = iot['timestamp'].dt.dayofweek
        
        # Daily aggregates
        daily = iot.groupby(['building_id', 'date']).agg({
            'total_energy_consumption_kwh': 'sum',
            'indoor_temperature': 'mean',
            'outdoor_temperature': 'mean',
            'co2_level': 'mean',
            'occupancy_ratio': 'mean'
        }).reset_index()
        
        # Monthly aggregates
        monthly = iot.groupby(['building_id', iot['timestamp'].dt.to_period('M')]).agg({
            'total_energy_consumption_kwh': 'sum',
            'indoor_temperature': 'mean',
            'outdoor_temperature': 'mean',
            'co2_level': 'mean',
            'occupancy_ratio': 'mean'
        }).reset_index()
        monthly.columns = ['building_id', 'month', 'total_energy_consumption_kwh',
                          'avg_indoor_temp', 'avg_outdoor_temp', 'avg_co2', 'avg_occupancy']
        monthly['month'] = monthly['month'].astype(str)
        
        # Hourly patterns (average by hour of day)
        hourly_patterns = iot.groupby(['building_id', 'hour']).agg({
            'total_energy_consumption_kwh': 'mean',
            'occupancy_ratio': 'mean'
        }).reset_index()
        
        return daily, monthly, hourly_patterns
    
    def create_ml_ready_dataset(self):
        """Create ML-ready dataset for predictive modeling."""
        print("Creating ML-ready dataset...")
        
        if 'iot_sensor' not in self.datasets or 'building_attributes' not in self.datasets:
            print("  Required datasets not available")
            return None
        
        # Sample IoT data (to reduce size - take every 24th record for daily data)
        iot = self.datasets['iot_sensor'].copy()
        iot['timestamp'] = pd.to_datetime(iot['timestamp'])
        iot_sampled = iot[iot['timestamp'].dt.hour == 12].copy()  # Noon readings
        
        # Merge with building attributes
        ml_dataset = iot_sampled.merge(
            self.datasets['building_attributes'],
            on='building_id',
            how='left',
            suffixes=('', '_building')
        )
        
        # Select relevant features for ML
        feature_columns = [
            'building_id', 'timestamp',
            # Building features
            'building_type', 'total_floor_area_m2', 'num_floors', 'building_age_years',
            'epc_rating', 'envelope_avg_u_value', 'window_wall_ratio',
            # Environmental features
            'outdoor_temperature', 'outdoor_humidity', 'solar_radiation',
            'indoor_temperature', 'indoor_humidity', 'co2_level', 'occupancy_ratio',
            # Target variable
            'total_energy_consumption_kwh'
        ]
        
        ml_dataset = ml_dataset[feature_columns]
        
        # Add temporal features
        ml_dataset['day_of_week'] = ml_dataset['timestamp'].dt.dayofweek
        ml_dataset['month'] = ml_dataset['timestamp'].dt.month
        ml_dataset['day_of_year'] = ml_dataset['timestamp'].dt.dayofyear
        
        return ml_dataset
    
    def generate_data_quality_report(self):
        """Generate data quality report."""
        print("\nGenerating data quality report...")
        
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'datasets': {}
        }
        
        for name, df in self.datasets.items():
            if df is not None:
                report['datasets'][name] = {
                    'records': len(df),
                    'columns': len(df.columns),
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
                    'missing_values': df.isnull().sum().to_dict(),
                    'dtypes': df.dtypes.astype(str).to_dict()
                }
        
        return report
    
    def run_integration(self):
        """Run complete integration pipeline."""
        print("=" * 80)
        print("DATA INTEGRATION PIPELINE")
        print("=" * 80)
        
        # Load all datasets
        self.load_datasets()
        
        # Create integrated datasets
        integrated_datasets = {}
        
        # Building master
        building_master = self.create_building_master()
        if building_master is not None:
            building_master.to_csv(f'{self.data_dir}/integrated_building_master.csv', index=False)
            building_master.to_excel(f'{self.data_dir}/integrated_building_master.xlsx', index=False)
            integrated_datasets['building_master'] = building_master
            print(f"  ✓ Building master: {len(building_master)} records")
        
        # Retrofit analysis
        retrofit_analysis = self.create_integrated_retrofit_analysis()
        if retrofit_analysis is not None:
            retrofit_analysis.to_csv(f'{self.data_dir}/integrated_retrofit_analysis.csv', index=False)
            retrofit_analysis.to_excel(f'{self.data_dir}/integrated_retrofit_analysis.xlsx', index=False)
            integrated_datasets['retrofit_analysis'] = retrofit_analysis
            print(f"  ✓ Retrofit analysis: {len(retrofit_analysis)} records")
        
        # Time series aggregates
        daily, monthly, hourly = self.create_time_series_aggregates()
        if daily is not None:
            daily.to_csv(f'{self.data_dir}/timeseries_daily.csv', index=False)
            integrated_datasets['daily'] = daily
            print(f"  ✓ Daily time series: {len(daily):,} records")
        
        if monthly is not None:
            monthly.to_csv(f'{self.data_dir}/timeseries_monthly.csv', index=False)
            integrated_datasets['monthly'] = monthly
            print(f"  ✓ Monthly time series: {len(monthly)} records")
        
        if hourly is not None:
            hourly.to_csv(f'{self.data_dir}/hourly_patterns.csv', index=False)
            integrated_datasets['hourly_patterns'] = hourly
            print(f"  ✓ Hourly patterns: {len(hourly)} records")
        
        # ML-ready dataset
        ml_dataset = self.create_ml_ready_dataset()
        if ml_dataset is not None:
            ml_dataset.to_csv(f'{self.data_dir}/ml_ready_dataset.csv', index=False)
            ml_dataset.to_parquet(f'{self.data_dir}/ml_ready_dataset.parquet', index=False)
            integrated_datasets['ml_dataset'] = ml_dataset
            print(f"  ✓ ML-ready dataset: {len(ml_dataset):,} records")
        
        # Generate quality report
        quality_report = self.generate_data_quality_report()
        with open(f'{self.data_dir}/data_quality_report.json', 'w') as f:
            json.dump(quality_report, f, indent=2)
        print(f"  ✓ Data quality report generated")
        
        # Generate integration summary
        summary = {
            'integration_timestamp': datetime.now().isoformat(),
            'source_datasets': {name: len(df) for name, df in self.datasets.items()},
            'integrated_datasets': {name: len(df) for name, df in integrated_datasets.items()},
            'total_buildings': len(self.datasets.get('building_attributes', [])),
            'total_iot_records': len(self.datasets.get('iot_sensor', [])),
            'data_coverage': {
                'temporal': '2023-01-01 to 2023-12-31',
                'spatial': f"{len(self.datasets.get('building_attributes', []))} buildings",
                'resolution': 'Hourly for IoT, Annual for energy'
            }
        }
        
        with open(f'{self.data_dir}/integration_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 80)
        print("INTEGRATION SUMMARY")
        print("=" * 80)
        print(f"Source datasets: {len(self.datasets)}")
        print(f"Integrated datasets: {len(integrated_datasets)}")
        print(f"Total buildings: {summary['total_buildings']}")
        print(f"Total IoT records: {summary['total_iot_records']:,}")
        
        print("\n✅ Data integration complete!")
        print(f"📁 All integrated datasets saved to: {self.data_dir}/")
        
        return integrated_datasets, quality_report, summary


def main():
    """Main function."""
    pipeline = DataIntegrationPipeline()
    integrated_datasets, quality_report, summary = pipeline.run_integration()
    
    print("\n" + "=" * 80)
    print("KEY OUTPUT FILES:")
    print("=" * 80)
    print("  • integrated_building_master.csv/xlsx - Complete building profiles")
    print("  • integrated_retrofit_analysis.csv/xlsx - Comprehensive retrofit analysis")
    print("  • ml_ready_dataset.csv/parquet - ML-ready feature dataset")
    print("  • timeseries_daily.csv - Daily aggregated IoT data")
    print("  • timeseries_monthly.csv - Monthly aggregated data")
    print("  • data_quality_report.json - Data quality metrics")
    print("  • integration_summary.json - Integration metadata")


if __name__ == "__main__":
    main()
