
"""
Data Integration Script for Building Retrofit Dataset
This script demonstrates how to merge and integrate the various dataset components
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class BuildingRetrofitDataIntegrator:
    """Class for integrating and analyzing the building retrofit dataset"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.building_data = {}
        self.iot_data = {}
        self.energy_data = {}
        self.lca_data = {}
    
    def load_all_data(self):
        """Load all dataset components"""
        print("Loading dataset components...")
        
        # Load building attributes
        self.building_data['basic_info'] = pd.read_csv(f"{self.data_dir}/building_basic_info.csv")
        self.building_data['geometric'] = pd.read_csv(f"{self.data_dir}/building_geometric_data.csv")
        self.building_data['thermal'] = pd.read_csv(f"{self.data_dir}/building_thermal_properties.csv")
        self.building_data['materials'] = pd.read_csv(f"{self.data_dir}/building_construction_materials.csv")
        
        # Load IoT data
        self.iot_data['energy'] = pd.read_csv(f"{self.data_dir}/iot_energy_consumption.csv")
        self.iot_data['environmental'] = pd.read_csv(f"{self.data_dir}/iot_environmental_parameters.csv")
        self.iot_data['weather'] = pd.read_csv(f"{self.data_dir}/iot_weather_conditions.csv")
        self.iot_data['occupancy'] = pd.read_csv(f"{self.data_dir}/iot_occupancy_patterns.csv")
        
        # Load energy performance data
        self.energy_data['historical'] = pd.read_csv(f"{self.data_dir}/energy_historical_consumption.csv")
        self.energy_data['ratings'] = pd.read_csv(f"{self.data_dir}/energy_efficiency_ratings.csv")
        self.energy_data['retrofit'] = pd.read_csv(f"{self.data_dir}/energy_retrofit_impact.csv")
        
        # Load LCA data
        self.lca_data['epds'] = pd.read_csv(f"{self.data_dir}/lca_material_epds.csv")
        self.lca_data['building_lca'] = pd.read_csv(f"{self.data_dir}/lca_building_lca.csv")
        
        print("All data loaded successfully!")
    
    def create_integrated_building_dataset(self):
        """Create an integrated dataset with all building information"""
        print("Creating integrated building dataset...")
        
        # Start with basic info
        integrated = self.building_data['basic_info'].copy()
        
        # Merge geometric data
        integrated = integrated.merge(
            self.building_data['geometric'], 
            on='building_id', 
            how='left'
        )
        
        # Merge thermal properties
        integrated = integrated.merge(
            self.building_data['thermal'], 
            on='building_id', 
            how='left'
        )
        
        # Merge materials data
        integrated = integrated.merge(
            self.building_data['materials'], 
            on='building_id', 
            how='left'
        )
        
        # Add energy performance summary
        energy_summary = self.energy_data['historical'].groupby('building_id').agg({
            'total_energy_kwh': 'mean',
            'energy_intensity_kwhm2': 'mean'
        }).reset_index()
        energy_summary.columns = ['building_id', 'avg_annual_energy_kwh', 'avg_energy_intensity_kwhm2']
        
        integrated = integrated.merge(energy_summary, on='building_id', how='left')
        
        # Add latest efficiency rating
        latest_ratings = self.energy_data['ratings'].loc[
            self.energy_data['ratings'].groupby('building_id')['rating_year'].idxmax()
        ][['building_id', 'eu_energy_rating', 'energy_performance_index']]
        
        integrated = integrated.merge(latest_ratings, on='building_id', how='left')
        
        return integrated
    
    def create_time_series_dataset(self, building_id):
        """Create a time series dataset for a specific building"""
        print(f"Creating time series dataset for building {building_id}...")
        
        # Get all IoT data for the building
        energy = self.iot_data['energy'][self.iot_data['energy']['building_id'] == building_id].copy()
        environmental = self.iot_data['environmental'][self.iot_data['environmental']['building_id'] == building_id].copy()
        weather = self.iot_data['weather'][self.iot_data['weather']['building_id'] == building_id].copy()
        occupancy = self.iot_data['occupancy'][self.iot_data['occupancy']['building_id'] == building_id].copy()
        
        # Convert timestamps
        for df in [energy, environmental, weather, occupancy]:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Merge all time series data
        time_series = energy[['total_consumption_kwh', 'heating_kwh', 'cooling_kwh']].copy()
        
        time_series = time_series.join(environmental[['co2_ppm', 'temperature_c', 'humidity_percent']])
        time_series = time_series.join(weather[['outdoor_temp_c', 'solar_irradiance_wm2']])
        time_series = time_series.join(occupancy[['occupancy_count', 'activity_level']])
        
        return time_series.dropna()
    
    def analyze_energy_patterns(self):
        """Analyze energy consumption patterns"""
        print("Analyzing energy patterns...")
        
        # Monthly energy consumption by building type
        energy_with_type = self.iot_data['energy'].merge(
            self.building_data['basic_info'][['building_id', 'building_type']], 
            on='building_id'
        )
        
        energy_with_type['month'] = pd.to_datetime(energy_with_type['timestamp']).dt.month
        monthly_consumption = energy_with_type.groupby(['building_type', 'month'])['total_consumption_kwh'].mean().reset_index()
        
        return monthly_consumption
    
    def identify_retrofit_candidates(self, energy_threshold=200):
        """Identify buildings that are good candidates for retrofit"""
        print("Identifying retrofit candidates...")
        
        # Calculate average energy intensity
        avg_energy = self.energy_data['historical'].groupby('building_id')['energy_intensity_kwhm2'].mean().reset_index()
        avg_energy.columns = ['building_id', 'avg_energy_intensity']
        
        # Get building info
        candidates = avg_energy.merge(
            self.building_data['basic_info'][['building_id', 'building_type', 'construction_year']], 
            on='building_id'
        )
        
        # Filter for high energy intensity and older buildings
        retrofit_candidates = candidates[
            (candidates['avg_energy_intensity'] > energy_threshold) & 
            (candidates['construction_year'] < 2000)
        ].sort_values('avg_energy_intensity', ascending=False)
        
        return retrofit_candidates

# Example usage
if __name__ == "__main__":
    integrator = BuildingRetrofitDataIntegrator()
    integrator.load_all_data()
    
    # Create integrated dataset
    integrated_data = integrator.create_integrated_building_dataset()
    print(f"Integrated dataset shape: {integrated_data.shape}")
    
    # Analyze energy patterns
    energy_patterns = integrator.analyze_energy_patterns()
    print("Energy patterns by building type:")
    print(energy_patterns.head())
    
    # Identify retrofit candidates
    candidates = integrator.identify_retrofit_candidates()
    print(f"Found {len(candidates)} retrofit candidates")
    print(candidates.head())
