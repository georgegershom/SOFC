#!/usr/bin/env python3
"""
Data Exploration and Visualization Tool for Contextual & External Data
Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization

This tool provides comprehensive analysis and visualization of the generated datasets.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self, data_path: str = "/workspace/data"):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.load_all_datasets()
    
    def load_all_datasets(self):
        """Load all generated datasets"""
        print("Loading all datasets...")
        
        # Weather & Climate Data
        self.datasets['tmy_data'] = pd.read_csv(self.data_path / 'weather_climate' / 'tmy_data.csv')
        self.datasets['climate_projections'] = pd.read_csv(self.data_path / 'weather_climate' / 'future_climate_projections.csv')
        
        # Economic & Market Data
        self.datasets['energy_prices'] = pd.read_csv(self.data_path / 'economic_market' / 'energy_prices.csv')
        self.datasets['material_costs'] = pd.read_csv(self.data_path / 'economic_market' / 'material_technology_costs.csv')
        self.datasets['labor_costs'] = pd.read_csv(self.data_path / 'economic_market' / 'labor_costs.csv')
        self.datasets['financial_params'] = pd.read_csv(self.data_path / 'economic_market' / 'financial_parameters.csv')
        
        # Geospatial & Regulatory Data
        self.datasets['geospatial'] = pd.read_csv(self.data_path / 'geospatial_regulatory' / 'geospatial_data.csv')
        self.datasets['carbon_intensity'] = pd.read_csv(self.data_path / 'geospatial_regulatory' / 'carbon_intensity.csv')
        self.datasets['building_codes'] = pd.read_csv(self.data_path / 'geospatial_regulatory' / 'building_codes.csv')
        
        print(f"Loaded {len(self.datasets)} datasets successfully!")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("CONTEXTUAL & EXTERNAL DATA SUMMARY REPORT")
        print("Dynamic Digital Twin Framework for Building Retrofit Optimization")
        print("="*80)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper().replace('_', ' ')}")
            print("-" * 50)
            print(f"Records: {len(df):,}")
            print(f"Columns: {len(df.columns)}")
            print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show sample data
            print("\nSample Data:")
            print(df.head(3).to_string())
            
            # Show data types
            print(f"\nData Types:")
            for col, dtype in df.dtypes.items():
                print(f"  {col}: {dtype}")
            
            # Show missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing Values:")
                for col, count in missing[missing > 0].items():
                    print(f"  {col}: {count}")
            else:
                print("\nNo missing values found!")
    
    def analyze_weather_climate_data(self):
        """Analyze weather and climate data"""
        print("\n" + "="*60)
        print("WEATHER & CLIMATE DATA ANALYSIS")
        print("="*60)
        
        # TMY Data Analysis
        tmy = self.datasets['tmy_data']
        print(f"\nTMY Data Analysis:")
        print(f"  Total Records: {len(tmy):,}")
        print(f"  Locations: {tmy['location_id'].nunique()}")
        print(f"  Date Range: {tmy['datetime'].min()} to {tmy['datetime'].max()}")
        
        # Temperature analysis
        temp_stats = tmy['dry_bulb_temperature_c'].describe()
        print(f"\nTemperature Statistics (°C):")
        print(f"  Mean: {temp_stats['mean']:.2f}")
        print(f"  Min: {temp_stats['min']:.2f}")
        print(f"  Max: {temp_stats['max']:.2f}")
        print(f"  Std: {temp_stats['std']:.2f}")
        
        # Solar radiation analysis
        solar_stats = tmy['global_horizontal_irradiance_whm2'].describe()
        print(f"\nSolar Radiation Statistics (Wh/m²):")
        print(f"  Mean: {solar_stats['mean']:.2f}")
        print(f"  Max: {solar_stats['max']:.2f}")
        
        # Climate Projections Analysis
        climate = self.datasets['climate_projections']
        print(f"\nClimate Projections Analysis:")
        print(f"  Total Records: {len(climate):,}")
        print(f"  Scenarios: {climate['scenario'].unique()}")
        print(f"  Year Range: {climate['year'].min()} to {climate['year'].max()}")
        
        # Temperature increase analysis
        temp_increase = climate.groupby('scenario')['temperature_increase_c'].mean()
        print(f"\nAverage Temperature Increase by Scenario:")
        for scenario, increase in temp_increase.items():
            print(f"  {scenario}: {increase:.2f}°C")
    
    def analyze_economic_data(self):
        """Analyze economic and market data"""
        print("\n" + "="*60)
        print("ECONOMIC & MARKET DATA ANALYSIS")
        print("="*60)
        
        # Energy Prices Analysis
        energy = self.datasets['energy_prices']
        print(f"\nEnergy Prices Analysis:")
        print(f"  Total Records: {len(energy):,}")
        print(f"  Regions: {energy['region_id'].nunique()}")
        print(f"  Countries: {energy['country'].nunique()}")
        
        # Price statistics by country
        price_stats = energy.groupby('country')['electricity_price_per_kwh'].agg(['mean', 'min', 'max', 'std'])
        print(f"\nElectricity Prices by Country (USD/kWh):")
        for country, stats in price_stats.iterrows():
            print(f"  {country}: Mean={stats['mean']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        
        # Material Costs Analysis
        materials = self.datasets['material_costs']
        print(f"\nMaterial Costs Analysis:")
        print(f"  Total Records: {len(materials):,}")
        print(f"  Categories: {materials['category'].nunique()}")
        print(f"  Materials: {materials['material_name'].nunique()}")
        
        # Cost trends by category
        cost_trends = materials.groupby(['category', 'year'])['regional_cost'].mean().unstack(level=0)
        print(f"\nCost Trends by Category (USD):")
        for category in materials['category'].unique():
            category_data = materials[materials['category'] == category]
            avg_cost = category_data['regional_cost'].mean()
            print(f"  {category}: ${avg_cost:.2f} per {category_data['unit'].iloc[0]}")
        
        # Labor Costs Analysis
        labor = self.datasets['labor_costs']
        print(f"\nLabor Costs Analysis:")
        print(f"  Total Records: {len(labor):,}")
        print(f"  Activities: {labor['activity'].nunique()}")
        print(f"  Regions: {labor['region_id'].nunique()}")
        
        # Labor cost by activity
        labor_costs = labor.groupby('activity')['hourly_rate_usd'].mean().sort_values(ascending=False)
        print(f"\nAverage Hourly Rates by Activity (USD):")
        for activity, rate in labor_costs.items():
            print(f"  {activity}: ${rate:.2f}")
    
    def analyze_geospatial_data(self):
        """Analyze geospatial and regulatory data"""
        print("\n" + "="*60)
        print("GEOSPATIAL & REGULATORY DATA ANALYSIS")
        print("="*60)
        
        # Geospatial Data Analysis
        geo = self.datasets['geospatial']
        print(f"\nGeospatial Data Analysis:")
        print(f"  Total Records: {len(geo):,}")
        print(f"  Locations: {geo['location_id'].nunique()}")
        print(f"  Countries: {geo['country'].nunique()}")
        
        # Climate zones
        climate_zones = geo['climate_zone'].value_counts()
        print(f"\nClimate Zone Distribution:")
        for zone, count in climate_zones.items():
            print(f"  {zone}: {count}")
        
        # Solar potential analysis
        solar_stats = geo['solar_potential_kwh_per_sqft'].describe()
        print(f"\nSolar Potential Statistics (kWh/sqft/year):")
        print(f"  Mean: {solar_stats['mean']:.2f}")
        print(f"  Min: {solar_stats['min']:.2f}")
        print(f"  Max: {solar_stats['max']:.2f}")
        
        # Carbon Intensity Analysis
        carbon = self.datasets['carbon_intensity']
        print(f"\nCarbon Intensity Analysis:")
        print(f"  Total Records: {len(carbon):,}")
        print(f"  Regions: {carbon['region_id'].nunique()}")
        
        # Carbon intensity by country
        carbon_stats = carbon.groupby('country')['carbon_intensity_kgco2_per_kwh'].agg(['mean', 'min', 'max'])
        print(f"\nCarbon Intensity by Country (kg CO2/kWh):")
        for country, stats in carbon_stats.iterrows():
            print(f"  {country}: Mean={stats['mean']:.4f}, Min={stats['min']:.4f}, Max={stats['max']:.4f}")
        
        # Building Codes Analysis
        codes = self.datasets['building_codes']
        print(f"\nBuilding Codes Analysis:")
        print(f"  Total Records: {len(codes):,}")
        print(f"  Energy Codes: {codes['energy_code'].nunique()}")
        
        # Energy code requirements
        print(f"\nEnergy Code Requirements:")
        for _, row in codes.iterrows():
            print(f"  {row['city']}, {row['country']}: {row['energy_code']} {row['energy_code_version']}")
            print(f"    Wall U-value: {row['u_value_walls_max']} W/m²K")
            print(f"    Window U-value: {row['u_value_windows_max']} W/m²K")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING DATA VISUALIZATIONS")
        print("="*60)
        
        # Create output directory for plots
        plot_dir = self.data_path / 'visualizations'
        plot_dir.mkdir(exist_ok=True)
        
        # 1. Weather Data Visualizations
        self._plot_weather_data(plot_dir)
        
        # 2. Economic Data Visualizations
        self._plot_economic_data(plot_dir)
        
        # 3. Geospatial Data Visualizations
        self._plot_geospatial_data(plot_dir)
        
        # 4. Carbon Intensity Visualizations
        self._plot_carbon_intensity(plot_dir)
        
        print(f"\nAll visualizations saved to: {plot_dir}")
    
    def _plot_weather_data(self, plot_dir):
        """Create weather data visualizations"""
        tmy = self.datasets['tmy_data']
        
        # Temperature distribution by location
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        for location in tmy['location_id'].unique()[:5]:  # Top 5 locations
            location_data = tmy[tmy['location_id'] == location]
            plt.hist(location_data['dry_bulb_temperature_c'], alpha=0.7, label=location, bins=30)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Frequency')
        plt.title('Temperature Distribution by Location')
        plt.legend()
        
        # Solar radiation by month
        plt.subplot(2, 2, 2)
        monthly_solar = tmy.groupby('month')['global_horizontal_irradiance_whm2'].mean()
        plt.plot(monthly_solar.index, monthly_solar.values, marker='o')
        plt.xlabel('Month')
        plt.ylabel('Solar Radiation (Wh/m²)')
        plt.title('Average Solar Radiation by Month')
        plt.grid(True)
        
        # Temperature vs Solar Radiation
        plt.subplot(2, 2, 3)
        sample_data = tmy.sample(10000)  # Sample for performance
        plt.scatter(sample_data['dry_bulb_temperature_c'], 
                   sample_data['global_horizontal_irradiance_whm2'], 
                   alpha=0.5, s=1)
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Solar Radiation (Wh/m²)')
        plt.title('Temperature vs Solar Radiation')
        
        # Climate projections
        plt.subplot(2, 2, 4)
        climate = self.datasets['climate_projections']
        for scenario in climate['scenario'].unique():
            scenario_data = climate[climate['scenario'] == scenario]
            yearly_temp = scenario_data.groupby('year')['projected_temperature_c'].mean()
            plt.plot(yearly_temp.index, yearly_temp.values, marker='o', label=scenario)
        plt.xlabel('Year')
        plt.ylabel('Projected Temperature (°C)')
        plt.title('Climate Projections by Scenario')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'weather_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_economic_data(self, plot_dir):
        """Create economic data visualizations"""
        energy = self.datasets['energy_prices']
        materials = self.datasets['material_costs']
        labor = self.datasets['labor_costs']
        
        plt.figure(figsize=(15, 12))
        
        # Energy prices by country
        plt.subplot(2, 3, 1)
        price_by_country = energy.groupby('country')['electricity_price_per_kwh'].mean().sort_values(ascending=False)
        plt.bar(range(len(price_by_country)), price_by_country.values)
        plt.xticks(range(len(price_by_country)), price_by_country.index, rotation=45)
        plt.ylabel('Electricity Price (USD/kWh)')
        plt.title('Average Electricity Prices by Country')
        
        # Material costs by category
        plt.subplot(2, 3, 2)
        material_costs = materials.groupby('category')['regional_cost'].mean().sort_values(ascending=False)
        plt.bar(range(len(material_costs)), material_costs.values)
        plt.xticks(range(len(material_costs)), material_costs.index, rotation=45)
        plt.ylabel('Average Cost (USD)')
        plt.title('Material Costs by Category')
        
        # Labor costs by activity
        plt.subplot(2, 3, 3)
        labor_costs = labor.groupby('activity')['hourly_rate_usd'].mean().sort_values(ascending=False)
        plt.bar(range(len(labor_costs)), labor_costs.values)
        plt.xticks(range(len(labor_costs)), [act.replace('_', ' ').title() for act in labor_costs.index], rotation=45)
        plt.ylabel('Hourly Rate (USD)')
        plt.title('Labor Costs by Activity')
        
        # Energy price trends over time
        plt.subplot(2, 3, 4)
        energy['datetime'] = pd.to_datetime(energy['datetime'])
        energy['year'] = energy['datetime'].dt.year
        yearly_prices = energy.groupby('year')['electricity_price_per_kwh'].mean()
        plt.plot(yearly_prices.index, yearly_prices.values, marker='o')
        plt.xlabel('Year')
        plt.ylabel('Electricity Price (USD/kWh)')
        plt.title('Energy Price Trends')
        plt.grid(True)
        
        # Material cost trends
        plt.subplot(2, 3, 5)
        for category in materials['category'].unique()[:3]:  # Top 3 categories
            category_data = materials[materials['category'] == category]
            yearly_costs = category_data.groupby('year')['regional_cost'].mean()
            plt.plot(yearly_costs.index, yearly_costs.values, marker='o', label=category)
        plt.xlabel('Year')
        plt.ylabel('Cost (USD)')
        plt.title('Material Cost Trends')
        plt.legend()
        plt.grid(True)
        
        # TOU pricing patterns
        plt.subplot(2, 3, 6)
        tou_prices = energy.groupby('tou_period')['electricity_price_per_kwh'].mean()
        plt.bar(range(len(tou_prices)), tou_prices.values)
        plt.xticks(range(len(tou_prices)), tou_prices.index, rotation=45)
        plt.ylabel('Electricity Price (USD/kWh)')
        plt.title('Time-of-Use Pricing Patterns')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'economic_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_geospatial_data(self, plot_dir):
        """Create geospatial data visualizations"""
        geo = self.datasets['geospatial']
        
        plt.figure(figsize=(15, 10))
        
        # Climate zones
        plt.subplot(2, 2, 1)
        climate_zones = geo['climate_zone'].value_counts()
        plt.pie(climate_zones.values, labels=climate_zones.index, autopct='%1.1f%%')
        plt.title('Distribution of Climate Zones')
        
        # Solar potential by location
        plt.subplot(2, 2, 2)
        solar_by_location = geo.set_index('location_id')['solar_potential_kwh_per_sqft'].sort_values(ascending=True)
        plt.barh(range(len(solar_by_location)), solar_by_location.values)
        plt.yticks(range(len(solar_by_location)), solar_by_location.index)
        plt.xlabel('Solar Potential (kWh/sqft/year)')
        plt.title('Solar Potential by Location')
        
        # Urban context distribution
        plt.subplot(2, 2, 3)
        urban_context = geo['urban_context'].value_counts()
        plt.bar(range(len(urban_context)), urban_context.values)
        plt.xticks(range(len(urban_context)), urban_context.index, rotation=45)
        plt.ylabel('Number of Locations')
        plt.title('Urban Context Distribution')
        
        # Air quality vs noise level
        plt.subplot(2, 2, 4)
        plt.scatter(geo['air_quality_index'], geo['noise_level_db'], alpha=0.7)
        plt.xlabel('Air Quality Index')
        plt.ylabel('Noise Level (dB)')
        plt.title('Air Quality vs Noise Level')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'geospatial_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_carbon_intensity(self, plot_dir):
        """Create carbon intensity visualizations"""
        carbon = self.datasets['carbon_intensity']
        
        plt.figure(figsize=(15, 10))
        
        # Carbon intensity by country
        plt.subplot(2, 2, 1)
        carbon_by_country = carbon.groupby('country')['carbon_intensity_kgco2_per_kwh'].mean().sort_values(ascending=False)
        plt.bar(range(len(carbon_by_country)), carbon_by_country.values)
        plt.xticks(range(len(carbon_by_country)), carbon_by_country.index, rotation=45)
        plt.ylabel('Carbon Intensity (kg CO2/kWh)')
        plt.title('Average Carbon Intensity by Country')
        
        # Carbon intensity trends over time
        plt.subplot(2, 2, 2)
        carbon['datetime'] = pd.to_datetime(carbon['datetime'])
        carbon['year'] = carbon['datetime'].dt.year
        yearly_carbon = carbon.groupby('year')['carbon_intensity_kgco2_per_kwh'].mean()
        plt.plot(yearly_carbon.index, yearly_carbon.values, marker='o')
        plt.xlabel('Year')
        plt.ylabel('Carbon Intensity (kg CO2/kWh)')
        plt.title('Carbon Intensity Trends')
        plt.grid(True)
        
        # Renewable energy percentage
        plt.subplot(2, 2, 3)
        renewable_by_country = carbon.groupby('country')['renewable_percentage'].mean().sort_values(ascending=False)
        plt.bar(range(len(renewable_by_country)), renewable_by_country.values)
        plt.xticks(range(len(renewable_by_country)), renewable_by_country.index, rotation=45)
        plt.ylabel('Renewable Energy (%)')
        plt.title('Renewable Energy Percentage by Country')
        
        # Grid mix composition
        plt.subplot(2, 2, 4)
        grid_mix = carbon.groupby('country')[['coal_percentage', 'natural_gas_percentage', 
                                           'nuclear_percentage', 'solar_percentage', 
                                           'wind_percentage', 'hydro_percentage']].mean()
        
        # Plot stacked bar chart
        bottom = np.zeros(len(grid_mix))
        colors = ['brown', 'orange', 'purple', 'yellow', 'green', 'blue']
        labels = ['Coal', 'Natural Gas', 'Nuclear', 'Solar', 'Wind', 'Hydro']
        
        for i, (col, color) in enumerate(zip(grid_mix.columns, colors)):
            plt.bar(range(len(grid_mix)), grid_mix[col], bottom=bottom, color=color, label=labels[i])
            bottom += grid_mix[col]
        
        plt.xticks(range(len(grid_mix)), grid_mix.index, rotation=45)
        plt.ylabel('Percentage (%)')
        plt.title('Grid Mix Composition by Country')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'carbon_intensity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_data_integration_guide(self):
        """Generate a guide for integrating the data with the Digital Twin Framework"""
        guide = """
# Data Integration Guide for Dynamic Digital Twin Framework

## Overview
This guide explains how to integrate the generated contextual and external data with your Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization.

## Dataset Structure

### 1. Weather & Climate Data
- **tmy_data.csv**: Typical Meteorological Year data with hourly weather parameters
- **future_climate_projections.csv**: Climate projections for different IPCC scenarios

**Key Fields:**
- `location_id`, `latitude`, `longitude`: Geographic identification
- `datetime`: Timestamp for temporal analysis
- `dry_bulb_temperature_c`: Temperature for thermal modeling
- `global_horizontal_irradiance_whm2`: Solar radiation for PV potential
- `relative_humidity_pct`: Humidity for comfort analysis

**Integration Points:**
- Use for building energy simulation baseline
- Input to HVAC load calculations
- Solar PV generation modeling
- Climate change impact assessment

### 2. Economic & Market Data
- **energy_prices.csv**: Time-varying energy pricing including TOU rates
- **material_technology_costs.csv**: Material and technology cost data
- **labor_costs.csv**: Labor cost data for different activities
- **financial_parameters.csv**: Financial parameters and incentives

**Key Fields:**
- `electricity_price_per_kwh`: For energy cost calculations
- `demand_charge_per_kw`: For peak demand optimization
- `regional_cost`: Material costs for LCC analysis
- `hourly_rate_usd`: Labor costs for installation
- `discount_rate_pct`: For NPV calculations

**Integration Points:**
- Life-cycle cost analysis
- ROI calculations for retrofit measures
- Optimization objective functions
- Financial feasibility assessment

### 3. Geospatial & Regulatory Data
- **geospatial_data.csv**: Location-specific environmental data
- **carbon_intensity.csv**: Grid carbon intensity factors
- **building_codes.csv**: Local building code requirements

**Key Fields:**
- `solar_potential_kwh_per_sqft`: For renewable energy sizing
- `carbon_intensity_kgco2_per_kwh`: For emissions calculations
- `u_value_walls_max`: Code compliance constraints
- `emissions_target_2030_pct`: Regulatory targets

**Integration Points:**
- Renewable energy potential assessment
- Carbon footprint calculations
- Code compliance verification
- Regulatory constraint modeling

## Integration Architecture

### Real-Time Data Integration
1. **IoT Sensor Data**: Integrate with building sensors for real-time conditions
2. **Weather API**: Connect to live weather data for current conditions
3. **Energy Market Data**: Real-time pricing from utility APIs
4. **Grid Data**: Live carbon intensity from grid operators

### Data Processing Pipeline
1. **Data Ingestion**: Load historical and real-time data
2. **Data Validation**: Ensure data quality and consistency
3. **Feature Engineering**: Create derived features for ML models
4. **Data Fusion**: Combine multiple data sources
5. **Model Input**: Feed processed data to optimization models

### Machine Learning Integration
1. **Weather Forecasting**: Use TMY data to train weather prediction models
2. **Energy Price Prediction**: Predict future energy costs
3. **Demand Forecasting**: Predict building energy demand
4. **Optimization**: Use DRL for multi-objective optimization

## Usage Examples

### Python Integration
```python
import pandas as pd
import numpy as np

# Load datasets
tmy_data = pd.read_csv('data/weather_climate/tmy_data.csv')
energy_prices = pd.read_csv('data/economic_market/energy_prices.csv')
carbon_intensity = pd.read_csv('data/geospatial_regulatory/carbon_intensity.csv')

# Filter data for specific location and time
location_data = tmy_data[tmy_data['location_id'] == 'New_York_USA']
energy_data = energy_prices[energy_prices['region_id'] == 'North_America_East']

# Calculate energy costs
energy_data['total_cost'] = (energy_data['electricity_price_per_kwh'] + 
                           energy_data['demand_charge_per_kw'] * 0.1)

# Calculate carbon emissions
carbon_data = carbon_intensity[carbon_intensity['region_id'] == 'North_America_East']
emissions = energy_data['electricity_price_per_kwh'] * carbon_data['carbon_intensity_kgco2_per_kwh']
```

### Optimization Model Integration
```python
# Example objective function incorporating multiple data sources
def objective_function(retrofit_measures, weather_data, energy_prices, carbon_intensity):
    # Calculate energy savings
    energy_savings = calculate_energy_savings(retrofit_measures, weather_data)
    
    # Calculate costs
    material_costs = get_material_costs(retrofit_measures)
    energy_costs = energy_savings * energy_prices['electricity_price_per_kwh']
    
    # Calculate emissions
    emissions = energy_savings * carbon_intensity['carbon_intensity_kgco2_per_kwh']
    
    # Multi-objective optimization
    return {
        'cost': material_costs - energy_costs,
        'emissions': -emissions,  # Negative for maximization
        'comfort': calculate_comfort_improvement(retrofit_measures, weather_data)
    }
```

## Data Updates and Maintenance

### Regular Updates
- **Weather Data**: Update annually with new TMY data
- **Energy Prices**: Update monthly with current market rates
- **Material Costs**: Update quarterly with market changes
- **Carbon Intensity**: Update daily with grid data

### Data Quality Monitoring
- Implement data validation checks
- Monitor for missing or anomalous data
- Set up alerts for data quality issues
- Regular data quality reports

## Performance Optimization

### Data Storage
- Use efficient data formats (Parquet, HDF5)
- Implement data compression
- Use appropriate indexing strategies
- Consider data partitioning by location/time

### Processing Optimization
- Use vectorized operations (NumPy, Pandas)
- Implement parallel processing for large datasets
- Use caching for frequently accessed data
- Optimize database queries

## Next Steps

1. **Data Validation**: Run validation scripts to ensure data quality
2. **Integration Testing**: Test data integration with your framework
3. **Performance Tuning**: Optimize data processing for your use case
4. **Real-time Integration**: Connect to live data sources
5. **Model Training**: Use data to train your ML models
6. **Optimization**: Implement multi-objective optimization algorithms

## Support and Documentation

- Dataset schemas are available in the validation results
- Sample integration code is provided in the examples
- Regular updates will be provided for new data
- Contact support for integration assistance
"""
        
        with open(self.data_path / 'integrated' / 'data_integration_guide.md', 'w') as f:
            f.write(guide)
        
        print(f"\nData integration guide saved to: {self.data_path / 'integrated' / 'data_integration_guide.md'}")

def main():
    """Main function to run data exploration"""
    print("Starting Data Exploration and Analysis")
    print("="*50)
    
    # Initialize explorer
    explorer = DataExplorer()
    
    # Generate comprehensive analysis
    explorer.generate_summary_report()
    explorer.analyze_weather_climate_data()
    explorer.analyze_economic_data()
    explorer.analyze_geospatial_data()
    
    # Create visualizations
    explorer.create_visualizations()
    
    # Generate integration guide
    explorer.generate_data_integration_guide()
    
    print("\n" + "="*50)
    print("Data exploration and analysis complete!")
    print("Check the visualizations and integration guide for detailed insights.")

if __name__ == "__main__":
    main()