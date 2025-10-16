#!/usr/bin/env python3
"""
IoT & Real-Time Monitoring Dataset Generator
For Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization

This module generates comprehensive building sensor data including:
- Energy Consumption (whole-building and sub-metered)
- Indoor Environmental Quality (IEQ)
- Occupancy & Usage Patterns
- External Weather Conditions
- Building System Operation (HVAC, setpoints)

Author: AI Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pytz
from scipy import stats
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

class IoTDatasetGenerator:
    """
    Comprehensive IoT dataset generator for building digital twin applications.
    Generates realistic sensor data with proper correlations and seasonal variations.
    """
    
    def __init__(self, start_date='2023-01-01', end_date='2023-12-31', 
                 timezone='US/Eastern', building_area=50000):  # sq ft
        """
        Initialize the dataset generator.
        
        Args:
            start_date: Start date for the dataset (YYYY-MM-DD)
            end_date: End date for the dataset (YYYY-MM-DD)
            timezone: Timezone for the building location
            building_area: Building area in square feet
        """
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.timezone = pytz.timezone(timezone)
        self.building_area = building_area
        
        # Create time index with 15-minute intervals
        self.time_index = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='15T',
            tz=self.timezone
        )
        
        # Building characteristics
        self.num_floors = 8
        self.num_zones_per_floor = 4
        self.total_zones = self.num_floors * self.num_zones_per_floor
        
        # Initialize random state for reproducibility
        np.random.seed(42)
        
    def generate_weather_data(self):
        """
        Generate realistic weather data including temperature, humidity, 
        solar irradiance, wind, and precipitation.
        """
        print("Generating weather data...")
        
        # Base temperature profile (seasonal variation)
        days_in_year = 365
        # Convert timezone-aware timestamps to naive for calculation
        time_index_naive = self.time_index.tz_localize(None)
        start_date_naive = self.start_date.tz_localize(None) if self.start_date.tzinfo else self.start_date
        day_of_year = np.array([(d - start_date_naive).days for d in time_index_naive])
        
        # Seasonal temperature variation (Fahrenheit)
        base_temp = 50 + 25 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        
        # Add daily temperature cycle
        hour_of_day = self.time_index.hour + self.time_index.minute / 60
        daily_cycle = 8 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
        
        # Add random weather variations
        weather_noise = np.random.normal(0, 5, len(self.time_index))
        
        # Temperature with heat island effect (urban buildings are warmer)
        temperature = base_temp + daily_cycle + weather_noise + 2
        
        # Relative humidity (inverse relationship with temperature)
        humidity = 60 - 0.3 * (temperature - 50) + np.random.normal(0, 10, len(self.time_index))
        humidity = np.clip(humidity, 20, 95)
        
        # Solar irradiance (W/m²)
        solar_angle = np.sin(2 * np.pi * day_of_year / 365) * np.sin(np.pi * hour_of_day / 12)
        solar_irradiance = np.maximum(0, 800 * solar_angle + np.random.normal(0, 50, len(self.time_index)))
        
        # Wind speed (m/s)
        wind_speed = 3 + 2 * np.sin(2 * np.pi * day_of_year / 365) + np.random.exponential(2, len(self.time_index))
        
        # Wind direction (degrees)
        wind_direction = np.random.uniform(0, 360, len(self.time_index))
        
        # Precipitation (mm)
        precipitation = np.random.exponential(0.5, len(self.time_index))
        # Reduce precipitation during summer
        summer_mask = (day_of_year > 150) & (day_of_year < 250)
        precipitation[summer_mask] *= 0.3
        
        weather_data = pd.DataFrame({
            'timestamp': self.time_index,
            'outdoor_temperature_f': temperature,
            'outdoor_humidity_rh': humidity,
            'solar_irradiance_wm2': solar_irradiance,
            'wind_speed_ms': wind_speed,
            'wind_direction_deg': wind_direction,
            'precipitation_mm': precipitation
        })
        
        return weather_data
    
    def generate_energy_consumption(self, weather_data):
        """
        Generate energy consumption data for whole-building and sub-metered systems.
        """
        print("Generating energy consumption data...")
        
        # Base energy consumption (kWh)
        base_consumption = 2000  # Base load in kWh per 15-min interval
        
        # HVAC energy consumption (correlated with temperature difference)
        temp_diff = np.abs(weather_data['outdoor_temperature_f'] - 72)  # 72F target
        hvac_consumption = 800 + 20 * temp_diff + np.random.normal(0, 100, len(self.time_index))
        hvac_consumption = np.maximum(200, hvac_consumption)
        
        # Lighting energy consumption (higher during business hours)
        hour_of_day = self.time_index.hour
        business_hours = ((hour_of_day >= 7) & (hour_of_day <= 19)) | (hour_of_day == 0)
        lighting_consumption = np.where(business_hours, 300, 50) + np.random.normal(0, 30, len(self.time_index))
        # Ensure lighting consumption is non-negative
        lighting_consumption = np.maximum(0, lighting_consumption)
        
        # Plug loads (computers, equipment)
        plug_consumption = 400 + 100 * business_hours + np.random.normal(0, 50, len(self.time_index))
        
        # Water heating
        water_heating = 200 + np.random.normal(0, 40, len(self.time_index))
        
        # Total building consumption
        total_consumption = hvac_consumption + lighting_consumption + plug_consumption + water_heating + base_consumption
        
        # Add some noise and realistic variations
        total_consumption += np.random.normal(0, 50, len(self.time_index))
        total_consumption = np.maximum(1000, total_consumption)
        
        energy_data = pd.DataFrame({
            'timestamp': self.time_index,
            'total_electricity_kwh': total_consumption,
            'hvac_electricity_kwh': hvac_consumption,
            'lighting_electricity_kwh': lighting_consumption,
            'plug_loads_kwh': plug_consumption,
            'water_heating_kwh': water_heating,
            'base_load_kwh': base_consumption
        })
        
        return energy_data
    
    def generate_ieq_data(self, weather_data, energy_data):
        """
        Generate Indoor Environmental Quality data including temperature, humidity,
        air quality, lighting, and acoustics.
        """
        print("Generating IEQ data...")
        
        # Indoor temperature (zones)
        zone_temps = []
        for zone in range(self.total_zones):
            # Base temperature influenced by outdoor temperature and HVAC
            base_temp = 0.3 * weather_data['outdoor_temperature_f'] + 0.7 * 72
            
            # HVAC influence
            hvac_influence = 0.1 * energy_data['hvac_electricity_kwh']
            
            # Zone-specific variations
            zone_variation = np.random.normal(0, 2, len(self.time_index))
            
            zone_temp = base_temp + hvac_influence + zone_variation
            zone_temps.append(zone_temp)
        
        # Indoor humidity (correlated with outdoor humidity and HVAC)
        base_humidity = 0.4 * weather_data['outdoor_humidity_rh'] + 0.6 * 45
        hvac_humidity_effect = -0.05 * energy_data['hvac_electricity_kwh']
        indoor_humidity = base_humidity + hvac_humidity_effect + np.random.normal(0, 5, len(self.time_index))
        indoor_humidity = np.clip(indoor_humidity, 20, 70)
        
        # CO2 levels (ppm) - higher during business hours and with more people
        hour_of_day = self.time_index.hour
        business_hours = (hour_of_day >= 8) & (hour_of_day <= 18)
        base_co2 = 400 + 200 * business_hours + np.random.normal(0, 50, len(self.time_index))
        base_co2 = np.clip(base_co2, 350, 1500)
        
        # PM2.5 levels (μg/m³)
        pm25 = 10 + 0.1 * weather_data['precipitation_mm'] + np.random.exponential(2, len(self.time_index))
        pm25 = np.clip(pm25, 5, 50)
        
        # TVOC levels (ppb)
        tvoc = 50 + 20 * business_hours + np.random.exponential(10, len(self.time_index))
        tvoc = np.clip(tvoc, 20, 200)
        
        # Illuminance levels (lux)
        # Natural lighting influenced by solar irradiance
        natural_light = 0.1 * weather_data['solar_irradiance_wm2']
        # Artificial lighting during business hours
        artificial_light = np.where(business_hours, 500, 50)
        illuminance = natural_light + artificial_light + np.random.normal(0, 50, len(self.time_index))
        illuminance = np.maximum(0, illuminance)
        
        # Noise levels (dB)
        base_noise = 40 + 20 * business_hours + np.random.normal(0, 5, len(self.time_index))
        base_noise = np.clip(base_noise, 30, 80)
        
        ieq_data = pd.DataFrame({
            'timestamp': self.time_index,
            'indoor_humidity_rh': indoor_humidity,
            'co2_ppm': base_co2,
            'pm25_ugm3': pm25,
            'tvoc_ppb': tvoc,
            'illuminance_lux': illuminance,
            'noise_level_db': base_noise
        })
        
        # Add zone-specific temperature data
        for i, zone_temp in enumerate(zone_temps):
            ieq_data[f'zone_{i+1}_temperature_f'] = zone_temp
        
        return ieq_data
    
    def generate_occupancy_data(self, ieq_data):
        """
        Generate occupancy and usage pattern data.
        """
        print("Generating occupancy data...")
        
        # Business hours pattern
        hour_of_day = self.time_index.hour
        day_of_week = self.time_index.dayofweek
        
        # Weekday vs weekend patterns
        weekday_mask = day_of_week < 5
        weekend_mask = day_of_week >= 5
        
        # Peak occupancy during business hours
        business_hours = (hour_of_day >= 8) & (hour_of_day <= 18)
        
        # Base occupancy
        base_occupancy = np.zeros(len(self.time_index))
        base_occupancy[weekday_mask & business_hours] = np.random.poisson(150, np.sum(weekday_mask & business_hours))
        base_occupancy[weekday_mask & ~business_hours] = np.random.poisson(20, np.sum(weekday_mask & ~business_hours))
        base_occupancy[weekend_mask] = np.random.poisson(10, np.sum(weekend_mask))
        
        # Space utilization (percentage)
        space_utilization = np.minimum(100, base_occupancy * 0.6 + np.random.normal(0, 10, len(self.time_index)))
        space_utilization = np.maximum(0, space_utilization)
        
        # Window operation (binary: 0=closed, 1=open)
        # More likely to be open during moderate temperatures
        temp_comfort = (ieq_data['zone_1_temperature_f'] > 65) & (ieq_data['zone_1_temperature_f'] < 80)
        window_open_prob = np.where(temp_comfort, 0.3, 0.1)
        window_operation = np.random.binomial(1, window_open_prob)
        
        # Blind operation (0=closed, 1=open)
        # More likely to be open during low solar irradiance
        blind_open_prob = np.where(ieq_data['illuminance_lux'] < 300, 0.7, 0.3)
        blind_operation = np.random.binomial(1, blind_open_prob)
        
        occupancy_data = pd.DataFrame({
            'timestamp': self.time_index,
            'occupant_count': base_occupancy,
            'space_utilization_pct': space_utilization,
            'window_operation': window_operation,
            'blind_operation': blind_operation
        })
        
        return occupancy_data
    
    def generate_hvac_data(self, weather_data, energy_data, ieq_data):
        """
        Generate HVAC system operation data.
        """
        print("Generating HVAC system data...")
        
        # Supply air temperature (F)
        target_temp = 72
        supply_temp = target_temp + np.random.normal(0, 2, len(self.time_index))
        
        # Return air temperature (influenced by zone temperatures)
        avg_zone_temp = np.mean([ieq_data[f'zone_{i+1}_temperature_f'] for i in range(min(4, self.total_zones))], axis=0)
        return_temp = avg_zone_temp + np.random.normal(0, 1, len(self.time_index))
        # Ensure return air temperature is within reasonable range
        return_temp = np.clip(return_temp, 65, 85)
        
        # Damper positions (0-100%)
        # More open when temperature difference is large
        temp_diff = np.abs(weather_data['outdoor_temperature_f'] - target_temp)
        damper_position = np.minimum(100, 30 + 0.5 * temp_diff + np.random.normal(0, 10, len(self.time_index)))
        damper_position = np.maximum(0, damper_position)
        
        # Fan speed (0-100%)
        fan_speed = 50 + 0.3 * energy_data['hvac_electricity_kwh'] + np.random.normal(0, 10, len(self.time_index))
        fan_speed = np.clip(fan_speed, 20, 100)
        
        # Valve positions (0-100%)
        valve_position = 40 + 0.2 * energy_data['hvac_electricity_kwh'] + np.random.normal(0, 5, len(self.time_index))
        valve_position = np.clip(valve_position, 0, 100)
        
        # Chiller status (0=off, 1=on)
        chiller_status = (weather_data['outdoor_temperature_f'] > 75).astype(int)
        
        # Boiler status (0=off, 1=on)
        boiler_status = (weather_data['outdoor_temperature_f'] < 65).astype(int)
        
        # Setpoints
        heating_setpoint = 70 + np.random.normal(0, 1, len(self.time_index))
        cooling_setpoint = 74 + np.random.normal(0, 1, len(self.time_index))
        # Ensure setpoints are within reasonable ranges
        heating_setpoint = np.clip(heating_setpoint, 65, 75)
        cooling_setpoint = np.clip(cooling_setpoint, 72, 80)
        
        hvac_data = pd.DataFrame({
            'timestamp': self.time_index,
            'supply_air_temp_f': supply_temp,
            'return_air_temp_f': return_temp,
            'damper_position_pct': damper_position,
            'fan_speed_pct': fan_speed,
            'valve_position_pct': valve_position,
            'chiller_status': chiller_status,
            'boiler_status': boiler_status,
            'heating_setpoint_f': heating_setpoint,
            'cooling_setpoint_f': cooling_setpoint
        })
        
        return hvac_data
    
    def generate_complete_dataset(self):
        """
        Generate the complete IoT dataset with all sensor data.
        """
        print("Starting complete IoT dataset generation...")
        print(f"Generating data from {self.start_date} to {self.end_date}")
        print(f"Total data points: {len(self.time_index)}")
        
        # Generate weather data first (needed for other calculations)
        weather_data = self.generate_weather_data()
        
        # Generate energy consumption data
        energy_data = self.generate_energy_consumption(weather_data)
        
        # Generate IEQ data
        ieq_data = self.generate_ieq_data(weather_data, energy_data)
        
        # Generate occupancy data
        occupancy_data = self.generate_occupancy_data(ieq_data)
        
        # Generate HVAC data
        hvac_data = self.generate_hvac_data(weather_data, energy_data, ieq_data)
        
        # Combine all data
        complete_dataset = weather_data.copy()
        
        # Merge energy data
        complete_dataset = complete_dataset.merge(energy_data, on='timestamp', how='left')
        
        # Merge IEQ data
        complete_dataset = complete_dataset.merge(ieq_data, on='timestamp', how='left')
        
        # Merge occupancy data
        complete_dataset = complete_dataset.merge(occupancy_data, on='timestamp', how='left')
        
        # Merge HVAC data
        complete_dataset = complete_dataset.merge(hvac_data, on='timestamp', how='left')
        
        print(f"Dataset generation complete! Shape: {complete_dataset.shape}")
        
        return complete_dataset
    
    def save_dataset(self, dataset, output_dir='iot_dataset'):
        """
        Save the dataset in multiple formats.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving dataset to {output_dir}/")
        
        # Create a copy for Excel export with timezone-naive timestamps
        dataset_excel = dataset.copy()
        dataset_excel['timestamp'] = dataset_excel['timestamp'].dt.tz_localize(None)
        
        # Save as CSV
        dataset.to_csv(f'{output_dir}/iot_building_dataset.csv', index=False)
        print("✓ Saved as CSV")
        
        # Save as Parquet (more efficient)
        dataset.to_parquet(f'{output_dir}/iot_building_dataset.parquet', index=False)
        print("✓ Saved as Parquet")
        
        # Save as Excel (for easy viewing)
        with pd.ExcelWriter(f'{output_dir}/iot_building_dataset.xlsx', engine='openpyxl') as writer:
            dataset_excel.to_excel(writer, sheet_name='Complete_Dataset', index=False)
            
            # Create separate sheets for different data categories
            weather_cols = [col for col in dataset.columns if any(x in col.lower() for x in ['outdoor', 'solar', 'wind', 'precipitation'])]
            energy_cols = [col for col in dataset.columns if any(x in col.lower() for x in ['electricity', 'kwh', 'energy'])]
            ieq_cols = [col for col in dataset.columns if any(x in col.lower() for x in ['indoor', 'co2', 'pm', 'tvoc', 'illuminance', 'noise', 'zone_', 'humidity'])]
            occupancy_cols = [col for col in dataset.columns if any(x in col.lower() for x in ['occupant', 'utilization', 'window', 'blind'])]
            hvac_cols = [col for col in dataset.columns if any(x in col.lower() for x in ['supply', 'return', 'damper', 'fan', 'valve', 'chiller', 'boiler', 'setpoint'])]
            
            dataset_excel[['timestamp'] + weather_cols].to_excel(writer, sheet_name='Weather_Data', index=False)
            dataset_excel[['timestamp'] + energy_cols].to_excel(writer, sheet_name='Energy_Data', index=False)
            dataset_excel[['timestamp'] + ieq_cols].to_excel(writer, sheet_name='IEQ_Data', index=False)
            dataset_excel[['timestamp'] + occupancy_cols].to_excel(writer, sheet_name='Occupancy_Data', index=False)
            dataset_excel[['timestamp'] + hvac_cols].to_excel(writer, sheet_name='HVAC_Data', index=False)
        
        print("✓ Saved as Excel with separate sheets")
        
        # Save metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'timezone': str(self.timezone),
            'building_area_sqft': self.building_area,
            'total_data_points': len(dataset),
            'sampling_frequency': '15 minutes',
            'columns': list(dataset.columns),
            'data_categories': {
                'weather': weather_cols,
                'energy': energy_cols,
                'ieq': ieq_cols,
                'occupancy': occupancy_cols,
                'hvac': hvac_cols
            }
        }
        
        import json
        with open(f'{output_dir}/dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✓ Saved metadata")
        
        return output_dir
    
    def create_visualizations(self, dataset, output_dir='iot_dataset'):
        """
        Create comprehensive visualizations of the generated dataset.
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig_size = (15, 10)
        
        # 1. Energy consumption over time
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Energy Consumption Analysis', fontsize=16)
        
        # Total electricity consumption
        axes[0, 0].plot(dataset['timestamp'], dataset['total_electricity_kwh'], alpha=0.7)
        axes[0, 0].set_title('Total Electricity Consumption')
        axes[0, 0].set_ylabel('kWh')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # HVAC vs outdoor temperature
        axes[0, 1].scatter(dataset['outdoor_temperature_f'], dataset['hvac_electricity_kwh'], alpha=0.5)
        axes[0, 1].set_title('HVAC Consumption vs Outdoor Temperature')
        axes[0, 1].set_xlabel('Outdoor Temperature (°F)')
        axes[0, 1].set_ylabel('HVAC Electricity (kWh)')
        
        # Daily energy pattern
        daily_energy = dataset.groupby(dataset['timestamp'].dt.hour)['total_electricity_kwh'].mean()
        axes[1, 0].plot(daily_energy.index, daily_energy.values, marker='o')
        axes[1, 0].set_title('Average Daily Energy Pattern')
        axes[1, 0].set_xlabel('Hour of Day')
        axes[1, 0].set_ylabel('Average Energy (kWh)')
        
        # Monthly energy consumption
        monthly_energy = dataset.groupby(dataset['timestamp'].dt.month)['total_electricity_kwh'].sum()
        axes[1, 1].bar(monthly_energy.index, monthly_energy.values)
        axes[1, 1].set_title('Monthly Energy Consumption')
        axes[1, 1].set_xlabel('Month')
        axes[1, 1].set_ylabel('Total Energy (kWh)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Indoor Environmental Quality
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Indoor Environmental Quality Analysis', fontsize=16)
        
        # Temperature comparison
        axes[0, 0].plot(dataset['timestamp'], dataset['outdoor_temperature_f'], label='Outdoor', alpha=0.7)
        axes[0, 0].plot(dataset['timestamp'], dataset['zone_1_temperature_f'], label='Zone 1', alpha=0.7)
        axes[0, 0].set_title('Temperature Comparison')
        axes[0, 0].set_ylabel('Temperature (°F)')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # CO2 levels
        axes[0, 1].plot(dataset['timestamp'], dataset['co2_ppm'], alpha=0.7)
        axes[0, 1].set_title('CO2 Levels')
        axes[0, 1].set_ylabel('CO2 (ppm)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Humidity
        axes[1, 0].plot(dataset['timestamp'], dataset['indoor_humidity_rh'], alpha=0.7)
        axes[1, 0].set_title('Indoor Humidity')
        axes[1, 0].set_ylabel('Relative Humidity (%)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Illuminance
        axes[1, 1].plot(dataset['timestamp'], dataset['illuminance_lux'], alpha=0.7)
        axes[1, 1].set_title('Illuminance Levels')
        axes[1, 1].set_ylabel('Illuminance (lux)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ieq_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Occupancy and HVAC correlation
        fig, axes = plt.subplots(2, 2, figsize=fig_size)
        fig.suptitle('Occupancy and System Operation Analysis', fontsize=16)
        
        # Occupancy over time
        axes[0, 0].plot(dataset['timestamp'], dataset['occupant_count'], alpha=0.7)
        axes[0, 0].set_title('Occupant Count')
        axes[0, 0].set_ylabel('Number of Occupants')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Space utilization
        axes[0, 1].plot(dataset['timestamp'], dataset['space_utilization_pct'], alpha=0.7)
        axes[0, 1].set_title('Space Utilization')
        axes[0, 1].set_ylabel('Utilization (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # HVAC operation
        axes[1, 0].plot(dataset['timestamp'], dataset['fan_speed_pct'], label='Fan Speed', alpha=0.7)
        axes[1, 0].plot(dataset['timestamp'], dataset['damper_position_pct'], label='Damper Position', alpha=0.7)
        axes[1, 0].set_title('HVAC Operation')
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].legend()
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Correlation heatmap
        corr_data = dataset[['occupant_count', 'co2_ppm', 'hvac_electricity_kwh', 
                           'outdoor_temperature_f', 'illuminance_lux']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Sensor Data Correlations')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/occupancy_hvac_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved")
        
        return True

def main():
    """
    Main function to generate the complete IoT dataset.
    """
    print("=" * 60)
    print("IoT & Real-Time Monitoring Dataset Generator")
    print("Dynamic Digital Twin Framework for Building Retrofit Optimization")
    print("=" * 60)
    
    # Initialize generator
    generator = IoTDatasetGenerator(
        start_date='2023-01-01',
        end_date='2023-12-31',
        timezone='US/Eastern',
        building_area=50000
    )
    
    # Generate complete dataset
    dataset = generator.generate_complete_dataset()
    
    # Save dataset
    output_dir = generator.save_dataset(dataset)
    
    # Create visualizations
    generator.create_visualizations(dataset, output_dir)
    
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}/")
    print(f"Total data points: {len(dataset):,}")
    print(f"Date range: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
    print(f"Sampling frequency: 15 minutes")
    print(f"File formats: CSV, Parquet, Excel")
    print(f"Visualizations: PNG files")
    print("\nDataset columns:")
    for i, col in enumerate(dataset.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return dataset

if __name__ == "__main__":
    dataset = main()