#!/usr/bin/env python3
"""
IoT Building Dataset Analysis Script
Provides comprehensive analysis and visualization of the generated dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class BuildingDatasetAnalyzer:
    def __init__(self, dataset_dir='.'):
        """Initialize the dataset analyzer"""
        self.dataset_dir = dataset_dir
        self.data = {}
        
    def load_all_data(self):
        """Load all dataset files"""
        print("Loading dataset files...")
        
        # Load energy data
        self.data['energy_whole'] = pd.read_csv(
            f'{self.dataset_dir}/energy/whole_building_energy.csv',
            parse_dates=['timestamp']
        )
        self.data['energy_sub'] = pd.read_csv(
            f'{self.dataset_dir}/energy/sub_metered_energy.csv',
            parse_dates=['timestamp']
        )
        
        # Load IEQ data
        self.data['ieq'] = pd.read_csv(
            f'{self.dataset_dir}/ieq/indoor_environmental_quality.csv',
            parse_dates=['timestamp']
        )
        
        # Load occupancy data
        self.data['occupancy'] = pd.read_csv(
            f'{self.dataset_dir}/occupancy/occupancy_usage.csv',
            parse_dates=['timestamp']
        )
        
        # Load weather data
        self.data['weather'] = pd.read_csv(
            f'{self.dataset_dir}/weather/weather_station.csv',
            parse_dates=['timestamp']
        )
        
        # Load HVAC data
        self.data['hvac'] = pd.read_csv(
            f'{self.dataset_dir}/hvac_systems/hvac_operation.csv',
            parse_dates=['timestamp']
        )
        
        # Load metadata
        with open(f'{self.dataset_dir}/dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print("✓ All data loaded successfully!")
        return self.data
    
    def print_dataset_summary(self):
        """Print comprehensive dataset summary"""
        print("\n" + "="*80)
        print("BUILDING IOT DATASET SUMMARY")
        print("="*80)
        
        # Dataset info
        info = self.metadata['dataset_info']
        print(f"\nDataset Information:")
        print(f"  Period: {info['start_date'][:10]} to {info['end_date'][:10]}")
        print(f"  Duration: {info['duration_days']} days")
        print(f"  Sampling: {info['sampling_interval_minutes']} minutes")
        print(f"  Records/stream: {info['total_records_per_stream']:,}")
        
        # Building config
        config = self.metadata['building_config']
        print(f"\nBuilding Configuration:")
        print(f"  Type: {config['building_type']}")
        print(f"  Area: {config['total_area_sqm']:,} m²")
        print(f"  Floors: {config['num_floors']}")
        print(f"  Zones: {config['num_zones']}")
        print(f"  Max Occupancy: {config['max_occupancy']}")
        
        # Data streams
        print(f"\nData Streams:")
        for stream_name, stream_info in self.metadata['data_streams'].items():
            if 'records' in stream_info:
                print(f"  {stream_name.upper()}: {stream_info['records']:,} records")
            else:
                for sub_name, sub_info in stream_info.items():
                    if isinstance(sub_info, dict) and 'records' in sub_info:
                        print(f"  {stream_name.upper()}/{sub_name}: {sub_info['records']:,} records")
        
        # File sizes
        print(f"\nDataset Files:")
        total_size = 0
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    total_size += size_mb
                    print(f"  {filepath}: {size_mb:.2f} MB")
        
        print(f"\nTotal Dataset Size: {total_size:.2f} MB")
        print("="*80 + "\n")
    
    def analyze_energy_patterns(self):
        """Analyze energy consumption patterns"""
        print("\nEnergy Consumption Analysis")
        print("-" * 60)
        
        df = self.data['energy_whole'].copy()
        
        # Overall statistics
        print("\nWhole-Building Energy Statistics:")
        print(df[['electricity_kw', 'gas_kw', 'water_m3']].describe())
        
        # Seasonal analysis
        df['month'] = df['timestamp'].dt.month
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        seasonal_energy = df.groupby('season')[['electricity_kw', 'gas_kw']].mean()
        print("\nSeasonal Average Energy (kW):")
        print(seasonal_energy)
        
        # Daily patterns
        df['hour'] = df['timestamp'].dt.hour
        df['is_weekday'] = df['timestamp'].dt.dayofweek < 5
        
        weekday_pattern = df[df['is_weekday']].groupby('hour')['electricity_kw'].mean()
        weekend_pattern = df[~df['is_weekday']].groupby('hour')['electricity_kw'].mean()
        
        print(f"\nPeak Electricity:")
        print(f"  Weekday: {weekday_pattern.max():.1f} kW at {weekday_pattern.idxmax()}:00")
        print(f"  Weekend: {weekend_pattern.max():.1f} kW at {weekend_pattern.idxmax()}:00")
        
        # Sub-metered breakdown
        sub_df = self.data['energy_sub'].copy()
        avg_breakdown = {
            'HVAC': sub_df['hvac_electricity_kw'].mean(),
            'Lighting': sub_df['lighting_electricity_kw'].mean(),
            'Plug Loads': sub_df['plug_loads_kw'].mean(),
            'Other': sub_df['other_loads_kw'].mean()
        }
        
        print(f"\nAverage Energy Breakdown:")
        total = sum(avg_breakdown.values())
        for category, value in avg_breakdown.items():
            pct = (value / total) * 100
            print(f"  {category}: {value:.1f} kW ({pct:.1f}%)")
    
    def analyze_ieq_quality(self):
        """Analyze Indoor Environmental Quality"""
        print("\n\nIndoor Environmental Quality Analysis")
        print("-" * 60)
        
        df = self.data['ieq'].copy()
        
        # Overall IEQ statistics
        ieq_metrics = ['temperature_c', 'relative_humidity_pct', 'co2_ppm', 
                       'pm25_ugm3', 'illuminance_lux', 'noise_db']
        
        print("\nIEQ Metrics Summary (All Zones):")
        print(df[ieq_metrics].describe())
        
        # Zone comparison
        zone_temps = df.groupby('zone_id')['temperature_c'].agg(['mean', 'std'])
        print("\nTemperature by Zone (°C):")
        print(zone_temps)
        
        # Comfort analysis
        df['is_comfortable_temp'] = (df['temperature_c'] >= 20) & (df['temperature_c'] <= 26)
        df['is_good_co2'] = df['co2_ppm'] <= 1000
        df['is_good_pm25'] = df['pm25_ugm3'] <= 35
        
        comfort_pct = df['is_comfortable_temp'].mean() * 100
        co2_pct = df['is_good_co2'].mean() * 100
        pm25_pct = df['is_good_pm25'].mean() * 100
        
        print(f"\nIEQ Compliance:")
        print(f"  Thermal Comfort (20-26°C): {comfort_pct:.1f}%")
        print(f"  Good CO2 (≤1000 ppm): {co2_pct:.1f}%")
        print(f"  Good PM2.5 (≤35 μg/m³): {pm25_pct:.1f}%")
    
    def analyze_occupancy_patterns(self):
        """Analyze occupancy and usage patterns"""
        print("\n\nOccupancy & Usage Pattern Analysis")
        print("-" * 60)
        
        df = self.data['occupancy'].copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['dayofweek'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['is_weekday'] = df['dayofweek'] < 5
        
        # Occupancy statistics
        print("\nOccupancy Statistics:")
        print(f"  Maximum: {df['occupant_count'].max():.0f} people")
        print(f"  Average (all time): {df['occupant_count'].mean():.1f} people")
        print(f"  Average (weekdays): {df[df['is_weekday']]['occupant_count'].mean():.1f} people")
        print(f"  Average (weekends): {df[~df['is_weekday']]['occupant_count'].mean():.1f} people")
        
        # Peak occupancy
        peak_hour = df[df['is_weekday']].groupby('hour')['occupant_count'].mean()
        print(f"\nPeak Occupancy:")
        print(f"  Time: {peak_hour.idxmax()}:00")
        print(f"  Count: {peak_hour.max():.0f} people")
        
        # Space utilization
        print(f"\nSpace Utilization (%):")
        print(f"  Desk: {df['desk_utilization_pct'].mean():.1f}% average")
        print(f"  Meeting Rooms: {df['meeting_room_utilization_pct'].mean():.1f}% average")
        
        # Window/blind operation
        print(f"\nWindow & Blind Operation:")
        print(f"  Windows Open: {df['windows_open_pct'].mean():.1f}% average")
        print(f"  Blinds Closed: {df['blinds_closed_pct'].mean():.1f}% average")
    
    def analyze_weather_conditions(self):
        """Analyze weather conditions"""
        print("\n\nWeather Conditions Analysis")
        print("-" * 60)
        
        df = self.data['weather'].copy()
        df['month'] = pd.to_datetime(df['timestamp']).dt.month
        
        # Temperature statistics
        print("\nTemperature Statistics (°C):")
        print(f"  Annual Range: {df['ambient_temperature_c'].min():.1f} to {df['ambient_temperature_c'].max():.1f}")
        print(f"  Annual Average: {df['ambient_temperature_c'].mean():.1f}")
        
        # Monthly averages
        monthly_temp = df.groupby('month')['ambient_temperature_c'].mean()
        print(f"\nTemperature by Month:")
        for month, temp in monthly_temp.items():
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
            print(f"  {month_name}: {temp:.1f}°C")
        
        # Solar irradiance
        print(f"\nSolar Irradiance (W/m²):")
        print(f"  Maximum: {df['solar_irradiance_wm2'].max():.0f}")
        print(f"  Daily Average: {df['solar_irradiance_wm2'].mean():.0f}")
        
        # Precipitation
        rainy_days = df.groupby(df['timestamp'].dt.date)['rainfall_mm'].sum()
        rainy_days_count = (rainy_days > 0).sum()
        print(f"\nPrecipitation:")
        print(f"  Total Annual: {df['rainfall_mm'].sum():.1f} mm")
        print(f"  Rainy Days: {rainy_days_count} days")
    
    def analyze_hvac_operation(self):
        """Analyze HVAC system operation"""
        print("\n\nHVAC System Operation Analysis")
        print("-" * 60)
        
        df = self.data['hvac'].copy()
        
        # System statistics by zone
        print("\nHVAC Operation Summary (All Zones):")
        hvac_metrics = ['supply_air_temp_c', 'return_air_temp_c', 'fan_speed_pct', 
                       'heating_valve_pct', 'cooling_valve_pct']
        print(df[hvac_metrics].describe())
        
        # Operational hours
        chiller_hours = df['chiller_status'].sum() * 0.25  # 15-min intervals
        boiler_hours = df['boiler_status'].sum() * 0.25
        
        print(f"\nSystem Runtime (hours/year):")
        print(f"  Chiller: {chiller_hours:,.0f} hours")
        print(f"  Boiler: {boiler_hours:,.0f} hours")
        
        # Setpoint analysis
        print(f"\nSetpoint Statistics (°C):")
        print(f"  Heating: {df['heating_setpoint_c'].mean():.1f} ± {df['heating_setpoint_c'].std():.1f}")
        print(f"  Cooling: {df['cooling_setpoint_c'].mean():.1f} ± {df['cooling_setpoint_c'].std():.1f}")
    
    def generate_correlation_analysis(self):
        """Generate correlation analysis between key variables"""
        print("\n\nCorrelation Analysis")
        print("-" * 60)
        
        # Merge key datasets
        df = self.data['energy_whole'].merge(
            self.data['weather'], on='timestamp'
        ).merge(
            self.data['occupancy'], on='timestamp'
        )
        
        # Select key variables
        variables = [
            'electricity_kw', 'gas_kw', 
            'ambient_temperature_c', 'solar_irradiance_wm2',
            'occupant_count'
        ]
        
        corr_matrix = df[variables].corr()
        
        print("\nCorrelation Matrix:")
        print(corr_matrix)
        
        # Key correlations
        print("\nKey Correlations:")
        print(f"  Electricity vs Outdoor Temp: {corr_matrix.loc['electricity_kw', 'ambient_temperature_c']:.3f}")
        print(f"  Electricity vs Occupancy: {corr_matrix.loc['electricity_kw', 'occupant_count']:.3f}")
        print(f"  Gas vs Outdoor Temp: {corr_matrix.loc['gas_kw', 'ambient_temperature_c']:.3f}")
    
    def create_visualizations(self, output_dir='analysis_plots'):
        """Create comprehensive visualizations"""
        print(f"\n\nGenerating visualizations...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Energy consumption over time
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        df_energy = self.data['energy_whole'].set_index('timestamp')
        df_energy['electricity_kw'].resample('D').mean().plot(ax=axes[0], color='steelblue')
        axes[0].set_title('Daily Average Electricity Consumption', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Electricity (kW)')
        axes[0].grid(True, alpha=0.3)
        
        # Energy breakdown
        df_sub = self.data['energy_sub'].set_index('timestamp')
        df_sub[['hvac_electricity_kw', 'lighting_electricity_kw', 
                'plug_loads_kw', 'other_loads_kw']].resample('D').mean().plot(
            ax=axes[1], stacked=False, alpha=0.7
        )
        axes[1].set_title('Energy Breakdown by End-Use', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Electricity (kW)')
        axes[1].legend(['HVAC', 'Lighting', 'Plug Loads', 'Other'])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/energy_consumption.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/energy_consumption.png")
        plt.close()
        
        # 2. IEQ conditions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df_ieq = self.data['ieq']
        zone_avg = df_ieq.groupby('zone_id').agg({
            'temperature_c': 'mean',
            'co2_ppm': 'mean',
            'relative_humidity_pct': 'mean',
            'illuminance_lux': 'mean'
        })
        
        zone_avg['temperature_c'].plot(kind='bar', ax=axes[0,0], color='coral')
        axes[0,0].set_title('Average Temperature by Zone', fontweight='bold')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].axhline(y=22, color='red', linestyle='--', alpha=0.5, label='Target')
        
        zone_avg['co2_ppm'].plot(kind='bar', ax=axes[0,1], color='skyblue')
        axes[0,1].set_title('Average CO₂ by Zone', fontweight='bold')
        axes[0,1].set_ylabel('CO₂ (ppm)')
        axes[0,1].axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Limit')
        
        zone_avg['relative_humidity_pct'].plot(kind='bar', ax=axes[1,0], color='lightgreen')
        axes[1,0].set_title('Average Humidity by Zone', fontweight='bold')
        axes[1,0].set_ylabel('Relative Humidity (%)')
        
        zone_avg['illuminance_lux'].plot(kind='bar', ax=axes[1,1], color='gold')
        axes[1,1].set_title('Average Illuminance by Zone', fontweight='bold')
        axes[1,1].set_ylabel('Illuminance (lux)')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ieq_conditions.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/ieq_conditions.png")
        plt.close()
        
        # 3. Weather conditions
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        df_weather = self.data['weather'].set_index('timestamp')
        
        df_weather['ambient_temperature_c'].resample('D').mean().plot(ax=axes[0,0], color='orangered')
        axes[0,0].set_title('Daily Average Temperature', fontweight='bold')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].grid(True, alpha=0.3)
        
        df_weather['solar_irradiance_wm2'].resample('D').mean().plot(ax=axes[0,1], color='gold')
        axes[0,1].set_title('Daily Average Solar Irradiance', fontweight='bold')
        axes[0,1].set_ylabel('Irradiance (W/m²)')
        axes[0,1].grid(True, alpha=0.3)
        
        df_weather['wind_speed_ms'].resample('D').mean().plot(ax=axes[1,0], color='steelblue')
        axes[1,0].set_title('Daily Average Wind Speed', fontweight='bold')
        axes[1,0].set_ylabel('Wind Speed (m/s)')
        axes[1,0].grid(True, alpha=0.3)
        
        df_weather['rainfall_mm'].resample('D').sum().plot(ax=axes[1,1], color='darkblue')
        axes[1,1].set_title('Daily Rainfall', fontweight='bold')
        axes[1,1].set_ylabel('Rainfall (mm)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/weather_conditions.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/weather_conditions.png")
        plt.close()
        
        # 4. Occupancy patterns
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        df_occ = self.data['occupancy'].copy()
        df_occ['hour'] = pd.to_datetime(df_occ['timestamp']).dt.hour
        df_occ['is_weekday'] = pd.to_datetime(df_occ['timestamp']).dt.dayofweek < 5
        
        weekday_occ = df_occ[df_occ['is_weekday']].groupby('hour')['occupant_count'].mean()
        weekend_occ = df_occ[~df_occ['is_weekday']].groupby('hour')['occupant_count'].mean()
        
        weekday_occ.plot(ax=axes[0], marker='o', label='Weekday', linewidth=2)
        weekend_occ.plot(ax=axes[0], marker='s', label='Weekend', linewidth=2)
        axes[0].set_title('Average Occupancy Pattern', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Occupant Count')
        axes[0].set_xlabel('Hour of Day')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Space utilization
        df_occ_ts = df_occ.set_index('timestamp')
        df_occ_ts[['desk_utilization_pct', 'meeting_room_utilization_pct']].resample('D').mean().plot(ax=axes[1])
        axes[1].set_title('Daily Space Utilization', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Utilization (%)')
        axes[1].legend(['Desk', 'Meeting Rooms'])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/occupancy_patterns.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/occupancy_patterns.png")
        plt.close()
        
        # 5. Correlation heatmap
        df_merged = self.data['energy_whole'].merge(
            self.data['weather'], on='timestamp'
        ).merge(
            self.data['occupancy'], on='timestamp'
        )
        
        variables = [
            'electricity_kw', 'gas_kw', 'water_m3',
            'ambient_temperature_c', 'solar_irradiance_wm2',
            'occupant_count', 'wind_speed_ms'
        ]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = df_merged[variables].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                   square=True, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Correlation Matrix: Energy, Weather & Occupancy', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_matrix.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/correlation_matrix.png")
        plt.close()
        
        print(f"\nAll visualizations saved to: {output_dir}/")
    
    def run_full_analysis(self):
        """Run complete dataset analysis"""
        self.load_all_data()
        self.print_dataset_summary()
        self.analyze_energy_patterns()
        self.analyze_ieq_quality()
        self.analyze_occupancy_patterns()
        self.analyze_weather_conditions()
        self.analyze_hvac_operation()
        self.generate_correlation_analysis()
        self.create_visualizations()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("\nDataset is ready for use in:")
        print("  • Digital Twin development")
        print("  • Deep Reinforcement Learning training")
        print("  • Building retrofit optimization")
        print("  • Energy modeling and prediction")
        print("  • Indoor environmental quality analysis")
        print("\n")

if __name__ == "__main__":
    analyzer = BuildingDatasetAnalyzer()
    analyzer.run_full_analysis()
