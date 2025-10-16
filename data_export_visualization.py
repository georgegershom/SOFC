#!/usr/bin/env python3
"""
Data Export and Visualization Tools
===================================

Comprehensive tools for exporting IoT building datasets and creating
interactive visualizations for analysis and validation.

This module provides:
- Multiple export formats (CSV, Parquet, HDF5, JSON)
- Interactive dashboards using Plotly/Dash
- Statistical analysis and correlation plots
- Time-series visualization tools
- Data quality assessment visualizations

Author: AI Assistant
Date: 2025-10-16
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import json
import os
import h5py
from typing import Dict, List, Tuple, Optional
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class DataExporter:
    """Handles exporting dataset to various formats."""
    
    def __init__(self, output_dir: str = "iot_building_dataset"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def export_all_formats(self, dataset: Dict[str, pd.DataFrame], 
                          metadata: Dict) -> Dict[str, str]:
        """Export dataset to all supported formats."""
        
        print("💾 Exporting dataset to multiple formats...")
        
        export_paths = {}
        
        # Export to CSV (most compatible)
        print("   📄 Exporting to CSV...")
        csv_paths = self.export_to_csv(dataset)
        export_paths['csv'] = csv_paths
        
        # Export to Parquet (efficient for analytics)
        print("   🗜️  Exporting to Parquet...")
        parquet_paths = self.export_to_parquet(dataset)
        export_paths['parquet'] = parquet_paths
        
        # Export to HDF5 (scientific computing)
        print("   🔬 Exporting to HDF5...")
        hdf5_path = self.export_to_hdf5(dataset)
        export_paths['hdf5'] = hdf5_path
        
        # Export metadata
        print("   📋 Exporting metadata...")
        metadata_path = self.export_metadata(metadata)
        export_paths['metadata'] = metadata_path
        
        # Create combined dataset
        print("   🔗 Creating combined dataset...")
        combined_path = self.create_combined_dataset(dataset)
        export_paths['combined'] = combined_path
        
        print(f"   ✅ All exports completed in: {self.output_dir}")
        
        return export_paths
    
    def export_to_csv(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export each dataframe to CSV format."""
        
        csv_dir = os.path.join(self.output_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        paths = {}
        
        for name, df in dataset.items():
            filename = f"{name}_data.csv"
            filepath = os.path.join(csv_dir, filename)
            df.to_csv(filepath, index=False)
            paths[name] = filepath
        
        return paths
    
    def export_to_parquet(self, dataset: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export each dataframe to Parquet format."""
        
        parquet_dir = os.path.join(self.output_dir, "parquet")
        os.makedirs(parquet_dir, exist_ok=True)
        
        paths = {}
        
        for name, df in dataset.items():
            filename = f"{name}_data.parquet"
            filepath = os.path.join(parquet_dir, filename)
            df.to_parquet(filepath, index=False, compression='snappy')
            paths[name] = filepath
        
        return paths
    
    def export_to_hdf5(self, dataset: Dict[str, pd.DataFrame]) -> str:
        """Export entire dataset to single HDF5 file."""
        
        filepath = os.path.join(self.output_dir, "iot_building_dataset.h5")
        
        with h5py.File(filepath, 'w') as f:
            for name, df in dataset.items():
                group = f.create_group(name)
                
                # Store timestamp as string
                timestamps = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S%z').values
                group.create_dataset('timestamp', data=timestamps.astype('S25'))
                
                # Store other columns as numeric arrays
                for col in df.columns:
                    if col != 'timestamp':
                        data = df[col].values
                        group.create_dataset(col, data=data, compression='gzip')
        
        return filepath
    
    def export_metadata(self, metadata: Dict) -> str:
        """Export metadata to JSON format."""
        
        filepath = os.path.join(self.output_dir, "metadata.json")
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return filepath
    
    def create_combined_dataset(self, dataset: Dict[str, pd.DataFrame]) -> str:
        """Create a single combined dataset with all parameters."""
        
        # Start with weather data as base
        combined_df = dataset['weather'].copy()
        
        # Add all other datasets
        for name, df in dataset.items():
            if name != 'weather':
                # Merge on timestamp
                combined_df = combined_df.merge(df, on='timestamp', how='left')
        
        # Export combined dataset
        filepath = os.path.join(self.output_dir, "combined_dataset.csv")
        combined_df.to_csv(filepath, index=False)
        
        return filepath

class DataVisualizer:
    """Creates comprehensive visualizations of the IoT building dataset."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        
    def create_comprehensive_dashboard(self, dataset: Dict[str, pd.DataFrame]) -> str:
        """Create a comprehensive interactive dashboard."""
        
        print("📊 Creating interactive dashboard...")
        
        # Initialize Dash app
        app = dash.Dash(__name__)
        
        # Define layout
        app.layout = html.Div([
            html.H1("IoT Building Dataset Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Select Data Category:"),
                    dcc.Dropdown(
                        id='category-dropdown',
                        options=[
                            {'label': 'Weather & External', 'value': 'weather'},
                            {'label': 'Energy Consumption', 'value': 'energy'},
                            {'label': 'Occupancy Patterns', 'value': 'occupancy'},
                            {'label': 'Indoor Air Quality', 'value': 'ieq'},
                            {'label': 'Building Systems', 'value': 'systems'}
                        ],
                        value='weather'
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Select Time Period:"),
                    dcc.DatePickerRange(
                        id='date-picker-range',
                        start_date=dataset['weather']['timestamp'].min(),
                        end_date=dataset['weather']['timestamp'].max(),
                        display_format='YYYY-MM-DD'
                    )
                ], style={'width': '40%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Aggregation:"),
                    dcc.Dropdown(
                        id='aggregation-dropdown',
                        options=[
                            {'label': 'Raw Data (15min)', 'value': 'raw'},
                            {'label': 'Hourly Average', 'value': 'hourly'},
                            {'label': 'Daily Average', 'value': 'daily'}
                        ],
                        value='hourly'
                    )
                ], style={'width': '25%', 'display': 'inline-block'})
            ], style={'marginBottom': 30}),
            
            # Main visualization area
            dcc.Graph(id='main-timeseries'),
            
            # Secondary plots
            html.Div([
                html.Div([
                    dcc.Graph(id='correlation-heatmap')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(id='distribution-plot')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Statistics table
            html.Div(id='statistics-table', style={'marginTop': 30})
        ])
        
        # Callbacks for interactivity
        @app.callback(
            [Output('main-timeseries', 'figure'),
             Output('correlation-heatmap', 'figure'),
             Output('distribution-plot', 'figure'),
             Output('statistics-table', 'children')],
            [Input('category-dropdown', 'value'),
             Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date'),
             Input('aggregation-dropdown', 'value')]
        )
        def update_dashboard(category, start_date, end_date, aggregation):
            # Filter data by date range
            df = dataset[category].copy()
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df[mask]
            
            # Apply aggregation
            if aggregation == 'hourly':
                df = df.set_index('timestamp').resample('H').mean().reset_index()
            elif aggregation == 'daily':
                df = df.set_index('timestamp').resample('D').mean().reset_index()
            
            # Create main time series plot
            main_fig = self._create_timeseries_plot(df, category)
            
            # Create correlation heatmap
            corr_fig = self._create_correlation_heatmap(df)
            
            # Create distribution plot
            dist_fig = self._create_distribution_plot(df)
            
            # Create statistics table
            stats_table = self._create_statistics_table(df)
            
            return main_fig, corr_fig, dist_fig, stats_table
        
        # Save dashboard as HTML
        dashboard_path = "iot_dashboard.html"
        
        # Note: In a real implementation, you would run app.run_server()
        # For this example, we'll create static plots instead
        
        return dashboard_path
    
    def create_static_visualizations(self, dataset: Dict[str, pd.DataFrame], 
                                   output_dir: str = "visualizations") -> Dict[str, str]:
        """Create static visualization plots."""
        
        print("📈 Creating static visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        viz_paths = {}
        
        # 1. Weather overview
        print("   🌤️  Weather overview...")
        weather_path = self._create_weather_overview(dataset['weather'], output_dir)
        viz_paths['weather_overview'] = weather_path
        
        # 2. Energy consumption analysis
        print("   ⚡ Energy consumption analysis...")
        energy_path = self._create_energy_analysis(dataset['energy'], output_dir)
        viz_paths['energy_analysis'] = energy_path
        
        # 3. Occupancy patterns
        print("   👥 Occupancy patterns...")
        occupancy_path = self._create_occupancy_analysis(dataset['occupancy'], output_dir)
        viz_paths['occupancy_patterns'] = occupancy_path
        
        # 4. IEQ analysis
        print("   🌡️  IEQ analysis...")
        ieq_path = self._create_ieq_analysis(dataset['ieq'], output_dir)
        viz_paths['ieq_analysis'] = ieq_path
        
        # 5. Systems performance
        print("   🔧 Systems performance...")
        systems_path = self._create_systems_analysis(dataset['systems'], output_dir)
        viz_paths['systems_performance'] = systems_path
        
        # 6. Correlation analysis
        print("   🔗 Correlation analysis...")
        correlation_path = self._create_correlation_analysis(dataset, output_dir)
        viz_paths['correlation_analysis'] = correlation_path
        
        # 7. Data quality assessment
        print("   ✅ Data quality assessment...")
        quality_path = self._create_data_quality_assessment(dataset, output_dir)
        viz_paths['data_quality'] = quality_path
        
        print(f"   ✅ All visualizations saved to: {output_dir}")
        
        return viz_paths
    
    def _create_weather_overview(self, weather_df: pd.DataFrame, output_dir: str) -> str:
        """Create weather overview visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Weather and External Conditions Overview', fontsize=16, fontweight='bold')
        
        # Temperature
        axes[0, 0].plot(weather_df['timestamp'], weather_df['ambient_temperature_c'], 
                       color='red', alpha=0.7)
        axes[0, 0].set_title('Ambient Temperature')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Solar irradiance
        axes[0, 1].plot(weather_df['timestamp'], weather_df['global_horizontal_irradiance_w_m2'], 
                       color='orange', alpha=0.7)
        axes[0, 1].set_title('Solar Irradiance')
        axes[0, 1].set_ylabel('Irradiance (W/m²)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Humidity
        axes[0, 2].plot(weather_df['timestamp'], weather_df['relative_humidity_pct'], 
                       color='blue', alpha=0.7)
        axes[0, 2].set_title('Relative Humidity')
        axes[0, 2].set_ylabel('Humidity (%)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Wind speed
        axes[1, 0].plot(weather_df['timestamp'], weather_df['wind_speed_m_s'], 
                       color='green', alpha=0.7)
        axes[1, 0].set_title('Wind Speed')
        axes[1, 0].set_ylabel('Speed (m/s)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rainfall
        axes[1, 1].bar(weather_df['timestamp'], weather_df['rainfall_mm_h'], 
                      color='skyblue', alpha=0.7, width=0.01)
        axes[1, 1].set_title('Rainfall')
        axes[1, 1].set_ylabel('Rainfall (mm/h)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Temperature vs Humidity scatter
        scatter = axes[1, 2].scatter(weather_df['ambient_temperature_c'], 
                                   weather_df['relative_humidity_pct'],
                                   c=weather_df.index, cmap='viridis', alpha=0.6)
        axes[1, 2].set_title('Temperature vs Humidity')
        axes[1, 2].set_xlabel('Temperature (°C)')
        axes[1, 2].set_ylabel('Humidity (%)')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'weather_overview.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_energy_analysis(self, energy_df: pd.DataFrame, output_dir: str) -> str:
        """Create energy consumption analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Energy Consumption Analysis', fontsize=16, fontweight='bold')
        
        # Total electricity consumption
        axes[0, 0].plot(energy_df['timestamp'], energy_df['total_electricity_kw'], 
                       color='red', alpha=0.7)
        axes[0, 0].set_title('Total Electricity Consumption')
        axes[0, 0].set_ylabel('Power (kW)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # HVAC breakdown
        hvac_cols = ['hvac_cooling_kw', 'hvac_heating_kw', 'hvac_fans_kw', 'hvac_pumps_kw']
        hvac_data = energy_df[hvac_cols].fillna(0)
        axes[0, 1].stackplot(energy_df['timestamp'], hvac_data.T, 
                           labels=hvac_cols, alpha=0.7)
        axes[0, 1].set_title('HVAC Energy Breakdown')
        axes[0, 1].set_ylabel('Power (kW)')
        axes[0, 1].legend(loc='upper right', fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Lighting zones
        lighting_cols = [col for col in energy_df.columns if 'lighting_zone' in col]
        if lighting_cols:
            lighting_data = energy_df[lighting_cols].fillna(0)
            axes[0, 2].stackplot(energy_df['timestamp'], lighting_data.T, alpha=0.7)
            axes[0, 2].set_title('Lighting Energy by Zone')
            axes[0, 2].set_ylabel('Power (kW)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Solar generation vs consumption
        if 'solar_pv_generation_kw' in energy_df.columns:
            axes[1, 0].plot(energy_df['timestamp'], energy_df['total_electricity_kw'], 
                           label='Consumption', color='red', alpha=0.7)
            axes[1, 0].plot(energy_df['timestamp'], energy_df['solar_pv_generation_kw'], 
                           label='Solar Generation', color='orange', alpha=0.7)
            axes[1, 0].set_title('Electricity Consumption vs Solar Generation')
            axes[1, 0].set_ylabel('Power (kW)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Daily energy profile (average)
        daily_profile = energy_df.groupby(energy_df['timestamp'].dt.hour)['total_electricity_kw'].mean()
        axes[1, 1].bar(daily_profile.index, daily_profile.values, color='blue', alpha=0.7)
        axes[1, 1].set_title('Average Daily Energy Profile')
        axes[1, 1].set_xlabel('Hour of Day')
        axes[1, 1].set_ylabel('Average Power (kW)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Energy intensity
        if 'total_site_eui_kwh_m2' in energy_df.columns:
            axes[1, 2].plot(energy_df['timestamp'], energy_df['total_site_eui_kwh_m2'], 
                           color='purple', alpha=0.7)
            axes[1, 2].set_title('Energy Use Intensity')
            axes[1, 2].set_ylabel('EUI (kWh/m²)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'energy_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_occupancy_analysis(self, occupancy_df: pd.DataFrame, output_dir: str) -> str:
        """Create occupancy patterns analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Occupancy Patterns Analysis', fontsize=16, fontweight='bold')
        
        # Total occupancy over time
        axes[0, 0].plot(occupancy_df['timestamp'], occupancy_df['total_occupancy'], 
                       color='blue', alpha=0.7)
        axes[0, 0].set_title('Total Building Occupancy')
        axes[0, 0].set_ylabel('People Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Weekly occupancy pattern
        occupancy_df['day_of_week'] = occupancy_df['timestamp'].dt.day_name()
        weekly_pattern = occupancy_df.groupby('day_of_week')['total_occupancy'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_pattern = weekly_pattern.reindex(day_order)
        
        axes[0, 1].bar(range(len(weekly_pattern)), weekly_pattern.values, 
                      color='green', alpha=0.7)
        axes[0, 1].set_title('Average Occupancy by Day of Week')
        axes[0, 1].set_xticks(range(len(weekly_pattern)))
        axes[0, 1].set_xticklabels([day[:3] for day in day_order], rotation=45)
        axes[0, 1].set_ylabel('Average People Count')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Hourly occupancy pattern
        hourly_pattern = occupancy_df.groupby(occupancy_df['timestamp'].dt.hour)['total_occupancy'].mean()
        axes[0, 2].plot(hourly_pattern.index, hourly_pattern.values, 
                       marker='o', color='red', alpha=0.7)
        axes[0, 2].set_title('Average Hourly Occupancy Pattern')
        axes[0, 2].set_xlabel('Hour of Day')
        axes[0, 2].set_ylabel('Average People Count')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Building utilization
        axes[1, 0].plot(occupancy_df['timestamp'], occupancy_df['building_utilization_pct'], 
                       color='orange', alpha=0.7)
        axes[1, 0].set_title('Building Utilization Percentage')
        axes[1, 0].set_ylabel('Utilization (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Zone occupancy heatmap (if available)
        zone_cols = [col for col in occupancy_df.columns if 'zone_' in col and '_occupancy' in col and '_pct' not in col]
        if len(zone_cols) >= 5:
            zone_data = occupancy_df[zone_cols[:5]].fillna(0)  # First 5 zones
            
            # Create heatmap data (sample every hour for readability)
            hourly_data = zone_data.iloc[::4]  # Every 4th point (hourly if 15-min data)
            
            im = axes[1, 1].imshow(hourly_data.T, aspect='auto', cmap='YlOrRd')
            axes[1, 1].set_title('Zone Occupancy Heatmap')
            axes[1, 1].set_xlabel('Time (hours)')
            axes[1, 1].set_ylabel('Zone')
            axes[1, 1].set_yticks(range(len(zone_cols[:5])))
            axes[1, 1].set_yticklabels([f'Zone {i+1}' for i in range(5)])
            plt.colorbar(im, ax=axes[1, 1], label='People Count')
        
        # Arrivals vs Departures
        if 'arrival_count' in occupancy_df.columns and 'departure_count' in occupancy_df.columns:
            axes[1, 2].plot(occupancy_df['timestamp'], occupancy_df['arrival_count'], 
                           label='Arrivals', color='green', alpha=0.7)
            axes[1, 2].plot(occupancy_df['timestamp'], occupancy_df['departure_count'], 
                           label='Departures', color='red', alpha=0.7)
            axes[1, 2].set_title('Arrivals vs Departures')
            axes[1, 2].set_ylabel('People Count')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'occupancy_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_ieq_analysis(self, ieq_df: pd.DataFrame, output_dir: str) -> str:
        """Create IEQ analysis visualization."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Indoor Environmental Quality Analysis', fontsize=16, fontweight='bold')
        
        # Building average temperature
        axes[0, 0].plot(ieq_df['timestamp'], ieq_df['building_avg_temp_c'], 
                       color='red', alpha=0.7)
        axes[0, 0].set_title('Building Average Temperature')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Building average CO2
        axes[0, 1].plot(ieq_df['timestamp'], ieq_df['building_avg_co2_ppm'], 
                       color='blue', alpha=0.7)
        axes[0, 1].axhline(y=1000, color='red', linestyle='--', alpha=0.7, label='1000 ppm threshold')
        axes[0, 1].set_title('Building Average CO2 Levels')
        axes[0, 1].set_ylabel('CO2 (ppm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Building average humidity
        axes[0, 2].plot(ieq_df['timestamp'], ieq_df['building_avg_rh_pct'], 
                       color='green', alpha=0.7)
        axes[0, 2].axhline(y=30, color='red', linestyle='--', alpha=0.5, label='Min comfort')
        axes[0, 2].axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Max comfort')
        axes[0, 2].set_title('Building Average Relative Humidity')
        axes[0, 2].set_ylabel('Humidity (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Zone temperature comparison (first 5 zones)
        temp_cols = [col for col in ieq_df.columns if 'zone_' in col and '_air_temp_c' in col][:5]
        if temp_cols:
            for i, col in enumerate(temp_cols):
                axes[1, 0].plot(ieq_df['timestamp'], ieq_df[col], 
                               alpha=0.7, label=f'Zone {i+1}')
            axes[1, 0].set_title('Zone Temperature Comparison')
            axes[1, 0].set_ylabel('Temperature (°C)')
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Air quality parameters
        pm25_cols = [col for col in ieq_df.columns if 'pm25' in col]
        if pm25_cols:
            axes[1, 1].plot(ieq_df['timestamp'], ieq_df[pm25_cols[0]], 
                           label='PM2.5', color='brown', alpha=0.7)
        
        tvoc_cols = [col for col in ieq_df.columns if 'tvoc' in col]
        if tvoc_cols:
            ax2 = axes[1, 1].twinx()
            ax2.plot(ieq_df['timestamp'], ieq_df[tvoc_cols[0]], 
                    label='TVOC', color='purple', alpha=0.7)
            ax2.set_ylabel('TVOC (ppb)')
        
        axes[1, 1].set_title('Air Quality Parameters')
        axes[1, 1].set_ylabel('PM2.5 (μg/m³)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Illuminance levels
        illuminance_cols = [col for col in ieq_df.columns if 'illuminance' in col]
        if illuminance_cols:
            axes[1, 2].plot(ieq_df['timestamp'], ieq_df[illuminance_cols[0]], 
                           color='yellow', alpha=0.7)
            axes[1, 2].axhline(y=500, color='red', linestyle='--', alpha=0.7, label='Target: 500 lux')
            axes[1, 2].set_title('Illuminance Levels')
            axes[1, 2].set_ylabel('Illuminance (lux)')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'ieq_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_systems_analysis(self, systems_df: pd.DataFrame, output_dir: str) -> str:
        """Create building systems analysis."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Building Systems Performance Analysis', fontsize=16, fontweight='bold')
        
        # Setpoints
        if 'building_cooling_setpoint_c' in systems_df.columns:
            axes[0, 0].plot(systems_df['timestamp'], systems_df['building_cooling_setpoint_c'], 
                           label='Cooling', color='blue', alpha=0.7)
        if 'building_heating_setpoint_c' in systems_df.columns:
            axes[0, 0].plot(systems_df['timestamp'], systems_df['building_heating_setpoint_c'], 
                           label='Heating', color='red', alpha=0.7)
        axes[0, 0].set_title('Temperature Setpoints')
        axes[0, 0].set_ylabel('Temperature (°C)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Chiller performance
        if 'chiller_capacity_pct' in systems_df.columns:
            axes[0, 1].plot(systems_df['timestamp'], systems_df['chiller_capacity_pct'], 
                           color='cyan', alpha=0.7)
            axes[0, 1].set_title('Chiller Capacity Utilization')
            axes[0, 1].set_ylabel('Capacity (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # System efficiency
        if 'hvac_system_efficiency_pct' in systems_df.columns:
            axes[0, 2].plot(systems_df['timestamp'], systems_df['hvac_system_efficiency_pct'], 
                           color='green', alpha=0.7)
            axes[0, 2].set_title('HVAC System Efficiency')
            axes[0, 2].set_ylabel('Efficiency (%)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Valve positions
        valve_cols = [col for col in systems_df.columns if 'valve' in col and 'pct' in col]
        if valve_cols:
            for i, col in enumerate(valve_cols[:3]):  # First 3 valves
                axes[1, 0].plot(systems_df['timestamp'], systems_df[col], 
                               alpha=0.7, label=col.replace('_', ' ').title())
            axes[1, 0].set_title('Valve Positions')
            axes[1, 0].set_ylabel('Position (%)')
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Fan speeds
        fan_cols = [col for col in systems_df.columns if 'fan' in col and 'speed' in col]
        if fan_cols:
            for i, col in enumerate(fan_cols[:3]):  # First 3 fans
                axes[1, 1].plot(systems_df['timestamp'], systems_df[col], 
                               alpha=0.7, label=col.replace('_', ' ').title())
            axes[1, 1].set_title('Fan Speeds')
            axes[1, 1].set_ylabel('Speed (%)')
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3)
        
        # System utilization
        if 'hvac_system_utilization_pct' in systems_df.columns:
            axes[1, 2].plot(systems_df['timestamp'], systems_df['hvac_system_utilization_pct'], 
                           color='purple', alpha=0.7)
            axes[1, 2].set_title('System Utilization')
            axes[1, 2].set_ylabel('Utilization (%)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'systems_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_correlation_analysis(self, dataset: Dict[str, pd.DataFrame], 
                                   output_dir: str) -> str:
        """Create correlation analysis across all systems."""
        
        # Combine key parameters from all datasets
        combined_data = {}
        
        # Weather
        weather_df = dataset['weather']
        combined_data['outdoor_temp'] = weather_df['ambient_temperature_c']
        combined_data['solar_irradiance'] = weather_df['global_horizontal_irradiance_w_m2']
        combined_data['humidity'] = weather_df['relative_humidity_pct']
        
        # Energy
        energy_df = dataset['energy']
        combined_data['total_electricity'] = energy_df['total_electricity_kw']
        combined_data['hvac_cooling'] = energy_df['hvac_cooling_kw']
        combined_data['hvac_heating'] = energy_df['hvac_heating_kw']
        
        # Occupancy
        occupancy_df = dataset['occupancy']
        combined_data['occupancy'] = occupancy_df['total_occupancy']
        
        # IEQ
        ieq_df = dataset['ieq']
        combined_data['indoor_temp'] = ieq_df['building_avg_temp_c']
        combined_data['co2_levels'] = ieq_df['building_avg_co2_ppm']
        
        # Create DataFrame
        corr_df = pd.DataFrame(combined_data)
        
        # Calculate correlation matrix
        correlation_matrix = corr_df.corr()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Cross-System Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Correlation heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, ax=axes[0])
        axes[0].set_title('Correlation Matrix')
        
        # Scatter plot matrix (subset)
        key_params = ['outdoor_temp', 'total_electricity', 'occupancy', 'indoor_temp', 'co2_levels']
        subset_df = corr_df[key_params]
        
        # Create scatter plot
        scatter_data = subset_df.sample(min(1000, len(subset_df)))  # Sample for performance
        
        # Plot outdoor temp vs electricity
        axes[1].scatter(scatter_data['outdoor_temp'], scatter_data['total_electricity'], 
                       alpha=0.6, c=scatter_data['occupancy'], cmap='viridis')
        axes[1].set_xlabel('Outdoor Temperature (°C)')
        axes[1].set_ylabel('Total Electricity (kW)')
        axes[1].set_title('Temperature vs Electricity (colored by occupancy)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'correlation_analysis.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_data_quality_assessment(self, dataset: Dict[str, pd.DataFrame], 
                                      output_dir: str) -> str:
        """Create data quality assessment visualization."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Data Quality Assessment', fontsize=16, fontweight='bold')
        
        # Data completeness
        completeness_data = {}
        for name, df in dataset.items():
            completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            completeness_data[name] = completeness
        
        axes[0, 0].bar(completeness_data.keys(), completeness_data.values(), 
                      color='green', alpha=0.7)
        axes[0, 0].set_title('Data Completeness by Category')
        axes[0, 0].set_ylabel('Completeness (%)')
        axes[0, 0].set_ylim(95, 100)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Data distribution example (temperature)
        weather_df = dataset['weather']
        axes[0, 1].hist(weather_df['ambient_temperature_c'], bins=50, 
                       alpha=0.7, color='red', edgecolor='black')
        axes[0, 1].set_title('Temperature Distribution')
        axes[0, 1].set_xlabel('Temperature (°C)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Outlier detection example (energy)
        energy_df = dataset['energy']
        Q1 = energy_df['total_electricity_kw'].quantile(0.25)
        Q3 = energy_df['total_electricity_kw'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = energy_df[(energy_df['total_electricity_kw'] < Q1 - 1.5*IQR) | 
                           (energy_df['total_electricity_kw'] > Q3 + 1.5*IQR)]
        
        axes[1, 0].boxplot(energy_df['total_electricity_kw'], vert=True)
        axes[1, 0].set_title(f'Energy Consumption Outliers\n({len(outliers)} outliers detected)')
        axes[1, 0].set_ylabel('Electricity (kW)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series continuity
        time_diffs = weather_df['timestamp'].diff().dt.total_seconds() / 60  # Minutes
        expected_interval = 15  # 15-minute intervals
        gaps = time_diffs[time_diffs > expected_interval * 1.5]
        
        axes[1, 1].hist(time_diffs.dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 1].axvline(x=expected_interval, color='red', linestyle='--', 
                          label=f'Expected: {expected_interval} min')
        axes[1, 1].set_title(f'Time Series Continuity\n({len(gaps)} gaps detected)')
        axes[1, 1].set_xlabel('Time Interval (minutes)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filepath = os.path.join(output_dir, 'data_quality_assessment.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_timeseries_plot(self, df: pd.DataFrame, category: str):
        """Create interactive time series plot."""
        # This would create a Plotly figure
        # Simplified for this example
        pass
    
    def _create_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap."""
        # This would create a Plotly heatmap
        # Simplified for this example
        pass
    
    def _create_distribution_plot(self, df: pd.DataFrame):
        """Create distribution plot."""
        # This would create a Plotly distribution plot
        # Simplified for this example
        pass
    
    def _create_statistics_table(self, df: pd.DataFrame):
        """Create statistics table."""
        # This would create an HTML table
        # Simplified for this example
        pass

def main():
    """Demonstrate the export and visualization tools."""
    
    print("💾📊 Data Export and Visualization Tools Demo")
    print("=" * 60)
    
    # This would normally use the complete dataset
    # For demo, create a simple sample
    from comprehensive_dataset_generator import ComprehensiveDatasetGenerator, BuildingConfig
    
    config = BuildingConfig()
    generator = ComprehensiveDatasetGenerator(config)
    
    # Generate sample dataset (1 week)
    print("Generating sample dataset...")
    dataset = generator.generate_complete_dataset(
        "2023-01-01 00:00:00", 
        "2023-01-07 23:45:00"
    )
    
    # Export data
    print("\n💾 Exporting data...")
    exporter = DataExporter("sample_export")
    export_paths = exporter.export_all_formats(dataset, generator.metadata)
    
    print("\nExport paths:")
    for format_type, paths in export_paths.items():
        print(f"  {format_type}: {paths}")
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    visualizer = DataVisualizer()
    viz_paths = visualizer.create_static_visualizations(dataset, "sample_visualizations")
    
    print("\nVisualization paths:")
    for viz_type, path in viz_paths.items():
        print(f"  {viz_type}: {path}")
    
    print("\n✅ Export and visualization demo complete!")
    
    return export_paths, viz_paths

if __name__ == "__main__":
    exports, visualizations = main()