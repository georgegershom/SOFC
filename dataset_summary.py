#!/usr/bin/env python3
"""
Dataset Summary and Statistics Generator

This script provides a comprehensive summary of the generated IoT building dataset,
including statistics, sample data, and usage examples.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def load_dataset():
    """Load the generated dataset."""
    print("Loading IoT Building Dataset...")
    dataset = pd.read_csv('iot_dataset/iot_building_dataset.csv')
    dataset['timestamp'] = pd.to_datetime(dataset['timestamp'], utc=True)
    print(f"✓ Dataset loaded: {dataset.shape[0]:,} records, {dataset.shape[1]} columns")
    return dataset

def generate_summary_statistics(dataset):
    """Generate comprehensive summary statistics."""
    print("\n" + "="*60)
    print("DATASET SUMMARY STATISTICS")
    print("="*60)
    
    # Basic info
    print(f"Dataset Period: {dataset['timestamp'].min()} to {dataset['timestamp'].max()}")
    print(f"Total Records: {len(dataset):,}")
    print(f"Sampling Frequency: 15 minutes")
    print(f"Building Area: 50,000 sq ft")
    print(f"Number of Zones: 32")
    
    # Energy consumption summary
    print(f"\n--- ENERGY CONSUMPTION ---")
    energy_cols = [col for col in dataset.columns if 'kwh' in col.lower()]
    for col in energy_cols:
        print(f"{col:25s}: {dataset[col].mean():8.1f} kWh (avg), {dataset[col].max():8.1f} kWh (max)")
    
    # Environmental conditions summary
    print(f"\n--- ENVIRONMENTAL CONDITIONS ---")
    env_cols = ['outdoor_temperature_f', 'indoor_humidity_rh', 'co2_ppm', 'illuminance_lux']
    for col in env_cols:
        if col in dataset.columns:
            print(f"{col:25s}: {dataset[col].mean():8.1f} (avg), {dataset[col].min():6.1f}-{dataset[col].max():6.1f} (range)")
    
    # Occupancy summary
    print(f"\n--- OCCUPANCY PATTERNS ---")
    print(f"{'occupant_count':25s}: {dataset['occupant_count'].mean():8.1f} (avg), {dataset['occupant_count'].max():8.0f} (max)")
    print(f"{'space_utilization_pct':25s}: {dataset['space_utilization_pct'].mean():8.1f}% (avg)")
    
    # HVAC operation summary
    print(f"\n--- HVAC OPERATION ---")
    hvac_cols = ['supply_air_temp_f', 'return_air_temp_f', 'fan_speed_pct', 'damper_position_pct']
    for col in hvac_cols:
        if col in dataset.columns:
            print(f"{col:25s}: {dataset[col].mean():8.1f} (avg), {dataset[col].min():6.1f}-{dataset[col].max():6.1f} (range)")

def show_sample_data(dataset):
    """Display sample data from the dataset."""
    print("\n" + "="*60)
    print("SAMPLE DATA (First 5 Records)")
    print("="*60)
    
    # Select key columns for display
    key_cols = [
        'timestamp', 'outdoor_temperature_f', 'total_electricity_kwh', 
        'co2_ppm', 'occupant_count', 'supply_air_temp_f'
    ]
    
    sample_data = dataset[key_cols].head()
    print(sample_data.to_string(index=False))

def show_correlations(dataset):
    """Display key correlations in the dataset."""
    print("\n" + "="*60)
    print("KEY CORRELATIONS")
    print("="*60)
    
    # Key correlation pairs
    correlations = [
        ('outdoor_temperature_f', 'hvac_electricity_kwh'),
        ('occupant_count', 'co2_ppm'),
        ('occupant_count', 'total_electricity_kwh'),
        ('solar_irradiance_wm2', 'illuminance_lux'),
        ('outdoor_temperature_f', 'indoor_humidity_rh')
    ]
    
    for var1, var2 in correlations:
        if var1 in dataset.columns and var2 in dataset.columns:
            corr = dataset[var1].corr(dataset[var2])
            print(f"{var1:25s} vs {var2:25s}: {corr:6.3f}")

def show_seasonal_patterns(dataset):
    """Display seasonal patterns in the data."""
    print("\n" + "="*60)
    print("SEASONAL PATTERNS")
    print("="*60)
    
    # Add month column
    timestamp_naive = dataset['timestamp'].dt.tz_localize(None)
    dataset['month'] = timestamp_naive.dt.month
    
    # Monthly averages
    monthly_stats = dataset.groupby('month').agg({
        'outdoor_temperature_f': 'mean',
        'total_electricity_kwh': 'mean',
        'occupant_count': 'mean',
        'co2_ppm': 'mean'
    }).round(1)
    
    print("Monthly Averages:")
    print(monthly_stats.to_string())

def show_usage_examples():
    """Display usage examples for the dataset."""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "title": "Load Dataset",
            "code": """
import pandas as pd
dataset = pd.read_csv('iot_dataset/iot_building_dataset.csv')
dataset['timestamp'] = pd.to_datetime(dataset['timestamp'])
            """
        },
        {
            "title": "Energy Analysis",
            "code": """
# Daily energy consumption
daily_energy = dataset.groupby(dataset['timestamp'].dt.date)['total_electricity_kwh'].sum()

# HVAC efficiency analysis
hvac_efficiency = dataset['hvac_electricity_kwh'] / (dataset['outdoor_temperature_f'] - 72).abs()
            """
        },
        {
            "title": "Occupancy Analysis",
            "code": """
# Business hours occupancy
business_hours = dataset['timestamp'].dt.hour.between(8, 18)
business_occupancy = dataset[business_hours]['occupant_count'].mean()

# CO2 correlation with occupancy
co2_occupancy_corr = dataset['co2_ppm'].corr(dataset['occupant_count'])
            """
        },
        {
            "title": "Time Series Forecasting",
            "code": """
# Prepare data for LSTM/RNN models
features = ['outdoor_temperature_f', 'occupant_count', 'total_electricity_kwh']
target = 'hvac_electricity_kwh'

X = dataset[features].values
y = dataset[target].values

# Create sequences for time series prediction
def create_sequences(X, y, seq_length=24):
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)
            """
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print("-" * 40)
        print(example['code'].strip())

def main():
    """Main function to generate dataset summary."""
    print("="*60)
    print("IoT BUILDING DATASET SUMMARY")
    print("Dynamic Digital Twin Framework")
    print("="*60)
    
    # Load dataset
    dataset = load_dataset()
    
    # Generate summaries
    generate_summary_statistics(dataset)
    show_sample_data(dataset)
    show_correlations(dataset)
    show_seasonal_patterns(dataset)
    show_usage_examples()
    
    print("\n" + "="*60)
    print("DATASET READY FOR AI MODEL TRAINING!")
    print("="*60)
    print("Files available:")
    print("  • iot_building_dataset.csv - Main dataset")
    print("  • iot_building_dataset.parquet - Efficient binary format")
    print("  • iot_building_dataset.xlsx - Excel with separate sheets")
    print("  • quality_report.json - Data quality assessment")
    print("  • dataset_metadata.json - Dataset specifications")
    print("  • *.png - Visualization files")
    
    return dataset

if __name__ == "__main__":
    dataset = main()