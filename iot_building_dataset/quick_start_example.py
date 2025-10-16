#!/usr/bin/env python3
"""
Quick Start Example for IoT Building Dataset
Demonstrates basic data loading and analysis
"""

import pandas as pd
import matplotlib.pyplot as plt
import json

print("="*80)
print("IoT BUILDING DATASET - QUICK START EXAMPLE")
print("="*80)

# 1. Load metadata
print("\n1. Loading dataset metadata...")
with open('dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"   Dataset period: {metadata['dataset_info']['start_date'][:10]} to {metadata['dataset_info']['end_date'][:10]}")
print(f"   Total zones: {metadata['building_config']['num_zones']}")
print(f"   Building area: {metadata['building_config']['total_area_sqm']:,} m²")

# 2. Load and preview energy data
print("\n2. Loading whole-building energy data...")
energy = pd.read_csv('energy/whole_building_energy.csv', parse_dates=['timestamp'])
print(f"   Records loaded: {len(energy):,}")
print("\n   First 5 records:")
print(energy.head())

# 3. Basic statistics
print("\n3. Energy consumption statistics:")
print(energy[['electricity_kw', 'gas_kw', 'water_m3']].describe())

# 4. Load weather data
print("\n4. Loading weather data...")
weather = pd.read_csv('weather/weather_station.csv', parse_dates=['timestamp'])
print(f"   Temperature range: {weather['ambient_temperature_c'].min():.1f}°C to {weather['ambient_temperature_c'].max():.1f}°C")
print(f"   Average solar irradiance: {weather['solar_irradiance_wm2'].mean():.0f} W/m²")

# 5. Merge datasets
print("\n5. Merging energy and weather data...")
df = energy.merge(weather, on='timestamp')
print(f"   Merged dataset: {len(df):,} records with {len(df.columns)} columns")

# 6. Calculate correlations
print("\n6. Key correlations:")
corr_temp = df['electricity_kw'].corr(df['ambient_temperature_c'])
print(f"   Electricity vs Outdoor Temperature: {corr_temp:.3f}")

corr_solar = df['electricity_kw'].corr(df['solar_irradiance_wm2'])
print(f"   Electricity vs Solar Irradiance: {corr_solar:.3f}")

# 7. Load occupancy data
print("\n7. Loading occupancy data...")
occupancy = pd.read_csv('occupancy/occupancy_usage.csv', parse_dates=['timestamp'])
print(f"   Max occupancy: {occupancy['occupant_count'].max():.0f} people")
print(f"   Average occupancy: {occupancy['occupant_count'].mean():.1f} people")

# 8. Load IEQ data for one zone
print("\n8. Loading Indoor Environmental Quality data...")
ieq = pd.read_csv('ieq/indoor_environmental_quality.csv', parse_dates=['timestamp'])
zone_1 = ieq[ieq['zone_id'] == 'Zone_01']
print(f"   Zone 01 average temperature: {zone_1['temperature_c'].mean():.1f}°C")
print(f"   Zone 01 average CO2: {zone_1['co2_ppm'].mean():.0f} ppm")

# 9. Simple visualization
print("\n9. Creating sample visualization...")
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Weekly energy pattern
energy_weekly = energy.set_index('timestamp').resample('D')['electricity_kw'].mean().head(7)
energy_weekly.plot(ax=axes[0], marker='o', linewidth=2, color='steelblue')
axes[0].set_title('First Week - Daily Average Electricity Consumption', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Electricity (kW)')
axes[0].grid(True, alpha=0.3)

# Plot 2: Energy vs Temperature scatter
axes[1].scatter(df['ambient_temperature_c'], df['electricity_kw'], alpha=0.1, s=1)
axes[1].set_xlabel('Outdoor Temperature (°C)')
axes[1].set_ylabel('Electricity Consumption (kW)')
axes[1].set_title('Electricity vs Outdoor Temperature', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_start_example.png', dpi=150, bbox_inches='tight')
print("   ✓ Saved: quick_start_example.png")
plt.close()

# 10. Summary
print("\n" + "="*80)
print("QUICK START COMPLETE")
print("="*80)
print("\nDataset successfully loaded and analyzed!")
print("\nNext steps:")
print("  1. Run 'python analyze_dataset.py' for comprehensive analysis")
print("  2. Explore individual CSV files for detailed data")
print("  3. Use this data for your Digital Twin and DRL research")
print("\nFor more information, see README.md")
print("="*80 + "\n")
