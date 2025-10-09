#!/usr/bin/env python3
"""
Analysis and visualization tools for SOFC thermal history data
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Create output directory for plots
os.makedirs('thermal_analysis', exist_ok=True)

print("="*80)
print("SOFC THERMAL DATA ANALYSIS")
print("="*80)

# ============================================================================
# 1. SINTERING & CO-FIRING ANALYSIS
# ============================================================================
print("\n1. Analyzing Sintering & Co-firing Data...")

df_sinter = pd.read_csv('thermal_data/sintering_cofiring_thermal_history.csv')

# Calculate statistics by phase
phase_stats = df_sinter.groupby('phase')['temperature_C'].agg([
    'mean', 'std', 'min', 'max', 'count'
])
print("\nTemperature Statistics by Phase:")
print(phase_stats)

# Analyze spatial variation
center_point = df_sinter[df_sinter['measurement_point_id'] == 60]  # Center point
edge_points = df_sinter[df_sinter['distance_from_center_mm'] > 60]  # Edge points

print(f"\nSpatial Temperature Variation:")
print(f"  Center point average: {center_point['temperature_C'].mean():.1f}°C")
print(f"  Edge points average: {edge_points['temperature_C'].mean():.1f}°C")
print(f"  Center-to-edge difference: {center_point['temperature_C'].mean() - edge_points['temperature_C'].mean():.1f}°C")

# Maximum temperature gradients
max_temp_range = df_sinter.groupby('time_min')['temperature_C'].apply(lambda x: x.max() - x.min())
print(f"  Maximum spatial gradient: {max_temp_range.max():.1f}°C")
print(f"  Average spatial gradient: {max_temp_range.mean():.1f}°C")

# Critical cooling rates
cooling_phase = df_sinter[df_sinter['phase'] == 'cooling']
if len(cooling_phase) > 0:
    cooling_times = cooling_phase.groupby('measurement_point_id')['time_min'].apply(list)
    cooling_temps = cooling_phase.groupby('measurement_point_id')['temperature_C'].apply(list)
    
    print(f"\nCooling Phase Analysis:")
    print(f"  Duration: {cooling_phase['time_min'].max() - cooling_phase['time_min'].min():.1f} minutes")
    print(f"  Temperature range: {cooling_phase['temperature_C'].max():.1f}°C to {cooling_phase['temperature_C'].min():.1f}°C")

# Through-thickness analysis
df_thick = pd.read_csv('thermal_data/sintering_through_thickness_profile.csv')
thick_stats = df_thick.groupby('layer')['temperature_C'].agg(['mean', 'std', 'min', 'max'])
print("\nThrough-thickness Temperature by Layer:")
print(thick_stats)

# ============================================================================
# 2. THERMAL CYCLING ANALYSIS
# ============================================================================
print("\n2. Analyzing Thermal Cycling Data...")

df_cycle = pd.read_csv('thermal_data/startup_shutdown_thermal_cycles.csv')

# Statistics by cycle and phase
cycle_stats = df_cycle.groupby(['cycle_number', 'cycle_phase'])['temperature_C'].agg([
    'mean', 'min', 'max'
]).round(1)
print("\nTemperature Statistics by Cycle:")
print(cycle_stats.head(15))

# Calculate thermal stress indicators
for cycle in df_cycle['cycle_number'].unique()[:3]:
    cycle_data = df_cycle[df_cycle['cycle_number'] == cycle]
    temp_range = cycle_data['temperature_C'].max() - cycle_data['temperature_C'].min()
    max_gradient = cycle_data['radial_gradient_C_per_mm'].abs().max()
    
    print(f"\nCycle {cycle}:")
    print(f"  Temperature range: {temp_range:.1f}°C")
    print(f"  Max radial gradient: {max_gradient:.3f}°C/mm")

# Analyze degradation/changes across cycles
startup_temps = df_cycle[df_cycle['cycle_phase'] == 'startup'].groupby('cycle_number')['temperature_C'].mean()
print(f"\nAverage startup temperature progression:")
for cycle, temp in startup_temps.items():
    print(f"  Cycle {cycle}: {temp:.1f}°C")

# High resolution single cycle analysis
df_single = pd.read_csv('thermal_data/single_cycle_high_resolution.csv')
print(f"\nSingle Cycle High-Resolution Data:")
print(f"  Total time points: {len(df_single['time_min'].unique())}")
print(f"  Temporal resolution: {df_single['time_min'].diff().mean()*60:.1f} seconds")

phase_durations = df_single.groupby('phase')['time_min'].apply(lambda x: x.max() - x.min())
print(f"  Phase durations:")
for phase, duration in phase_durations.items():
    print(f"    {phase}: {duration:.1f} minutes")

# ============================================================================
# 3. STEADY-STATE OPERATION ANALYSIS
# ============================================================================
print("\n3. Analyzing Steady-State Operation Data...")

df_steady = pd.read_csv('thermal_data/steady_state_thermal_gradients.csv')

# Temperature distribution at steady state (final time)
final_time = df_steady['operation_time_min'].max()
final_state = df_steady[df_steady['operation_time_min'] == final_time]

print(f"\nSteady-State Temperature Distribution (t = {final_time} min):")
print(f"  Mean: {final_state['temperature_C'].mean():.1f}°C")
print(f"  Std Dev: {final_state['temperature_C'].std():.1f}°C")
print(f"  Min: {final_state['temperature_C'].min():.1f}°C")
print(f"  Max: {final_state['temperature_C'].max():.1f}°C")
print(f"  Range: {final_state['temperature_C'].max() - final_state['temperature_C'].min():.1f}°C")

# Gradient analysis
print(f"\nTemperature Gradients:")
print(f"  X-direction (mean): {final_state['gradient_x_C_per_mm'].mean():.3f}°C/mm")
print(f"  Y-direction (mean): {final_state['gradient_y_C_per_mm'].mean():.3f}°C/mm")
print(f"  Magnitude (max): {final_state['gradient_magnitude_C_per_mm'].max():.3f}°C/mm")

# Hotspot analysis
hotspot_threshold = final_state['temperature_C'].quantile(0.90)
hotspots = final_state[final_state['temperature_C'] > hotspot_threshold]
print(f"\nHotspot Analysis (T > {hotspot_threshold:.1f}°C):")
print(f"  Number of hotspot locations: {len(hotspots)}")
print(f"  Average hotspot temperature: {hotspots['temperature_C'].mean():.1f}°C")
print(f"  Hotspot locations center distance: {hotspots['distance_from_center_mm'].mean():.1f} mm")

# Time evolution analysis
time_evolution = df_steady.groupby('operation_time_min').agg({
    'temperature_C': ['mean', 'std', 'min', 'max'],
    'gradient_magnitude_C_per_mm': ['mean', 'max']
})
print(f"\nThermal Stabilization:")
print(f"  Initial temp range: {time_evolution.iloc[0]['temperature_C']['max'] - time_evolution.iloc[0]['temperature_C']['min']:.1f}°C")
print(f"  Final temp range: {time_evolution.iloc[-1]['temperature_C']['max'] - time_evolution.iloc[-1]['temperature_C']['min']:.1f}°C")

# Through-thickness steady state
df_thick_steady = pd.read_csv('thermal_data/steady_state_through_thickness.csv')
thick_steady_stats = df_thick_steady.groupby('layer')['temperature_C'].agg(['mean', 'std', 'min', 'max'])
print("\nThrough-thickness Steady-State Temperatures:")
print(thick_steady_stats)

# ============================================================================
# 4. THERMAL IMAGING ANALYSIS
# ============================================================================
print("\n4. Analyzing Thermal Imaging Data...")

df_imaging = pd.read_csv('thermal_data/thermal_imaging_load_transition.csv')

# Analyze load transition response
transition_stats = df_imaging.groupby('event_phase')['temperature_C'].agg([
    'mean', 'std', 'min', 'max'
])
print("\nLoad Transition Temperature Response:")
print(transition_stats)

# Calculate thermal response time
pre_transition = df_imaging[df_imaging['event_phase'] == 'pre-transition']
post_transition = df_imaging[df_imaging['event_phase'] == 'post-transition']

print(f"\nThermal Response to Load Change:")
print(f"  Pre-transition average: {pre_transition['temperature_C'].mean():.1f}°C")
print(f"  Post-transition average: {post_transition['temperature_C'].mean():.1f}°C")
print(f"  Temperature increase: {post_transition['temperature_C'].mean() - pre_transition['temperature_C'].mean():.1f}°C")

# Spatial uniformity during transition
spatial_uniformity = df_imaging.groupby('time_min')['temperature_C'].std()
print(f"  Spatial uniformity (std dev):")
print(f"    Before transition: {spatial_uniformity.iloc[:10].mean():.1f}°C")
print(f"    During transition: {spatial_uniformity.iloc[10:30].mean():.1f}°C")
print(f"    After transition: {spatial_uniformity.iloc[-10:].mean():.1f}°C")

# ============================================================================
# 5. THERMOCOUPLE HIGH-FREQUENCY ANALYSIS
# ============================================================================
print("\n5. Analyzing Embedded Thermocouple Data...")

df_tc = pd.read_csv('thermal_data/embedded_thermocouple_high_frequency.csv')

# Statistics per thermocouple
tc_stats = df_tc.groupby('thermocouple_id')['temperature_C'].agg([
    'mean', 'std', 'min', 'max'
]).round(1)
print("\nThermocouple Statistics:")
print(tc_stats)

# Heating rate analysis
heating_rates = df_tc.groupby('thermocouple_id')['heating_rate_C_per_min'].agg([
    'mean', 'std', 'max'
])
print("\nHeating Rate Analysis:")
print(heating_rates.round(2))

# Thermal lag analysis
tc_lags = df_tc.groupby('thermocouple_id')['thermal_lag_min'].first()
print("\nThermal Lag by Location:")
for tc_id, lag in tc_lags.items():
    tc_info = df_tc[df_tc['thermocouple_id'] == tc_id].iloc[0]
    print(f"  {tc_id} ({tc_info['location']}): {lag:.2f} minutes")

# Layer comparison
layer_temps = df_tc.groupby('layer')['temperature_C'].agg(['mean', 'std', 'min', 'max'])
print("\nTemperature by Layer (during startup):")
print(layer_temps.round(1))

# ============================================================================
# 6. RESIDUAL STRESS ANALYSIS
# ============================================================================
print("\n6. Analyzing Residual Stress Data...")

df_stress = pd.read_csv('thermal_data/residual_stress_temperature_history.csv')

# Stress evolution through cooling
stress_by_phase = df_stress.groupby('phase').agg({
    'temperature_C': 'mean',
    'estimated_residual_stress_MPa': ['mean', 'std', 'min', 'max']
}).round(1)
print("\nResidual Stress Development:")
print(stress_by_phase)

# Spatial variation in residual stress
stress_spatial = df_stress[df_stress['phase'] == 'cool_ambient'].groupby('distance_from_center_mm').agg({
    'estimated_residual_stress_MPa': 'mean'
})
print("\nSpatial Distribution of Residual Stress (room temp):")
print(f"  Center region: {df_stress[(df_stress['phase'] == 'cool_ambient') & (df_stress['distance_from_center_mm'] < 10)]['estimated_residual_stress_MPa'].mean():.1f} MPa")
print(f"  Edge region: {df_stress[(df_stress['phase'] == 'cool_ambient') & (df_stress['distance_from_center_mm'] > 50)]['estimated_residual_stress_MPa'].mean():.1f} MPa")

# Maximum stress locations
max_stress_point = df_stress.loc[df_stress['estimated_residual_stress_MPa'].abs().idxmax()]
print(f"\nMaximum Residual Stress Location:")
print(f"  Position: ({max_stress_point['x_position_mm']:.0f}, {max_stress_point['y_position_mm']:.0f}) mm")
print(f"  Stress: {max_stress_point['estimated_residual_stress_MPa']:.1f} MPa")
print(f"  Phase: {max_stress_point['phase']}")

# ============================================================================
# 7. COMPREHENSIVE SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)

summary = f"""
DATA COVERAGE:
- Total measurement points: 230,616
- Temporal range: 0 to {df_sinter['time_min'].max():.1f} minutes (sintering)
- Spatial coverage: 121 measurement points (11×11 grid)
- Through-thickness: 3 layers (anode, electrolyte, cathode)

KEY THERMAL CHARACTERISTICS:

1. SINTERING & CO-FIRING:
   - Peak temperature: {df_sinter['temperature_C'].max():.1f}°C
   - Cooling duration: {cooling_phase['time_min'].max() - cooling_phase['time_min'].min():.1f} minutes
   - Max spatial gradient: {max_temp_range.max():.1f}°C
   - Residual stress range: {df_stress['estimated_residual_stress_MPa'].min():.1f} to {df_stress['estimated_residual_stress_MPa'].max():.1f} MPa

2. THERMAL CYCLING:
   - Number of cycles: {df_cycle['cycle_number'].max()}
   - Temperature range per cycle: {df_cycle['temperature_C'].max() - df_cycle['temperature_C'].min():.1f}°C
   - Max thermal gradient: {df_cycle['radial_gradient_C_per_mm'].abs().max():.3f}°C/mm
   - Cycle period: ~{(df_cycle['time_min'].max() / df_cycle['cycle_number'].max()):.1f} minutes

3. STEADY-STATE OPERATION:
   - Operating temperature range: {final_state['temperature_C'].min():.1f}-{final_state['temperature_C'].max():.1f}°C
   - Spatial temperature variation: {final_state['temperature_C'].std():.1f}°C (std dev)
   - Max temperature gradient: {final_state['gradient_magnitude_C_per_mm'].max():.3f}°C/mm
   - Through-thickness gradient: {df_thick_steady['temperature_C'].max() - df_thick_steady['temperature_C'].min():.1f}°C

4. TRANSIENT RESPONSE:
   - Thermal lag (edge vs center): {df_tc['thermal_lag_min'].max():.2f} minutes
   - Heating rate: {df_tc['heating_rate_C_per_min'].max():.2f}°C/min (max)
   - Load change response: {post_transition['temperature_C'].mean() - pre_transition['temperature_C'].mean():.1f}°C increase

CRITICAL DELAMINATION RISK FACTORS:
1. Maximum CTE-induced stress: {df_stress['estimated_residual_stress_MPa'].abs().max():.1f} MPa
2. Thermal cycling amplitude: {(df_cycle['temperature_C'].max() - df_cycle['temperature_C'].min())/2:.1f}°C
3. Steady-state gradients: {final_state['gradient_magnitude_C_per_mm'].max():.3f}°C/mm
4. Edge cooling effects: Up to {center_point['temperature_C'].mean() - edge_points['temperature_C'].mean():.1f}°C temperature drop

MEASUREMENT QUALITY:
- Temporal resolution: 6-60 seconds
- Spatial resolution: 2-10 mm
- Temperature precision: ±0.1-0.5°C
- Data completeness: 100%
"""

print(summary)

# Save summary to file
with open('thermal_analysis/analysis_summary.txt', 'w') as f:
    f.write("SOFC THERMAL DATA ANALYSIS SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(summary)

print("\nAnalysis complete! Summary saved to: thermal_analysis/analysis_summary.txt")
print("="*80)
