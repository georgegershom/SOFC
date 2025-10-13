#!/usr/bin/env python3
"""
Example Usage of SOFC Digital Twin Dataset
Demonstrates loading, analyzing, and visualizing the multi-fidelity data
"""

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Set up paths
data_dir = Path("data")

print("=" * 80)
print("SOFC Digital Twin Dataset - Example Usage")
print("=" * 80)

# 1. Load and explore simulation data
print("\n1. SIMULATION DATA")
print("-" * 40)

# Load simulation metadata
with open(data_dir / "simulation" / "simulation_metadata.json", "r") as f:
    simulations = json.load(f)

print(f"Total simulations: {len(simulations)}")
print(f"\nFirst simulation parameters:")
sim0 = simulations[0]
for param, value in sim0['parameters'].items():
    if isinstance(value, float):
        print(f"  {param}: {value:.2f}")
    else:
        print(f"  {param}: {value}")

print(f"\nFirst simulation outputs:")
for output, value in sim0['outputs'].items():
    if 'stress' in output:
        print(f"  {output}: {value/1e6:.2f} MPa")
    else:
        print(f"  {output}: {value:.4f}")

# Load field data for first simulation
temperature_field = np.load(data_dir / "simulation" / "temperature_0000.npy")
stress_field = np.load(data_dir / "simulation" / "stress_0000.npy")
current_field = np.load(data_dir / "simulation" / "current_density_0000.npy")

print(f"\nField data shapes:")
print(f"  Temperature: {temperature_field.shape}")
print(f"  Stress: {stress_field.shape}")
print(f"  Current density: {current_field.shape}")

# 2. Load and analyze experimental data
print("\n2. EXPERIMENTAL DATA")
print("-" * 40)

# Load operational data
op_data = pd.read_csv(data_dir / "experimental" / "operational_data.csv")
print(f"Operational data: {len(op_data)} points over {op_data['time_hours'].max():.1f} hours")
print(f"Columns: {list(op_data.columns)}")

# Calculate degradation
initial_voltage = op_data['voltage_V'].iloc[0]
final_voltage = op_data['voltage_V'].iloc[-1]
degradation_rate = (initial_voltage - final_voltage) / initial_voltage * 100
print(f"\nVoltage degradation: {initial_voltage:.3f} → {final_voltage:.3f} V ({degradation_rate:.1f}%)")

# Load EIS data
with open(data_dir / "experimental" / "eis_data.json", "r") as f:
    eis_data = json.load(f)
print(f"\nEIS measurements: {len(eis_data)}")
print(f"Frequency range: {eis_data[0]['frequencies_Hz'][0]:.2e} - {eis_data[0]['frequencies_Hz'][-1]:.2e} Hz")

# Load thermal images
thermal_images = np.load(data_dir / "experimental" / "thermal_images.npy")
print(f"\nThermal images: {thermal_images.shape}")
print(f"Temperature range: {thermal_images.min():.1f} - {thermal_images.max():.1f} °C")

# Load strain data
strain_data = pd.read_csv(data_dir / "experimental" / "strain_gauge_data.csv")
strain_cols = [col for col in strain_data.columns if 'strain_sensor' in col]
print(f"\nStrain sensors: {len(strain_cols)}")
print(f"Data points: {len(strain_data)}")

# Load acoustic emission events
ae_events = pd.read_csv(data_dir / "experimental" / "acoustic_emission_events.csv")
print(f"\nAcoustic emission events: {len(ae_events)}")
if not ae_events.empty:
    print(f"Event types: {ae_events['event_type'].value_counts().to_dict()}")

# 3. Load monitoring stream data
print("\n3. MONITORING DATA")
print("-" * 40)

stream_data = pd.read_csv(data_dir / "monitoring" / "realtime_stream.csv")
print(f"Stream data points: {len(stream_data)}")
print(f"Duration: {len(stream_data)} seconds")
print(f"Mean power: {stream_data['power_W'].mean():.1f} ± {stream_data['power_W'].std():.1f} W")

# 4. Load data splits
print("\n4. DATA ORGANIZATION")
print("-" * 40)

with open(data_dir / "data_splits.json", "r") as f:
    splits = json.load(f)

print(f"Train set: {len(splits['train'])} simulations")
print(f"Validation set: {len(splits['validation'])} simulations")
print(f"Test set: {len(splits['test'])} simulations")

# 5. Simple visualizations
print("\n5. CREATING VISUALIZATIONS")
print("-" * 40)

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Temperature field (mid-plane)
ax = axes[0, 0]
im = ax.imshow(temperature_field[:, :, temperature_field.shape[2]//2], cmap='hot')
ax.set_title('Temperature Field (Mid-plane)')
ax.set_xlabel('X [grid]')
ax.set_ylabel('Y [grid]')
plt.colorbar(im, ax=ax)

# Plot 2: Stress field (mid-plane)
ax = axes[0, 1]
im = ax.imshow(stress_field[:, :, stress_field.shape[2]//2] / 1e6, cmap='plasma')
ax.set_title('Von Mises Stress [MPa]')
ax.set_xlabel('X [grid]')
ax.set_ylabel('Y [grid]')
plt.colorbar(im, ax=ax)

# Plot 3: Voltage degradation
ax = axes[0, 2]
ax.plot(op_data['time_hours'], op_data['voltage_V'], 'b-', alpha=0.7)
ax.set_title('Voltage Degradation')
ax.set_xlabel('Time [hours]')
ax.set_ylabel('Voltage [V]')
ax.grid(True, alpha=0.3)

# Plot 4: EIS Nyquist plot
ax = axes[1, 0]
for i in [0, len(eis_data)//2, len(eis_data)-1]:
    if i < len(eis_data):
        measurement = eis_data[i]
        ax.plot(measurement['Z_real_ohm'], 
               [-z for z in measurement['Z_imag_ohm']], 
               'o-', label=f't={measurement["time_hours"]:.0f}h')
ax.set_title('EIS Evolution')
ax.set_xlabel('Z_real [Ω]')
ax.set_ylabel('-Z_imag [Ω]')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Thermal image example
ax = axes[1, 1]
im = ax.imshow(thermal_images[len(thermal_images)//2], cmap='hot')
ax.set_title(f'Thermal Image #{len(thermal_images)//2}')
ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')
plt.colorbar(im, ax=ax)

# Plot 6: Real-time monitoring
ax = axes[1, 2]
time_seconds = np.arange(len(stream_data))
ax.plot(time_seconds[:500], stream_data['voltage_V'].iloc[:500], 'g-', alpha=0.7)
ax.set_title('Real-time Voltage Monitor (first 500s)')
ax.set_xlabel('Time [seconds]')
ax.set_ylabel('Voltage [V]')
ax.grid(True, alpha=0.3)

plt.suptitle('SOFC Digital Twin Dataset Overview', fontsize=16, fontweight='bold')
plt.tight_layout()

# Save figure
output_file = 'dataset_overview.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved visualization to {output_file}")

plt.show()

# 6. Statistical summary
print("\n6. STATISTICAL SUMMARY")
print("-" * 40)

# Simulation statistics
voltages = [sim['outputs']['voltage'] for sim in simulations]
max_stresses = [sim['outputs']['max_stress']/1e6 for sim in simulations]
creep_damages = [sim['outputs']['creep_damage'] for sim in simulations]

print("Simulation outputs:")
print(f"  Voltage: {np.mean(voltages):.3f} ± {np.std(voltages):.3f} V")
print(f"  Max stress: {np.mean(max_stresses):.1f} ± {np.std(max_stresses):.1f} MPa")
print(f"  Creep damage: {np.mean(creep_damages):.4f} ± {np.std(creep_damages):.4f}")

# Experimental statistics
print("\nExperimental measurements:")
print(f"  Mean current: {op_data['current_A'].mean():.1f} ± {op_data['current_A'].std():.1f} A")
print(f"  Mean efficiency: {op_data['efficiency'].mean():.3f} ± {op_data['efficiency'].std():.3f}")
print(f"  Temperature range: {op_data['T_outlet_C'].min():.1f} - {op_data['T_outlet_C'].max():.1f} °C")

# 7. Example of preparing data for ML
print("\n7. PREPARING DATA FOR MACHINE LEARNING")
print("-" * 40)

# Extract features and targets from simulations
X_features = []
y_targets = []

for sim in simulations[:10]:  # Use first 10 for example
    # Input features
    features = [
        sim['parameters']['current_density'],
        sim['parameters']['fuel_utilization'],
        sim['parameters']['air_utilization'],
        sim['parameters']['fuel_temperature'],
        sim['parameters']['air_temperature'],
        sim['parameters']['crack_length'],
        sim['parameters']['porosity_change']
    ]
    X_features.append(features)
    
    # Output targets
    targets = [
        sim['outputs']['voltage'],
        sim['outputs']['max_stress'],
        sim['outputs']['creep_damage']
    ]
    y_targets.append(targets)

X_features = np.array(X_features)
y_targets = np.array(y_targets)

print(f"Feature matrix shape: {X_features.shape}")
print(f"Target matrix shape: {y_targets.shape}")

# Normalize features (example)
X_normalized = (X_features - X_features.mean(axis=0)) / (X_features.std(axis=0) + 1e-8)
print(f"Normalized features range: [{X_normalized.min():.2f}, {X_normalized.max():.2f}]")

print("\n" + "=" * 80)
print("Example usage complete!")
print("Dataset is ready for:")
print("  - Physics-informed neural network training")
print("  - Digital twin development")
print("  - Degradation modeling")
print("  - Adaptive monitoring system implementation")
print("=" * 80)