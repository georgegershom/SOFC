"""
Simplified SOFC Dataset Generator - Works with minimal dependencies
"""

import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import csv

# Create output directory structure
output_dir = Path("data")
for subdir in ['simulation', 'experimental', 'monitoring']:
    (output_dir / subdir).mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SOFC Digital Twin Dataset Generation (Simplified Version)")
print("=" * 80)

# 1. Generate simplified simulation data
print("\n[1/5] Generating simulation data...")

n_simulations = 100
simulation_data = []

# Parameter ranges
param_ranges = {
    'current_density': (1000, 10000),  # A/m²
    'fuel_utilization': (0.3, 0.85),
    'air_utilization': (0.2, 0.5),
    'fuel_temperature': (600, 800),  # °C
    'air_temperature': (500, 700),  # °C
    'crack_length': (0, 5),  # mm
    'porosity_change': (0.9, 1.2)  # relative
}

# Generate Latin Hypercube samples manually
np.random.seed(42)
for i in range(n_simulations):
    # Sample parameters
    params = {}
    for param, (min_val, max_val) in param_ranges.items():
        params[param] = np.random.uniform(min_val, max_val)
    
    # Simulate outputs (simplified physics model)
    voltage = 0.8 - 0.00002 * params['current_density'] - 0.1 * params['fuel_utilization']
    max_stress = 50e6 + 5e6 * params['fuel_utilization'] + 10e6 * params['crack_length'] / 5
    creep_damage = params['crack_length'] / 5 * 0.1 + np.random.uniform(0, 0.05)
    
    # Generate field data (smaller grids for simplicity)
    grid_size = (20, 20, 5)
    temperature_field = 800 + 50 * np.random.randn(*grid_size)
    stress_field = max_stress * (1 + 0.2 * np.random.randn(*grid_size))
    current_density_field = params['current_density'] * (1 + 0.1 * np.random.randn(*grid_size))
    
    sim_result = {
        'sim_id': i,
        'parameters': params,
        'outputs': {
            'voltage': float(voltage),
            'max_stress': float(max_stress),
            'creep_damage': float(creep_damage)
        },
        'field_stats': {
            'temperature_mean': float(temperature_field.mean()),
            'temperature_std': float(temperature_field.std()),
            'stress_mean': float(stress_field.mean()),
            'stress_std': float(stress_field.std()),
            'current_density_mean': float(current_density_field.mean()),
            'current_density_std': float(current_density_field.std())
        }
    }
    
    simulation_data.append(sim_result)
    
    # Save field data as numpy arrays
    np.save(output_dir / 'simulation' / f'temperature_{i:04d}.npy', temperature_field)
    np.save(output_dir / 'simulation' / f'stress_{i:04d}.npy', stress_field)
    np.save(output_dir / 'simulation' / f'current_density_{i:04d}.npy', current_density_field)

# Save simulation metadata
with open(output_dir / 'simulation' / 'simulation_metadata.json', 'w') as f:
    json.dump(simulation_data, f, indent=2)

print(f"  Generated {n_simulations} simulations")

# 2. Generate experimental operational data
print("\n[2/5] Generating experimental data...")

duration_hours = 1000
n_samples = duration_hours * 10  # 0.1 Hz sampling

op_data = []
for i in range(n_samples):
    time_h = i / 10
    
    # Simulate degradation
    degradation = 1 - 0.00005 * time_h
    load = 0.5 + 0.3 * np.sin(2 * np.pi * time_h / 100)
    
    row = {
        'time_hours': time_h,
        'current_A': 50 * load + np.random.normal(0, 0.1),
        'voltage_V': (0.8 - 0.1 * load) * degradation + np.random.normal(0, 0.001),
        'T_inlet_fuel_C': 700 + 50 * load + np.random.normal(0, 0.5),
        'T_inlet_air_C': 650 + 50 * load + np.random.normal(0, 0.5),
        'T_outlet_C': 800 + 50 * load + np.random.normal(0, 0.5),
        'fuel_flow_slpm': 10 * (1 + 0.5 * load) + np.random.normal(0, 0.02),
        'air_flow_slpm': 100 * (1 + 0.5 * load) + np.random.normal(0, 0.2)
    }
    row['power_W'] = row['current_A'] * row['voltage_V']
    row['efficiency'] = row['voltage_V'] / 1.253 * degradation
    
    op_data.append(row)

# Save operational data as CSV
with open(output_dir / 'experimental' / 'operational_data.csv', 'w', newline='') as f:
    if op_data:
        writer = csv.DictWriter(f, fieldnames=op_data[0].keys())
        writer.writeheader()
        writer.writerows(op_data)

print(f"  Generated {len(op_data)} operational data points")

# Generate EIS data
n_eis = 50
eis_data = []
frequencies = np.logspace(-2, 5, 30)  # 0.01 Hz to 100 kHz

for i in range(n_eis):
    degradation = 1 + 0.01 * i
    R_ohm = 0.1 * degradation
    R_ct = 0.2 * degradation
    
    omega = 2 * np.pi * frequencies
    Z_real = R_ohm + R_ct / (1 + (omega * R_ct * 1e-5)**2)
    Z_imag = -R_ct * omega * 1e-5 / (1 + (omega * R_ct * 1e-5)**2) - 0.05 / np.sqrt(omega)
    
    eis_measurement = {
        'measurement_id': i,
        'time_hours': i * 20,
        'frequencies_Hz': frequencies.tolist(),
        'Z_real_ohm': Z_real.tolist(),
        'Z_imag_ohm': Z_imag.tolist(),
        'R_ohmic': float(R_ohm),
        'R_charge_transfer': float(R_ct)
    }
    eis_data.append(eis_measurement)

with open(output_dir / 'experimental' / 'eis_data.json', 'w') as f:
    json.dump(eis_data, f, indent=2)

print(f"  Generated {n_eis} EIS measurements")

# Generate thermal images
n_thermal = 100
thermal_images = []
for i in range(n_thermal):
    # Simple 32x32 thermal image
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)
    
    T_base = 750 + 50 * np.sin(2*np.pi*X) * np.cos(2*np.pi*Y)
    T_base += np.random.normal(0, 2, (32, 32))
    T_base += 10 * (i / n_thermal) * (X + Y) / 2  # Degradation pattern
    
    thermal_images.append(T_base)

thermal_images = np.array(thermal_images)
np.save(output_dir / 'experimental' / 'thermal_images.npy', thermal_images)
print(f"  Generated {n_thermal} thermal images")

# Generate strain gauge data
n_sensors = 8
strain_data = []
for i in range(n_samples):
    time_h = i / 10
    row = {'time_hours': time_h}
    
    for sensor_id in range(n_sensors):
        thermal_strain = 100e-6 * np.sin(2 * np.pi * time_h / 24)
        mechanical_strain = 50e-6 * (1 + 0.2 * np.sin(2 * np.pi * time_h / 100))
        creep_strain = 10e-6 * time_h / duration_hours
        total_strain = thermal_strain + mechanical_strain + creep_strain + np.random.normal(0, 1e-6)
        row[f'strain_sensor_{sensor_id}_mu_strain'] = total_strain * 1e6

    strain_data.append(row)

with open(output_dir / 'experimental' / 'strain_gauge_data.csv', 'w', newline='') as f:
    if strain_data:
        writer = csv.DictWriter(f, fieldnames=strain_data[0].keys())
        writer.writeheader()
        writer.writerows(strain_data)

print(f"  Generated strain gauge data from {n_sensors} sensors")

# Generate acoustic emission events
n_events = np.random.poisson(50)
ae_events = []
for i in range(n_events):
    event = {
        'event_id': i,
        'time_hours': duration_hours * (0.5 + 0.5 * np.random.beta(2, 1)),
        'amplitude_dB': np.random.lognormal(3, 0.5),
        'duration_us': np.random.lognormal(2, 0.5),
        'location_mm': np.random.uniform(0, 100),
        'event_type': np.random.choice(['crack_initiation', 'crack_propagation', 'delamination'])
    }
    event['energy'] = event['amplitude_dB'] * event['duration_us'] / 1000
    ae_events.append(event)

with open(output_dir / 'experimental' / 'acoustic_emission_events.csv', 'w', newline='') as f:
    if ae_events:
        writer = csv.DictWriter(f, fieldnames=ae_events[0].keys())
        writer.writeheader()
        writer.writerows(ae_events)

print(f"  Generated {n_events} acoustic emission events")

# 3. Generate monitoring stream data
print("\n[3/5] Generating monitoring data...")

duration_seconds = 3600
stream_data = []
current_state = {'voltage': 0.75, 'current': 50.0, 'temperature': 800.0}

for i in range(duration_seconds):
    current_state['current'] += np.random.normal(0, 0.1)
    current_state['voltage'] -= 1e-6
    current_state['temperature'] += np.random.normal(0, 0.5)
    
    if np.random.rand() < 0.01:  # 1% chance of disturbance
        current_state['current'] += np.random.normal(0, 5)
    
    timestamp = (datetime.now() + timedelta(seconds=i)).isoformat()
    
    row = {
        'timestamp': timestamp,
        'voltage_V': current_state['voltage'] + np.random.normal(0, 0.001),
        'current_A': current_state['current'] + np.random.normal(0, 0.01),
        'temperature_C': current_state['temperature'] + np.random.normal(0, 0.5),
        'fuel_flow_slpm': 10.0 + np.random.normal(0, 0.1),
        'air_flow_slpm': 100.0 + np.random.normal(0, 1.0),
        'power_W': current_state['voltage'] * current_state['current']
    }
    stream_data.append(row)

with open(output_dir / 'monitoring' / 'realtime_stream.csv', 'w', newline='') as f:
    if stream_data:
        writer = csv.DictWriter(f, fieldnames=stream_data[0].keys())
        writer.writeheader()
        writer.writerows(stream_data)

print(f"  Generated {len(stream_data)} real-time data packets")

# Generate adaptive triggers
triggers = []
voltage_baseline = stream_data[0]['voltage_V'] if stream_data else 0.75
for i, data in enumerate(stream_data):
    if data['voltage_V'] < 0.95 * voltage_baseline:
        triggers.append({
            'time': i / 3600,
            'trigger_type': 'voltage_drop',
            'severity': 'medium',
            'action': 'update_degradation_model'
        })
    
    if data['temperature_C'] > 850:
        triggers.append({
            'time': i / 3600,
            'trigger_type': 'temperature_excursion',
            'severity': 'high',
            'action': 'high_fidelity_thermal_analysis'
        })

if triggers:
    with open(output_dir / 'monitoring' / 'adaptive_triggers.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=triggers[0].keys())
        writer.writeheader()
        writer.writerows(triggers)
    print(f"  Generated {len(triggers)} adaptive trigger events")

# 4. Create data splits
print("\n[4/5] Creating train/validation/test splits...")

indices = list(range(n_simulations))
np.random.shuffle(indices)

train_size = int(0.7 * n_simulations)
val_size = int(0.15 * n_simulations)

splits = {
    'train': indices[:train_size],
    'validation': indices[train_size:train_size+val_size],
    'test': indices[train_size+val_size:]
}

with open(output_dir / 'data_splits.json', 'w') as f:
    json.dump(splits, f, indent=2)

print(f"  Train: {len(splits['train'])} samples")
print(f"  Validation: {len(splits['validation'])} samples")
print(f"  Test: {len(splits['test'])} samples")

# 5. Generate metadata
print("\n[5/5] Creating metadata...")

metadata = {
    'dataset_name': 'SOFC Digital Twin Multi-Fidelity Dataset',
    'version': '1.0-simplified',
    'creation_date': datetime.now().isoformat(),
    'description': 'Simplified multi-physics dataset for SOFC digital twin',
    'data_sources': {
        'simulation': {
            'type': 'Simplified physics simulation',
            'n_samples': n_simulations,
            'grid_size': [20, 20, 5],
            'physics': ['electrochemical', 'thermal', 'structural']
        },
        'experimental': {
            'type': 'Synthetic experimental data',
            'duration_hours': duration_hours,
            'measurements': ['voltage', 'current', 'temperature', 'EIS', 'thermal_imaging', 
                           'strain_gauge', 'acoustic_emission']
        },
        'monitoring': {
            'type': 'Real-time monitoring streams',
            'frequency_hz': 1.0,
            'duration_seconds': duration_seconds
        }
    },
    'parameter_ranges': param_ranges,
    'units': {
        'current_density': 'A/m²',
        'temperature': '°C',
        'stress': 'Pa',
        'strain': 'dimensionless',
        'voltage': 'V',
        'current': 'A'
    }
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Calculate dataset size
total_size = 0
for dirpath, dirnames, filenames in os.walk(output_dir):
    for filename in filenames:
        filepath = os.path.join(dirpath, filename)
        total_size += os.path.getsize(filepath)

print("\n" + "=" * 80)
print("Dataset generation completed successfully!")
print(f"Total dataset size: ~{total_size / (1024 * 1024):.2f} MB")
print(f"Dataset saved to: {output_dir.absolute()}")
print("=" * 80)

print("\nDataset Contents:")
print("1. Simulation Data:")
print(f"   - {n_simulations} multi-physics simulations")
print("   - Temperature, stress, and current density fields")
print("   - Voltage, max stress, and creep damage outputs")

print("\n2. Experimental Data:")
print(f"   - {len(op_data)} operational data points over {duration_hours} hours")
print(f"   - {n_eis} EIS measurements")
print(f"   - {n_thermal} thermal images (32x32)")
print(f"   - {n_sensors} strain gauge sensors")
print(f"   - {n_events} acoustic emission events")

print("\n3. Monitoring Data:")
print(f"   - {duration_seconds} seconds of real-time stream at 1 Hz")
print(f"   - {len(triggers) if triggers else 0} adaptive trigger events")

print("\n4. Data Organization:")
print("   - Train/validation/test splits included")
print("   - Metadata and parameter ranges documented")
print("   - All data in standard formats (JSON, CSV, NPY)")

print("\nNext Steps:")
print("1. Use the visualization tools to explore the data")
print("2. Load data using the provided data_loader.py")
print("3. Train Physics-Informed Neural Networks")
print("4. Validate with experimental data")
print("5. Test adaptive monitoring system")