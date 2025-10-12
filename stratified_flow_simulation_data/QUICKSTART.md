# Quick Start Guide

## Get Started in 5 Minutes

This guide will help you quickly load and visualize the stratified flow simulation data.

---

## Prerequisites

```bash
pip install numpy scipy matplotlib
```

---

## Step 1: Verify Dataset (30 seconds)

Check that all files are present:

```python
import os
import numpy as np

# Check directory structure
folders = ['cfd_outputs', 'mathematical_model_outputs', 'validation_data', 'figures']
for folder in folders:
    exists = os.path.isdir(folder)
    print(f"{folder}: {'✓' if exists else '✗'}")

# Check a few key files
files = [
    'cfd_outputs/velocity_u.npy',
    'mathematical_model_outputs/attenuation_dB.npy',
    'validation_data/validation_statistics.json'
]
for file in files:
    exists = os.path.isfile(file)
    size = os.path.getsize(file) / 1e6 if exists else 0
    print(f"{file}: {'✓' if exists else '✗'} ({size:.1f} MB)")
```

---

## Step 2: Load Your First Dataset (1 minute)

```python
import numpy as np
import matplotlib.pyplot as plt
import json

# Load velocity field
velocity_u = np.load('cfd_outputs/velocity_u.npy')
print(f"Velocity shape: {velocity_u.shape}")  # (100, 100, 50)

# Load coordinates
with open('cfd_outputs/coordinates.json', 'r') as f:
    coords = json.load(f)
x = np.array(coords['x'])
z = np.array(coords['z'])

# Create a simple plot
plt.figure(figsize=(10, 5))
plt.contourf(x, z, velocity_u[50, :, :].T, levels=20, cmap='RdBu_r')
plt.colorbar(label='Velocity u (m/s)')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title('Velocity Field (y = 0.5 m)')
plt.axhline(y=0.25, color='k', linestyle='--', label='Interface')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Step 3: Explore Acoustic Data (2 minutes)

```python
# Load acoustic pressure propagation
acoustic = np.load('cfd_outputs/acoustic_pressure.npy')
print(f"Acoustic data shape: {acoustic.shape}")  # (1000, 100, 50)

# Extract time series at a specific location
# Point at x=0.5m, z=0.2m (in liquid layer)
t = np.array(coords['t'])
ix, iz = 50, 20
pressure_vs_time = acoustic[:, ix, iz]

# Plot time series
plt.figure(figsize=(10, 4))
plt.plot(t * 1000, pressure_vs_time)
plt.xlabel('Time (ms)')
plt.ylabel('Acoustic Pressure (Pa)')
plt.title(f'Acoustic Pressure at x={x[ix]:.2f}m, z={z[iz]:.2f}m')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot snapshot at t=0.5s
time_idx = 500
plt.figure(figsize=(10, 5))
plt.contourf(x, z, acoustic[time_idx, :, :].T, levels=30, cmap='seismic')
plt.colorbar(label='Pressure (Pa)')
plt.xlabel('x (m)')
plt.ylabel('z (m)')
plt.title(f'Acoustic Pressure at t = {t[time_idx]*1000:.1f} ms')
plt.axhline(y=0.25, color='k', linestyle='--', linewidth=2)
plt.tight_layout()
plt.show()
```

---

## Step 4: Analyze Mathematical Models (1 minute)

```python
# Load attenuation coefficients
attenuation_dB = np.load('mathematical_model_outputs/attenuation_dB.npy')

# Load parameters
with open('mathematical_model_outputs/parameters.json', 'r') as f:
    params = json.load(f)
frequencies = np.array(params['frequencies'])
void_fractions = np.array(params['void_fractions'])

# Plot attenuation vs frequency for different void fractions
plt.figure(figsize=(10, 6))
for i, alpha in enumerate([0, 5, 10, 15, 19]):
    plt.loglog(frequencies, attenuation_dB[:, alpha], 
               linewidth=2, label=f'α = {void_fractions[alpha]:.2f}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Attenuation (dB/m)')
plt.title('Attenuation vs Frequency')
plt.grid(True, alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Step 5: View Validation Results (1 minute)

```python
# Load validation statistics
with open('validation_data/validation_statistics.json', 'r') as f:
    stats = json.load(f)

# Print statistics
print("Validation Statistics:")
print("=" * 50)
for category, metrics in stats.items():
    print(f"\n{category.upper()}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Load and plot sound speed comparison
with open('validation_data/sound_speed_comparison.json', 'r') as f:
    speed_data = json.load(f)

alpha = np.array(speed_data['void_fraction'])
sim = np.array(speed_data['simulated'])
exp = np.array(speed_data['experimental'])
unc = np.array(speed_data['uncertainty'])

plt.figure(figsize=(10, 6))
plt.errorbar(alpha, exp, yerr=unc, fmt='ro', markersize=8, 
             capsize=5, label='Experimental', linewidth=2)
plt.plot(alpha, sim, 'b-', linewidth=2, label='Simulated')
plt.xlabel('Void Fraction α')
plt.ylabel('Sound Speed (m/s)')
plt.title('Sound Speed: Simulated vs Experimental')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

---

## Bonus: Generate All Figures

To generate all publication-quality figures at once:

```bash
python3 visualize_data.py
```

This creates 8 figures in the `figures/` directory:
1. `velocity_fields.png` - 3D velocity components
2. `vof_pressure.png` - Phase distribution and pressure
3. `turbulence_parameters.png` - k, ε, μ_t fields
4. `acoustic_propagation.png` - Time snapshots of acoustic waves
5. `sound_speed_predictions.png` - Sound speed models
6. `attenuation_coefficients.png` - Attenuation analysis
7. `wave_propagation.png` - Reflection and standing waves
8. `validation_comparison.png` - Model validation plots

---

## Common Tasks

### Task 1: Extract 1D Profile

```python
# Extract vertical profile at a specific x-y location
u = np.load('cfd_outputs/velocity_u.npy')
z = np.linspace(0, 0.5, 50)

# Profile at x=0.5m, y=0.5m
profile = u[50, 50, :]

plt.figure(figsize=(6, 8))
plt.plot(profile, z, 'b-', linewidth=2)
plt.ylabel('Height z (m)')
plt.xlabel('Velocity u (m/s)')
plt.axhline(y=0.25, color='r', linestyle='--', label='Interface')
plt.grid(True)
plt.legend()
plt.show()
```

### Task 2: Compute Average Properties

```python
# Compute mean velocity in each layer
u = np.load('cfd_outputs/velocity_u.npy')

# Lower layer (z < 0.25m, indices 0-25)
u_lower = u[:, :, 0:25].mean()

# Upper layer (z > 0.25m, indices 25-50)
u_upper = u[:, :, 25:50].mean()

print(f"Mean velocity in lower layer: {u_lower:.3f} m/s")
print(f"Mean velocity in upper layer: {u_upper:.3f} m/s")
print(f"Velocity ratio: {u_upper/u_lower:.2f}")
```

### Task 3: Export Data to CSV

```python
# Export sound speed data to CSV
with open('mathematical_model_outputs/parameters.json', 'r') as f:
    params = json.load(f)

sound_speed = np.load('mathematical_model_outputs/sound_speed_wood.npy')
void_fractions = np.array(params['void_fractions'])

# Create CSV
import pandas as pd
df = pd.DataFrame({
    'void_fraction': void_fractions,
    'sound_speed': sound_speed
})
df.to_csv('sound_speed_export.csv', index=False)
print("Exported to sound_speed_export.csv")
```

---

## Next Steps

1. **Read the Full Documentation**: See `README.md` for complete details
2. **Review Data Summary**: Check `DATASET_SUMMARY.md` for quick reference
3. **Explore Figures**: View generated plots in `figures/` directory
4. **Develop Your Models**: Use the data for your research
5. **Validate Your Results**: Compare with validation data provided

---

## Getting Help

- **Documentation**: See `README.md` and `DATASET_SUMMARY.md`
- **Code Examples**: Check the `visualize_data.py` script
- **File Structure**: Review directory listings in `README.md`

---

## Troubleshooting

**Problem**: "FileNotFoundError"  
**Solution**: Make sure you're in the `stratified_flow_simulation_data/` directory

**Problem**: "ModuleNotFoundError: No module named 'numpy'"  
**Solution**: Install required packages: `pip install numpy scipy matplotlib`

**Problem**: "MemoryError" when loading large files  
**Solution**: Use memory mapping:
```python
acoustic = np.load('cfd_outputs/acoustic_pressure.npy', mmap_mode='r')
```

**Problem**: Plots not showing  
**Solution**: Add `plt.show()` at the end or use `plt.savefig('output.png')`

---

Happy analyzing! 🎉
