#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

print("Starting test generation...")

# Simple test parameters
resolution = (64, 64, 32)  # Smaller for testing
voxel_size = 0.1

print(f"Resolution: {resolution}")
print(f"Voxel size: {voxel_size}")

# Create simple microstructure
microstructure = np.zeros(resolution, dtype=np.uint8)

# Add some simple phases
# Pore phase (0) - random spheres
from skimage import morphology
pore_mask = np.zeros(resolution, dtype=bool)
n_pores = 20
for _ in range(n_pores):
    center = np.random.uniform(0, min(resolution), 3)
    radius = np.random.uniform(2, 5)
    x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    sphere = dist <= radius
    pore_mask |= sphere

microstructure[pore_mask] = 0  # Pore

# Ni phase (1) - some spheres
ni_mask = np.zeros(resolution, dtype=bool)
n_ni = 10
for _ in range(n_ni):
    center = np.random.uniform(0, min(resolution), 3)
    radius = np.random.uniform(1, 3)
    x, y, z = np.ogrid[:resolution[0], :resolution[1], :resolution[2]]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    sphere = dist <= radius
    ni_mask |= sphere & ~pore_mask

microstructure[ni_mask] = 1  # Ni

# YSZ anode (2) - remaining solid
ysz_mask = ~pore_mask & ~ni_mask
microstructure[ysz_mask] = 2  # YSZ anode

# YSZ electrolyte (3) - top layer
electrolyte_thickness = 8
start_z = resolution[2] - electrolyte_thickness
if start_z >= 0:
    microstructure[:, :, start_z:] = 3  # YSZ electrolyte

print("Microstructure generated successfully!")
print(f"Shape: {microstructure.shape}")
print(f"Unique values: {np.unique(microstructure)}")

# Calculate phase fractions
total_voxels = np.prod(resolution)
phase_counts = {}
phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte'}

for phase_id, name in phase_names.items():
    count = np.sum(microstructure == phase_id)
    phase_counts[name] = count / total_voxels
    print(f"{name}: {count} voxels ({count/total_voxels:.3f})")

# Create output directory
os.makedirs('output', exist_ok=True)

# Save HDF5
print("Saving HDF5...")
with h5py.File('output/test_microstructure.h5', 'w') as f:
    f.create_dataset('microstructure', data=microstructure, compression='gzip')
    f.attrs['resolution'] = resolution
    f.attrs['voxel_size_um'] = voxel_size

# Save TIFF stack
print("Saving TIFF stack...")
import tifffile
for z in range(resolution[2]):
    slice_data = microstructure[:, :, z]
    filename = f"output/test_microstructure_z{z:03d}.tif"
    tifffile.imwrite(filename, slice_data.astype(np.uint8))

# Create visualization
print("Creating visualization...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

z_slices = [resolution[2]//4, resolution[2]//2, 3*resolution[2]//4, resolution[2]-1]
colors = {0: 'white', 1: 'gold', 2: 'lightblue', 3: 'darkblue'}

for i, z in enumerate(z_slices):
    if i >= 4:
        break
        
    slice_data = microstructure[:, :, z]
    
    # Create colored image
    colored_slice = np.zeros((*slice_data.shape, 3))
    for phase_id, color in colors.items():
        mask = slice_data == phase_id
        if color == 'white':
            colored_slice[mask] = [1, 1, 1]
        elif color == 'gold':
            colored_slice[mask] = [1, 0.84, 0]
        elif color == 'lightblue':
            colored_slice[mask] = [0.68, 0.85, 0.9]
        elif color == 'darkblue':
            colored_slice[mask] = [0, 0, 0.55]
    
    axes[i].imshow(colored_slice)
    axes[i].set_title(f'Z = {z * voxel_size:.1f} μm')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('output/test_microstructure_slices.png', dpi=300, bbox_inches='tight')
plt.close()

# Phase distribution plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

phases = list(phase_counts.keys())
fractions = list(phase_counts.values())
color_list = ['white', 'gold', 'lightblue', 'darkblue']

bars = ax1.bar(phases, fractions, color=color_list, edgecolor='black')
ax1.set_title('Phase Volume Fractions')
ax1.set_ylabel('Volume Fraction')
ax1.tick_params(axis='x', rotation=45)

for bar, fraction in zip(bars, fractions):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{fraction:.3f}', ha='center', va='bottom')

wedges, texts, autotexts = ax2.pie(fractions, labels=phases, colors=color_list, 
                                  autopct='%1.1f%%', startangle=90)
ax2.set_title('Phase Distribution')

plt.tight_layout()
plt.savefig('output/test_phase_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("TEST GENERATION COMPLETE")
print("="*50)
print("Files created:")
print("  - output/test_microstructure.h5")
print("  - output/test_microstructure_z*.tif")
print("  - output/test_microstructure_slices.png")
print("  - output/test_phase_distribution.png")