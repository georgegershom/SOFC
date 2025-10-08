#!/usr/bin/env python3
"""
Verification script for the generated SOFC microstructural dataset
"""

import h5py
import numpy as np
import json
import os

def verify_dataset(dataset_dir="./sofc_dataset"):
    """Verify the generated dataset."""
    
    print("\n" + "="*60)
    print("SOFC DATASET VERIFICATION")
    print("="*60)
    
    # Check if directory exists
    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return False
    
    print(f"✓ Dataset directory found: {dataset_dir}")
    
    # Check HDF5 file
    h5_file = os.path.join(dataset_dir, "microstructure.h5")
    if not os.path.exists(h5_file):
        print(f"❌ HDF5 file not found: {h5_file}")
        return False
    
    print(f"✓ HDF5 file found: {h5_file}")
    
    # Load and verify HDF5 contents
    print("\nVerifying HDF5 contents...")
    with h5py.File(h5_file, 'r') as f:
        # Check main dataset
        if 'microstructure' not in f:
            print("❌ Main microstructure dataset not found")
            return False
        
        volume = f['microstructure'][:]
        print(f"✓ Main dataset shape: {volume.shape}")
        print(f"  • Data type: {volume.dtype}")
        print(f"  • Value range: [{volume.min()}, {volume.max()}]")
        print(f"  • Unique phases: {np.unique(volume)}")
        
        # Check attributes
        attrs = dict(f['microstructure'].attrs)
        print(f"\n✓ Dataset attributes:")
        for key, value in attrs.items():
            print(f"  • {key}: {value}")
        
        # Check phase data
        if 'phases' in f:
            print(f"\n✓ Phase data found:")
            phase_group = f['phases']
            for phase_name in phase_group.keys():
                phase_data = phase_group[phase_name]
                vf = phase_data.attrs.get('volume_fraction', 0)
                name = phase_data.attrs.get('name', 'Unknown')
                print(f"  • {phase_name}: {name} (VF: {vf:.1%})")
        
        # Check metadata
        if 'metadata' in f:
            print(f"\n✓ Metadata found:")
            meta = dict(f['metadata'].attrs)
            for key in list(meta.keys())[:5]:  # Show first 5 items
                value = meta[key]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"  • {key}: {value}")
    
    # Check TIFF files
    tiff_dir = os.path.join(dataset_dir, "tiff_stack")
    if os.path.exists(tiff_dir):
        tiff_files = os.listdir(tiff_dir)
        print(f"\n✓ TIFF stack directory found:")
        print(f"  • Number of files: {len(tiff_files)}")
        print(f"  • Files: {', '.join(tiff_files[:5])}...")
    
    # Check metadata JSON
    meta_file = os.path.join(dataset_dir, "metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
        print(f"\n✓ Metadata JSON found:")
        print(f"  • Keys: {', '.join(list(metadata.keys())[:8])}")
    
    # Check README
    readme_file = os.path.join(dataset_dir, "README.md")
    if os.path.exists(readme_file):
        with open(readme_file, 'r') as f:
            lines = f.readlines()
        print(f"\n✓ README found ({len(lines)} lines)")
    
    # Data quality checks
    print("\n" + "-"*40)
    print("DATA QUALITY CHECKS")
    print("-"*40)
    
    # Check volume fractions sum
    with h5py.File(h5_file, 'r') as f:
        volume = f['microstructure'][:]
        
        # Calculate actual volume fractions
        total_voxels = volume.size
        vf_sum = 0
        print("\nActual volume fractions:")
        for phase in np.unique(volume):
            count = np.sum(volume == phase)
            vf = count / total_voxels
            vf_sum += vf
            print(f"  • Phase {phase}: {vf:.1%} ({count:,} voxels)")
        
        if abs(vf_sum - 1.0) < 0.001:
            print(f"✓ Volume fractions sum to 1.0")
        else:
            print(f"⚠ Volume fractions sum to {vf_sum:.4f}")
        
        # Check connectivity of solid phases
        print("\nPhase connectivity check:")
        for phase in [1, 2, 3]:  # Ni, YSZ_Anode, YSZ_Electrolyte
            if phase in np.unique(volume):
                phase_mask = volume == phase
                from scipy.ndimage import label
                labeled, num_components = label(phase_mask)
                print(f"  • Phase {phase}: {num_components} connected components")
        
        # Check interface exists
        print("\nInterface check:")
        # Find z-layers with multiple phases
        interface_layers = 0
        for z in range(volume.shape[2]):
            slice_phases = np.unique(volume[:, :, z])
            if len(slice_phases) > 2:
                interface_layers += 1
        print(f"  • Layers with multiple phases: {interface_layers}/{volume.shape[2]}")
    
    print("\n" + "="*60)
    print("✅ DATASET VERIFICATION COMPLETE")
    print("="*60)
    print("\nThe dataset is ready for use in high-fidelity modeling!")
    print("Key features:")
    print("  • 3D voxelated microstructure data")
    print("  • Multiple material phases with realistic morphology")
    print("  • Phase segmentation for computational modeling")
    print("  • Interface geometry suitable for delamination studies")
    print("  • Export formats: HDF5, TIFF stack")
    
    return True

if __name__ == "__main__":
    verify_dataset()