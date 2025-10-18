#!/usr/bin/env python3
"""
Generate Synthetic Micro-CT Image Descriptions and Metadata
This script creates detailed image metadata that would accompany actual Micro-CT scans

Author: Generated for PhD Research
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def generate_microct_image_catalog():
    """Generate catalog of Micro-CT scan images with metadata"""
    
    catalog = []
    
    rubber_contents = [0, 5, 10, 15, 20]
    temperatures = [20, 200, 400, 600, 800]
    
    for rubber in rubber_contents:
        for temp in temperatures:
            for replicate in [1, 2, 3]:
                
                # Generate specimen ID
                if temp == 20:
                    specimen_id = f"CTRL-R{rubber}-RT-{replicate:02d}"
                else:
                    specimen_id = f"HEAT-R{rubber}-{temp}-{replicate:02d}"
                
                # Scan parameters
                scan_info = {
                    'specimen_id': specimen_id,
                    'rubber_content_pct': rubber,
                    'temperature_C': temp,
                    'replicate': replicate,
                    
                    # Acquisition parameters
                    'scan_date': f'2024-{np.random.randint(1,13):02d}-{np.random.randint(1,29):02d}',
                    'x_ray_voltage_kV': 80,
                    'x_ray_current_uA': 125,
                    'resolution_um': 5.0 + np.random.uniform(-0.2, 0.2),
                    'rotation_step_deg': 0.4,
                    'number_of_projections': 900,
                    'exposure_time_ms': 850 + np.random.randint(-50, 50),
                    'frame_averaging': 3,
                    'al_filter_mm': 1.0,
                    
                    # Reconstruction parameters
                    'ring_artifact_correction': 8,
                    'beam_hardening_correction_pct': 40,
                    'reconstruction_software': 'NRecon v1.7.4.6',
                    
                    # Image stack info
                    'number_of_slices': 800 + np.random.randint(-50, 50),
                    'slice_thickness_um': 5.0,
                    'image_width_pixels': 1024,
                    'image_height_pixels': 1024,
                    'bit_depth': 16,
                    'file_format': 'BMP',
                    'total_file_size_GB': np.random.uniform(0.8, 1.5),
                    
                    # Analysis notes
                    'segmentation_method': 'Otsu + Manual',
                    'porosity_software': 'CTAn v1.18',
                    'crack_analysis_software': 'Avizo 9.0',
                    '3d_visualization_software': 'Dragonfly 2022.1',
                    
                    # Quality metrics
                    'signal_to_noise_ratio': 15 + np.random.uniform(-2, 5),
                    'contrast_to_noise_ratio': 8 + np.random.uniform(-1, 3),
                    'spatial_resolution_um': 6.5 + np.random.uniform(-0.5, 1.0),
                    
                    # Key observations
                    'observations': generate_observations(rubber, temp)
                }
                
                catalog.append(scan_info)
    
    return pd.DataFrame(catalog)

def generate_observations(rubber, temp):
    """Generate qualitative observations based on conditions"""
    
    obs = []
    
    if temp == 20:
        obs.append("No thermal damage")
        obs.append("Well-defined phase boundaries")
        if rubber > 0:
            obs.append(f"Rubber particles clearly visible ({rubber}%)")
            obs.append("ITZ at rubber-paste interface evident")
    
    if temp == 200:
        obs.append("Minor thermal microcracking")
        obs.append("Ettringite decomposition evident")
        if rubber > 0:
            obs.append("Initial rubber surface degradation")
            obs.append("Small voids forming at ITZ")
    
    if temp == 400:
        obs.append("Severe thermal cracking")
        obs.append("Major porosity increase")
        if rubber > 0:
            obs.append("Extensive rubber decomposition")
            obs.append("Large voids replacing rubber particles")
        else:
            obs.append("Paste densification in some regions")
    
    if temp == 600:
        obs.append("Massive porosity increase")
        obs.append("Interconnected crack networks")
        if rubber > 0:
            obs.append("Complete rubber burnout")
            obs.append("Only residual ash structure in voids")
        obs.append("Severe paste degradation")
    
    if temp == 800:
        obs.append("Extreme material degradation")
        obs.append("Sintering effects visible")
        if rubber > 0:
            obs.append("Total rubber combustion")
            obs.append("Massive void spaces")
        obs.append("Phase transformation evident")
    
    return "; ".join(obs)

def generate_typical_slice_descriptions():
    """Generate descriptions of typical 2D slices at different conditions"""
    
    descriptions = {
        'R0-RT': """
            Control specimen at room temperature shows dense cement paste with 
            well-defined aggregate particles. Porosity is minimal (~8%) with 
            most pores <10 μm. ITZ around aggregates is 20-30 μm thick with 
            slightly higher porosity than bulk paste. No visible cracks.
        """,
        
        'R10-RT': """
            10% rubber specimen at room temperature shows distributed rubber 
            particles (0.5-2 mm) with distinct ITZ zones. Porosity is elevated 
            (~16%) due to rubber particles and ITZ effects. Rubber particles 
            show clean boundaries with paste. No thermal damage evident.
        """,
        
        'R0-400C': """
            Control specimen after 400°C exposure shows significant thermal 
            cracking. Porosity increased to ~15% with new microcracks throughout 
            paste. Some cracks are >50 μm wide. Aggregate boundaries show slight 
            separation. Paste appears more porous due to water loss.
        """,
        
        'R10-400C': """
            10% rubber specimen after 400°C exposure shows catastrophic changes. 
            Porosity jumped to ~40% with large voids where rubber particles were. 
            Rubber has decomposed leaving irregular void spaces. Extensive 
            cracking radiates from void locations. ITZ zones are severely degraded.
        """,
        
        'R10-600C': """
            10% rubber specimen after 600°C exposure shows near-total degradation. 
            Porosity exceeds 70% with massive interconnected void network. All 
            rubber has combusted. Cracks connect most voids creating highly 
            permeable structure. Paste is severely decomposed with only skeleton 
            remaining.
        """,
        
        'R20-800C': """
            20% rubber specimen after 800°C exposure shows complete disintegration. 
            Porosity approaches 98% with only residual ash structure. Material 
            has essentially failed structurally. Some sintering of residual 
            particles visible. Original microstructure unrecognizable.
        """
    }
    
    return descriptions

def main():
    """Generate all synthetic Micro-CT image metadata"""
    
    print("Generating Micro-CT Image Catalog...")
    
    # Generate catalog
    catalog = generate_microct_image_catalog()
    
    # Save catalog
    output_dir = Path('../microct_data')
    output_dir.mkdir(exist_ok=True)
    
    catalog.to_csv(output_dir / 'microct_scan_catalog.csv', index=False)
    print(f"✓ Generated catalog with {len(catalog)} scans")
    
    # Generate descriptions
    descriptions = generate_typical_slice_descriptions()
    
    with open(output_dir / 'typical_slice_descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=2)
    print("✓ Generated typical slice descriptions")
    
    # Generate README for images
    readme_content = """# Micro-CT Image Data

## Overview

This folder contains metadata and descriptions for Micro-CT scan images. 
In a real research project, this would also contain:

- Raw projection images (.tif)
- Reconstructed slice stacks (.bmp or .tif)
- 3D volume files (.vol or .raw)
- Segmented volumes (.am or .tif)
- 3D renderings (.png or .mp4)

## File Structure (Typical)

```
microct_images/
├── raw_projections/
│   ├── CTRL-R0-RT-01/
│   │   ├── projection_0000.tif
│   │   ├── projection_0001.tif
│   │   └── ... (900 projections)
│   └── ...
├── reconstructed_slices/
│   ├── CTRL-R0-RT-01/
│   │   ├── slice_0001.bmp
│   │   ├── slice_0002.bmp
│   │   └── ... (800 slices)
│   └── ...
├── 3d_volumes/
│   ├── CTRL-R0-RT-01.vol
│   └── ...
└── segmented/
    ├── CTRL-R0-RT-01_porosity.tif
    ├── CTRL-R0-RT-01_cracks.tif
    └── ...
```

## Data Size

- Single scan: ~1-1.5 GB
- Total dataset: ~150-200 GB (75 scans)
- Reconstructed slices: 1024×1024×800 voxels @ 16-bit
- Voxel size: 5 μm

## How to View

1. **ImageJ/Fiji**: Open slice stacks
2. **Dragonfly**: Load .vol files for 3D visualization
3. **Avizo**: Advanced 3D analysis
4. **ParaView**: Volume rendering

## Analysis Workflow

1. Load reconstructed slices
2. Apply filters (median, Gaussian)
3. Segment phases (Otsu, watershed)
4. Analyze porosity (CTAn)
5. Extract crack networks (Avizo)
6. Export quantitative data

## Notes

This is a synthetic dataset for demonstration. Real Micro-CT images would 
show actual gray-scale intensities, artifacts, and noise typical of X-ray 
computed tomography.
"""
    
    with open(output_dir / 'IMAGE_README.md', 'w') as f:
        f.write(readme_content)
    print("✓ Generated image README")
    
    print("\n" + "="*60)
    print("Micro-CT Image Catalog Generation Complete!")
    print("="*60)
    print(f"\nTotal scans cataloged: {len(catalog)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("\nFiles created:")
    print("  - microct_scan_catalog.csv")
    print("  - typical_slice_descriptions.json")
    print("  - IMAGE_README.md")

if __name__ == "__main__":
    main()
