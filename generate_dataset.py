#!/usr/bin/env python3
"""
Simplified script to generate and export SOFC microstructural dataset
"""

import numpy as np
import os
import sys
import time
from datetime import datetime
import json
import h5py
import tifffile

# Import our generator
from sofc_microstructure_generator import SOFCMicrostructureGenerator

def main():
    """Generate the SOFC microstructural dataset."""
    
    print("\n" + "="*80)
    print("SOFC 3D MICROSTRUCTURAL DATASET GENERATOR")
    print("="*80)
    print("\nGenerating realistic 3D microstructure for SOFC electrode modeling...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Create output directory
    output_dir = "./sofc_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration
    print("Configuration:")
    print("  • Dimensions: 128 x 128 x 64 voxels")
    print("  • Voxel size: 0.5 µm")
    print("  • Physical size: 64 x 64 x 32 µm")
    print("  • Phases: Pore, Ni, YSZ (anode), YSZ (electrolyte), GDC interlayer")
    print()
    
    # Step 1: Generate microstructure
    print("STEP 1: Generating 3D microstructure...")
    print("-" * 40)
    
    start_time = time.time()
    
    try:
        generator = SOFCMicrostructureGenerator(
            dimensions=(128, 128, 64),
            voxel_size=0.5,
            seed=42
        )
        
        # Generate the structure
        structure = generator.generate_full_structure(include_interlayer=True)
        
        gen_time = time.time() - start_time
        print(f"✓ Generation completed in {gen_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Error during generation: {e}")
        return 1
    
    # Display results
    print("\nGenerated structure properties:")
    print("-" * 40)
    
    if 'volume_fractions' in generator.metadata:
        print("\nVolume Fractions:")
        for phase_id, fraction in generator.metadata['volume_fractions'].items():
            if fraction > 0:
                phase_name = generator.metadata['phases'].get(str(phase_id), f"Phase {phase_id}")
                print(f"  • {phase_name}: {fraction:.1%}")
    
    if 'tpb_density_m_per_m3' in generator.metadata:
        print(f"\nTriple Phase Boundary Density:")
        print(f"  • TPB: {generator.metadata['tpb_density_m_per_m3']:.2e} m/m³")
    
    if 'interface_areas_um2' in generator.metadata:
        print(f"\nInterface Areas:")
        for interface, area in generator.metadata['interface_areas_um2'].items():
            print(f"  • {interface}: {area:.2f} µm²")
    
    # Step 2: Export data
    print("\nSTEP 2: Exporting data...")
    print("-" * 40)
    
    # Export as HDF5
    print("Exporting to HDF5 format...")
    hdf5_file = os.path.join(output_dir, "microstructure.h5")
    
    try:
        with h5py.File(hdf5_file, 'w') as f:
            # Main dataset
            dset = f.create_dataset('microstructure', 
                                   data=generator.volume,
                                   compression='gzip',
                                   compression_opts=4)
            
            # Attributes
            dset.attrs['voxel_size'] = generator.voxel_size
            dset.attrs['voxel_unit'] = 'micrometer'
            dset.attrs['dimensions'] = generator.dimensions
            dset.attrs['creation_date'] = datetime.now().isoformat()
            
            # Phase information
            phase_group = f.create_group('phases')
            for phase_id, phase_name in generator.metadata.get('phases', {}).items():
                phase_mask = (generator.volume == int(phase_id))
                pdset = phase_group.create_dataset(
                    f'phase_{phase_id}',
                    data=phase_mask,
                    compression='gzip',
                    compression_opts=4
                )
                pdset.attrs['name'] = phase_name
                if 'volume_fractions' in generator.metadata:
                    pdset.attrs['volume_fraction'] = generator.metadata['volume_fractions'].get(int(phase_id), 0)
            
            # Metadata
            meta_group = f.create_group('metadata')
            for key, value in generator.metadata.items():
                if isinstance(value, (dict, list)):
                    meta_group.attrs[key] = json.dumps(value, default=str)
                elif not isinstance(value, np.ndarray):
                    meta_group.attrs[key] = value
        
        file_size = os.path.getsize(hdf5_file) / 1e6
        print(f"✓ HDF5 export complete: {hdf5_file} ({file_size:.2f} MB)")
        
    except Exception as e:
        print(f"✗ Error exporting HDF5: {e}")
    
    # Export as TIFF stack
    print("\nExporting TIFF stack...")
    tiff_dir = os.path.join(output_dir, "tiff_stack")
    os.makedirs(tiff_dir, exist_ok=True)
    
    try:
        # Save as multi-page TIFF
        tiff_file = os.path.join(tiff_dir, "complete_stack.tif")
        tifffile.imwrite(
            tiff_file,
            generator.volume,
            metadata={'spacing': generator.voxel_size, 'unit': 'um'}
        )
        print(f"✓ TIFF stack exported: {tiff_file}")
        
        # Save a few individual slices as examples
        for z in [0, generator.dimensions[2]//2, generator.dimensions[2]-1]:
            slice_file = os.path.join(tiff_dir, f"slice_z{z:03d}.tif")
            tifffile.imwrite(
                slice_file,
                generator.volume[:, :, z],
                metadata={'spacing': generator.voxel_size, 'unit': 'um', 'slice': z}
            )
        print(f"✓ Sample slices exported to {tiff_dir}")
        
    except Exception as e:
        print(f"✗ Error exporting TIFF: {e}")
    
    # Save metadata as JSON
    print("\nSaving metadata...")
    metadata_file = os.path.join(output_dir, "metadata.json")
    
    # Prepare metadata for JSON serialization
    json_metadata = {}
    for key, value in generator.metadata.items():
        if isinstance(value, np.ndarray):
            json_metadata[key] = value.tolist()
        else:
            json_metadata[key] = value
    
    with open(metadata_file, 'w') as f:
        json.dump(json_metadata, f, indent=2, default=str)
    print(f"✓ Metadata saved: {metadata_file}")
    
    # Create README for the dataset
    readme_content = f"""# SOFC 3D Microstructural Dataset

## Dataset Information
- **Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Dimensions**: {generator.dimensions[0]} x {generator.dimensions[1]} x {generator.dimensions[2]} voxels
- **Voxel Size**: {generator.voxel_size} µm
- **Physical Size**: {generator.dimensions[0] * generator.voxel_size} x {generator.dimensions[1] * generator.voxel_size} x {generator.dimensions[2] * generator.voxel_size} µm

## Phase Identification
| ID | Material | Volume Fraction |
|----|----------|-----------------|
"""
    
    for phase_id in range(6):
        phase_name = generator.metadata['phases'].get(str(phase_id), f"Phase {phase_id}")
        vf = generator.metadata.get('volume_fractions', {}).get(phase_id, 0)
        readme_content += f"| {phase_id} | {phase_name} | {vf:.1%} |\n"
    
    readme_content += f"""
## Key Properties
- **TPB Density**: {generator.metadata.get('tpb_density_m_per_m3', 0):.2e} m/m³
- **Anode Porosity**: ~35%
- **Electrolyte Density**: ~95%

## Files
- `microstructure.h5`: Complete dataset in HDF5 format
- `tiff_stack/`: TIFF image stack
- `metadata.json`: Complete metadata and properties
- `README.md`: This file

## Usage
### Loading in Python
```python
import h5py
import numpy as np

with h5py.File('microstructure.h5', 'r') as f:
    volume = f['microstructure'][:]
    voxel_size = f['microstructure'].attrs['voxel_size']
    # Access individual phases
    phase_ni = f['phases/phase_1'][:]
```

### Visualization
The data can be visualized using:
- ParaView (load .h5 or convert to .vtk)
- ImageJ/Fiji (load TIFF stack)
- Python (matplotlib, pyvista)

## Citation
If using this dataset, please acknowledge:
"SOFC 3D Microstructural Dataset - Synthetic data for electrode modeling"
"""
    
    readme_file = os.path.join(output_dir, "README.md")
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    print(f"✓ README created: {readme_file}")
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    print(f"\n📊 Summary:")
    print(f"  • Total voxels: {np.prod(generator.dimensions):,}")
    print(f"  • Physical volume: {np.prod([d * generator.voxel_size for d in generator.dimensions]):.2f} µm³")
    print(f"  • Processing time: {total_time:.2f} seconds")
    print(f"  • Output location: {os.path.abspath(output_dir)}")
    print(f"\n✅ Dataset ready for high-fidelity modeling!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())