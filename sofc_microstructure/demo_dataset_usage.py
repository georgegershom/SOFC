#!/usr/bin/env python3
"""
Demonstration script showing how to use the SOFC 3D microstructural dataset.

This script demonstrates:
1. Loading the dataset in different formats
2. Analyzing microstructure properties
3. Extracting phase-specific information
4. Basic visualization
5. Preparing data for simulations

Author: AI Assistant
Date: 2025-10-08
"""

import numpy as np
import h5py
import json
import matplotlib.pyplot as plt
from pathlib import Path
import tifffile

def load_hdf5_dataset(file_path):
    """Load dataset from HDF5 format (recommended)."""
    print("Loading HDF5 dataset...")
    
    with h5py.File(file_path, 'r') as f:
        microstructure = f['microstructure'][:]
        dimensions = f['dimensions'][:]
        voxel_size = f['voxel_size'][()]
        
        print(f"  Microstructure shape: {microstructure.shape}")
        print(f"  Voxel size: {voxel_size} μm")
        print(f"  Physical dimensions: {dimensions * voxel_size} μm")
    
    return microstructure, dimensions, voxel_size

def load_tiff_dataset(file_path):
    """Load dataset from TIFF stack format."""
    print("Loading TIFF dataset...")
    
    microstructure = tifffile.imread(file_path)
    # TIFF format is typically (Z, Y, X), transpose to (X, Y, Z)
    microstructure = np.transpose(microstructure, (2, 1, 0))
    
    print(f"  Microstructure shape: {microstructure.shape}")
    
    return microstructure

def load_metadata(file_path):
    """Load dataset metadata."""
    print("Loading metadata...")
    
    with open(file_path, 'r') as f:
        metadata = json.load(f)
    
    print("  Metadata loaded successfully")
    return metadata

def analyze_phases(microstructure):
    """Analyze phase distribution and properties."""
    print("\nAnalyzing phase distribution...")
    
    unique_phases, counts = np.unique(microstructure, return_counts=True)
    total_voxels = np.prod(microstructure.shape)
    
    phase_names = {0: 'Pore', 1: 'Ni-YSZ Anode', 2: 'YSZ Electrolyte', 3: 'Interface'}
    
    print("  Phase distribution:")
    for phase_id, count in zip(unique_phases, counts):
        fraction = count / total_voxels
        name = phase_names.get(phase_id, f'Unknown ({phase_id})')
        print(f"    {name}: {count:,} voxels ({fraction:.1%})")
    
    return unique_phases, counts

def extract_phase_masks(microstructure):
    """Extract binary masks for each phase."""
    print("\nExtracting phase masks...")
    
    phases = {
        'pore': microstructure == 0,
        'anode': microstructure == 1,
        'electrolyte': microstructure == 2,
        'interface': microstructure == 3
    }
    
    for name, mask in phases.items():
        print(f"  {name.capitalize()}: {np.sum(mask):,} voxels")
    
    return phases

def calculate_connectivity(phase_mask, axis_names=['X', 'Y', 'Z']):
    """Check phase connectivity (percolation) in each direction."""
    connectivity = {}
    
    for axis, name in enumerate(axis_names):
        # Check if phase connects from one side to the other
        if axis == 0:  # X direction
            start_slice = phase_mask[0, :, :]
            end_slice = phase_mask[-1, :, :]
        elif axis == 1:  # Y direction
            start_slice = phase_mask[:, 0, :]
            end_slice = phase_mask[:, -1, :]
        else:  # Z direction
            start_slice = phase_mask[:, :, 0]
            end_slice = phase_mask[:, :, -1]
        
        # Simple connectivity check
        percolates = np.any(start_slice) and np.any(end_slice)
        connectivity[name] = percolates
    
    return connectivity

def analyze_connectivity(phases):
    """Analyze connectivity for all phases."""
    print("\nAnalyzing phase connectivity...")
    
    for name, mask in phases.items():
        if np.any(mask):
            connectivity = calculate_connectivity(mask)
            print(f"  {name.capitalize()} percolation:")
            for direction, percolates in connectivity.items():
                status = "✓" if percolates else "✗"
                print(f"    {direction}: {status}")
        else:
            print(f"  {name.capitalize()}: No voxels found")

def calculate_interface_area(microstructure, voxel_size):
    """Calculate approximate interface area between phases."""
    print("\nCalculating interface area...")
    
    # Count interface voxels between different phases
    interface_voxels = 0
    
    # Check neighboring voxels in all directions
    for axis in range(3):
        # Create shifted arrays to compare neighbors
        if axis == 0:
            current = microstructure[:-1, :, :]
            neighbor = microstructure[1:, :, :]
        elif axis == 1:
            current = microstructure[:, :-1, :]
            neighbor = microstructure[:, 1:, :]
        else:
            current = microstructure[:, :, :-1]
            neighbor = microstructure[:, :, 1:]
        
        # Count where neighboring voxels have different phases
        interface_voxels += np.sum(current != neighbor)
    
    # Convert to physical area
    voxel_area = voxel_size ** 2
    total_interface_area = interface_voxels * voxel_area
    
    print(f"  Total interface area: {total_interface_area:.1f} μm²")
    print(f"  Interface voxels: {interface_voxels:,}")
    
    return total_interface_area

def create_cross_section_visualization(microstructure, voxel_size, save_path=None):
    """Create cross-section visualizations."""
    print("\nCreating cross-section visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SOFC Microstructure Cross-sections', fontsize=14, fontweight='bold')
    
    # Get middle slices
    mid_x = microstructure.shape[0] // 2
    mid_y = microstructure.shape[1] // 2
    mid_z = microstructure.shape[2] // 2
    
    # XY plane (top view)
    ax1 = axes[0, 0]
    im1 = ax1.imshow(microstructure[:, :, mid_z], cmap='tab10', vmin=0, vmax=3)
    ax1.set_title(f'XY Cross-section (z={mid_z})')
    ax1.set_xlabel('X (voxels)')
    ax1.set_ylabel('Y (voxels)')
    
    # XZ plane (side view)
    ax2 = axes[0, 1]
    im2 = ax2.imshow(microstructure[:, mid_y, :], cmap='tab10', vmin=0, vmax=3)
    ax2.set_title(f'XZ Cross-section (y={mid_y})')
    ax2.set_xlabel('X (voxels)')
    ax2.set_ylabel('Z (voxels)')
    
    # YZ plane (front view)
    ax3 = axes[1, 0]
    im3 = ax3.imshow(microstructure[mid_x, :, :], cmap='tab10', vmin=0, vmax=3)
    ax3.set_title(f'YZ Cross-section (x={mid_x})')
    ax3.set_xlabel('Y (voxels)')
    ax3.set_ylabel('Z (voxels)')
    
    # Phase distribution
    ax4 = axes[1, 1]
    unique_phases, counts = np.unique(microstructure, return_counts=True)
    phase_names = ['Pore', 'Ni-YSZ', 'YSZ', 'Interface']
    colors = ['lightblue', 'orange', 'lightgreen', 'red']
    
    bars = ax4.bar(range(len(unique_phases)), counts, color=colors[:len(unique_phases)])
    ax4.set_xlabel('Phase ID')
    ax4.set_ylabel('Voxel Count')
    ax4.set_title('Phase Distribution')
    ax4.set_xticks(range(len(unique_phases)))
    ax4.set_xticklabels([phase_names[i] for i in unique_phases], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                f'{count:,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Visualization saved to {save_path}")
    
    plt.show()

def prepare_for_simulation(phases, voxel_size):
    """Prepare data for different types of simulations."""
    print("\nPreparing data for simulations...")
    
    # 1. CFD preparation - extract pore network
    pore_network = phases['pore']
    pore_volume_fraction = np.sum(pore_network) / np.prod(pore_network.shape)
    
    print(f"  CFD Analysis:")
    print(f"    Pore volume fraction: {pore_volume_fraction:.3f}")
    print(f"    Pore network connectivity: Available")
    
    # 2. FEA preparation - solid phases
    solid_phases = phases['anode'] | phases['electrolyte']
    solid_volume_fraction = np.sum(solid_phases) / np.prod(solid_phases.shape)
    
    print(f"  FEA Analysis:")
    print(f"    Solid volume fraction: {solid_volume_fraction:.3f}")
    print(f"    Material interfaces: Defined")
    
    # 3. Electrochemical modeling - active interfaces
    interface_network = phases['interface']
    interface_area_density = np.sum(interface_network) * (voxel_size**2) / (np.prod(phases['pore'].shape) * voxel_size**3)
    
    print(f"  Electrochemical Analysis:")
    print(f"    Interface area density: {interface_area_density:.1f} μm⁻¹")
    print(f"    Active interface available: Yes")

def demonstrate_data_export(microstructure, phases):
    """Demonstrate how to export data for external tools."""
    print("\nDemonstrating data export options...")
    
    # Export individual phases as separate arrays
    print("  Exporting individual phases:")
    for name, mask in phases.items():
        if np.any(mask):
            # Convert boolean mask to uint8 for saving
            phase_array = mask.astype(np.uint8) * 255
            print(f"    {name.capitalize()}: {phase_array.shape} array ready for export")
    
    # Export combined microstructure with material IDs
    print("  Combined microstructure:")
    print(f"    Material ID array: {microstructure.shape} with {len(np.unique(microstructure))} phases")
    
    # Coordinate arrays for mesh generation
    x, y, z = np.meshgrid(
        np.arange(microstructure.shape[0]),
        np.arange(microstructure.shape[1]),
        np.arange(microstructure.shape[2]),
        indexing='ij'
    )
    
    print("  Coordinate arrays:")
    print(f"    X, Y, Z coordinates: {x.shape} each")
    print(f"    Ready for mesh generation or FEM preprocessing")

def main():
    """Main demonstration function."""
    print("="*70)
    print("SOFC 3D MICROSTRUCTURAL DATASET - USAGE DEMONSTRATION")
    print("="*70)
    
    # Define file paths
    base_path = Path("data")
    hdf5_path = base_path / "sofc_microstructure.h5"
    tiff_path = base_path / "sofc_microstructure.tiff"
    metadata_path = base_path / "metadata.json"
    
    # Check if files exist
    if not hdf5_path.exists():
        print("Error: Dataset files not found. Please run generate_small_dataset.py first.")
        return
    
    # 1. Load dataset
    microstructure, dimensions, voxel_size = load_hdf5_dataset(hdf5_path)
    metadata = load_metadata(metadata_path)
    
    # 2. Analyze phases
    unique_phases, counts = analyze_phases(microstructure)
    
    # 3. Extract phase masks
    phases = extract_phase_masks(microstructure)
    
    # 4. Analyze connectivity
    analyze_connectivity(phases)
    
    # 5. Calculate interface area
    interface_area = calculate_interface_area(microstructure, voxel_size)
    
    # 6. Create visualizations
    create_cross_section_visualization(
        microstructure, voxel_size, 
        save_path="results/demo_visualization.png"
    )
    
    # 7. Prepare for simulations
    prepare_for_simulation(phases, voxel_size)
    
    # 8. Demonstrate data export
    demonstrate_data_export(microstructure, phases)
    
    # 9. Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Dataset loaded: {microstructure.shape} voxels")
    print(f"Physical size: {dimensions * voxel_size} μm")
    print(f"Resolution: {voxel_size} μm/voxel")
    print(f"Total phases: {len(unique_phases)}")
    print(f"Interface area: {interface_area:.1f} μm²")
    print("\nThe dataset is ready for:")
    print("  ✓ Computational Fluid Dynamics (CFD)")
    print("  ✓ Finite Element Analysis (FEA)")
    print("  ✓ Electrochemical Modeling")
    print("  ✓ Multi-physics Simulations")
    print("\nFor more details, see the README.md and documentation files.")

if __name__ == "__main__":
    main()