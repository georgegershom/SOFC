#!/usr/bin/env python3
"""
Generate a smaller SOFC dataset for demonstration.
"""

import sys
import os
import time
import numpy as np

# Add source directory to path
sys.path.append('sofc_microstructure/src')

from microstructure_generator import SOFCMicrostructureGenerator
from interface_analyzer import InterfaceGeometryAnalyzer
from mesh_generator import SOFCMeshGenerator

def main():
    print("="*60)
    print("SOFC 3D MICROSTRUCTURAL DATASET GENERATION (SMALL)")
    print("="*60)
    
    # Create smaller dataset for demonstration
    generator = SOFCMicrostructureGenerator(
        dimensions=(200, 200, 100),  # Smaller size
        voxel_size=0.1,  # 100 nm voxels
        random_seed=42
    )
    
    # Generate microstructure
    print("Generating microstructure...")
    microstructure = generator.generate_realistic_microstructure(
        anode_thickness=10.0,
        electrolyte_thickness=6.0,
        interface_roughness=0.5
    )
    
    # Save dataset
    print("Saving dataset...")
    generator.save_dataset("sofc_microstructure/data")
    
    # Interface analysis
    print("Analyzing interface...")
    interface_analyzer = InterfaceGeometryAnalyzer(
        microstructure=microstructure,
        voxel_size=generator.voxel_size,
        phases=generator.phases
    )
    
    interface_surface = interface_analyzer.extract_interface_surface()
    morphology_metrics = interface_analyzer.analyze_interface_morphology()
    
    # Generate basic meshes
    print("Generating meshes...")
    mesh_generator = SOFCMeshGenerator(
        microstructure=microstructure,
        voxel_size=generator.voxel_size,
        phases=generator.phases
    )
    
    surface_meshes = mesh_generator.generate_surface_meshes(
        smooth_iterations=1,
        decimation_factor=0.3
    )
    
    # Export meshes
    mesh_generator.export_meshes("sofc_microstructure/data/meshes")
    
    # Create visualizations
    print("Creating visualizations...")
    try:
        generator.visualize_microstructure("sofc_microstructure/results/microstructure_analysis.png")
    except Exception as e:
        print(f"Warning: Could not create microstructure visualization: {e}")
    
    try:
        interface_analyzer.visualize_interface_analysis("sofc_microstructure/results/interface_analysis.png")
    except Exception as e:
        print(f"Warning: Could not create interface visualization: {e}")
    
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETED!")
    print("="*60)
    print(f"Microstructure: {microstructure.shape} voxels")
    print(f"Physical size: {generator.physical_size} μm")
    print(f"Porosity: {generator.metadata['porosity']:.3f}")
    print(f"Interface area: {morphology_metrics.get('interface_area_um2', 0):.1f} μm²")
    print(f"Surface meshes: {len(surface_meshes)} generated")
    print("\nFiles generated:")
    print("- sofc_microstructure/data/sofc_microstructure.h5")
    print("- sofc_microstructure/data/sofc_microstructure.tiff")
    print("- sofc_microstructure/data/slices/")
    print("- sofc_microstructure/data/meshes/")
    print("- sofc_microstructure/results/")

if __name__ == "__main__":
    main()