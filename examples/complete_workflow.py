"""
Complete Workflow Example for SOFC Microstructure Generation and Analysis

This script demonstrates the complete workflow from microstructure generation
to mesh creation and visualization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from microstructure_generator import SOFCMicrostructureGenerator
from microstructure_analysis import MicrostructureAnalyzer
from mesh_generator import MeshGenerator
from visualization_dashboard import MicrostructureVisualizer


def main():
    """
    Complete workflow demonstration.
    """
    print("="*60)
    print("SOFC MICROSTRUCTURE GENERATION - COMPLETE WORKFLOW")
    print("="*60)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate 3D Microstructure
    print("\n1. GENERATING 3D MICROSTRUCTURE")
    print("-" * 40)
    
    # Define parameters for realistic SOFC electrode
    resolution = (256, 256, 128)  # Width × Height × Depth
    voxel_size = 0.1  # 100 nm voxel size
    porosity = 0.3  # 30% porosity
    ni_ysz_ratio = 0.6  # 60% Ni in anode
    ysz_thickness = 10.0  # 10 μm electrolyte thickness
    
    print(f"Resolution: {resolution}")
    print(f"Voxel size: {voxel_size} μm")
    print(f"Target porosity: {porosity}")
    print(f"Ni/YSZ ratio: {ni_ysz_ratio}")
    print(f"Electrolyte thickness: {ysz_thickness} μm")
    
    # Create generator
    generator = SOFCMicrostructureGenerator(
        resolution=resolution,
        voxel_size=voxel_size,
        porosity=porosity,
        ni_ysz_ratio=ni_ysz_ratio,
        ysz_thickness=ysz_thickness
    )
    
    # Generate microstructure
    microstructure = generator.generate_microstructure()
    
    # Save in multiple formats
    print("\nSaving microstructure data...")
    generator.save_hdf5(str(output_dir / "sofc_microstructure.h5"))
    generator.save_tiff_stack(str(output_dir / "sofc_microstructure"))
    generator.save_vtk(str(output_dir / "sofc_microstructure.vtk"))
    
    # Print phase distribution
    print("\nPhase Distribution:")
    for phase, props in generator.phase_properties['phases'].items():
        print(f"  {phase}: {props['volume_fraction']:.3f} ({props['volume_um3']:.2f} μm³)")
    
    # Step 2: Comprehensive Analysis
    print("\n\n2. COMPREHENSIVE ANALYSIS")
    print("-" * 40)
    
    # Create analyzer
    analyzer = MicrostructureAnalyzer(microstructure, voxel_size)
    
    # Run all analyses
    print("Analyzing phase connectivity...")
    connectivity = analyzer.analyze_phase_connectivity()
    
    print("Analyzing pore network...")
    pore_network = analyzer.analyze_pore_network()
    
    print("Analyzing interface properties...")
    interfaces = analyzer.analyze_interface_properties()
    
    print("Estimating mechanical properties...")
    mechanical = analyzer.estimate_mechanical_properties()
    
    # Generate comprehensive report
    print("Generating analysis report...")
    report = analyzer.create_comprehensive_report()
    report.to_csv(str(output_dir / "microstructure_analysis.csv"))
    
    # Print key results
    print("\nKey Analysis Results:")
    print(f"  Pore tortuosity: {pore_network['tortuosity']:.3f}")
    print(f"  Pore connectivity: {pore_network['connectivity']:.1f}%")
    print(f"  Mean pore size: {pore_network['mean_pore_size']:.2f} μm³")
    
    if 'Anode_Electrolyte' in interfaces:
        print(f"  Interface area: {interfaces['Anode_Electrolyte']['area_um2']:.2f} μm²")
        print(f"  Interface roughness: {interfaces['Anode_Electrolyte']['roughness_um']:.3f} μm")
    
    print(f"  Effective Young's modulus: {mechanical['effective_youngs_modulus_Pa']/1e9:.1f} GPa")
    print(f"  Effective density: {mechanical['effective_density_kg_m3']:.0f} kg/m³")
    
    # Step 3: Mesh Generation
    print("\n\n3. MESH GENERATION")
    print("-" * 40)
    
    # Create mesh generator
    mesh_gen = MeshGenerator(microstructure, voxel_size)
    
    # Create meshes directory
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(exist_ok=True)
    
    # Generate structured hexahedral mesh
    print("Generating structured hexahedral mesh...")
    hex_mesh = mesh_gen.generate_structured_hex_mesh()
    mesh_gen.export_mesh(hex_mesh, str(meshes_dir / "structured_hex_mesh.vtk"), 'vtk')
    mesh_gen.export_mesh(hex_mesh, str(meshes_dir / "structured_hex_mesh.xdmf"), 'xdmf')
    
    # Generate unstructured tetrahedral mesh
    print("Generating unstructured tetrahedral mesh...")
    tet_mesh = mesh_gen.generate_unstructured_tet_mesh()
    if tet_mesh.n_cells > 0:
        mesh_gen.export_mesh(tet_mesh, str(meshes_dir / "unstructured_tet_mesh.vtk"), 'vtk')
        mesh_gen.export_mesh(tet_mesh, str(meshes_dir / "unstructured_tet_mesh.stl"), 'stl')
    
    # Generate surface mesh
    print("Generating surface mesh...")
    surface_mesh = mesh_gen.generate_surface_mesh()
    if surface_mesh.n_cells > 0:
        mesh_gen.export_mesh(surface_mesh, str(meshes_dir / "surface_mesh.stl"), 'stl')
        mesh_gen.export_mesh(surface_mesh, str(meshes_dir / "surface_mesh.obj"), 'obj')
    
    # Generate interface mesh
    print("Generating interface mesh...")
    interface_mesh = mesh_gen.generate_interface_mesh()
    if interface_mesh.n_cells > 0:
        mesh_gen.export_mesh(interface_mesh, str(meshes_dir / "interface_mesh.stl"), 'stl')
    
    # Calculate mesh statistics
    print("Calculating mesh statistics...")
    hex_stats = mesh_gen.create_mesh_statistics(hex_mesh)
    tet_stats = mesh_gen.create_mesh_statistics(tet_mesh) if tet_mesh.n_cells > 0 else {}
    
    print("\nMesh Statistics:")
    print(f"  Hexahedral mesh: {hex_stats['n_points']:,} points, {hex_stats['n_cells']:,} cells")
    if tet_stats:
        print(f"  Tetrahedral mesh: {tet_stats['n_points']:,} points, {tet_stats['n_cells']:,} cells")
    
    # Step 4: Visualization
    print("\n\n4. VISUALIZATION")
    print("-" * 40)
    
    # Create visualizer
    visualizer = MicrostructureVisualizer(microstructure, voxel_size)
    
    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Save static plots
    print("Creating static visualizations...")
    visualizer.save_static_plots(str(plots_dir))
    
    # Create 3D visualization with PyVista
    print("Creating 3D PyVista visualization...")
    try:
        plotter = generator.visualize_3d(str(plots_dir / "microstructure_3d.png"))
        print("3D visualization saved")
    except Exception as e:
        print(f"3D visualization failed: {e}")
    
    # Step 5: Summary and Next Steps
    print("\n\n5. WORKFLOW COMPLETE")
    print("-" * 40)
    
    print("\nGenerated Files:")
    print("  Data Files:")
    print(f"    - {output_dir}/sofc_microstructure.h5 (HDF5 format)")
    print(f"    - {output_dir}/sofc_microstructure_*.tif (TIFF stack)")
    print(f"    - {output_dir}/sofc_microstructure.vtk (VTK format)")
    
    print("  Analysis Results:")
    print(f"    - {output_dir}/microstructure_analysis.csv (Analysis report)")
    
    print("  Meshes:")
    print(f"    - {meshes_dir}/structured_hex_mesh.vtk/.xdmf")
    print(f"    - {meshes_dir}/unstructured_tet_mesh.vtk/.stl")
    print(f"    - {meshes_dir}/surface_mesh.stl/.obj")
    print(f"    - {meshes_dir}/interface_mesh.stl")
    
    print("  Visualizations:")
    print(f"    - {plots_dir}/3d_visualization.html/.png")
    print(f"    - {plots_dir}/cross_sections.html/.png")
    print(f"    - {plots_dir}/phase_distribution.html/.png")
    print(f"    - {plots_dir}/interface_analysis.html/.png")
    print(f"    - {plots_dir}/microstructure_3d.png")
    
    print("\nNext Steps:")
    print("  1. Open HTML files in browser for interactive visualization")
    print("  2. Import VTK/XDMF files into FEA software (ANSYS, Abaqus, etc.)")
    print("  3. Use STL files for 3D printing or CAD software")
    print("  4. Run interactive dashboard: python visualization_dashboard.py")
    print("  5. Customize parameters for specific applications")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()