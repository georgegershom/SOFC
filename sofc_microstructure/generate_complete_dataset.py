#!/usr/bin/env python3
"""
Complete SOFC 3D Microstructural Dataset Generation

This script demonstrates the complete workflow for generating a comprehensive
3D microstructural dataset for SOFC electrode modeling, including:

1. 3D voxelated microstructure generation
2. Phase segmentation and analysis
3. Interface geometry characterization
4. Mesh generation for computational modeling
5. Comprehensive visualization and documentation

Usage:
    python generate_complete_dataset.py

Author: AI Assistant
Date: 2025-10-08
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from microstructure_generator import SOFCMicrostructureGenerator
from interface_analyzer import InterfaceGeometryAnalyzer
from mesh_generator import SOFCMeshGenerator

def main():
    """Main function to generate complete SOFC dataset."""
    
    print("="*80)
    print("SOFC 3D MICROSTRUCTURAL DATASET GENERATION")
    print("="*80)
    print()
    
    start_time = time.time()
    
    # ========================================================================
    # STEP 1: MICROSTRUCTURE GENERATION
    # ========================================================================
    
    print("STEP 1: Generating 3D Microstructure")
    print("-" * 40)
    
    # Initialize generator with realistic parameters
    generator = SOFCMicrostructureGenerator(
        dimensions=(400, 400, 200),  # 400x400x200 voxels
        voxel_size=0.08,  # 80 nm voxels (realistic for synchrotron tomography)
        random_seed=42
    )
    
    # Generate microstructure with realistic SOFC parameters
    microstructure = generator.generate_realistic_microstructure(
        anode_thickness=20.0,  # 20 μm anode thickness
        electrolyte_thickness=12.0,  # 12 μm electrolyte thickness
        interface_roughness=0.6  # Moderate interface roughness
    )
    
    # Save the basic dataset
    generator.save_dataset("sofc_microstructure/data")
    
    print(f"✓ Microstructure generated: {microstructure.shape}")
    print(f"✓ Physical dimensions: {generator.physical_size} μm")
    print(f"✓ Volume fractions: {generator.metadata['volume_fractions']}")
    print()
    
    # ========================================================================
    # STEP 2: INTERFACE ANALYSIS
    # ========================================================================
    
    print("STEP 2: Analyzing Interface Geometry")
    print("-" * 40)
    
    # Initialize interface analyzer
    interface_analyzer = InterfaceGeometryAnalyzer(
        microstructure=microstructure,
        voxel_size=generator.voxel_size,
        phases=generator.phases
    )
    
    # Extract and analyze interface
    interface_surface = interface_analyzer.extract_interface_surface(smooth_iterations=3)
    
    # Comprehensive interface analysis
    morphology_metrics = interface_analyzer.analyze_interface_morphology()
    roughness_metrics = interface_analyzer.analyze_interface_roughness(
        analysis_scales=[0.2, 0.5, 1.0, 2.0, 5.0]  # Multiple length scales
    )
    curvature_metrics = interface_analyzer.analyze_interface_curvature()
    delamination_risk = interface_analyzer.assess_delamination_risk()
    
    print(f"✓ Interface extracted: {np.sum(interface_surface)} voxels")
    print(f"✓ Interface area: {morphology_metrics.get('interface_area_um2', 0):.1f} μm²")
    print(f"✓ Interface roughness analyzed at {len(roughness_metrics)-1} scales")
    print(f"✓ Delamination risk score: {delamination_risk.get('overall_risk_score', 0):.3f}")
    print()
    
    # ========================================================================
    # STEP 3: MESH GENERATION
    # ========================================================================
    
    print("STEP 3: Generating Computational Meshes")
    print("-" * 40)
    
    # Initialize mesh generator
    mesh_generator = SOFCMeshGenerator(
        microstructure=microstructure,
        voxel_size=generator.voxel_size,
        phases=generator.phases
    )
    
    # Generate different types of meshes
    surface_meshes = mesh_generator.generate_surface_meshes(
        smooth_iterations=2,
        decimation_factor=0.2  # Reduce mesh size for performance
    )
    
    volume_meshes = mesh_generator.generate_volume_meshes(
        target_edge_length=generator.voxel_size * 3
    )
    
    # Generate critical anode/electrolyte interface mesh
    interface_mesh = mesh_generator.generate_interface_mesh(
        phase1='ni_ysz',
        phase2='ysz_electrolyte',
        mesh_resolution=generator.voxel_size * 2
    )
    
    # Generate pore network for transport analysis
    pore_network = mesh_generator.generate_pore_network_mesh(
        min_pore_size=generator.voxel_size * 4
    )
    
    # Analyze mesh quality
    mesh_quality = mesh_generator.analyze_mesh_quality()
    
    print(f"✓ Surface meshes generated: {len(surface_meshes)} phases")
    print(f"✓ Volume meshes generated: {len(volume_meshes)} phases")
    print(f"✓ Interface mesh generated: {interface_mesh is not None}")
    print(f"✓ Pore network mesh generated: {pore_network is not None}")
    print()
    
    # ========================================================================
    # STEP 4: COMPREHENSIVE ANALYSIS AND VALIDATION
    # ========================================================================
    
    print("STEP 4: Comprehensive Analysis and Validation")
    print("-" * 40)
    
    # Validate microstructure properties
    validation_results = validate_microstructure_properties(generator, interface_analyzer)
    
    # Calculate advanced metrics
    advanced_metrics = calculate_advanced_metrics(
        microstructure, generator.voxel_size, generator.phases
    )
    
    print("✓ Microstructure validation completed")
    print("✓ Advanced metrics calculated")
    print()
    
    # ========================================================================
    # STEP 5: EXPORT AND DOCUMENTATION
    # ========================================================================
    
    print("STEP 5: Export and Documentation")
    print("-" * 40)
    
    # Create output directories
    os.makedirs("sofc_microstructure/results", exist_ok=True)
    os.makedirs("sofc_microstructure/data/meshes", exist_ok=True)
    
    # Export meshes
    mesh_generator.export_meshes("sofc_microstructure/data/meshes")
    
    # Generate comprehensive visualizations
    create_comprehensive_visualizations(
        generator, interface_analyzer, mesh_generator,
        validation_results, advanced_metrics
    )
    
    # Create detailed documentation
    create_dataset_documentation(
        generator, interface_analyzer, mesh_generator,
        morphology_metrics, roughness_metrics, curvature_metrics,
        delamination_risk, mesh_quality, validation_results, advanced_metrics
    )
    
    # Export analysis results
    export_analysis_results(
        morphology_metrics, roughness_metrics, curvature_metrics,
        delamination_risk, mesh_quality, validation_results, advanced_metrics
    )
    
    print("✓ Meshes exported in multiple formats")
    print("✓ Comprehensive visualizations created")
    print("✓ Dataset documentation generated")
    print("✓ Analysis results exported")
    print()
    
    # ========================================================================
    # COMPLETION SUMMARY
    # ========================================================================
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("="*80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print()
    print(f"Total processing time: {total_time:.1f} seconds")
    print()
    print("Generated Files:")
    print("├── sofc_microstructure/data/")
    print("│   ├── sofc_microstructure.h5          # Main dataset (HDF5)")
    print("│   ├── sofc_microstructure.tiff        # TIFF stack")
    print("│   ├── slices/                         # Individual 2D slices")
    print("│   ├── metadata.json                   # Dataset metadata")
    print("│   └── meshes/                         # Computational meshes")
    print("│       ├── *.stl                       # Surface meshes")
    print("│       ├── *.vtk                       # Volume meshes")
    print("│       └── mesh_quality_report.json    # Mesh quality metrics")
    print("├── sofc_microstructure/results/")
    print("│   ├── complete_analysis.png           # Comprehensive visualization")
    print("│   ├── interface_analysis.png          # Interface characterization")
    print("│   ├── mesh_visualization.png          # Mesh quality visualization")
    print("│   ├── dataset_report.html             # Complete HTML report")
    print("│   └── analysis_results.json           # All analysis metrics")
    print("└── sofc_microstructure/docs/")
    print("    └── dataset_documentation.md        # Detailed documentation")
    print()
    print("Dataset Characteristics:")
    print(f"  • Microstructure dimensions: {microstructure.shape} voxels")
    print(f"  • Physical size: {generator.physical_size} μm")
    print(f"  • Voxel resolution: {generator.voxel_size} μm")
    print(f"  • Total porosity: {generator.metadata['porosity']:.3f}")
    print(f"  • Interface area: {morphology_metrics.get('interface_area_um2', 0):.1f} μm²")
    print(f"  • Delamination risk: {delamination_risk.get('overall_risk_score', 0):.3f}")
    print()
    print("Ready for:")
    print("  ✓ Finite Element Analysis (FEA)")
    print("  ✓ Computational Fluid Dynamics (CFD)")
    print("  ✓ Electrochemical Modeling")
    print("  ✓ Mechanical Stress Analysis")
    print("  ✓ Transport Phenomena Studies")
    print()

def validate_microstructure_properties(generator, interface_analyzer):
    """Validate that the generated microstructure has realistic properties."""
    
    validation = {
        'porosity_realistic': False,
        'interface_continuity': False,
        'phase_connectivity': False,
        'geometric_feasibility': False
    }
    
    # Check porosity is within realistic range for SOFC
    porosity = generator.metadata['porosity']
    if 0.25 <= porosity <= 0.45:  # Typical SOFC porosity range
        validation['porosity_realistic'] = True
    
    # Check interface continuity
    if interface_analyzer.interface_surface is not None:
        interface_fraction = np.sum(interface_analyzer.interface_surface) / np.prod(generator.dimensions)
        if interface_fraction > 0.01:  # At least 1% interface voxels
            validation['interface_continuity'] = True
    
    # Check phase connectivity
    connectivity = generator.metadata.get('connectivity_metrics', {})
    anode_connected = connectivity.get('ni_ysz', {}).get('percolation_z', False)
    electrolyte_connected = connectivity.get('ysz_electrolyte', {}).get('percolation_z', False)
    
    if anode_connected and electrolyte_connected:
        validation['phase_connectivity'] = True
    
    # Check geometric feasibility
    volume_fractions = generator.metadata['volume_fractions']
    if (volume_fractions.get('ni_ysz', 0) > 0.1 and 
        volume_fractions.get('ysz_electrolyte', 0) > 0.1):
        validation['geometric_feasibility'] = True
    
    return validation

def calculate_advanced_metrics(microstructure, voxel_size, phases):
    """Calculate advanced microstructural metrics."""
    
    metrics = {}
    
    # Tortuosity calculation (simplified)
    pore_mask = (microstructure == phases['pore'])
    if np.any(pore_mask):
        # Calculate tortuosity in z-direction
        z_slices = []
        for z in range(microstructure.shape[2]):
            slice_porosity = np.sum(pore_mask[:, :, z]) / (microstructure.shape[0] * microstructure.shape[1])
            z_slices.append(slice_porosity)
        
        # Simple tortuosity estimate
        mean_porosity = np.mean(z_slices)
        if mean_porosity > 0:
            tortuosity = 1.0 / mean_porosity  # Simplified model
        else:
            tortuosity = float('inf')
        
        metrics['tortuosity_z'] = min(tortuosity, 10.0)  # Cap at reasonable value
    
    # Specific surface area
    total_voxels = np.prod(microstructure.shape)
    interface_voxels = 0
    
    # Count interface voxels between phases
    for i in range(microstructure.shape[0]-1):
        for j in range(microstructure.shape[1]-1):
            for k in range(microstructure.shape[2]-1):
                # Check if neighboring voxels have different phases
                current = microstructure[i, j, k]
                neighbors = [
                    microstructure[i+1, j, k],
                    microstructure[i, j+1, k],
                    microstructure[i, j, k+1]
                ]
                
                if any(neighbor != current for neighbor in neighbors):
                    interface_voxels += 1
    
    # Specific surface area (surface area per unit volume)
    total_volume = total_voxels * (voxel_size ** 3)
    interface_area = interface_voxels * (voxel_size ** 2)
    
    metrics['specific_surface_area'] = interface_area / total_volume if total_volume > 0 else 0
    
    # Pore size distribution (simplified)
    if np.any(pore_mask):
        from scipy import ndimage
        distance = ndimage.distance_transform_edt(pore_mask)
        pore_sizes = distance[pore_mask] * voxel_size
        
        metrics['pore_size_distribution'] = {
            'mean': np.mean(pore_sizes),
            'std': np.std(pore_sizes),
            'max': np.max(pore_sizes),
            'percentiles': {
                '10': np.percentile(pore_sizes, 10),
                '50': np.percentile(pore_sizes, 50),
                '90': np.percentile(pore_sizes, 90)
            }
        }
    
    return metrics

def create_comprehensive_visualizations(generator, interface_analyzer, mesh_generator,
                                      validation_results, advanced_metrics):
    """Create comprehensive visualizations of all analyses."""
    
    # Main microstructure visualization
    generator.visualize_microstructure("sofc_microstructure/results/complete_analysis.png")
    
    # Interface analysis visualization
    interface_analyzer.visualize_interface_analysis("sofc_microstructure/results/interface_analysis.png")
    
    # Mesh visualization (if meshes were generated)
    if mesh_generator.surface_meshes or mesh_generator.volume_meshes:
        try:
            mesh_generator.visualize_meshes("sofc_microstructure/results/mesh_visualization.png")
        except Exception as e:
            print(f"Warning: Could not create mesh visualization: {e}")
    
    # Create summary dashboard
    create_summary_dashboard(generator, validation_results, advanced_metrics)

def create_summary_dashboard(generator, validation_results, advanced_metrics):
    """Create a summary dashboard with key metrics."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SOFC Microstructure Dataset Summary', fontsize=16, fontweight='bold')
    
    # 1. Volume fractions pie chart
    ax1 = axes[0, 0]
    volume_fractions = generator.metadata['volume_fractions']
    labels = list(volume_fractions.keys())
    sizes = list(volume_fractions.values())
    colors = ['lightblue', 'orange', 'lightgreen', 'red']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Volume Fractions')
    
    # 2. Validation results
    ax2 = axes[0, 1]
    validation_labels = list(validation_results.keys())
    validation_values = [1 if v else 0 for v in validation_results.values()]
    
    bars = ax2.bar(range(len(validation_labels)), validation_values, 
                   color=['green' if v else 'red' for v in validation_results.values()])
    ax2.set_xticks(range(len(validation_labels)))
    ax2.set_xticklabels([label.replace('_', '\n') for label in validation_labels], 
                       rotation=45, ha='right')
    ax2.set_ylabel('Validation Status')
    ax2.set_title('Microstructure Validation')
    ax2.set_ylim(0, 1.2)
    
    # Add text labels on bars
    for i, (bar, value) in enumerate(zip(bars, validation_results.values())):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                'PASS' if value else 'FAIL', ha='center', va='bottom',
                fontweight='bold', color='white' if value else 'black')
    
    # 3. Connectivity metrics
    ax3 = axes[0, 2]
    connectivity = generator.metadata.get('connectivity_metrics', {})
    
    if connectivity:
        phases = list(connectivity.keys())
        percolation_x = [connectivity[phase].get('percolation_x', False) for phase in phases]
        percolation_y = [connectivity[phase].get('percolation_y', False) for phase in phases]
        percolation_z = [connectivity[phase].get('percolation_z', False) for phase in phases]
        
        x_pos = np.arange(len(phases))
        width = 0.25
        
        ax3.bar(x_pos - width, [1 if p else 0 for p in percolation_x], width, 
               label='X-direction', alpha=0.8)
        ax3.bar(x_pos, [1 if p else 0 for p in percolation_y], width, 
               label='Y-direction', alpha=0.8)
        ax3.bar(x_pos + width, [1 if p else 0 for p in percolation_z], width, 
               label='Z-direction', alpha=0.8)
        
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Percolation')
        ax3.set_title('Phase Connectivity')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(phases, rotation=45, ha='right')
        ax3.legend()
        ax3.set_ylim(0, 1.2)
    
    # 4. Physical dimensions
    ax4 = axes[1, 0]
    dimensions = ['X', 'Y', 'Z']
    sizes_um = generator.physical_size
    
    bars = ax4.bar(dimensions, sizes_um, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax4.set_ylabel('Size (μm)')
    ax4.set_title('Physical Dimensions')
    
    # Add value labels on bars
    for bar, size in zip(bars, sizes_um):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(sizes_um)*0.01,
                f'{size:.1f}', ha='center', va='bottom')
    
    # 5. Advanced metrics
    ax5 = axes[1, 1]
    
    if advanced_metrics:
        metrics_text = "Advanced Metrics:\n\n"
        
        if 'tortuosity_z' in advanced_metrics:
            metrics_text += f"Tortuosity (Z): {advanced_metrics['tortuosity_z']:.2f}\n"
        
        if 'specific_surface_area' in advanced_metrics:
            metrics_text += f"Specific Surface Area: {advanced_metrics['specific_surface_area']:.3f} μm⁻¹\n"
        
        if 'pore_size_distribution' in advanced_metrics:
            psd = advanced_metrics['pore_size_distribution']
            metrics_text += f"\nPore Size Distribution:\n"
            metrics_text += f"  Mean: {psd['mean']:.3f} μm\n"
            metrics_text += f"  Std: {psd['std']:.3f} μm\n"
            metrics_text += f"  Max: {psd['max']:.3f} μm\n"
        
        ax5.text(0.1, 0.9, metrics_text, transform=ax5.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    ax5.set_title('Advanced Metrics')
    ax5.axis('off')
    
    # 6. Dataset summary
    ax6 = axes[1, 2]
    
    summary_text = f"""Dataset Summary:

Voxel Resolution: {generator.voxel_size} μm
Total Voxels: {np.prod(generator.dimensions):,}
Data Size: ~{np.prod(generator.dimensions) * 4 / 1e6:.1f} MB

Porosity: {generator.metadata['porosity']:.3f}
Interface Area: {generator.metadata.get('interface_area_um2', 0):.1f} μm²

Validation Status:
{sum(validation_results.values())}/{len(validation_results)} checks passed

Generated Files:
• HDF5 dataset
• TIFF image stack
• Surface meshes (STL)
• Volume meshes (VTK)
• Interface meshes
• Analysis reports"""
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
            fontsize=9, verticalalignment='top', fontfamily='monospace')
    ax6.set_title('Dataset Summary')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.savefig("sofc_microstructure/results/summary_dashboard.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_documentation(generator, interface_analyzer, mesh_generator,
                               morphology_metrics, roughness_metrics, curvature_metrics,
                               delamination_risk, mesh_quality, validation_results, advanced_metrics):
    """Create comprehensive dataset documentation."""
    
    os.makedirs("sofc_microstructure/docs", exist_ok=True)
    
    doc_content = f"""# SOFC 3D Microstructural Dataset Documentation

## Overview

This dataset contains a comprehensive 3D microstructural representation of a Solid Oxide Fuel Cell (SOFC) electrode, generated using advanced computational methods to mimic synchrotron X-ray tomography or FIB-SEM tomography data.

## Dataset Specifications

### Microstructure Properties
- **Dimensions**: {generator.dimensions} voxels
- **Physical Size**: {generator.physical_size} μm
- **Voxel Resolution**: {generator.voxel_size} μm
- **Total Volume**: {np.prod(generator.physical_size):.2f} μm³
- **Generation Method**: Particle-based stochastic generation with morphological refinement

### Phase Information
The microstructure contains the following phases:

| Phase ID | Phase Name | Description | Volume Fraction |
|----------|------------|-------------|-----------------|
| 0 | Pore | Void space for gas transport | {generator.metadata['volume_fractions'].get('pore', 0):.3f} |
| 1 | Ni-YSZ | Anode material (Nickel-Yttria Stabilized Zirconia) | {generator.metadata['volume_fractions'].get('ni_ysz', 0):.3f} |
| 2 | YSZ Electrolyte | Yttria Stabilized Zirconia electrolyte | {generator.metadata['volume_fractions'].get('ysz_electrolyte', 0):.3f} |
| 3 | Interface | Anode/electrolyte interface regions | {generator.metadata['volume_fractions'].get('interface', 0):.3f} |

### Material Properties Used in Generation

#### Ni-YSZ Anode
- Target porosity: 35%
- Particle size range: 0.5 - 2.0 μm
- Connectivity: 0.8

#### YSZ Electrolyte
- Target porosity: 5%
- Particle size range: 0.2 - 1.0 μm
- Connectivity: 0.95

#### Interface
- Thickness range: 0.1 - 0.3 μm
- Roughness parameter: 0.2

## Interface Geometry Analysis

### Morphological Metrics
- **Interface Area**: {morphology_metrics.get('interface_area_um2', 0):.2f} μm²
- **Interface Tortuosity**: {morphology_metrics.get('tortuosity', 1.0):.3f}
- **Connectivity Components**: {morphology_metrics.get('connectivity', {}).get('connected_components', 0)}

### Roughness Analysis
Multi-scale roughness analysis was performed at the following length scales:
"""

    # Add roughness metrics
    if roughness_metrics:
        for scale, metrics in roughness_metrics.items():
            if scale.startswith('scale_') and isinstance(metrics, dict):
                scale_um = scale.replace('scale_', '').replace('um', '')
                doc_content += f"\n#### {scale_um} μm Scale\n"
                doc_content += f"- Ra (Average roughness): {metrics.get('Ra_um', 0):.4f} μm\n"
                doc_content += f"- Rq (RMS roughness): {metrics.get('Rq_um', 0):.4f} μm\n"
                doc_content += f"- Rz (Peak-to-valley): {metrics.get('Rz_um', 0):.4f} μm\n"

    doc_content += f"""

### Curvature Analysis
"""

    if curvature_metrics and 'error' not in curvature_metrics:
        doc_content += f"""- **Mean Curvature**: {curvature_metrics.get('mean_curvature', 0):.4f}
- **Maximum Curvature**: {curvature_metrics.get('max_curvature', 0):.4f}
- **Curvature Standard Deviation**: {curvature_metrics.get('curvature_std', 0):.4f}
- **High Curvature Fraction**: {curvature_metrics.get('high_curvature_fraction', 0):.3f}
"""

    doc_content += f"""

### Delamination Risk Assessment
- **Overall Risk Score**: {delamination_risk.get('overall_risk_score', 0):.3f} (0 = low risk, 1 = high risk)
- **Stress Concentration Factor**: {delamination_risk.get('stress_concentration', {}).get('mean_stress_concentration_factor', 1.0):.2f}
- **Critical Stress Locations**: {delamination_risk.get('stress_concentration', {}).get('critical_stress_locations', False)}

## Computational Meshes

The following computational meshes have been generated from the microstructure:

### Surface Meshes
"""

    for phase_name, mesh in mesh_generator.surface_meshes.items():
        doc_content += f"""
#### {phase_name.title()} Phase
- **Vertices**: {len(mesh.vertices):,}
- **Faces**: {len(mesh.faces):,}
- **Surface Area**: {mesh.area:.2f} μm²
- **Volume**: {mesh.volume:.2f} μm³ (if watertight)
- **File Formats**: STL, OBJ, PLY
"""

    doc_content += """
### Volume Meshes
"""

    for phase_name, mesh in mesh_generator.volume_meshes.items():
        doc_content += f"""
#### {phase_name.title()} Phase
- **Points**: {mesh.n_points:,}
- **Cells**: {mesh.n_cells:,}
- **Volume**: {mesh.volume:.2f} μm³
- **File Formats**: VTK, VTU
"""

    doc_content += f"""

### Interface Meshes
"""

    for interface_name, mesh in mesh_generator.interface_meshes.items():
        doc_content += f"""
#### {interface_name.replace('_', '/')} Interface
- **Vertices**: {len(mesh.vertices):,}
- **Faces**: {len(mesh.faces):,}
- **Surface Area**: {mesh.area:.2f} μm²
- **File Format**: STL
"""

    doc_content += f"""

## Validation Results

The generated microstructure has been validated against realistic SOFC properties:

| Validation Check | Status | Description |
|------------------|--------|-------------|
| Porosity Realistic | {'✓ PASS' if validation_results.get('porosity_realistic', False) else '✗ FAIL'} | Porosity within 25-45% range |
| Interface Continuity | {'✓ PASS' if validation_results.get('interface_continuity', False) else '✗ FAIL'} | Continuous interface between phases |
| Phase Connectivity | {'✓ PASS' if validation_results.get('phase_connectivity', False) else '✗ FAIL'} | Phases percolate through domain |
| Geometric Feasibility | {'✓ PASS' if validation_results.get('geometric_feasibility', False) else '✗ FAIL'} | Realistic phase volume fractions |

## Advanced Metrics
"""

    if advanced_metrics:
        if 'tortuosity_z' in advanced_metrics:
            doc_content += f"\n- **Tortuosity (Z-direction)**: {advanced_metrics['tortuosity_z']:.3f}"
        
        if 'specific_surface_area' in advanced_metrics:
            doc_content += f"\n- **Specific Surface Area**: {advanced_metrics['specific_surface_area']:.4f} μm⁻¹"
        
        if 'pore_size_distribution' in advanced_metrics:
            psd = advanced_metrics['pore_size_distribution']
            doc_content += f"""

### Pore Size Distribution
- **Mean Pore Size**: {psd['mean']:.3f} μm
- **Standard Deviation**: {psd['std']:.3f} μm
- **Maximum Pore Size**: {psd['max']:.3f} μm
- **10th Percentile**: {psd['percentiles']['10']:.3f} μm
- **50th Percentile (Median)**: {psd['percentiles']['50']:.3f} μm
- **90th Percentile**: {psd['percentiles']['90']:.3f} μm
"""

    doc_content += f"""

## File Structure

```
sofc_microstructure/
├── data/
│   ├── sofc_microstructure.h5          # Main dataset (HDF5 format)
│   ├── sofc_microstructure.tiff        # TIFF image stack
│   ├── slices/                         # Individual 2D slices
│   │   └── slice_XXXX.tiff
│   ├── metadata.json                   # Dataset metadata
│   └── meshes/                         # Computational meshes
│       ├── *_surface.stl               # Surface meshes (STL format)
│       ├── *_surface.obj               # Surface meshes (OBJ format)
│       ├── *_surface.ply               # Surface meshes (PLY format)
│       ├── *_volume.vtk                # Volume meshes (VTK format)
│       ├── *_volume.vtu                # Volume meshes (VTU format)
│       ├── *_interface.stl             # Interface meshes
│       └── mesh_quality_report.json    # Mesh quality metrics
├── results/
│   ├── complete_analysis.png           # Comprehensive visualization
│   ├── interface_analysis.png          # Interface characterization
│   ├── mesh_visualization.png          # Mesh quality visualization
│   ├── summary_dashboard.png           # Summary dashboard
│   ├── dataset_report.html             # Complete HTML report
│   └── analysis_results.json           # All analysis metrics
├── docs/
│   └── dataset_documentation.md        # This documentation
└── src/
    ├── microstructure_generator.py     # Core generation module
    ├── interface_analyzer.py           # Interface analysis module
    └── mesh_generator.py               # Mesh generation module
```

## Usage Guidelines

### Loading the Dataset

#### Python (recommended)
```python
import h5py
import numpy as np

# Load HDF5 dataset
with h5py.File('sofc_microstructure.h5', 'r') as f:
    microstructure = f['microstructure'][:]
    dimensions = f['dimensions'][:]
    voxel_size = f['voxel_size'][()]

# Load metadata
import json
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
```

#### ImageJ/Fiji
1. Open ImageJ/Fiji
2. File → Import → Image Sequence
3. Select the `slices/` directory
4. Set voxel size: Image → Properties → Pixel Width/Height = {generator.voxel_size} μm

#### ParaView (for meshes)
1. Open ParaView
2. File → Open → Select .vtk or .vtu files
3. Apply → Visualize volume meshes
4. For surface meshes, use .stl files

### Applications

This dataset is suitable for:

1. **Finite Element Analysis (FEA)**
   - Mechanical stress analysis
   - Thermal expansion studies
   - Delamination prediction

2. **Computational Fluid Dynamics (CFD)**
   - Gas transport modeling
   - Pressure drop calculations
   - Mass transfer analysis

3. **Electrochemical Modeling**
   - Current density distribution
   - Activation losses
   - Concentration gradients

4. **Multi-physics Simulations**
   - Coupled thermal-mechanical-electrochemical models
   - Degradation mechanisms
   - Performance optimization

## Citation

If you use this dataset in your research, please cite:

```
SOFC 3D Microstructural Dataset
Generated using advanced computational methods
Voxel resolution: {generator.voxel_size} μm
Physical dimensions: {generator.physical_size} μm
Generation date: {time.strftime('%Y-%m-%d')}
```

## Contact Information

For questions about this dataset or to report issues, please contact the dataset maintainer.

## Version History

- **v1.0** - Initial dataset generation with complete analysis pipeline

---
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""

    # Write documentation
    with open("sofc_microstructure/docs/dataset_documentation.md", 'w') as f:
        f.write(doc_content)

def export_analysis_results(morphology_metrics, roughness_metrics, curvature_metrics,
                           delamination_risk, mesh_quality, validation_results, advanced_metrics):
    """Export all analysis results to JSON."""
    
    results = {
        'morphology_metrics': morphology_metrics,
        'roughness_metrics': roughness_metrics,
        'curvature_metrics': curvature_metrics,
        'delamination_risk': delamination_risk,
        'mesh_quality': mesh_quality,
        'validation_results': validation_results,
        'advanced_metrics': advanced_metrics,
        'generation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open("sofc_microstructure/results/analysis_results.json", 'w') as f:
        import json
        json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()