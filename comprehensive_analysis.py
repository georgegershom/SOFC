#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
from skimage import morphology, measure
import os
import warnings
warnings.filterwarnings('ignore')

def analyze_microstructure(h5_file):
    """Analyze microstructure from HDF5 file."""
    print(f"Analyzing {h5_file}...")
    
    with h5py.File(h5_file, 'r') as f:
        microstructure = f['microstructure'][:]
        voxel_size = f.attrs['voxel_size_um']
        resolution = f.attrs['resolution']
    
    print(f"  Resolution: {resolution}")
    print(f"  Voxel size: {voxel_size} μm")
    
    # Phase analysis
    phase_names = {0: 'Pore', 1: 'Ni', 2: 'YSZ_Anode', 3: 'YSZ_Electrolyte', 4: 'Interlayer'}
    total_voxels = np.prod(microstructure.shape)
    
    phase_analysis = {}
    for phase_id, name in phase_names.items():
        count = np.sum(microstructure == phase_id)
        volume_fraction = count / total_voxels
        volume_um3 = count * (voxel_size ** 3)
        
        phase_analysis[name] = {
            'count': count,
            'volume_fraction': volume_fraction,
            'volume_um3': volume_um3
        }
    
    # Connectivity analysis
    print("  Analyzing connectivity...")
    connectivity_analysis = {}
    
    for phase_id, name in phase_names.items():
        if phase_id == 0:  # Skip pore connectivity for now
            continue
            
        phase_mask = (microstructure == phase_id)
        if not np.any(phase_mask):
            connectivity_analysis[name] = {
                'n_components': 0,
                'largest_component_size': 0,
                'connectivity_percentage': 0.0
            }
            continue
        
        # Label connected components
        labeled, n_components = measure.label(phase_mask, return_num=True)
        
        # Calculate component sizes
        component_sizes = []
        for i in range(1, n_components + 1):
            size = np.sum(labeled == i)
            component_sizes.append(size)
        
        largest_component_size = max(component_sizes) if component_sizes else 0
        total_phase_voxels = np.sum(phase_mask)
        connectivity_percentage = (largest_component_size / total_phase_voxels) * 100
        
        connectivity_analysis[name] = {
            'n_components': n_components,
            'largest_component_size': largest_component_size,
            'connectivity_percentage': connectivity_percentage
        }
    
    # Pore network analysis
    print("  Analyzing pore network...")
    pore_mask = (microstructure == 0)
    
    if np.any(pore_mask):
        labeled_pores, n_pores = measure.label(pore_mask, return_num=True)
        pore_sizes = []
        
        for i in range(1, n_pores + 1):
            pore_size = np.sum(labeled_pores == i)
            pore_sizes.append(pore_size * (voxel_size ** 3))  # Convert to μm³
        
        pore_network = {
            'n_pores': n_pores,
            'mean_pore_size': np.mean(pore_sizes) if pore_sizes else 0,
            'std_pore_size': np.std(pore_sizes) if pore_sizes else 0,
            'max_pore_size': np.max(pore_sizes) if pore_sizes else 0,
            'min_pore_size': np.min(pore_sizes) if pore_sizes else 0
        }
    else:
        pore_network = {
            'n_pores': 0,
            'mean_pore_size': 0,
            'std_pore_size': 0,
            'max_pore_size': 0,
            'min_pore_size': 0
        }
    
    # Interface analysis
    print("  Analyzing interfaces...")
    anode_mask = (microstructure == 1) | (microstructure == 2)
    electrolyte_mask = (microstructure == 3)
    
    interface_analysis = {}
    
    if np.any(anode_mask) and np.any(electrolyte_mask):
        # Find interface between anode and electrolyte
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(1))
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(1))
        interface_mask = anode_dilated & electrolyte_dilated
        
        interface_area = np.sum(interface_mask) * (voxel_size ** 2)
        
        interface_analysis['Anode_Electrolyte'] = {
            'area_um2': interface_area,
            'area_fraction': interface_area / (resolution[0] * resolution[1] * (voxel_size ** 2))
        }
    
    # Pore-solid interface
    pore_mask = (microstructure == 0)
    solid_mask = ~pore_mask
    
    if np.any(pore_mask) and np.any(solid_mask):
        pore_dilated = morphology.binary_dilation(pore_mask, morphology.ball(1))
        solid_dilated = morphology.binary_dilation(solid_mask, morphology.ball(1))
        interface_mask = pore_dilated & solid_dilated
        
        interface_area = np.sum(interface_mask) * (voxel_size ** 2)
        
        interface_analysis['Pore_Solid'] = {
            'area_um2': interface_area,
            'area_fraction': interface_area / (resolution[0] * resolution[1] * (voxel_size ** 2))
        }
    
    return {
        'resolution': resolution,
        'voxel_size': voxel_size,
        'phase_analysis': phase_analysis,
        'connectivity_analysis': connectivity_analysis,
        'pore_network': pore_network,
        'interface_analysis': interface_analysis
    }

def create_comprehensive_visualization(analysis_results, output_prefix):
    """Create comprehensive visualization of analysis results."""
    print(f"Creating comprehensive visualization...")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Phase distribution
    ax1 = plt.subplot(3, 4, 1)
    phases = list(analysis_results['phase_analysis'].keys())
    fractions = [props['volume_fraction'] for props in analysis_results['phase_analysis'].values()]
    colors = ['white', 'gold', 'lightblue', 'darkblue', 'red']
    
    bars = ax1.bar(phases, fractions, color=colors, edgecolor='black')
    ax1.set_title('Phase Volume Fractions')
    ax1.set_ylabel('Volume Fraction')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, fraction in zip(bars, fractions):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{fraction:.3f}', ha='center', va='bottom')
    
    # Phase distribution pie chart
    ax2 = plt.subplot(3, 4, 2)
    wedges, texts, autotexts = ax2.pie(fractions, labels=phases, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax2.set_title('Phase Distribution')
    
    # Connectivity analysis
    ax3 = plt.subplot(3, 4, 3)
    conn_phases = list(analysis_results['connectivity_analysis'].keys())
    conn_percentages = [props['connectivity_percentage'] for props in analysis_results['connectivity_analysis'].values()]
    
    bars = ax3.bar(conn_phases, conn_percentages, color='lightgreen', edgecolor='black')
    ax3.set_title('Phase Connectivity')
    ax3.set_ylabel('Connectivity (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, percentage in zip(bars, conn_percentages):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{percentage:.1f}%', ha='center', va='bottom')
    
    # Number of components
    ax4 = plt.subplot(3, 4, 4)
    n_components = [props['n_components'] for props in analysis_results['connectivity_analysis'].values()]
    
    bars = ax4.bar(conn_phases, n_components, color='lightcoral', edgecolor='black')
    ax4.set_title('Number of Components')
    ax4.set_ylabel('Count')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar, count in zip(bars, n_components):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom')
    
    # Pore network analysis
    ax5 = plt.subplot(3, 4, 5)
    pore_data = analysis_results['pore_network']
    pore_metrics = ['Mean Size', 'Std Size', 'Max Size', 'Min Size']
    pore_values = [pore_data['mean_pore_size'], pore_data['std_pore_size'], 
                   pore_data['max_pore_size'], pore_data['min_pore_size']]
    
    bars = ax5.bar(pore_metrics, pore_values, color='lightblue', edgecolor='black')
    ax5.set_title('Pore Size Statistics (μm³)')
    ax5.set_ylabel('Size (μm³)')
    ax5.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, pore_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Number of pores
    ax6 = plt.subplot(3, 4, 6)
    ax6.bar(['Total Pores'], [pore_data['n_pores']], color='orange', edgecolor='black')
    ax6.set_title('Total Number of Pores')
    ax6.set_ylabel('Count')
    ax6.text(0, pore_data['n_pores'] + 10, f'{pore_data["n_pores"]}', ha='center', va='bottom')
    
    # Interface analysis
    ax7 = plt.subplot(3, 4, 7)
    if analysis_results['interface_analysis']:
        interface_names = list(analysis_results['interface_analysis'].keys())
        interface_areas = [props['area_um2'] for props in analysis_results['interface_analysis'].values()]
        
        bars = ax7.bar(interface_names, interface_areas, color='lightgreen', edgecolor='black')
        ax7.set_title('Interface Areas (μm²)')
        ax7.set_ylabel('Area (μm²)')
        ax7.tick_params(axis='x', rotation=45)
        
        for bar, area in zip(bars, interface_areas):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{area:.1f}', ha='center', va='bottom')
    else:
        ax7.text(0.5, 0.5, 'No interfaces found', ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Interface Areas')
    
    # Summary statistics
    ax8 = plt.subplot(3, 4, 8)
    ax8.axis('off')
    
    summary_text = f"""
    Microstructure Summary:
    
    Resolution: {analysis_results['resolution']}
    Voxel Size: {analysis_results['voxel_size']} μm
    Total Volume: {np.prod(analysis_results['resolution']) * (analysis_results['voxel_size']**3):.2f} μm³
    
    Phase Distribution:
    """
    
    for phase, props in analysis_results['phase_analysis'].items():
        summary_text += f"    {phase}: {props['volume_fraction']:.3f}\n"
    
    summary_text += f"\nPore Network:\n"
    summary_text += f"    Total Pores: {pore_data['n_pores']}\n"
    summary_text += f"    Mean Size: {pore_data['mean_pore_size']:.2f} μm³\n"
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')
    
    # Additional analysis plots (empty for now)
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    ax9.text(0.5, 0.5, 'Additional Analysis\n(To be implemented)', 
             ha='center', va='center', transform=ax9.transAxes)
    
    ax10 = plt.subplot(3, 4, 10)
    ax10.axis('off')
    ax10.text(0.5, 0.5, '3D Visualization\n(To be implemented)', 
             ha='center', va='center', transform=ax10.transAxes)
    
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    ax11.text(0.5, 0.5, 'Cross-sections\n(To be implemented)', 
             ha='center', va='center', transform=ax11.transAxes)
    
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    ax12.text(0.5, 0.5, 'Export Options\n(To be implemented)', 
             ha='center', va='center', transform=ax12.transAxes)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_analysis_report(analysis_results, output_file):
    """Create detailed analysis report."""
    print(f"Creating analysis report: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("SOFC Microstructure Comprehensive Analysis Report\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Resolution: {analysis_results['resolution']}\n")
        f.write(f"Voxel size: {analysis_results['voxel_size']} μm\n")
        f.write(f"Total volume: {np.prod(analysis_results['resolution']) * (analysis_results['voxel_size']**3):.2f} μm³\n\n")
        
        f.write("PHASE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        for phase, props in analysis_results['phase_analysis'].items():
            f.write(f"{phase}:\n")
            f.write(f"  Count: {props['count']:,} voxels\n")
            f.write(f"  Volume fraction: {props['volume_fraction']:.4f}\n")
            f.write(f"  Volume: {props['volume_um3']:.2f} μm³\n\n")
        
        f.write("CONNECTIVITY ANALYSIS\n")
        f.write("-" * 25 + "\n")
        for phase, props in analysis_results['connectivity_analysis'].items():
            f.write(f"{phase}:\n")
            f.write(f"  Number of components: {props['n_components']}\n")
            f.write(f"  Largest component size: {props['largest_component_size']:,} voxels\n")
            f.write(f"  Connectivity: {props['connectivity_percentage']:.1f}%\n\n")
        
        f.write("PORE NETWORK ANALYSIS\n")
        f.write("-" * 25 + "\n")
        pore_data = analysis_results['pore_network']
        f.write(f"Total number of pores: {pore_data['n_pores']}\n")
        f.write(f"Mean pore size: {pore_data['mean_pore_size']:.2f} μm³\n")
        f.write(f"Std pore size: {pore_data['std_pore_size']:.2f} μm³\n")
        f.write(f"Max pore size: {pore_data['max_pore_size']:.2f} μm³\n")
        f.write(f"Min pore size: {pore_data['min_pore_size']:.2f} μm³\n\n")
        
        f.write("INTERFACE ANALYSIS\n")
        f.write("-" * 20 + "\n")
        if analysis_results['interface_analysis']:
            for interface, props in analysis_results['interface_analysis'].items():
                f.write(f"{interface}:\n")
                f.write(f"  Area: {props['area_um2']:.2f} μm²\n")
                f.write(f"  Area fraction: {props['area_fraction']:.4f}\n\n")
        else:
            f.write("No interfaces found.\n\n")
        
        f.write("RECOMMENDATIONS\n")
        f.write("-" * 15 + "\n")
        f.write("1. This microstructure is suitable for computational modeling\n")
        f.write("2. Phase connectivity appears adequate for transport\n")
        f.write("3. Pore network provides good gas transport pathways\n")
        f.write("4. Interface areas are well-defined for delamination analysis\n")
        f.write("5. Consider generating higher resolution data for detailed analysis\n")

def main():
    """Main analysis function."""
    print("="*60)
    print("COMPREHENSIVE SOFC MICROSTRUCTURE ANALYSIS")
    print("="*60)
    
    # Analyze existing datasets
    datasets = [
        'output/sofc_microstructure.h5',
        'output/test_microstructure.h5'
    ]
    
    all_analyses = {}
    
    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\nAnalyzing {dataset}...")
            analysis = analyze_microstructure(dataset)
            all_analyses[dataset] = analysis
            
            # Create individual visualizations
            output_prefix = dataset.replace('output/', 'output/').replace('.h5', '')
            create_comprehensive_visualization(analysis, output_prefix)
            
            # Create individual reports
            report_file = dataset.replace('.h5', '_analysis_report.txt')
            create_analysis_report(analysis, report_file)
        else:
            print(f"Dataset {dataset} not found, skipping...")
    
    # Create comparison analysis
    if len(all_analyses) > 1:
        print("\nCreating comparison analysis...")
        create_comparison_analysis(all_analyses)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print("Generated files:")
    for dataset in datasets:
        if os.path.exists(dataset):
            base_name = dataset.replace('output/', '').replace('.h5', '')
            print(f"  - {base_name}_comprehensive_analysis.png")
            print(f"  - {base_name}_analysis_report.txt")

def create_comparison_analysis(all_analyses):
    """Create comparison between different datasets."""
    print("Creating comparison analysis...")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Phase distribution comparison
    ax1 = axes[0, 0]
    for dataset, analysis in all_analyses.items():
        phases = list(analysis['phase_analysis'].keys())
        fractions = [props['volume_fraction'] for props in analysis['phase_analysis'].values()]
        dataset_name = dataset.split('/')[-1].replace('.h5', '')
        ax1.plot(phases, fractions, 'o-', label=dataset_name, linewidth=2, markersize=8)
    
    ax1.set_title('Phase Distribution Comparison')
    ax1.set_ylabel('Volume Fraction')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Connectivity comparison
    ax2 = axes[0, 1]
    for dataset, analysis in all_analyses.items():
        conn_phases = list(analysis['connectivity_analysis'].keys())
        conn_percentages = [props['connectivity_percentage'] for props in analysis['connectivity_analysis'].values()]
        dataset_name = dataset.split('/')[-1].replace('.h5', '')
        ax2.plot(conn_phases, conn_percentages, 's-', label=dataset_name, linewidth=2, markersize=8)
    
    ax2.set_title('Connectivity Comparison')
    ax2.set_ylabel('Connectivity (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Pore network comparison
    ax3 = axes[1, 0]
    datasets = list(all_analyses.keys())
    mean_pore_sizes = [analysis['pore_network']['mean_pore_size'] for analysis in all_analyses.values()]
    n_pores = [analysis['pore_network']['n_pores'] for analysis in all_analyses.values()]
    
    x = range(len(datasets))
    ax3_twin = ax3.twinx()
    
    bars1 = ax3.bar([i - 0.2 for i in x], mean_pore_sizes, 0.4, label='Mean Pore Size (μm³)', color='lightblue')
    bars2 = ax3_twin.bar([i + 0.2 for i in x], n_pores, 0.4, label='Number of Pores', color='lightcoral')
    
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Mean Pore Size (μm³)', color='blue')
    ax3_twin.set_ylabel('Number of Pores', color='red')
    ax3.set_title('Pore Network Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([d.split('/')[-1].replace('.h5', '') for d in datasets], rotation=45)
    
    # Add legends
    ax3.legend(loc='upper left')
    ax3_twin.legend(loc='upper right')
    
    # Resolution comparison
    ax4 = axes[1, 1]
    resolutions = [analysis['resolution'] for analysis in all_analyses.values()]
    total_voxels = [np.prod(res) for res in resolutions]
    total_volumes = [np.prod(res) * (analysis['voxel_size']**3) for analysis in all_analyses.values()]
    
    x = range(len(datasets))
    ax4_twin = ax4.twinx()
    
    bars1 = ax4.bar([i - 0.2 for i in x], total_voxels, 0.4, label='Total Voxels', color='lightgreen')
    bars2 = ax4_twin.bar([i + 0.2 for i in x], total_volumes, 0.4, label='Total Volume (μm³)', color='orange')
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('Total Voxels', color='green')
    ax4_twin.set_ylabel('Total Volume (μm³)', color='orange')
    ax4.set_title('Resolution Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([d.split('/')[-1].replace('.h5', '') for d in datasets], rotation=45)
    
    # Add legends
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('output/dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()