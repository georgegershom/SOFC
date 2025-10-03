"""
Example Analysis Scripts for SOFC Experimental Data

This script demonstrates how to load and analyze the generated experimental data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def analyze_dic_sintering():
    """Analyze DIC sintering strain data"""
    print("=" * 70)
    print("1. DIC SINTERING STRAIN ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv('sofc_experimental_data/dic_data/sintering_strain_data.csv')
    
    # Summary statistics by region
    print("\nStrain Summary by Region at 1500°C:")
    final_data = df[df['temperature_C'] == 1500]
    summary = final_data.groupby('region')['von_mises_strain'].describe()
    print(summary)
    
    # Identify maximum strain locations
    max_strain_row = final_data.loc[final_data['von_mises_strain'].idxmax()]
    print(f"\nMaximum strain: {max_strain_row['von_mises_strain']*100:.3f}% in {max_strain_row['region']}")
    
    # Plot strain evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    for region in df['region'].unique():
        region_data = df[df['region'] == region]
        ax.plot(region_data['temperature_C'], 
                region_data['von_mises_strain'] * 100,
                label=region, linewidth=2, marker='o', markersize=3, markevery=10)
    
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Von Mises Strain (%)', fontsize=12)
    ax.set_title('Strain Evolution During Sintering', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('analysis_sintering_strain.png', dpi=150)
    print("\n✓ Saved: analysis_sintering_strain.png")
    plt.close()

def analyze_xrd_stress():
    """Analyze XRD residual stress profiles"""
    print("\n" + "=" * 70)
    print("2. XRD RESIDUAL STRESS ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv('sofc_experimental_data/xrd_data/residual_stress_profiles.csv')
    
    # Compare stress levels across conditions
    print("\nPeak Von Mises Stress by Condition:")
    for condition in df['condition'].unique():
        max_stress = df[df['condition'] == condition]['von_mises_stress_MPa'].max()
        print(f"  {condition.replace('_', ' ').title()}: {max_stress:.1f} MPa")
    
    # Identify stress concentration zones
    as_sintered = df[df['condition'] == 'as_sintered']
    high_stress = as_sintered[as_sintered['von_mises_stress_MPa'] > 150]
    print(f"\nHigh stress regions (>150 MPa): {len(high_stress)} locations")
    print(f"Primarily in phases: {high_stress['phase'].value_counts().to_dict()}")
    
    # Plot stress profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Von Mises stress
    for condition in df['condition'].unique():
        condition_data = df[df['condition'] == condition]
        ax1.plot(condition_data['depth_um'], 
                condition_data['von_mises_stress_MPa'],
                label=condition.replace('_', ' '), linewidth=2)
    
    ax1.set_xlabel('Depth (μm)', fontsize=12)
    ax1.set_ylabel('Von Mises Stress (MPa)', fontsize=12)
    ax1.set_title('Residual Stress Profiles', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Phase distribution
    as_sintered = df[df['condition'] == 'as_sintered']
    ax2.scatter(as_sintered['depth_um'], 
               as_sintered['von_mises_stress_MPa'],
               c=as_sintered['phase'].astype('category').cat.codes,
               cmap='Set1', alpha=0.6, s=20)
    ax2.set_xlabel('Depth (μm)', fontsize=12)
    ax2.set_ylabel('Von Mises Stress (MPa)', fontsize=12)
    ax2.set_title('Stress by Phase (As-sintered)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_xrd_stress.png', dpi=150)
    print("\n✓ Saved: analysis_xrd_stress.png")
    plt.close()

def analyze_microcrack_threshold():
    """Analyze microcrack initiation threshold"""
    print("\n" + "=" * 70)
    print("3. MICROCRACK THRESHOLD ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv('sofc_experimental_data/xrd_data/microcrack_threshold_data.csv')
    
    # Calculate critical strain statistics
    cracked_specimens = df[df['cracked'] == True]
    uncracked_specimens = df[df['cracked'] == False]
    
    if len(cracked_specimens) > 0 and len(uncracked_specimens) > 0:
        min_crack_strain = cracked_specimens['applied_strain'].min()
        max_nocrack_strain = uncracked_specimens['applied_strain'].max()
        
        print(f"\nCritical strain range: {max_nocrack_strain*100:.2f}% - {min_crack_strain*100:.2f}%")
        print(f"Number of cracked specimens: {len(cracked_specimens)}/{len(df)}")
        
        # Average crack density
        if len(cracked_specimens) > 0:
            avg_density = cracked_specimens['crack_density_per_mm2'].mean()
            print(f"Average crack density (cracked samples): {avg_density:.2f} cracks/mm²")
    
    # Plot threshold behavior
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Crack occurrence probability
    axes[0].scatter(df['applied_strain'] * 100, 
                   df['cracked'].astype(int),
                   c=df['crack_density_per_mm2'], 
                   cmap='Reds', s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].axvline(x=2.0, color='r', linestyle='--', linewidth=2, label='εcr = 2.0%')
    axes[0].set_xlabel('Applied Strain (%)', fontsize=11)
    axes[0].set_ylabel('Cracked (0=No, 1=Yes)', fontsize=11)
    axes[0].set_title('Crack Initiation', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Crack density evolution
    cracked_data = df[df['cracked']]
    if len(cracked_data) > 0:
        axes[1].scatter(cracked_data['applied_strain'] * 100,
                       cracked_data['crack_density_per_mm2'],
                       s=80, alpha=0.7, c='crimson', edgecolors='k', linewidth=0.5)
        axes[1].set_xlabel('Applied Strain (%)', fontsize=11)
        axes[1].set_ylabel('Crack Density (cracks/mm²)', fontsize=11)
        axes[1].set_title('Crack Density vs Strain', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    
    # XRD peak broadening
    axes[2].scatter(df['applied_strain'] * 100, 
                   df['xrd_peak_fwhm_deg'],
                   c=df['cracked'].astype(int), 
                   cmap='RdYlGn_r', s=80, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[2].set_xlabel('Applied Strain (%)', fontsize=11)
    axes[2].set_ylabel('XRD Peak FWHM (°)', fontsize=11)
    axes[2].set_title('Peak Broadening (Damage)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_microcrack_threshold.png', dpi=150)
    print("\n✓ Saved: analysis_microcrack_threshold.png")
    plt.close()

def analyze_eds_composition():
    """Analyze EDS elemental composition"""
    print("\n" + "=" * 70)
    print("4. EDS ELEMENTAL COMPOSITION ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv('sofc_experimental_data/postmortem_data/eds_analysis/eds_line_scan_cross_section.csv')
    
    # Identify layer boundaries (simplified)
    print("\nElemental composition by layer:")
    
    # Anode region (0-300 μm)
    anode = df[df['distance_um'] < 300]
    print(f"\nAnode (Ni-YSZ) - 0-300 μm:")
    print(f"  Ni: {anode['Ni_wt%'].mean():.1f} ± {anode['Ni_wt%'].std():.1f} wt%")
    print(f"  Zr: {anode['Zr_wt%'].mean():.1f} ± {anode['Zr_wt%'].std():.1f} wt%")
    print(f"  Y:  {anode['Y_wt%'].mean():.1f} ± {anode['Y_wt%'].std():.1f} wt%")
    
    # Electrolyte (300-500 μm)
    electrolyte = df[(df['distance_um'] >= 300) & (df['distance_um'] < 500)]
    print(f"\nElectrolyte (YSZ) - 300-500 μm:")
    print(f"  Zr: {electrolyte['Zr_wt%'].mean():.1f} ± {electrolyte['Zr_wt%'].std():.1f} wt%")
    print(f"  Y:  {electrolyte['Y_wt%'].mean():.1f} ± {electrolyte['Y_wt%'].std():.1f} wt%")
    print(f"  O:  {electrolyte['O_wt%'].mean():.1f} ± {electrolyte['O_wt%'].std():.1f} wt%")
    
    # Cathode (500-800 μm)
    cathode = df[df['distance_um'] >= 500]
    print(f"\nCathode (LSM) - 500-800 μm:")
    print(f"  La: {cathode['La_wt%'].mean():.1f} ± {cathode['La_wt%'].std():.1f} wt%")
    print(f"  Sr: {cathode['Sr_wt%'].mean():.1f} ± {cathode['Sr_wt%'].std():.1f} wt%")
    print(f"  Mn: {cathode['Mn_wt%'].mean():.1f} ± {cathode['Mn_wt%'].std():.1f} wt%")
    
    # Plot composition profile
    fig, ax = plt.subplots(figsize=(12, 6))
    
    elements = {
        'Ni_wt%': ('Ni', 'gray'),
        'Zr_wt%': ('Zr', 'blue'),
        'Y_wt%': ('Y', 'green'),
        'O_wt%': ('O', 'red'),
        'La_wt%': ('La', 'purple'),
        'Sr_wt%': ('Sr', 'orange'),
        'Mn_wt%': ('Mn', 'brown')
    }
    
    for col, (label, color) in elements.items():
        ax.plot(df['distance_um'], df[col], label=label, linewidth=2, color=color)
    
    # Add layer boundaries
    ax.axvspan(0, 300, alpha=0.1, color='gray', label='Anode')
    ax.axvspan(300, 500, alpha=0.1, color='blue', label='Electrolyte')
    ax.axvspan(500, 800, alpha=0.1, color='red', label='Cathode')
    
    ax.set_xlabel('Distance (μm)', fontsize=12)
    ax.set_ylabel('Weight %', fontsize=12)
    ax.set_title('EDS Line Scan Across SOFC Stack', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_eds_composition.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: analysis_eds_composition.png")
    plt.close()

def analyze_nanoindentation():
    """Analyze nano-indentation mechanical properties"""
    print("\n" + "=" * 70)
    print("5. NANO-INDENTATION ANALYSIS")
    print("=" * 70)
    
    df = pd.read_csv('sofc_experimental_data/postmortem_data/nanoindentation/nanoindentation_map.csv')
    
    # Classify by layer based on depth
    def classify_layer(depth_um):
        if depth_um < 300:
            return 'Anode'
        elif depth_um < 500:
            return 'Electrolyte'
        else:
            return 'Cathode'
    
    df['layer'] = df['depth_um'].apply(classify_layer)
    
    # Summary statistics
    print("\nMechanical Properties by Layer:")
    summary = df.groupby('layer').agg({
        'youngs_modulus_GPa': ['mean', 'std'],
        'hardness_GPa': ['mean', 'std'],
        'creep_compliance': ['mean', 'std']
    }).round(2)
    print(summary)
    
    # Identify weak zones (interface regions)
    interfaces = df[((df['depth_um'] > 290) & (df['depth_um'] < 310)) |
                    ((df['depth_um'] > 490) & (df['depth_um'] < 510))]
    
    print(f"\nInterface regions (reduced properties):")
    print(f"  Average E: {interfaces['youngs_modulus_GPa'].mean():.1f} GPa")
    print(f"  Average H: {interfaces['hardness_GPa'].mean():.1f} GPa")
    
    # Create property comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    layers = ['Anode', 'Electrolyte', 'Cathode']
    colors = ['gray', 'blue', 'red']
    
    # Young's modulus
    for layer, color in zip(layers, colors):
        layer_data = df[df['layer'] == layer]
        axes[0].hist(layer_data['youngs_modulus_GPa'], 
                    bins=30, alpha=0.6, color=color, label=layer)
    axes[0].set_xlabel("Young's Modulus (GPa)", fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title("Young's Modulus Distribution", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Hardness
    for layer, color in zip(layers, colors):
        layer_data = df[df['layer'] == layer]
        axes[1].hist(layer_data['hardness_GPa'], 
                    bins=30, alpha=0.6, color=color, label=layer)
    axes[1].set_xlabel('Hardness (GPa)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Hardness Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Depth profile
    depth_profile = df.groupby('depth_um')['youngs_modulus_GPa'].mean()
    axes[2].plot(depth_profile.values, depth_profile.index, linewidth=2, color='darkblue')
    axes[2].axhspan(0, 300, alpha=0.1, color='gray', label='Anode')
    axes[2].axhspan(300, 500, alpha=0.1, color='blue', label='Electrolyte')
    axes[2].axhspan(500, 800, alpha=0.1, color='red', label='Cathode')
    axes[2].set_xlabel("Young's Modulus (GPa)", fontsize=11)
    axes[2].set_ylabel('Depth (μm)', fontsize=11)
    axes[2].set_title('Modulus vs Depth', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_nanoindentation.png', dpi=150)
    print("\n✓ Saved: analysis_nanoindentation.png")
    plt.close()

def generate_summary_report():
    """Generate comprehensive summary report"""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("=" * 70)
    
    report = []
    report.append("SOFC EXPERIMENTAL DATA ANALYSIS SUMMARY")
    report.append("=" * 70)
    report.append("")
    
    # DIC Data
    try:
        df_dic = pd.read_csv('sofc_experimental_data/dic_data/sintering_strain_data.csv')
        max_strain = df_dic['von_mises_strain'].max() * 100
        report.append(f"DIC Analysis:")
        report.append(f"  - Maximum strain observed: {max_strain:.3f}%")
        report.append(f"  - Data points collected: {len(df_dic)}")
    except:
        report.append("DIC data not found")
    
    # XRD Data
    try:
        df_xrd = pd.read_csv('sofc_experimental_data/xrd_data/residual_stress_profiles.csv')
        max_stress = df_xrd['von_mises_stress_MPa'].max()
        report.append(f"\nXRD Analysis:")
        report.append(f"  - Maximum Von Mises stress: {max_stress:.1f} MPa")
        report.append(f"  - Measurement points: {len(df_xrd)}")
    except:
        report.append("\nXRD data not found")
    
    # Crack Analysis
    try:
        df_crack = pd.read_csv('sofc_experimental_data/xrd_data/microcrack_threshold_data.csv')
        crack_rate = (df_crack['cracked'].sum() / len(df_crack)) * 100
        report.append(f"\nMicrocrack Analysis:")
        report.append(f"  - Specimens with cracks: {crack_rate:.1f}%")
        if df_crack['cracked'].any():
            avg_density = df_crack[df_crack['cracked']]['crack_density_per_mm2'].mean()
            report.append(f"  - Average crack density: {avg_density:.2f} cracks/mm²")
    except:
        report.append("\nCrack data not found")
    
    # SEM Analysis
    try:
        df_sem = pd.read_csv('sofc_experimental_data/postmortem_data/sem_analysis/crack_density_analysis.csv')
        report.append(f"\nSEM Analysis:")
        report.append(f"  - Total ROIs analyzed: {len(df_sem)}")
        report.append(f"  - Specimens examined: {df_sem['specimen'].nunique()}")
    except:
        report.append("\nSEM data not found")
    
    # Nano-indentation
    try:
        df_nano = pd.read_csv('sofc_experimental_data/postmortem_data/nanoindentation/nanoindentation_map.csv')
        report.append(f"\nNano-indentation Analysis:")
        report.append(f"  - Total indents: {len(df_nano)}")
        report.append(f"  - Young's modulus range: {df_nano['youngs_modulus_GPa'].min():.1f} - {df_nano['youngs_modulus_GPa'].max():.1f} GPa")
        report.append(f"  - Hardness range: {df_nano['hardness_GPa'].min():.1f} - {df_nano['hardness_GPa'].max():.1f} GPa")
    except:
        report.append("\nNano-indentation data not found")
    
    report.append("")
    report.append("=" * 70)
    report.append("Analysis complete. Visualizations saved as PNG files.")
    report.append("=" * 70)
    
    # Print report
    report_text = "\n".join(report)
    print("\n" + report_text)
    
    # Save report
    with open('ANALYSIS_SUMMARY_REPORT.txt', 'w') as f:
        f.write(report_text)
    
    print("\n✓ Saved: ANALYSIS_SUMMARY_REPORT.txt")

def main():
    """Run all analysis functions"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "SOFC EXPERIMENTAL DATA ANALYSIS" + " " * 22 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    # Check if data exists
    if not os.path.exists('sofc_experimental_data'):
        print("ERROR: Data directory not found!")
        print("Please run 'python3 generate_sofc_experimental_data.py' first.")
        return
    
    # Run analyses
    try:
        analyze_dic_sintering()
        analyze_xrd_stress()
        analyze_microcrack_threshold()
        analyze_eds_composition()
        analyze_nanoindentation()
        generate_summary_report()
        
        print("\n" + "=" * 70)
        print("ALL ANALYSES COMPLETE!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - analysis_sintering_strain.png")
        print("  - analysis_xrd_stress.png")
        print("  - analysis_microcrack_threshold.png")
        print("  - analysis_eds_composition.png")
        print("  - analysis_nanoindentation.png")
        print("  - ANALYSIS_SUMMARY_REPORT.txt")
        print()
        
    except Exception as e:
        print(f"\nERROR during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
