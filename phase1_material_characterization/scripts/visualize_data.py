#!/usr/bin/env python3
"""
Visualization script for Phase 1: Material Characterization Data
Fire-Resistant Rubberized Concrete Project
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define paths
DATA_DIR = Path('../data')
VIZ_DIR = Path('../visualizations')
VIZ_DIR.mkdir(exist_ok=True)

def load_data():
    """Load all JSON data files"""
    data = {}
    data['cement'] = json.load(open(DATA_DIR / 'cement_properties.json'))
    data['aggregates'] = json.load(open(DATA_DIR / 'aggregates_properties.json'))
    data['rubber'] = json.load(open(DATA_DIR / 'crumb_rubber_properties.json'))
    data['water_admix'] = json.load(open(DATA_DIR / 'water_and_admixtures.json'))
    data['mix_designs'] = json.load(open(DATA_DIR / 'mix_designs_matrix.json'))
    data['fresh_props'] = json.load(open(DATA_DIR / 'fresh_concrete_properties.json'))
    return data

def plot_cement_composition(data):
    """Plot cement chemical and mineralogical composition"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Chemical composition
    chem_comp = data['cement']['chemical_composition_xrf']
    major_oxides = {k: v for k, v in chem_comp.items() if v > 1 and k not in ['total', 'free_CaO']}
    
    ax = axes[0, 0]
    bars = ax.bar(major_oxides.keys(), major_oxides.values(), color='steelblue', edgecolor='black')
    ax.set_ylabel('Content (%)', fontsize=12)
    ax.set_title('Major Oxide Composition (XRF)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Bogue composition
    bogue = data['cement']['bogue_composition']
    bogue_comp = {k: v for k, v in bogue.items() if k != 'calculation_method'}
    
    ax = axes[0, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    wedges, texts, autotexts = ax.pie(bogue_comp.values(), labels=bogue_comp.keys(), 
                                        colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Bogue Composition', fontsize=14, fontweight='bold')
    
    # XRD phases
    xrd = data['cement']['mineralogical_analysis_xrd']['phases']
    major_phases = {k: v for k, v in xrd.items() if v > 2}
    
    ax = axes[1, 0]
    bars = ax.barh(list(major_phases.keys()), list(major_phases.values()), 
                   color='coral', edgecolor='black')
    ax.set_xlabel('Content (%)', fontsize=12)
    ax.set_title('XRD Mineralogical Phases (>2%)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Strength development
    strength = data['cement']['mechanical_properties']['compressive_strength']
    days = [3, 7, 28]
    values = [strength['3_days']['value'], strength['7_days']['value'], 
              strength['28_days']['value']]
    
    ax = axes[1, 1]
    ax.plot(days, values, 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel('Age (days)', fontsize=12)
    ax.set_ylabel('Compressive Strength (MPa)', fontsize=12)
    ax.set_title('Cement Mortar Strength Development', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    ax.set_ylim(20, 60)
    
    for day, val in zip(days, values):
        ax.text(day, val + 1, f'{val:.1f} MPa', ha='center', fontsize=10)
    
    plt.suptitle('Cement Characterization Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'cement_characterization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_aggregate_gradation(data):
    """Plot aggregate sieve analysis"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Coarse aggregate
    coarse = data['aggregates']['coarse_aggregate']['sieve_analysis']['sieve_data']
    coarse_df = pd.DataFrame(coarse)
    coarse_df = coarse_df[coarse_df['sieve_size_mm'] > 0]
    
    ax = axes[0]
    ax.semilogx(coarse_df['sieve_size_mm'], coarse_df['passing_percent'], 
                'o-', linewidth=2, markersize=6, label='Tested', color='blue')
    
    # Add ASTM limits for nominal 19mm aggregate
    sizes = [25, 19, 12.5, 9.5, 4.75, 2.36]
    lower = [100, 90, 40, 20, 0, 0]
    upper = [100, 100, 70, 55, 15, 5]
    
    ax.semilogx(sizes, lower, '--', color='red', alpha=0.5, label='ASTM Lower')
    ax.semilogx(sizes, upper, '--', color='green', alpha=0.5, label='ASTM Upper')
    ax.fill_between(sizes, lower, upper, alpha=0.2, color='gray')
    
    ax.set_xlabel('Sieve Size (mm)', fontsize=12)
    ax.set_ylabel('Percent Passing (%)', fontsize=12)
    ax.set_title('Coarse Aggregate Gradation (19mm Nominal)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(40, 0.5)
    ax.set_ylim(0, 105)
    ax.legend()
    
    # Fine aggregate
    fine = data['aggregates']['fine_aggregate']['sieve_analysis']['sieve_data']
    fine_df = pd.DataFrame(fine)
    fine_df = fine_df[fine_df['sieve_size_mm'] > 0]
    
    ax = axes[1]
    ax.semilogx(fine_df['sieve_size_mm'], fine_df['passing_percent'], 
                'o-', linewidth=2, markersize=6, label='Tested', color='brown')
    
    # Add ASTM limits for Zone II
    sizes = [4.75, 2.36, 1.18, 0.6, 0.3, 0.15]
    lower = [90, 75, 55, 35, 8, 0]
    upper = [100, 100, 90, 59, 30, 10]
    
    ax.semilogx(sizes, lower, '--', color='red', alpha=0.5, label='Zone II Lower')
    ax.semilogx(sizes, upper, '--', color='green', alpha=0.5, label='Zone II Upper')
    ax.fill_between(sizes, lower, upper, alpha=0.2, color='gray')
    
    ax.set_xlabel('Sieve Size (mm)', fontsize=12)
    ax.set_ylabel('Percent Passing (%)', fontsize=12)
    ax.set_title(f'Fine Aggregate Gradation (FM = {data["aggregates"]["fine_aggregate"]["sieve_analysis"]["fineness_modulus"]:.2f})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(10, 0.05)
    ax.set_ylim(0, 105)
    ax.legend()
    
    plt.suptitle('Aggregate Gradation Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'aggregate_gradation.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rubber_analysis(data):
    """Plot rubber characterization data"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    rubber_data = data['rubber']['crumb_rubber_characterization']
    
    # Particle size distribution - Fine
    ax = axes[0, 0]
    fine_sieve = rubber_data['particle_size_distribution']['size_range_1_4mm']['sieve_analysis']
    fine_df = pd.DataFrame(fine_sieve)
    fine_df = fine_df[fine_df['sieve_size_mm'] > 0]
    
    ax.semilogx(fine_df['sieve_size_mm'], fine_df['passing_percent'], 
                'o-', linewidth=2, markersize=6, color='purple', label='Fine (1-4mm)')
    
    # Particle size distribution - Coarse
    coarse_sieve = rubber_data['particle_size_distribution']['size_range_4_8mm']['sieve_analysis']
    coarse_df = pd.DataFrame(coarse_sieve)
    coarse_df = coarse_df[coarse_df['sieve_size_mm'] > 0]
    
    ax.semilogx(coarse_df['sieve_size_mm'], coarse_df['passing_percent'], 
                's-', linewidth=2, markersize=6, color='orange', label='Coarse (4-8mm)')
    
    ax.set_xlabel('Particle Size (mm)', fontsize=11)
    ax.set_ylabel('Percent Passing (%)', fontsize=11)
    ax.set_title('Rubber Particle Size Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    ax.set_xlim(10, 0.2)
    
    # Chemical composition
    ax = axes[0, 1]
    polymer_comp = rubber_data['chemical_composition']['polymer_composition']
    major_comp = {k: v for k, v in polymer_comp.items() 
                  if k != 'unit' and isinstance(v, (int, float)) and (v > 5 and ('rubber' in k.lower() or 'carbon' in k.lower()))}
    
    bars = ax.bar(range(len(major_comp)), list(major_comp.values()), 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_xticks(range(len(major_comp)))
    ax.set_xticklabels([k.replace('_', ' ').title()[:15] for k in major_comp.keys()], 
                        rotation=45, ha='right')
    ax.set_ylabel('Content (%)', fontsize=11)
    ax.set_title('Major Polymer Components', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # TGA analysis
    ax = axes[0, 2]
    tga_stages = rubber_data['thermal_analysis']['tga_thermogravimetric']['decomposition_stages']
    temps = []
    weights = []
    labels = []
    
    cumulative_loss = 0
    temps.append(25)
    weights.append(100)
    
    for stage in tga_stages[:4]:
        temp_range = stage['temperature_range'].split('-')
        temps.append(float(temp_range[1].replace('°C', '')))
        cumulative_loss += stage['weight_loss']
        weights.append(100 - cumulative_loss)
        labels.append(f"Stage {stage['stage']}")
    
    ax.plot(temps, weights, 'o-', linewidth=2, markersize=6, color='red')
    ax.set_xlabel('Temperature (°C)', fontsize=11)
    ax.set_ylabel('Weight Remaining (%)', fontsize=11)
    ax.set_title('TGA Decomposition Profile', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 650)
    ax.set_ylim(20, 105)
    
    # FTIR peaks
    ax = axes[1, 0]
    ftir_peaks = rubber_data['ftir_spectroscopy']['major_peaks_cm-1']
    wavenumbers = [p['wavenumber'] for p in ftir_peaks]
    assignments = [p['assignment'][:20] for p in ftir_peaks]
    
    bars = ax.barh(range(len(wavenumbers)), wavenumbers, color='teal')
    ax.set_yticks(range(len(assignments)))
    ax.set_yticklabels(assignments, fontsize=9)
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=11)
    ax.set_title('FTIR Major Peaks', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Surface properties
    ax = axes[1, 1]
    surface_props = {
        'Surface Area': rubber_data['surface_properties']['surface_area_bet']['value'],
        'Contact Angle': rubber_data['surface_properties']['water_contact_angle']['value'],
        'Surface Energy': rubber_data['surface_properties']['surface_energy']['total']
    }
    
    x = np.arange(len(surface_props))
    bars = ax.bar(x, surface_props.values(), color=['skyblue', 'lightcoral', 'lightgreen'])
    ax.set_xticks(x)
    ax.set_xticklabels(surface_props.keys(), rotation=45, ha='right')
    ax.set_title('Surface Properties', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(surface_props.items()):
        unit = 'm²/g' if 'Area' in k else '°' if 'Angle' in k else 'mJ/m²'
        ax.text(i, v + 2, f'{v:.1f} {unit}', ha='center', fontsize=9)
    
    # Mechanical properties
    ax = axes[1, 2]
    mech_props = rubber_data['mechanical_properties']
    props_data = {
        'Tensile\nStrength': mech_props['tensile_strength']['value'],
        'Shore A\nHardness': mech_props['shore_a_hardness'],
        'Resilience': mech_props['resilience']['value']
    }
    
    x = np.arange(len(props_data))
    bars = ax.bar(x, props_data.values(), color=['#FF9999', '#66B2FF', '#99FF99'])
    ax.set_xticks(x)
    ax.set_xticklabels(props_data.keys())
    ax.set_title('Mechanical Properties', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, (k, v) in enumerate(props_data.items()):
        unit = 'MPa' if 'Tensile' in k else '' if 'Hardness' in k else '%'
        ax.text(i, v + 1, f'{v:.1f} {unit}', ha='center', fontsize=9)
    
    plt.suptitle('Crumb Rubber Characterization Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'rubber_characterization.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_mix_design_comparison(data):
    """Plot mix design proportions comparison"""
    mix_data = data['mix_designs']['mix_designs']
    
    # Prepare data
    mix_names = []
    rubber_content = []
    water = []
    cement = []
    sp_dosage = []
    
    for mix_id, mix_info in mix_data.items():
        mix_names.append(mix_info['designation'][:15])
        rubber_content.append(mix_info['rubber_replacement']['percentage'])
        water.append(mix_info['proportions_kg_m3']['water'])
        cement.append(mix_info['proportions_kg_m3']['cement'])
        sp_dosage.append(mix_info['proportions_kg_m3']['superplasticizer'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rubber content bar chart
    ax = axes[0, 0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(mix_names)))
    bars = ax.bar(range(len(mix_names)), rubber_content, color=colors, edgecolor='black')
    ax.set_xticks(range(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Rubber Content (%)', fontsize=12)
    ax.set_title('Rubber Replacement Levels', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Water content
    ax = axes[0, 1]
    ax.plot(rubber_content, water, 'o-', linewidth=2, markersize=8, color='blue')
    ax.set_xlabel('Rubber Content (%)', fontsize=12)
    ax.set_ylabel('Water Content (kg/m³)', fontsize=12)
    ax.set_title('Water Demand vs Rubber Content', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Calculate trend
    z = np.polyfit(rubber_content, water, 1)
    p = np.poly1d(z)
    ax.plot(rubber_content, p(rubber_content), "--", alpha=0.5, color='red', 
            label=f'Trend: y = {z[0]:.2f}x + {z[1]:.1f}')
    ax.legend()
    
    # Superplasticizer dosage
    ax = axes[1, 0]
    ax.scatter(rubber_content, sp_dosage, s=100, c=rubber_content, cmap='viridis', 
               edgecolor='black', linewidth=1)
    ax.set_xlabel('Rubber Content (%)', fontsize=12)
    ax.set_ylabel('Superplasticizer (kg/m³)', fontsize=12)
    ax.set_title('Superplasticizer Demand', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Fit exponential trend
    from scipy.optimize import curve_fit
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c
    
    popt, _ = curve_fit(exp_func, rubber_content, sp_dosage)
    x_smooth = np.linspace(0, 20, 100)
    ax.plot(x_smooth, exp_func(x_smooth, *popt), '--', color='red', alpha=0.5,
            label='Exponential fit')
    ax.legend()
    
    # W/C ratio
    ax = axes[1, 1]
    wc_ratio = [w/c for w, c in zip(water, cement)]
    ax.bar(range(len(mix_names)), wc_ratio, color='coral', edgecolor='black')
    ax.set_xticks(range(len(mix_names)))
    ax.set_xticklabels(mix_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('W/C Ratio', fontsize=12)
    ax.set_title('Water-Cement Ratio', fontsize=14, fontweight='bold')
    ax.axhline(y=0.45, color='green', linestyle='--', alpha=0.5, label='Target W/C')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    plt.suptitle('Mix Design Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'mix_design_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_fresh_properties(data):
    """Plot fresh concrete properties"""
    fresh_data = data['fresh_props']['fresh_properties_results']
    
    # Prepare data
    mix_ids = []
    rubber_pct = []
    slump_initial = []
    air_content = []
    unit_weight = []
    setting_initial = []
    
    for mix_id, props in fresh_data.items():
        mix_ids.append(mix_id)
        # Extract rubber percentage from mix_id
        if 'CM' in mix_id:
            rubber_pct.append(0)
        else:
            pct = int(mix_id.split('-')[1])
            rubber_pct.append(pct)
        
        slump_initial.append(props['slump']['initial']['value'])
        air_content.append(props['air_content']['value'])
        unit_weight.append(props['unit_weight']['value'])
        setting_initial.append(props['setting_time']['initial']['value'])
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Slump vs rubber content
    ax = axes[0, 0]
    scatter = ax.scatter(rubber_pct, slump_initial, s=100, c=rubber_pct, 
                        cmap='coolwarm', edgecolor='black', linewidth=1)
    ax.set_xlabel('Rubber Content (%)', fontsize=12)
    ax.set_ylabel('Initial Slump (mm)', fontsize=12)
    ax.set_title('Workability vs Rubber Content', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(rubber_pct, slump_initial, 2)
    p = np.poly1d(z)
    x_smooth = np.linspace(0, 20, 100)
    ax.plot(x_smooth, p(x_smooth), '--', color='red', alpha=0.5, label='Quadratic fit')
    ax.legend()
    
    # Air content
    ax = axes[0, 1]
    ax.bar(range(len(mix_ids)), air_content, color=plt.cm.YlOrRd(np.array(rubber_pct)/20))
    ax.set_xticks(range(len(mix_ids)))
    ax.set_xticklabels([m[:8] for m in mix_ids], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Air Content (%)', fontsize=12)
    ax.set_title('Air Content by Mix', fontsize=14, fontweight='bold')
    ax.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='Target')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # Unit weight
    ax = axes[0, 2]
    ax.plot(rubber_pct, unit_weight, 'o-', linewidth=2, markersize=8, color='brown')
    ax.set_xlabel('Rubber Content (%)', fontsize=12)
    ax.set_ylabel('Unit Weight (kg/m³)', fontsize=12)
    ax.set_title('Density Reduction', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Calculate percentage reduction
    control_weight = unit_weight[0]
    for i, (rub, weight) in enumerate(zip(rubber_pct, unit_weight)):
        if rub > 0:
            reduction = (1 - weight/control_weight) * 100
            ax.annotate(f'-{reduction:.1f}%', xy=(rub, weight), 
                       xytext=(rub, weight-30), fontsize=8, ha='center')
    
    # Setting time
    ax = axes[1, 0]
    ax.scatter(rubber_pct, setting_initial, s=100, marker='^', color='purple', 
              edgecolor='black', linewidth=1)
    ax.set_xlabel('Rubber Content (%)', fontsize=12)
    ax.set_ylabel('Initial Setting Time (min)', fontsize=12)
    ax.set_title('Setting Time Delay', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Slump retention
    ax = axes[1, 1]
    slump_30 = []
    slump_60 = []
    for mix_id, props in fresh_data.items():
        slump_30.append(props['slump']['30_min']['value'])
        slump_60.append(props['slump']['60_min']['value'])
    
    x = np.arange(len(mix_ids))
    width = 0.35
    bars1 = ax.bar(x - width/2, slump_initial, width, label='Initial', color='skyblue')
    bars2 = ax.bar(x + width/2, slump_60, width, label='60 min', color='coral')
    
    ax.set_xticks(x)
    ax.set_xticklabels([m[:8] for m in mix_ids], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Slump (mm)', fontsize=12)
    ax.set_title('Slump Retention', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Rheology
    ax = axes[1, 2]
    yield_stress = []
    plastic_visc = []
    for mix_id, props in fresh_data.items():
        if 'rheology' in props:
            yield_stress.append(props['rheology']['yield_stress']['value'])
            plastic_visc.append(props['rheology']['plastic_viscosity']['value'])
    
    if yield_stress and plastic_visc:
        scatter = ax.scatter(yield_stress, plastic_visc, s=100, c=rubber_pct[:len(yield_stress)], 
                           cmap='plasma', edgecolor='black', linewidth=1)
        ax.set_xlabel('Yield Stress (Pa)', fontsize=12)
        ax.set_ylabel('Plastic Viscosity (Pa.s)', fontsize=12)
        ax.set_title('Rheological Properties', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Rubber %')
    
    plt.suptitle('Fresh Concrete Properties Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'fresh_properties_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_table(data):
    """Generate summary tables for the report"""
    # Mix design summary
    mix_data = data['mix_designs']['mix_designs']
    fresh_data = data['fresh_props']['fresh_properties_results']
    
    summary = []
    for mix_id, mix_info in mix_data.items():
        if mix_id in fresh_data:
            fresh = fresh_data[mix_id]
            summary.append({
                'Mix ID': mix_id,
                'Rubber %': mix_info['rubber_replacement']['percentage'],
                'Cement (kg/m³)': mix_info['proportions_kg_m3']['cement'],
                'Water (kg/m³)': mix_info['proportions_kg_m3']['water'],
                'W/C Ratio': mix_info['actual_wc_ratio'],
                'Slump (mm)': fresh['slump']['initial']['value'],
                'Air (%)': fresh['air_content']['value'],
                'Density (kg/m³)': fresh['unit_weight']['value']
            })
    
    df_summary = pd.DataFrame(summary)
    
    # Create a nice looking table plot
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_summary.values,
                    colLabels=df_summary.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.08, 0.08, 0.12, 0.1, 0.08, 0.1, 0.08, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style the header
    for i in range(len(df_summary.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(df_summary) + 1):
        for j in range(len(df_summary.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Mix Design and Fresh Properties Summary', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(VIZ_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save as CSV
    df_summary.to_csv(DATA_DIR / 'mix_design_summary.csv', index=False)
    print(f"Summary saved to {DATA_DIR / 'mix_design_summary.csv'}")

def main():
    """Main execution function"""
    print("Loading data...")
    data = load_data()
    
    print("Generating cement characterization plots...")
    plot_cement_composition(data)
    
    print("Generating aggregate gradation plots...")
    plot_aggregate_gradation(data)
    
    print("Generating rubber analysis plots...")
    plot_rubber_analysis(data)
    
    print("Generating mix design comparison plots...")
    plot_mix_design_comparison(data)
    
    print("Generating fresh properties plots...")
    plot_fresh_properties(data)
    
    print("Generating summary tables...")
    generate_summary_table(data)
    
    print(f"\nAll visualizations saved to {VIZ_DIR}")
    print("Data analysis complete!")

if __name__ == "__main__":
    main()