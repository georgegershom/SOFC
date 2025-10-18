#!/usr/bin/env python3
"""
Export all Phase 1 data to various formats for easy sharing and analysis
"""

import json
import pandas as pd
from pathlib import Path
import xlsxwriter

# Define paths
DATA_DIR = Path('../data')
EXPORT_DIR = Path('../exports')
EXPORT_DIR.mkdir(exist_ok=True)

def load_all_data():
    """Load all JSON data files"""
    data = {}
    for json_file in DATA_DIR.glob('*.json'):
        with open(json_file, 'r') as f:
            data[json_file.stem] = json.load(f)
    return data

def export_to_excel(data):
    """Export all data to a single Excel workbook with multiple sheets"""
    excel_file = EXPORT_DIR / 'phase1_complete_dataset.xlsx'
    
    with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        
        # Define formats
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#4CAF50',
            'font_color': 'white',
            'border': 1
        })
        
        # 1. Cement Properties Sheet
        cement_data = []
        cement = data['cement_properties']
        
        # Chemical composition
        for oxide, value in cement['chemical_composition_xrf'].items():
            if oxide not in ['total']:
                cement_data.append(['Chemical Composition', oxide, value, '%'])
        
        # Bogue composition
        for phase, value in cement['bogue_composition'].items():
            if phase != 'calculation_method':
                cement_data.append(['Bogue Composition', phase, value, '%'])
        
        # Physical properties
        cement_data.append(['Physical Properties', 'Specific Gravity', 
                          cement['physical_properties']['specific_gravity'], '-'])
        cement_data.append(['Physical Properties', 'Blaine Fineness', 
                          cement['physical_properties']['fineness_blaine']['value'], 'm²/kg'])
        
        df_cement = pd.DataFrame(cement_data, columns=['Category', 'Property', 'Value', 'Unit'])
        df_cement.to_excel(writer, sheet_name='Cement', index=False)
        
        # 2. Aggregates Sheet
        agg_data = []
        agg = data['aggregates_properties']
        
        # Coarse aggregate
        coarse = agg['coarse_aggregate']
        agg_data.append(['Coarse', 'Type', coarse['type'], '-'])
        agg_data.append(['Coarse', 'Specific Gravity (SSD)', 
                        coarse['physical_properties']['specific_gravity']['bulk_ssd'], '-'])
        agg_data.append(['Coarse', 'Water Absorption', 
                        coarse['physical_properties']['water_absorption']['value'], '%'])
        agg_data.append(['Coarse', 'Fineness Modulus', 
                        coarse['sieve_analysis']['fineness_modulus'], '-'])
        
        # Fine aggregate
        fine = agg['fine_aggregate']
        agg_data.append(['Fine', 'Type', fine['type'], '-'])
        agg_data.append(['Fine', 'Specific Gravity (SSD)', 
                        fine['physical_properties']['specific_gravity']['bulk_ssd'], '-'])
        agg_data.append(['Fine', 'Water Absorption', 
                        fine['physical_properties']['water_absorption']['value'], '%'])
        agg_data.append(['Fine', 'Fineness Modulus', 
                        fine['sieve_analysis']['fineness_modulus'], '-'])
        
        df_agg = pd.DataFrame(agg_data, columns=['Type', 'Property', 'Value', 'Unit'])
        df_agg.to_excel(writer, sheet_name='Aggregates', index=False)
        
        # 3. Rubber Properties Sheet
        rubber_data = []
        rubber = data['crumb_rubber_properties']['crumb_rubber_characterization']
        
        rubber_data.append(['Physical', 'Specific Gravity', 
                          rubber['physical_properties']['specific_gravity'], '-'])
        rubber_data.append(['Physical', 'Bulk Density (Loose)', 
                          rubber['physical_properties']['bulk_density']['loose']['value'], 'kg/m³'])
        rubber_data.append(['Physical', 'Mohs Hardness', 
                          rubber['physical_properties']['mohs_hardness'], '-'])
        rubber_data.append(['Surface', 'Surface Area (BET)', 
                          rubber['surface_properties']['surface_area_bet']['value'], 'm²/g'])
        rubber_data.append(['Surface', 'Water Contact Angle', 
                          rubber['surface_properties']['water_contact_angle']['value'], '°'])
        rubber_data.append(['Mechanical', 'Tensile Strength', 
                          rubber['mechanical_properties']['tensile_strength']['value'], 'MPa'])
        
        df_rubber = pd.DataFrame(rubber_data, columns=['Category', 'Property', 'Value', 'Unit'])
        df_rubber.to_excel(writer, sheet_name='Rubber', index=False)
        
        # 4. Mix Designs Sheet
        mix_designs = []
        mixes = data['mix_designs_matrix']['mix_designs']
        
        for mix_id, mix_data in mixes.items():
            # Get rubber content, handling different key names
            rubber_kg = mix_data['proportions_kg_m3'].get('crumb_rubber', 0)
            if rubber_kg == 0:
                rubber_kg = mix_data['proportions_kg_m3'].get('crumb_rubber_fine', 0) + \
                           mix_data['proportions_kg_m3'].get('crumb_rubber_coarse', 0)
            
            mix_designs.append({
                'Mix ID': mix_id,
                'Description': mix_data['designation'],
                'Rubber %': mix_data['rubber_replacement']['percentage'],
                'Cement (kg/m³)': mix_data['proportions_kg_m3']['cement'],
                'Water (kg/m³)': mix_data['proportions_kg_m3']['water'],
                'Fine Agg (kg/m³)': mix_data['proportions_kg_m3']['fine_aggregate'],
                'Coarse Agg (kg/m³)': mix_data['proportions_kg_m3']['coarse_aggregate'],
                'Rubber (kg/m³)': rubber_kg,
                'SP (kg/m³)': mix_data['proportions_kg_m3']['superplasticizer'],
                'W/C Ratio': mix_data['actual_wc_ratio']
            })
        
        df_mixes = pd.DataFrame(mix_designs)
        df_mixes.to_excel(writer, sheet_name='Mix Designs', index=False)
        
        # 5. Fresh Properties Sheet
        fresh_props = []
        fresh = data['fresh_concrete_properties']['fresh_properties_results']
        
        for mix_id, props in fresh.items():
            fresh_props.append({
                'Mix ID': mix_id,
                'Slump (mm)': props['slump']['initial']['value'],
                'Slump 30min (mm)': props['slump']['30_min']['value'],
                'Slump 60min (mm)': props['slump']['60_min']['value'],
                'Air Content (%)': props['air_content']['value'],
                'Unit Weight (kg/m³)': props['unit_weight']['value'],
                'Temperature (°C)': props['concrete_temperature']['value'],
                'Initial Set (min)': props['setting_time']['initial']['value'],
                'Final Set (min)': props['setting_time']['final']['value']
            })
        
        df_fresh = pd.DataFrame(fresh_props)
        df_fresh.to_excel(writer, sheet_name='Fresh Properties', index=False)
        
        # Format all sheets
        for sheet in writer.sheets.values():
            sheet.set_column('A:Z', 15)
    
    print(f"Excel file created: {excel_file}")
    return excel_file

def export_aggregates_sieve_analysis(data):
    """Export detailed sieve analysis data"""
    csv_file = EXPORT_DIR / 'sieve_analysis_data.csv'
    
    sieve_data = []
    
    # Coarse aggregate
    coarse_sieves = data['aggregates_properties']['coarse_aggregate']['sieve_analysis']['sieve_data']
    for sieve in coarse_sieves:
        if sieve.get('sieve_size_mm', 0) > 0:
            sieve_data.append({
                'Material': 'Coarse Aggregate',
                'Sieve Size (mm)': sieve['sieve_size_mm'],
                'Retained (g)': sieve.get('retained_g', 0),
                'Retained (%)': sieve.get('retained_percent', 0),
                'Cumulative Retained (%)': sieve.get('cumulative_retained_percent', 0),
                'Passing (%)': sieve.get('passing_percent', 0)
            })
    
    # Fine aggregate
    fine_sieves = data['aggregates_properties']['fine_aggregate']['sieve_analysis']['sieve_data']
    for sieve in fine_sieves:
        if sieve.get('sieve_size_mm', 0) > 0:
            sieve_data.append({
                'Material': 'Fine Aggregate',
                'Sieve Size (mm)': sieve['sieve_size_mm'],
                'Retained (g)': sieve.get('retained_g', 0),
                'Retained (%)': sieve.get('retained_percent', 0),
                'Cumulative Retained (%)': sieve.get('cumulative_retained_percent', 0),
                'Passing (%)': sieve.get('passing_percent', 0)
            })
    
    # Rubber - Fine
    rubber_fine = data['crumb_rubber_properties']['crumb_rubber_characterization']['particle_size_distribution']['size_range_1_4mm']['sieve_analysis']
    for sieve in rubber_fine:
        if sieve.get('sieve_size_mm', 0) > 0:
            sieve_data.append({
                'Material': 'Rubber Fine (1-4mm)',
                'Sieve Size (mm)': sieve['sieve_size_mm'],
                'Retained (g)': 0,  # Not provided
                'Retained (%)': sieve.get('retained_percent', 0),
                'Cumulative Retained (%)': sieve.get('cumulative_retained_percent', 0),
                'Passing (%)': sieve.get('passing_percent', 0)
            })
    
    # Rubber - Coarse
    rubber_coarse = data['crumb_rubber_properties']['crumb_rubber_characterization']['particle_size_distribution']['size_range_4_8mm']['sieve_analysis']
    for sieve in rubber_coarse:
        if sieve.get('sieve_size_mm', 0) > 0:
            sieve_data.append({
                'Material': 'Rubber Coarse (4-8mm)',
                'Sieve Size (mm)': sieve['sieve_size_mm'],
                'Retained (g)': 0,  # Not provided
                'Retained (%)': sieve.get('retained_percent', 0),
                'Cumulative Retained (%)': sieve.get('cumulative_retained_percent', 0),
                'Passing (%)': sieve.get('passing_percent', 0)
            })
    
    df_sieve = pd.DataFrame(sieve_data)
    df_sieve.to_csv(csv_file, index=False)
    
    print(f"Sieve analysis CSV created: {csv_file}")
    return csv_file

def export_chemical_analysis(data):
    """Export all chemical composition data"""
    csv_file = EXPORT_DIR / 'chemical_analysis.csv'
    
    chem_data = []
    
    # Cement XRF
    for oxide, value in data['cement_properties']['chemical_composition_xrf'].items():
        if oxide != 'total':
            chem_data.append({
                'Material': 'Cement',
                'Analysis': 'XRF',
                'Component': oxide,
                'Content (%)': value
            })
    
    # Cement XRD
    for phase, value in data['cement_properties']['mineralogical_analysis_xrd']['phases'].items():
        chem_data.append({
            'Material': 'Cement',
            'Analysis': 'XRD',
            'Component': phase,
            'Content (%)': value
        })
    
    # Rubber elemental
    for element, value in data['crumb_rubber_properties']['crumb_rubber_characterization']['chemical_composition']['elemental_analysis'].items():
        if element != 'unit':
            chem_data.append({
                'Material': 'Crumb Rubber',
                'Analysis': 'Elemental',
                'Component': element.capitalize(),
                'Content (%)': value
            })
    
    # Rubber polymer
    for polymer, value in data['crumb_rubber_properties']['crumb_rubber_characterization']['chemical_composition']['polymer_composition'].items():
        if polymer != 'unit' and isinstance(value, (int, float)):
            chem_data.append({
                'Material': 'Crumb Rubber',
                'Analysis': 'Polymer',
                'Component': polymer.replace('_', ' ').title(),
                'Content (%)': value
            })
    
    df_chem = pd.DataFrame(chem_data)
    df_chem.to_csv(csv_file, index=False)
    
    print(f"Chemical analysis CSV created: {csv_file}")
    return csv_file

def create_summary_statistics():
    """Create summary statistics file"""
    stats_file = EXPORT_DIR / 'summary_statistics.txt'
    
    with open(stats_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PHASE 1 MATERIAL CHARACTERIZATION - SUMMARY STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Dataset Overview:\n")
        f.write("-" * 40 + "\n")
        f.write("• Total mix designs: 12\n")
        f.write("• Rubber replacement levels: 0%, 5%, 10%, 15%, 20%\n")
        f.write("• Rubber sizes tested: Fine (1-4mm), Coarse (4-8mm), Mixed\n")
        f.write("• Special mixes: 2 with silica fume addition\n\n")
        
        f.write("Material Properties Ranges:\n")
        f.write("-" * 40 + "\n")
        f.write("Cement:\n")
        f.write("  • Specific Gravity: 3.15\n")
        f.write("  • Blaine Fineness: 382 m²/kg\n")
        f.write("  • 28-day Strength: 52.8 MPa\n\n")
        
        f.write("Aggregates:\n")
        f.write("  • Coarse SG: 2.68-2.76\n")
        f.write("  • Fine SG: 2.63-2.68\n")
        f.write("  • Coarse FM: 6.81\n")
        f.write("  • Fine FM: 2.72\n\n")
        
        f.write("Crumb Rubber:\n")
        f.write("  • Specific Gravity: 1.15\n")
        f.write("  • Bulk Density: 385-468 kg/m³\n")
        f.write("  • Contact Angle: 118° (Hydrophobic)\n\n")
        
        f.write("Fresh Properties Ranges:\n")
        f.write("-" * 40 + "\n")
        f.write("  • Slump: 115-180 mm\n")
        f.write("  • Air Content: 2.1-5.2%\n")
        f.write("  • Unit Weight: 2198-2385 kg/m³\n")
        f.write("  • Initial Setting: 285-345 min\n\n")
        
        f.write("Key Trends:\n")
        f.write("-" * 40 + "\n")
        f.write("  • Workability reduction: ~36% at 20% rubber\n")
        f.write("  • Density reduction: ~7.7% at 20% rubber\n")
        f.write("  • Air content increase: ~148% at 20% rubber\n")
        f.write("  • SP demand increase: ~100% at 20% rubber\n\n")
        
        f.write("Testing Standards Applied:\n")
        f.write("-" * 40 + "\n")
        f.write("  • ASTM C150 (Cement)\n")
        f.write("  • ASTM C136 (Sieve Analysis)\n")
        f.write("  • ASTM C143 (Slump)\n")
        f.write("  • ASTM C231 (Air Content)\n")
        f.write("  • ASTM C138 (Unit Weight)\n")
        f.write("  • ASTM C403 (Setting Time)\n")
        f.write("  • ASTM D5603 (Rubber Properties)\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Summary statistics created: {stats_file}")
    return stats_file

def main():
    """Main execution function"""
    print("Loading all data...")
    data = load_all_data()
    
    print("\nExporting to Excel...")
    export_to_excel(data)
    
    print("\nExporting sieve analysis data...")
    export_aggregates_sieve_analysis(data)
    
    print("\nExporting chemical analysis data...")
    export_chemical_analysis(data)
    
    print("\nCreating summary statistics...")
    create_summary_statistics()
    
    print(f"\n✅ All exports complete! Files saved to: {EXPORT_DIR}")
    print("\nExported files:")
    for file in EXPORT_DIR.iterdir():
        print(f"  • {file.name} ({file.stat().st_size / 1024:.1f} KB)")

if __name__ == "__main__":
    main()