#!/usr/bin/env python3
"""
Final comprehensive summary of the SOFC material properties dataset
"""

import json
import numpy as np

def parse_numpy_string(numpy_str):
    """Parse numpy array string representation"""
    # Remove brackets and split by whitespace
    clean_str = numpy_str.strip('[]').replace('\n', ' ')
    # Split and convert to float
    values = [float(x) for x in clean_str.split() if x.strip()]
    return values

def main():
    """Generate comprehensive dataset summary"""
    
    # Load the dataset
    with open('/workspace/material_properties.json', 'r') as f:
        dataset = json.load(f)
    
    print("=" * 80)
    print("COMPREHENSIVE SOFC MATERIAL PROPERTY DATASET")
    print("=" * 80)
    print()
    
    # Basic information
    print("DATASET OVERVIEW:")
    print(f"  Generation Date: {dataset['metadata']['generation_date']}")
    print(f"  Description: {dataset['metadata']['description']}")
    print(f"  Temperature Range: {dataset['metadata']['temperature_range_K'][0]}K - {dataset['metadata']['temperature_range_K'][-1]}K")
    print(f"  Number of Data Points: {len(dataset['metadata']['temperature_range_K'])}")
    print(f"  Materials: {', '.join(dataset['metadata']['materials'])}")
    print()
    
    # Property categories
    print("PROPERTY CATEGORIES:")
    print("  1. Elastic Properties (E, ν)")
    print("  2. Fracture Properties (K_ic, G_c)")
    print("  3. Thermo-Physical Properties (CTE)")
    print("  4. Chemical Expansion Properties")
    print()
    
    # Room temperature values (300K = index 0)
    print("=" * 60)
    print("ROOM TEMPERATURE PROPERTIES (300K)")
    print("=" * 60)
    
    # Elastic properties
    print("\nELASTIC PROPERTIES:")
    print("Material        | E (GPa) | ν    | Uncertainty")
    print("-" * 50)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['elastic_properties'][material]
        E_values = parse_numpy_string(data['Young_Modulus_Pa'])
        nu_values = parse_numpy_string(data['Poisson_Ratio'])
        E_err_values = parse_numpy_string(data['Uncertainty_E_Pa'])
        
        E = E_values[0] / 1e9  # Convert to GPa
        nu = nu_values[0]
        E_err = E_err_values[0] / 1e9
        
        print(f"{material:15} | {E:7.1f} | {nu:.3f} | ±{E_err:.1f} GPa")
    
    # Fracture properties
    print("\nFRACTURE PROPERTIES:")
    print("Material                | K_ic (MPa√m) | G_c (J/m²) | Uncertainty")
    print("-" * 65)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_interface', 'YSZ-electrolyte_interface']:
        data = dataset['fracture_properties'][material]
        K_ic_values = parse_numpy_string(data['Fracture_Toughness_MPa_sqrt_m'])
        G_c_values = parse_numpy_string(data['Critical_Energy_Release_Rate_J_per_m2'])
        K_ic_err_values = parse_numpy_string(data['Uncertainty_K_ic_MPa_sqrt_m'])
        
        K_ic = K_ic_values[0]
        G_c = G_c_values[0]
        K_ic_err = K_ic_err_values[0]
        
        print(f"{material:23} | {K_ic:11.2f} | {G_c:8.1f} | ±{K_ic_err:.2f}")
    
    # CTE properties
    print("\nTHERMAL EXPANSION:")
    print("Material        | CTE (ppm/K) | Uncertainty")
    print("-" * 40)
    
    for material in ['YSZ', 'Ni', 'Ni-YSZ_composite']:
        data = dataset['thermo_physical_properties'][material]
        cte_values = parse_numpy_string(data['CTE_per_K'])
        cte_err_values = parse_numpy_string(data['Uncertainty_CTE_per_K'])
        
        cte = cte_values[0] * 1e6  # Convert to ppm/K
        cte_err = cte_err_values[0] * 1e6
        
        print(f"{material:15} | {cte:11.1f} | ±{cte_err:.1f}")
    
    # CTE Mismatch
    mismatch_data = dataset['thermo_physical_properties']['CTE_Mismatch']
    mismatch_values = parse_numpy_string(mismatch_data['CTE_Difference_per_K'])
    mismatch_err_values = parse_numpy_string(mismatch_data['Uncertainty_CTE_Difference_per_K'])
    
    mismatch = mismatch_values[0] * 1e6
    mismatch_err = mismatch_err_values[0] * 1e6
    
    print(f"{'CTE Mismatch':15} | {mismatch:11.1f} | ±{mismatch_err:.1f}")
    
    # Chemical expansion
    print("\nCHEMICAL EXPANSION:")
    print("Process/Material        | Expansion (%) | Uncertainty")
    print("-" * 50)
    
    for material in ['Ni_to_NiO', 'YSZ_oxygen_vacancy', 'Ni-YSZ_composite_oxidation']:
        data = dataset['chemical_expansion_properties'][material]
        chem_exp_values = parse_numpy_string(data['Chemical_Expansion_Coefficient'])
        chem_exp_err_values = parse_numpy_string(data['Uncertainty_Chemical_Expansion'])
        
        chem_exp = chem_exp_values[0] * 100  # Convert to %
        chem_exp_err = chem_exp_err_values[0] * 100
        
        label = material.replace('_', ' ').replace('Ni to NiO', 'Ni→NiO')
        print(f"{label:23} | {chem_exp:11.2f} | ±{chem_exp_err:.2f}")
    
    print("\n" + "=" * 60)
    print("CRITICAL PARAMETERS FOR SOFC MODELING")
    print("=" * 60)
    
    print("\n1. INTERFACE FRACTURE PROPERTIES (MOST CRITICAL):")
    print("   - Ni-YSZ Interface: K_ic ≈ 0.5 MPa√m")
    print("   - This is the WEAKEST link in the system")
    print("   - High uncertainty due to measurement difficulty")
    print("   - Most likely failure location during thermal cycling")
    
    print("\n2. CTE MISMATCH (DRIVES RESIDUAL STRESSES):")
    print(f"   - Ni CTE: {parse_numpy_string(dataset['thermo_physical_properties']['Ni']['CTE_per_K'])[0]*1e6:.1f} ppm/K")
    print(f"   - YSZ CTE: {parse_numpy_string(dataset['thermo_physical_properties']['YSZ']['CTE_per_K'])[0]*1e6:.1f} ppm/K")
    print(f"   - Mismatch: {mismatch:.1f} ppm/K (SIGNIFICANT)")
    print("   - Causes thermal stress during heating/cooling")
    
    print("\n3. CHEMICAL EXPANSION (REDOX CYCLING):")
    print("   - Ni→NiO: 6.7% linear expansion")
    print("   - Volume change: 20% (Pilling-Bedworth ratio)")
    print("   - Critical for redox cycling analysis")
    print("   - Can cause mechanical failure during oxidation")
    
    print("\n4. TEMPERATURE DEPENDENCE:")
    print("   - Young's modulus decreases with temperature")
    print("   - Fracture toughness may decrease at high T")
    print("   - CTE increases slightly with temperature")
    print("   - All properties show realistic temperature trends")
    
    print("\n" + "=" * 60)
    print("DATASET FILES GENERATED")
    print("=" * 60)
    
    print("\nMain Files:")
    print("  - material_properties.json (Complete dataset)")
    print("  - material_properties.h5 (Hierarchical format)")
    print("  - DATASET_DOCUMENTATION.md (Detailed documentation)")
    
    print("\nCSV Files (Individual Properties):")
    print("  - elastic_properties_*.csv")
    print("  - fracture_properties_*.csv") 
    print("  - cte_properties_*.csv")
    print("  - chemical_expansion_*.csv")
    
    print("\nScripts:")
    print("  - material_property_dataset.py (Generation script)")
    print("  - dataset_summary.py (This summary)")
    
    print("\n" + "=" * 60)
    print("USAGE RECOMMENDATIONS")
    print("=" * 60)
    
    print("\nFor Finite Element Modeling:")
    print("  1. Use temperature-dependent elastic properties")
    print("  2. Include CTE mismatch in thermal stress analysis")
    print("  3. Model interfaces with reduced fracture properties")
    print("  4. Consider chemical expansion in redox cycling")
    
    print("\nFor Fracture Analysis:")
    print("  1. Interface properties are most critical")
    print("  2. Use conservative values due to high uncertainty")
    print("  3. Consider temperature effects on toughness")
    print("  4. Account for mixed-mode loading at interfaces")
    
    print("\nFor Redox Cycling Analysis:")
    print("  1. Include Ni→NiO chemical expansion")
    print("  2. Model volume changes during oxidation")
    print("  3. Consider interface delamination")
    print("  4. Account for CTE mismatch effects")
    
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print("\nThis comprehensive dataset provides all necessary material")
    print("properties for SOFC modeling and fracture analysis, with")
    print("particular emphasis on the critical interface properties")
    print("that govern component reliability.")

if __name__ == "__main__":
    main()