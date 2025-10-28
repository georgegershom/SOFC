"""
Example Usage of SOFC Material Properties Dataset
================================================

This script demonstrates various ways to use the SOFC material properties dataset
for different applications including FEA, CFD, and electrochemical modeling.
"""

import pandas as pd
import json
import numpy as np
from sofc_material_properties import SOFCDatasetGenerator

def load_and_explore_data():
    """Load and explore the dataset"""
    print("=" * 60)
    print("SOFC MATERIAL PROPERTIES DATASET EXPLORATION")
    print("=" * 60)
    
    # Load CSV data
    df = pd.read_csv('sofc_materials_dataset.csv')
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of materials: {len(df)}")
    print(f"Number of properties: {len(df.columns)}")
    
    print("\nMaterial overview:")
    print(df[['Material_ID', 'Name', 'Composition']].to_string(index=False))
    
    return df

def compare_thermal_properties(df):
    """Compare thermal properties across materials"""
    print("\n" + "=" * 60)
    print("THERMAL PROPERTIES COMPARISON")
    print("=" * 60)
    
    thermal_props = df[['Name', 'TEC_1/K', 'Thermal_Conductivity_W/mK', 'Specific_Heat_J/kgK']]
    
    print("\nThermal Expansion Coefficients (×10⁻⁶ K⁻¹):")
    for _, row in thermal_props.iterrows():
        tec_micro = row['TEC_1/K'] * 1e6
        print(f"  {row['Name']:<25}: {tec_micro:>6.1f}")
    
    print("\nThermal Conductivity (W/m·K):")
    for _, row in thermal_props.iterrows():
        print(f"  {row['Name']:<25}: {row['Thermal_Conductivity_W/mK']:>6.1f}")

def analyze_mechanical_properties(df):
    """Analyze mechanical properties for structural design"""
    print("\n" + "=" * 60)
    print("MECHANICAL PROPERTIES ANALYSIS")
    print("=" * 60)
    
    mech_props = df[['Name', 'Youngs_Modulus_GPa', 'Poissons_Ratio', 'Density_kg/m3']]
    
    print("\nYoung's Modulus (GPa) - Stiffness ranking:")
    sorted_by_modulus = mech_props.sort_values('Youngs_Modulus_GPa', ascending=False)
    for _, row in sorted_by_modulus.iterrows():
        print(f"  {row['Name']:<25}: {row['Youngs_Modulus_GPa']:>6.0f}")
    
    print("\nDensity (kg/m³):")
    for _, row in mech_props.iterrows():
        print(f"  {row['Name']:<25}: {row['Density_kg/m3']:>6.0f}")

def examine_porosity_effects(df):
    """Examine porosity and its effects on transport properties"""
    print("\n" + "=" * 60)
    print("POROSITY AND TRANSPORT PROPERTIES")
    print("=" * 60)
    
    transport_props = df[['Name', 'Porosity', 'Tortuosity', 'Ionic_Conductivity_S/m']]
    
    print("\nPorosity effects on transport:")
    for _, row in transport_props.iterrows():
        porosity_pct = row['Porosity'] * 100
        print(f"  {row['Name']:<25}:")
        print(f"    Porosity: {porosity_pct:>5.1f}%")
        print(f"    Tortuosity: {row['Tortuosity']:>5.1f}")
        print(f"    Ionic Conductivity: {row['Ionic_Conductivity_S/m']:>8.3f} S/m")
        print()

def electrochemical_performance_analysis(df):
    """Analyze electrochemical performance parameters"""
    print("\n" + "=" * 60)
    print("ELECTROCHEMICAL PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Focus on electrochemically active materials
    active_materials = df[df['Exchange_Current_Density_A/m2'] > 0]
    
    print("\nElectrochemically active materials:")
    for _, row in active_materials.iterrows():
        print(f"\n{row['Name']}:")
        print(f"  Exchange Current Density: {row['Exchange_Current_Density_A/m2']:>8.0f} A/m²")
        print(f"  Activation Overpotential: {row['Activation_Overpotential_V']:>8.3f} V")
        print(f"  Ionic Conductivity: {row['Ionic_Conductivity_S/m']:>8.3f} S/m")
        print(f"  Electronic Conductivity: {row['Electronic_Conductivity_S/m']:>8.1f} S/m")

def creep_analysis():
    """Analyze creep behavior for high-temperature operation"""
    print("\n" + "=" * 60)
    print("CREEP BEHAVIOR ANALYSIS")
    print("=" * 60)
    
    generator = SOFCDatasetGenerator()
    
    # Materials with creep data
    creep_materials = ['ni_ysz_anode', '8ysz_electrolyte', 'crofer22_interconnect']
    
    print("\nCreep parameters (Norton-Bailey model: ε̇ = B × σⁿ × exp(-Q/RT)):")
    
    for mat_id in creep_materials:
        material = generator.get_material(mat_id)
        if material.creep:
            print(f"\n{material.name}:")
            print(f"  B (pre-exponential): {material.creep.B:.2e} 1/Pa^n·s")
            print(f"  n (stress exponent): {material.creep.n:.1f}")
            print(f"  Q (activation energy): {material.creep.Q/1000:.0f} kJ/mol")
            
            # Calculate relative creep rates at 1073K, 50 MPa
            T = 1073  # K
            sigma = 50e6  # Pa
            R = 8.314  # J/mol·K
            
            creep_rate = material.creep.B * (sigma ** material.creep.n) * np.exp(-material.creep.Q / (R * T))
            print(f"  Creep rate at 1073K, 50 MPa: {creep_rate:.2e} 1/s")

def thermal_stress_compatibility():
    """Analyze thermal expansion compatibility"""
    print("\n" + "=" * 60)
    print("THERMAL EXPANSION COMPATIBILITY")
    print("=" * 60)
    
    df = pd.read_csv('sofc_materials_dataset.csv')
    
    # Calculate TEC differences (potential for thermal stress)
    tec_values = df[['Name', 'TEC_1/K']].copy()
    tec_values['TEC_micro'] = tec_values['TEC_1/K'] * 1e6
    
    print("\nThermal expansion coefficient matching:")
    print("(Smaller differences indicate better thermal compatibility)")
    
    # Compare anode-electrolyte compatibility
    ni_ysz_tec = tec_values[tec_values['Name'] == 'Ni-YSZ Anode']['TEC_micro'].iloc[0]
    ysz_tec = tec_values[tec_values['Name'] == '8YSZ Electrolyte']['TEC_micro'].iloc[0]
    
    print(f"\nAnode-Electrolyte compatibility:")
    print(f"  Ni-YSZ TEC: {ni_ysz_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  8YSZ TEC: {ysz_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  Difference: {abs(ni_ysz_tec - ysz_tec):.1f} ×10⁻⁶ K⁻¹")
    
    # Compare cathode-electrolyte compatibility
    lsm_tec = tec_values[tec_values['Name'] == 'LSM Cathode']['TEC_micro'].iloc[0]
    lscf_tec = tec_values[tec_values['Name'] == 'LSCF Cathode']['TEC_micro'].iloc[0]
    
    print(f"\nCathode-Electrolyte compatibility:")
    print(f"  LSM TEC: {lsm_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  8YSZ TEC: {ysz_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  Difference: {abs(lsm_tec - ysz_tec):.1f} ×10⁻⁶ K⁻¹")
    
    print(f"\n  LSCF TEC: {lscf_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  8YSZ TEC: {ysz_tec:.1f} ×10⁻⁶ K⁻¹")
    print(f"  Difference: {abs(lscf_tec - ysz_tec):.1f} ×10⁻⁶ K⁻¹")

def material_selection_guide():
    """Provide material selection guidance"""
    print("\n" + "=" * 60)
    print("MATERIAL SELECTION GUIDE")
    print("=" * 60)
    
    print("\n🔋 ANODE SELECTION:")
    print("  • Ni-YSZ: Standard choice")
    print("    - Good electronic/ionic conductivity")
    print("    - Moderate thermal expansion")
    print("    - 35% porosity for gas transport")
    print("    - Creep resistance at operating temperatures")
    
    print("\n⚡ ELECTROLYTE SELECTION:")
    print("  • 8YSZ: High-temperature operation (>800°C)")
    print("    - Excellent ionic conductivity at high T")
    print("    - Good mechanical strength")
    print("    - Low porosity (5%) for gas separation")
    print("  • CGO: Intermediate temperature operation (600-800°C)")
    print("    - Higher ionic conductivity at lower T")
    print("    - Some electronic conductivity (mixed conductor)")
    
    print("\n🔴 CATHODE SELECTION:")
    print("  • LSM: Traditional choice for high-temperature SOFCs")
    print("    - Good electronic conductivity")
    print("    - Thermal expansion close to YSZ")
    print("    - Requires TPB (triple phase boundary) for activity")
    print("  • LSCF: Advanced material for IT-SOFCs")
    print("    - Mixed ionic/electronic conductor")
    print("    - Higher catalytic activity")
    print("    - Higher thermal expansion (compatibility issues)")
    
    print("\n🔗 INTERCONNECT SELECTION:")
    print("  • Crofer22 APU: Metallic interconnect")
    print("    - High electronic conductivity")
    print("    - Good thermal expansion match")
    print("    - Oxidation resistance")
    print("    - Cost-effective for planar designs")

def export_for_simulation():
    """Export data in formats suitable for simulation software"""
    print("\n" + "=" * 60)
    print("EXPORTING DATA FOR SIMULATION SOFTWARE")
    print("=" * 60)
    
    generator = SOFCDatasetGenerator()
    
    # Export for ANSYS/COMSOL (material property tables)
    simulation_data = {}
    
    for mat_id, material in generator.materials.items():
        simulation_data[mat_id] = {
            'density': material.mechanical.density,
            'youngs_modulus': material.mechanical.youngs_modulus * 1e9,  # Convert to Pa
            'poissons_ratio': material.mechanical.poissons_ratio,
            'thermal_expansion': material.thermal.thermal_expansion_coefficient,
            'thermal_conductivity': material.thermal.thermal_conductivity,
            'specific_heat': material.thermal.specific_heat_capacity,
            'ionic_conductivity': material.electrochemical.ionic_conductivity,
            'electronic_conductivity': material.electrochemical.electronic_conductivity
        }
    
    # Save simulation-ready data
    with open('sofc_simulation_properties.json', 'w') as f:
        json.dump(simulation_data, f, indent=2)
    
    print("✅ Exported simulation-ready properties to 'sofc_simulation_properties.json'")
    
    # Create ANSYS material property format example
    ansys_format = """
! ANSYS APDL Material Properties for SOFC Materials
! Generated from SOFC Material Properties Dataset

! Ni-YSZ Anode
MP,DENS,1,6800          ! Density (kg/m³)
MP,EX,1,45e9            ! Young's Modulus (Pa)
MP,PRXY,1,0.31          ! Poisson's Ratio
MP,ALPX,1,12.5e-6       ! Thermal Expansion Coefficient (1/K)
MP,KXX,1,6.2            ! Thermal Conductivity (W/m·K)
MP,C,1,450              ! Specific Heat (J/kg·K)

! 8YSZ Electrolyte
MP,DENS,2,6100          ! Density (kg/m³)
MP,EX,2,200e9           ! Young's Modulus (Pa)
MP,PRXY,2,0.31          ! Poisson's Ratio
MP,ALPX,2,10.8e-6       ! Thermal Expansion Coefficient (1/K)
MP,KXX,2,2.16           ! Thermal Conductivity (W/m·K)
MP,C,2,470              ! Specific Heat (J/kg·K)
"""
    
    with open('ansys_material_properties.txt', 'w') as f:
        f.write(ansys_format)
    
    print("✅ Exported ANSYS APDL format to 'ansys_material_properties.txt'")

def main():
    """Main function to run all analyses"""
    print("🔩 SOFC MATERIAL PROPERTIES DATASET ANALYSIS")
    print("=" * 60)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Run various analyses
    compare_thermal_properties(df)
    analyze_mechanical_properties(df)
    examine_porosity_effects(df)
    electrochemical_performance_analysis(df)
    creep_analysis()
    thermal_stress_compatibility()
    material_selection_guide()
    export_for_simulation()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nFiles generated:")
    print("  • sofc_simulation_properties.json - Simulation-ready data")
    print("  • ansys_material_properties.txt - ANSYS APDL format")
    print("\nDataset files:")
    print("  • sofc_materials_dataset.csv - Complete dataset (CSV)")
    print("  • sofc_materials_dataset.json - Complete dataset (JSON)")
    print("  • sofc_material_properties.py - Python generator")

if __name__ == "__main__":
    main()