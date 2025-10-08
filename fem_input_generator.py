#!/usr/bin/env python3
"""
FEM Input Generator for YSZ Material Properties

This script generates input files for various FEM software packages (ANSYS, Abaqus, COMSOL)
using the YSZ material properties dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class FEMInputGenerator:
    """Class to generate FEM input files for different software packages."""
    
    def __init__(self, csv_file='ysz_material_properties_dataset.csv'):
        """Initialize with YSZ properties dataset."""
        self.df = pd.read_csv(csv_file)
        self.material_name = "YSZ_8mol_percent"
        
    def generate_ansys_input(self, output_file='ysz_ansys_material.inp'):
        """Generate ANSYS APDL material property input."""
        
        with open(output_file, 'w') as f:
            f.write("! YSZ Material Properties for ANSYS\n")
            f.write("! Generated from temperature-dependent dataset\n")
            f.write("! Material: 8 mol% Yttria-Stabilized Zirconia\n\n")
            
            # Material definition
            f.write(f"MP,EX,1,{self.df.iloc[0]['Youngs_Modulus_GPa']*1e9}  ! Young's Modulus at RT (Pa)\n")
            f.write(f"MP,PRXY,1,{self.df.iloc[0]['Poissons_Ratio']}  ! Poisson's Ratio at RT\n")
            f.write(f"MP,DENS,1,{self.df.iloc[0]['Density_g_cm3']*1000}  ! Density at RT (kg/m³)\n\n")
            
            # Temperature-dependent Young's Modulus
            f.write("! Temperature-dependent Young's Modulus\n")
            f.write("MPTEMP\n")
            f.write("MPTEMP,1")
            for i, temp in enumerate(self.df['Temperature_K']):
                if i % 6 == 0 and i > 0:
                    f.write("\nMPTEMP," + str(i//6 + 1))
                f.write(f",{temp}")
            f.write("\n\nMPDATA,EX,1,1")
            for i, E in enumerate(self.df['Youngs_Modulus_GPa']):
                if i % 6 == 0 and i > 0:
                    f.write("\nMPDATA,EX,1," + str(i//6 + 1))
                f.write(f",{E*1e9}")
            f.write("\n\n")
            
            # Temperature-dependent CTE
            f.write("! Temperature-dependent Coefficient of Thermal Expansion\n")
            f.write("MPDATA,ALPX,1,1")
            for i, cte in enumerate(self.df['CTE_1e6_per_K']):
                if i % 6 == 0 and i > 0:
                    f.write("\nMPDATA,ALPX,1," + str(i//6 + 1))
                f.write(f",{cte*1e-6}")
            f.write("\n\n")
            
            # Temperature-dependent Thermal Conductivity
            f.write("! Temperature-dependent Thermal Conductivity\n")
            f.write("MPDATA,KXX,1,1")
            for i, k in enumerate(self.df['Thermal_Conductivity_W_mK']):
                if i % 6 == 0 and i > 0:
                    f.write("\nMPDATA,KXX,1," + str(i//6 + 1))
                f.write(f",{k}")
            f.write("\n\n")
            
            # Creep properties (Norton Law)
            f.write("! Creep Properties (Norton Power Law)\n")
            f.write("! Rate equation: strain_rate = A * stress^n * exp(-Q/RT)\n")
            f.write(f"TB,CREEP,1,,,NORTON\n")
            f.write(f"TBDATA,1,{self.df.iloc[-1]['Creep_A_1_Pa_s']}  ! A parameter at high temp\n")
            f.write(f"TBDATA,2,{self.df.iloc[-1]['Creep_n']}  ! n parameter\n")
            f.write(f"TBDATA,3,{self.df.iloc[-1]['Creep_Q_kJ_mol']*1000}  ! Q parameter (J/mol)\n\n")
            
        print(f"ANSYS input file generated: {output_file}")
    
    def generate_abaqus_input(self, output_file='ysz_abaqus_material.inp'):
        """Generate Abaqus material property input."""
        
        with open(output_file, 'w') as f:
            f.write("** YSZ Material Properties for Abaqus\n")
            f.write("** Generated from temperature-dependent dataset\n")
            f.write("** Material: 8 mol% Yttria-Stabilized Zirconia\n")
            f.write("**\n")
            
            # Material definition
            f.write(f"*Material, name={self.material_name}\n")
            
            # Elastic properties
            f.write("*Elastic, type=ISOTROPIC\n")
            for _, row in self.df.iterrows():
                E = row['Youngs_Modulus_GPa'] * 1e9  # Convert to Pa
                nu = row['Poissons_Ratio']
                T = row['Temperature_K']
                f.write(f"{E:.3e}, {nu:.4f}, {T:.1f}\n")
            
            # Density
            f.write("*Density\n")
            for _, row in self.df.iterrows():
                rho = row['Density_g_cm3'] * 1000  # Convert to kg/m³
                T = row['Temperature_K']
                f.write(f"{rho:.1f}, {T:.1f}\n")
            
            # Thermal expansion
            f.write("*Expansion, type=ISOTROPIC\n")
            for _, row in self.df.iterrows():
                cte = row['CTE_1e6_per_K'] * 1e-6
                T = row['Temperature_K']
                f.write(f"{cte:.6e}, {T:.1f}\n")
            
            # Thermal conductivity
            f.write("*Conductivity\n")
            for _, row in self.df.iterrows():
                k = row['Thermal_Conductivity_W_mK']
                T = row['Temperature_K']
                f.write(f"{k:.4f}, {T:.1f}\n")
            
            # Specific heat (estimated)
            f.write("*Specific Heat\n")
            for _, row in self.df.iterrows():
                cp = 500  # Typical value for ceramics (J/kg·K)
                T = row['Temperature_K']
                f.write(f"{cp}, {T:.1f}\n")
            
            # Creep properties
            f.write("*Creep, law=POWER\n")
            for _, row in self.df.iterrows():
                A = row['Creep_A_1_Pa_s']
                n = row['Creep_n']
                T = row['Temperature_K']
                f.write(f"{A:.6e}, {n:.1f}, 0.0, {T:.1f}\n")
            
        print(f"Abaqus input file generated: {output_file}")
    
    def generate_comsol_input(self, output_file='ysz_comsol_material.txt'):
        """Generate COMSOL material property input (text format)."""
        
        with open(output_file, 'w') as f:
            f.write("% YSZ Material Properties for COMSOL Multiphysics\n")
            f.write("% Generated from temperature-dependent dataset\n")
            f.write("% Material: 8 mol% Yttria-Stabilized Zirconia\n")
            f.write("%\n")
            f.write("% Usage: Import this data into COMSOL material properties\n")
            f.write("% Use interpolation functions for temperature dependency\n\n")
            
            # Temperature array
            f.write("% Temperature array (K)\n")
            f.write("T = [")
            f.write(", ".join([f"{T:.1f}" for T in self.df['Temperature_K']]))
            f.write("];\n\n")
            
            # Young's Modulus
            f.write("% Young's Modulus (Pa)\n")
            f.write("E = [")
            f.write(", ".join([f"{E*1e9:.3e}" for E in self.df['Youngs_Modulus_GPa']]))
            f.write("];\n\n")
            
            # Poisson's Ratio
            f.write("% Poisson's Ratio\n")
            f.write("nu = [")
            f.write(", ".join([f"{nu:.4f}" for nu in self.df['Poissons_Ratio']]))
            f.write("];\n\n")
            
            # Density
            f.write("% Density (kg/m³)\n")
            f.write("rho = [")
            f.write(", ".join([f"{rho*1000:.1f}" for rho in self.df['Density_g_cm3']]))
            f.write("];\n\n")
            
            # Thermal Expansion Coefficient
            f.write("% Coefficient of Thermal Expansion (1/K)\n")
            f.write("alpha = [")
            f.write(", ".join([f"{cte*1e-6:.6e}" for cte in self.df['CTE_1e6_per_K']]))
            f.write("];\n\n")
            
            # Thermal Conductivity
            f.write("% Thermal Conductivity (W/m·K)\n")
            f.write("k = [")
            f.write(", ".join([f"{k:.4f}" for k in self.df['Thermal_Conductivity_W_mK']]))
            f.write("];\n\n")
            
            # Heat Capacity
            f.write("% Heat Capacity (J/kg·K) - estimated\n")
            f.write("Cp = ")
            f.write(str([500] * len(self.df)))  # Constant estimate
            f.write(";\n\n")
            
            # Creep parameters
            f.write("% Creep Parameters (Norton Law)\n")
            f.write("% strain_rate = A * stress^n * exp(-Q/RT)\n")
            f.write("creep_A = [")
            f.write(", ".join([f"{A:.6e}" for A in self.df['Creep_A_1_Pa_s']]))
            f.write("];\n")
            f.write(f"creep_n = {self.df.iloc[0]['Creep_n']};  % Stress exponent\n")
            f.write(f"creep_Q = {self.df.iloc[0]['Creep_Q_kJ_mol']*1000};  % Activation energy (J/mol)\n\n")
            
            # Fracture properties
            f.write("% Fracture Toughness (Pa·m^0.5)\n")
            f.write("K_IC = [")
            f.write(", ".join([f"{K*1e6:.3e}" for K in self.df['Fracture_Toughness_MPa_sqrt_m']]))
            f.write("];\n\n")
            
        print(f"COMSOL input file generated: {output_file}")
    
    def generate_material_card(self, output_file='ysz_material_card.txt'):
        """Generate a general material property card."""
        
        with open(output_file, 'w') as f:
            f.write("YSZ MATERIAL PROPERTY CARD\n")
            f.write("=" * 50 + "\n")
            f.write("Material: 8 mol% Yttria-Stabilized Zirconia\n")
            f.write("Application: SOFC Electrolyte\n")
            f.write("Temperature Range: 25°C to 1500°C\n\n")
            
            # Room temperature properties
            rt_data = self.df.iloc[0]
            f.write("ROOM TEMPERATURE PROPERTIES (25°C):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Young's Modulus:           {rt_data['Youngs_Modulus_GPa']:.1f} GPa\n")
            f.write(f"Poisson's Ratio:           {rt_data['Poissons_Ratio']:.3f}\n")
            f.write(f"Density:                   {rt_data['Density_g_cm3']:.2f} g/cm³\n")
            f.write(f"Thermal Expansion Coeff:   {rt_data['CTE_1e6_per_K']:.1f} × 10⁻⁶/K\n")
            f.write(f"Thermal Conductivity:      {rt_data['Thermal_Conductivity_W_mK']:.1f} W/m·K\n")
            f.write(f"Fracture Toughness:        {rt_data['Fracture_Toughness_MPa_sqrt_m']:.1f} MPa√m\n")
            f.write(f"Weibull Modulus:           {rt_data['Weibull_Modulus']:.1f}\n")
            f.write(f"Characteristic Strength:   {rt_data['Characteristic_Strength_MPa']:.0f} MPa\n\n")
            
            # High temperature properties
            ht_data = self.df.iloc[-1]
            f.write("HIGH TEMPERATURE PROPERTIES (1500°C):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Young's Modulus:           {ht_data['Youngs_Modulus_GPa']:.1f} GPa\n")
            f.write(f"Poisson's Ratio:           {ht_data['Poissons_Ratio']:.3f}\n")
            f.write(f"Density:                   {ht_data['Density_g_cm3']:.2f} g/cm³\n")
            f.write(f"Thermal Expansion Coeff:   {ht_data['CTE_1e6_per_K']:.1f} × 10⁻⁶/K\n")
            f.write(f"Thermal Conductivity:      {ht_data['Thermal_Conductivity_W_mK']:.2f} W/m·K\n")
            f.write(f"Fracture Toughness:        {ht_data['Fracture_Toughness_MPa_sqrt_m']:.1f} MPa√m\n")
            f.write(f"Creep Parameter A:         {ht_data['Creep_A_1_Pa_s']:.2e} Pa⁻¹s⁻¹\n\n")
            
            # Critical notes
            f.write("CRITICAL MODELING NOTES:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Young's Modulus decreases ~50% from RT to 1500°C\n")
            f.write("2. CTE increases ~56% over temperature range\n")
            f.write("3. Thermal conductivity drops dramatically at high T\n")
            f.write("4. Creep becomes significant above 800°C\n")
            f.write("5. Fracture toughness degrades with temperature\n")
            f.write("6. Use temperature-dependent properties for accuracy\n\n")
            
            # Usage recommendations
            f.write("FEM IMPLEMENTATION RECOMMENDATIONS:\n")
            f.write("-" * 40 + "\n")
            f.write("• Use coupled thermo-mechanical analysis\n")
            f.write("• Include temperature-dependent material properties\n")
            f.write("• Consider creep effects above 800°C\n")
            f.write("• Use Weibull statistics for failure analysis\n")
            f.write("• Validate with experimental data when possible\n")
            
        print(f"Material card generated: {output_file}")

def main():
    """Main function to generate all FEM input files."""
    
    print("YSZ FEM Input Generator")
    print("=" * 30)
    
    # Initialize generator
    generator = FEMInputGenerator()
    
    # Generate all input files
    generator.generate_ansys_input()
    generator.generate_abaqus_input()
    generator.generate_comsol_input()
    generator.generate_material_card()
    
    print("\nAll FEM input files generated successfully!")
    print("\nGenerated files:")
    print("- ysz_ansys_material.inp (ANSYS APDL)")
    print("- ysz_abaqus_material.inp (Abaqus)")
    print("- ysz_comsol_material.txt (COMSOL)")
    print("- ysz_material_card.txt (General reference)")

if __name__ == "__main__":
    main()