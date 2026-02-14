#!/usr/bin/env python3
"""
Generate Abaqus .inp file from material database CSV

This script reads the master material database and generates:
1. *MATERIAL blocks for UMAT (Elastic, Expansion, Depvar)
2. *UEL PROPERTY blocks for interface cohesive zones
3. *INITIAL CONDITIONS for temperature field
4. *AMPLITUDE for thermal loading

Usage:
    python generate_abaqus_input.py --input data/master_material_database.csv --output sofc_model.inp

Author: Generated for SOFC Fracture Dataset
Date: February 13, 2026
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


class AbaqusInputGenerator:
    """Generate Abaqus input file from material property database"""
    
    def __init__(self, csv_path):
        """
        Initialize generator with CSV database
        
        Parameters:
        -----------
        csv_path : str or Path
            Path to master_material_database.csv
        """
        self.df = pd.read_csv(csv_path)
        self.materials = {}
        self.interfaces = {}
        self._parse_database()
        
    def _parse_database(self):
        """Parse CSV database into material and interface dictionaries"""
        # Extract elastic properties at RT and 800C
        elastic_data = self.df[self.df['Category'] == 'Elastic'].copy()
        
        for material in ['8YSZ', 'GDC10', 'LSCF']:
            mat_name = material.replace('8YSZ', 'YSZ')
            self.materials[mat_name] = {
                'E_RT': self._get_value(elastic_data, f'E_{material}', 25),
                'E_800': self._get_value(elastic_data, f'E_{material}', 800),
                'nu_RT': self._get_value(elastic_data, f'nu_{material}', 25),
                'nu_800': self._get_value(elastic_data, f'nu_{material}', 800),
                'alpha': self._get_value_by_category('Thermal', f'alpha_{material}'),
            }
        
        # Extract chemical expansion
        chem_data = self.df[self.df['Category'] == 'Chemical'].copy()
        self.materials['GDC']['beta_iso'] = self._get_value_by_symbol(chem_data, 'beta_iso_GDC')
        self.materials['LSCF']['beta_11'] = self._get_value_by_symbol(chem_data, 'beta_11_LSCF')
        self.materials['LSCF']['beta_33'] = self._get_value_by_symbol(chem_data, 'beta_33_LSCF')
        self.materials['GDC']['delta_delta'] = self._get_value_by_symbol(chem_data, 'delta_delta_GDC')
        self.materials['LSCF']['delta_delta'] = self._get_value_by_symbol(chem_data, 'delta_delta_LSCF')
        
        # Extract fracture properties
        frac_data = self.df[self.df['Category'] == 'Fracture'].copy()
        for mat in ['YSZ', 'GDC', 'LSCF']:
            self.materials[mat]['Gc_b'] = self._get_value_by_symbol(frac_data, f'Gc_b_{mat}')
        
        # Extract interface properties
        cohesive_data = self.df[self.df['Category'] == 'Cohesive'].copy()
        for interface in ['YSZ_GDC', 'GDC_LSCF']:
            iface_name = interface.replace('_', '/')
            self.interfaces[iface_name] = {
                'Gc_I': self._get_value_by_symbol(frac_data, f'Gc_I_{interface}'),
                'Gc_II': self._get_value_by_symbol(frac_data, f'Gc_II_{interface}'),
                'T_max_n': self._get_value_by_symbol(cohesive_data, f'T_max_n_{interface}'),
                'T_max_t': self._get_value_by_symbol(cohesive_data, f'T_max_t_{interface}'),
                'eta_BK': 2.1,
            }
    
    def _get_value(self, df, symbol, temp):
        """Get value from dataframe filtered by symbol and temperature"""
        try:
            result = df[(df['Symbol'] == symbol) & (df['Temperature_C'] == temp)]['Value'].values
            return float(result[0]) if len(result) > 0 else 0.0
        except:
            return 0.0
    
    def _get_value_by_category(self, category, symbol):
        """Get value by category and symbol"""
        try:
            result = self.df[(self.df['Category'] == category) & 
                           (self.df['Symbol'].str.contains(symbol, na=False))]['Value'].values
            return float(result[0]) if len(result) > 0 else 0.0
        except:
            return 0.0
    
    def _get_value_by_symbol(self, df, symbol):
        """Get value by symbol from filtered dataframe"""
        try:
            result = df[df['Symbol'] == symbol]['Value'].values
            return float(result[0]) if len(result) > 0 else 0.0
        except:
            return 0.0
    
    def generate_material_block(self, material_name, output_file):
        """
        Generate *MATERIAL block for UMAT
        
        Parameters:
        -----------
        material_name : str
            Material name (YSZ, GDC, LSCF)
        output_file : file object
            Output file handle
        """
        mat = self.materials[material_name]
        
        # Material header
        output_file.write(f"*MATERIAL, NAME={material_name}\n")
        
        # Elastic properties at RT
        E_RT = mat['E_RT'] * 1e3  # Convert GPa to MPa
        nu_RT = mat['nu_RT']
        output_file.write(f"*ELASTIC, TYPE=ISOTROPIC\n")
        output_file.write(f"{E_RT:.6E}, {nu_RT:.6f}, 25.0\n")
        
        # Elastic properties at 800C
        E_800 = mat['E_800'] * 1e3
        nu_800 = mat['nu_800']
        output_file.write(f"{E_800:.6E}, {nu_800:.6f}, 800.0\n")
        
        # Thermal expansion
        alpha = mat['alpha'] * 1e-6  # Convert ppm/K to 1/K
        output_file.write(f"*EXPANSION, TYPE=ISO, ZERO=800.0\n")
        output_file.write(f"{alpha:.6E}\n")
        
        # Chemical expansion (if applicable)
        if material_name == 'GDC' and 'beta_iso' in mat:
            output_file.write(f"** Chemical Expansion Coefficient (Isotropic)\n")
            output_file.write(f"** beta_iso = {mat['beta_iso']:.6f} strain/delta_delta\n")
            output_file.write(f"** delta_delta = {mat.get('delta_delta', 0.0):.6f}\n")
            output_file.write(f"*USER DEFINED FIELD\n")
            
        elif material_name == 'LSCF' and 'beta_11' in mat:
            output_file.write(f"** Chemical Expansion Coefficient (Anisotropic)\n")
            output_file.write(f"** beta_11 = {mat['beta_11']:.6f}, beta_33 = {mat['beta_33']:.6f}\n")
            output_file.write(f"** delta_delta = {mat.get('delta_delta', 0.0):.6f}\n")
            output_file.write(f"*USER DEFINED FIELD\n")
        
        # Phase field fracture parameters
        Gc_b = mat.get('Gc_b', 0.0)
        output_file.write(f"** Bulk Fracture Toughness Gc = {Gc_b:.3f} J/m^2\n")
        output_file.write(f"*DEPVAR\n")
        output_file.write(f"10\n")  # 10 solution-dependent variables for phase field
        output_file.write(f"** SDV1: Phase field damage parameter d\n")
        output_file.write(f"** SDV2: Elastic strain energy density\n")
        output_file.write(f"** SDV3-8: Stress components\n")
        output_file.write(f"** SDV9: Chemical strain magnitude\n")
        output_file.write(f"** SDV10: History variable H\n")
        
        output_file.write("\n")
    
    def generate_cohesive_block(self, interface_name, output_file):
        """
        Generate cohesive zone properties for UEL
        
        Parameters:
        -----------
        interface_name : str
            Interface name (YSZ/GDC or GDC/LSCF)
        output_file : file object
            Output file handle
        """
        iface = self.interfaces[interface_name]
        
        output_file.write(f"** Cohesive Interface: {interface_name}\n")
        output_file.write(f"*UEL PROPERTY, ELSET={interface_name.replace('/', '_')}_COHESIVE\n")
        
        # Write cohesive parameters
        # Format: Gc_I, Gc_II, T_max_n, T_max_t, eta_BK, K_penalty
        Gc_I = iface['Gc_I']
        Gc_II = iface['Gc_II']
        T_max_n = iface['T_max_n']
        T_max_t = iface['T_max_t']
        eta_BK = iface['eta_BK']
        K_penalty = 1.0e6  # Large penalty stiffness
        
        output_file.write(f"{Gc_I:.6E}, {Gc_II:.6E}, {T_max_n:.6E}, {T_max_t:.6E}, {eta_BK:.3f}, {K_penalty:.6E}\n")
        output_file.write(f"** Gc_I [J/m^2], Gc_II [J/m^2], T_max_n [MPa], T_max_t [MPa], eta_BK, K_penalty [MPa/mm]\n")
        output_file.write(f"** Mixed-mode criterion: Gc = Gc_I + (Gc_II - Gc_I) * (mode_II_ratio)^eta_BK\n")
        output_file.write("\n")
    
    def generate_thermal_amplitude(self, output_file):
        """
        Generate thermal loading amplitude (800C to RT)
        
        Parameters:
        -----------
        output_file : file object
            Output file handle
        """
        output_file.write("** Thermal Loading Amplitude: Cooldown from 800C to 25C\n")
        output_file.write("*AMPLITUDE, NAME=COOLDOWN, DEFINITION=TABULAR\n")
        
        # Cooling schedule: 5 hour cooldown
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        temps = [800.0, 650.0, 500.0, 300.0, 100.0, 25.0]
        
        for t, T in zip(times, temps):
            output_file.write(f"{t:.3f}, {T:.3f}\n")
        
        output_file.write("\n")
    
    def generate_full_input(self, output_path):
        """
        Generate complete Abaqus input file
        
        Parameters:
        -----------
        output_path : str or Path
            Path to output .inp file
        """
        with open(output_path, 'w') as f:
            # Header
            f.write("**\n")
            f.write("** Abaqus Input File for SOFC Tri-Layer Fracture Simulation\n")
            f.write("** Generated from master_material_database.csv\n")
            f.write("** Date: February 13, 2026\n")
            f.write("**\n")
            f.write("** System: YSZ (10um) / GDC (5um) / LSCF (30um)\n")
            f.write("** Physics: Thermal + Chemical Expansion + Phase Field + Cohesive Zones\n")
            f.write("**\n\n")
            
            # Generate material blocks
            f.write("** ========================================\n")
            f.write("** MATERIAL DEFINITIONS\n")
            f.write("** ========================================\n\n")
            
            for material in ['YSZ', 'GDC', 'LSCF']:
                self.generate_material_block(material, f)
            
            # Generate cohesive zone properties
            f.write("** ========================================\n")
            f.write("** INTERFACE COHESIVE PROPERTIES\n")
            f.write("** ========================================\n\n")
            
            for interface in ['YSZ/GDC', 'GDC/LSCF']:
                self.generate_cohesive_block(interface, f)
            
            # Generate thermal amplitude
            f.write("** ========================================\n")
            f.write("** LOADING AMPLITUDES\n")
            f.write("** ========================================\n\n")
            
            self.generate_thermal_amplitude(f)
            
            # Add template for boundary conditions
            f.write("** ========================================\n")
            f.write("** BOUNDARY CONDITIONS (Template)\n")
            f.write("** ========================================\n\n")
            f.write("** *BOUNDARY\n")
            f.write("** BOTTOM_EDGE, 1, 2, 0.0  ! Fix displacement\n")
            f.write("** LEFT_EDGE, 1, 1, 0.0    ! Symmetry\n\n")
            
            # Add template for initial conditions
            f.write("** ========================================\n")
            f.write("** INITIAL CONDITIONS\n")
            f.write("** ========================================\n\n")
            f.write("** *INITIAL CONDITIONS, TYPE=TEMPERATURE\n")
            f.write("** ALL_NODES, 800.0\n\n")
            
            # Add step definition template
            f.write("** ========================================\n")
            f.write("** STEP: Thermal Cooldown\n")
            f.write("** ========================================\n\n")
            f.write("** *STEP, NAME=COOLDOWN, NLGEOM=YES, INC=1000\n")
            f.write("** *STATIC\n")
            f.write("** 0.01, 5.0, 1E-6, 0.1  ! Initial, total, min, max increment\n")
            f.write("** *TEMPERATURE, AMPLITUDE=COOLDOWN\n")
            f.write("** ALL_NODES, 800.0\n")
            f.write("** *OUTPUT, FIELD, FREQUENCY=10\n")
            f.write("** *ELEMENT OUTPUT\n")
            f.write("** S, E, SDV\n")
            f.write("** *NODE OUTPUT\n")
            f.write("** U, RF\n")
            f.write("** *END STEP\n\n")
            
        print(f"✓ Generated Abaqus input file: {output_path}")
        print(f"  - {len(self.materials)} materials defined")
        print(f"  - {len(self.interfaces)} interfaces defined")
        print(f"  - Thermal cooldown: 800°C → 25°C")
    
    def summary(self):
        """Print summary of material properties"""
        print("\n" + "="*60)
        print("MATERIAL PROPERTY SUMMARY")
        print("="*60)
        
        for mat_name, mat_props in self.materials.items():
            print(f"\n{mat_name}:")
            print(f"  E (RT): {mat_props['E_RT']:.1f} GPa")
            print(f"  E (800°C): {mat_props['E_800']:.1f} GPa")
            print(f"  ν: {mat_props['nu_RT']:.3f}")
            print(f"  α: {mat_props['alpha']:.2f} ppm/K")
            
            if 'beta_iso' in mat_props:
                print(f"  β_iso: {mat_props['beta_iso']:.4f} strain/Δδ")
                print(f"  Δδ: {mat_props.get('delta_delta', 0):.3f}")
            elif 'beta_11' in mat_props:
                print(f"  β₁₁: {mat_props['beta_11']:.4f}, β₃₃: {mat_props['beta_33']:.4f}")
                print(f"  Δδ: {mat_props.get('delta_delta', 0):.3f}")
            
            if 'Gc_b' in mat_props:
                print(f"  Gc,bulk: {mat_props['Gc_b']:.1f} J/m²")
        
        print("\n" + "-"*60)
        print("INTERFACE PROPERTIES")
        print("-"*60)
        
        for iface_name, iface_props in self.interfaces.items():
            print(f"\n{iface_name}:")
            print(f"  Gc,I: {iface_props['Gc_I']:.1f} J/m²")
            print(f"  Gc,II: {iface_props['Gc_II']:.1f} J/m²")
            print(f"  T_max,n: {iface_props['T_max_n']:.0f} MPa")
            print(f"  T_max,t: {iface_props['T_max_t']:.0f} MPa")
            print(f"  η_BK: {iface_props['eta_BK']:.2f}")
            print(f"  Gc,II/Gc,I: {iface_props['Gc_II']/iface_props['Gc_I']:.2f}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate Abaqus input file from material database CSV'
    )
    parser.add_argument('--input', '-i', 
                       default='data/master_material_database.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', '-o',
                       default='sofc_model.inp',
                       help='Output Abaqus .inp file path')
    parser.add_argument('--summary', '-s',
                       action='store_true',
                       help='Print material property summary')
    
    args = parser.parse_args()
    
    # Check if input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1
    
    # Generate input file
    print(f"Reading material database: {input_path}")
    generator = AbaqusInputGenerator(input_path)
    
    if args.summary:
        generator.summary()
    
    print(f"\nGenerating Abaqus input file...")
    generator.generate_full_input(args.output)
    
    print("\n✓ Success! Ready to run in Abaqus/Standard with custom UMAT/UEL")
    print("\nNext steps:")
    print("  1. Review generated .inp file")
    print("  2. Add mesh geometry and element definitions")
    print("  3. Compile and link UMAT/UEL subroutines")
    print("  4. Run: abaqus job=sofc_model user=umat_uel.f cpus=8")
    
    return 0


if __name__ == '__main__':
    exit(main())
