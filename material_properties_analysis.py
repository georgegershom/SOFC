#!/usr/bin/env python3
"""
Material Property Dataset Analysis and Visualization Tool
==========================================================

This script provides comprehensive analysis and visualization capabilities
for the SOFC material property database.

Features:
- Load and parse JSON material property dataset
- Temperature-dependent property interpolation
- Comparative analysis across materials
- Export to various formats (CSV, Excel, LaTeX tables)
- Visualization of property trends
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.interpolate import interp1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Interpolation features limited.")


class MaterialPropertyDatabase:
    """Class for managing and analyzing SOFC material properties."""
    
    def __init__(self, json_path: str):
        """Initialize database from JSON file."""
        self.json_path = Path(json_path)
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        
        self.materials = self.data['materials']
        self.metadata = self.data['dataset_metadata']
        self.references = self.data['references']
        
    def get_elastic_properties(self, material: str, temperature: float) -> Dict[str, float]:
        """
        Get elastic properties at specified temperature with interpolation.
        
        Args:
            material: Material name (e.g., '8YSZ_electrolyte')
            temperature: Temperature in °C
            
        Returns:
            Dictionary with E, nu, G, K values
        """
        mat_data = self.materials[material]['elastic_properties']
        
        # Extract Young's modulus
        E_data = mat_data['youngs_modulus']['values']
        temps_E = [d['temperature'] for d in E_data]
        values_E = [d['value'] for d in E_data]
        
        if SCIPY_AVAILABLE and len(temps_E) > 1:
            E_interp = interp1d(temps_E, values_E, kind='linear', fill_value='extrapolate')
            E = float(E_interp(temperature))
        else:
            # Simple nearest neighbor
            idx = np.argmin(np.abs(np.array(temps_E) - temperature))
            E = values_E[idx]
        
        # Extract Poisson's ratio
        nu_data = mat_data['poissons_ratio']['values']
        temps_nu = [d['temperature'] for d in nu_data]
        values_nu = [d['value'] for d in nu_data]
        
        if SCIPY_AVAILABLE and len(temps_nu) > 1:
            nu_interp = interp1d(temps_nu, values_nu, kind='linear', fill_value='extrapolate')
            nu = float(nu_interp(temperature))
        else:
            idx = np.argmin(np.abs(np.array(temps_nu) - temperature))
            nu = values_nu[idx]
        
        # Calculate derived properties
        G = E / (2 * (1 + nu))  # Shear modulus
        K = E / (3 * (1 - 2 * nu))  # Bulk modulus
        
        return {
            'youngs_modulus_GPa': E,
            'poissons_ratio': nu,
            'shear_modulus_GPa': G,
            'bulk_modulus_GPa': K,
            'temperature_C': temperature
        }
    
    def get_fracture_properties(self, material: str, temperature: float) -> Dict[str, float]:
        """Get fracture properties at specified temperature."""
        mat_data = self.materials[material]['fracture_properties']
        
        # Fracture toughness
        Kic_data = mat_data['fracture_toughness_mode_I']['values']
        temps = [d['temperature'] for d in Kic_data]
        values = [d['value'] for d in Kic_data]
        
        if SCIPY_AVAILABLE and len(temps) > 1:
            Kic_interp = interp1d(temps, values, kind='linear', fill_value='extrapolate')
            Kic = float(Kic_interp(temperature))
        else:
            idx = np.argmin(np.abs(np.array(temps) - temperature))
            Kic = values[idx]
        
        # Calculate G_Ic if not directly available
        elastic = self.get_elastic_properties(material, temperature)
        E = elastic['youngs_modulus_GPa'] * 1e3  # Convert to MPa
        nu = elastic['poissons_ratio']
        
        G_Ic = (Kic**2 * (1 - nu**2)) / E  # in MPa·m = J/m²
        
        return {
            'fracture_toughness_MPa_sqrtm': Kic,
            'critical_energy_release_rate_J_per_m2': G_Ic,
            'temperature_C': temperature
        }
    
    def get_thermal_expansion(self, material: str, temp_range: Tuple[float, float]) -> float:
        """Get mean CTE over temperature range."""
        mat_data = self.materials[material]['thermo_physical_properties']
        cte_data = mat_data['coefficient_thermal_expansion']['values']
        
        # Find CTE values in range
        cte_values = []
        for entry in cte_data:
            temp_str = entry['temp_range']
            T1, T2 = map(int, temp_str.split('-'))
            if (T1 >= temp_range[0] and T1 <= temp_range[1]) or \
               (T2 >= temp_range[0] and T2 <= temp_range[1]):
                cte_values.append(entry['cte'])
        
        return np.mean(cte_values) if cte_values else None
    
    def export_to_csv(self, output_dir: str = '.'):
        """Export all material properties to CSV files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Export elastic properties
        elastic_data = []
        for mat_name, mat_info in self.materials.items():
            if 'elastic_properties' not in mat_info:
                continue
            
            E_data = mat_info['elastic_properties']['youngs_modulus']['values']
            nu_data = mat_info['elastic_properties']['poissons_ratio']['values']
            
            for E_entry in E_data:
                T = E_entry['temperature']
                E = E_entry['value']
                E_unc = E_entry.get('uncertainty', np.nan)
                
                # Find matching nu
                nu_entry = next((n for n in nu_data if n['temperature'] == T), None)
                nu = nu_entry['value'] if nu_entry else np.nan
                nu_unc = nu_entry.get('uncertainty', np.nan) if nu_entry else np.nan
                
                elastic_data.append({
                    'material': mat_name,
                    'temperature_C': T,
                    'youngs_modulus_GPa': E,
                    'youngs_modulus_uncertainty': E_unc,
                    'poissons_ratio': nu,
                    'poissons_ratio_uncertainty': nu_unc,
                    'reference': E_entry.get('reference', '')
                })
        
        df_elastic = pd.DataFrame(elastic_data)
        df_elastic.to_csv(output_path / 'elastic_properties.csv', index=False)
        print(f"Exported elastic properties to {output_path / 'elastic_properties.csv'}")
        
        # Export fracture properties
        fracture_data = []
        for mat_name, mat_info in self.materials.items():
            if 'fracture_properties' not in mat_info:
                continue
            
            if 'fracture_toughness_mode_I' not in mat_info['fracture_properties']:
                continue
                
            Kic_data = mat_info['fracture_properties']['fracture_toughness_mode_I']['values']
            
            for entry in Kic_data:
                fracture_data.append({
                    'material': mat_name,
                    'temperature_C': entry['temperature'],
                    'fracture_toughness_MPa_sqrtm': entry['value'],
                    'uncertainty': entry.get('uncertainty', np.nan),
                    'test_method': entry.get('test_method', ''),
                    'reference': entry.get('reference', '')
                })
        
        df_fracture = pd.DataFrame(fracture_data)
        df_fracture.to_csv(output_path / 'fracture_properties.csv', index=False)
        print(f"Exported fracture properties to {output_path / 'fracture_properties.csv'}")
        
        # Export CTE data
        cte_data = []
        for mat_name, mat_info in self.materials.items():
            if 'thermo_physical_properties' not in mat_info:
                continue
            
            if 'coefficient_thermal_expansion' not in mat_info['thermo_physical_properties']:
                continue
                
            cte_entries = mat_info['thermo_physical_properties']['coefficient_thermal_expansion']['values']
            
            for entry in cte_entries:
                cte_data.append({
                    'material': mat_name,
                    'temp_range': entry['temp_range'],
                    'CTE_1e-6_K-1': entry['cte'],
                    'uncertainty': entry.get('uncertainty', np.nan),
                    'reference': entry.get('reference', '')
                })
        
        df_cte = pd.DataFrame(cte_data)
        df_cte.to_csv(output_path / 'thermal_expansion_coefficients.csv', index=False)
        print(f"Exported CTE data to {output_path / 'thermal_expansion_coefficients.csv'}")
        
        # Export interface properties
        interface_data = []
        interfaces = self.materials.get('interface_properties', {})
        for int_name, int_info in interfaces.items():
            if int_name == 'description' or int_name == 'note':
                continue
            
            if 'fracture_toughness_mode_I' in int_info:
                for entry in int_info['fracture_toughness_mode_I']['values']:
                    interface_data.append({
                        'interface': int_name,
                        'interface_full_name': int_info.get('interface_name', ''),
                        'criticality': int_info.get('criticality', ''),
                        'temperature_C': entry['temperature'],
                        'fracture_toughness_MPa_sqrtm': entry['value'],
                        'uncertainty': entry.get('uncertainty', np.nan),
                        'test_method': entry.get('test_method', ''),
                        'reference': entry.get('reference', '')
                    })
        
        df_interface = pd.DataFrame(interface_data)
        df_interface.to_csv(output_path / 'interface_fracture_properties.csv', index=False)
        print(f"Exported interface properties to {output_path / 'interface_fracture_properties.csv'}")
        
        # Export chemical expansion data
        chem_exp_data = []
        for mat_name, mat_info in self.materials.items():
            if 'chemical_expansion' not in mat_info:
                continue
            
            chem_exp = mat_info['chemical_expansion']
            
            if 'chemical_expansion_coefficient' in chem_exp:
                coeff = chem_exp['chemical_expansion_coefficient']
                chem_exp_data.append({
                    'material': mat_name,
                    'coefficient': coeff.get('value', np.nan),
                    'uncertainty': coeff.get('uncertainty', np.nan),
                    'temperature_C': coeff.get('temperature', np.nan),
                    'units': coeff.get('units', ''),
                    'reference': coeff.get('reference', ''),
                    'notes': coeff.get('note', '')
                })
        
        df_chem_exp = pd.DataFrame(chem_exp_data)
        df_chem_exp.to_csv(output_path / 'chemical_expansion_coefficients.csv', index=False)
        print(f"Exported chemical expansion data to {output_path / 'chemical_expansion_coefficients.csv'}")
        
        return {
            'elastic': df_elastic,
            'fracture': df_fracture,
            'cte': df_cte,
            'interface': df_interface,
            'chemical_expansion': df_chem_exp
        }
    
    def plot_youngs_modulus_vs_temperature(self, materials: Optional[List[str]] = None, 
                                           save_path: Optional[str] = None):
        """Plot Young's modulus vs temperature for selected materials."""
        if materials is None:
            materials = ['8YSZ_electrolyte', 'Ni_metal', 'Ni_YSZ_cermet', 'LSM_YSZ_cathode']
        
        plt.figure(figsize=(10, 6))
        
        for mat in materials:
            if mat not in self.materials:
                continue
            
            mat_data = self.materials[mat]['elastic_properties']['youngs_modulus']['values']
            temps = [d['temperature'] for d in mat_data]
            values = [d['value'] for d in mat_data]
            uncertainties = [d.get('uncertainty', 0) for d in mat_data]
            
            # Clean material name for legend
            clean_name = mat.replace('_', ' ').replace('YSZ', '8YSZ')
            
            plt.errorbar(temps, values, yerr=uncertainties, marker='o', 
                        label=clean_name, linewidth=2, capsize=4)
        
        plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        plt.ylabel("Young's Modulus (GPa)", fontsize=12, fontweight='bold')
        plt.title("Temperature-Dependent Young's Modulus for SOFC Materials", 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.savefig('youngs_modulus_vs_temperature.png', dpi=300, bbox_inches='tight')
            print("Saved plot to youngs_modulus_vs_temperature.png")
        
        plt.close()
    
    def plot_fracture_toughness_comparison(self, save_path: Optional[str] = None):
        """Compare fracture toughness across materials and interfaces."""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Collect data
        plot_data = []
        
        # Bulk materials
        bulk_materials = ['8YSZ_electrolyte', 'Ni_YSZ_cermet', 'LSM_YSZ_cathode', 'NiO']
        for mat in bulk_materials:
            if mat not in self.materials:
                continue
            if 'fracture_properties' not in self.materials[mat]:
                continue
            
            frac_data = self.materials[mat]['fracture_properties']
            if 'fracture_toughness_mode_I' not in frac_data:
                continue
            
            values = frac_data['fracture_toughness_mode_I']['values']
            for entry in values:
                if entry['temperature'] in [25, 800]:
                    plot_data.append({
                        'name': mat.replace('_', ' '),
                        'type': 'Bulk',
                        'temperature': entry['temperature'],
                        'value': entry['value'],
                        'uncertainty': entry.get('uncertainty', 0)
                    })
        
        # Interface properties
        interfaces = self.materials.get('interface_properties', {})
        for int_name, int_info in interfaces.items():
            if int_name in ['description', 'note']:
                continue
            
            if 'fracture_toughness_mode_I' not in int_info:
                continue
            
            values = int_info['fracture_toughness_mode_I']['values']
            for entry in values:
                if entry['temperature'] in [25, 800]:
                    plot_data.append({
                        'name': int_info.get('interface_name', int_name),
                        'type': 'Interface',
                        'temperature': entry['temperature'],
                        'value': entry['value'],
                        'uncertainty': entry.get('uncertainty', 0)
                    })
        
        # Plot
        df = pd.DataFrame(plot_data)
        
        # Get unique names from both temperatures
        df_25 = df[df['temperature'] == 25].copy()
        df_800 = df[df['temperature'] == 800].copy()
        
        # Get all unique names
        all_names = sorted(set(df_25['name'].tolist() + df_800['name'].tolist()))
        
        # Create aligned data
        values_25 = []
        values_800 = []
        errors_25 = []
        errors_800 = []
        types = []
        
        for name in all_names:
            # Get 25°C data
            data_25 = df_25[df_25['name'] == name]
            if len(data_25) > 0:
                values_25.append(data_25.iloc[0]['value'])
                errors_25.append(data_25.iloc[0]['uncertainty'])
                types.append(data_25.iloc[0]['type'])
            else:
                values_25.append(0)
                errors_25.append(0)
                # Try to get type from 800°C data
                data_800_temp = df_800[df_800['name'] == name]
                if len(data_800_temp) > 0:
                    types.append(data_800_temp.iloc[0]['type'])
                else:
                    types.append('Unknown')
            
            # Get 800°C data
            data_800 = df_800[df_800['name'] == name]
            if len(data_800) > 0:
                values_800.append(data_800.iloc[0]['value'])
                errors_800.append(data_800.iloc[0]['uncertainty'])
            else:
                values_800.append(0)
                errors_800.append(0)
        
        x = np.arange(len(all_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, values_25, width, yerr=errors_25,
                      label='25°C', capsize=4, alpha=0.8, color='steelblue')
        
        bars2 = ax.bar(x + width/2, values_800, width, yerr=errors_800,
                      label='800°C', capsize=4, alpha=0.8, color='coral')
        
        ax.set_ylabel('Fracture Toughness K$_{Ic}$ (MPa√m)', fontsize=12, fontweight='bold')
        ax.set_title('Fracture Toughness Comparison: Bulk Materials vs Interfaces', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add type labels
        for i, mat_type in enumerate(types):
            if mat_type == 'Interface':
                ax.axvspan(i-0.5, i+0.5, alpha=0.1, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.savefig('fracture_toughness_comparison.png', dpi=300, bbox_inches='tight')
            print("Saved plot to fracture_toughness_comparison.png")
        
        plt.close()
    
    def plot_thermal_expansion_mismatch(self, save_path: Optional[str] = None):
        """Plot CTE for all materials to visualize mismatch."""
        plt.figure(figsize=(11, 7))
        
        materials_to_plot = {
            '8YSZ_electrolyte': '8YSZ Electrolyte',
            'Ni_YSZ_cermet': 'Ni-YSZ Anode',
            'LSM_YSZ_cathode': 'LSM-YSZ Cathode',
            'Ni_metal': 'Ni Metal',
            'NiO': 'NiO'
        }
        
        for mat_key, mat_label in materials_to_plot.items():
            if mat_key not in self.materials:
                continue
            
            if 'thermo_physical_properties' not in self.materials[mat_key]:
                continue
            
            cte_data = self.materials[mat_key]['thermo_physical_properties']['coefficient_thermal_expansion']['values']
            
            # Extract temperature midpoints and CTE values
            temps = []
            ctes = []
            uncertainties = []
            
            for entry in cte_data:
                temp_range = entry['temp_range']
                T1, T2 = map(int, temp_range.split('-'))
                temps.append((T1 + T2) / 2)
                ctes.append(entry['cte'])
                uncertainties.append(entry.get('uncertainty', 0))
            
            plt.errorbar(temps, ctes, yerr=uncertainties, marker='o', 
                        label=mat_label, linewidth=2.5, markersize=7, capsize=4)
        
        plt.xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        plt.ylabel('CTE (×10⁻⁶ K⁻¹)', fontsize=12, fontweight='bold')
        plt.title('Thermal Expansion Coefficient Comparison - CTE Mismatch Analysis', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        else:
            plt.savefig('thermal_expansion_mismatch.png', dpi=300, bbox_inches='tight')
            print("Saved plot to thermal_expansion_mismatch.png")
        
        plt.close()
    
    def generate_summary_report(self, output_path: str = 'material_properties_summary.txt'):
        """Generate a comprehensive text summary of the database."""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SOFC MATERIAL PROPERTY DATABASE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Database Version: {self.metadata['version']}\n")
            f.write(f"Date Created: {self.metadata['date_created']}\n")
            f.write(f"Temperature Range: {self.metadata['temperature_range']}\n\n")
            
            f.write("DATA SOURCES:\n")
            for source in self.metadata['data_sources']:
                f.write(f"  - {source}\n")
            f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("MATERIAL SUMMARY AT KEY TEMPERATURES (25°C and 800°C)\n")
            f.write("="*80 + "\n\n")
            
            key_materials = ['8YSZ_electrolyte', 'Ni_YSZ_cermet', 'LSM_YSZ_cathode']
            
            for mat in key_materials:
                if mat not in self.materials:
                    continue
                
                f.write(f"\n{mat.replace('_', ' ').upper()}\n")
                f.write("-"*80 + "\n")
                
                # Elastic properties
                props_25 = self.get_elastic_properties(mat, 25)
                props_800 = self.get_elastic_properties(mat, 800)
                
                f.write(f"\nElastic Properties:\n")
                f.write(f"  Young's Modulus:  {props_25['youngs_modulus_GPa']:.1f} GPa (25°C) → "
                       f"{props_800['youngs_modulus_GPa']:.1f} GPa (800°C)\n")
                f.write(f"  Poisson's Ratio:  {props_25['poissons_ratio']:.3f} (25°C) → "
                       f"{props_800['poissons_ratio']:.3f} (800°C)\n")
                
                # Fracture properties
                if 'fracture_properties' in self.materials[mat]:
                    frac_25 = self.get_fracture_properties(mat, 25)
                    frac_800 = self.get_fracture_properties(mat, 800)
                    
                    f.write(f"\nFracture Properties:\n")
                    f.write(f"  Fracture Toughness: {frac_25['fracture_toughness_MPa_sqrtm']:.2f} MPa√m (25°C) → "
                           f"{frac_800['fracture_toughness_MPa_sqrtm']:.2f} MPa√m (800°C)\n")
                    f.write(f"  G_Ic:               {frac_25['critical_energy_release_rate_J_per_m2']:.1f} J/m² (25°C) → "
                           f"{frac_800['critical_energy_release_rate_J_per_m2']:.1f} J/m² (800°C)\n")
                
                # Thermal expansion
                cte = self.get_thermal_expansion(mat, (25, 800))
                if cte:
                    f.write(f"\nThermal Properties:\n")
                    f.write(f"  Mean CTE (25-800°C): {cte:.2f} ×10⁻⁶ K⁻¹\n")
                
                # Chemical expansion
                if 'chemical_expansion' in self.materials[mat]:
                    chem_exp = self.materials[mat]['chemical_expansion']
                    if 'chemical_expansion_coefficient' in chem_exp:
                        coeff = chem_exp['chemical_expansion_coefficient']
                        f.write(f"\nChemical Expansion:\n")
                        f.write(f"  Coefficient: {coeff['value']:.3f} {coeff['units']}\n")
                        f.write(f"  Note: {coeff.get('note', 'N/A')}\n")
            
            # Interface properties
            f.write("\n" + "="*80 + "\n")
            f.write("INTERFACE PROPERTIES (CRITICAL FOR FAILURE PREDICTION)\n")
            f.write("="*80 + "\n\n")
            
            interfaces = self.materials.get('interface_properties', {})
            for int_name, int_info in interfaces.items():
                if int_name in ['description', 'note']:
                    continue
                
                f.write(f"\n{int_info.get('interface_name', int_name)}\n")
                f.write(f"Criticality: {int_info.get('criticality', 'N/A')}\n")
                f.write("-"*80 + "\n")
                
                if 'fracture_toughness_mode_I' in int_info:
                    values = int_info['fracture_toughness_mode_I']['values']
                    f.write("Fracture Toughness K_Ic:\n")
                    for entry in values:
                        f.write(f"  {entry['temperature']}°C: {entry['value']:.2f} ± "
                               f"{entry.get('uncertainty', 0):.2f} MPa√m "
                               f"[{entry.get('reference', 'N/A')}]\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("REFERENCES\n")
            f.write("="*80 + "\n\n")
            
            for ref_key, ref_text in sorted(self.references.items())[:20]:
                f.write(f"[{ref_key}]\n{ref_text}\n\n")
            
            f.write(f"\n... and {len(self.references) - 20} more references.\n\n")
        
        print(f"Summary report generated: {output_path}")


def main():
    """Main execution function."""
    print("="*80)
    print("SOFC Material Property Database - Analysis Tool")
    print("="*80)
    print()
    
    # Load database
    print("Loading material property database...")
    db = MaterialPropertyDatabase('material_property_dataset.json')
    print(f"✓ Loaded {len(db.materials)} materials")
    print(f"✓ Database version: {db.metadata['version']}")
    print()
    
    # Export to CSV
    print("Exporting data to CSV files...")
    dataframes = db.export_to_csv(output_dir='.')
    print()
    
    # Generate plots
    print("Generating visualizations...")
    db.plot_youngs_modulus_vs_temperature()
    db.plot_fracture_toughness_comparison()
    db.plot_thermal_expansion_mismatch()
    print()
    
    # Generate summary report
    print("Generating summary report...")
    db.generate_summary_report()
    print()
    
    # Example property queries
    print("="*80)
    print("EXAMPLE PROPERTY QUERIES")
    print("="*80)
    print()
    
    print("8YSZ Electrolyte at 800°C:")
    props = db.get_elastic_properties('8YSZ_electrolyte', 800)
    print(f"  E = {props['youngs_modulus_GPa']:.1f} GPa")
    print(f"  ν = {props['poissons_ratio']:.3f}")
    print(f"  G = {props['shear_modulus_GPa']:.1f} GPa")
    print()
    
    frac_props = db.get_fracture_properties('8YSZ_electrolyte', 800)
    print(f"  K_Ic = {frac_props['fracture_toughness_MPa_sqrtm']:.2f} MPa√m")
    print(f"  G_Ic = {frac_props['critical_energy_release_rate_J_per_m2']:.1f} J/m²")
    print()
    
    print("Ni-YSZ Anode at 800°C:")
    props = db.get_elastic_properties('Ni_YSZ_cermet', 800)
    print(f"  E = {props['youngs_modulus_GPa']:.1f} GPa")
    print(f"  ν = {props['poissons_ratio']:.3f}")
    print()
    
    cte = db.get_thermal_expansion('Ni_YSZ_cermet', (25, 800))
    print(f"  Mean CTE (25-800°C) = {cte:.2f} ×10⁻⁶ K⁻¹")
    print()
    
    print("="*80)
    print("Analysis complete! Generated files:")
    print("  - elastic_properties.csv")
    print("  - fracture_properties.csv")
    print("  - thermal_expansion_coefficients.csv")
    print("  - interface_fracture_properties.csv")
    print("  - chemical_expansion_coefficients.csv")
    print("  - youngs_modulus_vs_temperature.png")
    print("  - fracture_toughness_comparison.png")
    print("  - thermal_expansion_mismatch.png")
    print("  - material_properties_summary.txt")
    print("="*80)


if __name__ == "__main__":
    main()
