#!/usr/bin/env python3
"""
Visualization and Analysis Tools for SOFC Material Property Data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MaterialPropertyVisualizer:
    """Visualize and analyze SOFC material property data"""
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.materials_color_map = {
            'YSZ-8mol': '#1f77b4',
            'YSZ-3mol': '#ff7f0e',
            'Ni': '#2ca02c',
            'NiO': '#d62728',
            'Ni-YSZ': '#9467bd',
            'LSM': '#8c564b',
            'LSCF': '#e377c2',
            'LSC': '#7f7f7f',
            'GDC': '#bcbd22',
            'Crofer22APU': '#17becf'
        }
    
    def plot_elastic_properties(self, df: pd.DataFrame):
        """Create comprehensive elastic property plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Elastic Properties of SOFC Materials', fontsize=16, fontweight='bold')
        
        # 1. Young's Modulus Distribution
        ax = axes[0, 0]
        for material in df['material'].unique():
            data = df[df['material'] == material]['youngs_modulus_GPa']
            ax.hist(data, alpha=0.5, label=material, bins=20, 
                   color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Young\'s Modulus (GPa)')
        ax.set_ylabel('Frequency')
        ax.set_title('Young\'s Modulus Distribution')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 2. Temperature Dependence
        ax = axes[0, 1]
        for material in df['material'].unique():
            mat_data = df[df['material'] == material]
            ax.scatter(mat_data['temperature_C'], mat_data['youngs_modulus_GPa'], 
                      alpha=0.3, s=10, label=material,
                      color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Temperature Effect on Elastic Modulus')
        
        # 3. Porosity Effect
        ax = axes[0, 2]
        for material in ['Ni-YSZ', 'LSM', 'LSCF']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['porosity'], mat_data['youngs_modulus_GPa'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Porosity (fraction)')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Porosity Effect on Elastic Modulus')
        ax.legend()
        
        # 4. Poisson's Ratio vs Young's Modulus
        ax = axes[1, 0]
        for material in df['material'].unique():
            mat_data = df[df['material'] == material]
            ax.scatter(mat_data['poissons_ratio'], mat_data['youngs_modulus_GPa'], 
                      alpha=0.5, s=30, label=material,
                      color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Poisson\'s Ratio')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Elastic Property Correlation')
        
        # 5. Processing Method Effect
        ax = axes[1, 1]
        processing_data = df.groupby(['material', 'processing_method'])['youngs_modulus_GPa'].mean().unstack()
        processing_data.plot(kind='bar', ax=ax)
        ax.set_xlabel('Material')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Processing Method Effect')
        ax.legend(title='Processing', bbox_to_anchor=(1.05, 1))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 6. Grain Size Effect
        ax = axes[1, 2]
        for material in ['YSZ-8mol', 'YSZ-3mol', 'GDC']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['grain_size_um'], mat_data['youngs_modulus_GPa'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Grain Size (μm)')
        ax.set_ylabel('Young\'s Modulus (GPa)')
        ax.set_title('Grain Size Effect')
        ax.set_xscale('log')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_fracture_properties(self, df: pd.DataFrame):
        """Visualize fracture properties"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Fracture Properties of SOFC Materials', fontsize=16, fontweight='bold')
        
        # 1. Fracture Toughness Comparison
        ax = axes[0, 0]
        materials = df['material'].unique()
        toughness_data = [df[df['material'] == m]['fracture_toughness_MPam05'].dropna() 
                          for m in materials]
        bp = ax.boxplot(toughness_data, labels=materials, patch_artist=True)
        for patch, material in zip(bp['boxes'], materials):
            patch.set_facecolor(self.materials_color_map.get(material, 'gray'))
        ax.set_ylabel('Fracture Toughness (MPa·m^0.5)')
        ax.set_title('Fracture Toughness Distribution')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Temperature Effect on Toughness
        ax = axes[0, 1]
        for material in ['YSZ-8mol', 'Ni-YSZ', 'LSCF']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['temperature_C'], mat_data['fracture_toughness_MPam05'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
                
                # Fit trend line
                z = np.polyfit(mat_data['temperature_C'].dropna(), 
                              mat_data['fracture_toughness_MPam05'].dropna(), 1)
                p = np.poly1d(z)
                temps = np.linspace(25, 1000, 100)
                ax.plot(temps, p(temps), "--", alpha=0.5,
                       color=self.materials_color_map.get(material, 'gray'))
        
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Fracture Toughness (MPa·m^0.5)')
        ax.set_title('Temperature Dependence of Fracture Toughness')
        ax.legend()
        
        # 3. Critical Energy Release Rate
        ax = axes[0, 2]
        for material in df['material'].unique():
            mat_data = df[df['material'] == material]['critical_energy_release_rate_Jm2'].dropna()
            if len(mat_data) > 0:
                ax.scatter([material] * len(mat_data), mat_data, 
                          alpha=0.3, s=10,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_ylabel('G_c (J/m²)')
        ax.set_yscale('log')
        ax.set_title('Critical Energy Release Rate')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 4. Weibull Analysis
        ax = axes[1, 0]
        weibull_materials = ['YSZ-8mol', 'YSZ-3mol', 'GDC', 'LSCF']
        for material in weibull_materials:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                weibull_m = mat_data['weibull_modulus'].dropna()
                char_strength = mat_data['characteristic_strength_MPa'].dropna()
                if len(weibull_m) > 0 and len(char_strength) > 0:
                    ax.scatter(weibull_m, char_strength, alpha=0.5, s=30, 
                              label=material,
                              color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Weibull Modulus')
        ax.set_ylabel('Characteristic Strength (MPa)')
        ax.set_title('Weibull Parameters')
        ax.legend()
        
        # 5. R-curve behavior
        ax = axes[1, 1]
        r_curve_mats = df[df['crack_extension_mm'] > 0]
        for material in ['YSZ-3mol', 'Ni-YSZ']:
            if material in r_curve_mats['material'].values:
                mat_data = r_curve_mats[r_curve_mats['material'] == material]
                ax.scatter(mat_data['crack_extension_mm'], 
                          mat_data['fracture_toughness_Rcurve_MPam05'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Crack Extension (mm)')
        ax.set_ylabel('K_R (MPa·m^0.5)')
        ax.set_title('R-Curve Behavior')
        ax.legend()
        
        # 6. Environment Effect
        ax = axes[1, 2]
        env_data = df.groupby(['material', 'test_environment'])['fracture_toughness_MPam05'].mean().unstack()
        if not env_data.empty:
            env_data.plot(kind='bar', ax=ax)
            ax.set_xlabel('Material')
            ax.set_ylabel('Fracture Toughness (MPa·m^0.5)')
            ax.set_title('Environment Effect on Fracture')
            ax.legend(title='Environment', bbox_to_anchor=(1.05, 1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
    
    def plot_interface_properties(self, df: pd.DataFrame):
        """Visualize interface fracture properties"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Interface Properties in SOFC Systems', fontsize=16, fontweight='bold')
        
        # 1. Interface Toughness Comparison
        ax = axes[0, 0]
        df['interface'] = df['material_1'] + '/' + df['material_2']
        interfaces = df['interface'].unique()
        toughness_data = [df[df['interface'] == i]['interface_toughness_MPam05'].dropna() 
                          for i in interfaces]
        bp = ax.boxplot(toughness_data, labels=interfaces, patch_artist=True)
        ax.set_ylabel('Interface Toughness (MPa·m^0.5)')
        ax.set_title('Interface Toughness Distribution')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 2. Thermal Cycling Effect
        ax = axes[0, 1]
        colors = plt.cm.viridis(np.linspace(0, 1, len(interfaces)))
        for i, (interface, color) in enumerate(zip(interfaces, colors)):
            int_data = df[df['interface'] == interface]
            ax.scatter(int_data['thermal_cycles'], int_data['interface_toughness_MPam05'], 
                      alpha=0.4, s=20, label=interface, color=color)
        ax.set_xlabel('Thermal Cycles')
        ax.set_ylabel('Interface Toughness (MPa·m^0.5)')
        ax.set_title('Thermal Cycling Degradation')
        ax.legend(bbox_to_anchor=(1.05, 1))
        
        # 3. Redox Cycling Effect (Ni-containing interfaces)
        ax = axes[0, 2]
        ni_interfaces = df[(df['material_1'].str.contains('Ni')) | 
                          (df['material_2'].str.contains('Ni'))]
        for interface in ni_interfaces['interface'].unique():
            int_data = ni_interfaces[ni_interfaces['interface'] == interface]
            ax.scatter(int_data['redox_cycles'], int_data['interface_toughness_MPam05'], 
                      alpha=0.4, s=20, label=interface)
        ax.set_xlabel('Redox Cycles')
        ax.set_ylabel('Interface Toughness (MPa·m^0.5)')
        ax.set_title('Redox Cycling Effect')
        ax.legend()
        
        # 4. Processing Method Effect
        ax = axes[1, 0]
        process_data = df.groupby(['interface', 'processing_method'])['interface_toughness_MPam05'].mean().unstack()
        if not process_data.empty:
            process_data.plot(kind='bar', ax=ax)
            ax.set_xlabel('Interface')
            ax.set_ylabel('Interface Toughness (MPa·m^0.5)')
            ax.set_title('Processing Method Effect')
            ax.legend(title='Processing')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 5. Mode Mixity Effect
        ax = axes[1, 1]
        ax.scatter(df['mode_I_fraction'], df['interface_toughness_MPam05'], 
                  c=df['residual_stress_MPa'], cmap='coolwarm', alpha=0.5)
        ax.set_xlabel('Mode I Fraction')
        ax.set_ylabel('Interface Toughness (MPa·m^0.5)')
        ax.set_title('Mode Mixity and Residual Stress')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Residual Stress (MPa)')
        
        # 6. CTE Mismatch vs Residual Stress
        ax = axes[1, 2]
        ax.scatter(df['CTE_mismatch_ppm_K'], df['residual_stress_MPa'], 
                  c=df['temperature_C'], cmap='plasma', alpha=0.5)
        ax.set_xlabel('CTE Mismatch (ppm/K)')
        ax.set_ylabel('Residual Stress (MPa)')
        ax.set_title('CTE Mismatch Effect')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Temperature (°C)')
        
        plt.tight_layout()
        return fig
    
    def plot_thermal_properties(self, df: pd.DataFrame):
        """Visualize thermal expansion properties"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Thermal Properties of SOFC Materials', fontsize=16, fontweight='bold')
        
        # 1. CTE Distribution
        ax = axes[0, 0]
        materials = df['material'].unique()
        cte_data = [df[df['material'] == m]['CTE_ppm_K'].dropna() for m in materials]
        bp = ax.boxplot(cte_data, labels=materials, patch_artist=True)
        for patch, material in zip(bp['boxes'], materials):
            patch.set_facecolor(self.materials_color_map.get(material, 'gray'))
        ax.set_ylabel('CTE (10⁻⁶/K)')
        ax.set_title('Coefficient of Thermal Expansion')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.axhline(y=12, color='r', linestyle='--', alpha=0.5, label='Target CTE')
        ax.legend()
        
        # 2. Temperature Dependence of CTE
        ax = axes[0, 1]
        for material in ['YSZ-8mol', 'Ni', 'LSCF', 'Crofer22APU']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['T_mean_C'], mat_data['CTE_ppm_K'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('CTE (10⁻⁶/K)')
        ax.set_title('Temperature Dependence of CTE')
        ax.legend()
        
        # 3. Thermal Conductivity
        ax = axes[1, 0]
        for material in df['material'].unique():
            mat_data = df[df['material'] == material]
            ax.scatter(mat_data['T_mean_C'], mat_data['thermal_conductivity_W_mK'], 
                      alpha=0.3, s=15, label=material,
                      color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Thermal Conductivity (W/m·K)')
        ax.set_yscale('log')
        ax.set_title('Thermal Conductivity vs Temperature')
        ax.legend(bbox_to_anchor=(1.05, 1), ncol=1)
        
        # 4. CTE Mismatch Matrix
        ax = axes[1, 1]
        materials_subset = ['YSZ-8mol', 'Ni', 'Ni-YSZ', 'LSCF', 'Crofer22APU']
        cte_means = []
        for mat in materials_subset:
            if mat in df['material'].values:
                cte_means.append(df[df['material'] == mat]['CTE_ppm_K'].mean())
            else:
                cte_means.append(np.nan)
        
        n = len(materials_subset)
        mismatch_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if not np.isnan(cte_means[i]) and not np.isnan(cte_means[j]):
                    mismatch_matrix[i, j] = abs(cte_means[i] - cte_means[j])
        
        im = ax.imshow(mismatch_matrix, cmap='RdYlGn_r')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(materials_subset, rotation=45, ha='right')
        ax.set_yticklabels(materials_subset)
        ax.set_title('CTE Mismatch Matrix (10⁻⁶/K)')
        
        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f'{mismatch_matrix[i, j]:.1f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        return fig
    
    def plot_chemical_expansion(self, df: pd.DataFrame):
        """Visualize chemical expansion properties"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Chemical Expansion in SOFC Materials', fontsize=16, fontweight='bold')
        
        # 1. Strain vs pO2
        ax = axes[0, 0]
        for material in ['LSCF', 'LSC', 'GDC']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                mat_data = mat_data[mat_data['pO2_final_atm'] > 0]
                ax.scatter(np.log10(mat_data['pO2_final_atm']), 
                          mat_data['linear_strain'], 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('log(pO₂/atm)')
        ax.set_ylabel('Linear Strain')
        ax.set_title('Oxygen Partial Pressure Effect')
        ax.legend()
        
        # 2. Ni/NiO Redox
        ax = axes[0, 1]
        ni_data = df[df['material'].isin(['Ni', 'NiO'])]
        for material in ['Ni', 'NiO']:
            if material in ni_data['material'].values:
                mat_data = ni_data[ni_data['material'] == material]
                ax.hist(mat_data['linear_strain'], alpha=0.6, bins=20, 
                       label=material,
                       color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Linear Strain')
        ax.set_ylabel('Frequency')
        ax.set_title('Ni/NiO Redox Strain')
        ax.legend()
        
        # 3. Kinetics
        ax = axes[0, 2]
        for material in df['material'].unique():
            mat_data = df[df['material'] == material]
            if len(mat_data) > 10:
                ax.scatter(mat_data['time_hours'], mat_data['conversion_fraction'], 
                          alpha=0.3, s=10, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Conversion Fraction')
        ax.set_xscale('log')
        ax.set_title('Chemical Expansion Kinetics')
        ax.legend(bbox_to_anchor=(1.05, 1))
        
        # 4. Temperature Effect
        ax = axes[1, 0]
        for material in ['LSCF', 'LSC', 'Ni']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['temperature_C'], abs(mat_data['linear_strain']), 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('|Linear Strain|')
        ax.set_title('Temperature Effect on Chemical Expansion')
        ax.legend()
        
        # 5. Stress Generation
        ax = axes[1, 1]
        ax.scatter(df['linear_strain'], df['constrained_stress_MPa'], 
                  c=df['porosity'], cmap='viridis', alpha=0.5)
        ax.set_xlabel('Linear Strain')
        ax.set_ylabel('Constrained Stress (MPa)')
        ax.set_title('Stress from Chemical Expansion')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Porosity')
        
        # 6. Porosity Effect
        ax = axes[1, 2]
        for material in ['Ni-YSZ', 'LSCF']:
            if material in df['material'].values:
                mat_data = df[df['material'] == material]
                ax.scatter(mat_data['porosity'], abs(mat_data['linear_strain']), 
                          alpha=0.4, s=20, label=material,
                          color=self.materials_color_map.get(material, 'gray'))
        ax.set_xlabel('Porosity')
        ax.set_ylabel('|Linear Strain|')
        ax.set_title('Porosity Effect on Chemical Expansion')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_correlation_matrix(self, df: pd.DataFrame, title: str = "Correlation Matrix"):
        """Create correlation matrix heatmap"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'id' not in col.lower()]
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, datasets: dict):
        """Generate comprehensive visualization report"""
        
        print("Generating visualization report...")
        
        # Create output directory
        output_dir = Path("material_property_visualizations")
        output_dir.mkdir(exist_ok=True)
        
        # Process each dataset
        for name, df in datasets.items():
            print(f"Processing {name}...")
            
            if name == 'elastic_properties':
                fig = self.plot_elastic_properties(df)
                fig.savefig(output_dir / f"{name}_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            elif name == 'fracture_properties':
                fig = self.plot_fracture_properties(df)
                fig.savefig(output_dir / f"{name}_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            elif name == 'interface_properties':
                fig = self.plot_interface_properties(df)
                fig.savefig(output_dir / f"{name}_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            elif name == 'thermal_properties':
                fig = self.plot_thermal_properties(df)
                fig.savefig(output_dir / f"{name}_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
                
            elif name == 'chemical_expansion':
                fig = self.plot_chemical_expansion(df)
                fig.savefig(output_dir / f"{name}_analysis.png", dpi=150, bbox_inches='tight')
                plt.close(fig)
            
            # Create correlation matrix
            fig = self.create_correlation_matrix(df, f"{name.replace('_', ' ').title()} - Correlations")
            fig.savefig(output_dir / f"{name}_correlations.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"Visualizations saved to {output_dir}/")
        return output_dir


def main():
    """Main execution function"""
    
    print("Loading datasets...")
    
    # Try to load existing CSV files
    import glob
    csv_files = glob.glob("sofc_material_*.csv")
    
    if not csv_files:
        print("No existing data found. Generating new datasets...")
        from generate_material_data import SOFCMaterialDataGenerator
        generator = SOFCMaterialDataGenerator()
        datasets = generator.generate_complete_dataset()
        generator.save_datasets(datasets, format='csv')
    else:
        print(f"Found {len(csv_files)} CSV files. Loading...")
        datasets = {}
        for file in csv_files:
            if 'summary' not in file:
                name = file.replace('sofc_material_', '').split('_20')[0]
                datasets[name] = pd.read_csv(file)
                print(f"  Loaded {name}: {len(datasets[name])} samples")
    
    # Create visualizer and generate report
    visualizer = MaterialPropertyVisualizer()
    output_dir = visualizer.generate_report(datasets)
    
    print("\nVisualization complete!")
    print(f"All figures saved to: {output_dir}/")
    
    return datasets


if __name__ == "__main__":
    datasets = main()