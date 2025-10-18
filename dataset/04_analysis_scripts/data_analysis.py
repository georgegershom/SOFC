#!/usr/bin/env python3
"""
Data Analysis Script for Fire-Resistant Rubberized Concrete Dataset
Phase 1: Material Characterization & Specimen Preparation

This script provides comprehensive analysis and visualization of:
- Constituent materials characterization
- Mix design optimization
- Fresh concrete properties
- Statistical analysis and correlations
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RubberizedConcreteAnalyzer:
    def __init__(self, data_path="../"):
        """Initialize the analyzer with data path"""
        self.data_path = Path(data_path)
        self.constituent_data = {}
        self.mix_data = {}
        self.fresh_data = {}
        self.load_data()
    
    def load_data(self):
        """Load all JSON data files"""
        try:
            # Load constituent materials data
            with open(self.data_path / "01_constituent_materials/cement_data.json", 'r') as f:
                self.constituent_data['cement'] = json.load(f)
            
            with open(self.data_path / "01_constituent_materials/aggregates_data.json", 'r') as f:
                self.constituent_data['aggregates'] = json.load(f)
            
            with open(self.data_path / "01_constituent_materials/crumb_rubber_data.json", 'r') as f:
                self.constituent_data['rubber'] = json.load(f)
            
            with open(self.data_path / "01_constituent_materials/water_admixtures_data.json", 'r') as f:
                self.constituent_data['water_admixtures'] = json.load(f)
            
            # Load mix design data
            with open(self.data_path / "02_mix_designs/mix_design_matrix.json", 'r') as f:
                self.mix_data = json.load(f)
            
            # Load fresh properties data
            with open(self.data_path / "03_fresh_properties/fresh_concrete_data.json", 'r') as f:
                self.fresh_data = json.load(f)
            
            print("✓ All data files loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def analyze_constituent_materials(self):
        """Analyze constituent materials properties"""
        print("\\n" + "="*60)
        print("CONSTITUENT MATERIALS ANALYSIS")
        print("="*60)
        
        # Cement analysis
        cement = self.constituent_data['cement']['cement_characterization']
        print(f"\\nCement Type: {cement['type']}")
        print(f"Specific Gravity: {cement['specific_gravity']}")
        print(f"Fineness (Blaine): {cement['fineness_blaine']} {cement['fineness_blaine_units']}")
        
        # Chemical composition
        chem_comp = cement['chemical_composition_xrf']
        print("\\nChemical Composition (XRF):")
        for oxide, content in chem_comp.items():
            if oxide != 'units':
                print(f"  {oxide}: {content}%")
        
        # Bogue composition
        bogue = cement['bogue_composition']
        print("\\nBogue Composition:")
        for compound, content in bogue.items():
            if compound not in ['units', 'calculation_method']:
                print(f"  {compound}: {content}%")
        
        # Aggregates analysis
        coarse_agg = self.constituent_data['aggregates']['coarse_aggregate']
        fine_agg = self.constituent_data['aggregates']['fine_aggregate']
        
        print(f"\\nCoarse Aggregate: {coarse_agg['type']}")
        print(f"  Specific Gravity (SSD): {coarse_agg['specific_gravity_bulk_ssd']}")
        print(f"  Water Absorption: {coarse_agg['water_absorption']}%")
        print(f"  LA Abrasion: {coarse_agg['los_angeles_abrasion']}%")
        
        print(f"\\nFine Aggregate: {fine_agg['type']}")
        print(f"  Specific Gravity (SSD): {fine_agg['specific_gravity_bulk_ssd']}")
        print(f"  Water Absorption: {fine_agg['water_absorption']}%")
        print(f"  Fineness Modulus: {fine_agg['fineness_modulus']}")
        
        # Rubber analysis
        rubber = self.constituent_data['rubber']['crumb_rubber_characterization']
        print(f"\\nCrumb Rubber Source: {rubber['source']['tire_type']}")
        print(f"Processing: {rubber['source']['processing_method']}")
        print(f"Specific Gravity: {rubber['physical_properties']['specific_gravity']}")
        print(f"Mohs Hardness: {rubber['physical_properties']['mohs_hardness']}")
        
        # Thermal properties comparison
        print("\\nThermal Properties Comparison:")
        print(f"Cement thermal expansion: {cement['thermal_properties']['coefficient_thermal_expansion']:.1e} /°C")
        print(f"Coarse agg thermal expansion: {coarse_agg['thermal_properties']['coefficient_thermal_expansion']:.1e} /°C")
        print(f"Fine agg thermal expansion: {fine_agg['thermal_properties']['coefficient_thermal_expansion']:.1e} /°C")
        print(f"Rubber thermal expansion: {rubber['thermal_properties']['coefficient_thermal_expansion']:.1e} /°C")
    
    def create_mix_design_dataframe(self):
        """Create DataFrame from mix design data for analysis"""
        mix_designs = self.mix_data['mix_design_matrix']['mix_designs']
        
        data = []
        for mix_id, mix_info in mix_designs.items():
            row = {
                'mix_id': mix_id,
                'rubber_content': mix_info['rubber_content'],
                'rubber_size': mix_info.get('rubber_size', 'N/A'),
                'cement': mix_info['proportions']['cement_opc'],
                'silica_fume': mix_info['proportions']['silica_fume'],
                'water': mix_info['proportions']['water'],
                'fine_aggregate': mix_info['proportions']['fine_aggregate'],
                'coarse_aggregate': mix_info['proportions']['coarse_aggregate'],
                'crumb_rubber': mix_info['proportions'].get('crumb_rubber', 0),
                'superplasticizer': mix_info['proportions']['superplasticizer'],
                'w_cm_ratio': mix_info['w_cm_ratio'],
                'total_cementitious': mix_info['total_cementitious']
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def create_fresh_properties_dataframe(self):
        """Create DataFrame from fresh properties data"""
        batch_data = self.fresh_data['fresh_concrete_properties']['batch_data']
        
        data = []
        for mix_id, mix_batches in batch_data.items():
            if 'average' in mix_batches:
                avg_data = mix_batches['average']
                row = {
                    'mix_id': mix_id,
                    'slump': avg_data['slump'],
                    'air_content': avg_data['air_content'],
                    'unit_weight': avg_data['unit_weight'],
                    'fresh_temperature': avg_data['fresh_temperature']
                }
                data.append(row)
        
        return pd.DataFrame(data)
    
    def analyze_mix_designs(self):
        """Analyze mix design matrix"""
        print("\\n" + "="*60)
        print("MIX DESIGN ANALYSIS")
        print("="*60)
        
        df_mix = self.create_mix_design_dataframe()
        
        print(f"\\nTotal number of mix designs: {len(df_mix)}")
        print(f"Rubber content range: {df_mix['rubber_content'].min()}% - {df_mix['rubber_content'].max()}%")
        print(f"W/CM ratio range: {df_mix['w_cm_ratio'].min():.2f} - {df_mix['w_cm_ratio'].max():.2f}")
        
        # Group by rubber content
        rubber_groups = df_mix.groupby('rubber_content')
        print("\\nMix designs by rubber content:")
        for rubber_pct, group in rubber_groups:
            print(f"  {rubber_pct}% rubber: {len(group)} mixes")
        
        # Analyze material consumption trends
        print("\\nMaterial consumption trends with rubber content:")
        correlation_matrix = df_mix[['rubber_content', 'fine_aggregate', 'crumb_rubber', 
                                   'superplasticizer', 'w_cm_ratio']].corr()
        
        print(f"Fine aggregate vs rubber content correlation: {correlation_matrix.loc['fine_aggregate', 'rubber_content']:.3f}")
        print(f"Superplasticizer vs rubber content correlation: {correlation_matrix.loc['superplasticizer', 'rubber_content']:.3f}")
        
        return df_mix
    
    def analyze_fresh_properties(self):
        """Analyze fresh concrete properties"""
        print("\\n" + "="*60)
        print("FRESH PROPERTIES ANALYSIS")
        print("="*60)
        
        df_fresh = self.create_fresh_properties_dataframe()
        df_mix = self.create_mix_design_dataframe()
        
        # Merge with mix design data
        df_combined = df_fresh.merge(df_mix[['mix_id', 'rubber_content', 'rubber_size']], on='mix_id')
        
        print(f"\\nFresh properties summary:")
        print(f"Slump range: {df_fresh['slump'].min():.1f} - {df_fresh['slump'].max():.1f} mm")
        print(f"Air content range: {df_fresh['air_content'].min():.1f} - {df_fresh['air_content'].max():.1f}%")
        print(f"Unit weight range: {df_fresh['unit_weight'].min():.0f} - {df_fresh['unit_weight'].max():.0f} kg/m³")
        
        # Analyze trends with rubber content
        correlations = df_combined[['rubber_content', 'slump', 'air_content', 'unit_weight']].corr()
        
        print("\\nCorrelations with rubber content:")
        print(f"  Slump: {correlations.loc['slump', 'rubber_content']:.3f}")
        print(f"  Air content: {correlations.loc['air_content', 'rubber_content']:.3f}")
        print(f"  Unit weight: {correlations.loc['unit_weight', 'rubber_content']:.3f}")
        
        # Analyze by rubber size
        size_analysis = df_combined.groupby('rubber_size').agg({
            'slump': ['mean', 'std'],
            'air_content': ['mean', 'std'],
            'unit_weight': ['mean', 'std']
        }).round(2)
        
        print("\\nProperties by rubber size:")
        print(size_analysis)
        
        return df_combined
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        df_mix = self.create_mix_design_dataframe()
        df_fresh = self.create_fresh_properties_dataframe()
        df_combined = df_fresh.merge(df_mix[['mix_id', 'rubber_content', 'rubber_size']], on='mix_id')
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Mix design proportions
        ax1 = plt.subplot(3, 3, 1)
        materials = ['cement', 'fine_aggregate', 'coarse_aggregate', 'crumb_rubber']
        bottom = np.zeros(len(df_mix))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, material in enumerate(materials):
            values = df_mix[material].values
            plt.bar(df_mix['mix_id'], values, bottom=bottom, label=material.replace('_', ' ').title(), 
                   color=colors[i], alpha=0.8)
            bottom += values
        
        plt.title('Mix Design Proportions', fontsize=14, fontweight='bold')
        plt.xlabel('Mix ID')
        plt.ylabel('Content (kg/m³)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Rubber content vs Slump
        ax2 = plt.subplot(3, 3, 2)
        scatter = plt.scatter(df_combined['rubber_content'], df_combined['slump'], 
                            c=df_combined['rubber_content'], cmap='viridis', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Rubber Content (%)')
        
        # Add trend line
        z = np.polyfit(df_combined['rubber_content'], df_combined['slump'], 1)
        p = np.poly1d(z)
        plt.plot(df_combined['rubber_content'], p(df_combined['rubber_content']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.title('Slump vs Rubber Content', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Content (%)')
        plt.ylabel('Slump (mm)')
        plt.grid(True, alpha=0.3)
        
        # 3. Rubber content vs Air content
        ax3 = plt.subplot(3, 3, 3)
        scatter = plt.scatter(df_combined['rubber_content'], df_combined['air_content'], 
                            c=df_combined['rubber_content'], cmap='plasma', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Rubber Content (%)')
        
        # Add trend line
        z = np.polyfit(df_combined['rubber_content'], df_combined['air_content'], 1)
        p = np.poly1d(z)
        plt.plot(df_combined['rubber_content'], p(df_combined['rubber_content']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.title('Air Content vs Rubber Content', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Content (%)')
        plt.ylabel('Air Content (%)')
        plt.grid(True, alpha=0.3)
        
        # 4. Rubber content vs Unit weight
        ax4 = plt.subplot(3, 3, 4)
        scatter = plt.scatter(df_combined['rubber_content'], df_combined['unit_weight'], 
                            c=df_combined['rubber_content'], cmap='coolwarm', s=100, alpha=0.7)
        plt.colorbar(scatter, label='Rubber Content (%)')
        
        # Add trend line
        z = np.polyfit(df_combined['rubber_content'], df_combined['unit_weight'], 1)
        p = np.poly1d(z)
        plt.plot(df_combined['rubber_content'], p(df_combined['rubber_content']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.title('Unit Weight vs Rubber Content', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Content (%)')
        plt.ylabel('Unit Weight (kg/m³)')
        plt.grid(True, alpha=0.3)
        
        # 5. Fresh properties comparison by rubber size
        ax5 = plt.subplot(3, 3, 5)
        rubber_sizes = df_combined['rubber_size'].unique()
        x_pos = np.arange(len(rubber_sizes))
        
        slump_means = [df_combined[df_combined['rubber_size'] == size]['slump'].mean() 
                      for size in rubber_sizes]
        slump_stds = [df_combined[df_combined['rubber_size'] == size]['slump'].std() 
                     for size in rubber_sizes]
        
        plt.bar(x_pos, slump_means, yerr=slump_stds, capsize=5, alpha=0.7, 
               color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        plt.title('Average Slump by Rubber Size', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Size')
        plt.ylabel('Slump (mm)')
        plt.xticks(x_pos, rubber_sizes, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 6. Correlation heatmap
        ax6 = plt.subplot(3, 3, 6)
        corr_data = df_combined[['rubber_content', 'slump', 'air_content', 'unit_weight', 
                               'fresh_temperature']].corr()
        
        sns.heatmap(corr_data, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Fresh Properties Correlation Matrix', fontsize=14, fontweight='bold')
        
        # 7. Superplasticizer dosage vs rubber content
        ax7 = plt.subplot(3, 3, 7)
        plt.scatter(df_mix['rubber_content'], df_mix['superplasticizer'], 
                   c=df_mix['rubber_content'], cmap='viridis', s=100, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(df_mix['rubber_content'], df_mix['superplasticizer'], 1)
        p = np.poly1d(z)
        plt.plot(df_mix['rubber_content'], p(df_mix['rubber_content']), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.title('Superplasticizer Dosage vs Rubber Content', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Content (%)')
        plt.ylabel('Superplasticizer (kg/m³)')
        plt.grid(True, alpha=0.3)
        
        # 8. W/CM ratio distribution
        ax8 = plt.subplot(3, 3, 8)
        plt.hist(df_mix['w_cm_ratio'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(df_mix['w_cm_ratio'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df_mix["w_cm_ratio"].mean():.3f}')
        plt.title('W/CM Ratio Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('W/CM Ratio')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Material efficiency plot
        ax9 = plt.subplot(3, 3, 9)
        df_mix['total_materials'] = (df_mix['cement'] + df_mix['silica_fume'] + 
                                   df_mix['fine_aggregate'] + df_mix['coarse_aggregate'] + 
                                   df_mix['crumb_rubber'])
        
        plt.scatter(df_mix['rubber_content'], df_mix['total_materials'], 
                   c=df_mix['rubber_content'], cmap='viridis', s=100, alpha=0.7)
        plt.title('Total Material Content vs Rubber Content', fontsize=14, fontweight='bold')
        plt.xlabel('Rubber Content (%)')
        plt.ylabel('Total Materials (kg/m³)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_path / '04_analysis_scripts/comprehensive_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Comprehensive visualization saved as 'comprehensive_analysis.png'")
    
    def generate_statistical_report(self):
        """Generate detailed statistical report"""
        print("\\n" + "="*60)
        print("STATISTICAL ANALYSIS REPORT")
        print("="*60)
        
        df_mix = self.create_mix_design_dataframe()
        df_fresh = self.create_fresh_properties_dataframe()
        df_combined = df_fresh.merge(df_mix[['mix_id', 'rubber_content', 'rubber_size']], on='mix_id')
        
        # Descriptive statistics
        print("\\nDESCRIPTIVE STATISTICS")
        print("-" * 40)
        
        stats_summary = df_combined[['rubber_content', 'slump', 'air_content', 
                                   'unit_weight', 'fresh_temperature']].describe()
        print(stats_summary.round(2))
        
        # Regression analysis
        print("\\nREGRESSION ANALYSIS")
        print("-" * 40)
        
        from scipy import stats
        
        # Slump vs rubber content
        slope_slump, intercept_slump, r_slump, p_slump, se_slump = stats.linregress(
            df_combined['rubber_content'], df_combined['slump'])
        print(f"Slump = {intercept_slump:.2f} + {slope_slump:.2f} × Rubber_Content")
        print(f"  R² = {r_slump**2:.3f}, p-value = {p_slump:.4f}")
        
        # Air content vs rubber content
        slope_air, intercept_air, r_air, p_air, se_air = stats.linregress(
            df_combined['rubber_content'], df_combined['air_content'])
        print(f"Air_Content = {intercept_air:.2f} + {slope_air:.3f} × Rubber_Content")
        print(f"  R² = {r_air**2:.3f}, p-value = {p_air:.4f}")
        
        # Unit weight vs rubber content
        slope_weight, intercept_weight, r_weight, p_weight, se_weight = stats.linregress(
            df_combined['rubber_content'], df_combined['unit_weight'])
        print(f"Unit_Weight = {intercept_weight:.0f} + {slope_weight:.1f} × Rubber_Content")
        print(f"  R² = {r_weight**2:.3f}, p-value = {p_weight:.4f}")
        
        # ANOVA for rubber size effect
        print("\\nANOVA ANALYSIS - Effect of Rubber Size")
        print("-" * 40)
        
        from scipy.stats import f_oneway
        
        # Group data by rubber size (excluding N/A)
        size_groups = df_combined[df_combined['rubber_size'] != 'N/A'].groupby('rubber_size')
        
        if len(size_groups) > 1:
            slump_groups = [group['slump'].values for name, group in size_groups]
            air_groups = [group['air_content'].values for name, group in size_groups]
            weight_groups = [group['unit_weight'].values for name, group in size_groups]
            
            f_slump, p_slump = f_oneway(*slump_groups)
            f_air, p_air = f_oneway(*air_groups)
            f_weight, p_weight = f_oneway(*weight_groups)
            
            print(f"Slump: F = {f_slump:.3f}, p-value = {p_slump:.4f}")
            print(f"Air Content: F = {f_air:.3f}, p-value = {p_air:.4f}")
            print(f"Unit Weight: F = {f_weight:.3f}, p-value = {p_weight:.4f}")
        
        # Quality control limits
        print("\\nQUALITY CONTROL LIMITS (Mean ± 2σ)")
        print("-" * 40)
        
        for prop in ['slump', 'air_content', 'unit_weight']:
            mean_val = df_combined[prop].mean()
            std_val = df_combined[prop].std()
            lcl = mean_val - 2 * std_val
            ucl = mean_val + 2 * std_val
            print(f"{prop.replace('_', ' ').title()}: {lcl:.1f} - {ucl:.1f}")
    
    def export_processed_data(self):
        """Export processed data for further analysis"""
        print("\\n" + "="*60)
        print("EXPORTING PROCESSED DATA")
        print("="*60)
        
        df_mix = self.create_mix_design_dataframe()
        df_fresh = self.create_fresh_properties_dataframe()
        df_combined = df_fresh.merge(df_mix, on='mix_id')
        
        # Export to CSV
        output_dir = self.data_path / '04_analysis_scripts'
        output_dir.mkdir(exist_ok=True)
        
        df_mix.to_csv(output_dir / 'mix_designs.csv', index=False)
        df_fresh.to_csv(output_dir / 'fresh_properties.csv', index=False)
        df_combined.to_csv(output_dir / 'combined_dataset.csv', index=False)
        
        print("✓ Exported mix_designs.csv")
        print("✓ Exported fresh_properties.csv") 
        print("✓ Exported combined_dataset.csv")
        
        # Create summary statistics file
        summary_stats = {
            'mix_design_summary': df_mix.describe().to_dict(),
            'fresh_properties_summary': df_fresh.describe().to_dict(),
            'correlations': df_combined[['rubber_content', 'slump', 'air_content', 
                                      'unit_weight']].corr().to_dict()
        }
        
        with open(output_dir / 'summary_statistics.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print("✓ Exported summary_statistics.json")
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("FIRE-RESISTANT RUBBERIZED CONCRETE DATASET ANALYSIS")
        print("=" * 80)
        print("Phase 1: Material Characterization & Specimen Preparation")
        print("=" * 80)
        
        self.analyze_constituent_materials()
        self.analyze_mix_designs()
        self.analyze_fresh_properties()
        self.create_visualizations()
        self.generate_statistical_report()
        self.export_processed_data()
        
        print("\\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("All data has been analyzed and visualizations created.")
        print("Check the output files in the 04_analysis_scripts directory.")


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = RubberizedConcreteAnalyzer()
    analyzer.run_complete_analysis()