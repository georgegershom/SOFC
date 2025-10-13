#!/usr/bin/env python3
"""
Geotechnical Data Analysis Toolkit
===================================

A comprehensive toolkit for analyzing geotechnical datasets related to
failure mechanisms of underground structures in sandy and clay soils.

Author: PhD Research Support
Date: October 2025
Version: 1.0
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class GeotechnicalDataAnalyzer:
    """Main class for geotechnical data analysis"""
    
    def __init__(self, data_directory='./'):
        """
        Initialize the analyzer with dataset directory
        
        Parameters:
        -----------
        data_directory : str
            Path to directory containing CSV files
        """
        self.data_dir = Path(data_directory)
        self.datasets = {}
        
    def load_all_datasets(self):
        """Load all available datasets"""
        dataset_files = {
            'sandy_properties': 'sandy_soils_properties.csv',
            'clay_properties': 'clay_soils_properties.csv',
            'sandy_liquefaction': 'sandy_soil_liquefaction_cases.csv',
            'clay_failures': 'clay_soil_failure_cases.csv',
            'mechanical_comparison': 'mechanical_properties_comparison.csv',
            'grain_size': 'grain_size_distribution_data.csv',
            'groundwater': 'groundwater_and_pore_pressure.csv'
        }
        
        for name, filename in dataset_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                self.datasets[name] = pd.read_csv(filepath)
                print(f"✓ Loaded {name}: {len(self.datasets[name])} records")
            else:
                print(f"✗ Could not find {filename}")
        
        return self.datasets
    
    def sandy_soil_summary(self):
        """Generate summary statistics for sandy soils"""
        if 'sandy_properties' not in self.datasets:
            print("Sandy soil properties dataset not loaded!")
            return None
        
        df = self.datasets['sandy_properties']
        
        summary = {
            'Friction Angle': {
                'Mean': df['Friction_Angle_degrees'].mean(),
                'Std': df['Friction_Angle_degrees'].std(),
                'Min': df['Friction_Angle_degrees'].min(),
                'Max': df['Friction_Angle_degrees'].max()
            },
            'Relative Density': {
                'Mean': df['Relative_Density_percent'].mean(),
                'Std': df['Relative_Density_percent'].std(),
                'Min': df['Relative_Density_percent'].min(),
                'Max': df['Relative_Density_percent'].max()
            },
            'SPT N-Value': {
                'Mean': df['SPT_N_Value'].mean(),
                'Std': df['SPT_N_Value'].std(),
                'Min': df['SPT_N_Value'].min(),
                'Max': df['SPT_N_Value'].max()
            },
            'Sand Content': {
                'Mean': df['Sand_Content_percent'].mean(),
                'Std': df['Sand_Content_percent'].std(),
                'Min': df['Sand_Content_percent'].min(),
                'Max': df['Sand_Content_percent'].max()
            }
        }
        
        print("\n" + "="*60)
        print("SANDY SOIL PROPERTIES SUMMARY")
        print("="*60)
        for param, stats_dict in summary.items():
            print(f"\n{param}:")
            for stat, value in stats_dict.items():
                print(f"  {stat:8s}: {value:8.2f}")
        
        return pd.DataFrame(summary).T
    
    def clay_soil_summary(self):
        """Generate summary statistics for clay soils"""
        if 'clay_properties' not in self.datasets:
            print("Clay soil properties dataset not loaded!")
            return None
        
        df = self.datasets['clay_properties']
        
        summary = {
            'Liquid Limit': {
                'Mean': df['Liquid_Limit_percent'].mean(),
                'Std': df['Liquid_Limit_percent'].std(),
                'Min': df['Liquid_Limit_percent'].min(),
                'Max': df['Liquid_Limit_percent'].max()
            },
            'Plasticity Index': {
                'Mean': df['Plasticity_Index'].mean(),
                'Std': df['Plasticity_Index'].std(),
                'Min': df['Plasticity_Index'].min(),
                'Max': df['Plasticity_Index'].max()
            },
            'Undrained Shear Strength': {
                'Mean': df['Undrained_Shear_Strength_kPa'].mean(),
                'Std': df['Undrained_Shear_Strength_kPa'].std(),
                'Min': df['Undrained_Shear_Strength_kPa'].min(),
                'Max': df['Undrained_Shear_Strength_kPa'].max()
            },
            'Sensitivity': {
                'Mean': df['Sensitivity'].mean(),
                'Std': df['Sensitivity'].std(),
                'Min': df['Sensitivity'].min(),
                'Max': df['Sensitivity'].max()
            },
            'OCR': {
                'Mean': df['Overconsolidation_Ratio'].mean(),
                'Std': df['Overconsolidation_Ratio'].std(),
                'Min': df['Overconsolidation_Ratio'].min(),
                'Max': df['Overconsolidation_Ratio'].max()
            }
        }
        
        print("\n" + "="*60)
        print("CLAY SOIL PROPERTIES SUMMARY")
        print("="*60)
        for param, stats_dict in summary.items():
            print(f"\n{param}:")
            for stat, value in stats_dict.items():
                print(f"  {stat:8s}: {value:8.2f}")
        
        return pd.DataFrame(summary).T
    
    def plot_sandy_soil_correlations(self, save_fig=False):
        """Plot key correlations for sandy soils"""
        if 'sandy_properties' not in self.datasets:
            print("Sandy soil properties dataset not loaded!")
            return
        
        df = self.datasets['sandy_properties']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Relative Density vs SPT N-value
        axes[0, 0].scatter(df['Relative_Density_percent'], df['SPT_N_Value'], 
                          alpha=0.6, s=100, c='blue')
        axes[0, 0].set_xlabel('Relative Density (%)', fontsize=12)
        axes[0, 0].set_ylabel('SPT N-Value', fontsize=12)
        axes[0, 0].set_title('Relative Density vs SPT N-Value', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Friction Angle vs Relative Density
        axes[0, 1].scatter(df['Relative_Density_percent'], df['Friction_Angle_degrees'],
                          alpha=0.6, s=100, c='green')
        axes[0, 1].set_xlabel('Relative Density (%)', fontsize=12)
        axes[0, 1].set_ylabel('Friction Angle (degrees)', fontsize=12)
        axes[0, 1].set_title('Friction Angle vs Relative Density', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # D50 vs Permeability
        axes[1, 0].scatter(df['D50_mm'], df['Permeability_m_s'], 
                          alpha=0.6, s=100, c='red')
        axes[1, 0].set_xlabel('D50 (mm)', fontsize=12)
        axes[1, 0].set_ylabel('Permeability (m/s)', fontsize=12)
        axes[1, 0].set_title('Grain Size vs Permeability', fontsize=14, fontweight='bold')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sand Content Distribution
        axes[1, 1].hist(df['Sand_Content_percent'], bins=15, 
                       alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Sand Content (%)', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Sand Content Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('sandy_soil_correlations.png', dpi=300, bbox_inches='tight')
            print("Figure saved as 'sandy_soil_correlations.png'")
        
        plt.show()
    
    def plot_clay_soil_plasticity_chart(self, save_fig=False):
        """Plot Casagrande plasticity chart for clay soils"""
        if 'clay_properties' not in self.datasets:
            print("Clay soil properties dataset not loaded!")
            return
        
        df = self.datasets['clay_properties']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot data points colored by clay mineral type
        mineral_types = df['Clay_Mineral_Type'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(mineral_types)))
        
        for i, mineral in enumerate(mineral_types):
            mask = df['Clay_Mineral_Type'] == mineral
            ax.scatter(df.loc[mask, 'Liquid_Limit_percent'], 
                      df.loc[mask, 'Plasticity_Index'],
                      label=mineral, alpha=0.7, s=100, c=[colors[i]])
        
        # Add A-line
        ll_range = np.linspace(0, 100, 100)
        a_line = 0.73 * (ll_range - 20)
        ax.plot(ll_range, a_line, 'k-', linewidth=2, label='A-line')
        
        # Add U-line
        u_line = 0.9 * (ll_range - 8)
        ax.plot(ll_range, u_line, 'k--', linewidth=1.5, label='U-line', alpha=0.7)
        
        ax.set_xlabel('Liquid Limit (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Plasticity Index', fontsize=14, fontweight='bold')
        ax.set_title('Casagrande Plasticity Chart with Mineralogy', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 60)
        
        # Add region labels
        ax.text(35, 5, 'CL-ML', fontsize=12, fontweight='bold', alpha=0.5)
        ax.text(50, 20, 'CL', fontsize=12, fontweight='bold', alpha=0.5)
        ax.text(70, 35, 'CH', fontsize=12, fontweight='bold', alpha=0.5)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('clay_plasticity_chart.png', dpi=300, bbox_inches='tight')
            print("Figure saved as 'clay_plasticity_chart.png'")
        
        plt.show()
    
    def analyze_liquefaction_cases(self):
        """Analyze liquefaction case studies"""
        if 'sandy_liquefaction' not in self.datasets:
            print("Liquefaction cases dataset not loaded!")
            return None
        
        df = self.datasets['sandy_liquefaction']
        
        print("\n" + "="*60)
        print("LIQUEFACTION CASE STUDIES ANALYSIS")
        print("="*60)
        
        print(f"\nTotal Cases: {len(df)}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nEarthquake Magnitude Range: {df['Earthquake_Magnitude'].min():.1f} to {df['Earthquake_Magnitude'].max():.1f}")
        print(f"Mean PGA: {df['Peak_Ground_Acceleration_g'].mean():.3f}g")
        
        print(f"\n--- Structural Response ---")
        print(f"Maximum Ground Settlement: {df['Max_Ground_Settlement_cm'].max():.1f} cm")
        print(f"Maximum Lateral Displacement: {df['Lateral_Displacement_cm'].max():.1f} cm")
        print(f"Maximum Uplift: {df['Uplift_Displacement_cm'].max():.1f} cm")
        
        print(f"\n--- Mitigation Effectiveness ---")
        mitigated = df['Mitigation_Present'].value_counts()
        print(mitigated)
        
        # Correlation between excess pore pressure ratio and damage
        corr = df['Excess_Pore_Pressure_Ratio'].corr(df['Max_Ground_Settlement_cm'])
        print(f"\nCorrelation (Excess PPR vs Settlement): {corr:.3f}")
        
        return df.describe()
    
    def analyze_clay_failures(self):
        """Analyze clay soil failure case studies"""
        if 'clay_failures' not in self.datasets:
            print("Clay failure cases dataset not loaded!")
            return None
        
        df = self.datasets['clay_failures']
        
        print("\n" + "="*60)
        print("CLAY SOIL FAILURE CASE STUDIES ANALYSIS")
        print("="*60)
        
        print(f"\nTotal Cases: {len(df)}")
        print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
        
        print(f"\n--- Failure Types ---")
        failure_types = df['Failure_Type'].value_counts()
        for failure_type, count in failure_types.items():
            print(f"  {failure_type}: {count}")
        
        print(f"\n--- Clay Types ---")
        clay_types = df['Clay_Type'].value_counts()
        for clay_type, count in clay_types.items():
            print(f"  {clay_type}: {count}")
        
        print(f"\n--- Damage Severity ---")
        damage_levels = df['Structure_Damage_Level'].value_counts()
        for level, count in damage_levels.items():
            print(f"  {level}: {count}")
        
        print(f"\n--- Key Metrics ---")
        print(f"Maximum Displacement: {df['Displacement_m'].max():.1f} m")
        print(f"Mean Sensitivity: {df['Sensitivity'].mean():.2f}")
        print(f"Mean Pore Pressure Ratio: {df['Pore_Pressure_Ratio'].mean():.3f}")
        
        # Progressive failure analysis
        progressive = df['Progressive_Failure'].value_counts()
        print(f"\n--- Progressive Failure Occurrence ---")
        print(progressive)
        
        return df.describe()
    
    def plot_liquefaction_analysis(self, save_fig=False):
        """Create comprehensive liquefaction analysis plots"""
        if 'sandy_liquefaction' not in self.datasets:
            print("Liquefaction cases dataset not loaded!")
            return
        
        df = self.datasets['sandy_liquefaction']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Excess PPR vs Settlement
        axes[0, 0].scatter(df['Excess_Pore_Pressure_Ratio'], 
                          df['Max_Ground_Settlement_cm'],
                          s=df['Earthquake_Magnitude']*20, 
                          alpha=0.6, c='red')
        axes[0, 0].set_xlabel('Excess Pore Pressure Ratio', fontsize=12)
        axes[0, 0].set_ylabel('Maximum Settlement (cm)', fontsize=12)
        axes[0, 0].set_title('Liquefaction Severity vs Settlement\n(Bubble size = Magnitude)', 
                            fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Relative Density vs Uplift
        axes[0, 1].scatter(df['Relative_Density_percent'], 
                          df['Uplift_Displacement_cm'],
                          alpha=0.6, s=100, c='blue')
        axes[0, 1].set_xlabel('Relative Density (%)', fontsize=12)
        axes[0, 1].set_ylabel('Uplift Displacement (cm)', fontsize=12)
        axes[0, 1].set_title('Relative Density vs Structural Uplift', 
                            fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Foundation Type Performance
        foundation_damage = df.groupby('Foundation_Type')['Structure_Damage_Level'].apply(
            lambda x: (x == 'Severe').sum()
        ).sort_values(ascending=False)
        axes[1, 0].barh(foundation_damage.index, foundation_damage.values, 
                       color='orange', alpha=0.7)
        axes[1, 0].set_xlabel('Number of Severe Damage Cases', fontsize=12)
        axes[1, 0].set_title('Severe Damage by Foundation Type', 
                            fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # Mitigation Effectiveness
        mitigated = df[df['Mitigation_Present'].str.startswith('Yes', na=False)]
        not_mitigated = df[df['Mitigation_Present'] == 'No']
        
        damage_comparison = pd.DataFrame({
            'With Mitigation': mitigated['Structure_Damage_Level'].value_counts(),
            'Without Mitigation': not_mitigated['Structure_Damage_Level'].value_counts()
        }).fillna(0)
        
        damage_comparison.plot(kind='bar', ax=axes[1, 1], 
                              color=['green', 'red'], alpha=0.7)
        axes[1, 1].set_xlabel('Damage Level', fontsize=12)
        axes[1, 1].set_ylabel('Number of Cases', fontsize=12)
        axes[1, 1].set_title('Mitigation Effectiveness', 
                            fontsize=12, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig('liquefaction_analysis.png', dpi=300, bbox_inches='tight')
            print("Figure saved as 'liquefaction_analysis.png'")
        
        plt.show()
    
    def export_summary_report(self, filename='analysis_summary.txt'):
        """Export comprehensive summary report"""
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("GEOTECHNICAL DATASETS COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Dataset overview
            f.write("DATASETS LOADED:\n")
            f.write("-" * 80 + "\n")
            for name, df in self.datasets.items():
                f.write(f"{name:30s}: {len(df):5d} records\n")
            f.write("\n")
            
            # Sandy soil summary
            if 'sandy_properties' in self.datasets:
                df = self.datasets['sandy_properties']
                f.write("\nSANDY SOIL PROPERTIES SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Friction Angle:    {df['Friction_Angle_degrees'].mean():.2f} ± {df['Friction_Angle_degrees'].std():.2f} degrees\n")
                f.write(f"Relative Density:  {df['Relative_Density_percent'].mean():.2f} ± {df['Relative_Density_percent'].std():.2f} %\n")
                f.write(f"SPT N-Value:       {df['SPT_N_Value'].mean():.2f} ± {df['SPT_N_Value'].std():.2f}\n")
                f.write(f"Permeability:      {df['Permeability_m_s'].mean():.2e} ± {df['Permeability_m_s'].std():.2e} m/s\n")
            
            # Clay soil summary
            if 'clay_properties' in self.datasets:
                df = self.datasets['clay_properties']
                f.write("\nCLAY SOIL PROPERTIES SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Liquid Limit:              {df['Liquid_Limit_percent'].mean():.2f} ± {df['Liquid_Limit_percent'].std():.2f} %\n")
                f.write(f"Plasticity Index:          {df['Plasticity_Index'].mean():.2f} ± {df['Plasticity_Index'].std():.2f}\n")
                f.write(f"Undrained Shear Strength:  {df['Undrained_Shear_Strength_kPa'].mean():.2f} ± {df['Undrained_Shear_Strength_kPa'].std():.2f} kPa\n")
                f.write(f"Sensitivity:               {df['Sensitivity'].mean():.2f} ± {df['Sensitivity'].std():.2f}\n")
                f.write(f"OCR:                       {df['Overconsolidation_Ratio'].mean():.2f} ± {df['Overconsolidation_Ratio'].std():.2f}\n")
            
            # Liquefaction cases
            if 'sandy_liquefaction' in self.datasets:
                df = self.datasets['sandy_liquefaction']
                f.write("\nLIQUEFACTION CASE STUDIES SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Cases:             {len(df)}\n")
                f.write(f"Magnitude Range:         {df['Earthquake_Magnitude'].min():.1f} - {df['Earthquake_Magnitude'].max():.1f}\n")
                f.write(f"Max Settlement:          {df['Max_Ground_Settlement_cm'].max():.1f} cm\n")
                f.write(f"Max Uplift:              {df['Uplift_Displacement_cm'].max():.1f} cm\n")
                f.write(f"With Mitigation:         {(df['Mitigation_Present'] != 'No').sum()} cases\n")
            
            # Clay failures
            if 'clay_failures' in self.datasets:
                df = self.datasets['clay_failures']
                f.write("\nCLAY SOIL FAILURE CASE STUDIES SUMMARY:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Total Cases:             {len(df)}\n")
                f.write(f"Max Displacement:        {df['Displacement_m'].max():.1f} m\n")
                f.write(f"Progressive Failures:    {(df['Progressive_Failure'] == 'Yes').sum()} cases\n")
                f.write(f"Severe Damage Cases:     {(df['Structure_Damage_Level'] == 'Severe').sum()}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("End of Report\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary report exported to '{filename}'")


def main():
    """Main execution function"""
    print("="*80)
    print("GEOTECHNICAL DATA ANALYSIS TOOLKIT")
    print("="*80)
    print("\nInitializing analyzer...")
    
    # Create analyzer instance
    analyzer = GeotechnicalDataAnalyzer()
    
    # Load all datasets
    print("\nLoading datasets...")
    analyzer.load_all_datasets()
    
    # Generate summaries
    print("\n" + "="*80)
    analyzer.sandy_soil_summary()
    analyzer.clay_soil_summary()
    
    # Analyze case studies
    analyzer.analyze_liquefaction_cases()
    analyzer.analyze_clay_failures()
    
    # Generate plots
    print("\n" + "="*80)
    print("Generating visualizations...")
    print("="*80)
    
    try:
        analyzer.plot_sandy_soil_correlations(save_fig=True)
        analyzer.plot_clay_soil_plasticity_chart(save_fig=True)
        analyzer.plot_liquefaction_analysis(save_fig=True)
        print("\n✓ All visualizations generated successfully!")
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Export summary report
    print("\nExporting comprehensive summary report...")
    analyzer.export_summary_report()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - sandy_soil_correlations.png")
    print("  - clay_plasticity_chart.png")
    print("  - liquefaction_analysis.png")
    print("  - analysis_summary.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
