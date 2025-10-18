#!/usr/bin/env python3
"""
Data Analysis Script for Fire-Resistant Structural Elements Using High-Performance Rubberized Concrete
Generated Synthetic Dataset Analysis and Visualization Tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class RubberizedConcreteAnalyzer:
    def __init__(self, data_dir="/workspace"):
        """Initialize the analyzer with data directory"""
        self.data_dir = Path(data_dir)
        self.load_data()
        
    def load_data(self):
        """Load all CSV datasets"""
        try:
            self.mix_proportions = pd.read_csv(self.data_dir / "mix_proportions_fresh_properties.csv")
            self.mechanical_props = pd.read_csv(self.data_dir / "mechanical_properties.csv")
            self.thermal_props = pd.read_csv(self.data_dir / "thermal_properties.csv")
            self.specimen_data = pd.read_csv(self.data_dir / "specimen_preparation_testing.csv")
            self.validation_data = pd.read_csv(self.data_dir / "data_validation_qa.csv")
            
            # Add CR_Content_% column to mechanical_props by merging with mix_proportions
            self.mechanical_props = self.mechanical_props.merge(
                self.mix_proportions[['Mix_ID', 'CR_Content_%']], 
                on='Mix_ID', 
                how='left'
            )
            
            # Add CR_Content_% column to thermal_props by merging with mix_proportions
            self.thermal_props = self.thermal_props.merge(
                self.mix_proportions[['Mix_ID', 'CR_Content_%']], 
                on='Mix_ID', 
                how='left'
            )
            
            print("✓ All datasets loaded successfully")
            print(f"Mechanical props columns: {list(self.mechanical_props.columns)}")
            print(f"Thermal props columns: {list(self.thermal_props.columns)}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*60)
        print("SYNTHETIC DATASET SUMMARY STATISTICS")
        print("="*60)
        
        # Mix proportions summary
        print(f"\n1. MIX DESIGNS: {len(self.mix_proportions)} total mixes")
        print(f"   - Control mixes: {len(self.mix_proportions[self.mix_proportions['CR_Content_%'] == 0])}")
        print(f"   - Rubberized mixes: {len(self.mix_proportions[self.mix_proportions['CR_Content_%'] > 0])}")
        print(f"   - Rubber content range: {self.mix_proportions['CR_Content_%'].min()}% - {self.mix_proportions['CR_Content_%'].max()}%")
        
        # Mechanical properties summary
        print(f"\n2. MECHANICAL PROPERTIES: {len(self.mechanical_props)} data points")
        print(f"   - Testing ages: {sorted(self.mechanical_props['Age_days'].unique())} days")
        print(f"   - Compressive strength range: {self.mechanical_props['Compressive_Strength_MPa'].min():.1f} - {self.mechanical_props['Compressive_Strength_MPa'].max():.1f} MPa")
        print(f"   - Modulus of elasticity range: {self.mechanical_props['Modulus_of_Elasticity_GPa'].min():.1f} - {self.mechanical_props['Modulus_of_Elasticity_GPa'].max():.1f} GPa")
        
        # Thermal properties summary
        print(f"\n3. THERMAL PROPERTIES: {len(self.thermal_props)} data points")
        print(f"   - Temperature range: {self.thermal_props['Temperature_C'].min()}°C - {self.thermal_props['Temperature_C'].max()}°C")
        print(f"   - Thermal conductivity range: {self.thermal_props['Thermal_Conductivity_W_mK'].min():.2f} - {self.thermal_props['Thermal_Conductivity_W_mK'].max():.2f} W/m·K")
        print(f"   - Fire resistance ratings: {sorted(self.thermal_props['Fire_Resistance_Rating_min'].unique())} minutes")
        
        # Specimen data summary
        print(f"\n4. SPECIMEN DATA: {len(self.specimen_data)} specimens")
        print(f"   - Specimen types: {self.specimen_data['Specimen_Type'].unique()}")
        print(f"   - Density range: {self.specimen_data['Density_kg_m3'].min()} - {self.specimen_data['Density_kg_m3'].max()} kg/m³")
        
    def analyze_rubber_effects(self):
        """Analyze the effects of rubber content and particle size"""
        print("\n" + "="*60)
        print("RUBBER CONTENT AND PARTICLE SIZE EFFECTS ANALYSIS")
        print("="*60)
        
        # Group by rubber content
        rubber_effects = self.mix_proportions.groupby('CR_Content_%').agg({
            'Fresh_Density_kg_m3': 'mean',
            'Slump_mm': 'mean',
            'Air_Content_%': 'mean'
        }).round(2)
        
        print("\n1. FRESH PROPERTIES vs RUBBER CONTENT:")
        print(rubber_effects)
        
        # Mechanical properties analysis
        mech_analysis = self.mechanical_props.groupby(['CR_Content_%', 'Age_days']).agg({
            'Compressive_Strength_MPa': 'mean',
            'Modulus_of_Elasticity_GPa': 'mean'
        }).round(2)
        
        print("\n2. MECHANICAL PROPERTIES vs RUBBER CONTENT (28-day):")
        mech_28d = mech_analysis[mech_analysis.index.get_level_values('Age_days') == 28]
        print(mech_28d)
        
        # Thermal properties analysis
        thermal_analysis = self.thermal_props.groupby(['CR_Content_%', 'Temperature_C']).agg({
            'Thermal_Conductivity_W_mK': 'mean',
            'Fire_Resistance_Rating_min': 'mean'
        }).round(3)
        
        print("\n3. THERMAL PROPERTIES vs RUBBER CONTENT (20°C):")
        thermal_20c = thermal_analysis[thermal_analysis.index.get_level_values('Temperature_C') == 20]
        print(thermal_20c)
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Compressive strength vs age
        ax1 = plt.subplot(2, 3, 1)
        for mix in self.mechanical_props['Mix_ID'].unique():
            mix_data = self.mechanical_props[self.mechanical_props['Mix_ID'] == mix]
            ax1.plot(mix_data['Age_days'], mix_data['Compressive_Strength_MPa'], 
                    marker='o', label=mix, linewidth=2)
        ax1.set_xlabel('Age (days)')
        ax1.set_ylabel('Compressive Strength (MPa)')
        ax1.set_title('Compressive Strength Development')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Fresh density vs rubber content
        ax2 = plt.subplot(2, 3, 2)
        density_data = self.mix_proportions.groupby('CR_Content_%')['Fresh_Density_kg_m3'].mean()
        ax2.plot(density_data.index, density_data.values, 'bo-', linewidth=2, markersize=8)
        ax2.set_xlabel('Rubber Content (%)')
        ax2.set_ylabel('Fresh Density (kg/m³)')
        ax2.set_title('Fresh Density vs Rubber Content')
        ax2.grid(True, alpha=0.3)
        
        # 3. Thermal conductivity vs temperature
        ax3 = plt.subplot(2, 3, 3)
        for mix in self.thermal_props['Mix_ID'].unique():
            mix_data = self.thermal_props[self.thermal_props['Mix_ID'] == mix]
            ax3.plot(mix_data['Temperature_C'], mix_data['Thermal_Conductivity_W_mK'], 
                    marker='s', label=mix, linewidth=2)
        ax3.set_xlabel('Temperature (°C)')
        ax3.set_ylabel('Thermal Conductivity (W/m·K)')
        ax3.set_title('Thermal Conductivity vs Temperature')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Fire resistance vs rubber content
        ax4 = plt.subplot(2, 3, 4)
        fire_resistance = self.thermal_props.groupby('CR_Content_%')['Fire_Resistance_Rating_min'].mean()
        ax4.bar(fire_resistance.index, fire_resistance.values, color='red', alpha=0.7)
        ax4.set_xlabel('Rubber Content (%)')
        ax4.set_ylabel('Fire Resistance Rating (min)')
        ax4.set_title('Fire Resistance vs Rubber Content')
        ax4.grid(True, alpha=0.3)
        
        # 5. Modulus of elasticity vs rubber content
        ax5 = plt.subplot(2, 3, 5)
        mod_data = self.mechanical_props[self.mechanical_props['Age_days'] == 28]
        mod_analysis = mod_data.groupby('CR_Content_%')['Modulus_of_Elasticity_GPa'].mean()
        ax5.plot(mod_analysis.index, mod_analysis.values, 'go-', linewidth=2, markersize=8)
        ax5.set_xlabel('Rubber Content (%)')
        ax5.set_ylabel('Modulus of Elasticity (GPa)')
        ax5.set_title('Modulus of Elasticity vs Rubber Content (28-day)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Workability vs rubber content
        ax6 = plt.subplot(2, 3, 6)
        workability_data = self.mix_proportions.groupby('CR_Content_%')['Slump_mm'].mean()
        ax6.plot(workability_data.index, workability_data.values, 'mo-', linewidth=2, markersize=8)
        ax6.set_xlabel('Rubber Content (%)')
        ax6.set_ylabel('Slump (mm)')
        ax6.set_title('Workability vs Rubber Content')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.data_dir / 'synthetic_dataset_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved as 'synthetic_dataset_analysis.png'")
        
    def export_analysis_results(self):
        """Export analysis results to files"""
        print("\n" + "="*60)
        print("EXPORTING ANALYSIS RESULTS")
        print("="*60)
        
        # Export summary statistics
        summary_stats = {
            'dataset_info': {
                'total_mixes': len(self.mix_proportions),
                'total_specimens': len(self.specimen_data),
                'total_mechanical_tests': len(self.mechanical_props),
                'total_thermal_tests': len(self.thermal_props)
            },
            'rubber_content_effects': self.mix_proportions.groupby('CR_Content_%').agg({
                'Fresh_Density_kg_m3': 'mean',
                'Slump_mm': 'mean',
                'Air_Content_%': 'mean'
            }).to_dict(),
            'mechanical_properties_28d': self.mechanical_props[
                self.mechanical_props['Age_days'] == 28
            ].groupby('CR_Content_%').agg({
                'Compressive_Strength_MPa': 'mean',
                'Modulus_of_Elasticity_GPa': 'mean'
            }).to_dict(),
            'thermal_properties_20c': self.thermal_props[
                self.thermal_props['Temperature_C'] == 20
            ].groupby('CR_Content_%').agg({
                'Thermal_Conductivity_W_mK': 'mean',
                'Fire_Resistance_Rating_min': 'mean'
            }).to_dict()
        }
        
        with open(self.data_dir / 'analysis_results.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print("✓ Analysis results exported to 'analysis_results.json'")
        
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting comprehensive analysis of synthetic rubberized concrete dataset...")
        
        self.generate_summary_statistics()
        self.analyze_rubber_effects()
        self.create_visualizations()
        self.export_analysis_results()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("  - synthetic_dataset_analysis.png (visualizations)")
        print("  - analysis_results.json (summary statistics)")
        print("  - All original CSV datasets")
        print("  - complete_dataset.json (comprehensive JSON format)")

if __name__ == "__main__":
    analyzer = RubberizedConcreteAnalyzer()
    analyzer.run_complete_analysis()