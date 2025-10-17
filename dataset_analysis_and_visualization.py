#!/usr/bin/env python3
"""
Rubberized Concrete Baseline Dataset Analysis and Visualization
Comprehensive analysis tools for the generated dataset

Author: AI Research Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    def __init__(self, data_dir="/workspace"):
        """Initialize the dataset analyzer."""
        self.data_dir = data_dir
        self.fresh_data = None
        self.mechanical_data = None
        self.physical_data = None
        self.mixture_data = None
        self.rubber_data = None
        
        # Load all datasets
        self.load_datasets()
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_datasets(self):
        """Load all dataset files."""
        print("Loading datasets...")
        
        # Load CSV files
        self.fresh_data = pd.read_csv(f"{self.data_dir}/fresh_state_properties.csv")
        self.mechanical_data = pd.read_csv(f"{self.data_dir}/mechanical_properties.csv")
        self.physical_data = pd.read_csv(f"{self.data_dir}/physical_properties.csv")
        
        # Load JSON files
        with open(f"{self.data_dir}/mixture_proportions.json", 'r') as f:
            self.mixture_data = json.load(f)
        
        with open(f"{self.data_dir}/rubber_characterization.json", 'r') as f:
            self.rubber_data = json.load(f)
        
        print("Datasets loaded successfully!")
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print("\n" + "="*60)
        print("COMPREHENSIVE DATASET SUMMARY")
        print("="*60)
        
        # Fresh state properties summary
        print("\n1. FRESH STATE PROPERTIES SUMMARY")
        print("-" * 40)
        fresh_summary = self.fresh_data.groupby('rubber_replacement').agg({
            'slump_flow': ['count', 'mean', 'std', 'min', 'max'],
            'air_content': ['mean', 'std', 'min', 'max'],
            'fresh_density': ['mean', 'std', 'min', 'max']
        }).round(2)
        print(fresh_summary)
        
        # Mechanical properties summary
        print("\n2. MECHANICAL PROPERTIES SUMMARY")
        print("-" * 40)
        mech_summary = self.mechanical_data.groupby(['rubber_replacement', 'age_days']).agg({
            'compressive_strength': ['count', 'mean', 'std', 'min', 'max'],
            'tensile_splitting_strength': ['mean', 'std', 'min', 'max'],
            'modulus_elasticity': ['mean', 'std', 'min', 'max']
        }).round(2)
        print(mech_summary)
        
        # Physical properties summary
        print("\n3. PHYSICAL PROPERTIES SUMMARY")
        print("-" * 40)
        phys_summary = self.physical_data.groupby('rubber_replacement').agg({
            'oven_dry_density': ['count', 'mean', 'std', 'min', 'max'],
            'ssd_density': ['mean', 'std', 'min', 'max'],
            'porosity': ['mean', 'std', 'min', 'max'],
            'ultrasonic_pulse_velocity': ['mean', 'std', 'min', 'max']
        }).round(2)
        print(phys_summary)
        
        return {
            'fresh_summary': fresh_summary,
            'mechanical_summary': mech_summary,
            'physical_summary': phys_summary
        }
    
    def create_fresh_state_plots(self):
        """Create comprehensive plots for fresh state properties."""
        print("\nGenerating fresh state property plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fresh State Properties vs Rubber Replacement Level', fontsize=16, fontweight='bold')
        
        # Slump flow
        sns.boxplot(data=self.fresh_data, x='rubber_replacement', y='slump_flow', ax=axes[0,0])
        axes[0,0].set_title('Slump Flow vs Rubber Content')
        axes[0,0].set_xlabel('Rubber Replacement (%)')
        axes[0,0].set_ylabel('Slump Flow (mm)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Air content
        sns.boxplot(data=self.fresh_data, x='rubber_replacement', y='air_content', ax=axes[0,1])
        axes[0,1].set_title('Air Content vs Rubber Content')
        axes[0,1].set_xlabel('Rubber Replacement (%)')
        axes[0,1].set_ylabel('Air Content (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Fresh density
        sns.boxplot(data=self.fresh_data, x='rubber_replacement', y='fresh_density', ax=axes[1,0])
        axes[1,0].set_title('Fresh Density vs Rubber Content')
        axes[1,0].set_xlabel('Rubber Replacement (%)')
        axes[1,0].set_ylabel('Fresh Density (kg/m³)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Workability rating
        workability_counts = self.fresh_data.groupby(['rubber_replacement', 'workability_rating']).size().unstack(fill_value=0)
        workability_counts.plot(kind='bar', stacked=True, ax=axes[1,1])
        axes[1,1].set_title('Workability Rating Distribution')
        axes[1,1].set_xlabel('Rubber Replacement (%)')
        axes[1,1].set_ylabel('Number of Tests')
        axes[1,1].legend(title='Workability Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/fresh_state_properties_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_mechanical_properties_plots(self):
        """Create comprehensive plots for mechanical properties."""
        print("\nGenerating mechanical properties plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Mechanical Properties vs Rubber Replacement Level', fontsize=16, fontweight='bold')
        
        # Compressive strength at 28 days
        mech_28d = self.mechanical_data[self.mechanical_data['age_days'] == 28]
        sns.boxplot(data=mech_28d, x='rubber_replacement', y='compressive_strength', ax=axes[0,0])
        axes[0,0].set_title('28-Day Compressive Strength vs Rubber Content')
        axes[0,0].set_xlabel('Rubber Replacement (%)')
        axes[0,0].set_ylabel('Compressive Strength (MPa)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Compressive strength development
        for replacement in sorted(self.mechanical_data['rubber_replacement'].unique()):
            subset = self.mechanical_data[self.mechanical_data['rubber_replacement'] == replacement]
            strength_by_age = subset.groupby('age_days')['compressive_strength'].mean()
            axes[0,1].plot(strength_by_age.index, strength_by_age.values, 
                          marker='o', linewidth=2, label=f'{replacement}% Rubber')
        axes[0,1].set_title('Compressive Strength Development')
        axes[0,1].set_xlabel('Age (days)')
        axes[0,1].set_ylabel('Compressive Strength (MPa)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Tensile splitting strength
        sns.boxplot(data=mech_28d, x='rubber_replacement', y='tensile_splitting_strength', ax=axes[1,0])
        axes[1,0].set_title('28-Day Tensile Splitting Strength vs Rubber Content')
        axes[1,0].set_xlabel('Rubber Replacement (%)')
        axes[1,0].set_ylabel('Tensile Splitting Strength (MPa)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Modulus of elasticity
        sns.boxplot(data=mech_28d, x='rubber_replacement', y='modulus_elasticity', ax=axes[1,1])
        axes[1,1].set_title('28-Day Modulus of Elasticity vs Rubber Content')
        axes[1,1].set_xlabel('Rubber Replacement (%)')
        axes[1,1].set_ylabel('Modulus of Elasticity (GPa)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/mechanical_properties_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_physical_properties_plots(self):
        """Create comprehensive plots for physical properties."""
        print("\nGenerating physical properties plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Physical Properties vs Rubber Replacement Level', fontsize=16, fontweight='bold')
        
        # Density comparison
        sns.boxplot(data=self.physical_data, x='rubber_replacement', y='oven_dry_density', ax=axes[0,0])
        axes[0,0].set_title('Oven-Dry Density vs Rubber Content')
        axes[0,0].set_xlabel('Rubber Replacement (%)')
        axes[0,0].set_ylabel('Oven-Dry Density (kg/m³)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Porosity
        sns.boxplot(data=self.physical_data, x='rubber_replacement', y='porosity', ax=axes[0,1])
        axes[0,1].set_title('Porosity vs Rubber Content')
        axes[0,1].set_xlabel('Rubber Replacement (%)')
        axes[0,1].set_ylabel('Porosity (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Ultrasonic pulse velocity
        sns.boxplot(data=self.physical_data, x='rubber_replacement', y='ultrasonic_pulse_velocity', ax=axes[1,0])
        axes[1,0].set_title('Ultrasonic Pulse Velocity vs Rubber Content')
        axes[1,0].set_xlabel('Rubber Replacement (%)')
        axes[1,0].set_ylabel('UPV (m/s)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Water absorption
        sns.boxplot(data=self.physical_data, x='rubber_replacement', y='water_absorption', ax=axes[1,1])
        axes[1,1].set_title('Water Absorption vs Rubber Content')
        axes[1,1].set_xlabel('Rubber Replacement (%)')
        axes[1,1].set_ylabel('Water Absorption (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/physical_properties_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_correlation_analysis(self):
        """Create correlation analysis between properties."""
        print("\nGenerating correlation analysis...")
        
        # Prepare data for correlation analysis
        correlation_data = self.fresh_data.merge(
            self.mechanical_data[self.mechanical_data['age_days'] == 28], 
            on=['mix_id', 'rubber_replacement'], 
            how='inner'
        ).merge(
            self.physical_data, 
            on=['mix_id', 'rubber_replacement'], 
            how='inner'
        )
        
        # Select numeric columns for correlation
        numeric_cols = [
            'rubber_replacement', 'slump_flow', 'air_content', 'fresh_density',
            'compressive_strength', 'tensile_splitting_strength', 'modulus_elasticity',
            'oven_dry_density', 'porosity', 'ultrasonic_pulse_velocity', 'water_absorption'
        ]
        
        corr_data = correlation_data[numeric_cols]
        correlation_matrix = corr_data.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8}, fmt='.2f')
        plt.title('Property Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def generate_rubber_characterization_report(self):
        """Generate detailed rubber characterization report."""
        print("\n" + "="*60)
        print("RUBBER AGGREGATE CHARACTERIZATION REPORT")
        print("="*60)
        
        print("\n1. RUBBER SOURCE AND TYPE")
        print("-" * 40)
        print(f"Source: {self.rubber_data[0]['source']}")
        print(f"Manufacturer: {self.rubber_data[0]['manufacturer']}")
        print(f"Particle Size Distribution:")
        for size, percentage in self.rubber_data[0]['particle_size_distribution'].items():
            print(f"  {size}: {percentage}%")
        
        print("\n2. PHYSICAL PROPERTIES")
        print("-" * 40)
        phys_props = self.rubber_data[0]['physical_properties']
        for prop, value in phys_props.items():
            if isinstance(value, float):
                print(f"{prop.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{prop.replace('_', ' ').title()}: {value}")
        
        print("\n3. CHEMICAL COMPOSITION")
        print("-" * 40)
        chem_comp = self.rubber_data[0]['chemical_composition']
        for component, percentage in chem_comp.items():
            print(f"{component.replace('_', ' ').title()}: {percentage}%")
        
        print("\n4. THERMAL PROPERTIES")
        print("-" * 40)
        thermal_props = self.rubber_data[0]['thermal_properties']
        for prop, value in thermal_props.items():
            print(f"{prop.replace('_', ' ').title()}: {value}°C")
        
        print("\n5. PRE-TREATMENT PROCEDURE")
        print("-" * 40)
        pretreatment = self.rubber_data[0]['pre_treatment']
        for step, value in pretreatment.items():
            print(f"{step.replace('_', ' ').title()}: {value}")
    
    def generate_mixture_design_report(self):
        """Generate detailed mixture design report."""
        print("\n" + "="*60)
        print("CONCRETE MIXTURE DESIGN REPORT")
        print("="*60)
        
        for mix_id, mix in self.mixture_data.items():
            print(f"\n{mix_id.upper()}")
            print("-" * 40)
            print(f"Cement Type: {mix['cement_type']}")
            print(f"Cement Content: {mix['cement_content']} kg/m³")
            print(f"Water-Cement Ratio: {mix['water_cement_ratio']}")
            print(f"Coarse Aggregate: {mix['coarse_aggregate']['content']} kg/m³ ({mix['coarse_aggregate']['type']})")
            print(f"Fine Aggregate: {mix['fine_aggregate']['content']} kg/m³ ({mix['fine_aggregate']['type']})")
            print(f"Water: {mix['water']['content']} kg/m³")
            print(f"Superplasticizer: {mix['superplasticizer']['content']} kg/m³ ({mix['superplasticizer']['dosage']}%)")
            
            if 'rubber_aggregate' in mix:
                rubber = mix['rubber_aggregate']
                print(f"Rubber Aggregate: {rubber['content']:.1f} kg/m³ ({rubber['replacement_level']}% replacement)")
                print(f"  Type: {rubber['type']}")
                print(f"  Particle Size: {rubber['particle_size']}")
                print(f"  Specific Gravity: {rubber['specific_gravity']}")
    
    def create_comprehensive_report(self):
        """Create a comprehensive analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RUBBERIZED CONCRETE BASELINE DATASET ANALYSIS")
        print("="*80)
        
        # Generate all analyses
        summaries = self.generate_summary_statistics()
        self.generate_rubber_characterization_report()
        self.generate_mixture_design_report()
        
        # Create visualizations
        self.create_fresh_state_plots()
        self.create_mechanical_properties_plots()
        self.create_physical_properties_plots()
        correlation_matrix = self.create_correlation_analysis()
        
        # Generate key findings
        print("\n" + "="*60)
        print("KEY FINDINGS AND INSIGHTS")
        print("="*60)
        
        # Analyze trends
        self._analyze_trends()
        
        print(f"\nAnalysis complete! All plots saved to {self.data_dir}/")
        print("Generated files:")
        print("- fresh_state_properties_analysis.png")
        print("- mechanical_properties_analysis.png") 
        print("- physical_properties_analysis.png")
        print("- correlation_analysis.png")
    
    def _analyze_trends(self):
        """Analyze key trends in the data."""
        print("\n1. FRESH STATE TRENDS")
        print("-" * 30)
        
        # Slump flow trend
        slump_trend = self.fresh_data.groupby('rubber_replacement')['slump_flow'].mean()
        print(f"Slump flow decreases with rubber content:")
        for replacement, slump in slump_trend.items():
            print(f"  {replacement}% rubber: {slump:.1f} mm")
        
        # Air content trend
        air_trend = self.fresh_data.groupby('rubber_replacement')['air_content'].mean()
        print(f"\nAir content increases with rubber content:")
        for replacement, air in air_trend.items():
            print(f"  {replacement}% rubber: {air:.1f}%")
        
        print("\n2. MECHANICAL PROPERTY TRENDS")
        print("-" * 30)
        
        # Compressive strength at 28 days
        mech_28d = self.mechanical_data[self.mechanical_data['age_days'] == 28]
        strength_trend = mech_28d.groupby('rubber_replacement')['compressive_strength'].mean()
        print(f"28-day compressive strength decreases with rubber content:")
        for replacement, strength in strength_trend.items():
            print(f"  {replacement}% rubber: {strength:.1f} MPa")
        
        print("\n3. PHYSICAL PROPERTY TRENDS")
        print("-" * 30)
        
        # Density trend
        density_trend = self.physical_data.groupby('rubber_replacement')['oven_dry_density'].mean()
        print(f"Oven-dry density decreases with rubber content:")
        for replacement, density in density_trend.items():
            print(f"  {replacement}% rubber: {density:.0f} kg/m³")
        
        # Porosity trend
        porosity_trend = self.physical_data.groupby('rubber_replacement')['porosity'].mean()
        print(f"Porosity increases with rubber content:")
        for replacement, porosity in porosity_trend.items():
            print(f"  {replacement}% rubber: {porosity:.1f}%")

def main():
    """Main function to run the complete analysis."""
    analyzer = DatasetAnalyzer()
    analyzer.create_comprehensive_report()

if __name__ == "__main__":
    main()