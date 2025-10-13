#!/usr/bin/env python3
"""
Geotechnical Dataset Analysis and Visualization
Comprehensive analysis of generated and downloaded geotechnical datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import os
import warnings
warnings.filterwarnings('ignore')

class GeotechnicalAnalyzer:
    def __init__(self, data_dir='.'):
        self.data_dir = data_dir
        self.sandy_soils = None
        self.clay_soils = None
        self.failure_cases = None
        self.underground_structures = None
        self.earthquake_data = None
        self.liquefaction_cases = None
        self.landslide_cases = None
        
    def load_datasets(self):
        """Load all available datasets"""
        print("Loading datasets...")
        
        try:
            self.sandy_soils = pd.read_csv(f'{self.data_dir}/sandy_soils/sandy_soil_properties.csv')
            print(f"Loaded sandy soils: {len(self.sandy_soils)} samples")
        except FileNotFoundError:
            print("Sandy soils dataset not found")
            
        try:
            self.clay_soils = pd.read_csv(f'{self.data_dir}/clay_soils/clay_soil_properties.csv')
            print(f"Loaded clay soils: {len(self.clay_soils)} samples")
        except FileNotFoundError:
            print("Clay soils dataset not found")
            
        try:
            self.failure_cases = pd.read_csv(f'{self.data_dir}/case_studies/failure_case_studies.csv')
            print(f"Loaded failure cases: {len(self.failure_cases)} samples")
        except FileNotFoundError:
            print("Failure cases dataset not found")
            
        try:
            self.underground_structures = pd.read_csv(f'{self.data_dir}/case_studies/underground_structures.csv')
            print(f"Loaded underground structures: {len(self.underground_structures)} samples")
        except FileNotFoundError:
            print("Underground structures dataset not found")
            
        try:
            self.earthquake_data = pd.read_csv(f'{self.data_dir}/real_data/usgs_earthquakes.csv')
            print(f"Loaded earthquake data: {len(self.earthquake_data)} samples")
        except FileNotFoundError:
            print("Earthquake data not found")
            
        try:
            self.liquefaction_cases = pd.read_csv(f'{self.data_dir}/real_data/liquefaction_cases.csv')
            print(f"Loaded liquefaction cases: {len(self.liquefaction_cases)} samples")
        except FileNotFoundError:
            print("Liquefaction cases not found")
            
        try:
            self.landslide_cases = pd.read_csv(f'{self.data_dir}/real_data/landslide_cases.csv')
            print(f"Loaded landslide cases: {len(self.landslide_cases)} samples")
        except FileNotFoundError:
            print("Landslide cases not found")
    
    def analyze_sandy_soils(self):
        """Comprehensive analysis of sandy soil properties"""
        if self.sandy_soils is None:
            return
            
        print("\nAnalyzing sandy soil properties...")
        
        # Basic statistics
        numeric_cols = self.sandy_soils.select_dtypes(include=[np.number]).columns
        stats_summary = self.sandy_soils[numeric_cols].describe()
        
        # Save statistics
        stats_summary.to_csv(f'{self.data_dir}/analysis/sandy_soils_statistics.csv')
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Sandy Soil Properties Analysis', fontsize=16)
        
        # Grain size distribution
        axes[0, 0].hist(self.sandy_soils['D50_mm'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 0].set_xlabel('D50 (mm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Grain Size Distribution (D50)')
        
        # Friction angle vs relative density
        axes[0, 1].scatter(self.sandy_soils['relative_density_pct'], 
                          self.sandy_soils['friction_angle_deg'], 
                          alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Relative Density (%)')
        axes[0, 1].set_ylabel('Friction Angle (degrees)')
        axes[0, 1].set_title('Friction Angle vs Relative Density')
        
        # SPT N-value distribution
        axes[0, 2].hist(self.sandy_soils['SPT_N60'], bins=30, alpha=0.7, color='green')
        axes[0, 2].set_xlabel('SPT N60')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('SPT N-value Distribution')
        
        # Liquefaction potential
        axes[1, 0].scatter(self.sandy_soils['SPT_N60'], 
                          self.sandy_soils['CRR_7_5'], 
                          alpha=0.6, color='red')
        axes[1, 0].set_xlabel('SPT N60')
        axes[1, 0].set_ylabel('Cyclic Resistance Ratio')
        axes[1, 0].set_title('Liquefaction Resistance')
        
        # Pore pressure relationships
        axes[1, 1].scatter(self.sandy_soils['initial_pore_pressure_kPa'], 
                          self.sandy_soils['excess_pore_pressure_kPa'], 
                          alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Initial Pore Pressure (kPa)')
        axes[1, 1].set_ylabel('Excess Pore Pressure (kPa)')
        axes[1, 1].set_title('Pore Pressure Relationships')
        
        # Permeability distribution
        axes[1, 2].hist(np.log10(self.sandy_soils['permeability_m_s']), 
                       bins=30, alpha=0.7, color='brown')
        axes[1, 2].set_xlabel('Log10 Permeability (m/s)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Permeability Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/sandy_soils_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation analysis
        correlation_matrix = self.sandy_soils[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Sandy Soils Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/sandy_soils_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Sandy soil analysis completed")
    
    def analyze_clay_soils(self):
        """Comprehensive analysis of clay soil properties"""
        if self.clay_soils is None:
            return
            
        print("\nAnalyzing clay soil properties...")
        
        # Basic statistics
        numeric_cols = self.clay_soils.select_dtypes(include=[np.number]).columns
        stats_summary = self.clay_soils[numeric_cols].describe()
        
        # Save statistics
        stats_summary.to_csv(f'{self.data_dir}/analysis/clay_soils_statistics.csv')
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clay Soil Properties Analysis', fontsize=16)
        
        # Atterberg limits
        axes[0, 0].scatter(self.clay_soils['liquid_limit_pct'], 
                          self.clay_soils['plasticity_index'], 
                          alpha=0.6, color='blue')
        axes[0, 0].set_xlabel('Liquid Limit (%)')
        axes[0, 0].set_ylabel('Plasticity Index')
        axes[0, 0].set_title('Atterberg Limits')
        
        # Undrained shear strength
        axes[0, 1].hist(self.clay_soils['undrained_shear_strength_kPa'], 
                       bins=30, alpha=0.7, color='green')
        axes[0, 1].set_xlabel('Undrained Shear Strength (kPa)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Undrained Shear Strength Distribution')
        
        # Preconsolidation stress
        axes[0, 2].hist(self.clay_soils['preconsolidation_stress_kPa'], 
                       bins=30, alpha=0.7, color='orange')
        axes[0, 2].set_xlabel('Preconsolidation Stress (kPa)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Preconsolidation Stress Distribution')
        
        # Sensitivity
        axes[1, 0].hist(self.clay_soils['sensitivity'], 
                       bins=30, alpha=0.7, color='red')
        axes[1, 0].set_xlabel('Sensitivity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sensitivity Distribution')
        
        # Mineral composition
        minerals = ['smectite_pct', 'illite_pct', 'kaolinite_pct', 'chlorite_pct']
        mineral_data = [self.clay_soils[col] for col in minerals]
        axes[1, 1].boxplot(mineral_data, labels=['Smectite', 'Illite', 'Kaolinite', 'Chlorite'])
        axes[1, 1].set_ylabel('Percentage (%)')
        axes[1, 1].set_title('Clay Mineral Composition')
        
        # OCR distribution
        axes[1, 2].hist(self.clay_soils['OCR'], 
                       bins=30, alpha=0.7, color='purple')
        axes[1, 2].set_xlabel('Overconsolidation Ratio')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('OCR Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/clay_soils_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Correlation analysis
        correlation_matrix = self.clay_soils[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Clay Soils Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/clay_soils_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Clay soil analysis completed")
    
    def analyze_failure_cases(self):
        """Analysis of failure case studies"""
        if self.failure_cases is None:
            return
            
        print("\nAnalyzing failure case studies...")
        
        # Failure type distribution
        failure_counts = self.failure_cases['failure_type'].value_counts()
        
        plt.figure(figsize=(10, 6))
        failure_counts.plot(kind='bar', color='coral')
        plt.title('Distribution of Failure Types')
        plt.xlabel('Failure Type')
        plt.ylabel('Number of Cases')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/failure_types_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Damage level analysis
        damage_counts = self.failure_cases['damage_level'].value_counts()
        
        plt.figure(figsize=(8, 6))
        damage_counts.plot(kind='pie', autopct='%1.1f%%', colors=['lightgreen', 'yellow', 'orange', 'red'])
        plt.title('Distribution of Damage Levels')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/damage_levels_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Economic impact analysis
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(np.log10(self.failure_cases['economic_loss_usd']), bins=30, alpha=0.7, color='blue')
        plt.xlabel('Log10 Economic Loss (USD)')
        plt.ylabel('Frequency')
        plt.title('Economic Loss Distribution')
        
        plt.subplot(2, 2, 2)
        plt.scatter(self.failure_cases['max_displacement_mm'], 
                   self.failure_cases['economic_loss_usd'], 
                   alpha=0.6, color='red')
        plt.xlabel('Max Displacement (mm)')
        plt.ylabel('Economic Loss (USD)')
        plt.title('Displacement vs Economic Loss')
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        plt.scatter(self.failure_cases['settlement_mm'], 
                   self.failure_cases['economic_loss_usd'], 
                   alpha=0.6, color='green')
        plt.xlabel('Settlement (mm)')
        plt.ylabel('Economic Loss (USD)')
        plt.title('Settlement vs Economic Loss')
        plt.yscale('log')
        
        plt.subplot(2, 2, 4)
        failure_cases_numeric = self.failure_cases.select_dtypes(include=[np.number])
        correlation_matrix = failure_cases_numeric.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f')
        plt.title('Failure Cases Correlation Matrix')
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/failure_cases_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Failure case analysis completed")
    
    def analyze_earthquake_data(self):
        """Analysis of earthquake data for liquefaction studies"""
        if self.earthquake_data is None:
            return
            
        print("\nAnalyzing earthquake data...")
        
        # Convert time column if it exists
        if 'datetime' in self.earthquake_data.columns:
            self.earthquake_data['year'] = pd.to_datetime(self.earthquake_data['datetime']).dt.year
        
        plt.figure(figsize=(15, 10))
        
        # Magnitude distribution
        plt.subplot(2, 3, 1)
        plt.hist(self.earthquake_data['magnitude'], bins=30, alpha=0.7, color='red')
        plt.xlabel('Magnitude')
        plt.ylabel('Frequency')
        plt.title('Earthquake Magnitude Distribution')
        
        # Depth distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.earthquake_data['depth_km'], bins=30, alpha=0.7, color='blue')
        plt.xlabel('Depth (km)')
        plt.ylabel('Frequency')
        plt.title('Earthquake Depth Distribution')
        
        # Magnitude vs Depth
        plt.subplot(2, 3, 3)
        plt.scatter(self.earthquake_data['magnitude'], 
                   self.earthquake_data['depth_km'], 
                   alpha=0.6, color='green')
        plt.xlabel('Magnitude')
        plt.ylabel('Depth (km)')
        plt.title('Magnitude vs Depth')
        
        # Temporal distribution
        if 'year' in self.earthquake_data.columns:
            plt.subplot(2, 3, 4)
            yearly_counts = self.earthquake_data['year'].value_counts().sort_index()
            plt.plot(yearly_counts.index, yearly_counts.values, marker='o')
            plt.xlabel('Year')
            plt.ylabel('Number of Earthquakes')
            plt.title('Earthquake Frequency Over Time')
        
        # Geographic distribution
        plt.subplot(2, 3, 5)
        plt.scatter(self.earthquake_data['longitude'], 
                   self.earthquake_data['latitude'], 
                   c=self.earthquake_data['magnitude'], 
                   cmap='Reds', alpha=0.6)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Geographic Distribution of Earthquakes')
        plt.colorbar(label='Magnitude')
        
        # Tsunami correlation
        if 'tsunami' in self.earthquake_data.columns:
            plt.subplot(2, 3, 6)
            tsunami_magnitude = self.earthquake_data[self.earthquake_data['tsunami'] == 1]['magnitude']
            no_tsunami_magnitude = self.earthquake_data[self.earthquake_data['tsunami'] == 0]['magnitude']
            plt.hist([tsunami_magnitude, no_tsunami_magnitude], 
                    bins=20, alpha=0.7, label=['With Tsunami', 'No Tsunami'], 
                    color=['red', 'blue'])
            plt.xlabel('Magnitude')
            plt.ylabel('Frequency')
            plt.title('Magnitude Distribution: Tsunami vs No Tsunami')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/visualizations/earthquake_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Earthquake data analysis completed")
    
    def create_interactive_dashboard(self):
        """Create interactive Plotly dashboard"""
        print("\nCreating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sandy Soils - Friction Angle vs Density', 
                          'Clay Soils - Atterberg Limits',
                          'Failure Cases - Economic Impact',
                          'Earthquake Magnitude Distribution',
                          'Liquefaction Cases - SPT vs CRR',
                          'Landslide Cases - Volume Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Sandy soils plot
        if self.sandy_soils is not None:
            fig.add_trace(
                go.Scatter(x=self.sandy_soils['relative_density_pct'],
                          y=self.sandy_soils['friction_angle_deg'],
                          mode='markers',
                          name='Sandy Soils',
                          marker=dict(color='blue', opacity=0.6)),
                row=1, col=1
            )
        
        # Clay soils plot
        if self.clay_soils is not None:
            fig.add_trace(
                go.Scatter(x=self.clay_soils['liquid_limit_pct'],
                          y=self.clay_soils['plasticity_index'],
                          mode='markers',
                          name='Clay Soils',
                          marker=dict(color='red', opacity=0.6)),
                row=1, col=2
            )
        
        # Failure cases plot
        if self.failure_cases is not None:
            failure_counts = self.failure_cases['failure_type'].value_counts()
            fig.add_trace(
                go.Bar(x=failure_counts.index,
                      y=failure_counts.values,
                      name='Failure Types',
                      marker_color='green'),
                row=2, col=1
            )
        
        # Earthquake data plot
        if self.earthquake_data is not None:
            fig.add_trace(
                go.Histogram(x=self.earthquake_data['magnitude'],
                           name='Earthquake Magnitude',
                           marker_color='orange'),
                row=2, col=2
            )
        
        # Liquefaction cases plot
        if self.liquefaction_cases is not None:
            fig.add_trace(
                go.Scatter(x=self.liquefaction_cases['SPT_N60'],
                          y=self.liquefaction_cases['max_settlement_mm'],
                          mode='markers',
                          name='Liquefaction Cases',
                          marker=dict(color='purple', opacity=0.6)),
                row=3, col=1
            )
        
        # Landslide cases plot
        if self.landslide_cases is not None:
            fig.add_trace(
                go.Histogram(x=np.log10(self.landslide_cases['volume_m3']),
                           name='Landslide Volume (log10)',
                           marker_color='brown'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Geotechnical Data Interactive Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save interactive dashboard
        fig.write_html(f'{self.data_dir}/visualizations/interactive_dashboard.html')
        
        print("Interactive dashboard created")
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report = []
        report.append("# Geotechnical Dataset Analysis Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Dataset summaries
        datasets = [
            ("Sandy Soils", self.sandy_soils),
            ("Clay Soils", self.clay_soils),
            ("Failure Cases", self.failure_cases),
            ("Underground Structures", self.underground_structures),
            ("Earthquake Data", self.earthquake_data),
            ("Liquefaction Cases", self.liquefaction_cases),
            ("Landslide Cases", self.landslide_cases)
        ]
        
        for name, df in datasets:
            if df is not None:
                report.append(f"## {name}")
                report.append(f"- **Samples**: {len(df)}")
                report.append(f"- **Variables**: {len(df.columns)}")
                report.append(f"- **Numeric Variables**: {len(df.select_dtypes(include=[np.number]).columns)}")
                report.append("")
        
        # Key findings
        report.append("## Key Findings")
        report.append("")
        
        if self.sandy_soils is not None:
            report.append("### Sandy Soils")
            report.append(f"- Average friction angle: {self.sandy_soils['friction_angle_deg'].mean():.1f}°")
            report.append(f"- Average relative density: {self.sandy_soils['relative_density_pct'].mean():.1f}%")
            report.append(f"- Average SPT N60: {self.sandy_soils['SPT_N60'].mean():.1f}")
            report.append("")
        
        if self.clay_soils is not None:
            report.append("### Clay Soils")
            report.append(f"- Average liquid limit: {self.clay_soils['liquid_limit_pct'].mean():.1f}%")
            report.append(f"- Average plasticity index: {self.clay_soils['plasticity_index'].mean():.1f}")
            report.append(f"- Average undrained shear strength: {self.clay_soils['undrained_shear_strength_kPa'].mean():.1f} kPa")
            report.append("")
        
        if self.failure_cases is not None:
            report.append("### Failure Cases")
            most_common_failure = self.failure_cases['failure_type'].mode().iloc[0]
            report.append(f"- Most common failure type: {most_common_failure}")
            report.append(f"- Average economic loss: ${self.failure_cases['economic_loss_usd'].mean():,.0f}")
            report.append("")
        
        # Save report
        with open(f'{self.data_dir}/analysis/summary_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        print("Summary report generated")

def main():
    """Main analysis function"""
    print("Starting Geotechnical Dataset Analysis...")
    print("=" * 50)
    
    # Create analysis directories
    os.makedirs('analysis', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    
    # Initialize analyzer
    analyzer = GeotechnicalAnalyzer()
    
    # Load datasets
    analyzer.load_datasets()
    
    # Perform analyses
    analyzer.analyze_sandy_soils()
    analyzer.analyze_clay_soils()
    analyzer.analyze_failure_cases()
    analyzer.analyze_earthquake_data()
    
    # Create interactive dashboard
    analyzer.create_interactive_dashboard()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    print("\nAnalysis completed!")
    print("Results saved in:")
    print("- analysis/")
    print("- visualizations/")

if __name__ == "__main__":
    main()