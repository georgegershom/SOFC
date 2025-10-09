#!/usr/bin/env python3
"""
Mechanical Boundary Conditions Dataset Analysis and Visualization

This script analyzes the fabricated mechanical boundary conditions dataset for SOFC research,
providing comprehensive statistical analysis, visualizations, and insights into experimental
setup parameters, fixture types, applied loads, and boundary conditions.

Author: Research Dataset Generator
Date: 2025-10-09
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MechanicalBoundaryAnalyzer:
    """
    Comprehensive analyzer for mechanical boundary conditions dataset
    """
    
    def __init__(self, csv_file='mechanical_boundary_conditions_dataset.csv'):
        """Initialize the analyzer with dataset"""
        self.df = pd.read_csv(csv_file)
        self.setup_plotting_style()
        
    def setup_plotting_style(self):
        """Setup consistent plotting style"""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
    def dataset_overview(self):
        """Provide comprehensive dataset overview"""
        print("="*80)
        print("MECHANICAL BOUNDARY CONDITIONS DATASET ANALYSIS")
        print("="*80)
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Experiments: {len(self.df)}")
        print(f"Number of Features: {len(self.df.columns)}")
        print("\nDataset Info:")
        print(self.df.info())
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        # Statistical summary
        print("\n" + "="*50)
        print("STATISTICAL SUMMARY")
        print("="*50)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
    def fixture_type_analysis(self):
        """Analyze fixture types and their characteristics"""
        print("\n" + "="*50)
        print("FIXTURE TYPE ANALYSIS")
        print("="*50)
        
        fixture_counts = self.df['fixture_type'].value_counts()
        print("Fixture Type Distribution:")
        print(fixture_counts)
        
        # Fixture type vs pressure analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Fixture type distribution
        axes[0,0].pie(fixture_counts.values, labels=fixture_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Distribution of Fixture Types')
        
        # Stack pressure by fixture type
        sns.boxplot(data=self.df, x='fixture_type', y='stack_pressure_mpa', ax=axes[0,1])
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        axes[0,1].set_title('Stack Pressure Distribution by Fixture Type')
        
        # Safety factor by fixture type
        sns.boxplot(data=self.df, x='fixture_type', y='safety_factor', ax=axes[1,0])
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        axes[1,0].set_title('Safety Factor by Fixture Type')
        
        # Test duration by fixture type
        sns.boxplot(data=self.df, x='fixture_type', y='test_duration_hours', ax=axes[1,1])
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        axes[1,1].set_title('Test Duration by Fixture Type')
        
        plt.tight_layout()
        plt.savefig('fixture_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def pressure_load_analysis(self):
        """Analyze pressure and load relationships"""
        print("\n" + "="*50)
        print("PRESSURE AND LOAD ANALYSIS")
        print("="*50)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Stack pressure distribution
        axes[0,0].hist(self.df['stack_pressure_mpa'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].set_xlabel('Stack Pressure (MPa)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Stack Pressure Distribution')
        axes[0,0].axvline(self.df['stack_pressure_mpa'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {self.df["stack_pressure_mpa"].mean():.3f}')
        axes[0,0].legend()
        
        # Load magnitude distribution
        axes[0,1].hist(self.df['load_magnitude_mpa'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,1].set_xlabel('Load Magnitude (MPa)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Load Magnitude Distribution')
        axes[0,1].axvline(self.df['load_magnitude_mpa'].mean(), color='red', linestyle='--',
                         label=f'Mean: {self.df["load_magnitude_mpa"].mean():.3f}')
        axes[0,1].legend()
        
        # Pressure vs Load correlation
        axes[0,2].scatter(self.df['stack_pressure_mpa'], self.df['load_magnitude_mpa'], alpha=0.6)
        axes[0,2].set_xlabel('Stack Pressure (MPa)')
        axes[0,2].set_ylabel('Load Magnitude (MPa)')
        axes[0,2].set_title('Stack Pressure vs Load Magnitude')
        
        # Calculate correlation
        corr = self.df['stack_pressure_mpa'].corr(self.df['load_magnitude_mpa'])
        axes[0,2].text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=axes[0,2].transAxes)
        
        # Safety factor vs pressure
        axes[1,0].scatter(self.df['stack_pressure_mpa'], self.df['safety_factor'], alpha=0.6)
        axes[1,0].set_xlabel('Stack Pressure (MPa)')
        axes[1,0].set_ylabel('Safety Factor')
        axes[1,0].set_title('Safety Factor vs Stack Pressure')
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', label='Safety Threshold')
        axes[1,0].legend()
        
        # Temperature vs pressure
        axes[1,1].scatter(self.df['temperature_c'], self.df['stack_pressure_mpa'], alpha=0.6)
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel('Stack Pressure (MPa)')
        axes[1,1].set_title('Temperature vs Stack Pressure')
        
        # Stress concentration factor distribution
        axes[1,2].hist(self.df['stress_concentration_factor'], bins=15, alpha=0.7, edgecolor='black')
        axes[1,2].set_xlabel('Stress Concentration Factor')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Stress Concentration Factor Distribution')
        
        plt.tight_layout()
        plt.savefig('pressure_load_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def boundary_condition_analysis(self):
        """Analyze boundary conditions and constraints"""
        print("\n" + "="*50)
        print("BOUNDARY CONDITION ANALYSIS")
        print("="*50)
        
        # Constraint type analysis
        constraint_counts = self.df['constraint_type'].value_counts()
        print("Constraint Type Distribution:")
        print(constraint_counts)
        
        # Applied load type analysis
        load_type_counts = self.df['applied_load_type'].value_counts()
        print("\nApplied Load Type Distribution:")
        print(load_type_counts)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Constraint types
        axes[0,0].pie(constraint_counts.values, labels=constraint_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Distribution of Constraint Types')
        
        # Load types
        axes[0,1].pie(load_type_counts.values, labels=load_type_counts.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribution of Applied Load Types')
        
        # Safety factor by constraint type
        sns.boxplot(data=self.df, x='constraint_type', y='safety_factor', ax=axes[1,0])
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        axes[1,0].set_title('Safety Factor by Constraint Type')
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Load direction analysis
        load_direction_counts = self.df['load_direction'].value_counts()
        axes[1,1].bar(range(len(load_direction_counts)), load_direction_counts.values)
        axes[1,1].set_xticks(range(len(load_direction_counts)))
        axes[1,1].set_xticklabels(load_direction_counts.index, rotation=45)
        axes[1,1].set_title('Load Direction Distribution')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig('boundary_condition_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def safety_analysis(self):
        """Comprehensive safety factor analysis"""
        print("\n" + "="*50)
        print("SAFETY FACTOR ANALYSIS")
        print("="*50)
        
        # Safety statistics
        safety_stats = self.df['safety_factor'].describe()
        print("Safety Factor Statistics:")
        print(safety_stats)
        
        # Risk assessment
        high_risk = (self.df['safety_factor'] < 1.0).sum()
        moderate_risk = ((self.df['safety_factor'] >= 1.0) & (self.df['safety_factor'] < 1.5)).sum()
        low_risk = (self.df['safety_factor'] >= 1.5).sum()
        
        print(f"\nRisk Assessment:")
        print(f"High Risk (SF < 1.0): {high_risk} experiments ({high_risk/len(self.df)*100:.1f}%)")
        print(f"Moderate Risk (1.0 ≤ SF < 1.5): {moderate_risk} experiments ({moderate_risk/len(self.df)*100:.1f}%)")
        print(f"Low Risk (SF ≥ 1.5): {low_risk} experiments ({low_risk/len(self.df)*100:.1f}%)")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Safety factor histogram
        axes[0,0].hist(self.df['safety_factor'], bins=20, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Failure Threshold')
        axes[0,0].axvline(x=1.5, color='orange', linestyle='--', linewidth=2, label='Design Margin')
        axes[0,0].set_xlabel('Safety Factor')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Safety Factor Distribution')
        axes[0,0].legend()
        
        # Risk categories pie chart
        risk_data = [high_risk, moderate_risk, low_risk]
        risk_labels = ['High Risk\n(SF < 1.0)', 'Moderate Risk\n(1.0 ≤ SF < 1.5)', 'Low Risk\n(SF ≥ 1.5)']
        colors = ['red', 'orange', 'green']
        axes[0,1].pie(risk_data, labels=risk_labels, colors=colors, autopct='%1.1f%%')
        axes[0,1].set_title('Risk Category Distribution')
        
        # Safety factor vs test duration
        axes[1,0].scatter(self.df['test_duration_hours'], self.df['safety_factor'], alpha=0.6)
        axes[1,0].set_xlabel('Test Duration (hours)')
        axes[1,0].set_ylabel('Safety Factor')
        axes[1,0].set_title('Safety Factor vs Test Duration')
        axes[1,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xscale('log')
        
        # Safety factor vs temperature
        axes[1,1].scatter(self.df['temperature_c'], self.df['safety_factor'], alpha=0.6)
        axes[1,1].set_xlabel('Temperature (°C)')
        axes[1,1].set_ylabel('Safety Factor')
        axes[1,1].set_title('Safety Factor vs Temperature')
        axes[1,1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('safety_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def correlation_analysis(self):
        """Analyze correlations between numerical variables"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Select numerical columns
        numerical_cols = ['stack_pressure_mpa', 'load_magnitude_mpa', 'temperature_c', 
                         'test_duration_hours', 'contact_pressure_mpa', 'friction_coefficient',
                         'stress_concentration_factor', 'safety_factor']
        
        correlation_matrix = self.df[numerical_cols].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        plt.title('Correlation Matrix of Numerical Variables')
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Identify strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'var1': correlation_matrix.columns[i],
                        'var2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if strong_correlations:
            print("\nStrong Correlations (|r| > 0.5):")
            for corr in strong_correlations:
                print(f"{corr['var1']} - {corr['var2']}: {corr['correlation']:.3f}")
        
    def clustering_analysis(self):
        """Perform clustering analysis on experimental conditions"""
        print("\n" + "="*50)
        print("CLUSTERING ANALYSIS")
        print("="*50)
        
        # Prepare data for clustering
        features = ['stack_pressure_mpa', 'load_magnitude_mpa', 'temperature_c', 
                   'stress_concentration_factor', 'safety_factor']
        
        X = self.df[features].copy()
        
        # Normalize the features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform K-means clustering
        n_clusters = 4
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        self.df['cluster'] = clusters
        
        # Analyze clusters
        print(f"Clustering Results ({n_clusters} clusters):")
        for i in range(n_clusters):
            cluster_data = self.df[self.df['cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data)} experiments):")
            print(f"  Mean Stack Pressure: {cluster_data['stack_pressure_mpa'].mean():.3f} MPa")
            print(f"  Mean Safety Factor: {cluster_data['safety_factor'].mean():.3f}")
            print(f"  Mean Temperature: {cluster_data['temperature_c'].mean():.1f}°C")
            print(f"  Common Fixture Types: {cluster_data['fixture_type'].mode().values}")
        
        # Visualize clusters
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pressure vs Safety Factor
        scatter = axes[0,0].scatter(self.df['stack_pressure_mpa'], self.df['safety_factor'], 
                                  c=clusters, cmap='viridis', alpha=0.7)
        axes[0,0].set_xlabel('Stack Pressure (MPa)')
        axes[0,0].set_ylabel('Safety Factor')
        axes[0,0].set_title('Clusters: Pressure vs Safety Factor')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Temperature vs Load Magnitude
        scatter = axes[0,1].scatter(self.df['temperature_c'], self.df['load_magnitude_mpa'], 
                                  c=clusters, cmap='viridis', alpha=0.7)
        axes[0,1].set_xlabel('Temperature (°C)')
        axes[0,1].set_ylabel('Load Magnitude (MPa)')
        axes[0,1].set_title('Clusters: Temperature vs Load Magnitude')
        plt.colorbar(scatter, ax=axes[0,1])
        
        # Stress Concentration vs Safety Factor
        scatter = axes[1,0].scatter(self.df['stress_concentration_factor'], self.df['safety_factor'], 
                                  c=clusters, cmap='viridis', alpha=0.7)
        axes[1,0].set_xlabel('Stress Concentration Factor')
        axes[1,0].set_ylabel('Safety Factor')
        axes[1,0].set_title('Clusters: Stress Concentration vs Safety Factor')
        plt.colorbar(scatter, ax=axes[1,0])
        
        # Cluster size distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        axes[1,1].bar(cluster_counts.index, cluster_counts.values)
        axes[1,1].set_xlabel('Cluster')
        axes[1,1].set_ylabel('Number of Experiments')
        axes[1,1].set_title('Cluster Size Distribution')
        
        plt.tight_layout()
        plt.savefig('clustering_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE SUMMARY REPORT")
        print("="*80)
        
        # Dataset summary
        print(f"Total Experiments Analyzed: {len(self.df)}")
        print(f"Unique Fixture Types: {self.df['fixture_type'].nunique()}")
        print(f"Unique Constraint Types: {self.df['constraint_type'].nunique()}")
        print(f"Unique Load Types: {self.df['applied_load_type'].nunique()}")
        
        # Pressure range
        print(f"\nPressure Range: {self.df['stack_pressure_mpa'].min():.3f} - {self.df['stack_pressure_mpa'].max():.3f} MPa")
        print(f"Average Pressure: {self.df['stack_pressure_mpa'].mean():.3f} ± {self.df['stack_pressure_mpa'].std():.3f} MPa")
        
        # Temperature range
        print(f"\nTemperature Range: {self.df['temperature_c'].min():.0f} - {self.df['temperature_c'].max():.0f}°C")
        print(f"Average Temperature: {self.df['temperature_c'].mean():.1f} ± {self.df['temperature_c'].std():.1f}°C")
        
        # Safety assessment
        critical_experiments = self.df[self.df['safety_factor'] < 1.0]
        print(f"\nCritical Experiments (SF < 1.0): {len(critical_experiments)}")
        if len(critical_experiments) > 0:
            print("Critical Experiment IDs:", critical_experiments['experiment_id'].tolist())
        
        # Duration analysis
        print(f"\nTest Duration Range: {self.df['test_duration_hours'].min():.1f} - {self.df['test_duration_hours'].max():.1f} hours")
        long_term_tests = self.df[self.df['test_duration_hours'] > 1000]
        print(f"Long-term Tests (>1000h): {len(long_term_tests)}")
        
        # Recommendations
        print("\n" + "="*50)
        print("RECOMMENDATIONS")
        print("="*50)
        
        # Most reliable fixture type
        fixture_safety = self.df.groupby('fixture_type')['safety_factor'].mean().sort_values(ascending=False)
        print(f"Most Reliable Fixture Type: {fixture_safety.index[0]} (Avg SF: {fixture_safety.iloc[0]:.3f})")
        
        # Optimal pressure range
        safe_experiments = self.df[self.df['safety_factor'] >= 1.5]
        if len(safe_experiments) > 0:
            optimal_pressure = safe_experiments['stack_pressure_mpa'].mean()
            print(f"Recommended Pressure Range: {optimal_pressure:.3f} ± {safe_experiments['stack_pressure_mpa'].std():.3f} MPa")
        
        # Temperature considerations
        temp_safety = self.df.groupby(pd.cut(self.df['temperature_c'], bins=5))['safety_factor'].mean()
        print(f"Temperature with Highest Safety: {temp_safety.idxmax()}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE - All visualizations saved as PNG files")
        print("="*80)

def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = MechanicalBoundaryAnalyzer()
        
        # Run comprehensive analysis
        analyzer.dataset_overview()
        analyzer.fixture_type_analysis()
        analyzer.pressure_load_analysis()
        analyzer.boundary_condition_analysis()
        analyzer.safety_analysis()
        analyzer.correlation_analysis()
        analyzer.clustering_analysis()
        analyzer.generate_summary_report()
        
    except FileNotFoundError:
        print("Error: mechanical_boundary_conditions_dataset.csv not found!")
        print("Please ensure the dataset file is in the current directory.")
    except Exception as e:
        print(f"An error occurred during analysis: {str(e)}")

if __name__ == "__main__":
    main()