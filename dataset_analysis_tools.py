#!/usr/bin/env python3
"""
Welding Dataset Analysis and Validation Tools
=============================================

This module provides comprehensive analysis and validation tools for the
welding inverse design dataset, including data quality checks, statistical
analysis, and visualization capabilities.

Author: AI Assistant
Date: 2025-10-14
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import json
import warnings
warnings.filterwarnings('ignore')

class WeldingDatasetAnalyzer:
    """
    Comprehensive analysis and validation tool for welding datasets.
    """
    
    def __init__(self, dataset_path=None, dataset_df=None):
        """Initialize the analyzer with dataset."""
        if dataset_df is not None:
            self.df = dataset_df.copy()
        elif dataset_path:
            self.df = pd.read_csv(dataset_path)
        else:
            raise ValueError("Either dataset_path or dataset_df must be provided")
        
        self.input_cols = [
            'laser_power_w', 'welding_speed_mm_s', 'pulse_frequency_hz', 
            'pulse_duration_ms', 'beam_focus_position_mm', 'beam_spot_size_um',
            'clamping_pressure_mpa', 'shield_gas_flow_rate_l_min', 
            'sheet_thickness_mm', 'overlap_distance_mm'
        ]
        
        self.output_cols = [
            'nugget_width_mm', 'penetration_depth_mm', 'haz_width_mm',
            'crack_presence', 'porosity_area_percent', 'spatter_rating',
            'tensile_shear_strength_n', 'peel_strength_n', 'contact_resistance_micro_ohm',
            'strength_degradation_percent', 'resistance_increase_percent',
            'cycles_to_failure', 'creep_time_to_failure_hours',
            'imc_thickness_post_aging_um', 'grain_size_change_percent'
        ]
        
        self.categorical_cols = ['material_combination', 'joint_type']
        
    def data_quality_report(self):
        """Generate comprehensive data quality report."""
        print("=== WELDING DATASET QUALITY REPORT ===")
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Total Samples: {len(self.df)}")
        print()
        
        # Data source distribution
        print("Data Source Distribution:")
        source_counts = self.df['data_source'].value_counts()
        for source, count in source_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
        print()
        
        # Missing data analysis
        print("Missing Data Analysis:")
        missing_data = self.df.isnull().sum()
        missing_percentage = (missing_data / len(self.df)) * 100
        
        for col in self.input_cols + self.output_cols:
            if col in self.df.columns:
                missing_count = missing_data[col]
                missing_pct = missing_percentage[col]
                if missing_count > 0:
                    print(f"  {col}: {missing_count} ({missing_pct:.1f}%)")
        
        if missing_data.sum() == 0:
            print("  No missing data found!")
        print()
        
        # Data type validation
        print("Data Type Validation:")
        for col in self.input_cols + self.output_cols:
            if col in self.df.columns:
                dtype = self.df[col].dtype
                if col in ['crack_presence', 'spatter_rating']:
                    expected = "integer"
                else:
                    expected = "numeric"
                print(f"  {col}: {dtype} ({expected})")
        print()
        
        # Statistical summary
        print("Statistical Summary (Key Parameters):")
        key_params = ['laser_power_w', 'welding_speed_mm_s', 'tensile_shear_strength_n', 
                     'contact_resistance_micro_ohm', 'cycles_to_failure']
        
        for param in key_params:
            if param in self.df.columns:
                data = self.df[param].dropna()
                if len(data) > 0:
                    print(f"  {param}:")
                    print(f"    Mean: {data.mean():.2f}, Std: {data.std():.2f}")
                    print(f"    Min: {data.min():.2f}, Max: {data.max():.2f}")
                    print(f"    Median: {data.median():.2f}")
        print()
        
        return {
            'total_samples': len(self.df),
            'data_sources': source_counts.to_dict(),
            'missing_data': missing_data.to_dict(),
            'missing_percentage': missing_percentage.to_dict()
        }
    
    def detect_outliers(self, method='isolation_forest'):
        """Detect outliers in the dataset using various methods."""
        print(f"=== OUTLIER DETECTION ({method.upper()}) ===")
        
        # Prepare numerical data
        numerical_cols = [col for col in self.input_cols + self.output_cols 
                         if col in self.df.columns and self.df[col].dtype in ['float64', 'int64']]
        
        data_for_outlier_detection = self.df[numerical_cols].dropna()
        
        if len(data_for_outlier_detection) == 0:
            print("No complete numerical data available for outlier detection.")
            return pd.DataFrame()
        
        if method == 'isolation_forest':
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data_for_outlier_detection)
            
            # Detect outliers
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(scaled_data)
            
            # Get outlier indices
            outlier_indices = data_for_outlier_detection.index[outlier_labels == -1]
            
        elif method == 'statistical':
            # Use Z-score method
            z_scores = np.abs(stats.zscore(data_for_outlier_detection))
            outlier_indices = data_for_outlier_detection.index[(z_scores > 3).any(axis=1)]
        
        outliers_df = self.df.loc[outlier_indices]
        
        print(f"Detected {len(outliers_df)} outliers ({len(outliers_df)/len(self.df)*100:.1f}% of data)")
        
        if len(outliers_df) > 0:
            print("\nOutlier Summary by Data Source:")
            outlier_sources = outliers_df['data_source'].value_counts()
            for source, count in outlier_sources.items():
                print(f"  {source}: {count}")
        
        return outliers_df
    
    def correlation_analysis(self):
        """Analyze correlations between input and output parameters."""
        print("=== CORRELATION ANALYSIS ===")
        
        # Select numerical columns
        numerical_cols = [col for col in self.input_cols + self.output_cols 
                         if col in self.df.columns and self.df[col].dtype in ['float64', 'int64']]
        
        correlation_data = self.df[numerical_cols].dropna()
        
        if len(correlation_data) == 0:
            print("No complete numerical data available for correlation analysis.")
            return None
        
        # Calculate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Find strong correlations with output parameters
        print("Strong Input-Output Correlations (|r| > 0.5):")
        
        output_corrs = {}
        for output_col in self.output_cols:
            if output_col in corr_matrix.columns:
                input_corrs = corr_matrix[output_col][self.input_cols]
                strong_corrs = input_corrs[abs(input_corrs) > 0.5].dropna()
                
                if len(strong_corrs) > 0:
                    print(f"\n  {output_col}:")
                    for input_col, corr_val in strong_corrs.items():
                        print(f"    {input_col}: {corr_val:.3f}")
                    output_corrs[output_col] = strong_corrs.to_dict()
        
        return corr_matrix, output_corrs
    
    def physics_validation(self):
        """Validate physical relationships in the data."""
        print("=== PHYSICS VALIDATION ===")
        
        validation_results = {}
        
        # Check heat input relationships
        if all(col in self.df.columns for col in ['laser_power_w', 'welding_speed_mm_s', 'nugget_width_mm']):
            # Calculate heat input
            heat_input = self.df['laser_power_w'] / self.df['welding_speed_mm_s']
            nugget_width = self.df['nugget_width_mm']
            
            # Remove NaN values
            mask = ~(pd.isna(heat_input) | pd.isna(nugget_width))
            if mask.sum() > 10:
                corr_heat_nugget = np.corrcoef(heat_input[mask], nugget_width[mask])[0, 1]
                validation_results['heat_input_nugget_correlation'] = corr_heat_nugget
                
                expected_positive = corr_heat_nugget > 0
                print(f"Heat Input vs Nugget Width correlation: {corr_heat_nugget:.3f}")
                print(f"  Expected positive correlation: {'✓' if expected_positive else '✗'}")
        
        # Check strength vs defects relationship
        if all(col in self.df.columns for col in ['tensile_shear_strength_n', 'crack_presence']):
            strength_with_cracks = self.df[self.df['crack_presence'] == 1]['tensile_shear_strength_n'].dropna()
            strength_without_cracks = self.df[self.df['crack_presence'] == 0]['tensile_shear_strength_n'].dropna()
            
            if len(strength_with_cracks) > 5 and len(strength_without_cracks) > 5:
                mean_with_cracks = strength_with_cracks.mean()
                mean_without_cracks = strength_without_cracks.mean()
                
                validation_results['strength_defect_relationship'] = {
                    'mean_with_cracks': mean_with_cracks,
                    'mean_without_cracks': mean_without_cracks
                }
                
                expected_lower = mean_with_cracks < mean_without_cracks
                print(f"Strength with cracks: {mean_with_cracks:.0f}N")
                print(f"Strength without cracks: {mean_without_cracks:.0f}N")
                print(f"  Expected lower strength with cracks: {'✓' if expected_lower else '✗'}")
        
        # Check material combination effects
        if 'material_combination' in self.df.columns and 'contact_resistance_micro_ohm' in self.df.columns:
            material_resistance = self.df.groupby('material_combination')['contact_resistance_micro_ohm'].mean().dropna()
            
            if len(material_resistance) > 1:
                validation_results['material_resistance'] = material_resistance.to_dict()
                print(f"\nContact Resistance by Material:")
                for material, resistance in material_resistance.items():
                    print(f"  {material}: {resistance:.1f} µΩ")
        
        return validation_results
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics."""
        print("=== SUMMARY STATISTICS ===")
        
        summary_stats = {}
        
        # Input parameters summary
        print("Input Parameters Summary:")
        input_summary = self.df[self.input_cols].describe()
        summary_stats['input_parameters'] = input_summary.to_dict()
        
        for col in self.input_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    print(f"  {col}: μ={data.mean():.2f}, σ={data.std():.2f}, range=[{data.min():.2f}, {data.max():.2f}]")
        
        print("\nOutput Parameters Summary:")
        output_summary = self.df[self.output_cols].describe()
        summary_stats['output_parameters'] = output_summary.to_dict()
        
        for col in self.output_cols:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    print(f"  {col}: μ={data.mean():.2f}, σ={data.std():.2f}, range=[{data.min():.2f}, {data.max():.2f}]")
        
        # Categorical parameters
        print("\nCategorical Parameters Distribution:")
        for col in self.categorical_cols:
            if col in self.df.columns:
                distribution = self.df[col].value_counts()
                summary_stats[f'{col}_distribution'] = distribution.to_dict()
                print(f"  {col}:")
                for category, count in distribution.items():
                    percentage = (count / len(self.df)) * 100
                    print(f"    {category}: {count} ({percentage:.1f}%)")
        
        return summary_stats
    
    def data_completeness_analysis(self):
        """Analyze data completeness across different tiers and parameters."""
        print("=== DATA COMPLETENESS ANALYSIS ===")
        
        completeness_results = {}
        
        # Completeness by data source
        print("Completeness by Data Source:")
        for source in self.df['data_source'].unique():
            source_data = self.df[self.df['data_source'] == source]
            
            # Calculate completeness for each parameter
            completeness = {}
            for col in self.input_cols + self.output_cols:
                if col in source_data.columns:
                    non_null_count = source_data[col].count()
                    total_count = len(source_data)
                    completeness_pct = (non_null_count / total_count) * 100
                    completeness[col] = completeness_pct
            
            completeness_results[source] = completeness
            
            # Print summary
            avg_completeness = np.mean(list(completeness.values()))
            print(f"  {source}: {avg_completeness:.1f}% average completeness")
            
            # Show parameters with low completeness
            low_completeness = {k: v for k, v in completeness.items() if v < 80}
            if low_completeness:
                print(f"    Parameters with <80% completeness:")
                for param, pct in low_completeness.items():
                    print(f"      {param}: {pct:.1f}%")
        
        return completeness_results
    
    def save_analysis_report(self, output_path='dataset_analysis_report.json'):
        """Save comprehensive analysis report to JSON file."""
        print(f"\n=== SAVING ANALYSIS REPORT ===")
        
        # Run all analyses
        quality_report = self.data_quality_report()
        outliers = self.detect_outliers()
        corr_matrix, strong_correlations = self.correlation_analysis()
        physics_validation = self.physics_validation()
        summary_stats = self.generate_summary_statistics()
        completeness = self.data_completeness_analysis()
        
        # Compile report
        analysis_report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.df),
                'shape': list(self.df.shape),
                'data_sources': self.df['data_source'].value_counts().to_dict()
            },
            'data_quality': quality_report,
            'outlier_detection': {
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(self.df) * 100,
                'outlier_weld_ids': outliers['weld_id'].tolist() if 'weld_id' in outliers.columns else []
            },
            'correlations': {
                'strong_input_output_correlations': strong_correlations
            },
            'physics_validation': physics_validation,
            'summary_statistics': summary_stats,
            'data_completeness': completeness
        }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        print(f"Analysis report saved to: {output_path}")
        return analysis_report

def main():
    """Main function to run dataset analysis."""
    print("Loading welding dataset for analysis...")
    
    try:
        # Load the master dataset
        analyzer = WeldingDatasetAnalyzer('welding_inverse_design_master_dataset.csv')
        
        # Generate comprehensive analysis report
        report = analyzer.save_analysis_report()
        
        print("\n=== ANALYSIS COMPLETE ===")
        print("Generated files:")
        print("- dataset_analysis_report.json")
        
        return analyzer, report
        
    except FileNotFoundError:
        print("Error: welding_inverse_design_master_dataset.csv not found!")
        print("Please run welding_dataset_generator.py first.")
        return None, None

if __name__ == "__main__":
    analyzer, report = main()