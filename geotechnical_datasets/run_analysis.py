#!/usr/bin/env python3
"""
Main script to run comprehensive geotechnical data analysis
"""

import os
import sys
sys.path.append('analysis_tools')

from data_loader import GeotechnicalDataLoader
from visualization import GeotechnicalVisualizer
from statistical_analysis import GeotechnicalAnalyzer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run comprehensive analysis of geotechnical datasets"""
    
    print("="*60)
    print("GEOTECHNICAL DATASET ANALYSIS")
    print("Underground Structure Failure Mechanisms Research")
    print("="*60)
    
    # Initialize components
    loader = GeotechnicalDataLoader()
    visualizer = GeotechnicalVisualizer()
    analyzer = GeotechnicalAnalyzer()
    
    # Load all datasets
    print("\n📁 LOADING DATASETS...")
    print("-"*40)
    
    # Sandy soils
    sandy_data, sandy_merged = loader.load_sandy_soils()
    print(f"✓ Sandy Soils: {len(sandy_merged)} samples loaded")
    
    # Clay soils
    clay_data, clay_merged = loader.load_clay_soils()
    print(f"✓ Clay Soils: {len(clay_merged)} samples loaded")
    
    # Case studies
    case_data = loader.load_case_studies()
    print(f"✓ Case Studies: {len(case_data['failures'])} failure cases")
    print(f"✓ Monitoring: {len(case_data['monitoring'])} monitoring records")
    
    # Spatial data
    spatial_data = loader.load_spatial_data()
    print(f"✓ Spatial Data: {len(spatial_data['grid'])} grid points")
    
    # Summary statistics
    print("\n📊 DATASET SUMMARY")
    print("-"*40)
    
    # Sandy soils summary
    print("\nSandy Soils - Key Statistics:")
    sandy_stats = sandy_merged[['sand_content_%', 'friction_angle_deg', 'relative_density_%', 'CSR']].describe()
    print(sandy_stats.round(2))
    
    # Clay soils summary
    print("\nClay Soils - Key Statistics:")
    clay_stats = clay_merged[['liquid_limit_%', 'plasticity_index', 'undrained_shear_strength_kPa', 'OCR']].describe()
    print(clay_stats.round(2))
    
    # Statistical Analysis
    print("\n🔬 STATISTICAL ANALYSIS")
    print("-"*40)
    
    # Correlation analysis for sandy soils
    print("\nSandy Soils - Top Correlations with Liquefaction (CSR):")
    sandy_corr = loader.calculate_correlations(sandy_merged, 'CSR')
    for param, corr in list(sandy_corr.items())[:5]:
        print(f"  • {param}: {corr:.3f}")
    
    # Correlation analysis for clay soils
    print("\nClay Soils - Top Correlations with Shear Strength:")
    clay_corr = loader.calculate_correlations(clay_merged, 'undrained_shear_strength_kPa')
    for param, corr in list(clay_corr.items())[:5]:
        print(f"  • {param}: {corr:.3f}")
    
    # PCA Analysis
    print("\n🎯 PRINCIPAL COMPONENT ANALYSIS")
    print("-"*40)
    
    # PCA for sandy soils
    sandy_pca = analyzer.perform_pca(
        sandy_merged,
        n_components=3,
        columns=['sand_content_%', 'relative_density_%', 'friction_angle_deg', 
                'void_ratio', 'CSR']
    )
    print("\nSandy Soils - Explained Variance by Components:")
    for i, var in enumerate(sandy_pca['explained_variance_ratio']):
        print(f"  • PC{i+1}: {var:.1%}")
    print(f"  • Total: {sandy_pca['cumulative_variance_ratio'][-1]:.1%}")
    
    # Machine Learning Analysis
    print("\n🤖 MACHINE LEARNING PREDICTIONS")
    print("-"*40)
    
    # Predict liquefaction safety factor
    print("\nPredicting Liquefaction Safety Factor:")
    regression_results = analyzer.regression_analysis(
        sandy_merged,
        target_col='safety_factor',
        feature_cols=['CSR', 'CRR', 'relative_density_%', 'N1_60', 'fines_content_%']
    )
    print(f"  • R² Score (Test): {regression_results['test_r2']:.3f}")
    print(f"  • RMSE (Test): {regression_results['test_rmse']:.3f}")
    print(f"  • Cross-validation R²: {regression_results['cv_mean']:.3f} ± {regression_results['cv_std']:.3f}")
    
    print("\nMost Important Features:")
    for _, row in regression_results['feature_importance'].head(3).iterrows():
        print(f"  • {row['feature']}: {row['importance']:.3f}")
    
    # Failure Risk Assessment
    print("\n⚠️ FAILURE RISK ASSESSMENT")
    print("-"*40)
    
    # Analyze failure cases by soil type
    failure_by_soil = case_data['failures'].groupby('soil_type').agg({
        'case_id': 'count',
        'economic_loss_million_USD': 'mean',
        'max_settlement_mm': 'mean',
        'recovery_time_days': 'mean'
    }).round(1)
    failure_by_soil.columns = ['Cases', 'Avg Loss ($M)', 'Avg Settlement (mm)', 'Avg Recovery (days)']
    print("\nFailure Statistics by Soil Type:")
    print(failure_by_soil)
    
    # High-risk conditions
    print("\n🔴 HIGH-RISK CONDITIONS IDENTIFIED:")
    
    # Sandy soils - liquefaction risk
    high_risk_sandy = sandy_merged[sandy_merged['safety_factor'] < 1.0]
    if len(high_risk_sandy) > 0:
        print(f"\nSandy Soils - Liquefaction Risk:")
        print(f"  • {len(high_risk_sandy)} samples with FS < 1.0")
        print(f"  • Average CSR: {high_risk_sandy['CSR'].mean():.3f}")
        print(f"  • Average relative density: {high_risk_sandy['relative_density_%'].mean():.1f}%")
    
    # Clay soils - slope stability risk
    high_risk_clay = clay_merged[clay_merged['factor_of_safety'] < 1.0]
    if len(high_risk_clay) > 0:
        print(f"\nClay Soils - Slope Stability Risk:")
        print(f"  • {len(high_risk_clay)} samples with FS < 1.0")
        print(f"  • Average shear strength: {high_risk_clay['undrained_shear_strength_kPa'].mean():.1f} kPa")
        print(f"  • Average plasticity index: {high_risk_clay['plasticity_index'].mean():.1f}")
    
    # Regional risk assessment
    print("\n🌍 REGIONAL RISK ASSESSMENT")
    print("-"*40)
    
    regional_risk = spatial_data['grid'].groupby('soil_type').agg({
        'shear_strength_kPa': 'mean',
        'bearing_capacity_kPa': 'mean',
        'erosion_risk': lambda x: (x == 'Very_High').sum() / len(x) * 100
    }).round(1)
    regional_risk.columns = ['Avg Shear Strength (kPa)', 'Avg Bearing Capacity (kPa)', '% Very High Erosion Risk']
    print(regional_risk.head(10))
    
    # Generate visualizations
    print("\n📈 GENERATING VISUALIZATIONS")
    print("-"*40)
    
    # Create output directory
    os.makedirs('output_figures', exist_ok=True)
    
    # Generate and save plots
    try:
        # 1. Grain size distribution
        fig1 = visualizer.plot_grain_size_distribution(sandy_merged)
        fig1.savefig('output_figures/grain_size_distribution.png', dpi=150, bbox_inches='tight')
        print("✓ Grain size distribution plot saved")
        
        # 2. Plasticity chart
        fig2 = visualizer.plot_plasticity_chart(clay_merged)
        fig2.savefig('output_figures/plasticity_chart.png', dpi=150, bbox_inches='tight')
        print("✓ Plasticity chart saved")
        
        # 3. Liquefaction assessment
        fig3 = visualizer.plot_liquefaction_assessment(sandy_merged)
        fig3.savefig('output_figures/liquefaction_assessment.png', dpi=150, bbox_inches='tight')
        print("✓ Liquefaction assessment plot saved")
        
        # 4. Failure analysis
        fig4 = visualizer.plot_failure_analysis(case_data['failures'])
        fig4.savefig('output_figures/failure_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Failure analysis plot saved")
        
        # 5. Correlation matrix for sandy soils
        fig5 = visualizer.plot_correlation_matrix(
            sandy_merged,
            columns=['sand_content_%', 'relative_density_%', 'friction_angle_deg', 
                    'void_ratio', 'CSR', 'CRR', 'safety_factor'],
            title="Sandy Soils Correlation Matrix"
        )
        fig5.savefig('output_figures/sandy_correlation_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Sandy soils correlation matrix saved")
        
        # 6. Correlation matrix for clay soils
        fig6 = visualizer.plot_correlation_matrix(
            clay_merged,
            columns=['liquid_limit_%', 'plasticity_index', 'water_content_%', 
                    'void_ratio', 'undrained_shear_strength_kPa', 'OCR'],
            title="Clay Soils Correlation Matrix"
        )
        fig6.savefig('output_figures/clay_correlation_matrix.png', dpi=150, bbox_inches='tight')
        print("✓ Clay soils correlation matrix saved")
        
    except Exception as e:
        print(f"⚠ Warning: Some visualizations could not be generated: {e}")
    
    # Export processed data
    print("\n💾 EXPORTING PROCESSED DATA")
    print("-"*40)
    
    # Create output directory
    os.makedirs('output_data', exist_ok=True)
    
    # Export merged datasets
    sandy_merged.to_csv('output_data/sandy_soils_merged.csv', index=False)
    print("✓ Sandy soils merged data exported")
    
    clay_merged.to_csv('output_data/clay_soils_merged.csv', index=False)
    print("✓ Clay soils merged data exported")
    
    # Export risk assessment
    risk_assessment = pd.DataFrame({
        'sandy_high_risk_samples': [len(high_risk_sandy)],
        'clay_high_risk_samples': [len(high_risk_clay)],
        'total_failure_cases': [len(case_data['failures'])],
        'average_economic_loss_million': [case_data['failures']['economic_loss_million_USD'].mean()],
        'average_recovery_days': [case_data['failures']['recovery_time_days'].mean()]
    })
    risk_assessment.to_csv('output_data/risk_assessment_summary.csv', index=False)
    print("✓ Risk assessment summary exported")
    
    print("\n" + "="*60)
    print("✅ ANALYSIS COMPLETE!")
    print("="*60)
    print("\n📂 Output Files:")
    print("  • Visualizations: ./output_figures/")
    print("  • Processed Data: ./output_data/")
    print("  • Raw Datasets: ./sandy_soils/, ./clay_soils/, ./case_studies/, ./spatial_data/")
    
    return {
        'sandy_data': sandy_merged,
        'clay_data': clay_merged,
        'case_data': case_data,
        'spatial_data': spatial_data
    }

if __name__ == "__main__":
    results = main()