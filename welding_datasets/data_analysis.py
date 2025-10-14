"""
Data Analysis and Visualization for Welding Inverse Design Dataset
Provides exploratory data analysis, visualizations, and statistical summaries
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def load_datasets():
    """Load all datasets"""
    exp_data = pd.read_csv('tier1_experimental_data.csv')
    sim_data = pd.read_csv('tier2_computational_data.csv')
    master_data = pd.read_csv('master_dataset.csv')
    
    with open('dataset_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    return exp_data, sim_data, master_data, metadata


def print_dataset_summary(df, name="Dataset"):
    """Print comprehensive dataset summary"""
    print(f"\n{'='*80}")
    print(f"{name} Summary")
    print(f"{'='*80}")
    print(f"Total Samples: {len(df)}")
    print(f"Features: {len(df.columns)}")
    print(f"Missing Values: {df.isnull().sum().sum()}")
    print(f"\nData Source Distribution:")
    print(df['Data_Source'].value_counts())
    print(f"\nMaterial Combination Distribution:")
    print(df['Material_Combination'].value_counts())
    print(f"\nJoint Type Distribution:")
    print(df['Joint_Type'].value_counts())


def analyze_input_parameters(df):
    """Analyze input parameter distributions"""
    print(f"\n{'='*80}")
    print("Input Parameters Statistics")
    print(f"{'='*80}")
    
    input_params = [
        'Laser_Power_W', 'Welding_Speed_mm_s', 'Pulse_Frequency_Hz',
        'Pulse_Duration_ms', 'Beam_Focus_Position_mm', 'Beam_Spot_Size_um',
        'Clamping_Pressure_kPa', 'Shield_Gas_Flow_L_min',
        'Sheet_Thickness_mm', 'Overlap_Distance_mm', 'Heat_Input_J_mm'
    ]
    
    stats = df[input_params].describe()
    print(stats)
    
    return stats


def analyze_output_parameters(df):
    """Analyze output parameter distributions"""
    print(f"\n{'='*80}")
    print("Output Parameters Statistics")
    print(f"{'='*80}")
    
    # Morphology
    print("\n--- Weld Morphology ---")
    morphology_params = ['Nugget_Width_mm', 'Penetration_Depth_mm', 'HAZ_Width_mm']
    print(df[morphology_params].describe())
    
    # Defects
    print("\n--- Defect Rates ---")
    print(f"Cracks: {df['Has_Cracks'].mean()*100:.2f}%")
    print(f"Porosity: {df['Has_Porosity'].mean()*100:.2f}%")
    print(f"Undercut: {df['Has_Undercut'].mean()*100:.2f}%")
    print(f"Expulsion: {df['Has_Expulsion'].mean()*100:.2f}%")
    
    # Mechanical properties
    print("\n--- Mechanical & Electrical Properties ---")
    mech_params = ['Tensile_Shear_Strength_N', 'Peel_Strength_N', 'Contact_Resistance_uOhm']
    print(df[mech_params].describe())
    
    # Extreme-temperature performance
    print("\n--- Extreme-Temperature Performance ---")
    extreme_params = [
        'Thermal_Cycling_Strength_Degradation_pct',
        'Thermal_Cycling_Resistance_Increase_pct',
        'Cycles_to_Failure',
        'Creep_Time_to_Failure_hours',
        'Overall_Quality_Score'
    ]
    print(df[extreme_params].describe())


def analyze_correlations(df):
    """Analyze correlations between key parameters"""
    print(f"\n{'='*80}")
    print("Key Correlations Analysis")
    print(f"{'='*80}")
    
    # Select key features for correlation analysis
    key_features = [
        'Laser_Power_W', 'Welding_Speed_mm_s', 'Heat_Input_J_mm',
        'Nugget_Width_mm', 'Penetration_Depth_mm',
        'Tensile_Shear_Strength_N', 'Contact_Resistance_uOhm',
        'Cycles_to_Failure', 'Overall_Quality_Score'
    ]
    
    corr_matrix = df[key_features].corr()
    
    print("\nStrongest Positive Correlations with Overall_Quality_Score:")
    quality_corr = corr_matrix['Overall_Quality_Score'].sort_values(ascending=False)
    print(quality_corr.head(6))
    
    print("\nStrongest Correlations with Cycles_to_Failure:")
    cycles_corr = corr_matrix['Cycles_to_Failure'].sort_values(ascending=False)
    print(cycles_corr.head(6))
    
    return corr_matrix


def analyze_by_material(df):
    """Analyze performance by material combination"""
    print(f"\n{'='*80}")
    print("Performance by Material Combination")
    print(f"{'='*80}")
    
    metrics = [
        'Tensile_Shear_Strength_N',
        'Contact_Resistance_uOhm',
        'Cycles_to_Failure',
        'Overall_Quality_Score'
    ]
    
    material_analysis = df.groupby('Material_Combination')[metrics].agg(['mean', 'std'])
    print(material_analysis)
    
    return material_analysis


def analyze_defect_impact(df):
    """Analyze impact of defects on performance"""
    print(f"\n{'='*80}")
    print("Defect Impact Analysis")
    print(f"{'='*80}")
    
    # Create defect flag
    df['Any_Defect'] = (df['Has_Cracks'] | df['Has_Porosity'] | 
                        df['Has_Undercut'] | df['Has_Expulsion']).astype(int)
    
    print("\nPerformance Comparison: No Defects vs. With Defects")
    
    metrics = [
        'Tensile_Shear_Strength_N',
        'Contact_Resistance_uOhm',
        'Cycles_to_Failure',
        'Overall_Quality_Score'
    ]
    
    comparison = df.groupby('Any_Defect')[metrics].mean()
    comparison.index = ['No Defects', 'With Defects']
    print(comparison)
    
    print("\nPerformance Degradation due to Defects:")
    degradation = ((comparison.loc['No Defects'] - comparison.loc['With Defects']) / 
                   comparison.loc['No Defects'] * 100)
    print(degradation)


def compare_experimental_vs_computational(exp_df, sim_df):
    """Compare experimental and computational data"""
    print(f"\n{'='*80}")
    print("Experimental vs Computational Data Comparison")
    print(f"{'='*80}")
    
    metrics = [
        'Tensile_Shear_Strength_N',
        'Contact_Resistance_uOhm',
        'Cycles_to_Failure',
        'Overall_Quality_Score'
    ]
    
    exp_means = exp_df[metrics].mean()
    sim_means = sim_df[metrics].mean()
    
    comparison = pd.DataFrame({
        'Experimental': exp_means,
        'Computational': sim_means,
        'Difference (%)': ((sim_means - exp_means) / exp_means * 100)
    })
    
    print(comparison)
    
    print("\nDefect Rate Comparison:")
    print(f"Experimental Cracks: {exp_df['Has_Cracks'].mean()*100:.2f}%")
    print(f"Computational Cracks: {sim_df['Has_Cracks'].mean()*100:.2f}%")
    print(f"Experimental Porosity: {exp_df['Has_Porosity'].mean()*100:.2f}%")
    print(f"Computational Porosity: {sim_df['Has_Porosity'].mean()*100:.2f}%")


def identify_optimal_parameters(df, top_n=10):
    """Identify parameter sets with best performance"""
    print(f"\n{'='*80}")
    print(f"Top {top_n} Welds by Overall Quality Score")
    print(f"{'='*80}")
    
    top_welds = df.nlargest(top_n, 'Overall_Quality_Score')
    
    display_cols = [
        'Weld_ID', 'Material_Combination', 'Laser_Power_W', 'Welding_Speed_mm_s',
        'Heat_Input_J_mm', 'Tensile_Shear_Strength_N', 'Contact_Resistance_uOhm',
        'Cycles_to_Failure', 'Overall_Quality_Score'
    ]
    
    print(top_welds[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("Average Parameters for Top Performers")
    print("="*80)
    
    input_params = [
        'Laser_Power_W', 'Welding_Speed_mm_s', 'Pulse_Frequency_Hz',
        'Heat_Input_J_mm', 'Beam_Spot_Size_um'
    ]
    
    print(top_welds[input_params].mean())


def identify_failure_modes(df):
    """Identify common failure modes"""
    print(f"\n{'='*80}")
    print("Failure Mode Analysis")
    print(f"{'='*80}")
    
    # Low performance threshold
    threshold = df['Overall_Quality_Score'].quantile(0.25)
    poor_welds = df[df['Overall_Quality_Score'] < threshold]
    
    print(f"\nWelds with Quality Score < {threshold:.2f} (bottom 25%): {len(poor_welds)}")
    
    print("\nCommon Characteristics of Poor-Quality Welds:")
    print(f"Defect Rate: {poor_welds['Has_Cracks'].mean()*100:.2f}% (vs {df['Has_Cracks'].mean()*100:.2f}% overall)")
    print(f"Average Heat Input: {poor_welds['Heat_Input_J_mm'].mean():.2f} J/mm")
    print(f"Average Cycles to Failure: {poor_welds['Cycles_to_Failure'].mean():.0f} cycles")
    
    print("\nMaterial Distribution in Poor Welds:")
    print(poor_welds['Material_Combination'].value_counts(normalize=True) * 100)


def generate_summary_report():
    """Generate complete analysis report"""
    print("\n" + "="*80)
    print("WELDING INVERSE DESIGN DATASET - COMPREHENSIVE ANALYSIS REPORT")
    print("="*80)
    
    # Load data
    exp_df, sim_df, master_df, metadata = load_datasets()
    
    # Dataset summaries
    print_dataset_summary(master_df, "Master Dataset")
    
    # Input parameters
    analyze_input_parameters(master_df)
    
    # Output parameters
    analyze_output_parameters(master_df)
    
    # Correlations
    corr_matrix = analyze_correlations(master_df)
    
    # Material analysis
    material_analysis = analyze_by_material(master_df)
    
    # Defect impact
    analyze_defect_impact(master_df)
    
    # Experimental vs Computational
    compare_experimental_vs_computational(exp_df, sim_df)
    
    # Optimal parameters
    identify_optimal_parameters(master_df, top_n=10)
    
    # Failure modes
    identify_failure_modes(master_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Total samples analyzed: {len(master_df)}")
    print(f"Report generated successfully!")


def create_quick_stats():
    """Generate quick statistics for reference"""
    master_df = pd.read_csv('master_dataset.csv')
    
    quick_stats = {
        'total_samples': len(master_df),
        'experimental_samples': len(master_df[master_df['Data_Source'] == 'Experimental']),
        'computational_samples': len(master_df[master_df['Data_Source'] == 'Computational']),
        'avg_quality_score': float(master_df['Overall_Quality_Score'].mean()),
        'avg_tensile_strength': float(master_df['Tensile_Shear_Strength_N'].mean()),
        'avg_cycles_to_failure': float(master_df['Cycles_to_Failure'].mean()),
        'defect_rate': float(master_df['Has_Cracks'].mean() + master_df['Has_Porosity'].mean()),
        'materials': master_df['Material_Combination'].unique().tolist(),
        'best_material_quality': master_df.groupby('Material_Combination')['Overall_Quality_Score'].mean().to_dict()
    }
    
    with open('quick_statistics.json', 'w') as f:
        json.dump(quick_stats, f, indent=2)
    
    print("Quick statistics saved to: quick_statistics.json")
    return quick_stats


if __name__ == '__main__':
    # Generate full analysis report
    generate_summary_report()
    
    # Generate quick stats
    create_quick_stats()
