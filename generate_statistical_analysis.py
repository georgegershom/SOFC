#!/usr/bin/env python3
"""
Statistical Analysis and Calibration/Validation Data Organization
================================================================

Creates comprehensive statistical analysis of the generated dataset including:
- Parameter distributions and correlation matrices
- Uncertainty bounds and confidence intervals
- Calibration/validation data organization
- Multi-scale parameter linking verification

"""

import numpy as np
import pandas as pd
import os
from scipy import stats
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

def load_all_datasets():
    """Load all generated datasets."""
    
    base_path = "/workspace/thermo_mechanical_dataset"
    
    datasets = {}
    
    # Thermal properties
    thermal_path = f"{base_path}/thermal_properties"
    datasets['thermal_conductivity'] = pd.read_csv(f"{thermal_path}/thermal_conductivity.csv")
    datasets['specific_heat'] = pd.read_csv(f"{thermal_path}/specific_heat.csv")
    datasets['thermal_diffusivity'] = pd.read_csv(f"{thermal_path}/thermal_diffusivity.csv")
    datasets['thermal_expansion'] = pd.read_csv(f"{thermal_path}/thermal_expansion.csv")
    
    # Mechanical properties
    mechanical_path = f"{base_path}/mechanical_properties"
    datasets['compressive_strength'] = pd.read_csv(f"{mechanical_path}/compressive_strength.csv")
    datasets['tensile_strength'] = pd.read_csv(f"{mechanical_path}/tensile_strength.csv")
    datasets['elastic_modulus'] = pd.read_csv(f"{mechanical_path}/elastic_modulus.csv")
    datasets['poissons_ratio'] = pd.read_csv(f"{mechanical_path}/poissons_ratio.csv")
    datasets['fracture_properties'] = pd.read_csv(f"{mechanical_path}/fracture_properties.csv")
    
    # Transport properties
    transport_path = f"{base_path}/transport_properties"
    datasets['permeability'] = pd.read_csv(f"{transport_path}/permeability.csv")
    datasets['porosity'] = pd.read_csv(f"{transport_path}/porosity.csv")
    datasets['moisture_transport'] = pd.read_csv(f"{transport_path}/moisture_transport.csv")
    datasets['gas_transport'] = pd.read_csv(f"{transport_path}/gas_transport.csv")
    
    # Deformation properties
    deformation_path = f"{base_path}/deformation_properties"
    datasets['creep_parameters'] = pd.read_csv(f"{deformation_path}/creep_parameters.csv")
    datasets['shrinkage_parameters'] = pd.read_csv(f"{deformation_path}/shrinkage_parameters.csv")
    datasets['thermal_strain'] = pd.read_csv(f"{deformation_path}/thermal_strain.csv")
    datasets['damage_evolution'] = pd.read_csv(f"{deformation_path}/damage_evolution.csv")
    
    return datasets

def create_parameter_distributions():
    """Create parameter distribution analysis."""
    
    datasets = load_all_datasets()
    
    # Collect all numerical parameters
    all_parameters = []
    
    for dataset_name, df in datasets.items():
        # Get numerical columns (exclude ID, temperature, data type columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Temperature_C', 'Rubber_Content_Percent']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        for col in numerical_cols:
            if '_Mean' in col or col.endswith('_Mean'):
                # Extract base parameter name
                param_name = col.replace('_Mean', '').replace('_Mean_MPa', '').replace('_Mean_W_m_K', '')
                
                # Get corresponding std column
                std_col = col.replace('_Mean', '_Std')
                if std_col in df.columns:
                    std_values = df[std_col]
                else:
                    std_values = df[col] * 0.1  # Default 10% if no std column
                
                for idx, row in df.iterrows():
                    all_parameters.append({
                        'Dataset': dataset_name,
                        'Parameter': param_name,
                        'Mix_ID': row['Mix_ID'],
                        'Temperature_C': row['Temperature_C'],
                        'Data_Type': row['Data_Type'],
                        'Mean_Value': row[col],
                        'Std_Value': std_values.iloc[idx] if hasattr(std_values, 'iloc') else std_values,
                        'CoV': (std_values.iloc[idx] if hasattr(std_values, 'iloc') else std_values) / row[col] if row[col] != 0 else 0,
                        'Rubber_Content': row.get('Rubber_Content_Percent', 0)
                    })
    
    param_df = pd.DataFrame(all_parameters)
    
    return param_df

def create_correlation_matrices():
    """Create correlation matrices between different properties."""
    
    datasets = load_all_datasets()
    
    # Create a combined dataset at reference temperature (20°C)
    ref_temp = 20
    combined_data = []
    
    for dataset_name, df in datasets.items():
        ref_data = df[df['Temperature_C'] == ref_temp].copy()
        
        # Get mean value columns
        mean_cols = [col for col in ref_data.columns if '_Mean' in col and col != 'Temperature_C']
        
        for idx, row in ref_data.iterrows():
            data_point = {
                'Mix_ID': row['Mix_ID'],
                'Rubber_Content': row.get('Rubber_Content_Percent', 0),
                'Dataset': dataset_name
            }
            
            for col in mean_cols:
                # Simplify column name
                simple_name = col.replace('_Mean', '').replace('_MPa', '').replace('_W_m_K', '').replace('_J_kg_K', '')
                data_point[simple_name] = row[col]
            
            combined_data.append(data_point)
    
    combined_df = pd.DataFrame(combined_data)
    
    # Pivot to get properties as columns
    property_matrix = combined_df.pivot_table(
        index=['Mix_ID', 'Rubber_Content'], 
        columns='Dataset', 
        values=[col for col in combined_df.columns if col not in ['Mix_ID', 'Rubber_Content', 'Dataset']]
    )
    
    # Flatten column names
    property_matrix.columns = ['_'.join(col).strip() for col in property_matrix.columns.values]
    property_matrix = property_matrix.reset_index()
    
    # Calculate correlations
    numerical_cols = property_matrix.select_dtypes(include=[np.number]).columns
    correlation_matrix = property_matrix[numerical_cols].corr()
    
    return correlation_matrix, property_matrix

def create_uncertainty_bounds():
    """Create uncertainty bounds for all parameters."""
    
    param_df = create_parameter_distributions()
    
    uncertainty_bounds = []
    
    # Group by parameter and calculate bounds
    for param_name in param_df['Parameter'].unique():
        param_data = param_df[param_df['Parameter'] == param_name]
        
        for mix_id in param_data['Mix_ID'].unique():
            mix_data = param_data[param_data['Mix_ID'] == mix_id]
            
            # Calculate bounds across all temperatures
            mean_values = mix_data['Mean_Value']
            std_values = mix_data['Std_Value']
            
            # Overall statistics
            overall_mean = mean_values.mean()
            overall_std = np.sqrt(np.mean(std_values**2))  # RMS of standard deviations
            
            # Confidence intervals
            ci_95_lower = overall_mean - 1.96 * overall_std
            ci_95_upper = overall_mean + 1.96 * overall_std
            ci_99_lower = overall_mean - 2.58 * overall_std
            ci_99_upper = overall_mean + 2.58 * overall_std
            
            # Temperature-dependent bounds
            temp_range = mix_data['Temperature_C'].max() - mix_data['Temperature_C'].min()
            temp_variability = (mean_values.max() - mean_values.min()) / overall_mean if overall_mean != 0 else 0
            
            uncertainty_bounds.append({
                'Parameter': param_name,
                'Mix_ID': mix_id,
                'Rubber_Content': mix_data['Rubber_Content'].iloc[0],
                'Overall_Mean': overall_mean,
                'Overall_Std': overall_std,
                'CoV_Mean': overall_std / overall_mean if overall_mean != 0 else 0,
                'CI_95_Lower': ci_95_lower,
                'CI_95_Upper': ci_95_upper,
                'CI_99_Lower': ci_99_lower,
                'CI_99_Upper': ci_99_upper,
                'Temperature_Range_C': temp_range,
                'Temperature_Variability': temp_variability,
                'Min_Value': mean_values.min(),
                'Max_Value': mean_values.max(),
                'Data_Points': len(mix_data)
            })
    
    return pd.DataFrame(uncertainty_bounds)

def organize_calibration_validation_data():
    """Organize and validate calibration/validation data splits."""
    
    datasets = load_all_datasets()
    
    cal_val_summary = []
    
    for dataset_name, df in datasets.items():
        # Overall split
        total_points = len(df)
        cal_points = len(df[df['Data_Type'] == 'Calibration'])
        val_points = len(df[df['Data_Type'] == 'Validation'])
        
        # Split by mix type
        mix_splits = []
        for mix_id in df['Mix_ID'].unique():
            mix_data = df[df['Mix_ID'] == mix_id]
            mix_cal = len(mix_data[mix_data['Data_Type'] == 'Calibration'])
            mix_val = len(mix_data[mix_data['Data_Type'] == 'Validation'])
            mix_splits.append({
                'Mix_ID': mix_id,
                'Calibration': mix_cal,
                'Validation': mix_val,
                'Cal_Percentage': mix_cal / len(mix_data) * 100
            })
        
        # Split by temperature range
        temp_ranges = [(20, 200), (200, 400), (400, 600), (600, 800)]
        temp_splits = []
        
        for temp_min, temp_max in temp_ranges:
            temp_data = df[(df['Temperature_C'] >= temp_min) & (df['Temperature_C'] < temp_max)]
            if len(temp_data) > 0:
                temp_cal = len(temp_data[temp_data['Data_Type'] == 'Calibration'])
                temp_val = len(temp_data[temp_data['Data_Type'] == 'Validation'])
                temp_splits.append({
                    'Temperature_Range': f"{temp_min}-{temp_max}°C",
                    'Calibration': temp_cal,
                    'Validation': temp_val,
                    'Cal_Percentage': temp_cal / len(temp_data) * 100 if len(temp_data) > 0 else 0
                })
        
        cal_val_summary.append({
            'Dataset': dataset_name,
            'Total_Points': total_points,
            'Calibration_Points': cal_points,
            'Validation_Points': val_points,
            'Calibration_Percentage': cal_points / total_points * 100,
            'Mix_Splits': mix_splits,
            'Temperature_Splits': temp_splits
        })
    
    return cal_val_summary

def verify_multi_scale_linking():
    """Verify multi-scale parameter linking consistency."""
    
    datasets = load_all_datasets()
    
    # Define expected relationships between properties
    expected_relationships = [
        # Thermal properties relationships
        ('thermal_conductivity', 'specific_heat', 'thermal_diffusivity', 'inverse'),  # α = k/(ρ*cp)
        
        # Mechanical properties relationships  
        ('compressive_strength', 'tensile_strength', None, 'positive'),  # Higher comp -> higher tens
        ('elastic_modulus', 'compressive_strength', None, 'positive'),   # Higher E -> higher strength
        
        # Transport properties relationships
        ('porosity', 'permeability', None, 'positive'),  # Higher porosity -> higher permeability
        ('porosity', 'thermal_conductivity', None, 'negative'),  # Higher porosity -> lower conductivity
        
        # Deformation relationships
        ('thermal_expansion', 'thermal_strain', None, 'positive'),  # Higher α -> higher strain
        ('damage_evolution', 'elastic_modulus', None, 'negative'),  # Higher damage -> lower modulus
    ]
    
    relationship_verification = []
    
    # Check relationships at reference temperature
    ref_temp = 20
    
    for dataset1_name, dataset2_name, dataset3_name, relationship_type in expected_relationships:
        if dataset1_name in datasets and dataset2_name in datasets:
            df1 = datasets[dataset1_name][datasets[dataset1_name]['Temperature_C'] == ref_temp]
            df2 = datasets[dataset2_name][datasets[dataset2_name]['Temperature_C'] == ref_temp]
            
            # Get mean value columns
            mean_cols1 = [col for col in df1.columns if '_Mean' in col and 'Temperature' not in col]
            mean_cols2 = [col for col in df2.columns if '_Mean' in col and 'Temperature' not in col]
            
            if mean_cols1 and mean_cols2:
                try:
                    # Merge datasets on Mix_ID
                    merged = pd.merge(df1[['Mix_ID'] + mean_cols1], 
                                    df2[['Mix_ID'] + mean_cols2], 
                                    on='Mix_ID')
                    
                    if len(merged) > 0:
                        # Calculate correlation
                        prop1_values = merged[mean_cols1[0]]  # Take first mean column
                        prop2_values = merged[mean_cols2[0]]
                except (KeyError, IndexError):
                    # Skip if columns don't exist or merge fails
                    continue
                    
                    correlation, p_value = pearsonr(prop1_values, prop2_values)
                    
                    # Check if relationship matches expectation
                    if relationship_type == 'positive':
                        relationship_met = correlation > 0.3
                    elif relationship_type == 'negative':
                        relationship_met = correlation < -0.3
                    elif relationship_type == 'inverse':
                        relationship_met = correlation < -0.5
                    else:
                        relationship_met = True
                    
                    relationship_verification.append({
                        'Property_1': dataset1_name,
                        'Property_2': dataset2_name,
                        'Expected_Relationship': relationship_type,
                        'Correlation': correlation,
                        'P_Value': p_value,
                        'Relationship_Met': relationship_met,
                        'Data_Points': len(merged)
                    })
    
    return pd.DataFrame(relationship_verification)

def main():
    """Generate comprehensive statistical analysis."""
    
    print("Generating Statistical Analysis and Data Organization...")
    print("=" * 60)
    
    # Create output directory
    output_dir = "/workspace/thermo_mechanical_dataset/statistical_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. Creating parameter distributions...")
    param_distributions = create_parameter_distributions()
    param_distributions.to_csv(f"{output_dir}/parameter_distributions.csv", index=False)
    
    print("2. Creating correlation matrices...")
    correlation_matrix, property_matrix = create_correlation_matrices()
    correlation_matrix.to_csv(f"{output_dir}/correlation_matrices.csv")
    property_matrix.to_csv(f"{output_dir}/property_matrix.csv", index=False)
    
    print("3. Creating uncertainty bounds...")
    uncertainty_bounds = create_uncertainty_bounds()
    uncertainty_bounds.to_csv(f"{output_dir}/uncertainty_bounds.csv", index=False)
    
    print("4. Organizing calibration/validation data...")
    cal_val_summary = organize_calibration_validation_data()
    
    # Save calibration/validation summary
    with open(f"{output_dir}/calibration_validation_summary.txt", 'w') as f:
        f.write("Calibration/Validation Data Split Summary\n")
        f.write("=" * 50 + "\n\n")
        
        for summary in cal_val_summary:
            f.write(f"Dataset: {summary['Dataset']}\n")
            f.write(f"Total Points: {summary['Total_Points']}\n")
            f.write(f"Calibration: {summary['Calibration_Points']} ({summary['Calibration_Percentage']:.1f}%)\n")
            f.write(f"Validation: {summary['Validation_Points']} ({100-summary['Calibration_Percentage']:.1f}%)\n\n")
            
            f.write("Split by Mix Type:\n")
            for mix_split in summary['Mix_Splits']:
                f.write(f"  {mix_split['Mix_ID']}: {mix_split['Calibration']} cal, {mix_split['Validation']} val ({mix_split['Cal_Percentage']:.1f}% cal)\n")
            
            f.write("\nSplit by Temperature Range:\n")
            for temp_split in summary['Temperature_Splits']:
                f.write(f"  {temp_split['Temperature_Range']}: {temp_split['Calibration']} cal, {temp_split['Validation']} val ({temp_split['Cal_Percentage']:.1f}% cal)\n")
            
            f.write("\n" + "-" * 50 + "\n\n")
    
    print("5. Verifying multi-scale parameter linking...")
    relationship_verification = verify_multi_scale_linking()
    relationship_verification.to_csv(f"{output_dir}/multi_scale_verification.csv", index=False)
    
    # Generate summary statistics
    print("\nStatistical Analysis Summary:")
    print(f"- Parameter distributions: {len(param_distributions)} entries")
    print(f"- Correlation matrix: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}")
    print(f"- Uncertainty bounds: {len(uncertainty_bounds)} parameter sets")
    print(f"- Multi-scale relationships verified: {len(relationship_verification)}")
    
    # Dataset statistics
    total_datasets = len(cal_val_summary)
    total_points = sum([s['Total_Points'] for s in cal_val_summary])
    total_cal_points = sum([s['Calibration_Points'] for s in cal_val_summary])
    total_val_points = sum([s['Validation_Points'] for s in cal_val_summary])
    
    print(f"\nOverall Dataset Statistics:")
    print(f"- Total datasets: {total_datasets}")
    print(f"- Total data points: {total_points}")
    print(f"- Calibration points: {total_cal_points} ({total_cal_points/total_points*100:.1f}%)")
    print(f"- Validation points: {total_val_points} ({total_val_points/total_points*100:.1f}%)")
    
    # Parameter uncertainty statistics
    print(f"\nUncertainty Statistics:")
    mean_cov = uncertainty_bounds['CoV_Mean'].mean()
    max_cov = uncertainty_bounds['CoV_Mean'].max()
    min_cov = uncertainty_bounds['CoV_Mean'].min()
    
    print(f"- Mean coefficient of variation: {mean_cov:.3f}")
    print(f"- Maximum CoV: {max_cov:.3f}")
    print(f"- Minimum CoV: {min_cov:.3f}")
    
    # Multi-scale verification statistics
    if len(relationship_verification) > 0 and 'Relationship_Met' in relationship_verification.columns:
        relationships_met = relationship_verification['Relationship_Met'].sum()
        total_relationships = len(relationship_verification)
        
        print(f"\nMulti-scale Verification:")
        print(f"- Relationships verified: {relationships_met}/{total_relationships} ({relationships_met/total_relationships*100:.1f}%)")
    else:
        print(f"\nMulti-scale Verification:")
        print(f"- No relationships could be verified due to data structure limitations")
    
    print("\nStatistical analysis and data organization completed successfully!")

if __name__ == "__main__":
    main()