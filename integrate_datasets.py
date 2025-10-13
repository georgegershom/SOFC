#!/usr/bin/env python3
"""
Integrate Primary and Secondary Datasets for Food Processing Firms Research
Creates a comprehensive dataset for analysis at the firm-year level
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_datasets():
    """Load primary and secondary datasets"""
    
    print("Loading primary and secondary datasets...")
    
    # Load primary survey data
    primary_df = pd.read_csv('/workspace/primary_survey_data.csv')
    
    # Load secondary macro data
    secondary_df = pd.read_csv('/workspace/secondary_macro_data.csv')
    
    return primary_df, secondary_df

def assign_survey_years(primary_df):
    """Assign survey years to firms based on survey dates"""
    
    # Convert survey_date to datetime
    primary_df['survey_date'] = pd.to_datetime(primary_df['survey_date'])
    
    # Extract year from survey date
    primary_df['survey_year'] = primary_df['survey_date'].dt.year
    
    # For this research, we'll assume all surveys were conducted in 2024
    # but firms report data for different years (2022-2024) based on their characteristics
    
    # Assign reporting years based on firm characteristics
    np.random.seed(42)
    reporting_years = []
    
    for i, row in primary_df.iterrows():
        # Larger firms and those with FDI are more likely to have recent data
        if row['firm_size'] == 'Large' and row['fdi_presence'] == 1:
            year = np.random.choice([2023, 2024], p=[0.3, 0.7])
        elif row['firm_size'] == 'Large':
            year = np.random.choice([2022, 2023, 2024], p=[0.2, 0.4, 0.4])
        elif row['fdi_presence'] == 1:
            year = np.random.choice([2022, 2023, 2024], p=[0.3, 0.4, 0.3])
        else:
            year = np.random.choice([2022, 2023, 2024], p=[0.4, 0.4, 0.2])
        
        reporting_years.append(year)
    
    primary_df['reporting_year'] = reporting_years
    
    return primary_df

def create_firm_year_panel(primary_df, secondary_df):
    """Create a firm-year panel dataset"""
    
    print("Creating firm-year panel dataset...")
    
    # For this cross-sectional study, each firm has one observation
    # But we merge with the secondary data based on their reporting year
    
    integrated_data = []
    
    for i, firm_row in primary_df.iterrows():
        reporting_year = firm_row['reporting_year']
        
        # Get corresponding secondary data for the reporting year
        secondary_year_data = secondary_df[secondary_df['year'] == reporting_year]
        
        if len(secondary_year_data) > 0:
            # Combine firm data with macro data
            combined_row = firm_row.to_dict()
            
            # Add secondary data (excluding year column to avoid duplication)
            secondary_dict = secondary_year_data.iloc[0].to_dict()
            for key, value in secondary_dict.items():
                if key != 'year':
                    combined_row[f'macro_{key}'] = value
            
            # Add the reporting year as macro_year for clarity
            combined_row['macro_year'] = reporting_year
            
            integrated_data.append(combined_row)
    
    # Create integrated DataFrame
    integrated_df = pd.DataFrame(integrated_data)
    
    return integrated_df

def add_calculated_variables(integrated_df):
    """Add calculated variables and indices"""
    
    print("Adding calculated variables and composite indices...")
    
    # Calculate composite scores for FDI constructs
    
    # Knowledge Absorption Score (average of 4 items)
    ka_cols = ['ka_acquire_external_knowledge', 'ka_assimilate_new_info', 
               'ka_transform_knowledge', 'ka_exploit_knowledge_commercially']
    integrated_df['knowledge_absorption_score'] = integrated_df[ka_cols].mean(axis=1).round(2)
    
    # Task Performance Score (average of 4 items)
    tp_cols = ['tp_work_quality_standards', 'tp_efficient_task_completion',
               'tp_productivity_levels', 'tp_goal_achievement']
    integrated_df['task_performance_score'] = integrated_df[tp_cols].mean(axis=1).round(2)
    
    # Innovation Score (average of 4 items)
    inn_cols = ['inn_product_innovation', 'inn_process_innovation',
                'inn_marketing_innovation', 'inn_organizational_innovation']
    integrated_df['innovation_score'] = integrated_df[inn_cols].mean(axis=1).round(2)
    
    # Firm Resources Score (normalized composite)
    # Normalize each component to 0-1 scale first
    fr_skilled_norm = (integrated_df['fr_skilled_labor_ratio'] - integrated_df['fr_skilled_labor_ratio'].min()) / \
                     (integrated_df['fr_skilled_labor_ratio'].max() - integrated_df['fr_skilled_labor_ratio'].min())
    
    fr_rd_norm = (integrated_df['fr_rd_spend_ratio'] - integrated_df['fr_rd_spend_ratio'].min()) / \
                 (integrated_df['fr_rd_spend_ratio'].max() - integrated_df['fr_rd_spend_ratio'].min())
    
    fr_liq_norm = (integrated_df['fr_liquidity_ratio'] - integrated_df['fr_liquidity_ratio'].min()) / \
                  (integrated_df['fr_liquidity_ratio'].max() - integrated_df['fr_liquidity_ratio'].min())
    
    # Composite firm resources score (0-5 scale)
    integrated_df['firm_resources_score'] = (
        (fr_skilled_norm + fr_rd_norm + integrated_df['fr_iot_adoption'] + fr_liq_norm) / 4 * 5
    ).round(2)
    
    # Overall Performance Index (composite of ROI, ROA, export intensity, market share, operational efficiency)
    # Normalize performance metrics
    roi_norm = (integrated_df['roi_percent'] - integrated_df['roi_percent'].min()) / \
               (integrated_df['roi_percent'].max() - integrated_df['roi_percent'].min())
    
    roa_norm = (integrated_df['roa_percent'] - integrated_df['roa_percent'].min()) / \
               (integrated_df['roa_percent'].max() - integrated_df['roa_percent'].min())
    
    export_norm = integrated_df['export_intensity_percent'] / 100  # Already 0-100
    
    market_norm = (integrated_df['market_share_percent'] - integrated_df['market_share_percent'].min()) / \
                  (integrated_df['market_share_percent'].max() - integrated_df['market_share_percent'].min())
    
    op_eff_norm = (integrated_df['operational_efficiency_score'] - 1) / 9  # 1-10 scale to 0-1
    
    # Overall performance index (0-10 scale)
    integrated_df['overall_performance_index'] = (
        (roi_norm + roa_norm + export_norm + market_norm + op_eff_norm) / 5 * 10
    ).round(2)
    
    # FDI Intensity (amount of FDI relative to firm size)
    integrated_df['fdi_intensity'] = np.where(
        integrated_df['fdi_presence'] == 1,
        integrated_df['fdi_amount_million_usd'] / integrated_df['total_assets_million_naira'] * 100,
        0
    ).round(3)
    
    # Firm Age Categories
    integrated_df['firm_age_category'] = pd.cut(
        integrated_df['firm_age'], 
        bins=[0, 5, 10, 20, 50], 
        labels=['Young (≤5 years)', 'Growing (6-10 years)', 'Mature (11-20 years)', 'Established (>20 years)']
    )
    
    # Export Orientation (binary based on export intensity)
    integrated_df['export_oriented'] = (integrated_df['export_intensity_percent'] > 10).astype(int)
    
    # Technology Adoption Level (based on IoT use and R&D spend)
    tech_conditions = [
        (integrated_df['fr_iot_adoption'] == 1) & (integrated_df['fr_rd_spend_ratio'] > 5),
        (integrated_df['fr_iot_adoption'] == 1) | (integrated_df['fr_rd_spend_ratio'] > 3),
        integrated_df['fr_rd_spend_ratio'] > 1
    ]
    tech_choices = ['High', 'Medium', 'Low']
    integrated_df['technology_adoption_level'] = np.select(tech_conditions, tech_choices, default='Very Low')
    
    return integrated_df

def add_interaction_terms(integrated_df):
    """Add interaction terms for moderation analysis"""
    
    print("Adding interaction terms for moderation analysis...")
    
    # FDI × Government Policy Interactions
    integrated_df['fdi_x_policy_effectiveness'] = (
        integrated_df['fdi_presence'] * integrated_df['policy_effectiveness_index']
    ).round(3)
    
    integrated_df['fdi_intensity_x_policy_effectiveness'] = (
        integrated_df['fdi_intensity'] * integrated_df['policy_effectiveness_index']
    ).round(3)
    
    # FDI × Macro Environment Interactions
    integrated_df['fdi_x_gdp_growth'] = (
        integrated_df['fdi_presence'] * integrated_df['macro_gdp_growth_rate']
    ).round(3)
    
    integrated_df['fdi_x_ease_of_business'] = (
        integrated_df['fdi_presence'] * integrated_df['macro_ease_of_doing_business_score']
    ).round(3)
    
    # Knowledge Absorption × Policy Interactions
    integrated_df['knowledge_absorption_x_policy'] = (
        integrated_df['knowledge_absorption_score'] * integrated_df['policy_effectiveness_index']
    ).round(3)
    
    return integrated_df

def create_final_dataset():
    """Create the final integrated dataset"""
    
    # Load datasets
    primary_df, secondary_df = load_datasets()
    
    # Assign survey years
    primary_df = assign_survey_years(primary_df)
    
    # Create firm-year panel
    integrated_df = create_firm_year_panel(primary_df, secondary_df)
    
    # Add calculated variables
    integrated_df = add_calculated_variables(integrated_df)
    
    # Add interaction terms
    integrated_df = add_interaction_terms(integrated_df)
    
    # Reorder columns for better organization
    id_cols = ['firm_id', 'survey_year', 'reporting_year', 'macro_year']
    
    firm_char_cols = [col for col in integrated_df.columns if col.startswith(('firm_', 'employees', 'total_assets', 'subsector', 'ownership'))]
    
    fdi_cols = [col for col in integrated_df.columns if col.startswith('fdi_') or col in ['knowledge_absorption_score', 'task_performance_score', 'innovation_score', 'firm_resources_score']]
    
    performance_cols = [col for col in integrated_df.columns if col.startswith(('roi_', 'roa_', 'export_', 'market_', 'operational_', 'overall_performance'))]
    
    policy_cols = [col for col in integrated_df.columns if col.startswith('gp_') or col == 'policy_effectiveness_index']
    
    macro_cols = [col for col in integrated_df.columns if col.startswith('macro_')]
    
    interaction_cols = [col for col in integrated_df.columns if '_x_' in col]
    
    other_cols = [col for col in integrated_df.columns if col not in id_cols + firm_char_cols + fdi_cols + performance_cols + policy_cols + macro_cols + interaction_cols]
    
    # Reorder columns
    column_order = id_cols + firm_char_cols + fdi_cols + performance_cols + policy_cols + macro_cols + interaction_cols + other_cols
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in integrated_df.columns]
    integrated_df = integrated_df[existing_columns]
    
    return integrated_df

if __name__ == "__main__":
    # Create the final integrated dataset
    final_df = create_final_dataset()
    
    # Save to CSV
    final_df.to_csv('/workspace/integrated_food_processing_dataset.csv', index=False)
    
    # Display summary statistics
    print(f"\nIntegrated Dataset Created Successfully!")
    print(f"Shape: {final_df.shape}")
    print(f"Columns: {len(final_df.columns)}")
    
    print(f"\nReporting Year Distribution:")
    print(final_df['reporting_year'].value_counts().sort_index())
    
    print(f"\nFDI Presence by Firm Size:")
    print(pd.crosstab(final_df['firm_size'], final_df['fdi_presence']))
    
    print(f"\nSample of key variables:")
    key_vars = ['firm_id', 'firm_size', 'fdi_presence', 'knowledge_absorption_score', 
                'overall_performance_index', 'policy_effectiveness_index', 'macro_gdp_growth_rate']
    print(final_df[key_vars].head(10))
    
    print(f"\nDataset saved as: integrated_food_processing_dataset.csv")
    print(f"Total firms: {len(final_df)}")
    print(f"Firms with FDI: {final_df['fdi_presence'].sum()}")
    print(f"Average performance index: {final_df['overall_performance_index'].mean():.2f}")