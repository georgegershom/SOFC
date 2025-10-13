#!/usr/bin/env python3
"""
Generate Primary Survey Data for Food Processing Firms in Lagos, Nigeria
Research Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_firm_ids(n_firms=300):
    """Generate unique firm IDs"""
    return [f"FIRM_{str(i+1).zfill(3)}" for i in range(n_firms)]

def generate_firm_characteristics():
    """Generate basic firm characteristics"""
    
    # Firm sizes based on stratified sampling (200 SMEs, 100 large)
    firm_sizes = ['SME'] * 200 + ['Large'] * 100
    random.shuffle(firm_sizes)
    
    # Firm ages (years in operation)
    firm_ages = np.random.gamma(2, 8, 300).astype(int)  # Skewed towards younger firms
    firm_ages = np.clip(firm_ages, 1, 50)  # Cap at 50 years
    
    # Number of employees
    employees = []
    for size in firm_sizes:
        if size == 'SME':
            emp = np.random.lognormal(3.5, 0.8)  # Log-normal for SMEs
            emp = int(np.clip(emp, 10, 249))  # SME definition: 10-249 employees
        else:
            emp = np.random.lognormal(6, 0.5)  # Larger for big firms
            emp = int(np.clip(emp, 250, 2000))
        employees.append(emp)
    
    # Total assets (in millions of Naira)
    assets = []
    for i, size in enumerate(firm_sizes):
        if size == 'SME':
            asset = np.random.lognormal(8, 1.2)  # Smaller assets for SMEs
            asset = np.clip(asset, 50, 999)  # 50M - 999M Naira
        else:
            asset = np.random.lognormal(10.5, 0.8)  # Larger assets
            asset = np.clip(asset, 1000, 50000)  # 1B - 50B Naira
        assets.append(round(asset, 2))
    
    # Food processing subsectors
    subsectors = np.random.choice([
        'Meat Processing', 'Dairy Products', 'Grain Milling', 
        'Bakery Products', 'Beverages', 'Fruits & Vegetables',
        'Fish Processing', 'Oil & Fats', 'Sugar & Confectionery'
    ], 300, p=[0.15, 0.12, 0.18, 0.15, 0.10, 0.12, 0.08, 0.06, 0.04])
    
    # Ownership types
    ownership_types = np.random.choice([
        'Local Private', 'Foreign', 'Joint Venture', 'Government'
    ], 300, p=[0.65, 0.20, 0.12, 0.03])
    
    return {
        'firm_size': firm_sizes,
        'firm_age': firm_ages,
        'employees': employees,
        'total_assets_million_naira': assets,
        'subsector': subsectors,
        'ownership_type': ownership_types
    }

def generate_fdi_constructs():
    """Generate FDI-related constructs using Likert scales"""
    
    # Knowledge Absorption (adapted from Zahra & George, 2002)
    # 5-point Likert scale: 1=Strongly Disagree, 5=Strongly Agree
    knowledge_absorption_items = {
        'ka_acquire_external_knowledge': np.random.normal(3.2, 0.8, 300),
        'ka_assimilate_new_info': np.random.normal(3.4, 0.7, 300),
        'ka_transform_knowledge': np.random.normal(3.1, 0.9, 300),
        'ka_exploit_knowledge_commercially': np.random.normal(3.3, 0.8, 300)
    }
    
    # Task Performance (adapted from Koopmans et al., 2013)
    task_performance_items = {
        'tp_work_quality_standards': np.random.normal(3.8, 0.6, 300),
        'tp_efficient_task_completion': np.random.normal(3.6, 0.7, 300),
        'tp_productivity_levels': np.random.normal(3.5, 0.8, 300),
        'tp_goal_achievement': np.random.normal(3.7, 0.6, 300)
    }
    
    # Innovation (adapted from OECD Oslo Manual)
    innovation_items = {
        'inn_product_innovation': np.random.normal(2.9, 1.0, 300),
        'inn_process_innovation': np.random.normal(3.1, 0.9, 300),
        'inn_marketing_innovation': np.random.normal(2.7, 1.1, 300),
        'inn_organizational_innovation': np.random.normal(3.0, 0.9, 300)
    }
    
    # Firm Resources
    firm_resources_items = {
        'fr_skilled_labor_ratio': np.random.beta(2, 3, 300) * 100,  # Percentage
        'fr_rd_spend_ratio': np.random.exponential(2, 300),  # % of revenue
        'fr_iot_adoption': np.random.choice([0, 1], 300, p=[0.7, 0.3]),  # Binary
        'fr_liquidity_ratio': np.random.gamma(2, 1.5, 300)
    }
    
    # Clip Likert scale items to 1-5 range
    for category in [knowledge_absorption_items, task_performance_items, innovation_items]:
        for key, values in category.items():
            category[key] = np.clip(values, 1, 5)
    
    # Clip resource ratios to reasonable ranges
    firm_resources_items['fr_skilled_labor_ratio'] = np.clip(firm_resources_items['fr_skilled_labor_ratio'], 5, 80)
    firm_resources_items['fr_rd_spend_ratio'] = np.clip(firm_resources_items['fr_rd_spend_ratio'], 0, 15)
    firm_resources_items['fr_liquidity_ratio'] = np.clip(firm_resources_items['fr_liquidity_ratio'], 0.5, 5.0)
    
    return {**knowledge_absorption_items, **task_performance_items, 
            **innovation_items, **firm_resources_items}

def generate_performance_metrics(firm_chars, fdi_constructs):
    """Generate firm performance metrics with realistic correlations"""
    
    n_firms = len(firm_chars['firm_size'])
    
    # Base performance influenced by firm characteristics and FDI constructs
    base_performance = np.zeros(n_firms)
    
    for i in range(n_firms):
        # Size effect
        size_effect = 0.3 if firm_chars['firm_size'][i] == 'Large' else 0
        
        # Age effect (inverted U-shape)
        age = firm_chars['firm_age'][i]
        age_effect = 0.02 * age - 0.0003 * age**2
        
        # FDI construct effects
        ka_avg = np.mean([fdi_constructs['ka_acquire_external_knowledge'][i],
                         fdi_constructs['ka_assimilate_new_info'][i],
                         fdi_constructs['ka_transform_knowledge'][i],
                         fdi_constructs['ka_exploit_knowledge_commercially'][i]])
        
        tp_avg = np.mean([fdi_constructs['tp_work_quality_standards'][i],
                         fdi_constructs['tp_efficient_task_completion'][i],
                         fdi_constructs['tp_productivity_levels'][i],
                         fdi_constructs['tp_goal_achievement'][i]])
        
        inn_avg = np.mean([fdi_constructs['inn_product_innovation'][i],
                          fdi_constructs['inn_process_innovation'][i],
                          fdi_constructs['inn_marketing_innovation'][i],
                          fdi_constructs['inn_organizational_innovation'][i]])
        
        base_performance[i] = size_effect + age_effect + 0.1*ka_avg + 0.15*tp_avg + 0.08*inn_avg
    
    # ROI (Return on Investment) - percentage
    roi = base_performance * 10 + np.random.normal(8, 4, n_firms)
    roi = np.clip(roi, -5, 35)  # Realistic range for ROI
    
    # ROA (Return on Assets) - percentage  
    roa = roi * 0.6 + np.random.normal(2, 2, n_firms)
    roa = np.clip(roa, -3, 25)
    
    # Export Intensity - percentage of revenue from exports
    export_intensity = np.random.exponential(8, n_firms)
    export_intensity = np.clip(export_intensity, 0, 60)
    
    # Market Share - percentage in their subsector
    market_share = np.random.gamma(1.5, 2, n_firms)
    market_share = np.clip(market_share, 0.1, 25)
    
    # Operational Efficiency Score (1-10 scale)
    op_efficiency = 5 + base_performance * 2 + np.random.normal(0, 1, n_firms)
    op_efficiency = np.clip(op_efficiency, 1, 10)
    
    return {
        'roi_percent': np.round(roi, 2),
        'roa_percent': np.round(roa, 2), 
        'export_intensity_percent': np.round(export_intensity, 1),
        'market_share_percent': np.round(market_share, 2),
        'operational_efficiency_score': np.round(op_efficiency, 1)
    }

def generate_government_policy_perception():
    """Generate government policy perception variables (7-point Likert scale)"""
    
    # 7-point scale: 1=Very Poor, 7=Excellent
    policy_items = {
        'gp_tax_incentives_effectiveness': np.random.normal(3.8, 1.2, 300),
        'gp_regulatory_stability': np.random.normal(3.2, 1.3, 300),
        'gp_infrastructure_support': np.random.normal(2.9, 1.1, 300),
        'gp_ease_of_doing_business': np.random.normal(3.5, 1.0, 300),
        'gp_corruption_experience': np.random.normal(4.2, 1.4, 300),  # Higher = more corruption experienced
        'gp_policy_consistency': np.random.normal(3.1, 1.2, 300),
        'gp_government_support_programs': np.random.normal(3.4, 1.1, 300)
    }
    
    # Clip to 1-7 range and round
    for key, values in policy_items.items():
        policy_items[key] = np.round(np.clip(values, 1, 7), 1)
    
    # Calculate composite Policy Effectiveness Index (1-7 scale)
    # Reverse corruption score for the index
    policy_effectiveness_index = (
        policy_items['gp_tax_incentives_effectiveness'] +
        policy_items['gp_regulatory_stability'] +
        policy_items['gp_infrastructure_support'] +
        policy_items['gp_ease_of_doing_business'] +
        (8 - policy_items['gp_corruption_experience']) +  # Reversed
        policy_items['gp_policy_consistency'] +
        policy_items['gp_government_support_programs']
    ) / 7
    
    policy_items['policy_effectiveness_index'] = np.round(policy_effectiveness_index, 2)
    
    return policy_items

def generate_fdi_presence_and_type():
    """Generate FDI presence and type variables"""
    
    # FDI Presence (binary)
    fdi_presence = np.random.choice([0, 1], 300, p=[0.4, 0.6])  # 60% have FDI
    
    # FDI Type (for firms with FDI presence)
    fdi_types = []
    fdi_amounts = []
    fdi_origin_countries = []
    
    for i, has_fdi in enumerate(fdi_presence):
        if has_fdi:
            # Type of FDI
            fdi_type = np.random.choice(['Greenfield', 'M&A', 'Joint Venture'], 
                                     p=[0.5, 0.3, 0.2])
            fdi_types.append(fdi_type)
            
            # FDI Amount (in millions USD)
            if fdi_type == 'Greenfield':
                amount = np.random.lognormal(2.5, 1.0)  # Higher for greenfield
            elif fdi_type == 'M&A':
                amount = np.random.lognormal(3.0, 0.8)  # Highest for M&A
            else:  # Joint Venture
                amount = np.random.lognormal(2.0, 0.9)  # Lower for JV
            
            fdi_amounts.append(round(amount, 2))
            
            # Origin Country
            origin = np.random.choice([
                'South Africa', 'UK', 'USA', 'Netherlands', 'China', 
                'India', 'Germany', 'France', 'UAE', 'Other'
            ], p=[0.25, 0.15, 0.12, 0.10, 0.08, 0.08, 0.06, 0.05, 0.05, 0.06])
            fdi_origin_countries.append(origin)
            
        else:
            fdi_types.append('None')
            fdi_amounts.append(0)
            fdi_origin_countries.append('None')
    
    return {
        'fdi_presence': fdi_presence,
        'fdi_type': fdi_types,
        'fdi_amount_million_usd': fdi_amounts,
        'fdi_origin_country': fdi_origin_countries
    }

def create_primary_dataset():
    """Create the complete primary survey dataset"""
    
    print("Generating primary survey data for 300 food processing firms in Lagos...")
    
    # Generate firm IDs
    firm_ids = generate_firm_ids(300)
    
    # Generate all components
    firm_chars = generate_firm_characteristics()
    fdi_constructs = generate_fdi_constructs()
    performance = generate_performance_metrics(firm_chars, fdi_constructs)
    gov_policy = generate_government_policy_perception()
    fdi_data = generate_fdi_presence_and_type()
    
    # Combine all data
    primary_data = {
        'firm_id': firm_ids,
        **firm_chars,
        **fdi_constructs,
        **performance,
        **gov_policy,
        **fdi_data
    }
    
    # Create DataFrame
    df = pd.DataFrame(primary_data)
    
    # Add survey metadata
    df['survey_date'] = pd.date_range(start='2024-01-15', end='2024-03-30', periods=300)
    df['respondent_position'] = np.random.choice([
        'CEO', 'Operations Manager', 'General Manager', 'Production Manager', 
        'Finance Manager', 'Business Development Manager'
    ], 300, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    
    # Add data quality indicators
    df['response_completeness'] = np.random.uniform(0.85, 1.0, 300)  # 85-100% complete
    df['response_time_minutes'] = np.random.gamma(3, 8, 300).astype(int)  # Survey completion time
    
    return df

if __name__ == "__main__":
    # Generate the primary dataset
    primary_df = create_primary_dataset()
    
    # Save to CSV
    primary_df.to_csv('/workspace/primary_survey_data.csv', index=False)
    
    # Display basic statistics
    print(f"\nPrimary Dataset Generated Successfully!")
    print(f"Shape: {primary_df.shape}")
    print(f"Columns: {len(primary_df.columns)}")
    print(f"\nSample of first 5 rows:")
    print(primary_df.head())
    
    print(f"\nFirm Size Distribution:")
    print(primary_df['firm_size'].value_counts())
    
    print(f"\nFDI Presence:")
    print(primary_df['fdi_presence'].value_counts())
    
    print(f"\nSubsector Distribution:")
    print(primary_df['subsector'].value_counts())