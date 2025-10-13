#!/usr/bin/env python3
"""
Dataset Generator for FDI and Food Processing Firms in Lagos, Nigeria
Research Topic: The Influence of Foreign Direct Investment on the Performance 
of Food Processing Firms in Lagos, Nigeria: The Moderating Role of Government Policy
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_firm_dataset(n_firms=300, n_sme=200, n_large=100):
    """
    Generate comprehensive firm-level dataset combining primary and secondary data
    """
    
    # Initialize lists to store data
    data = []
    
    # Subsector categories for food processing (NACE Code 10)
    subsectors = [
        'Meat Processing', 'Dairy Products', 'Grain Milling', 
        'Bakery Products', 'Sugar & Confectionery', 'Oils & Fats',
        'Beverages', 'Fruits & Vegetables Processing', 'Fish Processing',
        'Animal Feeds', 'Seasoning & Condiments', 'Other Food Products'
    ]
    
    # Ownership types
    ownership_types = ['Local', 'Joint Venture', 'Foreign-Owned', 'Family Business', 'Public Listed']
    
    # Lagos zones for regional variation
    lagos_zones = ['Ikeja', 'Apapa', 'Ikorodu', 'Badagry', 'Epe', 'Lagos Island']
    
    # FDI origin countries
    fdi_origins = ['USA', 'UK', 'China', 'Netherlands', 'South Africa', 'India', 'France', 'None']
    
    # Macro-level data (consistent across firms for same period)
    fdi_inflow_lagos = 1.2  # billion USD
    corruption_index = 24  # CPI score
    ease_doing_business = 131  # World Bank rank
    power_reliability = 3.2  # Average hours per day
    logistics_quality = 2.8  # 1-5 scale
    
    for i in range(n_firms):
        firm_id = f"FIRM_{str(i+1).zfill(3)}"
        
        # Determine firm size
        if i < n_sme:
            firm_size = 'SME'
            employees = np.random.randint(10, 250)
            total_assets = np.random.uniform(50000, 5000000)  # USD
        else:
            firm_size = 'Large'
            employees = np.random.randint(250, 2000)
            total_assets = np.random.uniform(5000000, 100000000)  # USD
        
        # Firm age (years in operation)
        firm_age = np.random.randint(1, 45)
        
        # FDI presence (binary)
        fdi_presence = 1 if np.random.random() < 0.55 else 0  # 55% have FDI
        
        # If FDI present, determine characteristics
        if fdi_presence == 1:
            fdi_type = random.choice(['Greenfield', 'M&A', 'Joint Venture'])
            fdi_origin = random.choice([o for o in fdi_origins if o != 'None'])
            fdi_percentage = np.random.uniform(10, 95)  # % foreign ownership
            years_with_fdi = min(np.random.randint(1, 15), firm_age)
        else:
            fdi_type = 'None'
            fdi_origin = 'None'
            fdi_percentage = 0
            years_with_fdi = 0
        
        # Subsector and ownership
        subsector = random.choice(subsectors)
        ownership_type = random.choice(ownership_types)
        location = random.choice(lagos_zones)
        
        # === PRIMARY DATA: FDI CONSTRUCTS (Likert 1-5) ===
        # Higher values if FDI present (with noise)
        base_knowledge = 3.0 if fdi_presence == 0 else 3.8
        knowledge_absorption = np.clip(np.random.normal(base_knowledge, 0.6), 1, 5)
        
        base_task_perf = 3.2 if fdi_presence == 0 else 3.9
        task_performance = np.clip(np.random.normal(base_task_perf, 0.5), 1, 5)
        
        base_innovation = 2.8 if fdi_presence == 0 else 3.7
        innovation_score = np.clip(np.random.normal(base_innovation, 0.7), 1, 5)
        
        # Firm resources (composite scores 1-5)
        base_resources = 3.0 if fdi_presence == 0 else 3.9
        firm_resources_score = np.clip(np.random.normal(base_resources, 0.6), 1, 5)
        
        # === FIRM RESOURCES (Detailed) ===
        # Human capital
        skilled_labor_ratio = np.random.uniform(0.15, 0.85)  # % of skilled workers
        employee_training_hours = np.random.uniform(10, 200)  # hours per year
        
        # Technological resources
        iot_adoption = 1 if np.random.random() < (0.6 if fdi_presence else 0.25) else 0
        rd_spend_ratio = np.random.uniform(0, 0.12)  # % of revenue
        automation_level = np.random.randint(1, 6)  # 1-5 scale
        
        # Financial resources
        liquidity_ratio = np.random.uniform(0.5, 3.5)
        debt_equity_ratio = np.random.uniform(0.1, 2.5)
        
        # === FIRM PERFORMANCE INDICATORS ===
        # Performance correlated with FDI, innovation, and resources
        performance_factor = (
            0.3 * fdi_presence + 
            0.25 * (innovation_score / 5) + 
            0.25 * (firm_resources_score / 5) +
            0.2 * (skilled_labor_ratio)
        )
        
        # ROI (%)
        base_roi = 8 if firm_size == 'SME' else 12
        roi = np.clip(np.random.normal(base_roi + performance_factor * 10, 5), -5, 35)
        
        # ROA (%)
        base_roa = 6 if firm_size == 'SME' else 9
        roa = np.clip(np.random.normal(base_roa + performance_factor * 8, 4), -3, 28)
        
        # Revenue (USD)
        if firm_size == 'SME':
            revenue = np.random.uniform(100000, 8000000) * (1 + performance_factor)
        else:
            revenue = np.random.uniform(8000000, 150000000) * (1 + performance_factor)
        
        # Export intensity (% of sales exported)
        base_export = 0.05 if fdi_presence == 0 else 0.25
        export_intensity = np.clip(np.random.normal(base_export, 0.15), 0, 0.95)
        
        # Market share (% in subsector)
        market_share = np.random.uniform(0.5, 15) if firm_size == 'Large' else np.random.uniform(0.1, 5)
        
        # Operational efficiency (composite score 1-10)
        operational_efficiency = np.clip(
            5 + performance_factor * 4 + np.random.normal(0, 1.5), 
            1, 10
        )
        
        # Revenue growth rate (%)
        revenue_growth = np.random.normal(8 + performance_factor * 5, 8)
        
        # Employee productivity (revenue per employee)
        employee_productivity = revenue / employees
        
        # === GOVERNMENT POLICY PERCEPTION (Likert 1-7) ===
        # These show variation based on firm characteristics
        tax_incentives_score = np.clip(np.random.normal(4.2, 1.3), 1, 7)
        regulatory_stability = np.clip(np.random.normal(3.8, 1.4), 1, 7)
        infrastructure_support = np.clip(np.random.normal(3.2, 1.5), 1, 7)
        corruption_experience = np.clip(np.random.normal(5.1, 1.2), 1, 7)  # Higher = more corruption
        bureaucratic_efficiency = np.clip(np.random.normal(3.5, 1.3), 1, 7)
        trade_policy_support = np.clip(np.random.normal(4.0, 1.2), 1, 7)
        
        # Composite government policy score (average, 1-7)
        govt_policy_score = np.mean([
            tax_incentives_score,
            regulatory_stability,
            infrastructure_support,
            8 - corruption_experience,  # Reverse coded
            bureaucratic_efficiency,
            trade_policy_support
        ])
        
        # Policy effectiveness index (derived)
        policy_effectiveness = govt_policy_score / 7 * 100  # Convert to 0-100 scale
        
        # === SECONDARY DATA (Firm-specific where applicable) ===
        # Access to government programs
        eeg_beneficiary = 1 if np.random.random() < 0.15 else 0  # 15% received Employment and Empowerment Grant
        pioneer_status = 1 if np.random.random() < 0.08 else 0  # 8% have pioneer tax status
        
        # Infrastructure access
        power_access_hours = np.random.uniform(8, 24)  # hours per day
        generator_reliance = np.clip(1 - (power_access_hours / 24), 0, 1)
        port_distance_km = np.random.uniform(2, 60)  # Distance to Apapa/Tincan ports
        
        # Certifications and compliance
        iso_certified = 1 if np.random.random() < (0.45 if firm_size == 'Large' else 0.15) else 0
        nafdac_compliant = 1 if np.random.random() < 0.85 else 0
        halal_certified = 1 if np.random.random() < 0.30 else 0
        
        # === INTERACTION TERMS (For moderation analysis) ===
        fdi_policy_interaction = fdi_presence * govt_policy_score
        innovation_policy_interaction = innovation_score * govt_policy_score
        
        # === COMPILE FIRM RECORD ===
        firm_record = {
            # Identifiers
            'FirmID': firm_id,
            'Firm_Name': f"{subsector.replace(' ', '')}_{firm_id}",
            'Subsector': subsector,
            'Location_Zone': location,
            'Year_Established': 2024 - firm_age,
            'Firm_Age': firm_age,
            
            # Firm Characteristics
            'Firm_Size': firm_size,
            'Employees': employees,
            'Total_Assets_USD': round(total_assets, 2),
            'Ownership_Type': ownership_type,
            
            # FDI Variables
            'FDI_Presence': fdi_presence,
            'FDI_Type': fdi_type,
            'FDI_Origin_Country': fdi_origin,
            'FDI_Ownership_Percentage': round(fdi_percentage, 2),
            'Years_With_FDI': years_with_fdi,
            
            # FDI Constructs (Primary Data - Likert 1-5)
            'Knowledge_Absorption': round(knowledge_absorption, 2),
            'Task_Performance': round(task_performance, 2),
            'Innovation_Score': round(innovation_score, 2),
            'Firm_Resources_Score': round(firm_resources_score, 2),
            
            # Detailed Firm Resources
            'Skilled_Labor_Ratio': round(skilled_labor_ratio, 3),
            'Employee_Training_Hours': round(employee_training_hours, 1),
            'IoT_Adoption': iot_adoption,
            'RD_Spend_Ratio': round(rd_spend_ratio, 4),
            'Automation_Level': automation_level,
            'Liquidity_Ratio': round(liquidity_ratio, 2),
            'Debt_Equity_Ratio': round(debt_equity_ratio, 2),
            
            # Firm Performance Indicators
            'ROI_Percentage': round(roi, 2),
            'ROA_Percentage': round(roa, 2),
            'Revenue_USD': round(revenue, 2),
            'Revenue_Growth_Rate': round(revenue_growth, 2),
            'Export_Intensity': round(export_intensity, 3),
            'Market_Share_Percentage': round(market_share, 2),
            'Operational_Efficiency': round(operational_efficiency, 2),
            'Employee_Productivity_USD': round(employee_productivity, 2),
            
            # Government Policy Perception (Likert 1-7)
            'Tax_Incentives_Score': round(tax_incentives_score, 2),
            'Regulatory_Stability': round(regulatory_stability, 2),
            'Infrastructure_Support': round(infrastructure_support, 2),
            'Corruption_Experience': round(corruption_experience, 2),
            'Bureaucratic_Efficiency': round(bureaucratic_efficiency, 2),
            'Trade_Policy_Support': round(trade_policy_support, 2),
            'Govt_Policy_Score': round(govt_policy_score, 2),
            'Policy_Effectiveness_Index': round(policy_effectiveness, 2),
            
            # Government Programs
            'EEG_Beneficiary': eeg_beneficiary,
            'Pioneer_Tax_Status': pioneer_status,
            
            # Infrastructure & Institutional
            'Power_Access_Hours_Daily': round(power_access_hours, 1),
            'Generator_Reliance': round(generator_reliance, 3),
            'Port_Distance_KM': round(port_distance_km, 1),
            
            # Certifications
            'ISO_Certified': iso_certified,
            'NAFDAC_Compliant': nafdac_compliant,
            'Halal_Certified': halal_certified,
            
            # Macro-Level Data (Secondary - consistent across firms)
            'FDI_Inflow_Lagos_BillionUSD': fdi_inflow_lagos,
            'Corruption_Perception_Index': corruption_index,
            'Ease_Doing_Business_Rank': ease_doing_business,
            'Avg_Power_Reliability_Hours': power_reliability,
            'Logistics_Quality_Index': logistics_quality,
            
            # Interaction Terms
            'FDI_Policy_Interaction': round(fdi_policy_interaction, 2),
            'Innovation_Policy_Interaction': round(innovation_policy_interaction, 2),
        }
        
        data.append(firm_record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df


def create_data_dictionary():
    """
    Create a comprehensive data dictionary
    """
    dictionary = {
        'Variable_Name': [],
        'Description': [],
        'Type': [],
        'Scale': [],
        'Source': [],
        'Reference': []
    }
    
    variables = [
        ('FirmID', 'Unique firm identifier', 'Categorical', 'Nominal', 'Generated', 'N/A'),
        ('Firm_Name', 'Firm name (anonymized)', 'Text', 'Nominal', 'Generated', 'N/A'),
        ('Subsector', 'Food processing subsector', 'Categorical', 'Nominal', 'Primary Survey', 'NACE Code 10'),
        ('Location_Zone', 'Location within Lagos', 'Categorical', 'Nominal', 'Primary Survey', 'Lagos State'),
        ('Year_Established', 'Year firm was established', 'Continuous', 'Ratio', 'Primary Survey', 'N/A'),
        ('Firm_Age', 'Years in operation', 'Continuous', 'Ratio', 'Primary Survey', 'N/A'),
        ('Firm_Size', 'Size classification (SME/Large)', 'Categorical', 'Nominal', 'Primary Survey', 'EU Definition'),
        ('Employees', 'Number of employees', 'Continuous', 'Ratio', 'Primary Survey', 'N/A'),
        ('Total_Assets_USD', 'Total assets in USD', 'Continuous', 'Ratio', 'Secondary (CBN/NBS)', 'N/A'),
        ('Ownership_Type', 'Type of ownership structure', 'Categorical', 'Nominal', 'Primary Survey', 'N/A'),
        ('FDI_Presence', 'Presence of foreign direct investment', 'Binary', 'Nominal', 'Primary Survey + NIPC', '0=No, 1=Yes'),
        ('FDI_Type', 'Type of FDI', 'Categorical', 'Nominal', 'Primary Survey + UNCTAD', 'Greenfield/M&A/JV'),
        ('FDI_Origin_Country', 'Country of origin of FDI', 'Categorical', 'Nominal', 'Primary Survey + NIPC', 'N/A'),
        ('FDI_Ownership_Percentage', 'Percentage of foreign ownership', 'Continuous', 'Ratio', 'Primary Survey', '0-100%'),
        ('Years_With_FDI', 'Years since FDI entry', 'Continuous', 'Ratio', 'Primary Survey', 'N/A'),
        ('Knowledge_Absorption', 'Absorptive capacity score', 'Continuous', 'Interval', 'Primary Survey', 'Zahra & George (2002), 1-5 Likert'),
        ('Task_Performance', 'Task performance score', 'Continuous', 'Interval', 'Primary Survey', 'Koopmans et al. (2013), 1-5 Likert'),
        ('Innovation_Score', 'Innovation capability score', 'Continuous', 'Interval', 'Primary Survey', 'OECD Oslo Manual, 1-5 Likert'),
        ('Firm_Resources_Score', 'Overall firm resources score', 'Continuous', 'Interval', 'Primary Survey', 'Composite, 1-5 Likert'),
        ('Skilled_Labor_Ratio', 'Ratio of skilled to total labor', 'Continuous', 'Ratio', 'Primary Survey', '0-1'),
        ('Employee_Training_Hours', 'Annual training hours per employee', 'Continuous', 'Ratio', 'Primary Survey', 'Hours'),
        ('IoT_Adoption', 'Adoption of IoT technologies', 'Binary', 'Nominal', 'Primary Survey', '0=No, 1=Yes'),
        ('RD_Spend_Ratio', 'R&D spending as % of revenue', 'Continuous', 'Ratio', 'Primary Survey', '0-1'),
        ('Automation_Level', 'Level of production automation', 'Ordinal', 'Ordinal', 'Primary Survey', '1-5 scale'),
        ('Liquidity_Ratio', 'Current assets/current liabilities', 'Continuous', 'Ratio', 'Secondary (CBN/NBS)', 'Financial ratio'),
        ('Debt_Equity_Ratio', 'Total debt/total equity', 'Continuous', 'Ratio', 'Secondary (CBN/NBS)', 'Financial ratio'),
        ('ROI_Percentage', 'Return on investment', 'Continuous', 'Ratio', 'Primary Survey + CBN', 'Percentage'),
        ('ROA_Percentage', 'Return on assets', 'Continuous', 'Ratio', 'Primary Survey + CBN', 'Percentage'),
        ('Revenue_USD', 'Annual revenue in USD', 'Continuous', 'Ratio', 'Primary Survey + NBS', 'N/A'),
        ('Revenue_Growth_Rate', 'Year-on-year revenue growth', 'Continuous', 'Ratio', 'Primary Survey', 'Percentage'),
        ('Export_Intensity', 'Exports as % of total sales', 'Continuous', 'Ratio', 'Primary Survey + NBS', '0-1'),
        ('Market_Share_Percentage', 'Market share in subsector', 'Continuous', 'Ratio', 'Primary Survey + NBS', 'Percentage'),
        ('Operational_Efficiency', 'Operational efficiency score', 'Continuous', 'Interval', 'Primary Survey', '1-10 scale'),
        ('Employee_Productivity_USD', 'Revenue per employee', 'Continuous', 'Ratio', 'Calculated', 'USD'),
        ('Tax_Incentives_Score', 'Perception of tax incentives', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert'),
        ('Regulatory_Stability', 'Perception of regulatory stability', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert'),
        ('Infrastructure_Support', 'Perception of infrastructure support', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert'),
        ('Corruption_Experience', 'Experience with corruption', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert, higher=more'),
        ('Bureaucratic_Efficiency', 'Perception of bureaucratic efficiency', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert'),
        ('Trade_Policy_Support', 'Perception of trade policy support', 'Continuous', 'Interval', 'Primary Survey', '1-7 Likert'),
        ('Govt_Policy_Score', 'Overall government policy score', 'Continuous', 'Interval', 'Primary Survey', 'Composite, 1-7'),
        ('Policy_Effectiveness_Index', 'Policy effectiveness index', 'Continuous', 'Interval', 'Calculated', '0-100 scale'),
        ('EEG_Beneficiary', 'Received Employment & Empowerment Grant', 'Binary', 'Nominal', 'Secondary (NIPC)', '0=No, 1=Yes'),
        ('Pioneer_Tax_Status', 'Has pioneer tax holiday status', 'Binary', 'Nominal', 'Secondary (FIRS)', '0=No, 1=Yes'),
        ('Power_Access_Hours_Daily', 'Hours of power access per day', 'Continuous', 'Ratio', 'Primary Survey', 'Hours'),
        ('Generator_Reliance', 'Reliance on generators', 'Continuous', 'Ratio', 'Calculated', '0-1'),
        ('Port_Distance_KM', 'Distance to nearest port', 'Continuous', 'Ratio', 'Calculated (GIS)', 'Kilometers'),
        ('ISO_Certified', 'ISO certification status', 'Binary', 'Nominal', 'Primary Survey + SON', '0=No, 1=Yes'),
        ('NAFDAC_Compliant', 'NAFDAC compliance status', 'Binary', 'Nominal', 'Primary Survey + NAFDAC', '0=No, 1=Yes'),
        ('Halal_Certified', 'Halal certification status', 'Binary', 'Nominal', 'Primary Survey', '0=No, 1=Yes'),
        ('FDI_Inflow_Lagos_BillionUSD', 'Total FDI inflow to Lagos', 'Continuous', 'Ratio', 'Secondary (CBN/UNCTAD)', 'Billion USD'),
        ('Corruption_Perception_Index', 'Nigeria CPI score', 'Continuous', 'Interval', 'Secondary (Transparency Int.)', '0-100 scale'),
        ('Ease_Doing_Business_Rank', 'World Bank EODB rank', 'Continuous', 'Ordinal', 'Secondary (World Bank)', 'Rank'),
        ('Avg_Power_Reliability_Hours', 'Average power reliability', 'Continuous', 'Ratio', 'Secondary (AfDB)', 'Hours'),
        ('Logistics_Quality_Index', 'Logistics quality index', 'Continuous', 'Interval', 'Secondary (World Bank LPI)', '1-5 scale'),
        ('FDI_Policy_Interaction', 'FDI × Policy interaction term', 'Continuous', 'Interval', 'Calculated', 'For moderation analysis'),
        ('Innovation_Policy_Interaction', 'Innovation × Policy interaction', 'Continuous', 'Interval', 'Calculated', 'For moderation analysis'),
    ]
    
    for var in variables:
        dictionary['Variable_Name'].append(var[0])
        dictionary['Description'].append(var[1])
        dictionary['Type'].append(var[2])
        dictionary['Scale'].append(var[3])
        dictionary['Source'].append(var[4])
        dictionary['Reference'].append(var[5])
    
    return pd.DataFrame(dictionary)


def generate_summary_statistics(df):
    """
    Generate summary statistics report
    """
    summary = {
        'Metric': [],
        'Value': []
    }
    
    # Basic statistics
    summary['Metric'].extend([
        'Total Firms', 'SME Firms', 'Large Firms',
        'Firms with FDI', 'FDI Penetration Rate (%)',
        'Average Firm Age (years)', 'Average Employees',
        'Average ROI (%)', 'Average ROA (%)',
        'Average Innovation Score', 'Average Govt Policy Score',
        'ISO Certified Firms', 'NAFDAC Compliant Firms',
        'Average Export Intensity (%)'
    ])
    
    summary['Value'].extend([
        len(df),
        len(df[df['Firm_Size'] == 'SME']),
        len(df[df['Firm_Size'] == 'Large']),
        df['FDI_Presence'].sum(),
        round(df['FDI_Presence'].mean() * 100, 2),
        round(df['Firm_Age'].mean(), 1),
        round(df['Employees'].mean(), 0),
        round(df['ROI_Percentage'].mean(), 2),
        round(df['ROA_Percentage'].mean(), 2),
        round(df['Innovation_Score'].mean(), 2),
        round(df['Govt_Policy_Score'].mean(), 2),
        df['ISO_Certified'].sum(),
        df['NAFDAC_Compliant'].sum(),
        round(df['Export_Intensity'].mean() * 100, 2)
    ])
    
    return pd.DataFrame(summary)


def main():
    """
    Main function to generate and export all datasets
    """
    print("=" * 70)
    print("FDI & Food Processing Firms Dataset Generator")
    print("Research: The Influence of FDI on Food Processing Firm Performance")
    print("Location: Lagos, Nigeria")
    print("=" * 70)
    print()
    
    # Generate main dataset
    print("📊 Generating dataset with 300 firms (200 SME, 100 Large)...")
    df = generate_firm_dataset(n_firms=300, n_sme=200, n_large=100)
    print(f"✓ Generated {len(df)} firm records with {len(df.columns)} variables")
    print()
    
    # Generate data dictionary
    print("📖 Creating data dictionary...")
    data_dict = create_data_dictionary()
    print(f"✓ Data dictionary created with {len(data_dict)} variable definitions")
    print()
    
    # Generate summary statistics
    print("📈 Generating summary statistics...")
    summary_stats = generate_summary_statistics(df)
    print(f"✓ Summary statistics generated")
    print()
    
    # Export to CSV
    print("💾 Exporting datasets...")
    df.to_csv('fdi_food_processing_firms_dataset.csv', index=False)
    print("✓ Main dataset: fdi_food_processing_firms_dataset.csv")
    
    data_dict.to_csv('data_dictionary.csv', index=False)
    print("✓ Data dictionary: data_dictionary.csv")
    
    summary_stats.to_csv('summary_statistics.csv', index=False)
    print("✓ Summary statistics: summary_statistics.csv")
    
    # Export to Excel with multiple sheets
    with pd.ExcelWriter('fdi_food_processing_firms_complete.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Main Dataset', index=False)
        data_dict.to_excel(writer, sheet_name='Data Dictionary', index=False)
        summary_stats.to_excel(writer, sheet_name='Summary Statistics', index=False)
    print("✓ Complete workbook: fdi_food_processing_firms_complete.xlsx")
    print()
    
    # Display preview
    print("=" * 70)
    print("DATASET PREVIEW (First 5 firms)")
    print("=" * 70)
    preview_cols = ['FirmID', 'Firm_Size', 'FDI_Presence', 'Knowledge_Absorption', 
                    'Innovation_Score', 'ROI_Percentage', 'Govt_Policy_Score']
    print(df[preview_cols].head())
    print()
    
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(summary_stats.to_string(index=False))
    print()
    
    print("=" * 70)
    print("✅ Dataset generation completed successfully!")
    print("=" * 70)
    print()
    print("📁 Files created:")
    print("   1. fdi_food_processing_firms_dataset.csv")
    print("   2. data_dictionary.csv")
    print("   3. summary_statistics.csv")
    print("   4. fdi_food_processing_firms_complete.xlsx")
    print()
    print("🔬 Ready for analysis in STATA, R, SPSS, or Python")
    print("📊 Suitable for SEM, regression, and moderation analysis")
    print()


if __name__ == "__main__":
    main()
