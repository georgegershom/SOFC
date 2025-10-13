"""
Food Processing Firms FDI Dataset Generator
============================================
Generates synthetic dataset for studying the influence of FDI on food processing firms
in Lagos, Nigeria, with government policy as moderating variable.

Research Topic: The Influence of Foreign Direct Investment on the Performance of 
Food Processing Firms in Lagos, Nigeria: The Moderating Role of Nigerian Government Policy

Author: Dataset Generator
Date: 2024
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FDIDatasetGenerator:
    """Generate synthetic firm-level survey data for FDI study"""
    
    def __init__(self, n_firms=500):
        self.n_firms = n_firms
        self.subsectors = [
            'Dairy Processing', 'Grain Milling', 'Beverages', 
            'Meat Processing', 'Fish Processing', 'Fruits & Vegetables',
            'Bakery Products', 'Confectionery', 'Oil & Fats', 'Other Food'
        ]
        self.ownership_types = [
            'Fully Nigerian', 'Foreign Majority', 'Joint Venture', 
            'Foreign Subsidiary', 'Nigerian Majority JV'
        ]
        self.firm_sizes = ['Micro', 'Small', 'Medium', 'Large']
        
    def generate_control_variables(self):
        """Generate firm control variables"""
        data = {}
        
        # Firm ID
        data['firm_id'] = [f'FIRM_{str(i+1).zfill(4)}' for i in range(self.n_firms)]
        
        # Survey response date
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 10, 1)
        data['survey_date'] = [start_date + timedelta(
            days=random.randint(0, (end_date - start_date).days)
        ) for _ in range(self.n_firms)]
        
        # Subsector with realistic distribution
        subsector_weights = [0.15, 0.12, 0.18, 0.08, 0.10, 0.09, 0.11, 0.07, 0.06, 0.04]
        data['subsector'] = np.random.choice(
            self.subsectors, size=self.n_firms, p=subsector_weights
        )
        
        # Ownership type (influences FDI exposure)
        ownership_weights = [0.35, 0.20, 0.25, 0.10, 0.10]
        data['ownership_type'] = np.random.choice(
            self.ownership_types, size=self.n_firms, p=ownership_weights
        )
        
        # Firm age (years) - log-normal distribution
        data['firm_age'] = np.round(np.random.lognormal(2.5, 0.8, self.n_firms))
        data['firm_age'] = np.clip(data['firm_age'], 1, 60)
        
        # Number of employees - varies by ownership and subsector
        employees = []
        for i in range(self.n_firms):
            if data['ownership_type'][i] in ['Foreign Majority', 'Foreign Subsidiary']:
                base = np.random.lognormal(5, 1.2)
            else:
                base = np.random.lognormal(4, 1.1)
            
            if data['subsector'][i] in ['Beverages', 'Dairy Processing', 'Oil & Fats']:
                base *= 1.5
                
            employees.append(int(np.clip(base, 5, 5000)))
        data['num_employees'] = employees
        
        # Firm size category based on employees
        data['firm_size'] = pd.cut(
            data['num_employees'],
            bins=[0, 10, 50, 250, 10000],
            labels=self.firm_sizes
        )
        
        # Total assets (million Naira) - correlated with employees
        data['total_assets_mn'] = np.round(
            data['num_employees'] * np.random.uniform(2, 8, self.n_firms) +
            np.random.normal(50, 20, self.n_firms), 2
        )
        data['total_assets_mn'] = np.clip(data['total_assets_mn'], 10, 50000)
        
        # Export intensity (% of revenue from exports)
        export_base = np.random.beta(2, 5, self.n_firms) * 100
        # Foreign firms export more
        for i in range(self.n_firms):
            if 'Foreign' in data['ownership_type'][i]:
                export_base[i] *= 1.5
        data['export_intensity'] = np.round(np.clip(export_base, 0, 100), 1)
        
        return pd.DataFrame(data)
    
    def generate_fdi_constructs(self, control_df):
        """Generate FDI-related constructs using Likert scales"""
        data = {}
        
        # Knowledge Absorption (Zahra & George, 2002) - 5 items, 1-5 scale
        # Higher for foreign-affiliated firms
        base_knowledge = []
        for _, row in control_df.iterrows():
            if 'Foreign' in row['ownership_type']:
                mean = 3.8
            else:
                mean = 3.2
            base_knowledge.append(mean)
        
        data['knowledge_acquisition'] = np.round(np.random.normal(base_knowledge, 0.7, self.n_firms))
        data['knowledge_assimilation'] = np.round(np.random.normal(base_knowledge, 0.6, self.n_firms))
        data['knowledge_transformation'] = np.round(np.random.normal(base_knowledge, 0.8, self.n_firms))
        data['knowledge_exploitation'] = np.round(np.random.normal(base_knowledge, 0.7, self.n_firms))
        
        # Clip to valid Likert range
        for col in ['knowledge_acquisition', 'knowledge_assimilation', 
                   'knowledge_transformation', 'knowledge_exploitation']:
            data[col] = np.clip(data[col], 1, 5)
        
        # Calculate composite knowledge absorption score
        data['knowledge_absorption_avg'] = np.round(
            (data['knowledge_acquisition'] + data['knowledge_assimilation'] + 
             data['knowledge_transformation'] + data['knowledge_exploitation']) / 4, 2
        )
        
        # Task Performance (Koopmans et al., 2013) - 1-5 scale
        # Correlated with knowledge absorption
        data['task_quality'] = np.round(
            data['knowledge_absorption_avg'] * 0.6 + 
            np.random.normal(1.8, 0.5, self.n_firms)
        )
        data['task_efficiency'] = np.round(
            data['knowledge_absorption_avg'] * 0.5 + 
            np.random.normal(2, 0.6, self.n_firms)
        )
        data['task_innovation'] = np.round(
            data['knowledge_absorption_avg'] * 0.7 + 
            np.random.normal(1.5, 0.6, self.n_firms)
        )
        
        for col in ['task_quality', 'task_efficiency', 'task_innovation']:
            data[col] = np.clip(data[col], 1, 5)
        
        data['task_performance_avg'] = np.round(
            (data['task_quality'] + data['task_efficiency'] + data['task_innovation']) / 3, 2
        )
        
        # Innovation (Oslo Manual) - 1-5 scale
        data['product_innovation'] = np.round(
            data['knowledge_absorption_avg'] * 0.8 + 
            np.random.normal(1.2, 0.7, self.n_firms)
        )
        data['process_innovation'] = np.round(
            data['knowledge_absorption_avg'] * 0.7 + 
            np.random.normal(1.5, 0.6, self.n_firms)
        )
        data['marketing_innovation'] = np.round(
            data['knowledge_absorption_avg'] * 0.6 + 
            np.random.normal(1.8, 0.8, self.n_firms)
        )
        data['organizational_innovation'] = np.round(
            data['knowledge_absorption_avg'] * 0.5 + 
            np.random.normal(2, 0.7, self.n_firms)
        )
        
        for col in ['product_innovation', 'process_innovation', 
                   'marketing_innovation', 'organizational_innovation']:
            data[col] = np.clip(data[col], 1, 5)
        
        data['innovation_avg'] = np.round(
            (data['product_innovation'] + data['process_innovation'] + 
             data['marketing_innovation'] + data['organizational_innovation']) / 4, 2
        )
        
        # FDI Intensity (% of foreign ownership)
        fdi_intensity = []
        for _, row in control_df.iterrows():
            if row['ownership_type'] == 'Fully Nigerian':
                fdi_intensity.append(0)
            elif row['ownership_type'] == 'Foreign Subsidiary':
                fdi_intensity.append(np.random.uniform(90, 100))
            elif row['ownership_type'] == 'Foreign Majority':
                fdi_intensity.append(np.random.uniform(51, 89))
            elif row['ownership_type'] == 'Joint Venture':
                fdi_intensity.append(np.random.uniform(40, 60))
            else:  # Nigerian Majority JV
                fdi_intensity.append(np.random.uniform(10, 49))
        
        data['fdi_ownership_pct'] = np.round(fdi_intensity, 1)
        
        return pd.DataFrame(data)
    
    def generate_firm_performance(self, control_df, fdi_df):
        """Generate firm performance metrics"""
        data = {}
        
        # Base performance influenced by FDI constructs
        base_performance = (
            fdi_df['knowledge_absorption_avg'] * 0.3 +
            fdi_df['task_performance_avg'] * 0.3 +
            fdi_df['innovation_avg'] * 0.2 +
            np.random.normal(3, 0.5, self.n_firms)
        ) / 5  # Normalize to 0-1 scale
        
        # ROI (%) - influenced by FDI and firm characteristics
        roi_base = base_performance * 25 + np.random.normal(10, 5, self.n_firms)
        # Larger firms tend to have more stable ROI
        for i in range(self.n_firms):
            if control_df.iloc[i]['firm_size'] == 'Large':
                roi_base[i] += 5
            elif control_df.iloc[i]['firm_size'] == 'Micro':
                roi_base[i] -= 3
        
        data['roi_pct'] = np.round(np.clip(roi_base, -10, 50), 2)
        
        # ROA (%) - correlated with ROI but slightly lower
        data['roa_pct'] = np.round(
            data['roi_pct'] * 0.7 + np.random.normal(0, 3, self.n_firms), 2
        )
        data['roa_pct'] = np.clip(data['roa_pct'], -15, 40)
        
        # Revenue growth (% YoY)
        data['revenue_growth_pct'] = np.round(
            base_performance * 30 + np.random.normal(5, 8, self.n_firms), 2
        )
        data['revenue_growth_pct'] = np.clip(data['revenue_growth_pct'], -20, 60)
        
        # Market share (%) - influenced by size and performance
        market_share = []
        for i in range(self.n_firms):
            if control_df.iloc[i]['firm_size'] == 'Large':
                base = np.random.uniform(5, 25)
            elif control_df.iloc[i]['firm_size'] == 'Medium':
                base = np.random.uniform(2, 10)
            elif control_df.iloc[i]['firm_size'] == 'Small':
                base = np.random.uniform(0.5, 3)
            else:
                base = np.random.uniform(0.1, 1)
            
            base *= (1 + base_performance[i] * 0.5)
            market_share.append(base)
        
        data['market_share_pct'] = np.round(market_share, 2)
        
        # Operational efficiency (1-5 scale)
        data['operational_efficiency'] = np.round(
            fdi_df['task_performance_avg'] * 0.7 + 
            np.random.normal(1.5, 0.5, self.n_firms)
        )
        data['operational_efficiency'] = np.clip(data['operational_efficiency'], 1, 5)
        
        # Productivity (output per employee, index)
        data['productivity_index'] = np.round(
            base_performance * 100 + np.random.normal(80, 15, self.n_firms), 1
        )
        data['productivity_index'] = np.clip(data['productivity_index'], 40, 150)
        
        return pd.DataFrame(data)
    
    def generate_firm_resources(self, control_df, fdi_df):
        """Generate firm resource variables"""
        data = {}
        
        # Human Capital
        # Skilled labor ratio (%)
        skilled_base = fdi_df['fdi_ownership_pct'] * 0.3 + 30
        data['skilled_labor_ratio'] = np.round(
            skilled_base + np.random.normal(0, 10, self.n_firms), 1
        )
        data['skilled_labor_ratio'] = np.clip(data['skilled_labor_ratio'], 5, 90)
        
        # Training hours per employee per year
        data['training_hours_per_emp'] = np.round(
            fdi_df['knowledge_absorption_avg'] * 10 + 
            np.random.normal(20, 10, self.n_firms), 1
        )
        data['training_hours_per_emp'] = np.clip(data['training_hours_per_emp'], 0, 100)
        
        # Management quality (1-5 scale)
        data['management_quality'] = np.round(
            fdi_df['task_performance_avg'] * 0.8 + 
            np.random.normal(1, 0.5, self.n_firms)
        )
        data['management_quality'] = np.clip(data['management_quality'], 1, 5)
        
        # Technological Resources
        # IoT adoption (binary)
        iot_prob = fdi_df['innovation_avg'] / 5 * 0.6 + 0.1
        data['iot_adoption'] = np.random.binomial(1, iot_prob)
        
        # R&D spending (% of revenue)
        rd_base = fdi_df['innovation_avg'] * 0.8
        data['rd_spend_pct'] = np.round(
            rd_base + np.random.exponential(0.5, self.n_firms), 2
        )
        data['rd_spend_pct'] = np.clip(data['rd_spend_pct'], 0, 15)
        
        # Technology sophistication (1-5 scale)
        data['tech_sophistication'] = np.round(
            fdi_df['innovation_avg'] * 0.7 + 
            data['iot_adoption'] * 0.5 +
            np.random.normal(1, 0.6, self.n_firms)
        )
        data['tech_sophistication'] = np.clip(data['tech_sophistication'], 1, 5)
        
        # ERP system adoption (binary)
        erp_prob = (control_df['firm_size'].map({'Micro': 0.1, 'Small': 0.3, 
                                                 'Medium': 0.6, 'Large': 0.9}))
        data['erp_adoption'] = np.random.binomial(1, erp_prob)
        
        # Financial Resources
        # Liquidity ratio
        data['liquidity_ratio'] = np.round(
            np.random.gamma(2, 0.5, self.n_firms), 2
        )
        data['liquidity_ratio'] = np.clip(data['liquidity_ratio'], 0.5, 5)
        
        # Access to credit (1-5 scale)
        credit_base = []
        for i in range(self.n_firms):
            if control_df.iloc[i]['firm_size'] == 'Large':
                credit_base.append(4)
            elif control_df.iloc[i]['firm_size'] == 'Medium':
                credit_base.append(3.5)
            elif control_df.iloc[i]['firm_size'] == 'Small':
                credit_base.append(2.8)
            else:
                credit_base.append(2)
        
        data['access_to_credit'] = np.round(
            credit_base + np.random.normal(0, 0.7, self.n_firms)
        )
        data['access_to_credit'] = np.clip(data['access_to_credit'], 1, 5)
        
        # Investment capacity (million Naira)
        data['investment_capacity_mn'] = np.round(
            control_df['total_assets_mn'] * 0.15 * data['liquidity_ratio'] +
            np.random.normal(0, 50, self.n_firms), 2
        )
        data['investment_capacity_mn'] = np.clip(data['investment_capacity_mn'], 0, 5000)
        
        return pd.DataFrame(data)
    
    def generate_government_policy(self, control_df, fdi_df, performance_df):
        """Generate government policy perception variables (1-7 scale)"""
        data = {}
        
        # Base policy perception influenced by firm performance
        base_perception = 4 + (performance_df['roi_pct'] / 50) * 2
        
        # Tax incentives effectiveness
        data['tax_incentives_eff'] = np.round(
            base_perception + np.random.normal(0, 1, self.n_firms)
        )
        data['tax_incentives_eff'] = np.clip(data['tax_incentives_eff'], 1, 7)
        
        # Regulatory stability
        data['regulatory_stability'] = np.round(
            base_perception - 0.5 + np.random.normal(0, 1.2, self.n_firms)
        )
        data['regulatory_stability'] = np.clip(data['regulatory_stability'], 1, 7)
        
        # Infrastructure support
        data['infrastructure_support'] = np.round(
            base_perception - 1 + np.random.normal(0, 1.3, self.n_firms)
        )
        data['infrastructure_support'] = np.clip(data['infrastructure_support'], 1, 7)
        
        # Trade facilitation
        trade_base = base_perception + (control_df['export_intensity'] / 100) * 2
        data['trade_facilitation'] = np.round(
            trade_base + np.random.normal(-0.5, 1, self.n_firms)
        )
        data['trade_facilitation'] = np.clip(data['trade_facilitation'], 1, 7)
        
        # Investment protection
        invest_base = []
        for i in range(self.n_firms):
            if 'Foreign' in control_df.iloc[i]['ownership_type']:
                invest_base.append(base_perception[i] + 0.5)
            else:
                invest_base.append(base_perception[i])
        
        data['investment_protection'] = np.round(
            invest_base + np.random.normal(0, 0.9, self.n_firms)
        )
        data['investment_protection'] = np.clip(data['investment_protection'], 1, 7)
        
        # Bureaucratic efficiency (reverse coded - lower is better)
        data['bureaucratic_burden'] = np.round(
            8 - base_perception + np.random.normal(0, 1.1, self.n_firms)
        )
        data['bureaucratic_burden'] = np.clip(data['bureaucratic_burden'], 1, 7)
        
        # Corruption experience (reverse coded - lower is better)
        corruption_base = 8 - base_perception + np.random.exponential(0.5, self.n_firms)
        data['corruption_experience'] = np.round(corruption_base)
        data['corruption_experience'] = np.clip(data['corruption_experience'], 1, 7)
        
        # Policy consistency
        data['policy_consistency'] = np.round(
            base_perception - 0.3 + np.random.normal(0, 1, self.n_firms)
        )
        data['policy_consistency'] = np.clip(data['policy_consistency'], 1, 7)
        
        # Government support programs utilization (binary)
        support_prob = base_perception / 10 + 0.2
        data['govt_support_utilized'] = np.random.binomial(1, support_prob)
        
        # Overall policy effectiveness index (composite)
        data['policy_effectiveness_index'] = np.round(
            (data['tax_incentives_eff'] + data['regulatory_stability'] + 
             data['infrastructure_support'] + data['trade_facilitation'] +
             data['investment_protection'] + (8 - data['bureaucratic_burden']) +
             (8 - data['corruption_experience']) + data['policy_consistency']) / 8, 2
        )
        
        return pd.DataFrame(data)
    
    def add_interactions_and_moderations(self, full_df):
        """Add interaction terms and moderation effects"""
        
        # Create FDI composite score
        full_df['fdi_composite'] = (
            full_df['fdi_ownership_pct'] / 100 * 0.3 +
            full_df['knowledge_absorption_avg'] / 5 * 0.35 +
            full_df['innovation_avg'] / 5 * 0.35
        )
        
        # Create performance composite score
        full_df['performance_composite'] = (
            (full_df['roi_pct'] + 50) / 100 * 0.25 +  # Normalize ROI
            (full_df['roa_pct'] + 50) / 100 * 0.25 +  # Normalize ROA
            full_df['operational_efficiency'] / 5 * 0.25 +
            full_df['productivity_index'] / 150 * 0.25
        )
        
        # Interaction: FDI × Government Policy
        full_df['fdi_x_policy'] = full_df['fdi_composite'] * full_df['policy_effectiveness_index']
        
        # Moderated effect on performance
        # Performance = β0 + β1*FDI + β2*Policy + β3*(FDI×Policy) + controls
        # The interaction term should show that policy moderates the FDI-performance relationship
        
        # Add some noise but maintain theoretical relationships
        moderation_effect = (
            full_df['fdi_composite'] * 0.4 +
            full_df['policy_effectiveness_index'] / 7 * 0.2 +
            full_df['fdi_x_policy'] * 0.3 +
            np.random.normal(0, 0.1, self.n_firms)
        )
        
        # Adjust performance based on moderation
        full_df['performance_adjusted'] = full_df['performance_composite'] * (0.7 + moderation_effect * 0.3)
        
        # Three-way interaction: FDI × Policy × Firm Size
        size_dummy = full_df['firm_size'].map({'Micro': 0, 'Small': 0.33, 'Medium': 0.67, 'Large': 1}).astype(float)
        full_df['fdi_x_policy_x_size'] = full_df['fdi_x_policy'] * size_dummy
        
        return full_df
    
    def add_missing_values(self, df, missing_pct=0.02):
        """Add realistic missing values to simulate survey non-response"""
        
        # Columns that shouldn't have missing values
        no_missing = ['firm_id', 'survey_date', 'subsector', 'ownership_type', 
                     'firm_size', 'num_employees']
        
        # Add missing values to other columns
        for col in df.columns:
            if col not in no_missing and np.random.random() < 0.3:  # 30% chance of having missing values
                missing_mask = np.random.random(self.n_firms) < missing_pct
                df.loc[missing_mask, col] = np.nan
        
        return df
    
    def generate_full_dataset(self):
        """Generate complete dataset"""
        
        print("Generating firm control variables...")
        control_df = self.generate_control_variables()
        
        print("Generating FDI constructs...")
        fdi_df = self.generate_fdi_constructs(control_df)
        
        print("Generating firm performance metrics...")
        performance_df = self.generate_firm_performance(control_df, fdi_df)
        
        print("Generating firm resources...")
        resources_df = self.generate_firm_resources(control_df, fdi_df)
        
        print("Generating government policy perceptions...")
        policy_df = self.generate_government_policy(control_df, fdi_df, performance_df)
        
        # Combine all dataframes
        full_df = pd.concat([control_df, fdi_df, performance_df, resources_df, policy_df], axis=1)
        
        print("Adding interaction terms and moderation effects...")
        full_df = self.add_interactions_and_moderations(full_df)
        
        print("Adding realistic missing values...")
        full_df = self.add_missing_values(full_df)
        
        return full_df
    
    def generate_metadata(self, df):
        """Generate metadata and data dictionary"""
        
        metadata = {
            'dataset_name': 'Food Processing Firms FDI Study - Lagos, Nigeria',
            'n_observations': len(df),
            'n_variables': len(df.columns),
            'collection_period': '2024-01-01 to 2024-10-01',
            'missing_data': df.isnull().sum().to_dict(),
            'variable_types': df.dtypes.astype(str).to_dict(),
            'summary_stats': {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            metadata['summary_stats'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'missing': int(df[col].isnull().sum())
            }
        
        return metadata


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("FOOD PROCESSING FIRMS FDI DATASET GENERATOR")
    print("Lagos, Nigeria - Government Policy Moderation Study")
    print("=" * 70)
    print()
    
    # Initialize generator with 500 firms
    generator = FDIDatasetGenerator(n_firms=500)
    
    # Generate complete dataset
    df = generator.generate_full_dataset()
    
    print(f"\n✓ Generated dataset with {len(df)} firms and {len(df.columns)} variables")
    
    # Save to CSV
    csv_filename = 'lagos_food_processing_fdi_dataset.csv'
    df.to_csv(csv_filename, index=False)
    print(f"✓ Saved to {csv_filename}")
    
    # Save to Excel with multiple sheets
    excel_filename = 'lagos_food_processing_fdi_dataset.xlsx'
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Main Data', index=False)
        
        # Add summary statistics sheet
        summary_stats = df.describe()
        summary_stats.to_excel(writer, sheet_name='Summary Statistics')
        
        # Add correlation matrix sheet
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        correlation_matrix.to_excel(writer, sheet_name='Correlations')
    
    print(f"✓ Saved to {excel_filename}")
    
    # Generate metadata
    metadata = generator.generate_metadata(df)
    
    # Save metadata to JSON
    import json
    with open('dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print("✓ Saved metadata to dataset_metadata.json")
    
    print("\n" + "=" * 70)
    print("DATASET GENERATION COMPLETE")
    print("=" * 70)
    
    # Display sample of the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    
    print("\nDataset Shape:", df.shape)
    print("\nColumn Categories:")
    print("- Control Variables:", [col for col in control_df.columns if col in df.columns])
    print("- FDI Constructs:", [col for col in fdi_df.columns if col in df.columns])
    print("- Performance Metrics:", [col for col in performance_df.columns if col in df.columns])
    print("- Firm Resources:", [col for col in resources_df.columns if col in df.columns])
    print("- Government Policy:", [col for col in policy_df.columns if col in df.columns])
    
    return df


if __name__ == "__main__":
    # Generate the dataset
    control_df = pd.DataFrame()  # Define these for the print statements
    fdi_df = pd.DataFrame()
    performance_df = pd.DataFrame()
    resources_df = pd.DataFrame()
    policy_df = pd.DataFrame()
    
    dataset = main()