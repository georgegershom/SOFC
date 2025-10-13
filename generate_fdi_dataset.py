#!/usr/bin/env python3
"""
Dataset Generator for FDI Study: Food Processing Firms in Lagos, Nigeria
Topic: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy

This script generates realistic fabricated data for research purposes.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FDIDatasetGenerator:
    def __init__(self, n_firms=300):
        self.n_firms = n_firms
        self.food_subsectors = [
            'Grain Processing', 'Dairy Products', 'Meat Processing', 
            'Beverage Manufacturing', 'Snack Foods', 'Bakery Products',
            'Canned Foods', 'Spice Processing', 'Oil & Fats', 'Confectionery'
        ]
        self.ownership_types = ['Domestic', 'Foreign', 'Joint Venture']
        
    def generate_firm_ids(self):
        """Generate unique firm identifiers"""
        return [f"FPF{str(i).zfill(4)}" for i in range(1, self.n_firms + 1)]
    
    def generate_fdi_constructs(self):
        """Generate FDI construct variables (Likert 1-5)"""
        # Based on Zahra & George (2002) absorptive capacity dimensions
        data = {}
        
        # Knowledge Absorption (4 items)
        data['knowledge_acquisition'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.05, 0.15, 0.35, 0.35, 0.10])
        data['knowledge_assimilation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.08, 0.17, 0.30, 0.35, 0.10])
        data['knowledge_transformation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.10, 0.20, 0.35, 0.25, 0.10])
        data['knowledge_exploitation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.07, 0.18, 0.30, 0.35, 0.10])
        
        # Task Performance (Koopmans et al., 2013) - 5 items
        data['task_proficiency'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.05, 0.15, 0.25, 0.40, 0.15])
        data['task_adaptability'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.08, 0.17, 0.30, 0.30, 0.15])
        data['task_proactivity'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        data['contextual_performance'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.06, 0.14, 0.30, 0.35, 0.15])
        data['counterproductive_behavior'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.40, 0.30, 0.20, 0.08, 0.02])  # Reverse scored
        
        # Innovation (OECD Oslo Manual) - 4 items
        data['product_innovation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.15, 0.25, 0.30, 0.20, 0.10])
        data['process_innovation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.12, 0.23, 0.35, 0.20, 0.10])
        data['organizational_innovation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.18, 0.27, 0.30, 0.15, 0.10])
        data['marketing_innovation'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.20, 0.30, 0.25, 0.15, 0.10])
        
        return data
    
    def generate_firm_performance(self):
        """Generate firm performance metrics"""
        data = {}
        
        # ROI (Return on Investment) - percentage
        data['roi_percent'] = np.random.normal(12.5, 8.2, self.n_firms)
        data['roi_percent'] = np.clip(data['roi_percent'], -15, 45)  # Realistic range
        
        # ROA (Return on Assets) - percentage
        data['roa_percent'] = np.random.normal(8.3, 5.1, self.n_firms)
        data['roa_percent'] = np.clip(data['roa_percent'], -10, 25)
        
        # Export Intensity (% of revenue from exports)
        export_prob = np.random.random(self.n_firms)
        data['export_intensity_percent'] = np.where(
            export_prob < 0.4, 0,  # 40% don't export
            np.random.exponential(15, self.n_firms)
        )
        data['export_intensity_percent'] = np.clip(data['export_intensity_percent'], 0, 80)
        
        # Market Share (% of local market in subsector)
        data['market_share_percent'] = np.random.exponential(5, self.n_firms)
        data['market_share_percent'] = np.clip(data['market_share_percent'], 0.1, 35)
        
        # Operational Efficiency (1-5 scale)
        data['operational_efficiency'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.08, 0.17, 0.35, 0.30, 0.10])
        
        return data
    
    def generate_firm_resources(self):
        """Generate firm resource variables"""
        data = {}
        
        # Human Capital
        data['skilled_labor_ratio'] = np.random.beta(2, 5, self.n_firms) * 100  # % of skilled workers
        data['training_investment'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.12, 0.18, 0.30, 0.25, 0.15])
        data['management_quality'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.08, 0.17, 0.35, 0.30, 0.10])
        
        # Technological Resources
        data['iot_adoption'] = np.random.choice([0, 1], self.n_firms, p=[0.65, 0.35])  # Binary
        data['rd_spend_percent'] = np.random.exponential(2.5, self.n_firms)  # % of revenue
        data['rd_spend_percent'] = np.clip(data['rd_spend_percent'], 0, 15)
        data['technology_sophistication'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.15, 0.25, 0.30, 0.20, 0.10])
        data['digital_infrastructure'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.10, 0.20, 0.35, 0.25, 0.10])
        
        # Financial Resources
        data['liquidity_ratio'] = np.random.gamma(2, 0.8, self.n_firms)  # Current ratio
        data['liquidity_ratio'] = np.clip(data['liquidity_ratio'], 0.5, 5.0)
        data['debt_equity_ratio'] = np.random.gamma(1.5, 0.6, self.n_firms)
        data['debt_equity_ratio'] = np.clip(data['debt_equity_ratio'], 0.1, 3.0)
        data['financial_flexibility'] = np.random.choice([1,2,3,4,5], self.n_firms, p=[0.12, 0.18, 0.30, 0.25, 0.15])
        
        return data
    
    def generate_government_policy_perception(self):
        """Generate government policy perception variables (Likert 1-7)"""
        data = {}
        
        # Tax Incentives
        data['tax_incentive_effectiveness'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms, 
                                                              p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03])
        data['tax_incentive_accessibility'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                              p=[0.18, 0.22, 0.25, 0.18, 0.10, 0.05, 0.02])
        
        # Regulatory Stability
        data['regulatory_predictability'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                            p=[0.20, 0.25, 0.20, 0.15, 0.10, 0.07, 0.03])
        data['policy_consistency'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                     p=[0.22, 0.23, 0.20, 0.15, 0.10, 0.07, 0.03])
        data['bureaucratic_efficiency'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                          p=[0.25, 0.25, 0.20, 0.15, 0.08, 0.05, 0.02])
        
        # Infrastructure Support
        data['transport_infrastructure'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                           p=[0.12, 0.18, 0.25, 0.20, 0.15, 0.07, 0.03])
        data['power_supply_reliability'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                           p=[0.30, 0.25, 0.20, 0.12, 0.08, 0.03, 0.02])
        data['telecommunications'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                     p=[0.08, 0.12, 0.20, 0.25, 0.20, 0.10, 0.05])
        
        # Corruption Experience
        data['corruption_frequency'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                       p=[0.05, 0.08, 0.12, 0.20, 0.25, 0.20, 0.10])  # Higher = more corruption
        data['corruption_impact'] = np.random.choice([1,2,3,4,5,6,7], self.n_firms,
                                                    p=[0.08, 0.10, 0.15, 0.22, 0.20, 0.15, 0.10])
        
        # Policy Effectiveness Index (composite)
        policy_components = np.array([
            data['tax_incentive_effectiveness'],
            data['regulatory_predictability'],
            data['transport_infrastructure'],
            8 - data['corruption_frequency']  # Reverse corruption for positive index
        ])
        data['policy_effectiveness_index'] = np.mean(policy_components, axis=0)
        
        return data
    
    def generate_control_variables(self):
        """Generate control variables"""
        data = {}
        
        # Firm Size
        data['employees'] = np.random.lognormal(4.5, 1.2, self.n_firms).astype(int)
        data['employees'] = np.clip(data['employees'], 10, 2000)
        
        # Assets (in millions of Naira)
        data['total_assets_million_naira'] = np.random.lognormal(8.5, 1.5, self.n_firms)
        data['total_assets_million_naira'] = np.clip(data['total_assets_million_naira'], 50, 50000)
        
        # Firm Age (years since establishment)
        current_year = 2024
        establishment_years = np.random.choice(range(1985, 2020), self.n_firms, 
                                             p=self._age_distribution())
        data['firm_age_years'] = current_year - establishment_years
        
        # Export Intensity Category
        data['export_category'] = np.where(
            data.get('export_intensity_percent', np.zeros(self.n_firms)) == 0, 'Non-exporter',
            np.where(data.get('export_intensity_percent', np.zeros(self.n_firms)) < 25, 'Low exporter',
                    np.where(data.get('export_intensity_percent', np.zeros(self.n_firms)) < 50, 'Medium exporter',
                            'High exporter'))
        )
        
        # Subsector
        data['subsector'] = np.random.choice(self.food_subsectors, self.n_firms)
        
        # Ownership Type
        data['ownership_type'] = np.random.choice(self.ownership_types, self.n_firms, 
                                                p=[0.60, 0.25, 0.15])  # Mostly domestic
        
        # FDI Status (derived from ownership)
        data['has_fdi'] = np.where(data['ownership_type'] == 'Domestic', 0, 1)
        
        # Location within Lagos
        lagos_areas = ['Victoria Island', 'Ikeja', 'Apapa', 'Ikorodu', 'Agege', 'Alimosho', 'Mushin']
        data['location_lagos'] = np.random.choice(lagos_areas, self.n_firms)
        
        return data
    
    def _age_distribution(self):
        """Generate realistic age distribution for firms"""
        # More firms established in recent decades
        years = list(range(1985, 2020))
        weights = []
        for year in years:
            if year < 1995:
                weights.append(0.5)
            elif year < 2005:
                weights.append(1.0)
            elif year < 2015:
                weights.append(1.5)
            else:
                weights.append(2.0)
        
        # Normalize weights
        total_weight = sum(weights)
        return [w/total_weight for w in weights]
    
    def add_correlations(self, df):
        """Add realistic correlations between variables"""
        # FDI firms tend to perform better
        fdi_mask = df['has_fdi'] == 1
        
        # Boost performance for FDI firms
        df.loc[fdi_mask, 'roi_percent'] += np.random.normal(3, 2, sum(fdi_mask))
        df.loc[fdi_mask, 'roa_percent'] += np.random.normal(2, 1.5, sum(fdi_mask))
        df.loc[fdi_mask, 'operational_efficiency'] = np.minimum(5, 
            df.loc[fdi_mask, 'operational_efficiency'] + np.random.choice([0, 1], sum(fdi_mask), p=[0.6, 0.4]))
        
        # Larger firms tend to have better resources
        large_firms = df['employees'] > df['employees'].quantile(0.75)
        df.loc[large_firms, 'rd_spend_percent'] += np.random.exponential(1, sum(large_firms))
        df.loc[large_firms, 'technology_sophistication'] = np.minimum(5,
            df.loc[large_firms, 'technology_sophistication'] + np.random.choice([0, 1], sum(large_firms), p=[0.5, 0.5]))
        
        # Older firms have more stable performance but less innovation
        old_firms = df['firm_age_years'] > df['firm_age_years'].quantile(0.75)
        df.loc[old_firms, 'product_innovation'] = np.maximum(1,
            df.loc[old_firms, 'product_innovation'] - np.random.choice([0, 1], sum(old_firms), p=[0.7, 0.3]))
        
        return df
    
    def generate_dataset(self):
        """Generate the complete dataset"""
        print("Generating FDI dataset for food processing firms in Lagos...")
        
        # Initialize dataset
        dataset = {}
        
        # Generate firm IDs
        dataset['firm_id'] = self.generate_firm_ids()
        
        # Generate all variable categories
        print("Generating FDI constructs...")
        fdi_data = self.generate_fdi_constructs()
        dataset.update(fdi_data)
        
        print("Generating firm performance data...")
        performance_data = self.generate_firm_performance()
        dataset.update(performance_data)
        
        print("Generating firm resources data...")
        resources_data = self.generate_firm_resources()
        dataset.update(resources_data)
        
        print("Generating government policy perception data...")
        policy_data = self.generate_government_policy_perception()
        dataset.update(policy_data)
        
        print("Generating control variables...")
        control_data = self.generate_control_variables()
        dataset.update(control_data)
        
        # Create DataFrame
        df = pd.DataFrame(dataset)
        
        # Add realistic correlations
        print("Adding realistic correlations...")
        df = self.add_correlations(df)
        
        # Round numeric columns appropriately
        df = self._round_columns(df)
        
        print(f"Dataset generated successfully with {len(df)} firms and {len(df.columns)} variables.")
        return df
    
    def _round_columns(self, df):
        """Round numeric columns to appropriate decimal places"""
        # Percentages to 1 decimal
        percent_cols = [col for col in df.columns if 'percent' in col or 'ratio' in col]
        for col in percent_cols:
            if col in df.columns:
                df[col] = df[col].round(1)
        
        # Likert scales to integers
        likert_cols = ['knowledge_acquisition', 'knowledge_assimilation', 'knowledge_transformation',
                      'knowledge_exploitation', 'task_proficiency', 'task_adaptability', 'task_proactivity',
                      'contextual_performance', 'counterproductive_behavior', 'product_innovation',
                      'process_innovation', 'organizational_innovation', 'marketing_innovation',
                      'operational_efficiency', 'training_investment', 'management_quality',
                      'technology_sophistication', 'digital_infrastructure', 'financial_flexibility',
                      'tax_incentive_effectiveness', 'tax_incentive_accessibility', 'regulatory_predictability',
                      'policy_consistency', 'bureaucratic_efficiency', 'transport_infrastructure',
                      'power_supply_reliability', 'telecommunications', 'corruption_frequency', 'corruption_impact']
        
        for col in likert_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Financial ratios to 2 decimals
        financial_cols = ['liquidity_ratio', 'debt_equity_ratio', 'policy_effectiveness_index']
        for col in financial_cols:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        # Assets to whole numbers
        if 'total_assets_million_naira' in df.columns:
            df['total_assets_million_naira'] = df['total_assets_million_naira'].round(0).astype(int)
        
        return df

def main():
    """Main function to generate and save the dataset"""
    # Generate dataset
    generator = FDIDatasetGenerator(n_firms=300)
    df = generator.generate_dataset()
    
    # Save to CSV
    output_file = 'lagos_food_processing_fdi_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Display basic statistics
    print(f"\nDataset Summary:")
    print(f"Number of firms: {len(df)}")
    print(f"Number of variables: {len(df.columns)}")
    print(f"\nOwnership distribution:")
    print(df['ownership_type'].value_counts())
    print(f"\nSubsector distribution:")
    print(df['subsector'].value_counts())
    
    # Display first few rows
    print(f"\nFirst 5 rows preview:")
    print(df.head())
    
    return df

if __name__ == "__main__":
    dataset = main()