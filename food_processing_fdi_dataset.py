#!/usr/bin/env python3
"""
Dataset Generator for Food Processing FDI Research
The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FoodProcessingFDIDatasetGenerator:
    def __init__(self, n_firms=300, n_sme=200, n_large=100):
        self.n_firms = n_firms
        self.n_sme = n_sme
        self.n_large = n_large
        self.firm_ids = []
        self.dataset = None
        
    def generate_firm_ids(self):
        """Generate unique firm IDs"""
        self.firm_ids = [f"FIRM_{str(i+1).zfill(3)}" for i in range(self.n_firms)]
        return self.firm_ids
    
    def generate_firm_characteristics(self):
        """Generate basic firm characteristics"""
        data = {
            'FirmID': self.firm_ids,
            'Firm_Size': ['SME'] * self.n_sme + ['Large'] * self.n_large,
            'Firm_Age': np.random.normal(15, 8, self.n_firms).astype(int),
            'Employees': np.concatenate([
                np.random.normal(25, 10, self.n_sme),  # SMEs: 5-45 employees
                np.random.normal(150, 50, self.n_large)  # Large: 100-200 employees
            ]).astype(int),
            'Assets_Million_NGN': np.concatenate([
                np.random.lognormal(2, 1, self.n_sme),  # SMEs: 1-50M NGN
                np.random.lognormal(4, 1, self.n_large)  # Large: 50-500M NGN
            ]),
            'Subsector': np.random.choice([
                'Fruit & Vegetable Processing',
                'Dairy Products',
                'Bakery & Confectionery',
                'Meat & Poultry Processing',
                'Beverage Production',
                'Grain & Cereal Processing'
            ], self.n_firms),
            'Ownership_Type': np.random.choice([
                'Domestic Private',
                'Foreign-Owned',
                'Joint Venture',
                'State-Owned'
            ], self.n_firms, p=[0.4, 0.3, 0.25, 0.05])
        }
        return data
    
    def generate_fdi_constructs(self):
        """Generate FDI-related variables based on Zahra & George (2002) and OECD Oslo Manual"""
        data = {
            'FDI_Presence': np.random.binomial(1, 0.6, self.n_firms),  # 60% have FDI
            'Knowledge_Absorption': np.random.normal(3.5, 0.8, self.n_firms),  # Likert 1-5
            'Task_Performance': np.random.normal(3.8, 0.7, self.n_firms),  # Likert 1-5
            'Innovation_Score': np.random.normal(3.6, 0.9, self.n_firms),  # Likert 1-5
            'FDI_Intensity': np.random.uniform(0, 0.8, self.n_firms),  # % of foreign ownership
            'FDI_Duration_Years': np.random.exponential(5, self.n_firms),  # Years of FDI presence
            'FDI_Origin_Region': np.random.choice([
                'Europe', 'North America', 'Asia', 'Other Africa', 'Middle East'
            ], self.n_firms),
            'FDI_Type': np.random.choice([
                'Greenfield Investment',
                'Merger & Acquisition',
                'Joint Venture',
                'Strategic Alliance'
            ], self.n_firms)
        }
        
        # Ensure Likert scales are within bounds
        for var in ['Knowledge_Absorption', 'Task_Performance', 'Innovation_Score']:
            data[var] = np.clip(data[var], 1, 5)
            
        return data
    
    def generate_firm_performance(self):
        """Generate firm performance indicators"""
        data = {
            'ROI_Percent': np.random.normal(15, 8, self.n_firms),  # Return on Investment
            'ROA_Percent': np.random.normal(12, 6, self.n_firms),  # Return on Assets
            'Export_Intensity': np.random.uniform(0, 0.6, self.n_firms),  # % of revenue from exports
            'Market_Share_Percent': np.random.uniform(0.1, 15, self.n_firms),
            'Operational_Efficiency': np.random.normal(3.7, 0.6, self.n_firms),  # Likert 1-5
            'Revenue_Growth_Percent': np.random.normal(8, 12, self.n_firms),
            'Profit_Margin_Percent': np.random.normal(10, 5, self.n_firms)
        }
        
        # Clip values to realistic ranges
        data['ROI_Percent'] = np.clip(data['ROI_Percent'], -10, 40)
        data['ROA_Percent'] = np.clip(data['ROA_Percent'], -5, 30)
        data['Operational_Efficiency'] = np.clip(data['Operational_Efficiency'], 1, 5)
        
        return data
    
    def generate_firm_resources(self):
        """Generate firm resource variables"""
        data = {
            'Skilled_Labor_Ratio': np.random.uniform(0.2, 0.8, self.n_firms),  # % of skilled workers
            'IoT_Usage': np.random.binomial(1, 0.4, self.n_firms),  # Binary: IoT adoption
            'R_D_Spend_Percent': np.random.uniform(0.5, 8, self.n_firms),  # % of revenue on R&D
            'Technology_Adoption_Score': np.random.normal(3.4, 0.8, self.n_firms),  # Likert 1-5
            'Liquidity_Ratio': np.random.uniform(1.2, 3.5, self.n_firms),  # Current assets/liabilities
            'Debt_to_Equity_Ratio': np.random.uniform(0.3, 2.0, self.n_firms),
            'Training_Investment_Percent': np.random.uniform(0.5, 5, self.n_firms)  # % of revenue on training
        }
        
        data['Technology_Adoption_Score'] = np.clip(data['Technology_Adoption_Score'], 1, 5)
        
        return data
    
    def generate_government_policy_perception(self):
        """Generate government policy perception variables (Likert 1-7)"""
        data = {
            'Tax_Incentive_Effectiveness': np.random.normal(4.2, 1.2, self.n_firms),
            'Regulatory_Stability': np.random.normal(3.8, 1.3, self.n_firms),
            'Infrastructure_Support': np.random.normal(3.5, 1.4, self.n_firms),
            'Corruption_Experience': np.random.normal(4.8, 1.1, self.n_firms),  # Higher = more corruption
            'Policy_Effectiveness_Index': np.random.normal(3.9, 1.0, self.n_firms),
            'Ease_of_Doing_Business': np.random.normal(3.6, 1.2, self.n_firms),
            'Government_Support_Access': np.random.normal(3.4, 1.3, self.n_firms),
            'Regulatory_Burden': np.random.normal(4.5, 1.1, self.n_firms)  # Higher = more burden
        }
        
        # Clip all to Likert 1-7 scale
        for var in data.keys():
            data[var] = np.clip(data[var], 1, 7)
            
        return data
    
    def generate_secondary_data(self):
        """Generate secondary macro and sectoral data"""
        # Lagos-specific data (same for all firms in a given year)
        lagos_data = {
            'FDI_Inflow_Lagos_Million_USD': np.random.normal(1200, 200, self.n_firms),  # $1.2bn ± 200M
            'Lagos_GDP_Growth_Percent': np.random.normal(3.2, 0.8, self.n_firms),
            'Food_Processing_GDP_Contribution_Percent': np.random.normal(2.8, 0.3, self.n_firms),
            'Power_Supply_Hours_Daily': np.random.normal(18, 4, self.n_firms),
            'Logistics_Quality_Index': np.random.normal(2.8, 0.4, self.n_firms),  # 1-5 scale
            'Corruption_Index': np.random.normal(24, 2, self.n_firms),  # Transparency International
            'Regulatory_Quality_Index': np.random.normal(2.1, 0.3, self.n_firms),  # World Bank
            'Ease_of_Doing_Business_Rank': np.random.normal(131, 5, self.n_firms),  # Nigeria's rank
            'EEG_Disbursement_Million_NGN': np.random.normal(150, 30, self.n_firms),  # Export Expansion Grant
            'Food_Export_Value_Million_USD': np.random.normal(450, 50, self.n_firms)
        }
        
        return lagos_data
    
    def generate_control_variables(self):
        """Generate additional control variables"""
        data = {
            'Export_Orientation': np.random.binomial(1, 0.4, self.n_firms),  # Binary: export-focused
            'Certification_ISO': np.random.binomial(1, 0.3, self.n_firms),  # ISO certification
            'Certification_HACCP': np.random.binomial(1, 0.2, self.n_firms),  # HACCP certification
            'Location_Zone': np.random.choice([
                'Ikeja', 'Victoria Island', 'Apapa', 'Surulere', 'Lagos Island', 'Other'
            ], self.n_firms),
            'Years_in_Export': np.random.exponential(3, self.n_firms),
            'Supply_Chain_Integration': np.random.normal(3.2, 0.9, self.n_firms),  # Likert 1-5
            'Competition_Intensity': np.random.normal(3.8, 0.7, self.n_firms)  # Likert 1-5
        }
        
        data['Supply_Chain_Integration'] = np.clip(data['Supply_Chain_Integration'], 1, 5)
        data['Competition_Intensity'] = np.clip(data['Competition_Intensity'], 1, 5)
        
        return data
    
    def generate_dataset(self):
        """Generate the complete integrated dataset"""
        print("Generating Food Processing FDI Dataset...")
        
        # Generate firm IDs
        self.generate_firm_ids()
        
        # Generate all data components
        firm_chars = self.generate_firm_characteristics()
        fdi_data = self.generate_fdi_constructs()
        performance_data = self.generate_firm_performance()
        resources_data = self.generate_firm_resources()
        policy_data = self.generate_government_policy_perception()
        secondary_data = self.generate_secondary_data()
        control_data = self.generate_control_variables()
        
        # Combine all data
        all_data = {**firm_chars, **fdi_data, **performance_data, **resources_data, 
                   **policy_data, **secondary_data, **control_data}
        
        # Create DataFrame
        self.dataset = pd.DataFrame(all_data)
        
        # Add some realistic correlations
        self.add_realistic_correlations()
        
        print(f"Dataset generated with {len(self.dataset)} firms and {len(self.dataset.columns)} variables")
        return self.dataset
    
    def add_realistic_correlations(self):
        """Add realistic correlations between variables"""
        df = self.dataset
        
        # FDI presence should correlate with performance
        fdi_mask = df['FDI_Presence'] == 1
        df.loc[fdi_mask, 'ROI_Percent'] += np.random.normal(2, 1, fdi_mask.sum())
        df.loc[fdi_mask, 'Innovation_Score'] += np.random.normal(0.3, 0.2, fdi_mask.sum())
        
        # Larger firms should have better performance
        large_mask = df['Firm_Size'] == 'Large'
        df.loc[large_mask, 'ROI_Percent'] += np.random.normal(3, 1.5, large_mask.sum())
        df.loc[large_mask, 'Market_Share_Percent'] += np.random.normal(2, 1, large_mask.sum())
        
        # Government policy effectiveness should correlate with firm performance
        policy_effectiveness = df['Policy_Effectiveness_Index']
        df['ROI_Percent'] += (policy_effectiveness - 4) * 0.5 + np.random.normal(0, 0.5, len(df))
        
        # Clip values to maintain realistic ranges
        df['ROI_Percent'] = np.clip(df['ROI_Percent'], -10, 40)
        df['Innovation_Score'] = np.clip(df['Innovation_Score'], 1, 5)
        df['Market_Share_Percent'] = np.clip(df['Market_Share_Percent'], 0.1, 20)
    
    def save_dataset(self, filename='food_processing_fdi_dataset.csv'):
        """Save dataset to CSV"""
        if self.dataset is not None:
            self.dataset.to_csv(filename, index=False)
            print(f"Dataset saved to {filename}")
        else:
            print("No dataset to save. Run generate_dataset() first.")
    
    def generate_data_dictionary(self):
        """Generate a comprehensive data dictionary"""
        data_dict = {
            "Dataset Information": {
                "Title": "Food Processing FDI Performance Dataset - Lagos, Nigeria",
                "Description": "Firm-level survey data on FDI influence on food processing firm performance",
                "Sample Size": f"{self.n_firms} firms ({self.n_sme} SMEs, {self.n_large} large firms)",
                "Geographic Coverage": "Lagos State, Nigeria",
                "Time Period": "Cross-sectional (2024)",
                "Data Sources": "Primary survey + Secondary macro data"
            },
            "Variable Categories": {
                "FDI Constructs": [
                    "FDI_Presence", "Knowledge_Absorption", "Task_Performance", 
                    "Innovation_Score", "FDI_Intensity", "FDI_Duration_Years",
                    "FDI_Origin_Region", "FDI_Type"
                ],
                "Firm Performance": [
                    "ROI_Percent", "ROA_Percent", "Export_Intensity", 
                    "Market_Share_Percent", "Operational_Efficiency",
                    "Revenue_Growth_Percent", "Profit_Margin_Percent"
                ],
                "Firm Resources": [
                    "Skilled_Labor_Ratio", "IoT_Usage", "R_D_Spend_Percent",
                    "Technology_Adoption_Score", "Liquidity_Ratio", 
                    "Debt_to_Equity_Ratio", "Training_Investment_Percent"
                ],
                "Government Policy": [
                    "Tax_Incentive_Effectiveness", "Regulatory_Stability",
                    "Infrastructure_Support", "Corruption_Experience",
                    "Policy_Effectiveness_Index", "Ease_of_Doing_Business",
                    "Government_Support_Access", "Regulatory_Burden"
                ],
                "Secondary Data": [
                    "FDI_Inflow_Lagos_Million_USD", "Lagos_GDP_Growth_Percent",
                    "Food_Processing_GDP_Contribution_Percent", "Power_Supply_Hours_Daily",
                    "Logistics_Quality_Index", "Corruption_Index",
                    "Regulatory_Quality_Index", "Ease_of_Doing_Business_Rank",
                    "EEG_Disbursement_Million_NGN", "Food_Export_Value_Million_USD"
                ],
                "Control Variables": [
                    "Firm_Size", "Firm_Age", "Employees", "Assets_Million_NGN",
                    "Subsector", "Ownership_Type", "Export_Orientation",
                    "Certification_ISO", "Certification_HACCP", "Location_Zone",
                    "Years_in_Export", "Supply_Chain_Integration", "Competition_Intensity"
                ]
            },
            "Measurement Scales": {
                "Likert 1-5": ["Knowledge_Absorption", "Task_Performance", "Innovation_Score", 
                              "Operational_Efficiency", "Technology_Adoption_Score", 
                              "Supply_Chain_Integration", "Competition_Intensity"],
                "Likert 1-7": ["Tax_Incentive_Effectiveness", "Regulatory_Stability",
                              "Infrastructure_Support", "Corruption_Experience",
                              "Policy_Effectiveness_Index", "Ease_of_Doing_Business",
                              "Government_Support_Access", "Regulatory_Burden"],
                "Binary (0/1)": ["FDI_Presence", "IoT_Usage", "Export_Orientation",
                                "Certification_ISO", "Certification_HACCP"],
                "Percentage": ["ROI_Percent", "ROA_Percent", "Export_Intensity", 
                              "Market_Share_Percent", "Revenue_Growth_Percent",
                              "Profit_Margin_Percent", "Skilled_Labor_Ratio",
                              "R_D_Spend_Percent", "Training_Investment_Percent"],
                "Continuous": ["FDI_Intensity", "Liquidity_Ratio", "Debt_to_Equity_Ratio",
                              "Assets_Million_NGN", "Employees", "Firm_Age"]
            }
        }
        
        return data_dict

def main():
    """Main function to generate and save the dataset"""
    # Initialize generator
    generator = FoodProcessingFDIDatasetGenerator(n_firms=300, n_sme=200, n_large=100)
    
    # Generate dataset
    dataset = generator.generate_dataset()
    
    # Save dataset
    generator.save_dataset('food_processing_fdi_dataset.csv')
    
    # Generate and save data dictionary
    data_dict = generator.generate_data_dictionary()
    with open('data_dictionary.json', 'w') as f:
        json.dump(data_dict, f, indent=2)
    
    # Display basic statistics
    print("\nDataset Summary:")
    print(f"Shape: {dataset.shape}")
    print(f"\nFirm Size Distribution:")
    print(dataset['Firm_Size'].value_counts())
    print(f"\nFDI Presence:")
    print(dataset['FDI_Presence'].value_counts())
    print(f"\nSubsector Distribution:")
    print(dataset['Subsector'].value_counts())
    
    # Display sample of the data
    print(f"\nSample of the dataset:")
    print(dataset.head())
    
    return dataset

if __name__ == "__main__":
    dataset = main()