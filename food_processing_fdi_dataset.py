#!/usr/bin/env python3
"""
Dataset Generator for Food Processing FDI Study
The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: 
The Moderating Role of the Nigerian Government Policy

This script generates a comprehensive dataset with firm-level survey data including:
- FDI Constructs (Knowledge absorption, task performance, innovation, firm resources)
- Firm Performance (ROI, ROA, export intensity, market share, operational efficiency)
- Firm Resources (Human capital, technological, financial)
- Government Policy Perception (Tax incentives, regulatory stability, infrastructure support)
- Control Variables (Firm size, age, subsector, ownership type)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FoodProcessingFDIDatasetGenerator:
    def __init__(self, n_firms=500):
        self.n_firms = n_firms
        self.firm_ids = [f"FP_{i+1:04d}" for i in range(n_firms)]
        
        # Lagos food processing subsectors
        self.subsectors = [
            "Beverage Manufacturing",
            "Dairy Processing", 
            "Grain Milling",
            "Meat Processing",
            "Oil & Fat Processing",
            "Snack Food Manufacturing",
            "Sugar Refining",
            "Vegetable Processing",
            "Bakery Products",
            "Confectionery"
        ]
        
        # Ownership types
        self.ownership_types = [
            "Domestic Private",
            "Foreign Direct Investment",
            "Joint Venture",
            "State-Owned",
            "Multinational Subsidiary"
        ]
        
        # Lagos areas for location
        self.lagos_areas = [
            "Ikeja", "Victoria Island", "Lekki", "Surulere", "Yaba",
            "Mushin", "Oshodi", "Apapa", "Lagos Island", "Alaba",
            "Ikorodu", "Badagry", "Epe", "Ajeromi-Ifelodun", "Amuwo-Odofin"
        ]

    def generate_fdi_constructs(self):
        """Generate FDI-related constructs using Likert scales"""
        data = {}
        
        # Knowledge Absorption (adapted from Zahra & George, 2002)
        data['knowledge_absorption_1'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['knowledge_absorption_2'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['knowledge_absorption_3'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['knowledge_absorption_4'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        
        # Task Performance (adapted from Koopmans et al., 2013)
        data['task_performance_1'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['task_performance_2'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['task_performance_3'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        
        # Innovation (OECD Oslo Manual adapted)
        data['innovation_1'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        data['innovation_2'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        data['innovation_3'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        data['innovation_4'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.10, 0.20, 0.30, 0.25, 0.15])
        
        # Firm Resources (Likert scale)
        data['firm_resources_1'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['firm_resources_2'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        data['firm_resources_3'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.25, 0.35, 0.20])
        
        return data

    def generate_firm_performance(self):
        """Generate firm performance metrics"""
        data = {}
        
        # ROI (Return on Investment) - percentage
        data['roi'] = np.random.normal(12.5, 8.0, self.n_firms)
        data['roi'] = np.clip(data['roi'], -5, 35)  # Realistic range
        
        # ROA (Return on Assets) - percentage
        data['roa'] = np.random.normal(8.0, 5.5, self.n_firms)
        data['roa'] = np.clip(data['roa'], -2, 25)  # Realistic range
        
        # Export Intensity - percentage of revenue from exports
        data['export_intensity'] = np.random.beta(2, 5, self.n_firms) * 100
        data['export_intensity'] = np.clip(data['export_intensity'], 0, 80)
        
        # Market Share - percentage
        data['market_share'] = np.random.beta(1.5, 8, self.n_firms) * 100
        data['market_share'] = np.clip(data['market_share'], 0.1, 25)
        
        # Operational Efficiency (Likert scale 1-5)
        data['operational_efficiency'] = np.random.choice([1, 2, 3, 4, 5], self.n_firms, p=[0.05, 0.15, 0.30, 0.35, 0.15])
        
        return data

    def generate_firm_resources(self):
        """Generate firm resource variables"""
        data = {}
        
        # Human Capital - Skilled labor ratio (percentage)
        data['skilled_labor_ratio'] = np.random.beta(3, 2, self.n_firms) * 100
        data['skilled_labor_ratio'] = np.clip(data['skilled_labor_ratio'], 10, 90)
        
        # IoT Use - Binary (yes/no)
        data['iot_use'] = np.random.choice([0, 1], self.n_firms, p=[0.4, 0.6])
        
        # R&D Spend - percentage of revenue
        data['rd_spend_ratio'] = np.random.beta(2, 8, self.n_firms) * 10
        data['rd_spend_ratio'] = np.clip(data['rd_spend_ratio'], 0, 8)
        
        # Liquidity Ratio - current assets/current liabilities
        data['liquidity_ratio'] = np.random.normal(1.8, 0.6, self.n_firms)
        data['liquidity_ratio'] = np.clip(data['liquidity_ratio'], 0.5, 4.0)
        
        return data

    def generate_government_policy_perception(self):
        """Generate government policy perception variables (Likert 1-7)"""
        data = {}
        
        # Tax Incentives
        data['tax_incentives_1'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])
        data['tax_incentives_2'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])
        data['tax_incentives_3'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05])
        
        # Regulatory Stability
        data['regulatory_stability_1'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03])
        data['regulatory_stability_2'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03])
        data['regulatory_stability_3'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.15, 0.20, 0.25, 0.20, 0.10, 0.07, 0.03])
        
        # Infrastructure Support
        data['infrastructure_support_1'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.04, 0.01])
        data['infrastructure_support_2'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.04, 0.01])
        data['infrastructure_support_3'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.20, 0.25, 0.25, 0.15, 0.10, 0.04, 0.01])
        
        # Corruption Experience (reverse coded - higher = less corruption)
        data['corruption_experience'] = np.random.choice([1, 2, 3, 4, 5, 6, 7], self.n_firms, p=[0.25, 0.30, 0.25, 0.15, 0.04, 0.01, 0.00])
        
        # Policy Effectiveness Index (composite score)
        policy_vars = ['tax_incentives_1', 'tax_incentives_2', 'tax_incentives_3',
                      'regulatory_stability_1', 'regulatory_stability_2', 'regulatory_stability_3',
                      'infrastructure_support_1', 'infrastructure_support_2', 'infrastructure_support_3']
        
        return data

    def generate_control_variables(self):
        """Generate control variables"""
        data = {}
        
        # Firm Size - Number of employees
        data['employees'] = np.random.lognormal(4.5, 1.2, self.n_firms)
        data['employees'] = np.clip(data['employees'], 5, 2000).astype(int)
        
        # Firm Assets - in millions NGN
        data['total_assets'] = np.random.lognormal(8.0, 1.5, self.n_firms)
        data['total_assets'] = np.clip(data['total_assets'], 10, 5000)
        
        # Firm Age - years since establishment
        data['firm_age'] = np.random.choice(range(1, 51), self.n_firms, p=[0.05] + [0.95/49] * 49)
        
        # Subsector
        data['subsector'] = np.random.choice(self.subsectors, self.n_firms)
        
        # Ownership Type
        data['ownership_type'] = np.random.choice(self.ownership_types, self.n_firms, p=[0.35, 0.25, 0.20, 0.10, 0.10])
        
        # Location in Lagos
        data['location'] = np.random.choice(self.lagos_areas, self.n_firms)
        
        # Export Status (Binary)
        data['export_status'] = np.random.choice([0, 1], self.n_firms, p=[0.3, 0.7])
        
        return data

    def generate_dataset(self):
        """Generate the complete dataset"""
        print(f"Generating dataset for {self.n_firms} food processing firms in Lagos...")
        
        # Generate all variable categories
        fdi_data = self.generate_fdi_constructs()
        performance_data = self.generate_firm_performance()
        resources_data = self.generate_firm_resources()
        policy_data = self.generate_government_policy_perception()
        control_data = self.generate_control_variables()
        
        # Combine all data
        all_data = {
            'firm_id': self.firm_ids,
            **fdi_data,
            **performance_data,
            **resources_data,
            **policy_data,
            **control_data
        }
        
        # Create DataFrame
        df = pd.DataFrame(all_data)
        
        # Calculate composite scores
        df['knowledge_absorption_score'] = df[['knowledge_absorption_1', 'knowledge_absorption_2', 
                                             'knowledge_absorption_3', 'knowledge_absorption_4']].mean(axis=1)
        
        df['task_performance_score'] = df[['task_performance_1', 'task_performance_2', 
                                         'task_performance_3']].mean(axis=1)
        
        df['innovation_score'] = df[['innovation_1', 'innovation_2', 'innovation_3', 'innovation_4']].mean(axis=1)
        
        df['firm_resources_score'] = df[['firm_resources_1', 'firm_resources_2', 'firm_resources_3']].mean(axis=1)
        
        df['tax_incentives_score'] = df[['tax_incentives_1', 'tax_incentives_2', 'tax_incentives_3']].mean(axis=1)
        
        df['regulatory_stability_score'] = df[['regulatory_stability_1', 'regulatory_stability_2', 
                                             'regulatory_stability_3']].mean(axis=1)
        
        df['infrastructure_support_score'] = df[['infrastructure_support_1', 'infrastructure_support_2', 
                                               'infrastructure_support_3']].mean(axis=1)
        
        # Policy Effectiveness Index
        policy_vars = ['tax_incentives_score', 'regulatory_stability_score', 'infrastructure_support_score']
        df['policy_effectiveness_index'] = df[policy_vars].mean(axis=1)
        
        # Add some realistic correlations
        self.add_realistic_correlations(df)
        
        return df

    def add_realistic_correlations(self, df):
        """Add realistic correlations between variables"""
        # FDI firms tend to have higher performance
        fdi_mask = df['ownership_type'].isin(['Foreign Direct Investment', 'Multinational Subsidiary'])
        df.loc[fdi_mask, 'roi'] *= 1.2
        df.loc[fdi_mask, 'roa'] *= 1.15
        df.loc[fdi_mask, 'export_intensity'] *= 1.3
        
        # Larger firms tend to have better resources
        large_firm_mask = df['employees'] > df['employees'].quantile(0.7)
        df.loc[large_firm_mask, 'skilled_labor_ratio'] *= 1.1
        df.loc[large_firm_mask, 'rd_spend_ratio'] *= 1.2
        
        # Older firms tend to have more stable performance
        old_firm_mask = df['firm_age'] > 20
        df.loc[old_firm_mask, 'roi'] *= 1.05
        df.loc[old_firm_mask, 'roa'] *= 1.05

    def create_data_dictionary(self):
        """Create a comprehensive data dictionary"""
        data_dict = {
            "dataset_info": {
                "title": "Food Processing FDI Dataset - Lagos, Nigeria",
                "description": "Firm-level survey data on Foreign Direct Investment influence on food processing firms",
                "sample_size": self.n_firms,
                "location": "Lagos, Nigeria",
                "generated_date": datetime.now().strftime("%Y-%m-%d"),
                "variables_count": 45
            },
            "variable_categories": {
                "fdi_constructs": {
                    "description": "Foreign Direct Investment related constructs using Likert scales",
                    "variables": {
                        "knowledge_absorption_1-4": "Firm's ability to absorb and utilize foreign knowledge (1-5 scale)",
                        "task_performance_1-3": "Performance in core business tasks (1-5 scale)",
                        "innovation_1-4": "Innovation capabilities and activities (1-5 scale)",
                        "firm_resources_1-3": "Available firm resources (1-5 scale)"
                    }
                },
                "firm_performance": {
                    "description": "Quantitative measures of firm performance",
                    "variables": {
                        "roi": "Return on Investment (percentage)",
                        "roa": "Return on Assets (percentage)",
                        "export_intensity": "Percentage of revenue from exports",
                        "market_share": "Market share percentage",
                        "operational_efficiency": "Operational efficiency rating (1-5 scale)"
                    }
                },
                "firm_resources": {
                    "description": "Human, technological, and financial resources",
                    "variables": {
                        "skilled_labor_ratio": "Percentage of skilled workers",
                        "iot_use": "Use of Internet of Things technology (0/1)",
                        "rd_spend_ratio": "R&D spending as percentage of revenue",
                        "liquidity_ratio": "Current assets to current liabilities ratio"
                    }
                },
                "government_policy_perception": {
                    "description": "Firm perceptions of government policies (1-7 scale)",
                    "variables": {
                        "tax_incentives_1-3": "Perception of tax incentive effectiveness",
                        "regulatory_stability_1-3": "Perception of regulatory stability",
                        "infrastructure_support_1-3": "Perception of infrastructure support",
                        "corruption_experience": "Experience with corruption (reverse coded)",
                        "policy_effectiveness_index": "Composite policy effectiveness score"
                    }
                },
                "control_variables": {
                    "description": "Firm characteristics and control variables",
                    "variables": {
                        "firm_id": "Unique firm identifier",
                        "employees": "Number of employees",
                        "total_assets": "Total assets in millions NGN",
                        "firm_age": "Years since establishment",
                        "subsector": "Food processing subsector",
                        "ownership_type": "Type of firm ownership",
                        "location": "Location within Lagos",
                        "export_status": "Export activity status (0/1)"
                    }
                }
            },
            "composite_scores": {
                "knowledge_absorption_score": "Average of knowledge absorption items",
                "task_performance_score": "Average of task performance items",
                "innovation_score": "Average of innovation items",
                "firm_resources_score": "Average of firm resources items",
                "tax_incentives_score": "Average of tax incentives items",
                "regulatory_stability_score": "Average of regulatory stability items",
                "infrastructure_support_score": "Average of infrastructure support items"
            }
        }
        return data_dict

    def export_dataset(self, df, data_dict):
        """Export dataset in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs("food_processing_fdi_dataset", exist_ok=True)
        
        # Export CSV
        csv_path = f"food_processing_fdi_dataset/food_processing_fdi_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Dataset exported to: {csv_path}")
        
        # Export Excel with multiple sheets
        excel_path = f"food_processing_fdi_dataset/food_processing_fdi_data_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Main_Dataset', index=False)
            
            # Create summary statistics sheet
            summary_stats = df.describe()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Create correlation matrix sheet
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            corr_matrix.to_excel(writer, sheet_name='Correlation_Matrix')
        
        print(f"Excel file exported to: {excel_path}")
        
        # Export JSON
        json_path = f"food_processing_fdi_dataset/food_processing_fdi_data_{timestamp}.json"
        df.to_json(json_path, orient='records', indent=2)
        print(f"JSON file exported to: {json_path}")
        
        # Export data dictionary
        dict_path = f"food_processing_fdi_dataset/data_dictionary_{timestamp}.json"
        with open(dict_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
        print(f"Data dictionary exported to: {dict_path}")
        
        # Create README
        readme_path = "food_processing_fdi_dataset/README.md"
        readme_content = f"""# Food Processing FDI Dataset

## Dataset Information
- **Title**: The Influence of Foreign Direct Investment on the Performance of Food Processing Firms in Lagos, Nigeria: The Moderating Role of the Nigerian Government Policy
- **Sample Size**: {self.n_firms} firms
- **Location**: Lagos, Nigeria
- **Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Files Included
1. `food_processing_fdi_data_{timestamp}.csv` - Main dataset in CSV format
2. `food_processing_fdi_data_{timestamp}.xlsx` - Excel file with multiple sheets
3. `food_processing_fdi_data_{timestamp}.json` - JSON format
4. `data_dictionary_{timestamp}.json` - Comprehensive data dictionary

## Variable Categories
- **FDI Constructs**: Knowledge absorption, task performance, innovation, firm resources
- **Firm Performance**: ROI, ROA, export intensity, market share, operational efficiency
- **Firm Resources**: Human capital, technological resources, financial resources
- **Government Policy Perception**: Tax incentives, regulatory stability, infrastructure support
- **Control Variables**: Firm size, age, subsector, ownership type, location

## Usage Notes
- All Likert scale variables use appropriate ranges (1-5 or 1-7)
- Composite scores are calculated as means of related items
- Data includes realistic correlations between variables
- Suitable for regression analysis, structural equation modeling, and other statistical analyses

## Research Context
This dataset supports research on how Foreign Direct Investment influences the performance of food processing firms in Lagos, Nigeria, with particular attention to the moderating role of government policies.
"""
        
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"README file created: {readme_path}")
        
        return {
            'csv_path': csv_path,
            'excel_path': excel_path,
            'json_path': json_path,
            'dict_path': dict_path,
            'readme_path': readme_path
        }

def main():
    """Main function to generate and export the dataset"""
    print("🧾 Food Processing FDI Dataset Generator")
    print("=" * 50)
    
    # Generate dataset
    generator = FoodProcessingFDIDatasetGenerator(n_firms=500)
    df = generator.generate_dataset()
    data_dict = generator.create_data_dictionary()
    
    # Export dataset
    file_paths = generator.export_dataset(df, data_dict)
    
    # Display summary
    print("\n📊 Dataset Summary:")
    print(f"Total firms: {len(df)}")
    print(f"Total variables: {len(df.columns)}")
    print(f"FDI firms: {len(df[df['ownership_type'].isin(['Foreign Direct Investment', 'Multinational Subsidiary'])])}")
    print(f"Exporting firms: {df['export_status'].sum()}")
    print(f"Average ROI: {df['roi'].mean():.2f}%")
    print(f"Average ROA: {df['roa'].mean():.2f}%")
    
    print("\n✅ Dataset generation completed successfully!")
    print("All files have been saved to the 'food_processing_fdi_dataset' directory.")

if __name__ == "__main__":
    main()