#!/usr/bin/env python3
"""
Generate synthetic datasets for PhD thesis on Open Innovation in Tanzanian SMEs
This script creates realistic survey data and interview transcripts for analysis
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from faker import Faker

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()

# Configuration
N_RESPONDENTS = 313  # Sample size as mentioned in the thesis
SECTORS = ['Manufacturing', 'Retail/Wholesale', 'ICT/Business Services', 'Agriculture/Agribusiness']
LOCATIONS = ['Dar es Salaam', 'Arusha', 'Mwanza']
FIRM_SIZES = ['Small (5-49 employees)', 'Medium (50-99 employees)']

def generate_sme_demographics():
    """Generate demographic data for SMEs"""
    data = []
    
    for i in range(N_RESPONDENTS):
        # Weighted random selection to reflect realistic distribution
        sector = np.random.choice(SECTORS, p=[0.25, 0.35, 0.20, 0.20])
        location = np.random.choice(LOCATIONS, p=[0.50, 0.25, 0.25])
        firm_size = np.random.choice(FIRM_SIZES, p=[0.70, 0.30])
        
        # Correlated variables
        if sector == 'ICT/Business Services':
            firm_age = np.random.randint(1, 10)
            education_level = np.random.choice(['University', 'Diploma', 'Secondary'], p=[0.6, 0.3, 0.1])
            revenue = np.random.uniform(50, 500) * 1e6  # Higher revenue for ICT
        else:
            firm_age = np.random.randint(1, 25)
            education_level = np.random.choice(['University', 'Diploma', 'Secondary', 'Primary'], 
                                             p=[0.15, 0.35, 0.40, 0.10])
            revenue = np.random.uniform(10, 200) * 1e6
        
        owner_age = max(25, min(65, np.random.normal(40, 10)))
        employee_count = np.random.randint(5, 50) if 'Small' in firm_size else np.random.randint(50, 100)
        
        data.append({
            'respondent_id': f'SME_{i+1:04d}',
            'sector': sector,
            'location': location,
            'firm_size': firm_size,
            'firm_age_years': firm_age,
            'owner_age': int(owner_age),
            'owner_gender': np.random.choice(['Male', 'Female'], p=[0.65, 0.35]),
            'owner_education': education_level,
            'employee_count': employee_count,
            'annual_revenue_tzs': revenue,
            'export_activity': np.random.choice(['Yes', 'No'], p=[0.15, 0.85]),
            'formal_registration': np.random.choice(['Yes', 'No'], p=[0.85, 0.15])
        })
    
    return pd.DataFrame(data)

def generate_organizational_barriers(demographics_df):
    """Generate data on organizational barriers"""
    barriers_data = []
    
    for idx, row in demographics_df.iterrows():
        # Base scores influenced by demographics
        base_modifier = 0
        if row['owner_education'] in ['Secondary', 'Primary']:
            base_modifier += 0.3
        if row['firm_age_years'] > 10:
            base_modifier += 0.2
        if row['sector'] != 'ICT/Business Services':
            base_modifier += 0.2
        
        # Generate barrier scores (1-7 Likert scale)
        barriers = {
            'respondent_id': row['respondent_id'],
            
            # Structural barriers
            'barrier_rigid_hierarchy': min(7, max(1, np.random.normal(4.5 + base_modifier, 1.2))),
            'barrier_centralized_decision': min(7, max(1, np.random.normal(4.8 + base_modifier, 1.1))),
            'barrier_departmental_silos': min(7, max(1, np.random.normal(4.2 + base_modifier, 1.3))),
            
            # Cultural barriers  
            'barrier_risk_aversion': min(7, max(1, np.random.normal(5.2 + base_modifier, 1.0))),
            'barrier_nih_syndrome': min(7, max(1, np.random.normal(4.9 + base_modifier, 1.2))),
            'barrier_resistance_external': min(7, max(1, np.random.normal(4.6 + base_modifier, 1.1))),
            
            # Resource barriers
            'barrier_financial_constraints': min(7, max(1, np.random.normal(5.8 + base_modifier*0.5, 0.9))),
            'barrier_human_capital': min(7, max(1, np.random.normal(5.1 + base_modifier, 1.0))),
            'barrier_tech_infrastructure': min(7, max(1, np.random.normal(5.3 + base_modifier, 1.1))),
            
            # Cognitive barriers
            'barrier_limited_awareness': min(7, max(1, np.random.normal(4.7 + base_modifier, 1.2))),
            'barrier_knowledge_gaps': min(7, max(1, np.random.normal(4.9 + base_modifier, 1.1))),
            
            # Relational barriers
            'barrier_trust_deficit': min(7, max(1, np.random.normal(5.0 + base_modifier, 1.0))),
            'barrier_weak_networks': min(7, max(1, np.random.normal(4.8 + base_modifier, 1.2))),
            'barrier_poor_collaboration': min(7, max(1, np.random.normal(4.6 + base_modifier, 1.3)))
        }
        
        barriers_data.append(barriers)
    
    return pd.DataFrame(barriers_data)

def generate_digital_literacy(demographics_df):
    """Generate digital literacy scores"""
    digital_data = []
    
    for idx, row in demographics_df.iterrows():
        # Base digital literacy influenced by sector and education
        base_score = 3.0
        if row['sector'] == 'ICT/Business Services':
            base_score = 5.5
        elif row['sector'] == 'Manufacturing':
            base_score = 3.2
        
        if row['owner_education'] == 'University':
            base_score += 1.0
        elif row['owner_education'] == 'Primary':
            base_score -= 1.0
        
        if row['owner_age'] < 35:
            base_score += 0.5
        elif row['owner_age'] > 50:
            base_score -= 0.5
        
        # Generate digital literacy dimensions (1-7 scale)
        digital = {
            'respondent_id': row['respondent_id'],
            'dl_technical': min(7, max(1, np.random.normal(base_score, 0.8))),
            'dl_informational': min(7, max(1, np.random.normal(base_score + 0.3, 0.9))),
            'dl_communicative': min(7, max(1, np.random.normal(base_score + 0.1, 0.8))),
            'dl_strategic': min(7, max(1, np.random.normal(base_score - 0.2, 1.0))),
            
            # Digital tool usage
            'uses_computer': 1 if np.random.random() < (base_score/7) else 0,
            'uses_internet_daily': 1 if np.random.random() < (base_score/7 + 0.1) else 0,
            'uses_cloud_services': 1 if np.random.random() < (base_score/7 - 0.2) else 0,
            'uses_social_media_business': 1 if np.random.random() < (base_score/7 + 0.2) else 0,
            'uses_digital_payments': 1 if np.random.random() < 0.7 else 0,  # High adoption
            'uses_erp_software': 1 if np.random.random() < (base_score/7 - 0.3) else 0,
        }
        
        # Calculate composite digital literacy score
        digital['dl_composite'] = np.mean([
            digital['dl_technical'],
            digital['dl_informational'],
            digital['dl_communicative'],
            digital['dl_strategic']
        ])
        
        digital_data.append(digital)
    
    return pd.DataFrame(digital_data)

def generate_oi_adoption(demographics_df, barriers_df, digital_df):
    """Generate open innovation adoption outcomes"""
    oi_data = []
    
    # Merge dataframes for analysis
    full_df = demographics_df.merge(barriers_df, on='respondent_id')
    full_df = full_df.merge(digital_df, on='respondent_id')
    
    for idx, row in full_df.iterrows():
        # Calculate aggregate barrier score
        barrier_cols = [col for col in barriers_df.columns if 'barrier_' in col]
        avg_barrier = row[barrier_cols].mean()
        
        # Digital literacy composite
        dl_composite = row['dl_composite']
        
        # Base OI adoption influenced by barriers and digital literacy
        # Negative relationship with barriers
        base_oi = 7 - (avg_barrier * 0.6)
        
        # Moderation effect: digital literacy reduces barrier impact
        moderation_effect = dl_composite * 0.15 * (7 - avg_barrier)
        
        # Add some noise and sector effects
        if row['sector'] == 'ICT/Business Services':
            sector_bonus = 1.0
        elif row['sector'] == 'Manufacturing':
            sector_bonus = 0.3
        else:
            sector_bonus = 0
        
        oi_score = base_oi + moderation_effect + sector_bonus + np.random.normal(0, 0.5)
        oi_score = min(7, max(1, oi_score))
        
        # Generate specific OI practices (binary)
        oi_practices = {
            'respondent_id': row['respondent_id'],
            'oi_adoption_score': oi_score,
            
            # Breadth of OI
            'oi_customer_collab': 1 if np.random.random() < (oi_score/7) else 0,
            'oi_supplier_collab': 1 if np.random.random() < (oi_score/7 - 0.1) else 0,
            'oi_competitor_collab': 1 if np.random.random() < (oi_score/7 - 0.3) else 0,
            'oi_university_collab': 1 if np.random.random() < (oi_score/7 - 0.2) else 0,
            'oi_consultant_use': 1 if np.random.random() < (oi_score/7 - 0.15) else 0,
            
            # Depth of OI
            'oi_joint_rd': 1 if np.random.random() < (oi_score/7 - 0.25) else 0,
            'oi_licensing_in': 1 if np.random.random() < (oi_score/7 - 0.35) else 0,
            'oi_licensing_out': 1 if np.random.random() < (oi_score/7 - 0.4) else 0,
            'oi_crowdsourcing': 1 if np.random.random() < (oi_score/7 - 0.4) else 0,
            
            # OI outcomes
            'new_products_last_year': max(0, int(np.random.poisson(oi_score/2))),
            'process_improvements': max(0, int(np.random.poisson(oi_score/2.5))),
            'revenue_from_innovation_pct': min(50, max(0, np.random.normal(oi_score * 3, 5)))
        }
        
        # Calculate OI breadth and depth
        breadth_cols = ['oi_customer_collab', 'oi_supplier_collab', 'oi_competitor_collab', 
                       'oi_university_collab', 'oi_consultant_use']
        depth_cols = ['oi_joint_rd', 'oi_licensing_in', 'oi_licensing_out', 'oi_crowdsourcing']
        
        oi_practices['oi_breadth'] = sum(oi_practices[col] for col in breadth_cols)
        oi_practices['oi_depth'] = sum(oi_practices[col] for col in depth_cols)
        
        oi_data.append(oi_practices)
    
    return pd.DataFrame(oi_data)

def generate_qualitative_themes():
    """Generate qualitative interview themes and sample quotes"""
    themes = {
        'organizational_barriers': {
            'rigid_structures': [
                "All decisions must go through the owner. Even small collaborations need approval.",
                "We have a very hierarchical system. Junior staff cannot directly engage with external partners.",
                "The bureaucracy in our organization makes it difficult to respond quickly to collaboration opportunities."
            ],
            'cultural_resistance': [
                "We prefer to keep our innovations secret. Sharing with others might help competitors.",
                "There is a strong belief that good ideas must come from within the company.",
                "Previous bad experiences with partners have made us very cautious about collaboration."
            ],
            'resource_constraints': [
                "We simply don't have the funds to invest in collaborative projects.",
                "Our staff are already overworked. We cannot spare anyone for external projects.",
                "The cost of protecting our intellectual property is too high for us."
            ],
            'trust_issues': [
                "We have been cheated before by partners who stole our ideas.",
                "It's difficult to find trustworthy partners in our industry.",
                "Without proper legal protection, we cannot risk sharing our knowledge."
            ]
        },
        'digital_literacy_impact': {
            'enabling_collaboration': [
                "WhatsApp groups have made it easier to coordinate with suppliers and customers.",
                "Online platforms help us find partners we would never have met otherwise.",
                "Digital tools allow us to share documents and collaborate despite distance."
            ],
            'knowledge_access': [
                "YouTube tutorials have taught us new production techniques.",
                "We use Google to research market trends and competitor products.",
                "Online forums connect us with experts who can solve technical problems."
            ],
            'capability_gaps': [
                "Many of our staff struggle with basic computer tasks.",
                "We know digital tools exist but don't know how to use them effectively.",
                "The younger employees are good with technology but lack business experience."
            ]
        },
        'contextual_factors': {
            'infrastructure_challenges': [
                "Internet connectivity is unreliable, making online collaboration difficult.",
                "Power outages disrupt our digital operations regularly.",
                "The cost of data is too high for sustained online engagement."
            ],
            'policy_environment': [
                "Government regulations are unclear about digital business operations.",
                "There is no support for SMEs trying to innovate collaboratively.",
                "Tax policies do not recognize innovation expenses."
            ],
            'market_dynamics': [
                "Customers still prefer face-to-face interactions over digital channels.",
                "The local market is too small to justify innovation investments.",
                "Competition from imports makes it hard to recover innovation costs."
            ]
        }
    }
    
    return themes

def save_datasets():
    """Generate and save all datasets"""
    print("Generating SME demographics...")
    demographics_df = generate_sme_demographics()
    
    print("Generating organizational barriers data...")
    barriers_df = generate_organizational_barriers(demographics_df)
    
    print("Generating digital literacy data...")
    digital_df = generate_digital_literacy(demographics_df)
    
    print("Generating OI adoption data...")
    oi_df = generate_oi_adoption(demographics_df, barriers_df, digital_df)
    
    print("Generating qualitative themes...")
    themes = generate_qualitative_themes()
    
    # Merge all quantitative data
    full_dataset = demographics_df.merge(barriers_df, on='respondent_id')
    full_dataset = full_dataset.merge(digital_df, on='respondent_id')
    full_dataset = full_dataset.merge(oi_df, on='respondent_id')
    
    # Save datasets
    full_dataset.to_csv('thesis_survey_data.csv', index=False)
    demographics_df.to_csv('sme_demographics.csv', index=False)
    barriers_df.to_csv('organizational_barriers.csv', index=False)
    digital_df.to_csv('digital_literacy.csv', index=False)
    oi_df.to_csv('oi_adoption.csv', index=False)
    
    with open('qualitative_themes.json', 'w') as f:
        json.dump(themes, f, indent=2)
    
    print(f"\nDatasets generated successfully!")
    print(f"Total respondents: {len(full_dataset)}")
    print(f"Variables: {len(full_dataset.columns)}")
    print(f"\nFiles created:")
    print("- thesis_survey_data.csv (full dataset)")
    print("- sme_demographics.csv")
    print("- organizational_barriers.csv")
    print("- digital_literacy.csv")
    print("- oi_adoption.csv")
    print("- qualitative_themes.json")
    
    # Generate summary statistics
    print("\n=== Summary Statistics ===")
    print(f"\nSector Distribution:")
    print(demographics_df['sector'].value_counts())
    print(f"\nLocation Distribution:")
    print(demographics_df['location'].value_counts())
    print(f"\nMean Barrier Scores:")
    barrier_cols = [col for col in barriers_df.columns if 'barrier_' in col]
    print(barriers_df[barrier_cols].mean().sort_values(ascending=False))
    print(f"\nMean Digital Literacy: {digital_df['dl_composite'].mean():.2f}")
    print(f"Mean OI Adoption Score: {oi_df['oi_adoption_score'].mean():.2f}")
    
    return full_dataset

if __name__ == "__main__":
    dataset = save_datasets()