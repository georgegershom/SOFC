"""
PhD Research Dataset Generator: FDI Influence on Food Processing Firms
Jiangsu University - Lagos, Nigeria Study
Author: PhD Research Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class FDIDataGenerator:
    def __init__(self, n_sme=200, n_large=100):
        """
        Initialize data generator with stratified sample sizes
        """
        self.n_sme = n_sme
        self.n_large = n_large
        self.n_total = n_sme + n_large
        self.data = []
        
    def generate_firm_code(self, index, is_sme):
        """Generate unique firm identifier"""
        prefix = "SME" if is_sme else "LRG"
        return f"{prefix}{str(index).zfill(3)}"
    
    def generate_firm_characteristics(self, is_sme):
        """Generate basic firm characteristics"""
        # Years of operation - larger firms tend to be older
        if is_sme:
            years_operation = np.random.gamma(3, 2) + 1
        else:
            years_operation = np.random.gamma(5, 3) + 5
        years_operation = min(50, int(years_operation))
        
        # Number of employees
        if is_sme:
            employees_cat = np.random.choice([1, 2], p=[0.6, 0.4])  # 1-50 or 51-250
        else:
            employees_cat = np.random.choice([3, 4], p=[0.7, 0.3])  # 251-500 or 500+
        
        # Annual revenue (in million Naira)
        if is_sme:
            if employees_cat == 1:
                revenue_cat = np.random.choice([1, 2], p=[0.7, 0.3])  # <50M or 50M-500M
            else:
                revenue_cat = np.random.choice([2, 3], p=[0.6, 0.4])  # 50M-500M or 500M-5B
        else:
            revenue_cat = np.random.choice([3, 4], p=[0.4, 0.6])  # 500M-5B or >5B
        
        # Ownership type - larger firms more likely to have FDI
        if is_sme:
            ownership = np.random.choice([1, 2, 3], p=[0.7, 0.15, 0.15])  # Local, Foreign, JV
        else:
            ownership = np.random.choice([1, 2, 3], p=[0.4, 0.3, 0.3])
        
        return {
            'years_operation': years_operation,
            'employees_category': employees_cat,
            'revenue_category': revenue_cat,
            'ownership_type': ownership
        }
    
    def generate_fdi_engagement(self, ownership_type, firm_size):
        """Generate FDI engagement data"""
        # Probability of having FDI based on ownership and size
        if ownership_type == 1:  # Local
            has_fdi = np.random.choice([0, 1], p=[0.75, 0.25])
        elif ownership_type == 2:  # Foreign-owned
            has_fdi = 1
        else:  # Joint venture
            has_fdi = 1
        
        if has_fdi:
            # Type of FDI (can be multiple)
            fdi_equity = np.random.choice([0, 1], p=[0.4, 0.6])
            fdi_jv = np.random.choice([0, 1], p=[0.5, 0.5])
            fdi_tech = np.random.choice([0, 1], p=[0.3, 0.7])
            fdi_mgmt = np.random.choice([0, 1], p=[0.6, 0.4])
            
            # Ensure at least one type is selected
            if not any([fdi_equity, fdi_jv, fdi_tech, fdi_mgmt]):
                fdi_tech = 1
            
            # Years with FDI partnership
            years_fdi = np.random.gamma(2, 1.5)
            years_fdi = min(20, max(1, int(years_fdi)))
        else:
            fdi_equity = fdi_jv = fdi_tech = fdi_mgmt = 0
            years_fdi = 0
        
        return {
            'has_fdi': has_fdi,
            'fdi_equity': fdi_equity,
            'fdi_joint_venture': fdi_jv,
            'fdi_tech_transfer': fdi_tech,
            'fdi_mgmt_contract': fdi_mgmt,
            'years_fdi_partnership': years_fdi
        }
    
    def generate_knowledge_absorption(self, has_fdi, years_fdi):
        """Generate knowledge absorption scores (Zahra & George Scale)"""
        # Base scores influenced by FDI presence and duration
        if has_fdi:
            base_mean = 3.5 + (years_fdi * 0.05)
            base_std = 0.8
        else:
            base_mean = 2.5
            base_std = 1.0
        
        # Generate correlated scores for 4 items
        ka_scores = []
        for i in range(4):
            score = np.random.normal(base_mean, base_std)
            score = max(1, min(5, round(score)))  # Bound between 1-5
            ka_scores.append(score)
        
        return {
            'ka_technical_manuals': ka_scores[0],
            'ka_staff_training': ka_scores[1],
            'ka_tech_adaptation': ka_scores[2],
            'ka_knowledge_commercialization': ka_scores[3],
            'ka_average': np.mean(ka_scores)
        }
    
    def generate_task_performance(self, has_fdi, ka_average):
        """Generate task performance scores (Koopmans et al. Scale)"""
        # Performance influenced by FDI and knowledge absorption
        base_mean = 2.8 + (has_fdi * 0.5) + (ka_average * 0.2)
        base_std = 0.7
        
        tp_scores = []
        for i in range(4):
            score = np.random.normal(base_mean, base_std)
            score = max(1, min(5, round(score)))
            tp_scores.append(score)
        
        return {
            'tp_production_efficiency': tp_scores[0],
            'tp_quality_control': tp_scores[1],
            'tp_order_fulfillment': tp_scores[2],
            'tp_employee_productivity': tp_scores[3],
            'tp_average': np.mean(tp_scores)
        }
    
    def generate_innovation(self, has_fdi, firm_size, ka_average):
        """Generate innovation metrics (OECD Oslo Manual)"""
        # R&D spending as % of revenue
        if has_fdi and not firm_size:  # Large firms with FDI
            rd_spending = np.random.gamma(2, 1.5) + 1
        elif has_fdi:
            rd_spending = np.random.gamma(1.5, 1) + 0.5
        else:
            rd_spending = np.random.gamma(1, 0.5) + 0.1
        rd_spending = min(10, round(rd_spending, 1))
        
        # New products launched (influenced by R&D and KA)
        product_mean = 2 + (rd_spending * 0.5) + (ka_average * 0.3)
        new_products = max(0, int(np.random.poisson(product_mean)))
        
        # Process innovations
        innovation_prob = 0.3 + (has_fdi * 0.2) + (ka_average * 0.05)
        iot_systems = np.random.choice([0, 1], p=[1-innovation_prob, innovation_prob])
        automation = np.random.choice([0, 1], p=[1-innovation_prob*0.8, innovation_prob*0.8])
        quality_mgmt = np.random.choice([0, 1], p=[1-innovation_prob*1.2, min(1, innovation_prob*1.2)])
        other_innovation = np.random.choice([0, 1], p=[1-innovation_prob*0.6, innovation_prob*0.6])
        
        return {
            'rd_spending_percent': rd_spending,
            'new_products_3years': new_products,
            'process_inn_iot': iot_systems,
            'process_inn_automation': automation,
            'process_inn_quality': quality_mgmt,
            'process_inn_other': other_innovation,
            'innovation_score': (rd_spending/10 + new_products/20 + 
                               (iot_systems + automation + quality_mgmt + other_innovation)/4) / 3 * 5
        }
    
    def generate_firm_resources(self, firm_size, has_fdi):
        """Generate firm resources data"""
        # Human resources
        if firm_size:  # Large firm
            skilled_workforce = np.random.beta(7, 3) * 100
            training_hours = np.random.gamma(3, 10) + 20
        else:  # SME
            skilled_workforce = np.random.beta(5, 5) * 100
            training_hours = np.random.gamma(2, 8) + 10
        
        # Add FDI influence
        if has_fdi:
            skilled_workforce = min(100, skilled_workforce * 1.2)
            training_hours = training_hours * 1.3
        
        # Technological resources
        modern_equipment = np.random.choice([0, 1], 
                                          p=[0.3 if not has_fdi else 0.1, 
                                             0.7 if not has_fdi else 0.9])
        machinery_age = np.random.gamma(3, 2) + 1 if modern_equipment else np.random.gamma(5, 2) + 5
        
        # Financial resources
        if firm_size and has_fdi:
            access_credit = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])  # Easy, Moderate, Difficult
        elif firm_size or has_fdi:
            access_credit = np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        else:
            access_credit = np.random.choice([1, 2, 3], p=[0.1, 0.4, 0.5])
        
        reinvestment_rate = np.random.beta(3, 7) * 100 if has_fdi else np.random.beta(2, 8) * 100
        
        return {
            'hr_skilled_workforce_percent': round(skilled_workforce, 1),
            'hr_training_hours': round(training_hours),
            'tech_modern_equipment': modern_equipment,
            'tech_machinery_age': round(machinery_age),
            'fin_access_credit': access_credit,
            'fin_reinvestment_rate': round(reinvestment_rate, 1),
            'firm_resources_score': (skilled_workforce/20 + training_hours/50 + 
                                    modern_equipment*2 + (10-machinery_age)/2 + 
                                    (4-access_credit) + reinvestment_rate/20) / 6
        }
    
    def generate_government_policy(self, has_fdi, firm_size):
        """Generate government policy perception scores"""
        # Larger firms and those with FDI tend to have more favorable views
        base_score = 3.5 if not firm_size else 4.0
        if has_fdi:
            base_score += 0.5
        
        policy_scores = []
        policy_areas = ['tax_incentives', 'regulatory_stability', 
                       'infrastructure_support', 'ease_permits']
        
        for area in policy_areas:
            score = np.random.normal(base_score, 1.2)
            score = max(1, min(7, round(score)))
            policy_scores.append(score)
        
        return {
            'gp_tax_incentives': policy_scores[0],
            'gp_regulatory_stability': policy_scores[1],
            'gp_infrastructure_support': policy_scores[2],
            'gp_ease_permits': policy_scores[3],
            'gp_average': np.mean(policy_scores)
        }
    
    def generate_performance_metrics(self, has_fdi, ka_avg, tp_avg, inn_score, fr_score, gp_avg):
        """Generate performance metrics"""
        # Complex relationships based on theoretical model
        base_performance = 30 + (has_fdi * 10) + (ka_avg * 3) + (tp_avg * 4) + \
                          (inn_score * 3) + (fr_score * 2) + (gp_avg * 1.5)
        
        # Add moderation effect of government policy
        if gp_avg > 4:
            base_performance *= 1.1
        
        # Financial metrics
        roi = np.random.normal(base_performance/4, 5)
        roi = max(5, min(40, roi))
        
        roa = roi * 0.7 + np.random.normal(0, 2)
        roa = max(3, min(30, roa))
        
        export_intensity = 0 if not has_fdi else np.random.beta(2, 5) * 100
        
        # Operational metrics
        capacity_utilization = np.random.normal(base_performance * 1.5, 10)
        capacity_utilization = max(40, min(95, capacity_utilization))
        
        market_share = np.random.beta(2, 10) * 100 if not has_fdi else np.random.beta(3, 8) * 100
        
        return {
            'perf_roi': round(roi, 1),
            'perf_roa': round(roa, 1),
            'perf_export_intensity': round(export_intensity, 1),
            'perf_capacity_utilization': round(capacity_utilization, 1),
            'perf_market_share': round(market_share, 1),
            'performance_composite': round((roi/40 + roa/30 + export_intensity/100 + 
                                           capacity_utilization/100 + market_share/100) / 5 * 100, 1)
        }
    
    def generate_survey_metadata(self, index):
        """Generate survey metadata"""
        # Generate survey date (last 6 months)
        days_ago = random.randint(0, 180)
        survey_date = datetime.now() - timedelta(days=days_ago)
        
        # Lagos industrial zones
        zones = ['Ikeja', 'Apapa', 'Oshodi', 'Mushin', 'Agege', 
                'Ikorodu', 'Lagos Island', 'Victoria Island']
        
        return {
            'survey_date': survey_date.strftime('%Y-%m-%d'),
            'industrial_zone': random.choice(zones),
            'response_time_minutes': np.random.normal(22, 3)
        }
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print("Generating synthetic FDI survey data...")
        print(f"Target: {self.n_total} firms ({self.n_sme} SMEs, {self.n_large} large firms)")
        
        for i in range(self.n_total):
            is_sme = i < self.n_sme
            firm_index = i if is_sme else i - self.n_sme
            
            # Generate all data components
            record = {}
            
            # Firm identification
            record['firm_code'] = self.generate_firm_code(firm_index + 1, is_sme)
            record['firm_size'] = 'SME' if is_sme else 'Large'
            
            # Survey metadata
            metadata = self.generate_survey_metadata(i)
            record.update(metadata)
            
            # Firm characteristics
            firm_char = self.generate_firm_characteristics(is_sme)
            record.update(firm_char)
            
            # FDI engagement
            fdi_data = self.generate_fdi_engagement(firm_char['ownership_type'], not is_sme)
            record.update(fdi_data)
            
            # Knowledge absorption
            ka_data = self.generate_knowledge_absorption(
                fdi_data['has_fdi'], 
                fdi_data['years_fdi_partnership']
            )
            record.update(ka_data)
            
            # Task performance
            tp_data = self.generate_task_performance(
                fdi_data['has_fdi'], 
                ka_data['ka_average']
            )
            record.update(tp_data)
            
            # Innovation
            inn_data = self.generate_innovation(
                fdi_data['has_fdi'], 
                not is_sme, 
                ka_data['ka_average']
            )
            record.update(inn_data)
            
            # Firm resources
            fr_data = self.generate_firm_resources(not is_sme, fdi_data['has_fdi'])
            record.update(fr_data)
            
            # Government policy perception
            gp_data = self.generate_government_policy(fdi_data['has_fdi'], not is_sme)
            record.update(gp_data)
            
            # Performance metrics
            perf_data = self.generate_performance_metrics(
                fdi_data['has_fdi'],
                ka_data['ka_average'],
                tp_data['tp_average'],
                inn_data['innovation_score'],
                fr_data['firm_resources_score'],
                gp_data['gp_average']
            )
            record.update(perf_data)
            
            self.data.append(record)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{self.n_total} records...")
        
        # Convert to DataFrame
        self.df = pd.DataFrame(self.data)
        print(f"✓ Dataset generation complete: {len(self.df)} records")
        
        return self.df
    
    def add_missing_values(self, missing_rate=0.03):
        """Add realistic missing values to simulate non-response"""
        print(f"Adding {missing_rate*100}% missing values...")
        
        # Columns that can have missing values (not identifiers or key variables)
        missable_columns = [col for col in self.df.columns 
                          if not col.startswith('firm_') and 
                          col not in ['survey_date', 'industrial_zone', 'has_fdi']]
        
        for col in missable_columns:
            missing_mask = np.random.random(len(self.df)) < missing_rate
            self.df.loc[missing_mask, col] = np.nan
        
        print(f"✓ Missing values added")
        return self.df
    
    def create_variable_labels(self):
        """Create variable labels and descriptions"""
        labels = {
            # Identification
            'firm_code': 'Unique firm identifier',
            'firm_size': 'Firm size category (SME/Large)',
            'survey_date': 'Date of survey completion',
            'industrial_zone': 'Lagos industrial zone location',
            'response_time_minutes': 'Time taken to complete survey (minutes)',
            
            # Firm characteristics
            'years_operation': 'Years in operation',
            'employees_category': 'Number of employees (1=1-50, 2=51-250, 3=251-500, 4=500+)',
            'revenue_category': 'Annual revenue (1=<50M, 2=50M-500M, 3=500M-5B, 4=>5B)',
            'ownership_type': 'Ownership type (1=Local, 2=Foreign-owned, 3=Joint venture)',
            
            # FDI engagement
            'has_fdi': 'Has foreign direct investment (0=No, 1=Yes)',
            'fdi_equity': 'FDI type: Equity investment (0=No, 1=Yes)',
            'fdi_joint_venture': 'FDI type: Joint venture (0=No, 1=Yes)',
            'fdi_tech_transfer': 'FDI type: Technology transfer (0=No, 1=Yes)',
            'fdi_mgmt_contract': 'FDI type: Management contract (0=No, 1=Yes)',
            'years_fdi_partnership': 'Years with FDI partnership',
            
            # Knowledge absorption (1-5 scale)
            'ka_technical_manuals': 'KA: Acquire technical manuals from FDI partners',
            'ka_staff_training': 'KA: Staff receive training from foreign partners',
            'ka_tech_adaptation': 'KA: Adapt foreign technology to local needs',
            'ka_knowledge_commercialization': 'KA: Commercialize knowledge from FDI',
            'ka_average': 'Knowledge absorption average score',
            
            # Task performance (1-5 scale)
            'tp_production_efficiency': 'TP: Production efficiency',
            'tp_quality_control': 'TP: Quality control',
            'tp_order_fulfillment': 'TP: Order fulfillment time',
            'tp_employee_productivity': 'TP: Employee productivity',
            'tp_average': 'Task performance average score',
            
            # Innovation
            'rd_spending_percent': 'R&D spending as % of revenue',
            'new_products_3years': 'New products launched (past 3 years)',
            'process_inn_iot': 'Process innovation: IoT systems (0=No, 1=Yes)',
            'process_inn_automation': 'Process innovation: Automation (0=No, 1=Yes)',
            'process_inn_quality': 'Process innovation: Quality management (0=No, 1=Yes)',
            'process_inn_other': 'Process innovation: Other (0=No, 1=Yes)',
            'innovation_score': 'Innovation composite score',
            
            # Firm resources
            'hr_skilled_workforce_percent': '% of skilled workforce',
            'hr_training_hours': 'Annual training hours per employee',
            'tech_modern_equipment': 'Use of modern equipment (0=No, 1=Yes)',
            'tech_machinery_age': 'Age of primary machinery (years)',
            'fin_access_credit': 'Access to credit (1=Easy, 2=Moderate, 3=Difficult)',
            'fin_reinvestment_rate': 'Reinvestment rate (%)',
            'firm_resources_score': 'Firm resources composite score',
            
            # Government policy (1-7 scale)
            'gp_tax_incentives': 'GP: Tax incentives effectiveness',
            'gp_regulatory_stability': 'GP: Regulatory stability',
            'gp_infrastructure_support': 'GP: Infrastructure support',
            'gp_ease_permits': 'GP: Ease of obtaining permits',
            'gp_average': 'Government policy average score',
            
            # Performance metrics
            'perf_roi': 'Return on Investment (%)',
            'perf_roa': 'Return on Assets (%)',
            'perf_export_intensity': 'Export intensity (%)',
            'perf_capacity_utilization': 'Production capacity utilization (%)',
            'perf_market_share': 'Market share in Lagos (%)',
            'performance_composite': 'Overall performance composite score'
        }
        
        return labels
    
    def export_data(self):
        """Export data to multiple formats"""
        print("\nExporting data to multiple formats...")
        
        # CSV format
        self.df.to_csv('fdi_survey_data.csv', index=False)
        print("✓ Exported to CSV: fdi_survey_data.csv")
        
        # Excel format with multiple sheets
        with pd.ExcelWriter('fdi_survey_data.xlsx', engine='openpyxl') as writer:
            # Main data
            self.df.to_excel(writer, sheet_name='Survey_Data', index=False)
            
            # Variable labels
            labels_df = pd.DataFrame(list(self.create_variable_labels().items()),
                                    columns=['Variable', 'Description'])
            labels_df.to_excel(writer, sheet_name='Variable_Labels', index=False)
            
            # Summary statistics
            summary_stats = self.df.describe()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Correlation matrix for key variables
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlation_matrix = self.df[numeric_cols].corr()
            correlation_matrix.to_excel(writer, sheet_name='Correlations')
        
        print("✓ Exported to Excel: fdi_survey_data.xlsx")
        
        # SPSS syntax file
        self.create_spss_syntax()
        print("✓ Created SPSS syntax: import_fdi_data.sps")
        
        # Stata format
        try:
            self.df.to_stata('fdi_survey_data.dta', write_index=False, version=118)
            print("✓ Exported to Stata: fdi_survey_data.dta")
        except:
            print("  Note: Stata export requires additional configuration")
        
        return True
    
    def create_spss_syntax(self):
        """Create SPSS syntax for importing and labeling data"""
        syntax = """* SPSS Syntax for FDI Survey Data Import
* Generated: {}

* Import CSV data
GET DATA /TYPE=TXT
  /FILE='fdi_survey_data.csv'
  /DELCASE=LINE
  /DELIMITERS=","
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
""".format(datetime.now().strftime('%Y-%m-%d'))
        
        # Add variable definitions
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                syntax += f"    {col} F8.2\n"
            else:
                syntax += f"    {col} A20\n"
        
        syntax += ".\n\n"
        
        # Add variable labels
        syntax += "* Variable Labels\n"
        labels = self.create_variable_labels()
        for var, label in labels.items():
            syntax += f"VARIABLE LABELS {var} '{label}'.\n"
        
        syntax += "\n* Value Labels\n"
        syntax += """
VALUE LABELS employees_category
    1 '1-50 employees'
    2 '51-250 employees'
    3 '251-500 employees'
    4 '500+ employees'.

VALUE LABELS revenue_category
    1 'Less than 50M Naira'
    2 '50M-500M Naira'
    3 '500M-5B Naira'
    4 'More than 5B Naira'.

VALUE LABELS ownership_type
    1 'Local'
    2 'Foreign-owned'
    3 'Joint venture'.

VALUE LABELS fin_access_credit
    1 'Easy'
    2 'Moderate'
    3 'Difficult'.

* Save as SPSS file
SAVE OUTFILE='fdi_survey_data.sav'.
EXECUTE.
"""
        
        with open('import_fdi_data.sps', 'w') as f:
            f.write(syntax)
        
        return True
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\nGenerating summary report...")
        
        report = """
================================================================================
FDI INFLUENCE ON FOOD PROCESSING FIRMS - SYNTHETIC DATASET SUMMARY
================================================================================
Generated: {}
Total Sample Size: {} firms
- SMEs: {} firms ({:.1f}%)
- Large firms: {} firms ({:.1f}%)

SECTION 1: SAMPLE CHARACTERISTICS
================================================================================
""".format(
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            self.n_total,
            self.n_sme, (self.n_sme/self.n_total)*100,
            self.n_large, (self.n_large/self.n_total)*100
        )
        
        # FDI engagement summary
        fdi_firms = self.df['has_fdi'].sum()
        report += f"""
FDI Engagement:
- Firms with FDI: {fdi_firms} ({(fdi_firms/self.n_total)*100:.1f}%)
- Firms without FDI: {self.n_total - fdi_firms} ({((self.n_total - fdi_firms)/self.n_total)*100:.1f}%)

Average years of FDI partnership: {self.df[self.df['has_fdi']==1]['years_fdi_partnership'].mean():.1f} years
"""
        
        # Geographic distribution
        report += "\nGeographic Distribution (Industrial Zones):\n"
        zone_dist = self.df['industrial_zone'].value_counts()
        for zone, count in zone_dist.items():
            report += f"- {zone}: {count} firms ({(count/self.n_total)*100:.1f}%)\n"
        
        # Key metrics summary
        report += """
SECTION 2: KEY PERFORMANCE INDICATORS
================================================================================
"""
        key_metrics = ['perf_roi', 'perf_roa', 'perf_export_intensity', 
                      'perf_capacity_utilization', 'perf_market_share']
        
        for metric in key_metrics:
            label = self.create_variable_labels()[metric]
            mean_val = self.df[metric].mean()
            std_val = self.df[metric].std()
            report += f"\n{label}:\n"
            report += f"  Overall: Mean={mean_val:.2f}, SD={std_val:.2f}\n"
            
            # Compare FDI vs non-FDI
            fdi_mean = self.df[self.df['has_fdi']==1][metric].mean()
            no_fdi_mean = self.df[self.df['has_fdi']==0][metric].mean()
            report += f"  With FDI: {fdi_mean:.2f}\n"
            report += f"  Without FDI: {no_fdi_mean:.2f}\n"
            report += f"  Difference: {fdi_mean - no_fdi_mean:.2f}\n"
        
        # Correlation insights
        report += """
SECTION 3: KEY CORRELATIONS
================================================================================
"""
        correlations = [
            ('has_fdi', 'performance_composite', 'FDI → Performance'),
            ('ka_average', 'performance_composite', 'Knowledge Absorption → Performance'),
            ('innovation_score', 'performance_composite', 'Innovation → Performance'),
            ('gp_average', 'performance_composite', 'Gov Policy → Performance')
        ]
        
        for var1, var2, label in correlations:
            corr = self.df[[var1, var2]].corr().iloc[0, 1]
            report += f"{label}: r = {corr:.3f}\n"
        
        # Data quality
        report += """
SECTION 4: DATA QUALITY METRICS
================================================================================
"""
        missing_summary = self.df.isnull().sum()
        total_missing = missing_summary.sum()
        total_cells = len(self.df) * len(self.df.columns)
        
        report += f"Total missing values: {total_missing} ({(total_missing/total_cells)*100:.2f}% of all data points)\n"
        report += f"Variables with most missing values:\n"
        
        top_missing = missing_summary.nlargest(5)
        for var, count in top_missing.items():
            if count > 0:
                report += f"  - {var}: {count} missing ({(count/len(self.df))*100:.1f}%)\n"
        
        # Statistical tests preview
        report += """
SECTION 5: PRELIMINARY STATISTICAL TESTS
================================================================================
"""
        # T-test for performance difference
        fdi_perf = self.df[self.df['has_fdi']==1]['performance_composite']
        no_fdi_perf = self.df[self.df['has_fdi']==0]['performance_composite']
        t_stat, p_value = stats.ttest_ind(fdi_perf, no_fdi_perf)
        
        report += f"T-test: Performance (FDI vs Non-FDI)\n"
        report += f"  t-statistic: {t_stat:.3f}\n"
        report += f"  p-value: {p_value:.4f}\n"
        report += f"  Result: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05\n"
        
        # Save report
        with open('fdi_data_summary_report.txt', 'w') as f:
            f.write(report)
        
        print("✓ Summary report saved: fdi_data_summary_report.txt")
        print("\nKey Statistics Preview:")
        print(f"- Firms with FDI: {(fdi_firms/self.n_total)*100:.1f}%")
        print(f"- Average ROI: {self.df['perf_roi'].mean():.1f}%")
        print(f"- Average ROA: {self.df['perf_roa'].mean():.1f}%")
        print(f"- FDI-Performance Correlation: r={self.df[['has_fdi', 'performance_composite']].corr().iloc[0, 1]:.3f}")
        
        return report

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("FDI SURVEY DATA GENERATOR")
    print("PhD Research: Jiangsu University")
    print("Topic: FDI Influence on Food Processing Firms in Lagos, Nigeria")
    print("="*80)
    
    # Initialize generator with stratified sample sizes
    generator = FDIDataGenerator(n_sme=200, n_large=100)
    
    # Generate main dataset
    df = generator.generate_dataset()
    
    # Add realistic missing values
    df = generator.add_missing_values(missing_rate=0.02)
    
    # Export to multiple formats
    generator.export_data()
    
    # Generate summary report
    generator.generate_summary_report()
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("1. fdi_survey_data.csv - Main dataset (CSV format)")
    print("2. fdi_survey_data.xlsx - Excel workbook with multiple sheets")
    print("3. import_fdi_data.sps - SPSS syntax file")
    print("4. fdi_data_summary_report.txt - Detailed summary report")
    print("\nDataset is ready for SEM analysis in AMOS, R (lavaan), or other software!")