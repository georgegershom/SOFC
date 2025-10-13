"""
Data Dictionary and Codebook Generator for FDI Dataset
=======================================================
Creates comprehensive documentation for the food processing firms dataset
"""

import pandas as pd
import json

def create_data_dictionary():
    """Create comprehensive data dictionary"""
    
    data_dict = {
        "CONTROL_VARIABLES": {
            "firm_id": {
                "description": "Unique firm identifier",
                "type": "String",
                "format": "FIRM_XXXX",
                "example": "FIRM_0001"
            },
            "survey_date": {
                "description": "Date of survey completion",
                "type": "Date",
                "format": "YYYY-MM-DD",
                "range": "2024-01-01 to 2024-10-01"
            },
            "subsector": {
                "description": "Food processing subsector classification",
                "type": "Categorical",
                "values": {
                    "Dairy Processing": "Milk and dairy products",
                    "Grain Milling": "Flour and grain products",
                    "Beverages": "Non-alcoholic and alcoholic beverages",
                    "Meat Processing": "Meat and poultry products",
                    "Fish Processing": "Fish and seafood products",
                    "Fruits & Vegetables": "Processed fruits and vegetables",
                    "Bakery Products": "Bread and bakery items",
                    "Confectionery": "Sweets and confectionery",
                    "Oil & Fats": "Edible oils and fats",
                    "Other Food": "Other food processing"
                }
            },
            "ownership_type": {
                "description": "Firm ownership structure",
                "type": "Categorical",
                "values": {
                    "Fully Nigerian": "100% Nigerian ownership",
                    "Foreign Majority": "51-89% foreign ownership",
                    "Joint Venture": "40-60% foreign ownership",
                    "Foreign Subsidiary": "90-100% foreign ownership",
                    "Nigerian Majority JV": "10-49% foreign ownership"
                }
            },
            "firm_age": {
                "description": "Years since firm establishment",
                "type": "Continuous",
                "unit": "Years",
                "range": "1-60"
            },
            "num_employees": {
                "description": "Total number of employees",
                "type": "Continuous",
                "unit": "Count",
                "range": "5-5000"
            },
            "firm_size": {
                "description": "Firm size category based on employees",
                "type": "Categorical",
                "values": {
                    "Micro": "< 10 employees",
                    "Small": "10-49 employees",
                    "Medium": "50-249 employees",
                    "Large": "≥ 250 employees"
                }
            },
            "total_assets_mn": {
                "description": "Total firm assets",
                "type": "Continuous",
                "unit": "Million Naira",
                "range": "10-50000"
            },
            "export_intensity": {
                "description": "Percentage of revenue from exports",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "0-100"
            }
        },
        
        "FDI_CONSTRUCTS": {
            "knowledge_acquisition": {
                "description": "Ability to identify and acquire external knowledge",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Low, 5=Very High",
                "source": "Zahra & George (2002)"
            },
            "knowledge_assimilation": {
                "description": "Ability to analyze and understand external knowledge",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Low, 5=Very High",
                "source": "Zahra & George (2002)"
            },
            "knowledge_transformation": {
                "description": "Ability to combine existing and new knowledge",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Low, 5=Very High",
                "source": "Zahra & George (2002)"
            },
            "knowledge_exploitation": {
                "description": "Ability to apply knowledge for commercial ends",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Low, 5=Very High",
                "source": "Zahra & George (2002)"
            },
            "knowledge_absorption_avg": {
                "description": "Average knowledge absorption capacity",
                "type": "Composite Score",
                "scale": "1-5",
                "calculation": "Mean of 4 knowledge dimensions"
            },
            "task_quality": {
                "description": "Quality of task execution",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Poor, 5=Excellent",
                "source": "Koopmans et al. (2013)"
            },
            "task_efficiency": {
                "description": "Efficiency in task completion",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Inefficient, 5=Very Efficient",
                "source": "Koopmans et al. (2013)"
            },
            "task_innovation": {
                "description": "Innovation in task approaches",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Not Innovative, 5=Highly Innovative",
                "source": "Koopmans et al. (2013)"
            },
            "task_performance_avg": {
                "description": "Average task performance",
                "type": "Composite Score",
                "scale": "1-5",
                "calculation": "Mean of 3 task dimensions"
            },
            "product_innovation": {
                "description": "Introduction of new or improved products",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=No Innovation, 5=High Innovation",
                "source": "OECD Oslo Manual"
            },
            "process_innovation": {
                "description": "Implementation of new production processes",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=No Innovation, 5=High Innovation",
                "source": "OECD Oslo Manual"
            },
            "marketing_innovation": {
                "description": "New marketing methods implementation",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=No Innovation, 5=High Innovation",
                "source": "OECD Oslo Manual"
            },
            "organizational_innovation": {
                "description": "New organizational methods in business practices",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=No Innovation, 5=High Innovation",
                "source": "OECD Oslo Manual"
            },
            "innovation_avg": {
                "description": "Average innovation capacity",
                "type": "Composite Score",
                "scale": "1-5",
                "calculation": "Mean of 4 innovation dimensions"
            },
            "fdi_ownership_pct": {
                "description": "Percentage of foreign ownership",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "0-100"
            }
        },
        
        "FIRM_PERFORMANCE": {
            "roi_pct": {
                "description": "Return on Investment",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "-10 to 50",
                "interpretation": "Annual ROI percentage"
            },
            "roa_pct": {
                "description": "Return on Assets",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "-15 to 40",
                "interpretation": "Annual ROA percentage"
            },
            "revenue_growth_pct": {
                "description": "Year-over-year revenue growth",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "-20 to 60",
                "interpretation": "YoY revenue growth rate"
            },
            "market_share_pct": {
                "description": "Market share in subsector",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "0.1-25",
                "interpretation": "Percentage of subsector market"
            },
            "operational_efficiency": {
                "description": "Overall operational efficiency rating",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Inefficient, 5=Very Efficient"
            },
            "productivity_index": {
                "description": "Labor productivity index",
                "type": "Continuous",
                "unit": "Index",
                "range": "40-150",
                "interpretation": "100 = industry average"
            }
        },
        
        "FIRM_RESOURCES": {
            "skilled_labor_ratio": {
                "description": "Percentage of skilled workers",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "5-90",
                "category": "Human Capital"
            },
            "training_hours_per_emp": {
                "description": "Annual training hours per employee",
                "type": "Continuous",
                "unit": "Hours",
                "range": "0-100",
                "category": "Human Capital"
            },
            "management_quality": {
                "description": "Quality of management practices",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Poor, 5=Excellent",
                "category": "Human Capital"
            },
            "iot_adoption": {
                "description": "Internet of Things technology adoption",
                "type": "Binary",
                "values": {"0": "Not adopted", "1": "Adopted"},
                "category": "Technological"
            },
            "rd_spend_pct": {
                "description": "R&D spending as percentage of revenue",
                "type": "Continuous",
                "unit": "Percentage",
                "range": "0-15",
                "category": "Technological"
            },
            "tech_sophistication": {
                "description": "Overall technology sophistication level",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Basic, 5=Advanced",
                "category": "Technological"
            },
            "erp_adoption": {
                "description": "Enterprise Resource Planning system adoption",
                "type": "Binary",
                "values": {"0": "Not adopted", "1": "Adopted"},
                "category": "Technological"
            },
            "liquidity_ratio": {
                "description": "Current assets to current liabilities ratio",
                "type": "Continuous",
                "unit": "Ratio",
                "range": "0.5-5",
                "interpretation": ">1 = good liquidity",
                "category": "Financial"
            },
            "access_to_credit": {
                "description": "Ease of accessing credit facilities",
                "type": "Likert Scale",
                "scale": "1-5",
                "interpretation": "1=Very Difficult, 5=Very Easy",
                "category": "Financial"
            },
            "investment_capacity_mn": {
                "description": "Available capital for investment",
                "type": "Continuous",
                "unit": "Million Naira",
                "range": "0-5000",
                "category": "Financial"
            }
        },
        
        "GOVERNMENT_POLICY": {
            "tax_incentives_eff": {
                "description": "Effectiveness of tax incentive policies",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Completely Ineffective, 7=Highly Effective"
            },
            "regulatory_stability": {
                "description": "Stability and predictability of regulations",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Very Unstable, 7=Very Stable"
            },
            "infrastructure_support": {
                "description": "Quality of government infrastructure support",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Very Poor, 7=Excellent"
            },
            "trade_facilitation": {
                "description": "Effectiveness of trade facilitation measures",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Very Poor, 7=Excellent"
            },
            "investment_protection": {
                "description": "Level of investment protection provided",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=No Protection, 7=Strong Protection"
            },
            "bureaucratic_burden": {
                "description": "Level of bureaucratic complexity (reverse coded)",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Low Burden, 7=High Burden"
            },
            "corruption_experience": {
                "description": "Experience with corruption (reverse coded)",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=No Corruption, 7=High Corruption"
            },
            "policy_consistency": {
                "description": "Consistency of government policies over time",
                "type": "Likert Scale",
                "scale": "1-7",
                "interpretation": "1=Very Inconsistent, 7=Very Consistent"
            },
            "govt_support_utilized": {
                "description": "Utilization of government support programs",
                "type": "Binary",
                "values": {"0": "Not utilized", "1": "Utilized"}
            },
            "policy_effectiveness_index": {
                "description": "Overall government policy effectiveness",
                "type": "Composite Score",
                "scale": "1-7",
                "calculation": "Mean of 8 policy dimensions (with reverse coding applied)"
            }
        },
        
        "INTERACTION_VARIABLES": {
            "fdi_composite": {
                "description": "Composite FDI exposure score",
                "type": "Composite Score",
                "scale": "0-1",
                "calculation": "Weighted average of FDI ownership, knowledge absorption, and innovation"
            },
            "performance_composite": {
                "description": "Composite firm performance score",
                "type": "Composite Score",
                "scale": "0-1",
                "calculation": "Weighted average of ROI, ROA, efficiency, and productivity"
            },
            "fdi_x_policy": {
                "description": "Interaction term: FDI × Government Policy",
                "type": "Interaction",
                "calculation": "fdi_composite × policy_effectiveness_index",
                "interpretation": "Tests moderation effect of policy on FDI-performance relationship"
            },
            "performance_adjusted": {
                "description": "Performance adjusted for moderation effects",
                "type": "Composite Score",
                "calculation": "Performance with FDI-policy interaction effects"
            },
            "fdi_x_policy_x_size": {
                "description": "Three-way interaction: FDI × Policy × Size",
                "type": "Interaction",
                "calculation": "fdi_x_policy × firm_size_dummy",
                "interpretation": "Tests if moderation varies by firm size"
            }
        }
    }
    
    return data_dict


def create_codebook():
    """Create detailed codebook with variable descriptions and coding schemes"""
    
    codebook = """
================================================================================
CODEBOOK: Food Processing Firms FDI Study - Lagos, Nigeria
================================================================================

STUDY INFORMATION
-----------------
Title: The Influence of Foreign Direct Investment on the Performance of Food 
       Processing Firms in Lagos, Nigeria: The Moderating Role of Nigerian 
       Government Policy

Data Collection: January 2024 - October 2024
Sample Size: 500 firms
Location: Lagos, Nigeria
Industry: Food Processing

DATA STRUCTURE
--------------
- Unit of Analysis: Firm
- Data Type: Cross-sectional survey
- Missing Data: MCAR (Missing Completely at Random) ~2%

VARIABLE NAMING CONVENTIONS
---------------------------
- _pct: Percentage values (0-100)
- _mn: Values in millions of Naira
- _avg: Average/composite scores
- _x_: Interaction terms

SCALE INTERPRETATIONS
----------------------

Likert Scale 1-5:
1 = Strongly Disagree / Very Low / Very Poor
2 = Disagree / Low / Poor  
3 = Neutral / Moderate / Fair
4 = Agree / High / Good
5 = Strongly Agree / Very High / Excellent

Likert Scale 1-7:
1 = Completely Ineffective / Strongly Disagree
2 = Very Ineffective / Disagree
3 = Ineffective / Somewhat Disagree
4 = Neutral
5 = Effective / Somewhat Agree
6 = Very Effective / Agree
7 = Highly Effective / Strongly Agree

Binary Variables:
0 = No / Not Present / Not Adopted
1 = Yes / Present / Adopted

KEY CONSTRUCTS
--------------

1. FOREIGN DIRECT INVESTMENT (FDI)
   - Measured through ownership percentage and knowledge transfer metrics
   - Based on Zahra & George (2002) absorptive capacity framework
   
2. FIRM PERFORMANCE
   - Financial: ROI, ROA, Revenue Growth
   - Operational: Efficiency, Productivity
   - Market: Market Share, Export Intensity
   
3. FIRM RESOURCES (RBV Theory)
   - Human Capital: Skills, Training, Management
   - Technological: IoT, R&D, ERP Systems
   - Financial: Liquidity, Credit Access, Investment Capacity
   
4. GOVERNMENT POLICY (Moderator)
   - Regulatory Environment: Stability, Consistency
   - Support Measures: Tax Incentives, Infrastructure
   - Business Climate: Bureaucracy, Corruption
   
5. INTERACTION EFFECTS
   - Primary: FDI × Policy → Performance
   - Secondary: Three-way with Firm Size

DATA QUALITY NOTES
------------------
- Self-reported data validated where possible with secondary sources
- Likert scales tested for reliability (Cronbach's α > 0.7)
- Common method bias addressed through survey design
- Outliers winsorized at 1st and 99th percentiles

ANALYSIS RECOMMENDATIONS
------------------------
1. Check multicollinearity (VIF < 10)
2. Test for heteroscedasticity
3. Consider hierarchical regression for moderation
4. Use robust standard errors
5. Control for industry subsector effects

CITATION
--------
When using this dataset, please cite:
"Synthetic Dataset for FDI Study of Food Processing Firms in Lagos, Nigeria (2024)"

================================================================================
"""
    
    return codebook


def save_documentation():
    """Save all documentation files"""
    
    # Create data dictionary
    data_dict = create_data_dictionary()
    
    # Save as JSON
    with open('data_dictionary.json', 'w') as f:
        json.dump(data_dict, f, indent=2)
    print("✓ Saved data_dictionary.json")
    
    # Create and save codebook
    codebook = create_codebook()
    with open('CODEBOOK.txt', 'w') as f:
        f.write(codebook)
    print("✓ Saved CODEBOOK.txt")
    
    # Create variable list CSV
    var_list = []
    for category, variables in data_dict.items():
        for var_name, var_info in variables.items():
            var_list.append({
                'Category': category,
                'Variable': var_name,
                'Type': var_info.get('type', ''),
                'Description': var_info.get('description', ''),
                'Scale/Unit': var_info.get('scale', var_info.get('unit', '')),
                'Range': var_info.get('range', '')
            })
    
    var_df = pd.DataFrame(var_list)
    var_df.to_csv('variable_list.csv', index=False)
    print("✓ Saved variable_list.csv")
    
    return data_dict, codebook


if __name__ == "__main__":
    save_documentation()