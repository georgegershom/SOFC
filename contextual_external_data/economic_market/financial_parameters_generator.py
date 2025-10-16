#!/usr/bin/env python3
"""
Financial Parameters Data Generator
Generates comprehensive financial data for building retrofit economic analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class FinancialParametersGenerator:
    def __init__(self, location="United States", currency="USD"):
        self.location = location
        self.currency = currency
        
        # Base financial parameters (2023 baseline)
        self.base_parameters = {
            'discount_rates': {
                'federal_government': 0.025,
                'state_local_government': 0.035,
                'commercial_real_estate': 0.065,
                'residential_owner': 0.045,
                'utility_company': 0.055,
                'energy_service_company': 0.085
            },
            'inflation_rates': {
                'general_inflation': 0.032,
                'energy_inflation': 0.045,
                'construction_inflation': 0.038,
                'labor_inflation': 0.042
            },
            'tax_rates': {
                'federal_corporate': 0.21,
                'average_state_corporate': 0.065,
                'property_tax_rate': 0.012,
                'sales_tax_rate': 0.075
            },
            'financing_terms': {
                'commercial_loan_rate': 0.075,
                'residential_mortgage_rate': 0.065,
                'equipment_financing_rate': 0.085,
                'green_bond_rate': 0.045,
                'pace_financing_rate': 0.055
            }
        }
        
        # Government incentives and rebates
        self.incentive_programs = {
            'federal_tax_credits': {
                'residential_solar_itc': {
                    'credit_rate': 0.30,
                    'max_credit': None,
                    'expiration_year': 2032,
                    'eligible_technologies': ['solar_pv', 'solar_thermal', 'battery_storage']
                },
                'commercial_solar_itc': {
                    'credit_rate': 0.30,
                    'max_credit': None,
                    'expiration_year': 2032,
                    'eligible_technologies': ['solar_pv', 'solar_thermal', 'battery_storage']
                },
                'section_179d_deduction': {
                    'deduction_per_sqft': 1.88,
                    'max_deduction_per_sqft': 5.65,
                    'eligible_technologies': ['lighting', 'hvac', 'building_envelope']
                }
            },
            'state_programs': {
                'california_sgip': {
                    'rebate_per_kwh': 200,
                    'max_rebate': 50000,
                    'eligible_technologies': ['battery_storage']
                },
                'new_york_nyserda': {
                    'heat_pump_rebate': 1500,
                    'insulation_rebate_per_sqft': 0.50,
                    'solar_rebate_per_watt': 0.40
                },
                'massachusetts_mass_save': {
                    'heat_pump_rebate': 1250,
                    'insulation_rebate': 2000,
                    'air_sealing_rebate': 2000
                }
            },
            'utility_programs': {
                'demand_response_payments': {
                    'payment_per_kw': 75,
                    'duration_months': 12,
                    'participation_requirements': 'automated_controls'
                },
                'energy_efficiency_rebates': {
                    'hvac_rebate': 500,
                    'lighting_rebate_per_fixture': 25,
                    'smart_thermostat_rebate': 100
                }
            }
        }
    
    def generate_discount_rate_scenarios(self, start_year=2024, end_year=2040):
        """Generate discount rate scenarios over time"""
        
        years = list(range(start_year, end_year + 1))
        scenarios = ['low', 'base', 'high']
        
        discount_data = []
        
        for scenario in scenarios:
            for year in years:
                years_from_base = year - 2023
                
                # Define scenario adjustments
                if scenario == 'low':
                    rate_adjustment = -0.01  # 1% lower rates
                elif scenario == 'high':
                    rate_adjustment = 0.015  # 1.5% higher rates
                else:  # base
                    rate_adjustment = 0.005 * years_from_base / 10  # Gradual increase
                
                for entity, base_rate in self.base_parameters['discount_rates'].items():
                    adjusted_rate = max(0.01, base_rate + rate_adjustment)  # Floor at 1%
                    
                    discount_data.append({
                        'year': year,
                        'scenario': scenario,
                        'entity_type': entity,
                        'discount_rate': adjusted_rate,
                        'base_rate': base_rate,
                        'adjustment': rate_adjustment,
                        'location': self.location
                    })
        
        return pd.DataFrame(discount_data)
    
    def generate_inflation_forecasts(self, start_year=2024, end_year=2040):
        """Generate inflation rate forecasts by category"""
        
        years = list(range(start_year, end_year + 1))
        inflation_data = []
        
        # Define inflation scenarios
        scenarios = {
            'low_inflation': {
                'general_inflation': 0.02,
                'energy_inflation': 0.025,
                'construction_inflation': 0.022,
                'labor_inflation': 0.028
            },
            'base_case': {
                'general_inflation': 0.032,
                'energy_inflation': 0.045,
                'construction_inflation': 0.038,
                'labor_inflation': 0.042
            },
            'high_inflation': {
                'general_inflation': 0.045,
                'energy_inflation': 0.065,
                'construction_inflation': 0.055,
                'labor_inflation': 0.058
            }
        }
        
        for scenario_name, scenario_rates in scenarios.items():
            for year in years:
                for category, annual_rate in scenario_rates.items():
                    # Add some year-to-year volatility
                    np.random.seed(year + hash(scenario_name + category))
                    volatility = np.random.normal(0, 0.005)  # 0.5% standard deviation
                    adjusted_rate = max(0, annual_rate + volatility)
                    
                    inflation_data.append({
                        'year': year,
                        'scenario': scenario_name,
                        'inflation_category': category,
                        'inflation_rate': adjusted_rate,
                        'base_rate': annual_rate,
                        'volatility_adjustment': volatility,
                        'location': self.location
                    })
        
        return pd.DataFrame(inflation_data)
    
    def generate_financing_options(self, year=2023):
        """Generate detailed financing options for retrofit projects"""
        
        financing_options = []
        
        # Traditional financing
        traditional_options = {
            'conventional_loan': {
                'interest_rate': 0.075,
                'term_years': 10,
                'down_payment_required': 0.20,
                'max_loan_amount': 5000000,
                'qualification_requirements': 'Good credit, cash flow positive'
            },
            'sba_504_loan': {
                'interest_rate': 0.065,
                'term_years': 20,
                'down_payment_required': 0.10,
                'max_loan_amount': 5500000,
                'qualification_requirements': 'Small business, owner-occupied'
            },
            'equipment_financing': {
                'interest_rate': 0.085,
                'term_years': 7,
                'down_payment_required': 0.15,
                'max_loan_amount': 2000000,
                'qualification_requirements': 'Equipment as collateral'
            }
        }
        
        # Green financing
        green_options = {
            'green_bond': {
                'interest_rate': 0.045,
                'term_years': 15,
                'down_payment_required': 0.05,
                'max_loan_amount': 10000000,
                'qualification_requirements': 'Green building certification'
            },
            'pace_financing': {
                'interest_rate': 0.055,
                'term_years': 20,
                'down_payment_required': 0.00,
                'max_loan_amount': 3000000,
                'qualification_requirements': 'Property tax assessment'
            },
            'esco_performance_contract': {
                'interest_rate': 0.065,
                'term_years': 15,
                'down_payment_required': 0.00,
                'max_loan_amount': 8000000,
                'qualification_requirements': 'Guaranteed energy savings'
            }
        }
        
        # Combine all options
        all_options = {**traditional_options, **green_options}
        
        for option_name, details in all_options.items():
            financing_options.append({
                'year': year,
                'financing_type': option_name,
                'category': 'green' if option_name in green_options else 'traditional',
                'interest_rate': details['interest_rate'],
                'term_years': details['term_years'],
                'down_payment_required': details['down_payment_required'],
                'max_loan_amount': details['max_loan_amount'],
                'qualification_requirements': details['qualification_requirements'],
                'location': self.location
            })
        
        return pd.DataFrame(financing_options)
    
    def generate_incentive_values(self, year=2023):
        """Generate current values of government incentives and rebates"""
        
        incentive_data = []
        
        # Process federal incentives
        for program_name, program_details in self.incentive_programs['federal_tax_credits'].items():
            incentive_data.append({
                'year': year,
                'incentive_level': 'federal',
                'program_name': program_name,
                'incentive_type': 'tax_credit',
                'value_type': 'percentage' if 'credit_rate' in program_details else 'fixed_amount',
                'value': program_details.get('credit_rate', program_details.get('deduction_per_sqft')),
                'max_value': program_details.get('max_credit', program_details.get('max_deduction_per_sqft')),
                'expiration_year': program_details.get('expiration_year'),
                'eligible_technologies': program_details['eligible_technologies'],
                'location': self.location
            })
        
        # Process state incentives
        for state, programs in self.incentive_programs['state_programs'].items():
            for program_name, program_details in programs.items():
                for key, value in program_details.items():
                    if 'rebate' in key and isinstance(value, (int, float)):
                        incentive_data.append({
                            'year': year,
                            'incentive_level': 'state',
                            'state': state,
                            'program_name': program_name,
                            'incentive_type': 'rebate',
                            'measure': key,
                            'value': value,
                            'location': self.location
                        })
        
        # Process utility incentives
        for program_name, program_details in self.incentive_programs['utility_programs'].items():
            for key, value in program_details.items():
                if isinstance(value, (int, float)):
                    incentive_data.append({
                        'year': year,
                        'incentive_level': 'utility',
                        'program_name': program_name,
                        'incentive_type': 'rebate',
                        'measure': key,
                        'value': value,
                        'location': self.location
                    })
        
        return pd.DataFrame(incentive_data)
    
    def calculate_project_economics(self, project_data):
        """Calculate comprehensive project economics"""
        
        # Extract project parameters
        initial_cost = project_data['initial_cost']
        annual_savings = project_data['annual_savings']
        project_life = project_data['project_life']
        discount_rate = project_data.get('discount_rate', 0.065)
        inflation_rate = project_data.get('inflation_rate', 0.032)
        
        # Calculate cash flows
        cash_flows = []
        for year in range(project_life + 1):
            if year == 0:
                cash_flow = -initial_cost
            else:
                # Escalate savings with inflation
                escalated_savings = annual_savings * ((1 + inflation_rate) ** year)
                cash_flow = escalated_savings
            
            cash_flows.append({
                'year': year,
                'cash_flow': cash_flow,
                'present_value': cash_flow / ((1 + discount_rate) ** year)
            })
        
        # Calculate financial metrics
        npv = sum([cf['present_value'] for cf in cash_flows])
        
        # Simple payback period
        cumulative_savings = 0
        payback_period = None
        for year in range(1, project_life + 1):
            escalated_savings = annual_savings * ((1 + inflation_rate) ** year)
            cumulative_savings += escalated_savings
            if cumulative_savings >= initial_cost and payback_period is None:
                payback_period = year
        
        # IRR calculation (simplified)
        irr = self._calculate_irr([cf['cash_flow'] for cf in cash_flows])
        
        # Savings-to-Investment Ratio
        sir = sum([cf['present_value'] for cf in cash_flows[1:]]) / abs(cash_flows[0]['present_value'])
        
        return {
            'cash_flows': cash_flows,
            'financial_metrics': {
                'net_present_value': npv,
                'internal_rate_of_return': irr,
                'simple_payback_period': payback_period,
                'savings_to_investment_ratio': sir,
                'benefit_cost_ratio': sir,
                'total_life_cycle_savings': sum([cf['cash_flow'] for cf in cash_flows[1:]])
            },
            'assumptions': {
                'discount_rate': discount_rate,
                'inflation_rate': inflation_rate,
                'project_life': project_life,
                'initial_cost': initial_cost,
                'annual_savings': annual_savings
            }
        }
    
    def _calculate_irr(self, cash_flows, precision=0.0001):
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        
        def npv_function(rate):
            return sum([cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows)])
        
        def npv_derivative(rate):
            return sum([-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows)])
        
        # Initial guess
        rate = 0.1
        
        for _ in range(100):  # Maximum iterations
            npv = npv_function(rate)
            if abs(npv) < precision:
                return rate
            
            derivative = npv_derivative(rate)
            if abs(derivative) < precision:
                return None  # Cannot calculate IRR
            
            rate = rate - npv / derivative
        
        return rate if abs(npv_function(rate)) < precision else None
    
    def generate_sensitivity_analysis_parameters(self):
        """Generate parameters for sensitivity analysis"""
        
        sensitivity_parameters = {
            'discount_rate': {
                'base_value': 0.065,
                'low_value': 0.035,
                'high_value': 0.095,
                'distribution': 'normal',
                'std_dev': 0.015
            },
            'energy_price_escalation': {
                'base_value': 0.045,
                'low_value': 0.02,
                'high_value': 0.08,
                'distribution': 'normal',
                'std_dev': 0.015
            },
            'construction_cost_escalation': {
                'base_value': 0.038,
                'low_value': 0.015,
                'high_value': 0.065,
                'distribution': 'normal',
                'std_dev': 0.012
            },
            'equipment_performance_degradation': {
                'base_value': 0.005,
                'low_value': 0.002,
                'high_value': 0.01,
                'distribution': 'uniform'
            },
            'maintenance_cost_escalation': {
                'base_value': 0.035,
                'low_value': 0.02,
                'high_value': 0.055,
                'distribution': 'normal',
                'std_dev': 0.01
            }
        }
        
        return sensitivity_parameters

def main():
    """Generate financial parameters data"""
    
    print("Generating financial parameters data...")
    
    generator = FinancialParametersGenerator()
    
    # Generate all financial datasets
    discount_rates = generator.generate_discount_rate_scenarios()
    inflation_forecasts = generator.generate_inflation_forecasts()
    financing_options = generator.generate_financing_options()
    incentive_values = generator.generate_incentive_values()
    sensitivity_params = generator.generate_sensitivity_analysis_parameters()
    
    # Create output directory
    output_dir = "economic_market/financial_parameters"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    discount_rates.to_csv(f"{output_dir}/discount_rate_scenarios_2024_2040.csv", index=False)
    inflation_forecasts.to_csv(f"{output_dir}/inflation_forecasts_2024_2040.csv", index=False)
    financing_options.to_csv(f"{output_dir}/financing_options_2023.csv", index=False)
    incentive_values.to_csv(f"{output_dir}/government_incentives_2023.csv", index=False)
    
    # Save JSON data
    with open(f"{output_dir}/sensitivity_analysis_parameters.json", 'w') as f:
        json.dump(sensitivity_params, f, indent=2)
    
    # Generate sample project economics
    sample_project = {
        'initial_cost': 150000,
        'annual_savings': 18000,
        'project_life': 20,
        'discount_rate': 0.065,
        'inflation_rate': 0.032
    }
    
    project_economics = generator.calculate_project_economics(sample_project)
    
    with open(f"{output_dir}/sample_project_economics.json", 'w') as f:
        json.dump(project_economics, f, indent=2, default=str)
    
    # Generate comprehensive financial summary
    financial_summary = {
        'generation_date': datetime.now().isoformat(),
        'location': generator.location,
        'currency': generator.currency,
        'base_parameters': generator.base_parameters,
        'available_incentives': len(incentive_values),
        'financing_options': len(financing_options),
        'forecast_years': '2024-2040',
        'key_assumptions': {
            'base_discount_rate': '6.5%',
            'base_inflation_rate': '3.2%',
            'energy_price_escalation': '4.5%',
            'analysis_period': '20 years'
        }
    }
    
    with open(f"{output_dir}/financial_parameters_summary.json", 'w') as f:
        json.dump(financial_summary, f, indent=2)
    
    print("Financial parameters data generation completed!")

if __name__ == "__main__":
    main()