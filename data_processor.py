#!/usr/bin/env python3
"""
Retrofit Intervention & Lifecycle Data Processor
Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization

This module provides comprehensive data processing capabilities for:
- Retrofit measure analysis and optimization
- Life cycle assessment calculations
- Operational carbon footprint analysis
- Maintenance scheduling and cost optimization
- Performance degradation modeling
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RetrofitScenario:
    """Data structure for retrofit scenario analysis"""
    scenario_id: str
    building_type: str
    climate_zone: str
    floor_area_m2: float
    measures: List[str]
    analysis_period_years: int = 25
    discount_rate: float = 0.03
    
@dataclass
class LCAResults:
    """Life cycle assessment results structure"""
    embodied_carbon_kg_co2e: float
    operational_carbon_kg_co2e: float
    total_carbon_kg_co2e: float
    carbon_payback_years: float
    uncertainty_range: Tuple[float, float]

@dataclass
class EconomicResults:
    """Economic analysis results structure"""
    initial_cost_usd: float
    annual_energy_savings_usd: float
    maintenance_costs_usd: float
    net_present_value_usd: float
    payback_period_years: float
    internal_rate_of_return: float

class RetrofitDataProcessor:
    """Main class for processing retrofit intervention and lifecycle data"""
    
    def __init__(self, data_directory: str = "/workspace"):
        """Initialize the data processor with database files"""
        self.data_dir = Path(data_directory)
        self.retrofit_measures = self._load_json("retrofit_measure_library.json")
        self.lci_database = self._load_json("lifecycle_inventory_database.json")
        self.operational_data = self._load_json("operational_carbon_database.json")
        self.maintenance_data = self._load_json("maintenance_operational_database.json")
        
        logger.info("Retrofit Data Processor initialized successfully")
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON data file with error handling"""
        try:
            with open(self.data_dir / filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Data file not found: {filename}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in: {filename}")
            raise
    
    def get_measure_data(self, measure_id: str) -> Dict:
        """Retrieve comprehensive data for a specific retrofit measure"""
        # Search through all measure categories
        for category in self.retrofit_measures.values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        for measure in subcategory:
                            if measure.get('measure_id') == measure_id:
                                return measure
                    elif isinstance(subcategory, dict) and subcategory.get('measure_id') == measure_id:
                        return subcategory
        
        raise ValueError(f"Measure ID {measure_id} not found in database")
    
    def calculate_embodied_carbon(self, measure_id: str, quantity: float, 
                                 functional_unit: str = "m2") -> Dict:
        """Calculate embodied carbon for a retrofit measure"""
        measure = self.get_measure_data(measure_id)
        
        # Find corresponding LCI data
        material_data = None
        for category in self.lci_database['materials'].values():
            for material_id, material in category.items():
                if material_id.startswith(measure_id.split('_')[0]):
                    material_data = material
                    break
            if material_data:
                break
        
        if not material_data:
            logger.warning(f"No LCI data found for measure {measure_id}")
            return {"error": "No LCI data available"}
        
        embodied_carbon = material_data['embodied_carbon']
        
        results = {
            'measure_id': measure_id,
            'quantity': quantity,
            'functional_unit': functional_unit,
            'embodied_carbon_breakdown': {
                'A1_A3_kg_co2e': embodied_carbon['A1_A3']['gwp_kg_co2e'] * quantity,
                'A4_A5_kg_co2e': embodied_carbon['A4_A5']['gwp_kg_co2e'] * quantity,
                'B1_B7_kg_co2e': embodied_carbon['B1_B7']['gwp_kg_co2e'] * quantity,
                'C1_C4_kg_co2e': embodied_carbon['C1_C4']['gwp_kg_co2e'] * quantity
            },
            'total_embodied_carbon_kg_co2e': embodied_carbon['total_lifecycle']['gwp_kg_co2e'] * quantity,
            'uncertainty': embodied_carbon['total_lifecycle'].get('uncertainty', 0.2)
        }
        
        return results
    
    def calculate_operational_savings(self, scenario: RetrofitScenario) -> Dict:
        """Calculate operational energy and carbon savings for a retrofit scenario"""
        building_baseline = self._get_building_baseline(scenario.building_type, scenario.climate_zone)
        
        if not building_baseline:
            raise ValueError(f"No baseline data for {scenario.building_type} in {scenario.climate_zone}")
        
        total_savings = {
            'heating_kwh': 0,
            'cooling_kwh': 0,
            'lighting_kwh': 0,
            'equipment_kwh': 0,
            'total_kwh': 0
        }
        
        # Calculate savings from each measure
        for measure_id in scenario.measures:
            measure_savings = self._calculate_measure_savings(measure_id, building_baseline, scenario)
            for key in total_savings:
                total_savings[key] += measure_savings.get(key, 0)
        
        # Apply building area scaling
        area_factor = scenario.floor_area_m2 / building_baseline.get('floor_area_m2', 1)
        for key in total_savings:
            total_savings[key] *= area_factor
        
        # Calculate carbon savings
        grid_carbon_intensity = self._get_grid_carbon_intensity(scenario.climate_zone)
        carbon_savings_kg_co2e = total_savings['total_kwh'] * grid_carbon_intensity
        
        results = {
            'scenario_id': scenario.scenario_id,
            'annual_energy_savings': total_savings,
            'annual_carbon_savings_kg_co2e': carbon_savings_kg_co2e,
            'lifecycle_carbon_savings_kg_co2e': carbon_savings_kg_co2e * scenario.analysis_period_years,
            'grid_carbon_intensity_kg_co2e_per_kwh': grid_carbon_intensity
        }
        
        return results
    
    def _get_building_baseline(self, building_type: str, climate_zone: str) -> Dict:
        """Get baseline energy use for building type and climate zone"""
        baselines = self.operational_data.get('building_baselines', {})
        
        # Search for building type
        for category, buildings in baselines.items():
            for building_id, building_data in buildings.items():
                if building_data.get('building_type') == building_type:
                    # Get climate-specific data if available
                    climate_data = building_data.get('by_climate_zone', {})
                    for zone_key, zone_data in climate_data.items():
                        if climate_zone.lower() in zone_key.lower():
                            return {**building_data, **zone_data}
                    # Return general data if climate-specific not found
                    return building_data
        
        return None
    
    def _calculate_measure_savings(self, measure_id: str, baseline: Dict, scenario: RetrofitScenario) -> Dict:
        """Calculate energy savings for a specific measure"""
        savings_factors = self.operational_data.get('energy_savings_factors', {})
        
        # Find savings factors for this measure
        measure_savings = {}
        for category, measures in savings_factors.items():
            for measure_type, measure_data in measures.items():
                if measure_id in str(measure_data):
                    # Apply savings factors to baseline consumption
                    heating_savings = baseline.get('heating_kwh', 0) * measure_data.get('heating_savings_factor', 0)
                    cooling_savings = baseline.get('cooling_kwh', 0) * measure_data.get('cooling_savings_factor', 0)
                    lighting_savings = baseline.get('lighting_kwh', 0) * measure_data.get('lighting_interaction', 0)
                    
                    measure_savings = {
                        'heating_kwh': heating_savings,
                        'cooling_kwh': cooling_savings,
                        'lighting_kwh': lighting_savings,
                        'equipment_kwh': 0,
                        'total_kwh': heating_savings + cooling_savings + lighting_savings
                    }
                    break
        
        return measure_savings
    
    def _get_grid_carbon_intensity(self, climate_zone: str) -> float:
        """Get grid carbon intensity for climate zone"""
        carbon_factors = self.operational_data.get('carbon_intensity_factors', {})
        electricity_grid = carbon_factors.get('electricity_grid', {})
        
        # Use national average as default
        return electricity_grid.get('hourly_profiles', {}).get('US_national_average', {}).get('annual_average_kg_co2e_per_kwh', 0.386)
    
    def calculate_maintenance_costs(self, measure_id: str, analysis_period_years: int = 25) -> Dict:
        """Calculate lifecycle maintenance costs for a retrofit measure"""
        # Find maintenance data for the measure
        maintenance_schedule = None
        for category in self.maintenance_data.get('maintenance_schedules', {}).values():
            for system_type in category.values():
                if system_type.get('system_id') == measure_id:
                    maintenance_schedule = system_type
                    break
        
        if not maintenance_schedule:
            logger.warning(f"No maintenance data found for measure {measure_id}")
            return {"error": "No maintenance data available"}
        
        # Calculate costs over analysis period
        total_costs = 0
        cost_breakdown = []
        
        for task in maintenance_schedule.get('maintenance_tasks', []):
            frequency_months = task.get('frequency_months', 12)
            task_cost = task.get('total_cost', 0)
            
            # Calculate number of occurrences over analysis period
            occurrences = int((analysis_period_years * 12) / frequency_months)
            task_total_cost = task_cost * occurrences
            total_costs += task_total_cost
            
            cost_breakdown.append({
                'task_name': task.get('task_name'),
                'frequency_months': frequency_months,
                'cost_per_occurrence': task_cost,
                'total_occurrences': occurrences,
                'total_cost': task_total_cost
            })
        
        # Apply cost escalation
        escalation_rate = self.maintenance_data.get('cost_escalation_factors', {}).get('labor_costs', {}).get('annual_escalation_rate', 0.035)
        escalated_costs = self._apply_cost_escalation(total_costs, analysis_period_years, escalation_rate)
        
        results = {
            'measure_id': measure_id,
            'analysis_period_years': analysis_period_years,
            'nominal_maintenance_costs': total_costs,
            'escalated_maintenance_costs': escalated_costs,
            'cost_breakdown': cost_breakdown,
            'annual_escalation_rate': escalation_rate
        }
        
        return results
    
    def _apply_cost_escalation(self, base_cost: float, years: int, escalation_rate: float) -> float:
        """Apply cost escalation over analysis period"""
        # Simple compound escalation model
        return base_cost * ((1 + escalation_rate) ** years)
    
    def perform_lifecycle_assessment(self, scenario: RetrofitScenario) -> LCAResults:
        """Perform comprehensive lifecycle assessment for retrofit scenario"""
        
        # Calculate embodied carbon for all measures
        total_embodied_carbon = 0
        embodied_uncertainty = 0
        
        for measure_id in scenario.measures:
            # Assume 1000 m² application for example
            embodied_results = self.calculate_embodied_carbon(measure_id, 1000.0)
            if 'error' not in embodied_results:
                total_embodied_carbon += embodied_results['total_embodied_carbon_kg_co2e']
                embodied_uncertainty += embodied_results['uncertainty'] ** 2
        
        embodied_uncertainty = np.sqrt(embodied_uncertainty)
        
        # Calculate operational carbon savings
        operational_results = self.calculate_operational_savings(scenario)
        operational_carbon_savings = operational_results['lifecycle_carbon_savings_kg_co2e']
        
        # Calculate net carbon impact
        net_carbon_impact = total_embodied_carbon - operational_carbon_savings
        
        # Calculate carbon payback period
        annual_carbon_savings = operational_results['annual_carbon_savings_kg_co2e']
        carbon_payback_years = total_embodied_carbon / annual_carbon_savings if annual_carbon_savings > 0 else float('inf')
        
        # Calculate uncertainty range
        uncertainty_factor = 0.2  # 20% default uncertainty
        uncertainty_range = (
            net_carbon_impact * (1 - uncertainty_factor),
            net_carbon_impact * (1 + uncertainty_factor)
        )
        
        return LCAResults(
            embodied_carbon_kg_co2e=total_embodied_carbon,
            operational_carbon_kg_co2e=-operational_carbon_savings,  # Negative for savings
            total_carbon_kg_co2e=net_carbon_impact,
            carbon_payback_years=carbon_payback_years,
            uncertainty_range=uncertainty_range
        )
    
    def perform_economic_analysis(self, scenario: RetrofitScenario, energy_cost_per_kwh: float = 0.12) -> EconomicResults:
        """Perform comprehensive economic analysis for retrofit scenario"""
        
        # Calculate initial costs
        total_initial_cost = 0
        for measure_id in scenario.measures:
            measure_data = self.get_measure_data(measure_id)
            # Use cost per square meter and scale by building area
            cost_per_m2 = measure_data.get('costs', {}).get('total_cost_per_sqm', 0)
            if cost_per_m2 == 0:
                cost_per_m2 = measure_data.get('costs', {}).get('total_cost', 0) / 1000  # Assume per 1000 m²
            total_initial_cost += cost_per_m2 * scenario.floor_area_m2
        
        # Calculate annual energy savings value
        operational_results = self.calculate_operational_savings(scenario)
        annual_energy_savings_kwh = operational_results['annual_energy_savings']['total_kwh']
        annual_energy_savings_usd = annual_energy_savings_kwh * energy_cost_per_kwh
        
        # Calculate maintenance costs
        total_maintenance_costs = 0
        for measure_id in scenario.measures:
            maintenance_results = self.calculate_maintenance_costs(measure_id, scenario.analysis_period_years)
            if 'error' not in maintenance_results:
                total_maintenance_costs += maintenance_results['escalated_maintenance_costs']
        
        # Calculate net present value
        npv = self._calculate_npv(
            initial_cost=total_initial_cost,
            annual_savings=annual_energy_savings_usd,
            maintenance_costs=total_maintenance_costs,
            analysis_period=scenario.analysis_period_years,
            discount_rate=scenario.discount_rate
        )
        
        # Calculate simple payback period
        payback_period = total_initial_cost / annual_energy_savings_usd if annual_energy_savings_usd > 0 else float('inf')
        
        # Calculate internal rate of return (simplified)
        irr = self._calculate_irr(total_initial_cost, annual_energy_savings_usd, scenario.analysis_period_years)
        
        return EconomicResults(
            initial_cost_usd=total_initial_cost,
            annual_energy_savings_usd=annual_energy_savings_usd,
            maintenance_costs_usd=total_maintenance_costs,
            net_present_value_usd=npv,
            payback_period_years=payback_period,
            internal_rate_of_return=irr
        )
    
    def _calculate_npv(self, initial_cost: float, annual_savings: float, 
                      maintenance_costs: float, analysis_period: int, discount_rate: float) -> float:
        """Calculate net present value"""
        # Present value of annual savings
        pv_savings = sum([annual_savings / ((1 + discount_rate) ** year) for year in range(1, analysis_period + 1)])
        
        # Present value of maintenance costs (distributed over analysis period)
        annual_maintenance = maintenance_costs / analysis_period
        pv_maintenance = sum([annual_maintenance / ((1 + discount_rate) ** year) for year in range(1, analysis_period + 1)])
        
        return pv_savings - initial_cost - pv_maintenance
    
    def _calculate_irr(self, initial_cost: float, annual_savings: float, analysis_period: int) -> float:
        """Calculate internal rate of return (simplified approximation)"""
        # Simplified IRR calculation using approximation
        if annual_savings <= 0:
            return 0.0
        
        # Use iterative approach to find IRR
        for rate in np.arange(0, 1, 0.001):
            npv = sum([annual_savings / ((1 + rate) ** year) for year in range(1, analysis_period + 1)]) - initial_cost
            if npv <= 0:
                return rate
        
        return 1.0  # Return 100% if very high IRR
    
    def optimize_retrofit_portfolio(self, building_type: str, climate_zone: str, 
                                  floor_area_m2: float, budget_limit: float = None) -> Dict:
        """Optimize retrofit measure portfolio for maximum benefit"""
        
        # Get all available measures
        available_measures = []
        for category in self.retrofit_measures.values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        for measure in subcategory:
                            if self._is_measure_applicable(measure, building_type, climate_zone):
                                available_measures.append(measure['measure_id'])
        
        # Evaluate each measure individually
        measure_evaluations = []
        for measure_id in available_measures:
            scenario = RetrofitScenario(
                scenario_id=f"single_{measure_id}",
                building_type=building_type,
                climate_zone=climate_zone,
                floor_area_m2=floor_area_m2,
                measures=[measure_id]
            )
            
            try:
                lca_results = self.perform_lifecycle_assessment(scenario)
                economic_results = self.perform_economic_analysis(scenario)
                
                measure_evaluations.append({
                    'measure_id': measure_id,
                    'carbon_savings_kg_co2e': -lca_results.operational_carbon_kg_co2e,
                    'carbon_payback_years': lca_results.carbon_payback_years,
                    'initial_cost_usd': economic_results.initial_cost_usd,
                    'npv_usd': economic_results.net_present_value_usd,
                    'payback_years': economic_results.payback_period_years,
                    'cost_effectiveness_usd_per_kg_co2e': economic_results.initial_cost_usd / max(-lca_results.operational_carbon_kg_co2e, 1)
                })
            except Exception as e:
                logger.warning(f"Could not evaluate measure {measure_id}: {str(e)}")
        
        # Sort by cost-effectiveness (lowest cost per kg CO2e saved)
        measure_evaluations.sort(key=lambda x: x['cost_effectiveness_usd_per_kg_co2e'])
        
        # Select optimal portfolio within budget
        selected_measures = []
        total_cost = 0
        
        for measure_eval in measure_evaluations:
            if budget_limit is None or (total_cost + measure_eval['initial_cost_usd']) <= budget_limit:
                selected_measures.append(measure_eval)
                total_cost += measure_eval['initial_cost_usd']
        
        return {
            'building_type': building_type,
            'climate_zone': climate_zone,
            'floor_area_m2': floor_area_m2,
            'budget_limit_usd': budget_limit,
            'all_measures_evaluated': measure_evaluations,
            'optimal_portfolio': selected_measures,
            'total_portfolio_cost_usd': total_cost,
            'total_carbon_savings_kg_co2e': sum([m['carbon_savings_kg_co2e'] for m in selected_measures]),
            'portfolio_cost_effectiveness': total_cost / max(sum([m['carbon_savings_kg_co2e'] for m in selected_measures]), 1)
        }
    
    def _is_measure_applicable(self, measure: Dict, building_type: str, climate_zone: str) -> bool:
        """Check if a retrofit measure is applicable to building and climate"""
        applicability = measure.get('applicability', {})
        
        # Check building type compatibility
        building_types = applicability.get('building_types', [])
        if building_types and building_type not in building_types:
            return False
        
        # Check climate zone compatibility
        climate_zones = applicability.get('climate_zones', [])
        if climate_zones and climate_zone not in climate_zones:
            return False
        
        return True
    
    def generate_performance_report(self, scenario: RetrofitScenario) -> Dict:
        """Generate comprehensive performance report for retrofit scenario"""
        
        try:
            # Perform all analyses
            lca_results = self.perform_lifecycle_assessment(scenario)
            economic_results = self.perform_economic_analysis(scenario)
            operational_results = self.calculate_operational_savings(scenario)
            
            # Compile comprehensive report
            report = {
                'scenario_summary': asdict(scenario),
                'lifecycle_assessment': asdict(lca_results),
                'economic_analysis': asdict(economic_results),
                'operational_performance': operational_results,
                'key_metrics': {
                    'carbon_payback_years': lca_results.carbon_payback_years,
                    'financial_payback_years': economic_results.payback_period_years,
                    'net_present_value_usd': economic_results.net_present_value_usd,
                    'total_carbon_impact_kg_co2e': lca_results.total_carbon_kg_co2e,
                    'annual_energy_savings_kwh': operational_results['annual_energy_savings']['total_kwh'],
                    'cost_effectiveness_usd_per_kg_co2e_saved': economic_results.initial_cost_usd / max(-lca_results.operational_carbon_kg_co2e, 1)
                },
                'recommendations': self._generate_recommendations(lca_results, economic_results),
                'report_generated': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, lca_results: LCAResults, economic_results: EconomicResults) -> List[str]:
        """Generate recommendations based on analysis results"""
        recommendations = []
        
        # Carbon performance recommendations
        if lca_results.carbon_payback_years < 5:
            recommendations.append("Excellent carbon performance - measure pays back embodied carbon quickly")
        elif lca_results.carbon_payback_years > 15:
            recommendations.append("Consider alternative measures with better carbon payback")
        
        # Economic performance recommendations
        if economic_results.net_present_value_usd > 0:
            recommendations.append("Economically attractive investment with positive NPV")
        else:
            recommendations.append("Consider incentives or financing to improve economic viability")
        
        if economic_results.payback_period_years < 7:
            recommendations.append("Short payback period makes this an attractive investment")
        
        # Overall recommendations
        if lca_results.total_carbon_kg_co2e < 0:
            recommendations.append("Net carbon negative - excellent environmental performance")
        
        return recommendations

def main():
    """Example usage of the RetrofitDataProcessor"""
    
    # Initialize processor
    processor = RetrofitDataProcessor()
    
    # Create example scenario
    scenario = RetrofitScenario(
        scenario_id="office_retrofit_example",
        building_type="small_office",
        climate_zone="4A",
        floor_area_m2=500,
        measures=["INS_001", "WIN_001", "HVAC_001", "SOLAR_001"],
        analysis_period_years=25,
        discount_rate=0.03
    )
    
    # Generate comprehensive report
    report = processor.generate_performance_report(scenario)
    
    # Print key results
    print("=== RETROFIT ANALYSIS REPORT ===")
    print(f"Scenario: {scenario.scenario_id}")
    print(f"Building: {scenario.building_type} ({scenario.floor_area_m2} m²)")
    print(f"Climate Zone: {scenario.climate_zone}")
    print(f"Measures: {', '.join(scenario.measures)}")
    print("\n=== KEY METRICS ===")
    
    if 'key_metrics' in report:
        metrics = report['key_metrics']
        print(f"Carbon Payback: {metrics['carbon_payback_years']:.1f} years")
        print(f"Financial Payback: {metrics['financial_payback_years']:.1f} years")
        print(f"Net Present Value: ${metrics['net_present_value_usd']:,.0f}")
        print(f"Total Carbon Impact: {metrics['total_carbon_impact_kg_co2e']:,.0f} kg CO2e")
        print(f"Annual Energy Savings: {metrics['annual_energy_savings_kwh']:,.0f} kWh")
        print(f"Cost Effectiveness: ${metrics['cost_effectiveness_usd_per_kg_co2e_saved']:.2f}/kg CO2e saved")
    
    # Demonstrate optimization
    print("\n=== PORTFOLIO OPTIMIZATION ===")
    optimization_results = processor.optimize_retrofit_portfolio(
        building_type="small_office",
        climate_zone="4A",
        floor_area_m2=500,
        budget_limit=50000
    )
    
    print(f"Budget Limit: ${optimization_results['budget_limit_usd']:,}")
    print(f"Optimal Portfolio Cost: ${optimization_results['total_portfolio_cost_usd']:,.0f}")
    print(f"Total Carbon Savings: {optimization_results['total_carbon_savings_kg_co2e']:,.0f} kg CO2e")
    print(f"Portfolio Cost Effectiveness: ${optimization_results['portfolio_cost_effectiveness']:.2f}/kg CO2e")
    
    print("\nSelected Measures:")
    for measure in optimization_results['optimal_portfolio'][:5]:  # Show top 5
        print(f"  {measure['measure_id']}: ${measure['initial_cost_usd']:,.0f}, "
              f"{measure['carbon_savings_kg_co2e']:,.0f} kg CO2e saved")

if __name__ == "__main__":
    main()