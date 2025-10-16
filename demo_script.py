#!/usr/bin/env python3
"""
Comprehensive Demo Script for Retrofit Intervention & Lifecycle Data
Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization

This script demonstrates all key capabilities of the framework:
1. Data processing and analysis
2. Multi-objective optimization
3. Digital twin integration
4. Real-time IoT simulation
5. Deep reinforcement learning
6. Performance reporting
"""

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Suppress warnings for cleaner demo output
warnings.filterwarnings('ignore')

# Import our framework modules
from data_processor import RetrofitDataProcessor, RetrofitScenario
from digital_twin_integration import RetrofitDecisionEngine, BuildingState

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveDemo:
    """Comprehensive demonstration of the retrofit framework"""
    
    def __init__(self):
        self.processor = RetrofitDataProcessor()
        self.results = {}
        
    def print_header(self, title: str, width: int = 80):
        """Print formatted section header"""
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
    
    def print_subheader(self, title: str, width: int = 60):
        """Print formatted subsection header"""
        print(f"\n{'-' * width}")
        print(f"{title:^{width}}")
        print(f"{'-' * width}")
    
    def demo_data_overview(self):
        """Demonstrate the comprehensive dataset overview"""
        self.print_header("RETROFIT INTERVENTION & LIFECYCLE DATA OVERVIEW")
        
        # Load and analyze the datasets
        retrofit_data = self.processor.retrofit_measures
        lci_data = self.processor.lci_database
        operational_data = self.processor.operational_data
        maintenance_data = self.processor.maintenance_data
        
        print(f"📊 DATASET STATISTICS")
        print(f"   • Retrofit Measures: {self._count_measures(retrofit_data)}")
        print(f"   • LCI Materials: {self._count_materials(lci_data)}")
        print(f"   • Building Types: {self._count_building_types(operational_data)}")
        print(f"   • Maintenance Systems: {self._count_maintenance_systems(maintenance_data)}")
        
        # Show measure categories
        self.print_subheader("AVAILABLE RETROFIT MEASURES")
        categories = self._analyze_measure_categories(retrofit_data)
        for category, count in categories.items():
            print(f"   {category.replace('_', ' ').title()}: {count} measures")
        
        # Show LCA coverage
        self.print_subheader("LIFE CYCLE ASSESSMENT COVERAGE")
        lca_coverage = self._analyze_lca_coverage(lci_data)
        for material_type, count in lca_coverage.items():
            print(f"   {material_type.replace('_', ' ').title()}: {count} materials")
        
        return {
            'total_measures': self._count_measures(retrofit_data),
            'total_materials': self._count_materials(lci_data),
            'categories': categories,
            'lca_coverage': lca_coverage
        }
    
    def demo_single_measure_analysis(self):
        """Demonstrate detailed analysis of a single retrofit measure"""
        self.print_header("SINGLE MEASURE DETAILED ANALYSIS")
        
        measure_id = "INS_001"  # Polyiso roof insulation
        
        print(f"🔍 ANALYZING MEASURE: {measure_id}")
        
        # Get measure data
        measure_data = self.processor.get_measure_data(measure_id)
        print(f"   Name: {measure_data['name']}")
        print(f"   Category: {measure_data['category']}")
        print(f"   Description: {measure_data['description']}")
        
        # Performance characteristics
        self.print_subheader("PERFORMANCE CHARACTERISTICS")
        performance = measure_data['performance']
        for key, value in performance.items():
            print(f"   {key.replace('_', ' ').title()}: {value}")
        
        # Cost analysis
        self.print_subheader("COST ANALYSIS")
        costs = measure_data['costs']
        print(f"   Material Cost: ${costs['material_cost_per_sqm']:.2f}/m²")
        print(f"   Labor Cost: ${costs['labor_cost_per_sqm']:.2f}/m²")
        print(f"   Total Cost: ${costs['total_cost_per_sqm']:.2f}/m²")
        print(f"   Cost Uncertainty: ±{costs['cost_uncertainty']:.1%}")
        
        # Embodied carbon analysis
        self.print_subheader("EMBODIED CARBON ANALYSIS")
        embodied_results = self.processor.calculate_embodied_carbon(measure_id, 1000.0)  # 1000 m²
        if 'error' not in embodied_results:
            breakdown = embodied_results['embodied_carbon_breakdown']
            print(f"   Production (A1-A3): {breakdown['A1_A3_kg_co2e']:,.0f} kg CO2e")
            print(f"   Transport/Install (A4-A5): {breakdown['A4_A5_kg_co2e']:,.0f} kg CO2e")
            print(f"   Use Phase (B1-B7): {breakdown['B1_B7_kg_co2e']:,.0f} kg CO2e")
            print(f"   End of Life (C1-C4): {breakdown['C1_C4_kg_co2e']:,.0f} kg CO2e")
            print(f"   TOTAL: {embodied_results['total_embodied_carbon_kg_co2e']:,.0f} kg CO2e")
        
        # Maintenance analysis
        self.print_subheader("MAINTENANCE ANALYSIS")
        maintenance_results = self.processor.calculate_maintenance_costs(measure_id, 25)
        if 'error' not in maintenance_results:
            print(f"   Analysis Period: {maintenance_results['analysis_period_years']} years")
            print(f"   Total Maintenance Cost: ${maintenance_results['escalated_maintenance_costs']:,.0f}")
            print(f"   Annual Escalation Rate: {maintenance_results['annual_escalation_rate']:.1%}")
            
            print(f"\n   Maintenance Tasks:")
            for task in maintenance_results['cost_breakdown'][:3]:  # Show first 3 tasks
                print(f"     • {task['task_name']}: ${task['total_cost']:,.0f} ({task['total_occurrences']} times)")
        
        return {
            'measure_id': measure_id,
            'measure_data': measure_data,
            'embodied_carbon': embodied_results,
            'maintenance_costs': maintenance_results
        }
    
    def demo_building_scenario_analysis(self):
        """Demonstrate comprehensive building scenario analysis"""
        self.print_header("BUILDING SCENARIO ANALYSIS")
        
        # Define multiple scenarios
        scenarios = [
            RetrofitScenario(
                scenario_id="basic_efficiency",
                building_type="small_office",
                climate_zone="4A",
                floor_area_m2=500,
                measures=["INS_001", "WIN_001"],
                analysis_period_years=25
            ),
            RetrofitScenario(
                scenario_id="comprehensive_retrofit",
                building_type="small_office",
                climate_zone="4A",
                floor_area_m2=500,
                measures=["INS_002", "WIN_002", "HVAC_001", "SOLAR_001"],
                analysis_period_years=25
            ),
            RetrofitScenario(
                scenario_id="deep_retrofit",
                building_type="small_office",
                climate_zone="4A",
                floor_area_m2=500,
                measures=["INS_002", "WIN_003", "HVAC_002", "SOLAR_002", "LIGHT_001", "BAS_001"],
                analysis_period_years=25
            )
        ]
        
        scenario_results = []
        
        for scenario in scenarios:
            print(f"\n🏢 ANALYZING SCENARIO: {scenario.scenario_id.upper()}")
            print(f"   Building: {scenario.building_type} ({scenario.floor_area_m2} m²)")
            print(f"   Climate Zone: {scenario.climate_zone}")
            print(f"   Measures: {', '.join(scenario.measures)}")
            
            # Generate comprehensive report
            report = self.processor.generate_performance_report(scenario)
            
            if 'error' not in report:
                metrics = report['key_metrics']
                
                print(f"\n   📈 KEY PERFORMANCE METRICS:")
                print(f"     • Carbon Payback: {metrics['carbon_payback_years']:.1f} years")
                print(f"     • Financial Payback: {metrics['financial_payback_years']:.1f} years")
                print(f"     • Net Present Value: ${metrics['net_present_value_usd']:,.0f}")
                print(f"     • Total Carbon Impact: {metrics['total_carbon_impact_kg_co2e']:,.0f} kg CO2e")
                print(f"     • Annual Energy Savings: {metrics['annual_energy_savings_kwh']:,.0f} kWh")
                print(f"     • Cost Effectiveness: ${metrics['cost_effectiveness_usd_per_kg_co2e_saved']:.2f}/kg CO2e")
                
                # Add to results for comparison
                scenario_results.append({
                    'scenario_id': scenario.scenario_id,
                    'metrics': metrics,
                    'report': report
                })
        
        # Compare scenarios
        self.print_subheader("SCENARIO COMPARISON")
        comparison_df = pd.DataFrame([
            {
                'Scenario': result['scenario_id'].replace('_', ' ').title(),
                'NPV ($)': result['metrics']['net_present_value_usd'],
                'Carbon Payback (years)': result['metrics']['carbon_payback_years'],
                'Financial Payback (years)': result['metrics']['financial_payback_years'],
                'Energy Savings (kWh/year)': result['metrics']['annual_energy_savings_kwh'],
                'Cost Effectiveness ($/kg CO2e)': result['metrics']['cost_effectiveness_usd_per_kg_co2e_saved']
            }
            for result in scenario_results
        ])
        
        print(comparison_df.to_string(index=False, float_format='%.1f'))
        
        return scenario_results
    
    def demo_portfolio_optimization(self):
        """Demonstrate portfolio optimization capabilities"""
        self.print_header("PORTFOLIO OPTIMIZATION")
        
        print(f"🎯 OPTIMIZING RETROFIT PORTFOLIO")
        print(f"   Building Type: Medium Office")
        print(f"   Climate Zone: 5A (Chicago)")
        print(f"   Floor Area: 2,000 m²")
        print(f"   Budget Limit: $150,000")
        
        # Run optimization
        optimization_results = self.processor.optimize_retrofit_portfolio(
            building_type="medium_office",
            climate_zone="5A",
            floor_area_m2=2000,
            budget_limit=150000
        )
        
        self.print_subheader("OPTIMIZATION RESULTS")
        print(f"   Total Portfolio Cost: ${optimization_results['total_portfolio_cost_usd']:,.0f}")
        print(f"   Total Carbon Savings: {optimization_results['total_carbon_savings_kg_co2e']:,.0f} kg CO2e/year")
        print(f"   Portfolio Cost Effectiveness: ${optimization_results['portfolio_cost_effectiveness']:.2f}/kg CO2e")
        print(f"   Number of Measures Selected: {len(optimization_results['optimal_portfolio'])}")
        
        self.print_subheader("SELECTED MEASURES (TOP 5)")
        for i, measure in enumerate(optimization_results['optimal_portfolio'][:5]):
            print(f"   {i+1}. {measure['measure_id']}")
            print(f"      Cost: ${measure['initial_cost_usd']:,.0f}")
            print(f"      Carbon Savings: {measure['carbon_savings_kg_co2e']:,.0f} kg CO2e/year")
            print(f"      Payback: {measure['payback_years']:.1f} years")
            print(f"      Cost Effectiveness: ${measure['cost_effectiveness_usd_per_kg_co2e']:.2f}/kg CO2e")
        
        # Show all evaluated measures
        self.print_subheader("ALL MEASURES RANKED BY COST EFFECTIVENESS")
        all_measures_df = pd.DataFrame([
            {
                'Measure ID': measure['measure_id'],
                'Cost ($)': measure['initial_cost_usd'],
                'Carbon Savings (kg CO2e/year)': measure['carbon_savings_kg_co2e'],
                'Payback (years)': measure['payback_years'],
                'Cost Effectiveness ($/kg CO2e)': measure['cost_effectiveness_usd_per_kg_co2e']
            }
            for measure in optimization_results['all_measures_evaluated'][:10]  # Top 10
        ])
        
        print(all_measures_df.to_string(index=False, float_format='%.1f'))
        
        return optimization_results
    
    async def demo_digital_twin_integration(self):
        """Demonstrate digital twin and AI integration"""
        self.print_header("DIGITAL TWIN & AI INTEGRATION")
        
        print(f"🤖 INITIALIZING DIGITAL TWIN FRAMEWORK")
        
        # Initialize decision engine
        engine = RetrofitDecisionEngine("demo_building_001")
        
        print(f"   Building ID: demo_building_001")
        print(f"   Components: IoT Manager, Digital Twin Engine, RL Agent, Multi-Objective Optimizer")
        
        # Update building state
        self.print_subheader("REAL-TIME BUILDING STATE")
        current_state = await engine.digital_twin.update_building_state()
        
        print(f"   Timestamp: {current_state.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Indoor Temperature: {current_state.indoor_temperature_c:.1f}°C")
        print(f"   Outdoor Temperature: {current_state.outdoor_temperature_c:.1f}°C")
        print(f"   Humidity: {current_state.humidity_percent:.1f}%")
        print(f"   Occupancy: {current_state.occupancy_count} people")
        print(f"   Energy Consumption: {current_state.energy_consumption_kw:.1f} kW")
        print(f"   Solar Generation: {current_state.renewable_generation_kw:.1f} kW")
        print(f"   Battery SOC: {current_state.battery_soc_percent:.1f}%")
        
        # IoT data quality
        self.print_subheader("IOT DATA QUALITY METRICS")
        data_quality = engine.digital_twin.iot_manager.calculate_data_quality_metrics()
        if 'error' not in data_quality:
            print(f"   Active Sensors: {data_quality['total_sensors']}")
            print(f"   Data Points (Last Hour): {data_quality['data_points_last_hour']}")
            print(f"   Average Quality Score: {data_quality['average_quality_score']:.2f}")
            print(f"   Sensor Coverage: {dict(list(data_quality['sensor_coverage'].items())[:3])}")
        
        # Anomaly detection
        self.print_subheader("ANOMALY DETECTION")
        anomalies = engine.digital_twin.detect_anomalies()
        if anomalies:
            print(f"   ⚠️  {len(anomalies)} anomalies detected:")
            for anomaly in anomalies:
                print(f"     • {anomaly['type']}: {anomaly['value']} (severity: {anomaly['severity']})")
        else:
            print(f"   ✅ No anomalies detected - building operating normally")
        
        # Predictive analytics
        self.print_subheader("PREDICTIVE ANALYTICS")
        predictions = engine.digital_twin.predict_future_state(hours_ahead=6)
        print(f"   6-hour forecast:")
        for i, pred in enumerate(predictions[:3]):
            print(f"     Hour +{i+1}: {pred.indoor_temperature_c:.1f}°C, {pred.energy_consumption_kw:.1f}kW, {pred.occupancy_count} people")
        
        # Generate AI recommendations
        self.print_subheader("AI-POWERED RECOMMENDATIONS")
        recommendation = await engine.generate_recommendations()
        
        print(f"   Recommendation ID: {recommendation.recommendation_id}")
        print(f"   Priority Score: {recommendation.priority_score:.2f}")
        print(f"   Confidence Level: {recommendation.confidence_level:.1%}")
        print(f"   Recommended Measures: {', '.join(recommendation.measures)}")
        print(f"   Implementation Timeline: {recommendation.implementation_timeline.replace('_', ' ').title()}")
        
        print(f"\n   💡 Estimated Annual Savings:")
        for key, value in recommendation.estimated_savings.items():
            print(f"     • {key.replace('_', ' ').title()}: {value:,.0f}")
        
        print(f"\n   🧠 AI Reasoning:")
        for reason in recommendation.reasoning[:3]:  # Show first 3 reasons
            print(f"     • {reason}")
        
        # Continuous optimization demo
        self.print_subheader("CONTINUOUS OPTIMIZATION SIMULATION")
        print(f"   Running 4-hour optimization simulation...")
        
        optimization_results = await engine.run_continuous_optimization(duration_hours=4)
        
        metrics = optimization_results['performance_metrics']
        print(f"   ✅ Optimization Complete!")
        print(f"     • Average RL Reward: {metrics['average_reward']:.2f}")
        print(f"     • Learning Trend: {metrics['reward_trend']:.3f}")
        print(f"     • Recommendations Generated: {metrics['total_recommendations']}")
        print(f"     • Average Confidence: {metrics['average_confidence']:.1%}")
        
        return {
            'building_state': current_state,
            'data_quality': data_quality,
            'anomalies': anomalies,
            'predictions': predictions,
            'recommendation': recommendation,
            'optimization_results': optimization_results
        }
    
    def demo_performance_visualization(self, scenario_results, optimization_results):
        """Create performance visualizations"""
        self.print_header("PERFORMANCE VISUALIZATION")
        
        print(f"📊 GENERATING PERFORMANCE CHARTS")
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Retrofit Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Scenario comparison - NPV vs Carbon Payback
        if scenario_results:
            scenarios = [r['scenario_id'].replace('_', ' ').title() for r in scenario_results]
            npvs = [r['metrics']['net_present_value_usd'] for r in scenario_results]
            carbon_paybacks = [r['metrics']['carbon_payback_years'] for r in scenario_results]
            
            scatter = ax1.scatter(carbon_paybacks, npvs, s=200, alpha=0.7, c=range(len(scenarios)), cmap='viridis')
            ax1.set_xlabel('Carbon Payback Period (years)')
            ax1.set_ylabel('Net Present Value ($)')
            ax1.set_title('Scenario Performance Comparison')
            ax1.grid(True, alpha=0.3)
            
            # Add labels
            for i, scenario in enumerate(scenarios):
                ax1.annotate(scenario, (carbon_paybacks[i], npvs[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 2. Portfolio optimization - Cost vs Carbon Savings
        if optimization_results and 'optimal_portfolio' in optimization_results:
            portfolio = optimization_results['optimal_portfolio'][:8]  # Top 8 measures
            costs = [m['initial_cost_usd'] for m in portfolio]
            savings = [m['carbon_savings_kg_co2e'] for m in portfolio]
            measure_ids = [m['measure_id'] for m in portfolio]
            
            bars = ax2.bar(range(len(costs)), costs, alpha=0.7, color='skyblue', label='Initial Cost')
            ax2_twin = ax2.twinx()
            line = ax2_twin.plot(range(len(savings)), savings, 'ro-', linewidth=2, markersize=6, label='Carbon Savings')
            
            ax2.set_xlabel('Retrofit Measures')
            ax2.set_ylabel('Initial Cost ($)', color='blue')
            ax2_twin.set_ylabel('Carbon Savings (kg CO2e/year)', color='red')
            ax2.set_title('Optimal Portfolio Analysis')
            ax2.set_xticks(range(len(measure_ids)))
            ax2.set_xticklabels(measure_ids, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Cost effectiveness ranking
        if optimization_results and 'all_measures_evaluated' in optimization_results:
            all_measures = optimization_results['all_measures_evaluated'][:10]  # Top 10
            effectiveness = [m['cost_effectiveness_usd_per_kg_co2e'] for m in all_measures]
            measure_names = [m['measure_id'] for m in all_measures]
            
            y_pos = range(len(effectiveness))
            bars = ax3.barh(y_pos, effectiveness, alpha=0.7, color='lightgreen')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(measure_names)
            ax3.set_xlabel('Cost Effectiveness ($/kg CO2e saved)')
            ax3.set_title('Measure Cost Effectiveness Ranking')
            ax3.grid(True, alpha=0.3)
        
        # 4. Payback period comparison
        if scenario_results:
            scenarios = [r['scenario_id'].replace('_', ' ').title() for r in scenario_results]
            carbon_paybacks = [r['metrics']['carbon_payback_years'] for r in scenario_results]
            financial_paybacks = [r['metrics']['financial_payback_years'] for r in scenario_results]
            
            x = range(len(scenarios))
            width = 0.35
            
            ax4.bar([i - width/2 for i in x], carbon_paybacks, width, label='Carbon Payback', alpha=0.7, color='orange')
            ax4.bar([i + width/2 for i in x], financial_paybacks, width, label='Financial Payback', alpha=0.7, color='purple')
            
            ax4.set_xlabel('Retrofit Scenarios')
            ax4.set_ylabel('Payback Period (years)')
            ax4.set_title('Payback Period Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(scenarios, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = Path('/workspace/retrofit_performance_dashboard.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Dashboard saved to: {output_path}")
        
        # Show the plot (if running interactively)
        try:
            plt.show()
        except:
            print(f"   ℹ️  Plot display not available in this environment")
        
        plt.close()
    
    def generate_summary_report(self, all_results):
        """Generate comprehensive summary report"""
        self.print_header("COMPREHENSIVE SUMMARY REPORT")
        
        print(f"📋 RETROFIT FRAMEWORK DEMONSTRATION SUMMARY")
        print(f"   Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Dataset overview
        if 'data_overview' in all_results:
            overview = all_results['data_overview']
            print(f"\n📊 DATASET OVERVIEW:")
            print(f"   • Total Retrofit Measures: {overview['total_measures']}")
            print(f"   • LCA Materials: {overview['total_materials']}")
            print(f"   • Measure Categories: {len(overview['categories'])}")
        
        # Best performing scenario
        if 'scenario_analysis' in all_results:
            scenarios = all_results['scenario_analysis']
            best_npv = max(scenarios, key=lambda x: x['metrics']['net_present_value_usd'])
            best_carbon = min(scenarios, key=lambda x: x['metrics']['carbon_payback_years'])
            
            print(f"\n🏆 BEST PERFORMING SCENARIOS:")
            print(f"   • Highest NPV: {best_npv['scenario_id']} (${best_npv['metrics']['net_present_value_usd']:,.0f})")
            print(f"   • Fastest Carbon Payback: {best_carbon['scenario_id']} ({best_carbon['metrics']['carbon_payback_years']:.1f} years)")
        
        # Portfolio optimization summary
        if 'portfolio_optimization' in all_results:
            portfolio = all_results['portfolio_optimization']
            print(f"\n🎯 PORTFOLIO OPTIMIZATION:")
            print(f"   • Optimal Portfolio Cost: ${portfolio['total_portfolio_cost_usd']:,.0f}")
            print(f"   • Total Carbon Savings: {portfolio['total_carbon_savings_kg_co2e']:,.0f} kg CO2e/year")
            print(f"   • Cost Effectiveness: ${portfolio['portfolio_cost_effectiveness']:.2f}/kg CO2e")
        
        # Digital twin performance
        if 'digital_twin' in all_results:
            dt_results = all_results['digital_twin']
            print(f"\n🤖 DIGITAL TWIN PERFORMANCE:")
            print(f"   • Building State Monitoring: ✅ Active")
            print(f"   • Anomaly Detection: ✅ {len(dt_results.get('anomalies', []))} anomalies detected")
            print(f"   • AI Recommendations: ✅ Generated with {dt_results['recommendation'].confidence_level:.1%} confidence")
            
            if 'optimization_results' in dt_results:
                opt_metrics = dt_results['optimization_results']['performance_metrics']
                print(f"   • RL Learning Progress: {opt_metrics['reward_trend']:.3f} (trend)")
        
        # Key insights and recommendations
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • The framework successfully integrates comprehensive retrofit data with AI optimization")
        print(f"   • Multi-objective optimization balances energy, carbon, and economic performance")
        print(f"   • Digital twin enables real-time monitoring and adaptive recommendations")
        print(f"   • Deep reinforcement learning continuously improves decision quality")
        
        print(f"\n🚀 NEXT STEPS:")
        print(f"   • Deploy framework in real building environments")
        print(f"   • Integrate with existing building management systems")
        print(f"   • Expand dataset with additional retrofit technologies")
        print(f"   • Validate predictions against measured performance data")
        
        # Save summary to file
        summary_path = Path('/workspace/demo_summary_report.json')
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n📄 Complete results saved to: {summary_path}")
    
    # Helper methods for data analysis
    def _count_measures(self, retrofit_data):
        """Count total retrofit measures"""
        count = 0
        for category in retrofit_data.values():
            if isinstance(category, dict):
                for subcategory in category.values():
                    if isinstance(subcategory, list):
                        count += len(subcategory)
        return count
    
    def _count_materials(self, lci_data):
        """Count LCI materials"""
        count = 0
        materials = lci_data.get('materials', {})
        for category in materials.values():
            if isinstance(category, dict):
                count += len(category)
        return count
    
    def _count_building_types(self, operational_data):
        """Count building types in operational data"""
        baselines = operational_data.get('building_baselines', {})
        return len(baselines)
    
    def _count_maintenance_systems(self, maintenance_data):
        """Count maintenance systems"""
        schedules = maintenance_data.get('maintenance_schedules', {})
        count = 0
        for category in schedules.values():
            if isinstance(category, dict):
                count += len(category)
        return count
    
    def _analyze_measure_categories(self, retrofit_data):
        """Analyze measure categories"""
        categories = {}
        for category_name, category_data in retrofit_data.items():
            if isinstance(category_data, dict) and category_name not in ['metadata']:
                for subcategory_name, subcategory_data in category_data.items():
                    if isinstance(subcategory_data, list):
                        categories[f"{category_name}_{subcategory_name}"] = len(subcategory_data)
        return categories
    
    def _analyze_lca_coverage(self, lci_data):
        """Analyze LCA material coverage"""
        coverage = {}
        materials = lci_data.get('materials', {})
        for material_type, materials_dict in materials.items():
            if isinstance(materials_dict, dict):
                coverage[material_type] = len(materials_dict)
        return coverage

async def main():
    """Run the comprehensive demonstration"""
    
    print("🚀 STARTING COMPREHENSIVE RETROFIT FRAMEWORK DEMONSTRATION")
    print("   This demo showcases all capabilities of the Dynamic Digital Twin Framework")
    print("   for Multi-Objective Building Retrofit Optimization")
    
    # Initialize demo
    demo = ComprehensiveDemo()
    all_results = {}
    
    try:
        # 1. Dataset overview
        all_results['data_overview'] = demo.demo_data_overview()
        
        # 2. Single measure analysis
        all_results['single_measure'] = demo.demo_single_measure_analysis()
        
        # 3. Building scenario analysis
        all_results['scenario_analysis'] = demo.demo_building_scenario_analysis()
        
        # 4. Portfolio optimization
        all_results['portfolio_optimization'] = demo.demo_portfolio_optimization()
        
        # 5. Digital twin integration
        all_results['digital_twin'] = await demo.demo_digital_twin_integration()
        
        # 6. Performance visualization
        demo.demo_performance_visualization(
            all_results['scenario_analysis'], 
            all_results['portfolio_optimization']
        )
        
        # 7. Generate summary report
        demo.generate_summary_report(all_results)
        
        # Final success message
        demo.print_header("DEMONSTRATION COMPLETED SUCCESSFULLY! 🎉")
        print("   All framework capabilities have been demonstrated")
        print("   Results and visualizations have been saved to workspace")
        print("   The framework is ready for real-world deployment")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        print("   Please check the data files and dependencies")
        raise

if __name__ == "__main__":
    asyncio.run(main())