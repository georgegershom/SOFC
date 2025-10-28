"""
Petroleum Cities Systems Map - Complete Demonstration
====================================================

This script demonstrates the complete functionality of the petroleum cities
systems analysis framework, including all visualizations and analyses.

Run this script to see the full capabilities of the system.
"""

import matplotlib.pyplot as plt
import numpy as np
from petroleum_cities_systems_map import PetroleumCitiesSystemsMap
from advanced_systems_analysis import AdvancedSystemsAnalysis
import warnings
warnings.filterwarnings('ignore')

def main():
    """Main demonstration function."""
    print("🌍 PETROLEUM CITIES SYSTEMS MAP - COMPLETE DEMONSTRATION")
    print("=" * 70)
    print()
    
    # 1. Create and display the main systems map
    print("📊 Creating Systems Map Visualization...")
    systems_map = PetroleumCitiesSystemsMap("Port Harcourt")
    fig1 = systems_map.create_systems_map()
    fig1.savefig('/workspace/demo_systems_map.png', dpi=300, bbox_inches='tight')
    print("✅ Systems map saved as 'demo_systems_map.png'")
    print()
    
    # 2. Run system dynamics simulation
    print("🔄 Running System Dynamics Simulation...")
    simulation_data = systems_map.simulate_system_dynamics(time_steps=150)
    fig2 = systems_map.plot_simulation_results(simulation_data)
    fig2.savefig('/workspace/demo_simulation.png', dpi=300, bbox_inches='tight')
    print("✅ Simulation results saved as 'demo_simulation.png'")
    print()
    
    # 3. Calculate and display vulnerability metrics
    print("📈 Calculating Vulnerability Metrics...")
    current_vulnerability = systems_map.calculate_vulnerability_index()
    print(f"   Current Composite Vulnerability Index: {current_vulnerability:.3f}")
    
    # 4. Optimize intervention strategy
    print("🎯 Optimizing Intervention Strategy...")
    optimization_results = systems_map.optimize_intervention_strategy()
    print(f"   Optimal interventions:")
    for intervention, strength in optimization_results['optimal_interventions'].items():
        print(f"     {intervention}: +{strength:.3f}")
    print(f"   Vulnerability reduction: {optimization_results['current_vulnerability']:.3f} → {optimization_results['min_vulnerability']:.3f}")
    print()
    
    # 5. Advanced analysis
    print("🔬 Running Advanced Systems Analysis...")
    advanced_analysis = AdvancedSystemsAnalysis("Port Harcourt")
    
    # Monte Carlo simulation
    print("   Running Monte Carlo simulation...")
    mc_results = advanced_analysis.monte_carlo_simulation(n_simulations=200, time_steps=50)
    print(f"   Mean vulnerability: {mc_results['final_vulnerability'].mean():.3f} ± {mc_results['final_vulnerability'].std():.3f}")
    
    # Policy scenario analysis
    print("   Analyzing policy scenarios...")
    policy_results = advanced_analysis.policy_scenario_analysis()
    best_scenario = policy_results.loc[policy_results['improvement'].idxmax()]
    print(f"   Best scenario: {best_scenario['scenario']} (improvement: {best_scenario['improvement_percent']:.1f}%)")
    
    # Network analysis
    print("   Performing network analysis...")
    network_results = advanced_analysis.network_analysis()
    print(f"   Network density: {network_results['network_metrics']['density']:.3f}")
    print(f"   Number of nodes: {network_results['network_metrics']['number_of_nodes']}")
    print(f"   Number of edges: {network_results['network_metrics']['number_of_edges']}")
    print()
    
    # 6. Create 3D vulnerability surface
    print("🌐 Creating 3D Vulnerability Surface...")
    fig3 = advanced_analysis.create_3d_vulnerability_surface()
    fig3.savefig('/workspace/demo_3d_surface.png', dpi=300, bbox_inches='tight')
    print("✅ 3D surface saved as 'demo_3d_surface.png'")
    print()
    
    # 7. Multi-city comparison
    print("🏙️  Multi-City Comparison Analysis...")
    cities = ["Port Harcourt", "Warri", "Bonny"]
    city_results = {}
    
    for city in cities:
        city_map = PetroleumCitiesSystemsMap(city)
        vulnerability = city_map.calculate_vulnerability_index()
        city_results[city] = vulnerability
        print(f"   {city}: CVI = {vulnerability:.3f}")
    
    # Create comparison visualization
    fig4, ax = plt.subplots(figsize=(12, 8))
    cities_list = list(city_results.keys())
    vulnerabilities = list(city_results.values())
    
    bars = ax.bar(cities_list, vulnerabilities, 
                  color=['#E74C3C', '#F39C12', '#8E44AD'], alpha=0.8)
    ax.set_ylabel('Composite Vulnerability Index', fontweight='bold', fontsize=12)
    ax.set_title('Vulnerability Comparison Across Petroleum Cities', 
                fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, vulnerabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig4.savefig('/workspace/demo_city_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ City comparison saved as 'demo_city_comparison.png'")
    print()
    
    # 8. Generate comprehensive report
    print("📋 Generating Comprehensive Report...")
    report = advanced_analysis.create_comprehensive_report()
    with open('/workspace/demo_comprehensive_report.txt', 'w') as f:
        f.write(report)
    print("✅ Comprehensive report saved as 'demo_comprehensive_report.txt'")
    print()
    
    # 9. Display key insights
    print("💡 KEY INSIGHTS FROM THE ANALYSIS")
    print("=" * 50)
    print(f"• Current system vulnerability: {current_vulnerability:.3f}")
    print(f"• Most effective intervention: {best_scenario['scenario']}")
    print(f"• Improvement potential: {best_scenario['improvement_percent']:.1f}%")
    print(f"• System stability (MC std): {mc_results['final_vulnerability'].std():.3f}")
    print(f"• Network complexity: {network_results['network_metrics']['density']:.3f} density")
    print()
    
    # 10. Summary of generated files
    print("📁 GENERATED FILES SUMMARY")
    print("=" * 30)
    files = [
        "demo_systems_map.png - Main systems map visualization",
        "demo_simulation.png - System dynamics simulation results", 
        "demo_3d_surface.png - 3D vulnerability surface",
        "demo_city_comparison.png - Multi-city vulnerability comparison",
        "demo_comprehensive_report.txt - Detailed analysis report"
    ]
    
    for file in files:
        print(f"✅ {file}")
    
    print()
    print("🎉 DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("The petroleum cities systems analysis framework has successfully")
    print("generated comprehensive visualizations and analyses demonstrating")
    print("the three dominant reinforcing feedback loops and their implications")
    print("for building resilience in oil-dependent urban areas.")
    print()
    print("All files are ready for presentation and further analysis!")


if __name__ == "__main__":
    main()