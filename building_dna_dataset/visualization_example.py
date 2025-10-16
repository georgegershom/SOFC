#!/usr/bin/env python3
"""
Building DNA Dataset Visualization Examples
============================================

This script demonstrates how to load and visualize the Building DNA dataset
for digital twin and retrofit optimization applications.

Requirements:
    pip install pandas matplotlib seaborn plotly numpy

Author: Building Digital Twin Research Team
Version: 2024.1
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Dataset root directory
DATASET_ROOT = Path(__file__).parent


def load_json(file_path):
    """Load JSON file from dataset"""
    with open(DATASET_ROOT / file_path, 'r') as f:
        return json.load(f)


def visualize_envelope_performance():
    """Visualize building envelope thermal performance by orientation"""
    
    # Load window data
    windows = load_json('construction_materials/windows_doors.json')
    
    # Extract window performance by orientation
    inventory = windows['window_inventory_by_orientation']
    
    orientations = list(inventory.keys())
    areas = [inventory[o]['area_m2'] for o in orientations]
    u_values = [inventory[o]['average_u_value_w_m2k'] for o in orientations]
    shgcs = [inventory[o]['average_shgc'] for o in orientations]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Window area by orientation
    axes[0].bar(orientations, areas, color='steelblue')
    axes[0].set_title('Window Area by Orientation', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Area (m²)')
    axes[0].set_xlabel('Orientation')
    
    # U-values by orientation
    colors = ['red' if u > 3.0 else 'green' for u in u_values]
    axes[1].bar(orientations, u_values, color=colors)
    axes[1].axhline(y=1.8, color='orange', linestyle='--', label='ASHRAE 90.1 Limit')
    axes[1].set_title('Window U-Values by Orientation', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('U-Value (W/m²K)')
    axes[1].set_xlabel('Orientation')
    axes[1].legend()
    
    # SHGC by orientation
    axes[2].bar(orientations, shgcs, color='darkorange')
    axes[2].set_title('Solar Heat Gain Coefficient', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('SHGC')
    axes[2].set_xlabel('Orientation')
    axes[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('envelope_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: envelope_performance.png")


def visualize_hvac_performance():
    """Visualize HVAC system performance over time"""
    
    # Load HVAC historical data
    hvac_df = pd.read_csv(DATASET_ROOT / 'systems/hvac/hvac_historical_performance.csv')
    hvac_df['date'] = pd.to_datetime(hvac_df['date'])
    
    # Calculate monthly efficiency
    hvac_df['heating_eff'] = np.where(
        hvac_df['gas_consumption_m3'] > 0,
        hvac_df['heating_energy_kwh'] / (hvac_df['gas_consumption_m3'] * 10.55),
        np.nan
    )
    hvac_df['cooling_eer'] = np.where(
        hvac_df['cooling_energy_kwh'] > 0,
        hvac_df['cooling_energy_kwh'] / hvac_df['cooling_runtime_hours'],
        np.nan
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Energy consumption by RTU
    for rtu_id in hvac_df['rtu_id'].unique():
        rtu_data = hvac_df[hvac_df['rtu_id'] == rtu_id]
        axes[0, 0].plot(rtu_data['date'], 
                       rtu_data['heating_energy_kwh'] + rtu_data['cooling_energy_kwh'],
                       marker='o', label=rtu_id, linewidth=2)
    axes[0, 0].set_title('Total HVAC Energy Consumption by RTU', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Energy (kWh)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Heating efficiency over time
    for rtu_id in hvac_df['rtu_id'].unique():
        rtu_data = hvac_df[hvac_df['rtu_id'] == rtu_id]
        axes[0, 1].plot(rtu_data['date'], rtu_data['heating_eff'], 
                       marker='s', label=rtu_id, linewidth=2)
    axes[0, 1].axhline(y=0.80, color='green', linestyle='--', label='Rated Efficiency')
    axes[0, 1].set_title('Heating Efficiency Degradation', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Efficiency (heating kWh / gas kWh)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Filter pressure drop
    for rtu_id in hvac_df['rtu_id'].unique():
        rtu_data = hvac_df[hvac_df['rtu_id'] == rtu_id]
        axes[1, 0].plot(rtu_data['date'], rtu_data['filter_pressure_drop_pa'],
                       marker='^', label=rtu_id, linewidth=2)
    axes[1, 0].axhline(y=250, color='red', linestyle='--', label='Replacement Threshold')
    axes[1, 0].set_title('Filter Pressure Drop Over Time', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Pressure Drop (Pa)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Runtime hours comparison
    monthly_runtime = hvac_df.groupby('rtu_id').agg({
        'heating_runtime_hours': 'sum',
        'cooling_runtime_hours': 'sum'
    })
    x = np.arange(len(monthly_runtime.index))
    width = 0.35
    axes[1, 1].bar(x - width/2, monthly_runtime['heating_runtime_hours'], 
                  width, label='Heating', color='orangered')
    axes[1, 1].bar(x + width/2, monthly_runtime['cooling_runtime_hours'], 
                  width, label='Cooling', color='steelblue')
    axes[1, 1].set_title('Annual Runtime Hours by RTU', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Runtime (hours)')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(monthly_runtime.index)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('hvac_performance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: hvac_performance.png")


def visualize_energy_breakdown():
    """Visualize building energy consumption breakdown"""
    
    # Load building metadata
    building = load_json('metadata/building_info.json')
    
    # Energy breakdown
    energy_data = {
        'Heating': building['energy_summary']['annual_heating_load_kwh'],
        'Cooling': building['energy_summary']['annual_cooling_load_kwh'],
        'Lighting': building['energy_summary']['annual_lighting_load_kwh'],
        'Plug Loads': building['energy_summary']['annual_plug_loads_kwh']
    }
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Pie chart
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    explode = (0.05, 0.05, 0.05, 0.05)
    axes[0].pie(energy_data.values(), labels=energy_data.keys(), autopct='%1.1f%%',
               colors=colors, explode=explode, startangle=90, textprops={'fontsize': 12})
    axes[0].set_title('Annual Energy Consumption Breakdown', fontsize=16, fontweight='bold')
    
    # Bar chart with comparison
    end_uses = list(energy_data.keys())
    current = list(energy_data.values())
    
    # Calculate retrofit potential (example reductions)
    retrofit_savings = {
        'Heating': 0.28,  # 28% reduction
        'Cooling': 0.18,  # 18% reduction
        'Lighting': 0.72,  # 72% reduction
        'Plug Loads': 0.15  # 15% reduction
    }
    post_retrofit = [current[i] * (1 - retrofit_savings[end_uses[i]]) 
                     for i in range(len(end_uses))]
    
    x = np.arange(len(end_uses))
    width = 0.35
    
    bars1 = axes[1].bar(x - width/2, current, width, label='Current', 
                       color='coral', edgecolor='black', linewidth=1.5)
    bars2 = axes[1].bar(x + width/2, post_retrofit, width, label='Post-Retrofit',
                       color='lightgreen', edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height/1000)}k', ha='center', va='bottom', fontsize=10)
    
    axes[1].set_title('Energy Consumption: Current vs. Post-Retrofit', 
                     fontsize=16, fontweight='bold')
    axes[1].set_ylabel('Energy Consumption (kWh/year)', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(end_uses)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('energy_breakdown.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: energy_breakdown.png")


def visualize_retrofit_scenarios():
    """Visualize retrofit investment vs. savings"""
    
    # Load renewable energy data
    renewable = load_json('systems/renewable/renewable_energy_systems.json')
    
    # Compile retrofit scenarios
    scenarios = []
    
    # Windows (from windows_doors.json)
    windows = load_json('construction_materials/windows_doors.json')
    scenarios.append({
        'name': 'Window\nReplacement',
        'cost': windows['retrofit_recommendations']['estimated_cost_usd'],
        'savings': windows['retrofit_recommendations']['estimated_energy_savings_kwh_year'] * 0.20,  # $0.20/kWh
        'payback': windows['retrofit_recommendations']['simple_payback_years']
    })
    
    # HVAC (from hvac_system_specifications.json)
    hvac = load_json('systems/hvac/hvac_system_specifications.json')
    scenarios.append({
        'name': 'HVAC\nReplacement',
        'cost': hvac['retrofit_opportunities']['priority_1']['estimated_cost_usd'],
        'savings': hvac['retrofit_opportunities']['priority_1']['estimated_savings_kwh_year'] * 0.20,
        'payback': hvac['retrofit_opportunities']['priority_1']['simple_payback_years']
    })
    
    # Lighting
    lighting = load_json('systems/lighting/lighting_system_inventory.json')
    scenarios.append({
        'name': 'LED\nRetrofit',
        'cost': lighting['retrofit_opportunities']['priority_1']['estimated_cost_usd'],
        'savings': lighting['retrofit_opportunities']['priority_1']['annual_savings_kwh'] * 0.20,
        'payback': lighting['retrofit_opportunities']['priority_1']['simple_payback_years']
    })
    
    # Solar PV
    scenarios.append({
        'name': 'Solar PV\n(161.5 kW)',
        'cost': renewable['solar_pv_potential']['financial_analysis']['net_system_cost_usd'],
        'savings': renewable['solar_pv_potential']['financial_analysis']['annual_energy_savings_usd'],
        'payback': renewable['solar_pv_potential']['financial_analysis']['simple_payback_years']
    })
    
    # Air sealing
    air_tight = load_json('environmental/air_tightness_data.json')
    scenarios.append({
        'name': 'Air Sealing\nPackage',
        'cost': air_tight['recommended_air_sealing_measures']['combined_potential']['total_cost_usd'],
        'savings': air_tight['recommended_air_sealing_measures']['combined_potential']['total_annual_savings_kwh'] * 0.12,
        'payback': air_tight['recommended_air_sealing_measures']['combined_potential']['simple_payback_years']
    })
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: Cost vs. Annual Savings
    names = [s['name'] for s in scenarios]
    costs = [s['cost']/1000 for s in scenarios]  # Convert to thousands
    savings = [s['savings']/1000 for s in scenarios]  # Convert to thousands
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, costs, width, label='Investment Cost',
                       color='crimson', edgecolor='black', linewidth=1.5)
    bars2 = axes[0].bar(x + width/2, savings, width, label='Annual Savings',
                       color='seagreen', edgecolor='black', linewidth=1.5)
    
    axes[0].set_title('Retrofit Measures: Investment vs. Annual Savings', 
                     fontsize=16, fontweight='bold')
    axes[0].set_ylabel('Amount ($1,000s)', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Simple Payback Period
    paybacks = [s['payback'] for s in scenarios]
    colors_payback = ['green' if p < 10 else 'orange' if p < 20 else 'red' 
                      for p in paybacks]
    
    bars = axes[1].barh(names, paybacks, color=colors_payback, 
                        edgecolor='black', linewidth=1.5)
    axes[1].axvline(x=10, color='orange', linestyle='--', linewidth=2, 
                   label='10-year threshold')
    axes[1].set_title('Simple Payback Period by Retrofit Measure', 
                     fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Payback Period (years)', fontsize=12)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        axes[1].text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{paybacks[i]:.1f} yr', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('retrofit_scenarios.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: retrofit_scenarios.png")


def create_summary_report():
    """Generate a summary report of key building metrics"""
    
    print("\n" + "="*80)
    print("BUILDING DNA DATASET - SUMMARY REPORT")
    print("="*80 + "\n")
    
    # Load data
    building = load_json('metadata/building_info.json')
    walls = load_json('construction_materials/wall_assemblies.json')
    windows = load_json('construction_materials/windows_doors.json')
    hvac = load_json('systems/hvac/hvac_system_specifications.json')
    
    print(f"Building: {building['building_name']}")
    print(f"Location: {building['location']['address']}")
    print(f"Year Built: {building['general_characteristics']['year_built']}")
    print(f"Gross Floor Area: {building['general_characteristics']['gross_floor_area_m2']:,.0f} m²")
    print(f"\n{'CURRENT PERFORMANCE':^80}")
    print("-" * 80)
    print(f"  Annual Energy Consumption: {building['energy_summary']['annual_energy_consumption_kwh']:,.0f} kWh")
    print(f"  Energy Use Intensity (EUI): {building['energy_summary']['eui_kwh_m2_year']:.1f} kWh/m²/year")
    print(f"  Peak Heating Demand: {building['energy_summary']['peak_heating_demand_kw']:.0f} kW")
    print(f"  Peak Cooling Demand: {building['energy_summary']['peak_cooling_demand_kw']:.0f} kW")
    
    print(f"\n{'ENVELOPE PERFORMANCE':^80}")
    print("-" * 80)
    print(f"  Average Wall U-value: {building['envelope_summary']['average_wall_u_value_w_m2k']:.2f} W/m²K")
    print(f"  Average Roof U-value: {building['envelope_summary']['average_roof_u_value_w_m2k']:.2f} W/m²K")
    print(f"  Average Window U-value: {building['envelope_summary']['average_window_u_value_w_m2k']:.2f} W/m²K")
    print(f"  Window-to-Wall Ratio: {building['envelope_summary']['window_to_wall_ratio']:.2f}")
    print(f"  Air Tightness (ACH50): {building['envelope_summary']['estimated_infiltration_ach50']:.2f}")
    
    print(f"\n{'RETROFIT POTENTIAL':^80}")
    print("-" * 80)
    print(f"  EUI Reduction Potential: {building['retrofit_potential']['estimated_eui_reduction_potential_percent']:.0f}%")
    print(f"  Envelope Retrofit Feasibility: {building['retrofit_potential']['envelope_retrofit_feasibility']}")
    print(f"  HVAC Retrofit Feasibility: {building['retrofit_potential']['hvac_retrofit_feasibility']}")
    print(f"  Renewable Integration Potential: {building['retrofit_potential']['renewable_integration_potential']}")
    
    print("\n" + "="*80)
    print("Report generated successfully!")
    print("="*80 + "\n")


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("BUILDING DNA DATASET - VISUALIZATION EXAMPLES")
    print("="*80 + "\n")
    
    print("Generating visualizations...\n")
    
    # Generate all visualizations
    visualize_envelope_performance()
    visualize_hvac_performance()
    visualize_energy_breakdown()
    visualize_retrofit_scenarios()
    
    # Generate summary report
    create_summary_report()
    
    print("\n✓ All visualizations completed successfully!")
    print("\nGenerated files:")
    print("  - envelope_performance.png")
    print("  - hvac_performance.png")
    print("  - energy_breakdown.png")
    print("  - retrofit_scenarios.png")
    print("\nDataset location:", DATASET_ROOT)
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
