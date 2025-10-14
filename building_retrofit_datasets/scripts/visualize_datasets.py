"""
Dataset Visualization Script
Quick visualization examples for exploring the building retrofit datasets.

Note: Requires matplotlib and seaborn. Install with:
pip install matplotlib seaborn
"""

import pandas as pd
import numpy as np

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('whitegrid')
    VIZ_AVAILABLE = True
except ImportError:
    print("⚠️  Matplotlib and Seaborn not installed.")
    print("Install with: pip install matplotlib seaborn")
    VIZ_AVAILABLE = False


def load_data():
    """Load key datasets."""
    print("Loading datasets...")
    
    data = {}
    data['buildings'] = pd.read_csv('../data/integrated_building_master.csv')
    data['iot'] = pd.read_parquet('../data/iot_sensor_data.parquet')
    data['iot']['timestamp'] = pd.to_datetime(data['iot']['timestamp'])
    data['retrofits'] = pd.read_csv('../data/integrated_retrofit_analysis.csv')
    data['monthly'] = pd.read_csv('../data/timeseries_monthly.csv')
    
    print(f"  ✓ Buildings: {len(data['buildings'])}")
    print(f"  ✓ IoT records: {len(data['iot']):,}")
    print(f"  ✓ Retrofit scenarios: {len(data['retrofits'])}")
    
    return data


def plot_building_distribution(buildings):
    """Plot building type and EPC distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Building type distribution
    buildings['building_type'].value_counts().plot(kind='bar', ax=axes[0], color='steelblue')
    axes[0].set_title('Building Type Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Building Type')
    axes[0].set_ylabel('Count')
    axes[0].tick_params(axis='x', rotation=45)
    
    # EPC rating distribution
    epc_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    epc_counts = buildings['epc_rating'].value_counts().reindex(epc_order, fill_value=0)
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b', '#7f8c8d']
    epc_counts.plot(kind='bar', ax=axes[1], color=colors)
    axes[1].set_title('EPC Rating Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('EPC Rating')
    axes[1].set_ylabel('Count')
    axes[1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig('../data/building_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: building_distribution.png")


def plot_energy_consumption_patterns(iot):
    """Plot energy consumption patterns."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Daily average consumption by building type
    iot['date'] = iot['timestamp'].dt.date
    daily_by_type = iot.groupby(['date', 'building_type'])['total_energy_consumption_kwh'].sum().reset_index()
    
    for building_type in iot['building_type'].unique():
        type_data = daily_by_type[daily_by_type['building_type'] == building_type]
        axes[0, 0].plot(pd.to_datetime(type_data['date']), 
                       type_data['total_energy_consumption_kwh'], 
                       label=building_type, alpha=0.7)
    
    axes[0, 0].set_title('Daily Energy Consumption by Building Type', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Energy Consumption (kWh)')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Hourly pattern (average across all days)
    iot['hour'] = iot['timestamp'].dt.hour
    hourly_avg = iot.groupby('hour')['total_energy_consumption_kwh'].mean()
    axes[0, 1].plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2, markersize=6)
    axes[0, 1].set_title('Average Hourly Consumption Pattern', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Average Consumption (kWh)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Energy breakdown by end-use
    end_uses = ['hvac_consumption_kwh', 'lighting_consumption_kwh', 'equipment_consumption_kwh']
    totals = [iot[col].sum() for col in end_uses]
    colors_pie = ['#e74c3c', '#f39c12', '#3498db']
    axes[1, 0].pie(totals, labels=['HVAC', 'Lighting', 'Equipment'], 
                   autopct='%1.1f%%', colors=colors_pie, startangle=90)
    axes[1, 0].set_title('Energy Consumption by End-Use', fontsize=12, fontweight='bold')
    
    # 4. Indoor vs Outdoor Temperature
    sample_building = iot[iot['building_id'] == 'BLD_001'].head(168)  # One week
    ax2 = axes[1, 1]
    ax2.plot(sample_building['timestamp'], sample_building['outdoor_temperature'], 
             label='Outdoor', color='blue', linewidth=2)
    ax2.plot(sample_building['timestamp'], sample_building['indoor_temperature'], 
             label='Indoor', color='red', linewidth=2)
    ax2.set_title('Temperature Profile (BLD_001 - Sample Week)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Temperature (°C)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../data/energy_patterns.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: energy_patterns.png")


def plot_retrofit_analysis(retrofits, buildings):
    """Plot retrofit analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Savings vs Cost by scenario
    for scenario in retrofits['scenario_name'].unique():
        scenario_data = retrofits[retrofits['scenario_name'] == scenario]
        axes[0, 0].scatter(scenario_data['total_cost_eur'], 
                          scenario_data['total_energy_saving_pct'],
                          label=scenario, s=100, alpha=0.6)
    
    axes[0, 0].set_title('Energy Savings vs Investment Cost', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Investment Cost (EUR)')
    axes[0, 0].set_ylabel('Energy Savings (%)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Payback period distribution
    retrofits.boxplot(column='simple_payback_years', by='scenario_name', ax=axes[0, 1])
    axes[0, 1].set_title('Payback Period by Scenario', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Scenario')
    axes[0, 1].set_ylabel('Payback Period (years)')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45)
    
    # 3. ROI by building type
    merged = retrofits.merge(buildings[['building_id', 'building_type']], on='building_id')
    deep_retrofit = merged[merged['scenario_name'] == 'Deep Retrofit']
    
    roi_by_type = deep_retrofit.groupby('building_type')['roi_percent'].mean().sort_values()
    roi_by_type.plot(kind='barh', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Average ROI by Building Type (Deep Retrofit)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('ROI (%)')
    axes[1, 0].set_ylabel('Building Type')
    
    # 4. Carbon savings potential
    carbon_savings = deep_retrofit.groupby('building_type')['annual_carbon_saving_kgco2'].sum() / 1000  # Convert to tonnes
    carbon_savings.plot(kind='bar', ax=axes[1, 1], color='darkgreen')
    axes[1, 1].set_title('Total Carbon Savings Potential by Type (Deep Retrofit)', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Building Type')
    axes[1, 1].set_ylabel('Carbon Savings (tonnes CO₂/year)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../data/retrofit_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: retrofit_analysis.png")


def plot_thermal_performance(buildings):
    """Plot thermal performance analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. U-value vs EPC rating
    epc_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    buildings['epc_rating_cat'] = pd.Categorical(buildings['epc_rating'], categories=epc_order, ordered=True)
    
    buildings.boxplot(column='envelope_avg_u_value', by='epc_rating', ax=axes[0])
    axes[0].set_title('Envelope U-value vs EPC Rating', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('EPC Rating')
    axes[0].set_ylabel('U-value (W/m²K)')
    plt.sca(axes[0])
    plt.xticks(rotation=0)
    
    # 2. Building age vs U-value
    axes[1].scatter(buildings['building_age_years'], buildings['envelope_avg_u_value'], 
                   c=buildings['epc_rating_cat'].cat.codes, cmap='RdYlGn_r', s=100, alpha=0.6)
    axes[1].set_title('Building Age vs Thermal Performance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Building Age (years)')
    axes[1].set_ylabel('Envelope U-value (W/m²K)')
    axes[1].grid(True, alpha=0.3)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlGn_r', 
                               norm=plt.Normalize(vmin=0, vmax=len(epc_order)-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes[1])
    cbar.set_ticks(range(len(epc_order)))
    cbar.set_ticklabels(epc_order)
    cbar.set_label('EPC Rating')
    
    plt.tight_layout()
    plt.savefig('../data/thermal_performance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: thermal_performance.png")


def main():
    """Main visualization function."""
    print("=" * 80)
    print("BUILDING RETROFIT DATASET VISUALIZATION")
    print("=" * 80)
    
    if not VIZ_AVAILABLE:
        return
    
    # Load data
    data = load_data()
    
    print("\nGenerating visualizations...")
    
    # Create plots
    plot_building_distribution(data['buildings'])
    plot_energy_consumption_patterns(data['iot'])
    plot_retrofit_analysis(data['retrofits'], data['buildings'])
    plot_thermal_performance(data['buildings'])
    
    print("\n" + "=" * 80)
    print("✅ Visualizations complete!")
    print("=" * 80)
    print("\nGenerated files in ../data/:")
    print("  • building_distribution.png")
    print("  • energy_patterns.png")
    print("  • retrofit_analysis.png")
    print("  • thermal_performance.png")
    print("\nOpen these files to explore the datasets visually!")


if __name__ == "__main__":
    main()
