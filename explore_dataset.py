"""
Dataset Exploration and Analysis Script
Demonstrates the capabilities and insights from the Building Retrofit Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetExplorer:
    """Comprehensive exploration and analysis of the building retrofit dataset"""
    
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all dataset components"""
        print("Loading dataset components...")
        
        # Load building attributes
        self.data['basic_info'] = pd.read_csv(f"{self.data_dir}/building_basic_info.csv")
        self.data['geometric'] = pd.read_csv(f"{self.data_dir}/building_geometric_data.csv")
        self.data['thermal'] = pd.read_csv(f"{self.data_dir}/building_thermal_properties.csv")
        self.data['materials'] = pd.read_csv(f"{self.data_dir}/building_construction_materials.csv")
        
        # Load IoT data
        self.data['energy'] = pd.read_csv(f"{self.data_dir}/iot_energy_consumption.csv")
        self.data['environmental'] = pd.read_csv(f"{self.data_dir}/iot_environmental_parameters.csv")
        self.data['weather'] = pd.read_csv(f"{self.data_dir}/iot_weather_conditions.csv")
        self.data['occupancy'] = pd.read_csv(f"{self.data_dir}/iot_occupancy_patterns.csv")
        
        # Load energy performance data
        self.data['historical'] = pd.read_csv(f"{self.data_dir}/energy_historical_consumption.csv")
        self.data['ratings'] = pd.read_csv(f"{self.data_dir}/energy_efficiency_ratings.csv")
        self.data['retrofit'] = pd.read_csv(f"{self.data_dir}/energy_retrofit_impact.csv")
        
        # Load LCA data
        self.data['epds'] = pd.read_csv(f"{self.data_dir}/lca_material_epds.csv")
        self.data['building_lca'] = pd.read_csv(f"{self.data_dir}/lca_building_lca.csv")
        
        print("All data loaded successfully!")
    
    def create_integrated_dataset(self):
        """Create an integrated dataset combining all building information"""
        print("Creating integrated building dataset...")
        
        # Start with basic info
        integrated = self.data['basic_info'].copy()
        
        # Merge all building data
        for key in ['geometric', 'thermal', 'materials']:
            integrated = integrated.merge(self.data[key], on='building_id', how='left')
        
        # Add energy performance summary
        energy_summary = self.data['historical'].groupby('building_id').agg({
            'total_energy_kwh': 'mean',
            'energy_intensity_kwhm2': 'mean'
        }).reset_index()
        energy_summary.columns = ['building_id', 'avg_annual_energy_kwh', 'avg_energy_intensity_kwhm2']
        
        integrated = integrated.merge(energy_summary, on='building_id', how='left')
        
        # Add latest efficiency rating
        latest_ratings = self.data['ratings'].loc[
            self.data['ratings'].groupby('building_id')['rating_year'].idxmax()
        ][['building_id', 'eu_energy_rating', 'energy_performance_index']]
        
        integrated = integrated.merge(latest_ratings, on='building_id', how='left')
        
        # Add retrofit information
        retrofit_summary = self.data['retrofit'].groupby('building_id').agg({
            'energy_savings_percent': 'max',
            'retrofit_type': 'first',
            'cost_euro': 'sum'
        }).reset_index()
        
        integrated = integrated.merge(retrofit_summary, on='building_id', how='left')
        integrated['has_retrofit'] = integrated['energy_savings_percent'].notna()
        
        return integrated
    
    def analyze_building_characteristics(self):
        """Analyze building characteristics and their distribution"""
        print("\n" + "="*60)
        print("BUILDING CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        integrated = self.create_integrated_dataset()
        
        # Building type distribution
        print("\n1. Building Type Distribution:")
        type_dist = integrated['building_type'].value_counts()
        for building_type, count in type_dist.items():
            percentage = (count / len(integrated)) * 100
            print(f"   {building_type}: {count} buildings ({percentage:.1f}%)")
        
        # Construction year analysis
        print(f"\n2. Construction Year Analysis:")
        print(f"   Oldest building: {integrated['construction_year'].min()}")
        print(f"   Newest building: {integrated['construction_year'].max()}")
        print(f"   Average construction year: {integrated['construction_year'].mean():.1f}")
        
        # Age distribution
        current_year = 2024
        integrated['building_age'] = current_year - integrated['construction_year']
        age_groups = pd.cut(integrated['building_age'], 
                           bins=[0, 20, 40, 60, 100], 
                           labels=['0-20 years', '21-40 years', '41-60 years', '60+ years'])
        print(f"\n3. Building Age Distribution:")
        age_dist = age_groups.value_counts()
        for age_group, count in age_dist.items():
            percentage = (count / len(integrated)) * 100
            print(f"   {age_group}: {count} buildings ({percentage:.1f}%)")
        
        # Energy efficiency ratings
        print(f"\n4. Energy Efficiency Ratings:")
        rating_dist = integrated['eu_energy_rating'].value_counts().sort_index()
        for rating, count in rating_dist.items():
            percentage = (count / len(integrated)) * 100
            print(f"   Rating {rating}: {count} buildings ({percentage:.1f}%)")
        
        # Retrofit status
        retrofit_count = integrated['has_retrofit'].sum()
        print(f"\n5. Retrofit Status:")
        print(f"   Buildings with retrofits: {retrofit_count} ({retrofit_count/len(integrated)*100:.1f}%)")
        print(f"   Buildings without retrofits: {len(integrated) - retrofit_count} ({(len(integrated) - retrofit_count)/len(integrated)*100:.1f}%)")
    
    def analyze_energy_patterns(self):
        """Analyze energy consumption patterns"""
        print("\n" + "="*60)
        print("ENERGY CONSUMPTION ANALYSIS")
        print("="*60)
        
        # Convert timestamps
        energy_data = self.data['energy'].copy()
        energy_data['timestamp'] = pd.to_datetime(energy_data['timestamp'])
        energy_data['month'] = energy_data['timestamp'].dt.month
        energy_data['year'] = energy_data['timestamp'].dt.year
        
        # Add building type information
        energy_with_type = energy_data.merge(
            self.data['basic_info'][['building_id', 'building_type']], 
            on='building_id'
        )
        
        # Overall energy statistics
        print("\n1. Overall Energy Consumption Statistics:")
        print(f"   Average daily consumption: {energy_data['total_consumption_kwh'].mean():.1f} kWh")
        print(f"   Median daily consumption: {energy_data['total_consumption_kwh'].median():.1f} kWh")
        print(f"   Standard deviation: {energy_data['total_consumption_kwh'].std():.1f} kWh")
        print(f"   Range: {energy_data['total_consumption_kwh'].min():.1f} - {energy_data['total_consumption_kwh'].max():.1f} kWh")
        
        # Energy consumption by building type
        print(f"\n2. Energy Consumption by Building Type:")
        energy_by_type = energy_with_type.groupby('building_type')['total_consumption_kwh'].agg(['mean', 'std']).round(1)
        for building_type, row in energy_by_type.iterrows():
            print(f"   {building_type}: {row['mean']:.1f} ± {row['std']:.1f} kWh/day")
        
        # Seasonal patterns
        print(f"\n3. Seasonal Energy Patterns:")
        seasonal_energy = energy_data.groupby('month')['total_consumption_kwh'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, consumption in seasonal_energy.items():
            print(f"   {months[month-1]}: {consumption:.1f} kWh/day")
        
        # End-use breakdown
        print(f"\n4. End-Use Energy Breakdown:")
        end_use_cols = ['heating_kwh', 'cooling_kwh', 'lighting_kwh', 'appliances_kwh', 'hvac_kwh']
        end_use_totals = energy_data[end_use_cols].sum()
        total_energy = end_use_totals.sum()
        for col in end_use_cols:
            percentage = (end_use_totals[col] / total_energy) * 100
            print(f"   {col.replace('_kwh', '').title()}: {percentage:.1f}%")
    
    def analyze_environmental_conditions(self):
        """Analyze indoor environmental conditions"""
        print("\n" + "="*60)
        print("INDOOR ENVIRONMENTAL CONDITIONS ANALYSIS")
        print("="*60)
        
        env_data = self.data['environmental'].copy()
        env_data['timestamp'] = pd.to_datetime(env_data['timestamp'])
        
        # Temperature analysis
        print("\n1. Temperature Analysis:")
        print(f"   Average temperature: {env_data['temperature_c'].mean():.1f}°C")
        print(f"   Temperature range: {env_data['temperature_c'].min():.1f} - {env_data['temperature_c'].max():.1f}°C")
        print(f"   Standard deviation: {env_data['temperature_c'].std():.1f}°C")
        
        # CO2 levels
        print(f"\n2. CO2 Levels Analysis:")
        print(f"   Average CO2: {env_data['co2_ppm'].mean():.0f} ppm")
        print(f"   CO2 range: {env_data['co2_ppm'].min():.0f} - {env_data['co2_ppm'].max():.0f} ppm")
        
        # Air quality assessment
        good_air_quality = (env_data['co2_ppm'] < 1000) & (env_data['tvoc_ppb'] < 500) & (env_data['pm25_ugm3'] < 25)
        good_air_percentage = (good_air_quality.sum() / len(env_data)) * 100
        print(f"\n3. Air Quality Assessment:")
        print(f"   Buildings with good air quality: {good_air_percentage:.1f}% of measurements")
        print(f"   (CO2 < 1000 ppm, TVOC < 500 ppb, PM2.5 < 25 μg/m³)")
    
    def identify_retrofit_opportunities(self):
        """Identify buildings with high retrofit potential"""
        print("\n" + "="*60)
        print("RETROFIT OPPORTUNITY ANALYSIS")
        print("="*60)
        
        integrated = self.create_integrated_dataset()
        
        # High energy intensity buildings
        high_energy_threshold = integrated['avg_energy_intensity_kwhm2'].quantile(0.75)
        high_energy_buildings = integrated[
            (integrated['avg_energy_intensity_kwhm2'] > high_energy_threshold) & 
            (~integrated['has_retrofit'])
        ]
        
        print(f"\n1. High Energy Intensity Buildings (No Retrofit):")
        print(f"   Threshold: >{high_energy_threshold:.1f} kWh/m²/year")
        print(f"   Number of buildings: {len(high_energy_buildings)}")
        print(f"   Average energy intensity: {high_energy_buildings['avg_energy_intensity_kwhm2'].mean():.1f} kWh/m²/year")
        
        # Old buildings with poor efficiency
        current_year = 2024
        integrated['building_age'] = current_year - integrated['construction_year']
        old_inefficient = integrated[
            (integrated['building_age'] > 40) & 
            (integrated['eu_energy_rating'].isin(['E', 'F', 'G'])) &
            (~integrated['has_retrofit'])
        ]
        
        print(f"\n2. Old Buildings with Poor Efficiency (No Retrofit):")
        print(f"   Age > 40 years, Rating E/F/G")
        print(f"   Number of buildings: {len(old_inefficient)}")
        if len(old_inefficient) > 0:
            print(f"   Average age: {old_inefficient['building_age'].mean():.1f} years")
            print(f"   Average energy intensity: {old_inefficient['avg_energy_intensity_kwhm2'].mean():.1f} kWh/m²/year")
        
        # Retrofit impact analysis
        if len(self.data['retrofit']) > 0:
            print(f"\n3. Retrofit Impact Analysis:")
            print(f"   Number of completed retrofits: {len(self.data['retrofit'])}")
            print(f"   Average energy savings: {self.data['retrofit']['energy_savings_percent'].mean():.1f}%")
            print(f"   Average cost: €{self.data['retrofit']['cost_euro'].mean():,.0f}")
            print(f"   Average payback period: {self.data['retrofit']['payback_period_years'].mean():.1f} years")
    
    def analyze_lca_impacts(self):
        """Analyze lifecycle assessment impacts"""
        print("\n" + "="*60)
        print("LIFECYCLE ASSESSMENT ANALYSIS")
        print("="*60)
        
        # Material impact analysis
        print("\n1. Material Environmental Impact:")
        epd_data = self.data['epds']
        print(f"   Most impactful material: {epd_data.loc[epd_data['global_warming_potential_kgco2eq'].idxmax(), 'material_name']}")
        print(f"   Least impactful material: {epd_data.loc[epd_data['global_warming_potential_kgco2eq'].idxmin(), 'material_name']}")
        
        # Building LCA by lifecycle stage
        print(f"\n2. Building LCA by Lifecycle Stage:")
        lca_by_stage = self.data['building_lca'].groupby('lifecycle_stage').agg({
            'total_gwp_kgco2eq': 'mean',
            'total_pe_mj': 'mean'
        }).round(0)
        
        for stage, row in lca_by_stage.iterrows():
            print(f"   {stage.title()}: {row['total_gwp_kgco2eq']:.0f} kg CO2eq, {row['total_pe_mj']:.0f} MJ")
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of the dataset"""
        print("\nCreating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Building type distribution
        ax1 = plt.subplot(4, 3, 1)
        integrated = self.create_integrated_dataset()
        building_types = integrated['building_type'].value_counts()
        ax1.pie(building_types.values, labels=building_types.index, autopct='%1.1f%%')
        ax1.set_title('Building Type Distribution', fontsize=14, fontweight='bold')
        
        # 2. Construction year distribution
        ax2 = plt.subplot(4, 3, 2)
        ax2.hist(integrated['construction_year'], bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Construction Year Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Number of Buildings')
        
        # 3. Energy efficiency ratings
        ax3 = plt.subplot(4, 3, 3)
        rating_counts = integrated['eu_energy_rating'].value_counts().sort_index()
        ax3.bar(rating_counts.index, rating_counts.values, alpha=0.7)
        ax3.set_title('Energy Efficiency Ratings', fontsize=14, fontweight='bold')
        ax3.set_xlabel('EU Energy Rating')
        ax3.set_ylabel('Number of Buildings')
        
        # 4. Energy consumption by building type
        ax4 = plt.subplot(4, 3, 4)
        energy_data = self.data['energy'].copy()
        energy_data = energy_data.merge(
            self.data['basic_info'][['building_id', 'building_type']], 
            on='building_id'
        )
        energy_by_type = energy_data.groupby('building_type')['total_consumption_kwh'].mean()
        ax4.bar(energy_by_type.index, energy_by_type.values, alpha=0.7)
        ax4.set_title('Average Daily Energy Consumption by Building Type', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Building Type')
        ax4.set_ylabel('Average Consumption (kWh/day)')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Seasonal energy patterns
        ax5 = plt.subplot(4, 3, 5)
        energy_data['timestamp'] = pd.to_datetime(energy_data['timestamp'])
        energy_data['month'] = energy_data['timestamp'].dt.month
        seasonal_energy = energy_data.groupby('month')['total_consumption_kwh'].mean()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax5.plot(range(1, 13), seasonal_energy.values, marker='o', linewidth=2, markersize=6)
        ax5.set_title('Seasonal Energy Consumption Pattern', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Month')
        ax5.set_ylabel('Average Consumption (kWh/day)')
        ax5.set_xticks(range(1, 13))
        ax5.set_xticklabels(months)
        ax5.grid(True, alpha=0.3)
        
        # 6. Energy intensity vs building age
        ax6 = plt.subplot(4, 3, 6)
        current_year = 2024
        integrated['building_age'] = current_year - integrated['construction_year']
        ax6.scatter(integrated['building_age'], integrated['avg_energy_intensity_kwhm2'], 
                   alpha=0.6, s=50)
        ax6.set_title('Energy Intensity vs Building Age', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Building Age (years)')
        ax6.set_ylabel('Energy Intensity (kWh/m²/year)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Indoor temperature distribution
        ax7 = plt.subplot(4, 3, 7)
        env_data = self.data['environmental']
        ax7.hist(env_data['temperature_c'], bins=30, alpha=0.7, edgecolor='black')
        ax7.axvline(env_data['temperature_c'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {env_data["temperature_c"].mean():.1f}°C')
        ax7.set_title('Indoor Temperature Distribution', fontsize=14, fontweight='bold')
        ax7.set_xlabel('Temperature (°C)')
        ax7.set_ylabel('Frequency')
        ax7.legend()
        
        # 8. CO2 levels over time
        ax8 = plt.subplot(4, 3, 8)
        env_data['timestamp'] = pd.to_datetime(env_data['timestamp'])
        daily_co2 = env_data.groupby(env_data['timestamp'].dt.date)['co2_ppm'].mean()
        ax8.plot(daily_co2.index, daily_co2.values, alpha=0.7, linewidth=1)
        ax8.set_title('Daily Average CO2 Levels', fontsize=14, fontweight='bold')
        ax8.set_xlabel('Date')
        ax8.set_ylabel('CO2 (ppm)')
        ax8.tick_params(axis='x', rotation=45)
        
        # 9. Retrofit impact analysis
        ax9 = plt.subplot(4, 3, 9)
        if len(self.data['retrofit']) > 0:
            retrofit_types = self.data['retrofit']['retrofit_type'].value_counts()
            ax9.bar(retrofit_types.index, retrofit_types.values, alpha=0.7)
            ax9.set_title('Retrofit Types Distribution', fontsize=14, fontweight='bold')
            ax9.set_xlabel('Retrofit Type')
            ax9.set_ylabel('Number of Retrofits')
            ax9.tick_params(axis='x', rotation=45)
        else:
            ax9.text(0.5, 0.5, 'No Retrofit Data', ha='center', va='center', transform=ax9.transAxes)
            ax9.set_title('Retrofit Types Distribution', fontsize=14, fontweight='bold')
        
        # 10. Material environmental impact
        ax10 = plt.subplot(4, 3, 10)
        epd_data = self.data['epds']
        ax10.barh(epd_data['material_name'], epd_data['global_warming_potential_kgco2eq'], alpha=0.7)
        ax10.set_title('Material Global Warming Potential', fontsize=14, fontweight='bold')
        ax10.set_xlabel('GWP (kg CO2eq)')
        
        # 11. Energy consumption distribution
        ax11 = plt.subplot(4, 3, 11)
        ax11.hist(energy_data['total_consumption_kwh'], bins=50, alpha=0.7, edgecolor='black')
        ax11.axvline(energy_data['total_consumption_kwh'].mean(), color='red', linestyle='--',
                    label=f'Mean: {energy_data["total_consumption_kwh"].mean():.1f} kWh')
        ax11.set_title('Daily Energy Consumption Distribution', fontsize=14, fontweight='bold')
        ax11.set_xlabel('Energy Consumption (kWh/day)')
        ax11.set_ylabel('Frequency')
        ax11.legend()
        
        # 12. Correlation heatmap
        ax12 = plt.subplot(4, 3, 12)
        numeric_cols = integrated.select_dtypes(include=[np.number]).columns
        correlation_matrix = integrated[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax12, cbar_kws={'shrink': 0.8})
        ax12.set_title('Building Attributes Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.data_dir}/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive visualizations saved to: {self.data_dir}/comprehensive_analysis.png")
    
    def generate_insights_summary(self):
        """Generate a summary of key insights from the dataset"""
        print("\n" + "="*60)
        print("KEY INSIGHTS SUMMARY")
        print("="*60)
        
        integrated = self.create_integrated_dataset()
        
        # Key statistics
        total_buildings = len(integrated)
        avg_energy_intensity = integrated['avg_energy_intensity_kwhm2'].mean()
        retrofit_rate = integrated['has_retrofit'].mean() * 100
        
        print(f"\nDataset Overview:")
        print(f"  • Total buildings: {total_buildings}")
        print(f"  • Average energy intensity: {avg_energy_intensity:.1f} kWh/m²/year")
        print(f"  • Retrofit rate: {retrofit_rate:.1f}%")
        
        # Energy efficiency insights
        poor_efficiency = integrated[integrated['eu_energy_rating'].isin(['E', 'F', 'G'])]
        print(f"\nEnergy Efficiency Insights:")
        print(f"  • Buildings with poor efficiency (E/F/G): {len(poor_efficiency)} ({len(poor_efficiency)/total_buildings*100:.1f}%)")
        print(f"  • Most common building type: {integrated['building_type'].mode().iloc[0]}")
        print(f"  • Average building age: {(2024 - integrated['construction_year']).mean():.1f} years")
        
        # Retrofit opportunities
        high_energy_no_retrofit = integrated[
            (integrated['avg_energy_intensity_kwhm2'] > integrated['avg_energy_intensity_kwhm2'].quantile(0.75)) &
            (~integrated['has_retrofit'])
        ]
        print(f"\nRetrofit Opportunities:")
        print(f"  • High-energy buildings without retrofit: {len(high_energy_no_retrofit)}")
        if len(high_energy_no_retrofit) > 0:
            potential_savings = high_energy_no_retrofit['avg_energy_intensity_kwhm2'].sum() * 0.3  # Assume 30% savings
            print(f"  • Potential energy savings: {potential_savings:.0f} kWh/year")
        
        # Environmental insights
        env_data = self.data['environmental']
        good_air_quality = ((env_data['co2_ppm'] < 1000) & 
                           (env_data['tvoc_ppb'] < 500) & 
                           (env_data['pm25_ugm3'] < 25)).mean() * 100
        
        print(f"\nEnvironmental Insights:")
        print(f"  • Buildings with good air quality: {good_air_quality:.1f}% of measurements")
        print(f"  • Average indoor temperature: {env_data['temperature_c'].mean():.1f}°C")
        print(f"  • Average CO2 levels: {env_data['co2_ppm'].mean():.0f} ppm")
    
    def run_complete_analysis(self):
        """Run the complete dataset analysis"""
        print("="*60)
        print("BUILDING RETROFIT DATASET - COMPREHENSIVE ANALYSIS")
        print("="*60)
        
        # Run all analyses
        self.analyze_building_characteristics()
        self.analyze_energy_patterns()
        self.analyze_environmental_conditions()
        self.identify_retrofit_opportunities()
        self.analyze_lca_impacts()
        self.generate_insights_summary()
        
        # Create visualizations
        self.create_comprehensive_visualizations()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("All analysis results and visualizations have been generated.")
        print("Check the dataset directory for output files.")

if __name__ == "__main__":
    explorer = DatasetExplorer()
    explorer.run_complete_analysis()