import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("Loading datasets...")
# Load all datasets
pipeline_data = pd.read_csv('underground_structure_failure_datasets/csv_data/01_pipeline_leakage_sinkholes.csv')
embankment_data = pd.read_csv('underground_structure_failure_datasets/csv_data/02_instrumented_embankment.csv')
excavation_data = pd.read_csv('underground_structure_failure_datasets/csv_data/03_deep_excavation_london_clay.csv')
geohazard_data = pd.read_csv('underground_structure_failure_datasets/csv_data/04_geohazard_susceptibility.csv')
met_data = pd.read_csv('underground_structure_failure_datasets/csv_data/05_meteorological_embankment.csv')
sandy_soil = pd.read_csv('underground_structure_failure_datasets/csv_data/06a_sandy_soil_properties.csv')
clay_soil = pd.read_csv('underground_structure_failure_datasets/csv_data/06b_clay_soil_properties.csv')
tunnel_data = pd.read_csv('underground_structure_failure_datasets/csv_data/07_tunnel_monitoring.csv')
centrifuge_data = pd.read_csv('underground_structure_failure_datasets/csv_data/08_centrifuge_physical_modeling.csv')
fem_data = pd.read_csv('underground_structure_failure_datasets/csv_data/09_synthetic_fem_parametric.csv')

print("Creating Figure 1: Pipeline Leakage Analysis...")
# Figure 1: Pipeline Leakage Sinkholes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Pipeline Leakage Sinkholes Analysis', fontsize=16, fontweight='bold')

# Time-Settlement relationship
axes[0,0].scatter(pipeline_data['Time_hours'], pipeline_data['Settlement_mm'], 
                 c=pipeline_data['Flow_Rate_L_per_min'], cmap='viridis', alpha=0.6, s=30)
axes[0,0].set_xlabel('Time (hours)')
axes[0,0].set_ylabel('Settlement (mm)')
axes[0,0].set_title('Time-Settlement Relationship')
cbar = plt.colorbar(axes[0,0].collections[0], ax=axes[0,0])
cbar.set_label('Flow Rate (L/min)')

# Failure occurrence by flow rate
failure_by_flow = pipeline_data.groupby(pd.cut(pipeline_data['Flow_Rate_L_per_min'], bins=5))['Failure_Occurred'].mean()
axes[0,1].bar(range(len(failure_by_flow)), failure_by_flow.values, color='coral', edgecolor='black')
axes[0,1].set_xlabel('Flow Rate Range')
axes[0,1].set_ylabel('Failure Probability')
axes[0,1].set_title('Failure Probability by Flow Rate')
axes[0,1].set_xticks(range(len(failure_by_flow)))
axes[0,1].set_xticklabels(['Low', 'Med-Low', 'Med', 'Med-High', 'High'], rotation=45)

# Sinkhole diameter distribution
axes[1,0].hist(pipeline_data['Sinkhole_Diameter_m'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
axes[1,0].set_xlabel('Sinkhole Diameter (m)')
axes[1,0].set_ylabel('Frequency')
axes[1,0].set_title('Sinkhole Diameter Distribution')
axes[1,0].axvline(pipeline_data['Sinkhole_Diameter_m'].mean(), color='red', linestyle='--', label='Mean')
axes[1,0].legend()

# Cavity volume vs settlement
axes[1,1].scatter(pipeline_data['Cavity_Volume_m3'], pipeline_data['Settlement_mm'], 
                 c=pipeline_data['Failure_Occurred'], cmap='RdYlGn_r', alpha=0.6, s=30)
axes[1,1].set_xlabel('Cavity Volume (m³)')
axes[1,1].set_ylabel('Settlement (mm)')
axes[1,1].set_title('Cavity Volume vs Settlement')
cbar2 = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
cbar2.set_label('Failure (0=No, 1=Yes)')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/01_pipeline_leakage_analysis.png', bbox_inches='tight')
plt.close()

print("Creating Figure 2: Embankment Monitoring...")
# Figure 2: Instrumented Embankment
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Instrumented Embankment Monitoring Data', fontsize=16, fontweight='bold')

# Wave velocity profiles
axes[0,0].scatter(embankment_data['P_Wave_Velocity_m_per_s'], embankment_data['Depth_m'], 
                 c='blue', alpha=0.5, s=20, label='P-Wave')
axes[0,0].scatter(embankment_data['S_Wave_Velocity_m_per_s'], embankment_data['Depth_m'], 
                 c='red', alpha=0.5, s=20, label='S-Wave')
axes[0,0].invert_yaxis()
axes[0,0].set_xlabel('Wave Velocity (m/s)')
axes[0,0].set_ylabel('Depth (m)')
axes[0,0].set_title('Seismic Wave Velocity Profile')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Displacement by loading stage
stage_disp = embankment_data.groupby('Loading_Stage')['Extensometer_Displacement_mm'].mean()
axes[0,1].plot(stage_disp.index, stage_disp.values, marker='o', linewidth=2, markersize=8, color='darkgreen')
axes[0,1].fill_between(stage_disp.index, 0, stage_disp.values, alpha=0.3, color='darkgreen')
axes[0,1].set_xlabel('Loading Stage')
axes[0,1].set_ylabel('Mean Displacement (mm)')
axes[0,1].set_title('Displacement vs Loading Stage')
axes[0,1].grid(True, alpha=0.3)

# Failure mode distribution
failure_counts = embankment_data['Failure_Mode'].value_counts()
colors = ['green', 'orange', 'coral', 'red']
axes[1,0].pie(failure_counts.values, labels=failure_counts.index, autopct='%1.1f%%', 
             colors=colors, startangle=90)
axes[1,0].set_title('Failure Mode Distribution')

# Pressure vs displacement
axes[1,1].scatter(embankment_data['Pressure_Cell_kPa'], embankment_data['Extensometer_Displacement_mm'],
                 c=embankment_data['Undrained_Shear_Strength_kPa'], cmap='plasma', alpha=0.6, s=30)
axes[1,1].set_xlabel('Pressure (kPa)')
axes[1,1].set_ylabel('Displacement (mm)')
axes[1,1].set_title('Pressure vs Displacement')
cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
cbar.set_label('Shear Strength (kPa)')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/02_embankment_monitoring.png', bbox_inches='tight')
plt.close()

print("Creating Figure 3: Deep Excavation Analysis...")
# Figure 3: Deep Excavation in London Clay
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Deep Excavation in London Clay - Parametric Analysis', fontsize=16, fontweight='bold')

# Excavation depth vs horizontal displacement
scatter = axes[0,0].scatter(excavation_data['Excavation_Depth_m'], excavation_data['Horizontal_Displacement_mm'],
                           c=excavation_data['Safety_Factor'], cmap='RdYlGn', alpha=0.6, s=30, vmin=0.5, vmax=3)
axes[0,0].set_xlabel('Excavation Depth (m)')
axes[0,0].set_ylabel('Horizontal Displacement (mm)')
axes[0,0].set_title('Depth vs Wall Displacement')
cbar = plt.colorbar(scatter, ax=axes[0,0])
cbar.set_label('Safety Factor')

# Wall thickness effect
axes[0,1].scatter(excavation_data['Wall_Thickness_m'], excavation_data['Max_Bending_Moment_kNm'],
                 alpha=0.5, s=30, c='darkblue')
axes[0,1].set_xlabel('Wall Thickness (m)')
axes[0,1].set_ylabel('Max Bending Moment (kNm)')
axes[0,1].set_title('Wall Thickness vs Bending Moment')
axes[0,1].grid(True, alpha=0.3)

# Risk level distribution
risk_counts = excavation_data['Failure_Risk'].value_counts()
risk_order = ['Low', 'Medium', 'High', 'Critical']
risk_colors = ['green', 'yellow', 'orange', 'red']
risk_data = [risk_counts.get(r, 0) for r in risk_order]
axes[1,0].bar(risk_order, risk_data, color=risk_colors, edgecolor='black')
axes[1,0].set_xlabel('Risk Level')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('Failure Risk Distribution')

# Settlement vs safety factor
axes[1,1].scatter(excavation_data['Safety_Factor'], excavation_data['Ground_Settlement_mm'],
                 c=excavation_data['Excavation_Depth_m'], cmap='coolwarm', alpha=0.6, s=30)
axes[1,1].axvline(x=1.5, color='red', linestyle='--', label='Typical Min SF=1.5')
axes[1,1].set_xlabel('Safety Factor')
axes[1,1].set_ylabel('Ground Settlement (mm)')
axes[1,1].set_title('Safety Factor vs Settlement')
axes[1,1].legend()
cbar2 = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
cbar2.set_label('Excavation Depth (m)')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/03_deep_excavation_analysis.png', bbox_inches='tight')
plt.close()

print("Creating Figure 4: Geohazard Susceptibility Map...")
# Figure 4: Geohazard Susceptibility
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Geohazard Susceptibility Analysis', fontsize=16, fontweight='bold')

# Spatial distribution
scatter = axes[0,0].scatter(geohazard_data['Longitude'], geohazard_data['Latitude'],
                           c=geohazard_data['Failure_Probability'], cmap='hot', alpha=0.6, s=40)
axes[0,0].set_xlabel('Longitude')
axes[0,0].set_ylabel('Latitude')
axes[0,0].set_title('Spatial Distribution of Failure Probability')
cbar = plt.colorbar(scatter, ax=axes[0,0])
cbar.set_label('Failure Probability')
axes[0,0].grid(True, alpha=0.3)

# Ground movement potential by soil type
movement_by_soil = geohazard_data.groupby(['Soil_Type', 'Ground_Movement_Potential']).size().unstack(fill_value=0)
movement_by_soil.plot(kind='bar', stacked=True, ax=axes[0,1], 
                     color=['green', 'lightgreen', 'yellow', 'orange', 'red'])
axes[0,1].set_xlabel('Soil Type')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Ground Movement Potential by Soil Type')
axes[0,1].legend(title='Movement Potential', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Asset age vs corrosion rate
axes[1,0].scatter(geohazard_data['Asset_Age_years'], geohazard_data['Corrosion_Rate_mm_per_year'],
                 c=geohazard_data['Failure_Probability'], cmap='Reds', alpha=0.6, s=30)
axes[1,0].set_xlabel('Asset Age (years)')
axes[1,0].set_ylabel('Corrosion Rate (mm/year)')
axes[1,0].set_title('Asset Age vs Corrosion Rate')
cbar2 = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])
cbar2.set_label('Failure Probability')

# Asset type distribution
asset_counts = geohazard_data['Asset_Type'].value_counts()
axes[1,1].pie(asset_counts.values, labels=asset_counts.index, autopct='%1.1f%%', startangle=90)
axes[1,1].set_title('Asset Type Distribution')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/04_geohazard_susceptibility.png', bbox_inches='tight')
plt.close()

print("Creating Figure 5: Meteorological Impact Analysis...")
# Figure 5: Meteorological Data
met_data['Date'] = pd.to_datetime(met_data['Date'])

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Meteorological Impact on Clay Embankment', fontsize=16, fontweight='bold')

# Precipitation time series
axes[0,0].plot(met_data['Date'], met_data['Precipitation_mm'], linewidth=0.8, color='blue', alpha=0.7)
axes[0,0].set_xlabel('Date')
axes[0,0].set_ylabel('Precipitation (mm)')
axes[0,0].set_title('Daily Precipitation')
axes[0,0].grid(True, alpha=0.3)

# Temperature variation
axes[0,1].plot(met_data['Date'], met_data['Temperature_C'], linewidth=0.8, color='red', alpha=0.7)
axes[0,1].set_xlabel('Date')
axes[0,1].set_ylabel('Temperature (°C)')
axes[0,1].set_title('Temperature Variation')
axes[0,1].grid(True, alpha=0.3)

# Soil moisture vs pore pressure
axes[1,0].scatter(met_data['Soil_Moisture_percent'], met_data['Pore_Water_Pressure_kPa'],
                 alpha=0.5, s=20, c='green')
axes[1,0].set_xlabel('Soil Moisture (%)')
axes[1,0].set_ylabel('Pore Water Pressure (kPa)')
axes[1,0].set_title('Soil Moisture vs Pore Pressure')
axes[1,0].grid(True, alpha=0.3)

# Embankment settlement over time
axes[1,1].plot(met_data['Date'], met_data['Embankment_Settlement_mm'], linewidth=1, color='darkred')
axes[1,1].set_xlabel('Date')
axes[1,1].set_ylabel('Cumulative Settlement (mm)')
axes[1,1].set_title('Embankment Settlement Over Time')
axes[1,1].grid(True, alpha=0.3)

# Alert level distribution
alert_counts = met_data['Alert_Level'].value_counts()
alert_order = ['Green', 'Yellow', 'Orange', 'Red']
alert_colors = ['green', 'yellow', 'orange', 'red']
alert_data = [alert_counts.get(a, 0) for a in alert_order]
axes[2,0].bar(alert_order, alert_data, color=alert_colors, edgecolor='black')
axes[2,0].set_xlabel('Alert Level')
axes[2,0].set_ylabel('Count (days)')
axes[2,0].set_title('Alert Level Distribution')

# Water balance components
monthly_avg = met_data.groupby(pd.Grouper(key='Date', freq='ME'))[['Precipitation_mm', 'Evapotranspiration_mm']].mean()
axes[2,1].plot(monthly_avg.index, monthly_avg['Precipitation_mm'], marker='o', label='Precipitation', linewidth=2)
axes[2,1].plot(monthly_avg.index, monthly_avg['Evapotranspiration_mm'], marker='s', label='Evapotranspiration', linewidth=2)
axes[2,1].set_xlabel('Date')
axes[2,1].set_ylabel('Average (mm)')
axes[2,1].set_title('Monthly Water Balance')
axes[2,1].legend()
axes[2,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/05_meteorological_analysis.png', bbox_inches='tight')
plt.close()

print("Creating Figure 6: Soil Properties Comparison...")
# Figure 6: Soil Properties Comparison (Sand vs Clay)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Soil Properties: Sand vs Clay Comparison', fontsize=16, fontweight='bold')

# Friction angle comparison
axes[0,0].hist(sandy_soil['Internal_Friction_Angle_degrees'], bins=25, alpha=0.6, label='Sandy Soil', color='orange', edgecolor='black')
axes[0,0].hist(clay_soil['Effective_Friction_Angle_degrees'], bins=25, alpha=0.6, label='Clay Soil', color='brown', edgecolor='black')
axes[0,0].set_xlabel('Friction Angle (degrees)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Internal Friction Angle Distribution')
axes[0,0].legend()

# Permeability comparison (log scale)
axes[0,1].hist(np.log10(sandy_soil['Permeability_m_per_s']), bins=25, alpha=0.6, label='Sandy Soil', color='orange', edgecolor='black')
axes[0,1].hist(np.log10(clay_soil['Permeability_m_per_s']), bins=25, alpha=0.6, label='Clay Soil', color='brown', edgecolor='black')
axes[0,1].set_xlabel('log₁₀(Permeability) [m/s]')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Permeability Distribution (Log Scale)')
axes[0,1].legend()

# Young's modulus comparison
axes[0,2].boxplot([sandy_soil['Youngs_Modulus_MPa'], clay_soil['Youngs_Modulus_MPa']], 
                  labels=['Sandy Soil', 'Clay Soil'], patch_artist=True,
                  boxprops=dict(facecolor='lightblue', edgecolor='black'),
                  medianprops=dict(color='red', linewidth=2))
axes[0,2].set_ylabel("Young's Modulus (MPa)")
axes[0,2].set_title("Young's Modulus Comparison")
axes[0,2].grid(True, alpha=0.3)

# Density comparison
axes[1,0].scatter(sandy_soil['Dry_Density_kg_per_m3'], sandy_soil['Saturated_Density_kg_per_m3'], 
                 alpha=0.5, s=30, label='Sandy Soil', c='orange')
axes[1,0].scatter(clay_soil['Dry_Density_kg_per_m3'], clay_soil['Saturated_Density_kg_per_m3'], 
                 alpha=0.5, s=30, label='Clay Soil', c='brown')
axes[1,0].plot([1200, 2000], [1200, 2000], 'k--', alpha=0.3)
axes[1,0].set_xlabel('Dry Density (kg/m³)')
axes[1,0].set_ylabel('Saturated Density (kg/m³)')
axes[1,0].set_title('Dry vs Saturated Density')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Clay plasticity chart
axes[1,1].scatter(clay_soil['Liquid_Limit_percent'], clay_soil['Plasticity_Index'], 
                 c=clay_soil['Undrained_Shear_Strength_kPa'], cmap='viridis', s=40, alpha=0.6)
axes[1,1].plot([0, 100], [0, 0.73*100], 'r--', label='A-Line', linewidth=2)
axes[1,1].set_xlabel('Liquid Limit (%)')
axes[1,1].set_ylabel('Plasticity Index (%)')
axes[1,1].set_title('Casagrande Plasticity Chart (Clay)')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
cbar.set_label('Undrained Strength (kPa)')

# Sand grain size distribution
axes[1,2].scatter(sandy_soil['D50_mm'], sandy_soil['Uniformity_Coefficient_Cu'], 
                 c=sandy_soil['Relative_Density_percent'], cmap='YlOrRd', s=40, alpha=0.6)
axes[1,2].set_xlabel('D₅₀ (mm)')
axes[1,2].set_ylabel('Uniformity Coefficient (Cᵤ)')
axes[1,2].set_title('Sand Gradation Characteristics')
axes[1,2].grid(True, alpha=0.3)
cbar2 = plt.colorbar(axes[1,2].collections[0], ax=axes[1,2])
cbar2.set_label('Relative Density (%)')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/06_soil_properties_comparison.png', bbox_inches='tight')
plt.close()

print("Creating Figure 7: Tunnel Monitoring Analysis...")
# Figure 7: Tunnel Monitoring
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Tunnel Monitoring and Risk Assessment', fontsize=16, fontweight='bold')

# Tunnel depth vs settlement by soil type
for soil in tunnel_data['Soil_Type'].unique():
    data = tunnel_data[tunnel_data['Soil_Type'] == soil]
    axes[0,0].scatter(data['Tunnel_Depth_m'], data['Vertical_Settlement_mm'], 
                     label=soil, alpha=0.6, s=30)
axes[0,0].set_xlabel('Tunnel Depth (m)')
axes[0,0].set_ylabel('Vertical Settlement (mm)')
axes[0,0].set_title('Depth vs Settlement by Soil Type')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Risk level by construction method
risk_by_method = tunnel_data.groupby(['Construction_Method', 'Risk_Level']).size().unstack(fill_value=0)
risk_by_method.plot(kind='bar', stacked=True, ax=axes[0,1], 
                   color=['green', 'yellow', 'orange', 'red'])
axes[0,1].set_xlabel('Construction Method')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Risk Level by Construction Method')
axes[0,1].legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[0,1].tick_params(axis='x', rotation=45)

# Cover depth ratio vs safety
axes[1,0].scatter(tunnel_data['Cover_Depth_Ratio'], tunnel_data['Crown_Settlement_mm'],
                 c=tunnel_data['Earth_Pressure_kPa'], cmap='plasma', alpha=0.6, s=30)
axes[1,0].axvline(x=2.0, color='red', linestyle='--', label='Typical Min C/D=2.0')
axes[1,0].set_xlabel('Cover to Depth Ratio')
axes[1,0].set_ylabel('Crown Settlement (mm)')
axes[1,0].set_title('Cover Depth Ratio vs Settlement')
axes[1,0].legend()
cbar = plt.colorbar(axes[1,0].collections[0], ax=axes[1,0])
cbar.set_label('Earth Pressure (kPa)')

# Failure mode distribution
failure_counts = tunnel_data['Failure_Mode'].value_counts()
axes[1,1].barh(range(len(failure_counts)), failure_counts.values, color='steelblue', edgecolor='black')
axes[1,1].set_yticks(range(len(failure_counts)))
axes[1,1].set_yticklabels(failure_counts.index)
axes[1,1].set_xlabel('Count')
axes[1,1].set_title('Tunnel Failure Mode Distribution')
axes[1,1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/07_tunnel_monitoring.png', bbox_inches='tight')
plt.close()

print("Creating Figure 8: Centrifuge Testing Results...")
# Figure 8: Centrifuge/Physical Modeling
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Centrifuge Physical Modeling Results', fontsize=16, fontweight='bold')

# g-level effect on settlement
glevel_settlement = centrifuge_data.groupby('Centrifuge_g_Level')['Settlement_mm'].mean()
axes[0,0].plot(glevel_settlement.index, glevel_settlement.values, marker='o', linewidth=2, markersize=10, color='darkblue')
axes[0,0].set_xlabel('Centrifuge g-Level')
axes[0,0].set_ylabel('Mean Settlement (mm)')
axes[0,0].set_title('Effect of g-Level on Settlement')
axes[0,0].grid(True, alpha=0.3)

# Bearing capacity by soil type
bc_by_soil = centrifuge_data.groupby('Soil_Type')['Bearing_Capacity_kPa'].mean().sort_values()
axes[0,1].barh(range(len(bc_by_soil)), bc_by_soil.values, color='coral', edgecolor='black')
axes[0,1].set_yticks(range(len(bc_by_soil)))
axes[0,1].set_yticklabels(bc_by_soil.index)
axes[0,1].set_xlabel('Mean Bearing Capacity (kPa)')
axes[0,1].set_title('Bearing Capacity by Soil Type')
axes[0,1].grid(True, alpha=0.3, axis='x')

# Load-settlement curves by structure type
for structure in centrifuge_data['Structure_Type'].unique():
    data = centrifuge_data[centrifuge_data['Structure_Type'] == structure]
    sorted_data = data.sort_values('Applied_Load_kN')
    axes[1,0].plot(sorted_data['Applied_Load_kN'], sorted_data['Settlement_mm'], 
                  marker='o', label=structure, alpha=0.7, linewidth=1.5)
axes[1,0].set_xlabel('Applied Load (kN)')
axes[1,0].set_ylabel('Settlement (mm)')
axes[1,0].set_title('Load-Settlement Behavior')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Failure mechanism distribution
mech_counts = centrifuge_data['Failure_Mechanism'].value_counts()
colors = plt.cm.Set3(range(len(mech_counts)))
axes[1,1].pie(mech_counts.values, labels=mech_counts.index, autopct='%1.1f%%', 
             colors=colors, startangle=90)
axes[1,1].set_title('Failure Mechanism Distribution')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/08_centrifuge_modeling.png', bbox_inches='tight')
plt.close()

print("Creating Figure 9: FEM Parametric Study...")
# Figure 9: Synthetic FEM Parametric Study
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('FEM Parametric Study: Underground Structure Failure', fontsize=16, fontweight='bold')

# Safety factor by soil type
sf_by_soil = fem_data.groupby('Soil_Type')['Min_Safety_Factor'].mean()
axes[0,0].bar(sf_by_soil.index, sf_by_soil.values, color=['orange', 'brown', 'gray'], edgecolor='black')
axes[0,0].axhline(y=1.5, color='red', linestyle='--', label='Target SF=1.5')
axes[0,0].set_xlabel('Soil Type')
axes[0,0].set_ylabel('Mean Safety Factor')
axes[0,0].set_title('Safety Factor by Soil Type')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3, axis='y')

# H/D ratio effect on displacement
axes[0,1].scatter(fem_data['HD_Ratio'], fem_data['Max_Vertical_Displacement_mm'],
                 c=fem_data['Embedment_Depth_m'], cmap='viridis', alpha=0.6, s=30)
axes[0,1].set_xlabel('H/D Ratio')
axes[0,1].set_ylabel('Max Vertical Displacement (mm)')
axes[0,1].set_title('H/D Ratio Effect on Displacement')
cbar = plt.colorbar(axes[0,1].collections[0], ax=axes[0,1])
cbar.set_label('Embedment Depth (m)')

# Groundwater condition effect
gw_failure = fem_data.groupby('Groundwater_Condition')['Failure_Occurred'].mean()
axes[0,2].bar(gw_failure.index, gw_failure.values, color=['green', 'yellow', 'blue'], edgecolor='black')
axes[0,2].set_xlabel('Groundwater Condition')
axes[0,2].set_ylabel('Failure Probability')
axes[0,2].set_title('Effect of Groundwater on Failure')
axes[0,2].tick_params(axis='x', rotation=45)
axes[0,2].grid(True, alpha=0.3, axis='y')

# Friction angle vs cohesion failure map
scatter = axes[1,0].scatter(fem_data['Friction_Angle_degrees'], fem_data['Cohesion_kPa'],
                           c=fem_data['Failure_Occurred'], cmap='RdYlGn_r', alpha=0.6, s=30)
axes[1,0].set_xlabel('Friction Angle (degrees)')
axes[1,0].set_ylabel('Cohesion (kPa)')
axes[1,0].set_title('Failure Map: φ vs c')
cbar2 = plt.colorbar(scatter, ax=axes[1,0])
cbar2.set_label('Failure (0=No, 1=Yes)')

# Structure type failure rates
structure_failure = fem_data.groupby('Structure_Type')['Failure_Occurred'].mean().sort_values()
axes[1,1].barh(range(len(structure_failure)), structure_failure.values, color='steelblue', edgecolor='black')
axes[1,1].set_yticks(range(len(structure_failure)))
axes[1,1].set_yticklabels(structure_failure.index)
axes[1,1].set_xlabel('Failure Rate')
axes[1,1].set_title('Failure Rate by Structure Type')
axes[1,1].grid(True, alpha=0.3, axis='x')

# Failure mode distribution
failure_modes = fem_data['Failure_Mode'].value_counts()
axes[1,2].barh(range(len(failure_modes)), failure_modes.values, color='coral', edgecolor='black')
axes[1,2].set_yticks(range(len(failure_modes)))
axes[1,2].set_yticklabels(failure_modes.index, fontsize=8)
axes[1,2].set_xlabel('Count')
axes[1,2].set_title('Failure Mode Distribution')
axes[1,2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('underground_structure_failure_datasets/figures/09_fem_parametric_study.png', bbox_inches='tight')
plt.close()

print("Creating Figure 10: Comprehensive Summary Dashboard...")
# Figure 10: Summary Dashboard
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
fig.suptitle('Underground Structure Failure: Comprehensive Summary Dashboard', fontsize=18, fontweight='bold')

# Overall failure rates by soil type across all datasets
ax1 = fig.add_subplot(gs[0, :2])
soil_failure_tunnel = tunnel_data.groupby('Soil_Type')['Risk_Level'].apply(lambda x: (x.isin(['High', 'Critical'])).mean())
soil_failure_fem = fem_data.groupby('Soil_Type')['Failure_Occurred'].mean()
x = np.arange(len(soil_failure_fem))
width = 0.35
ax1.bar(x - width/2, soil_failure_fem.values, width, label='FEM Simulation', color='steelblue', edgecolor='black')
if len(soil_failure_tunnel) > 0:
    ax1.bar(x + width/2, soil_failure_tunnel.values[:len(x)], width, label='Field Data', color='coral', edgecolor='black')
ax1.set_xlabel('Soil Type', fontsize=12)
ax1.set_ylabel('Failure/High Risk Probability', fontsize=12)
ax1.set_title('Failure Probability Comparison: Sand vs Clay', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(soil_failure_fem.index)
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Key parameter ranges
ax2 = fig.add_subplot(gs[0, 2])
params = ['E\n(MPa)', 'φ\n(deg)', 'c\n(kPa)', 'SF']
sand_vals = [sandy_soil['Youngs_Modulus_MPa'].mean(), 
             sandy_soil['Internal_Friction_Angle_degrees'].mean(),
             sandy_soil['Cohesion_kPa'].mean(),
             fem_data[fem_data['Soil_Type']=='Sand']['Min_Safety_Factor'].mean()]
clay_vals = [clay_soil['Youngs_Modulus_MPa'].mean(),
            clay_soil['Effective_Friction_Angle_degrees'].mean(),
            clay_soil['Effective_Cohesion_kPa'].mean(),
            fem_data[fem_data['Soil_Type']=='Clay']['Min_Safety_Factor'].mean()]

x = np.arange(len(params))
width = 0.35
ax2.bar(x - width/2, sand_vals, width, label='Sand', color='orange', edgecolor='black')
ax2.bar(x + width/2, clay_vals, width, label='Clay', color='brown', edgecolor='black')
ax2.set_ylabel('Mean Value', fontsize=10)
ax2.set_title('Key Parameters:\nSand vs Clay', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(params, fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Displacement analysis
ax3 = fig.add_subplot(gs[1, 0])
all_settlement = pd.concat([
    pd.Series(tunnel_data['Vertical_Settlement_mm'].values, name='Tunnel'),
    pd.Series(excavation_data['Ground_Settlement_mm'].values, name='Excavation'),
    pd.Series(centrifuge_data['Settlement_mm'].values, name='Centrifuge')
], axis=1)
bp = ax3.boxplot([all_settlement['Tunnel'].dropna(), 
                   all_settlement['Excavation'].dropna(),
                   all_settlement['Centrifuge'].dropna()],
                 labels=['Tunnel', 'Excavation', 'Centrifuge'],
                 patch_artist=True)
for patch in bp['boxes']:
    patch.set_facecolor('lightblue')
    patch.set_edgecolor('black')
ax3.set_ylabel('Settlement (mm)', fontsize=10)
ax3.set_title('Settlement Comparison\nAcross Datasets', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.tick_params(axis='x', rotation=45)

# Environmental factors correlation
ax4 = fig.add_subplot(gs[1, 1])
monthly_precip = met_data.groupby(pd.Grouper(key='Date', freq='ME'))['Precipitation_mm'].sum()
monthly_settlement = met_data.groupby(pd.Grouper(key='Date', freq='ME'))['Embankment_Settlement_mm'].last()
ax4_twin = ax4.twinx()
ax4.plot(monthly_precip.index, monthly_precip.values, color='blue', marker='o', label='Precipitation', linewidth=2)
ax4_twin.plot(monthly_settlement.index, monthly_settlement.values, color='red', marker='s', label='Settlement', linewidth=2)
ax4.set_xlabel('Date', fontsize=10)
ax4.set_ylabel('Precipitation (mm)', fontsize=10, color='blue')
ax4_twin.set_ylabel('Settlement (mm)', fontsize=10, color='red')
ax4.set_title('Climate Impact on\nEmbankment', fontsize=12, fontweight='bold')
ax4.tick_params(axis='y', labelcolor='blue')
ax4_twin.tick_params(axis='y', labelcolor='red')
ax4.grid(True, alpha=0.3)

# Construction method safety
ax5 = fig.add_subplot(gs[1, 2])
method_sf = tunnel_data.groupby('Construction_Method').size()
colors_method = plt.cm.Set3(range(len(method_sf)))
ax5.pie(method_sf.values, labels=method_sf.index, autopct='%1.1f%%', colors=colors_method, startangle=90)
ax5.set_title('Construction Method\nDistribution', fontsize=12, fontweight='bold')

# Failure mechanism heatmap
ax6 = fig.add_subplot(gs[2, :])
failure_matrix = fem_data.groupby(['Soil_Type', 'Failure_Mode']).size().unstack(fill_value=0)
im = ax6.imshow(failure_matrix.values, cmap='YlOrRd', aspect='auto')
ax6.set_xticks(np.arange(len(failure_matrix.columns)))
ax6.set_yticks(np.arange(len(failure_matrix.index)))
ax6.set_xticklabels(failure_matrix.columns, rotation=45, ha='right', fontsize=9)
ax6.set_yticklabels(failure_matrix.index, fontsize=10)
ax6.set_title('Failure Mode Distribution by Soil Type (FEM Study)', fontsize=14, fontweight='bold')
ax6.set_xlabel('Failure Mode', fontsize=12)
ax6.set_ylabel('Soil Type', fontsize=12)

# Add text annotations
for i in range(len(failure_matrix.index)):
    for j in range(len(failure_matrix.columns)):
        text = ax6.text(j, i, int(failure_matrix.values[i, j]),
                       ha="center", va="center", color="black", fontsize=9)

plt.colorbar(im, ax=ax6, label='Count')

plt.savefig('underground_structure_failure_datasets/figures/10_comprehensive_summary_dashboard.png', bbox_inches='tight')
plt.close()

print("\n" + "="*70)
print("All visualizations created successfully!")
print("="*70)
print(f"\nTotal figures generated: 10 PNG files")
print(f"Location: underground_structure_failure_datasets/figures/")
print("\nFigure List:")
print("  1. Pipeline Leakage Analysis")
print("  2. Embankment Monitoring")
print("  3. Deep Excavation Analysis")
print("  4. Geohazard Susceptibility")
print("  5. Meteorological Analysis")
print("  6. Soil Properties Comparison")
print("  7. Tunnel Monitoring")
print("  8. Centrifuge Modeling")
print("  9. FEM Parametric Study")
print(" 10. Comprehensive Summary Dashboard")
print("="*70)
