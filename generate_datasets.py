"""
=============================================================================
Fabricated Dataset Generator
Topic: Failure Mechanism of Underground Structures in Sandy and Clay Soils
=============================================================================
Generates realistic synthetic datasets for geotechnical research, including:
  1. Physical Model - Sinkhole settlement due to pipeline leakage
  2. Field Monitoring - Instrumented embankment on clay formation
  3. Numerical Simulation - Deep excavation in London Clay
  4. Geohazard Susceptibility - Ground movement potential mapping
  5. Environmental Context - Meteorological data for embankment failure
  6. Soil Properties - Sandy and Clay soil index/mechanical properties
  7. Tunnel Risk Dataset - Structural & monitoring data
  8. Centrifuge Modeling - Experimental physical modeling data
  9. FEM Parametric Study - Synthetic numerical simulation results

All CSVs are saved to ./datasets/ and packaged into datasets.zip
Figures are saved to ./figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.interpolate import make_interp_spline
from scipy.stats import norm, lognorm
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

DATASETS_DIR = "datasets"
FIGURES_DIR = "figures"
os.makedirs(DATASETS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 200,
})

# ============================================================================
# DATASET 1: Physical Model - Sinkhole Settlement (Pipeline Leakage)
# ============================================================================
def generate_sinkhole_settlement():
    print("Generating Dataset 1: Sinkhole Settlement...")
    time_hours = np.arange(0, 121, 1)
    flow_conditions = {
        'Low_Flow_0.5_L_min': {'rate': 0.5, 'max_settle': 45, 'k': 0.035},
        'Medium_Flow_2.0_L_min': {'rate': 2.0, 'max_settle': 120, 'k': 0.055},
        'High_Flow_5.0_L_min': {'rate': 5.0, 'max_settle': 280, 'k': 0.08},
        'Burst_Flow_10.0_L_min': {'rate': 10.0, 'max_settle': 450, 'k': 0.12},
    }

    records = []
    for name, params in flow_conditions.items():
        for t in time_hours:
            settlement = params['max_settle'] * (1 - np.exp(-params['k'] * t))
            settlement += np.random.normal(0, params['max_settle'] * 0.015)
            settlement = max(0, settlement)
            pore_pressure = 10 + 0.3 * params['rate'] * t * np.exp(-0.01 * t)
            pore_pressure += np.random.normal(0, 1.5)
            moisture = 18 + 12 * (1 - np.exp(-0.02 * params['rate'] * t))
            moisture += np.random.normal(0, 0.8)
            records.append({
                'Time_hours': t,
                'Flow_Condition': name,
                'Flow_Rate_L_per_min': params['rate'],
                'Settlement_mm': round(settlement, 2),
                'Pore_Water_Pressure_kPa': round(pore_pressure, 2),
                'Moisture_Content_pct': round(min(moisture, 42), 2),
                'Soil_Type': 'Sandy_Clay',
                'Depth_m': round(np.random.uniform(1.5, 4.5), 1),
                'Void_Ratio': round(np.random.uniform(0.55, 0.85), 3),
            })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/01_sinkhole_settlement_pipeline_leakage.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50']
    for i, (name, params) in enumerate(flow_conditions.items()):
        sub = df[df['Flow_Condition'] == name]
        label = name.replace('_', ' ')
        axes[0].plot(sub['Time_hours'], sub['Settlement_mm'], color=colors[i], alpha=0.85, label=label, linewidth=1.5)
        axes[1].plot(sub['Time_hours'], sub['Pore_Water_Pressure_kPa'], color=colors[i], alpha=0.85, label=label, linewidth=1.5)
        axes[2].plot(sub['Time_hours'], sub['Moisture_Content_pct'], color=colors[i], alpha=0.85, label=label, linewidth=1.5)

    axes[0].set_xlabel('Time (hours)'); axes[0].set_ylabel('Settlement (mm)')
    axes[0].set_title('Surface Settlement vs Time'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()
    axes[1].set_xlabel('Time (hours)'); axes[1].set_ylabel('Pore Water Pressure (kPa)')
    axes[1].set_title('Pore Water Pressure vs Time'); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (hours)'); axes[2].set_ylabel('Moisture Content (%)')
    axes[2].set_title('Moisture Content vs Time'); axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    fig.suptitle('Physical Model: Sinkhole Settlement Due to Pipeline Leakage in Sandy Clay', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig01_sinkhole_settlement.png")
    plt.close()
    return df


# ============================================================================
# DATASET 2: Field Monitoring - Instrumented Embankment on Clay
# ============================================================================
def generate_embankment_monitoring():
    print("Generating Dataset 2: Embankment Monitoring...")
    n_boreholes = 6
    depths = np.arange(0, 30.5, 0.5)
    records = []

    for bh in range(1, n_boreholes + 1):
        base_vp = 300 + 50 * bh
        base_vs = 150 + 25 * bh
        for d in depths:
            vp = base_vp + 35 * d + np.random.normal(0, 20)
            vs = base_vs + 18 * d + np.random.normal(0, 12)
            ext_disp = 0.5 * d * np.exp(-0.05 * d) * (1 + 0.2 * bh) + np.random.normal(0, 0.3)
            pressure = 9.81 * 1.85 * d * (1 + 0.05 * np.sin(d / 5)) + np.random.normal(0, 5)
            records.append({
                'Borehole_ID': f'BH-{bh:02d}',
                'Depth_m': d,
                'P_Wave_Velocity_m_s': round(max(vp, 200), 1),
                'S_Wave_Velocity_m_s': round(max(vs, 100), 1),
                'Vp_Vs_Ratio': round(max(vp, 200) / max(vs, 100), 3),
                'Extensometer_Displacement_mm': round(max(ext_disp, 0), 3),
                'Earth_Pressure_kPa': round(max(pressure, 0), 2),
                'Formation': 'Mudstone_Clay',
                'Loading_Stage': np.random.choice(['Pre-load', 'During_Load', 'Post_Load'], p=[0.2, 0.5, 0.3]),
            })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/02_embankment_monitoring_clay.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    cmap = plt.cm.viridis
    for i, bh in enumerate(range(1, n_boreholes + 1)):
        sub = df[df['Borehole_ID'] == f'BH-{bh:02d}']
        c = cmap(i / (n_boreholes - 1))
        axes[0].plot(sub['P_Wave_Velocity_m_s'], sub['Depth_m'], color=c, label=f'BH-{bh:02d}', linewidth=1.3)
        axes[1].plot(sub['Extensometer_Displacement_mm'], sub['Depth_m'], color=c, label=f'BH-{bh:02d}', linewidth=1.3)
        axes[2].plot(sub['Earth_Pressure_kPa'], sub['Depth_m'], color=c, label=f'BH-{bh:02d}', linewidth=1.3)

    for ax in axes:
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_ylabel('Depth (m)')

    axes[0].set_xlabel('P-Wave Velocity (m/s)'); axes[0].set_title('P-Wave Velocity Profile')
    axes[1].set_xlabel('Displacement (mm)'); axes[1].set_title('Extensometer Displacement')
    axes[2].set_xlabel('Earth Pressure (kPa)'); axes[2].set_title('Earth Pressure Distribution')
    fig.suptitle('Field Monitoring: Instrumented Embankment on Clay (Mudstone Formation)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig02_embankment_monitoring.png")
    plt.close()
    return df


# ============================================================================
# DATASET 3: Numerical Simulation - Deep Excavation in London Clay
# ============================================================================
def generate_deep_excavation():
    print("Generating Dataset 3: Deep Excavation Simulation...")
    exc_depths = [6, 8, 10, 12, 15, 18, 20]
    exc_widths = [10, 15, 20, 25, 30]
    prop_spacings = [2.0, 3.0, 4.0, 5.0]
    wall_thicknesses = [0.6, 0.8, 1.0, 1.2]

    records = []
    for H in exc_depths:
        for W in exc_widths:
            for ps in prop_spacings:
                for wt in wall_thicknesses:
                    stiffness_factor = wt ** 3 / 12
                    base_disp = 0.002 * H ** 1.5 * (W / 20) ** 0.5 * (ps / 3) ** 0.8 / (stiffness_factor ** 0.3)
                    max_horiz_disp = base_disp * 1000 + np.random.normal(0, 1.5)
                    max_settle = max_horiz_disp * np.random.uniform(0.6, 1.0)
                    bending_moment = 0.5 * 19 * H ** 2 * (ps / wt) + np.random.normal(0, 50)
                    fos = 2.5 / (H / (10 * wt)) * (1 / (ps / 3)) + np.random.normal(0, 0.1)
                    fos = max(fos, 0.5)
                    failure = 1 if fos < 1.2 else 0
                    records.append({
                        'Excavation_Depth_m': H,
                        'Excavation_Width_m': W,
                        'Prop_Spacing_m': ps,
                        'Wall_Thickness_m': wt,
                        'Max_Horizontal_Displacement_mm': round(max(max_horiz_disp, 0.5), 2),
                        'Max_Surface_Settlement_mm': round(max(max_settle, 0.2), 2),
                        'Max_Bending_Moment_kNm_per_m': round(max(bending_moment, 10), 1),
                        'Factor_of_Safety': round(fos, 3),
                        'Failure_Flag': failure,
                        'Soil_Type': 'London_Clay',
                        'Cu_kPa': round(70 + 5 * H + np.random.normal(0, 5), 1),
                        'Ko': round(1.5 + np.random.normal(0, 0.1), 2),
                    })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/03_deep_excavation_london_clay.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    for wt in wall_thicknesses:
        sub = df[(df['Wall_Thickness_m'] == wt) & (df['Prop_Spacing_m'] == 3.0) & (df['Excavation_Width_m'] == 20)]
        axes[0, 0].plot(sub['Excavation_Depth_m'], sub['Max_Horizontal_Displacement_mm'], 'o-', label=f't={wt}m', markersize=5)
    axes[0, 0].set_xlabel('Excavation Depth (m)'); axes[0, 0].set_ylabel('Max Horizontal Disp. (mm)')
    axes[0, 0].set_title('Effect of Wall Thickness on Displacement'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    for ps in prop_spacings:
        sub = df[(df['Prop_Spacing_m'] == ps) & (df['Wall_Thickness_m'] == 1.0) & (df['Excavation_Width_m'] == 20)]
        axes[0, 1].plot(sub['Excavation_Depth_m'], sub['Max_Horizontal_Displacement_mm'], 's-', label=f'Spacing={ps}m', markersize=5)
    axes[0, 1].set_xlabel('Excavation Depth (m)'); axes[0, 1].set_ylabel('Max Horizontal Disp. (mm)')
    axes[0, 1].set_title('Effect of Prop Spacing on Displacement'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    scatter = axes[1, 0].scatter(df['Factor_of_Safety'], df['Max_Horizontal_Displacement_mm'],
                                  c=df['Excavation_Depth_m'], cmap='RdYlGn_r', alpha=0.5, s=15, edgecolors='none')
    plt.colorbar(scatter, ax=axes[1, 0], label='Excavation Depth (m)')
    axes[1, 0].axvline(x=1.2, color='red', linestyle='--', label='FoS = 1.2')
    axes[1, 0].set_xlabel('Factor of Safety'); axes[1, 0].set_ylabel('Max Horizontal Disp. (mm)')
    axes[1, 0].set_title('FoS vs Displacement'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    fail_rate = df.groupby('Excavation_Depth_m')['Failure_Flag'].mean() * 100
    axes[1, 1].bar(fail_rate.index, fail_rate.values, color='#E91E63', alpha=0.8, width=1.2)
    axes[1, 1].set_xlabel('Excavation Depth (m)'); axes[1, 1].set_ylabel('Failure Rate (%)')
    axes[1, 1].set_title('Failure Rate by Excavation Depth'); axes[1, 1].grid(True, alpha=0.3, axis='y')

    fig.suptitle('Numerical Simulation: Deep Excavation in London Clay', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig03_deep_excavation.png")
    plt.close()
    return df


# ============================================================================
# DATASET 4: Geohazard Susceptibility - Ground Movement Potential
# ============================================================================
def generate_geohazard():
    print("Generating Dataset 4: Geohazard Susceptibility...")
    n = 2000
    lon = np.random.uniform(-2.5, 1.5, n)
    lat = np.random.uniform(51.0, 53.5, n)
    soil_types = np.random.choice(['Swelling_Clay', 'Compressible_Ground', 'Running_Sand', 'Firm_Clay', 'Dense_Sand'], n, p=[0.25, 0.2, 0.15, 0.25, 0.15])

    susceptibility_map = {
        'Swelling_Clay': {'shrink_swell': (0.6, 0.9), 'compress': (0.2, 0.5), 'collapse': (0.05, 0.2)},
        'Compressible_Ground': {'shrink_swell': (0.2, 0.5), 'compress': (0.6, 0.95), 'collapse': (0.3, 0.6)},
        'Running_Sand': {'shrink_swell': (0.01, 0.1), 'compress': (0.1, 0.4), 'collapse': (0.5, 0.9)},
        'Firm_Clay': {'shrink_swell': (0.3, 0.6), 'compress': (0.1, 0.3), 'collapse': (0.02, 0.15)},
        'Dense_Sand': {'shrink_swell': (0.01, 0.05), 'compress': (0.05, 0.2), 'collapse': (0.1, 0.35)},
    }

    records = []
    for i in range(n):
        st = soil_types[i]
        s = susceptibility_map[st]
        shrink_swell = np.random.uniform(*s['shrink_swell'])
        compressibility = np.random.uniform(*s['compress'])
        collapse_pot = np.random.uniform(*s['collapse'])
        overall = 0.4 * shrink_swell + 0.35 * compressibility + 0.25 * collapse_pot
        risk = 'Very_High' if overall > 0.7 else 'High' if overall > 0.5 else 'Moderate' if overall > 0.3 else 'Low'
        records.append({
            'Longitude': round(lon[i], 5),
            'Latitude': round(lat[i], 5),
            'Soil_Type': st,
            'Shrink_Swell_Potential': round(shrink_swell, 3),
            'Compressibility_Index': round(compressibility, 3),
            'Collapse_Potential': round(collapse_pot, 3),
            'Overall_Susceptibility': round(overall, 3),
            'Risk_Category': risk,
            'Groundwater_Depth_m': round(np.random.uniform(0.5, 15), 1),
            'Asset_Proximity_m': round(np.random.exponential(50), 1),
            'Pipe_Material': np.random.choice(['Cast_Iron', 'PVC', 'Steel', 'Concrete']),
            'Pipe_Age_years': np.random.randint(5, 120),
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/04_geohazard_susceptibility.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    risk_colors = {'Low': '#4CAF50', 'Moderate': '#FFC107', 'High': '#FF9800', 'Very_High': '#F44336'}
    for risk, color in risk_colors.items():
        sub = df[df['Risk_Category'] == risk]
        axes[0, 0].scatter(sub['Longitude'], sub['Latitude'], c=color, s=8, alpha=0.6, label=risk.replace('_', ' '))
    axes[0, 0].set_xlabel('Longitude'); axes[0, 0].set_ylabel('Latitude')
    axes[0, 0].set_title('Geohazard Risk Map'); axes[0, 0].legend(markerscale=3); axes[0, 0].grid(True, alpha=0.2)

    soil_order = ['Swelling_Clay', 'Compressible_Ground', 'Running_Sand', 'Firm_Clay', 'Dense_Sand']
    risk_order = ['Low', 'Moderate', 'High', 'Very_High']
    ct = pd.crosstab(df['Soil_Type'], df['Risk_Category']).reindex(columns=risk_order, fill_value=0)
    ct = ct.reindex(soil_order, fill_value=0)
    ct.plot(kind='bar', stacked=True, ax=axes[0, 1], color=[risk_colors[r] for r in risk_order])
    axes[0, 1].set_title('Risk Distribution by Soil Type'); axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=30)

    soil_palette = {'Swelling_Clay': '#8B4513', 'Compressible_Ground': '#696969',
                    'Running_Sand': '#DAA520', 'Firm_Clay': '#A0522D', 'Dense_Sand': '#F4A460'}
    for st in soil_order:
        sub = df[df['Soil_Type'] == st]
        axes[1, 0].scatter(sub['Shrink_Swell_Potential'], sub['Compressibility_Index'],
                           c=soil_palette[st], s=12, alpha=0.5, label=st.replace('_', ' '))
    axes[1, 0].set_xlabel('Shrink-Swell Potential'); axes[1, 0].set_ylabel('Compressibility Index')
    axes[1, 0].set_title('Shrink-Swell vs Compressibility'); axes[1, 0].legend(fontsize=8, markerscale=2); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(df['Overall_Susceptibility'], bins=40, color='#5C6BC0', alpha=0.8, edgecolor='white')
    axes[1, 1].axvline(x=0.3, color='green', linestyle='--', label='Low/Moderate')
    axes[1, 1].axvline(x=0.5, color='orange', linestyle='--', label='Moderate/High')
    axes[1, 1].axvline(x=0.7, color='red', linestyle='--', label='High/Very High')
    axes[1, 1].set_xlabel('Overall Susceptibility'); axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Susceptibility Distribution'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle('Geohazard Susceptibility: Ground Movement Potential (BGS-style)', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig04_geohazard_susceptibility.png")
    plt.close()
    return df


# ============================================================================
# DATASET 5: Environmental Context - Meteorological Data
# ============================================================================
def generate_meteorological():
    print("Generating Dataset 5: Meteorological Data...")
    dates = pd.date_range('2018-01-01', '2024-12-31', freq='D')
    n = len(dates)

    day_of_year = dates.dayofyear
    temp_seasonal = 10 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temperature = temp_seasonal + np.random.normal(0, 3, n)

    precip_seasonal = 2.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 300) / 365)
    precipitation = np.maximum(0, precip_seasonal * np.random.exponential(1, n))

    humidity = 70 + 15 * np.sin(2 * np.pi * (day_of_year - 300) / 365) + np.random.normal(0, 8, n)
    humidity = np.clip(humidity, 20, 100)

    wind_speed = np.abs(np.random.weibull(2.2, n) * 5)

    cumulative_precip_30d = pd.Series(precipitation).rolling(30, min_periods=1).sum().values
    evapotranspiration = np.maximum(0, 0.5 + 0.15 * temperature + np.random.normal(0, 0.3, n))
    water_balance = precipitation - evapotranspiration

    suction = 50 - 2 * water_balance
    suction = np.clip(suction, 0, 200)
    suction += np.random.normal(0, 5, n)
    suction = np.maximum(suction, 0)

    shrinkage_strain = 0.0005 * suction * (1 + 0.01 * temperature)
    shrinkage_strain += np.random.normal(0, 0.002, n)
    shrinkage_strain = np.maximum(shrinkage_strain, 0)

    embankment_risk = (0.3 * (cumulative_precip_30d / cumulative_precip_30d.max()) +
                       0.3 * (shrinkage_strain / max(shrinkage_strain.max(), 0.001)) +
                       0.2 * (np.clip(temperature, 0, 35) / 35) +
                       0.2 * (wind_speed / max(wind_speed.max(), 0.001)))
    embankment_risk = np.clip(embankment_risk, 0, 1)

    df = pd.DataFrame({
        'Date': dates,
        'Temperature_C': np.round(temperature, 1),
        'Precipitation_mm': np.round(precipitation, 2),
        'Relative_Humidity_pct': np.round(humidity, 1),
        'Wind_Speed_m_s': np.round(wind_speed, 1),
        'Cumulative_Precip_30d_mm': np.round(cumulative_precip_30d, 1),
        'Evapotranspiration_mm': np.round(evapotranspiration, 2),
        'Water_Balance_mm': np.round(water_balance, 2),
        'Soil_Suction_kPa': np.round(suction, 1),
        'Shrinkage_Strain': np.round(shrinkage_strain, 5),
        'Embankment_Deterioration_Index': np.round(embankment_risk, 4),
        'Soil_Type': 'Clay',
    })
    df.to_csv(f"{DATASETS_DIR}/05_meteorological_embankment_failure.csv", index=False)

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    monthly = df.set_index('Date').select_dtypes(include='number').resample('ME').mean()

    axes[0, 0].plot(monthly.index, monthly['Temperature_C'], color='#E91E63', linewidth=1.2)
    axes[0, 0].fill_between(monthly.index, monthly['Temperature_C'] - 3, monthly['Temperature_C'] + 3, alpha=0.15, color='#E91E63')
    axes[0, 0].set_title('Monthly Mean Temperature'); axes[0, 0].set_ylabel('Temperature (°C)'); axes[0, 0].grid(True, alpha=0.3)

    monthly_precip = df.set_index('Date').select_dtypes(include='number').resample('ME').sum()
    axes[0, 1].bar(monthly_precip.index, monthly_precip['Precipitation_mm'], width=25, color='#2196F3', alpha=0.7)
    axes[0, 1].set_title('Monthly Total Precipitation'); axes[0, 1].set_ylabel('Precipitation (mm)'); axes[0, 1].grid(True, alpha=0.3, axis='y')

    axes[1, 0].plot(monthly.index, monthly['Water_Balance_mm'], color='#009688', linewidth=1.2)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1, 0].fill_between(monthly.index, monthly['Water_Balance_mm'], alpha=0.3,
                             where=monthly['Water_Balance_mm'] > 0, color='blue', label='Surplus')
    axes[1, 0].fill_between(monthly.index, monthly['Water_Balance_mm'], alpha=0.3,
                             where=monthly['Water_Balance_mm'] < 0, color='red', label='Deficit')
    axes[1, 0].set_title('Water Balance'); axes[1, 0].set_ylabel('Water Balance (mm)'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(monthly.index, monthly['Soil_Suction_kPa'], color='#795548', linewidth=1.2)
    axes[1, 1].set_title('Soil Suction'); axes[1, 1].set_ylabel('Suction (kPa)'); axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(monthly.index, monthly['Shrinkage_Strain'] * 100, color='#FF5722', linewidth=1.2)
    axes[2, 0].set_title('Shrinkage Strain'); axes[2, 0].set_ylabel('Strain (%)'); axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(monthly.index, monthly['Embankment_Deterioration_Index'], color='#673AB7', linewidth=1.2)
    axes[2, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
    axes[2, 1].set_title('Embankment Deterioration Index'); axes[2, 1].set_ylabel('Index'); axes[2, 1].legend(); axes[2, 1].grid(True, alpha=0.3)

    fig.suptitle('Environmental Context: Meteorological Data for Clay Embankment Failure Prediction', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig05_meteorological_data.png")
    plt.close()
    return df


# ============================================================================
# DATASET 6: Soil Properties - Sandy and Clay Soils
# ============================================================================
def generate_soil_properties():
    print("Generating Dataset 6: Soil Properties...")

    # Sandy soil samples
    sandy_records = []
    for i in range(200):
        Dr = np.random.uniform(0.2, 0.95)
        D10 = np.random.uniform(0.05, 0.5)
        D50 = D10 * np.random.uniform(2.5, 8)
        D90 = D50 * np.random.uniform(2, 5)
        Cu_coeff = D50 / D10 * np.random.uniform(0.8, 1.5)
        phi = 28 + 15 * Dr + np.random.normal(0, 1.5)
        psi = max(0, phi - 30 + np.random.normal(0, 2))
        E = (10 + 40 * Dr) * 1000 + np.random.normal(0, 2000)
        nu = 0.25 + 0.1 * (1 - Dr) + np.random.normal(0, 0.02)
        gamma = 15 + 5 * Dr + np.random.normal(0, 0.3)
        emax = 0.9 - 0.15 * Dr + np.random.normal(0, 0.03)
        emin = 0.5 - 0.1 * Dr + np.random.normal(0, 0.02)
        k = 10 ** (np.random.uniform(-5, -2))
        sandy_records.append({
            'Sample_ID': f'SAND-{i+1:04d}',
            'Soil_Type': 'Sand',
            'Relative_Density_Dr': round(Dr, 3),
            'D10_mm': round(D10, 4),
            'D50_mm': round(D50, 4),
            'D90_mm': round(D90, 4),
            'Uniformity_Coefficient_Cu': round(Cu_coeff, 2),
            'Internal_Friction_Angle_deg': round(phi, 1),
            'Dilation_Angle_deg': round(max(psi, 0), 1),
            'Elastic_Modulus_kPa': round(max(E, 5000), 0),
            'Poissons_Ratio': round(np.clip(nu, 0.15, 0.4), 3),
            'Unit_Weight_kN_m3': round(gamma, 2),
            'Void_Ratio_max': round(emax, 3),
            'Void_Ratio_min': round(max(emin, 0.3), 3),
            'Permeability_m_s': f'{k:.2e}',
            'Cohesion_kPa': round(np.random.uniform(0, 5), 1),
        })

    # Clay soil samples
    clay_records = []
    for i in range(200):
        PI = np.random.uniform(10, 60)
        LL = 20 + PI + np.random.normal(0, 3)
        PL = LL - PI
        cu = np.random.uniform(15, 200)
        Cc = 0.009 * (LL - 10) + np.random.normal(0, 0.02)
        Cs = Cc / np.random.uniform(4, 8)
        E = 200 * cu + np.random.normal(0, 500)
        nu = 0.35 + 0.1 * (PI / 60) + np.random.normal(0, 0.02)
        gamma = 16 + 4 * (cu / 200) + np.random.normal(0, 0.4)
        phi_clay = 35 - 0.3 * PI + np.random.normal(0, 2)
        OCR = np.random.lognormal(0.5, 0.5)
        k = 10 ** (np.random.uniform(-10, -7))
        w = PL + 0.5 * (LL - PL) * np.random.uniform(0.3, 1.2)
        clay_records.append({
            'Sample_ID': f'CLAY-{i+1:04d}',
            'Soil_Type': 'Clay',
            'Undrained_Shear_Strength_kPa': round(cu, 1),
            'Plasticity_Index_PI': round(PI, 1),
            'Liquid_Limit_LL': round(LL, 1),
            'Plastic_Limit_PL': round(PL, 1),
            'Compression_Index_Cc': round(max(Cc, 0.05), 4),
            'Swelling_Index_Cs': round(max(Cs, 0.005), 4),
            'Elastic_Modulus_kPa': round(max(E, 3000), 0),
            'Poissons_Ratio': round(np.clip(nu, 0.3, 0.5), 3),
            'Unit_Weight_kN_m3': round(gamma, 2),
            'Friction_Angle_deg': round(max(phi_clay, 10), 1),
            'OCR': round(max(OCR, 1.0), 2),
            'Permeability_m_s': f'{k:.2e}',
            'Moisture_Content_pct': round(w, 1),
            'Cohesion_kPa': round(cu * 0.5 + np.random.normal(0, 5), 1),
        })

    df_sand = pd.DataFrame(sandy_records)
    df_clay = pd.DataFrame(clay_records)
    df_sand.to_csv(f"{DATASETS_DIR}/06a_soil_properties_sand.csv", index=False)
    df_clay.to_csv(f"{DATASETS_DIR}/06b_soil_properties_clay.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    axes[0, 0].scatter(df_sand['Relative_Density_Dr'], df_sand['Internal_Friction_Angle_deg'], s=15, alpha=0.6, c='#DAA520', edgecolors='none')
    axes[0, 0].set_xlabel('Relative Density ($D_r$)'); axes[0, 0].set_ylabel('Friction Angle (°)')
    axes[0, 0].set_title('Sand: $D_r$ vs $\\phi$'); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(df_sand['D50_mm'], df_sand['Elastic_Modulus_kPa'].astype(float) / 1000, s=15, alpha=0.6, c='#FF9800', edgecolors='none')
    axes[0, 1].set_xlabel('$D_{50}$ (mm)'); axes[0, 1].set_ylabel('Elastic Modulus (MPa)')
    axes[0, 1].set_title('Sand: Grain Size vs Stiffness'); axes[0, 1].grid(True, alpha=0.3)

    x_data = df_sand['Relative_Density_Dr'].values
    y_data = df_sand['Dilation_Angle_deg'].values
    axes[0, 2].scatter(x_data, y_data, s=15, alpha=0.6, c='#4CAF50', edgecolors='none')
    axes[0, 2].set_xlabel('Relative Density ($D_r$)'); axes[0, 2].set_ylabel('Dilation Angle (°)')
    axes[0, 2].set_title('Sand: $D_r$ vs Dilation Angle $\\psi$'); axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].scatter(df_clay['Plasticity_Index_PI'], df_clay['Undrained_Shear_Strength_kPa'], s=15, alpha=0.6, c='#8B4513', edgecolors='none')
    axes[1, 0].set_xlabel('Plasticity Index (PI)'); axes[1, 0].set_ylabel('$c_u$ (kPa)')
    axes[1, 0].set_title('Clay: PI vs Undrained Shear Strength'); axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(df_clay['Liquid_Limit_LL'], df_clay['Compression_Index_Cc'], s=15, alpha=0.6, c='#5C6BC0', edgecolors='none')
    axes[1, 1].set_xlabel('Liquid Limit (LL)'); axes[1, 1].set_ylabel('$C_c$')
    axes[1, 1].set_title('Clay: LL vs Compression Index'); axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].scatter(df_clay['Plasticity_Index_PI'], df_clay['Friction_Angle_deg'], s=15, alpha=0.6, c='#E91E63', edgecolors='none')
    axes[1, 2].set_xlabel('Plasticity Index (PI)'); axes[1, 2].set_ylabel("$\\phi'$ (°)")
    axes[1, 2].set_title("Clay: PI vs Effective Friction Angle"); axes[1, 2].grid(True, alpha=0.3)

    fig.suptitle('Soil Property Datasets: Sandy and Clay Soils', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig06_soil_properties.png")
    plt.close()
    return df_sand, df_clay


# ============================================================================
# DATASET 7: Tunnel Risk Dataset - Structural & Monitoring
# ============================================================================
def generate_tunnel_risk():
    print("Generating Dataset 7: Tunnel Risk Dataset...")
    n = 1200
    records = []
    for i in range(n):
        soil = np.random.choice(['Clay', 'Sand', 'Mixed_Clay_Sand'], p=[0.4, 0.35, 0.25])
        depth = np.random.uniform(5, 35)
        diameter = np.random.uniform(3, 12)
        hd_ratio = depth / diameter
        lining_thick = diameter * np.random.uniform(0.02, 0.06)

        if soil == 'Clay':
            cu = np.random.uniform(25, 180)
            phi = np.random.uniform(18, 28)
            K0 = np.random.uniform(0.8, 2.0)
            gwl = depth * np.random.uniform(0.3, 0.8)
        elif soil == 'Sand':
            cu = np.random.uniform(0, 8)
            phi = np.random.uniform(28, 42)
            K0 = np.random.uniform(0.35, 0.6)
            gwl = depth * np.random.uniform(0.1, 0.6)
        else:
            cu = np.random.uniform(10, 80)
            phi = np.random.uniform(22, 35)
            K0 = np.random.uniform(0.5, 1.2)
            gwl = depth * np.random.uniform(0.2, 0.7)

        volume_loss = 0.5 + 2 * (1 / (cu + 1)) * depth + np.random.normal(0, 0.3)
        volume_loss = np.clip(volume_loss, 0.3, 8)

        settlement = 0.3 * volume_loss * diameter ** 2 / (4 * depth) * 1000
        settlement += np.random.normal(0, 2)
        settlement = max(settlement, 0)

        lat_disp = settlement * np.random.uniform(0.2, 0.6)
        pore_pressure = 9.81 * gwl + np.random.normal(0, 10)

        crown_convergence = 0.001 * depth * diameter / lining_thick + np.random.normal(0, 0.5)
        crown_convergence = max(crown_convergence, 0)

        risk_score = (0.25 * min(volume_loss / 5, 1) +
                      0.25 * min(settlement / 50, 1) +
                      0.2 * min(depth / 30, 1) +
                      0.15 * (1 - min(cu / 150, 1)) +
                      0.15 * min(crown_convergence / 10, 1))

        failure = 1 if risk_score > 0.55 else 0

        records.append({
            'Tunnel_ID': f'TUN-{i+1:04d}',
            'Soil_Type': soil,
            'Tunnel_Depth_m': round(depth, 1),
            'Tunnel_Diameter_m': round(diameter, 2),
            'H_D_Ratio': round(hd_ratio, 2),
            'Lining_Thickness_m': round(lining_thick, 3),
            'Undrained_Shear_Strength_kPa': round(cu, 1),
            'Friction_Angle_deg': round(phi, 1),
            'K0': round(K0, 3),
            'Groundwater_Level_m': round(gwl, 1),
            'Volume_Loss_pct': round(volume_loss, 3),
            'Surface_Settlement_mm': round(settlement, 2),
            'Lateral_Displacement_mm': round(lat_disp, 2),
            'Pore_Water_Pressure_kPa': round(max(pore_pressure, 0), 1),
            'Crown_Convergence_mm': round(crown_convergence, 2),
            'Risk_Score': round(risk_score, 4),
            'Failure_Label': failure,
        })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/07_tunnel_risk_dataset.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    soil_colors = {'Clay': '#8B4513', 'Sand': '#DAA520', 'Mixed_Clay_Sand': '#708090'}

    for soil, color in soil_colors.items():
        sub = df[df['Soil_Type'] == soil]
        axes[0, 0].scatter(sub['Tunnel_Depth_m'], sub['Surface_Settlement_mm'], s=10, alpha=0.5, c=color, label=soil.replace('_', ' '))
    axes[0, 0].set_xlabel('Tunnel Depth (m)'); axes[0, 0].set_ylabel('Surface Settlement (mm)')
    axes[0, 0].set_title('Depth vs Settlement by Soil Type'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    for soil, color in soil_colors.items():
        sub = df[df['Soil_Type'] == soil]
        axes[0, 1].scatter(sub['Volume_Loss_pct'], sub['Surface_Settlement_mm'], s=10, alpha=0.5, c=color, label=soil.replace('_', ' '))
    axes[0, 1].set_xlabel('Volume Loss (%)'); axes[0, 1].set_ylabel('Surface Settlement (mm)')
    axes[0, 1].set_title('Volume Loss vs Settlement'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    scatter = axes[0, 2].scatter(df['H_D_Ratio'], df['Crown_Convergence_mm'], c=df['Risk_Score'], cmap='RdYlGn_r', s=10, alpha=0.6)
    plt.colorbar(scatter, ax=axes[0, 2], label='Risk Score')
    axes[0, 2].set_xlabel('H/D Ratio'); axes[0, 2].set_ylabel('Crown Convergence (mm)')
    axes[0, 2].set_title('H/D Ratio vs Crown Convergence'); axes[0, 2].grid(True, alpha=0.3)

    for soil, color in soil_colors.items():
        sub = df[df['Soil_Type'] == soil]
        axes[1, 0].hist(sub['Risk_Score'], bins=30, alpha=0.5, color=color, label=soil.replace('_', ' '))
    axes[1, 0].axvline(x=0.55, color='red', linestyle='--', label='Failure Threshold')
    axes[1, 0].set_xlabel('Risk Score'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Risk Score Distribution'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)

    fail_by_soil = df.groupby('Soil_Type')['Failure_Label'].mean() * 100
    axes[1, 1].bar(fail_by_soil.index, fail_by_soil.values, color=[soil_colors[s] for s in fail_by_soil.index], alpha=0.8)
    axes[1, 1].set_ylabel('Failure Rate (%)'); axes[1, 1].set_title('Failure Rate by Soil Type'); axes[1, 1].grid(True, alpha=0.3, axis='y')

    corr_cols = ['Tunnel_Depth_m', 'Tunnel_Diameter_m', 'Volume_Loss_pct', 'Surface_Settlement_mm', 'Crown_Convergence_mm', 'Risk_Score']
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 2], square=True,
                xticklabels=[c.replace('_', '\n') for c in corr_cols], yticklabels=[c.replace('_', '\n') for c in corr_cols])
    axes[1, 2].set_title('Correlation Matrix')

    fig.suptitle('Tunnel Risk Dataset: Structural & Monitoring Data', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig07_tunnel_risk.png")
    plt.close()
    return df


# ============================================================================
# DATASET 8: Centrifuge Modeling - Experimental Physical Modeling
# ============================================================================
def generate_centrifuge():
    print("Generating Dataset 8: Centrifuge Modeling...")
    g_levels = [40, 60, 80, 100]
    soil_types = ['Sand', 'Clay']
    moisture_contents = {'Sand': [5, 10, 15], 'Clay': [20, 30, 40]}
    loading_rates = [0.1, 0.5, 1.0]

    records = []
    test_id = 0
    for g in g_levels:
        for soil in soil_types:
            for mc in moisture_contents[soil]:
                for lr in loading_rates:
                    test_id += 1
                    n_timesteps = 50
                    for t in range(n_timesteps):
                        time_s = t * 60
                        scale_factor = g

                        if soil == 'Sand':
                            settlement = 0.5 * lr * t * (mc / 10) * (g / 80) * (1 - np.exp(-0.05 * t))
                            settlement += np.random.normal(0, settlement * 0.05 + 0.1)
                            lat_disp = settlement * np.random.uniform(0.15, 0.35)
                            pore_pressure = 9.81 * mc / 100 * g / 80 * t * 0.3 * np.exp(-0.02 * t)
                        else:
                            settlement = 0.3 * lr * t * (mc / 30) * (g / 80) ** 0.7 * np.log1p(t)
                            settlement += np.random.normal(0, settlement * 0.05 + 0.1)
                            lat_disp = settlement * np.random.uniform(0.3, 0.6)
                            pore_pressure = 15 + 5 * mc / 30 * g / 80 * np.log1p(t)

                        pore_pressure += np.random.normal(0, 2)

                        disp_x = lat_disp * np.random.uniform(0.8, 1.2)
                        disp_y = settlement
                        strain = np.sqrt(disp_x ** 2 + disp_y ** 2) / (100 * g / 80)

                        shear_band = 1 if (strain > 0.05 and t > 20) else 0

                        records.append({
                            'Test_ID': f'CEN-{test_id:03d}',
                            'G_Level': g,
                            'Soil_Type': soil,
                            'Moisture_Content_pct': mc,
                            'Loading_Rate_mm_min': lr,
                            'Time_s': time_s,
                            'Scale_Factor_N': scale_factor,
                            'Settlement_mm_model': round(max(settlement, 0), 3),
                            'Settlement_mm_prototype': round(max(settlement * scale_factor, 0), 1),
                            'Lateral_Displacement_mm': round(max(lat_disp, 0), 3),
                            'Pore_Water_Pressure_kPa': round(max(pore_pressure, 0), 2),
                            'PIV_Displacement_X_mm': round(disp_x, 3),
                            'PIV_Displacement_Y_mm': round(max(disp_y, 0), 3),
                            'Shear_Strain': round(max(strain, 0), 5),
                            'Shear_Band_Detected': shear_band,
                        })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/08_centrifuge_modeling.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for g in g_levels:
        sub = df[(df['G_Level'] == g) & (df['Soil_Type'] == 'Sand') & (df['Moisture_Content_pct'] == 10) & (df['Loading_Rate_mm_min'] == 0.5)]
        axes[0, 0].plot(sub['Time_s'] / 60, sub['Settlement_mm_model'], label=f'{g}g', linewidth=1.3)
    axes[0, 0].set_xlabel('Time (min)'); axes[0, 0].set_ylabel('Model Settlement (mm)')
    axes[0, 0].set_title('Sand: Settlement at Different g-Levels'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    for g in g_levels:
        sub = df[(df['G_Level'] == g) & (df['Soil_Type'] == 'Clay') & (df['Moisture_Content_pct'] == 30) & (df['Loading_Rate_mm_min'] == 0.5)]
        axes[0, 1].plot(sub['Time_s'] / 60, sub['Settlement_mm_model'], label=f'{g}g', linewidth=1.3)
    axes[0, 1].set_xlabel('Time (min)'); axes[0, 1].set_ylabel('Model Settlement (mm)')
    axes[0, 1].set_title('Clay: Settlement at Different g-Levels'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    for lr in loading_rates:
        sub = df[(df['G_Level'] == 80) & (df['Soil_Type'] == 'Sand') & (df['Moisture_Content_pct'] == 10) & (df['Loading_Rate_mm_min'] == lr)]
        axes[0, 2].plot(sub['Time_s'] / 60, sub['Pore_Water_Pressure_kPa'], label=f'Rate={lr} mm/min', linewidth=1.3)
    axes[0, 2].set_xlabel('Time (min)'); axes[0, 2].set_ylabel('PWP (kPa)')
    axes[0, 2].set_title('Sand 80g: Pore Pressure Evolution'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    # PIV-style displacement heatmap for one test
    test_sub = df[(df['Test_ID'] == 'CEN-013') & (df['Time_s'] == 2940)]
    if len(test_sub) == 0:
        test_sub = df[df['Test_ID'] == 'CEN-013'].tail(1)

    nx, ny = 20, 15
    x_grid = np.linspace(0, 200, nx)
    y_grid = np.linspace(0, 150, ny)
    X, Y = np.meshgrid(x_grid, y_grid)
    disp_magnitude = 5 * np.exp(-((X - 100) ** 2 / (2 * 40 ** 2) + (Y - 75) ** 2 / (2 * 30 ** 2)))
    disp_magnitude += np.random.normal(0, 0.3, disp_magnitude.shape)
    im = axes[1, 0].pcolormesh(X, Y, disp_magnitude, cmap='hot_r', shading='auto')
    plt.colorbar(im, ax=axes[1, 0], label='Displacement (mm)')
    axes[1, 0].set_xlabel('X (mm)'); axes[1, 0].set_ylabel('Y (mm)')
    axes[1, 0].set_title('PIV Displacement Field (Centrifuge)'); axes[1, 0].set_aspect('equal')

    # Shear strain field
    strain_field = np.gradient(disp_magnitude, axis=0) ** 2 + np.gradient(disp_magnitude, axis=1) ** 2
    strain_field = np.sqrt(strain_field)
    im2 = axes[1, 1].pcolormesh(X, Y, strain_field, cmap='YlOrRd', shading='auto')
    plt.colorbar(im2, ax=axes[1, 1], label='Shear Strain')
    axes[1, 1].set_xlabel('X (mm)'); axes[1, 1].set_ylabel('Y (mm)')
    axes[1, 1].set_title('PIV Shear Strain Field'); axes[1, 1].set_aspect('equal')

    shear_rate = df.groupby(['Soil_Type', 'G_Level'])['Shear_Band_Detected'].mean().unstack()
    shear_rate.plot(kind='bar', ax=axes[1, 2], colormap='viridis')
    axes[1, 2].set_xlabel('Soil Type'); axes[1, 2].set_ylabel('Shear Band Detection Rate')
    axes[1, 2].set_title('Shear Band Detection by Soil & g-Level'); axes[1, 2].grid(True, alpha=0.3, axis='y')
    axes[1, 2].tick_params(axis='x', rotation=0)

    fig.suptitle('Centrifuge Modeling: Experimental Physical Model Data', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig08_centrifuge_modeling.png")
    plt.close()
    return df


# ============================================================================
# DATASET 9: FEM Parametric Study - Synthetic Numerical Simulation
# ============================================================================
def generate_fem_parametric():
    print("Generating Dataset 9: FEM Parametric Study...")
    soil_types = ['Sand', 'Clay']
    structure_types = ['Tunnel', 'Basement', 'Retaining_Wall']
    hd_ratios = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    gw_conditions = ['Dry', 'Partially_Saturated', 'Fully_Saturated']

    records = []
    sim_id = 0
    for soil in soil_types:
        for struct in structure_types:
            for hd in hd_ratios:
                for gw in gw_conditions:
                    for rep in range(5):
                        sim_id += 1

                        if soil == 'Sand':
                            E = np.random.uniform(15000, 50000)
                            nu = np.random.uniform(0.25, 0.35)
                            c = np.random.uniform(0, 5)
                            phi = np.random.uniform(28, 42)
                            gamma = np.random.uniform(16, 20)
                        else:
                            E = np.random.uniform(5000, 30000)
                            nu = np.random.uniform(0.3, 0.45)
                            c = np.random.uniform(15, 100)
                            phi = np.random.uniform(18, 30)
                            gamma = np.random.uniform(16, 21)

                        sat_factor = {'Dry': 1.0, 'Partially_Saturated': 1.3, 'Fully_Saturated': 1.8}[gw]

                        max_disp = (gamma * hd * 10 / E * 1000) * sat_factor
                        max_disp *= {'Tunnel': 1.2, 'Basement': 0.8, 'Retaining_Wall': 1.5}[struct]
                        max_disp += np.random.normal(0, max_disp * 0.1)

                        settlement = max_disp * np.random.uniform(0.5, 1.2)
                        lateral = max_disp * np.random.uniform(0.2, 0.8)
                        heave = max_disp * np.random.uniform(0.05, 0.3) if soil == 'Clay' else 0

                        pore_pressure = gamma * hd * 10 * {'Dry': 0, 'Partially_Saturated': 0.3, 'Fully_Saturated': 0.6}[gw]
                        pore_pressure += np.random.normal(0, 5)

                        plastic_strain = max_disp / 1000 * np.random.uniform(0.5, 2)

                        failure_modes = []
                        if settlement > 30:
                            failure_modes.append('Excessive_Settlement')
                        if lateral > 20:
                            failure_modes.append('Lateral_Instability')
                        if heave > 10:
                            failure_modes.append('Heave')
                        if pore_pressure > 150 and soil == 'Sand':
                            failure_modes.append('Piping')
                        if plastic_strain > 0.05:
                            failure_modes.append('Structural_Buckling')

                        if not failure_modes:
                            failure_modes.append('No_Failure')

                        records.append({
                            'Simulation_ID': f'FEM-{sim_id:04d}',
                            'Soil_Type': soil,
                            'Structure_Type': struct,
                            'H_D_Ratio': hd,
                            'Groundwater_Condition': gw,
                            'Elastic_Modulus_kPa': round(E, 0),
                            'Poissons_Ratio': round(nu, 3),
                            'Cohesion_kPa': round(c, 1),
                            'Friction_Angle_deg': round(phi, 1),
                            'Unit_Weight_kN_m3': round(gamma, 2),
                            'Max_Displacement_mm': round(max(max_disp, 0.1), 2),
                            'Surface_Settlement_mm': round(max(settlement, 0), 2),
                            'Lateral_Displacement_mm': round(max(lateral, 0), 2),
                            'Heave_mm': round(max(heave, 0), 2),
                            'Pore_Water_Pressure_kPa': round(max(pore_pressure, 0), 1),
                            'Max_Plastic_Strain': round(max(plastic_strain, 0), 5),
                            'Primary_Failure_Mode': failure_modes[0],
                            'All_Failure_Modes': ';'.join(failure_modes),
                            'Failure_Detected': 0 if failure_modes == ['No_Failure'] else 1,
                        })

    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/09_fem_parametric_study.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    soil_colors = {'Sand': '#DAA520', 'Clay': '#8B4513'}

    for soil, color in soil_colors.items():
        sub = df[df['Soil_Type'] == soil]
        mean_disp = sub.groupby('H_D_Ratio')['Max_Displacement_mm'].mean()
        std_disp = sub.groupby('H_D_Ratio')['Max_Displacement_mm'].std()
        axes[0, 0].errorbar(mean_disp.index, mean_disp.values, yerr=std_disp.values, fmt='o-', color=color, label=soil, capsize=3, linewidth=1.5)
    axes[0, 0].set_xlabel('H/D Ratio'); axes[0, 0].set_ylabel('Max Displacement (mm)')
    axes[0, 0].set_title('H/D Ratio vs Maximum Displacement'); axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

    gw_colors = {'Dry': '#4CAF50', 'Partially_Saturated': '#FF9800', 'Fully_Saturated': '#2196F3'}
    for gw, color in gw_colors.items():
        sub = df[df['Groundwater_Condition'] == gw]
        mean_settle = sub.groupby('H_D_Ratio')['Surface_Settlement_mm'].mean()
        axes[0, 1].plot(mean_settle.index, mean_settle.values, 's-', color=color, label=gw.replace('_', ' '), linewidth=1.5, markersize=6)
    axes[0, 1].set_xlabel('H/D Ratio'); axes[0, 1].set_ylabel('Settlement (mm)')
    axes[0, 1].set_title('Groundwater Effect on Settlement'); axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

    struct_colors = {'Tunnel': '#E91E63', 'Basement': '#3F51B5', 'Retaining_Wall': '#009688'}
    for struct, color in struct_colors.items():
        sub = df[df['Structure_Type'] == struct]
        mean_disp = sub.groupby('H_D_Ratio')['Max_Displacement_mm'].mean()
        axes[0, 2].plot(mean_disp.index, mean_disp.values, '^-', color=color, label=struct.replace('_', ' '), linewidth=1.5, markersize=6)
    axes[0, 2].set_xlabel('H/D Ratio'); axes[0, 2].set_ylabel('Max Displacement (mm)')
    axes[0, 2].set_title('Structure Type Comparison'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

    fail_modes = df[df['Failure_Detected'] == 1]['Primary_Failure_Mode'].value_counts()
    fail_modes.plot(kind='barh', ax=axes[1, 0], color='#E91E63', alpha=0.8)
    axes[1, 0].set_xlabel('Count'); axes[1, 0].set_title('Failure Mode Distribution'); axes[1, 0].grid(True, alpha=0.3, axis='x')

    for soil, color in soil_colors.items():
        sub = df[df['Soil_Type'] == soil]
        axes[1, 1].scatter(sub['Max_Displacement_mm'], sub['Max_Plastic_Strain'] * 100, s=10, alpha=0.4, c=color, label=soil)
    axes[1, 1].set_xlabel('Max Displacement (mm)'); axes[1, 1].set_ylabel('Max Plastic Strain (%)')
    axes[1, 1].set_title('Displacement vs Plastic Strain'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

    ct = pd.crosstab(df['Structure_Type'], df['Groundwater_Condition'], values=df['Failure_Detected'], aggfunc='mean') * 100
    ct = ct[['Dry', 'Partially_Saturated', 'Fully_Saturated']]
    ct.plot(kind='bar', ax=axes[1, 2], color=[gw_colors[g] for g in ['Dry', 'Partially_Saturated', 'Fully_Saturated']])
    axes[1, 2].set_ylabel('Failure Rate (%)'); axes[1, 2].set_title('Failure Rate: Structure vs GW Condition')
    axes[1, 2].tick_params(axis='x', rotation=20); axes[1, 2].grid(True, alpha=0.3, axis='y')

    fig.suptitle('FEM Parametric Study: Underground Structure Failure Mechanisms', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig09_fem_parametric.png")
    plt.close()
    return df


# ============================================================================
# DATASET 10: Summary / Constitutive Properties Reference Table
# ============================================================================
def generate_summary_table():
    print("Generating Dataset 10: Summary Reference Table...")
    records = [
        {'Data_Category': 'Constitutive', 'Key_Variables': 'E, nu, c, phi', 'Typical_Source': 'Lab tests (Triaxial/Oedometer)', 'Soil_Type': 'Sand', 'E_kPa_range': '15000-50000', 'nu_range': '0.25-0.35', 'c_kPa_range': '0-5', 'phi_deg_range': '28-42'},
        {'Data_Category': 'Constitutive', 'Key_Variables': 'E, nu, c, phi', 'Typical_Source': 'Lab tests (Triaxial/Oedometer)', 'Soil_Type': 'Clay', 'E_kPa_range': '5000-30000', 'nu_range': '0.30-0.45', 'c_kPa_range': '15-200', 'phi_deg_range': '18-30'},
        {'Data_Category': 'Geometric', 'Key_Variables': 'Wall thickness, Tunnel diameter, Excavation depth', 'Typical_Source': 'Engineering design specs', 'Soil_Type': 'Both', 'E_kPa_range': '-', 'nu_range': '-', 'c_kPa_range': '-', 'phi_deg_range': '-'},
        {'Data_Category': 'Observation', 'Key_Variables': 'Settlement (S), Pore water pressure (u)', 'Typical_Source': 'Field sensors or Centrifuge tests', 'Soil_Type': 'Both', 'E_kPa_range': '-', 'nu_range': '-', 'c_kPa_range': '-', 'phi_deg_range': '-'},
        {'Data_Category': 'Risk/Mode', 'Key_Variables': 'Failure vs Non-failure labels', 'Typical_Source': 'Numerical Parametric Studies', 'Soil_Type': 'Both', 'E_kPa_range': '-', 'nu_range': '-', 'c_kPa_range': '-', 'phi_deg_range': '-'},
    ]
    df = pd.DataFrame(records)
    df.to_csv(f"{DATASETS_DIR}/10_summary_reference_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#37474F')
            cell.set_text_props(color='white', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#ECEFF1')
        cell.set_edgecolor('#B0BEC5')

    ax.set_title('Summary Table: Data Categories for Underground Structure Failure Analysis', fontsize=13, pad=20, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig10_summary_table.png")
    plt.close()
    return df


# ============================================================================
# MASTER FIGURE: Overview of All Failure Mechanisms
# ============================================================================
def generate_master_overview():
    print("Generating Master Overview Figure...")
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    # 1. Settlement comparison: Sand vs Clay
    ax1 = fig.add_subplot(gs[0, 0:2])
    depths = np.linspace(0, 30, 100)
    settle_sand = 0.5 * depths ** 1.3 + np.random.normal(0, 1, 100)
    settle_clay = 0.8 * depths ** 1.1 * np.log1p(depths * 0.1) + np.random.normal(0, 1, 100)
    ax1.plot(depths, settle_sand, color='#DAA520', linewidth=2, label='Sandy Soil')
    ax1.plot(depths, settle_clay, color='#8B4513', linewidth=2, label='Clay Soil')
    ax1.fill_between(depths, settle_sand - 3, settle_sand + 3, alpha=0.15, color='#DAA520')
    ax1.fill_between(depths, settle_clay - 3, settle_clay + 3, alpha=0.15, color='#8B4513')
    ax1.set_xlabel('Structure Depth (m)'); ax1.set_ylabel('Settlement (mm)')
    ax1.set_title('(a) Settlement vs Depth: Sand vs Clay'); ax1.legend(); ax1.grid(True, alpha=0.3)

    # 2. Failure mode pie charts
    ax2 = fig.add_subplot(gs[0, 2])
    sand_modes = ['Piping\n35%', 'Settlement\n25%', 'Collapse\n20%', 'Lateral\n15%', 'Other\n5%']
    sand_vals = [35, 25, 20, 15, 5]
    colors_pie = ['#F44336', '#FF9800', '#FFC107', '#4CAF50', '#9E9E9E']
    ax2.pie(sand_vals, labels=sand_modes, colors=colors_pie, autopct='', startangle=90, textprops={'fontsize': 8})
    ax2.set_title('(b) Sandy Soil Failures')

    ax3 = fig.add_subplot(gs[0, 3])
    clay_modes = ['Heave\n30%', 'Settlement\n28%', 'Buckling\n18%', 'Lateral\n14%', 'Other\n10%']
    clay_vals = [30, 28, 18, 14, 10]
    colors_pie2 = ['#2196F3', '#FF9800', '#E91E63', '#4CAF50', '#9E9E9E']
    ax3.pie(clay_vals, labels=clay_modes, colors=colors_pie2, autopct='', startangle=90, textprops={'fontsize': 8})
    ax3.set_title('(c) Clay Soil Failures')

    # 3. FoS vs H/D ratio
    ax4 = fig.add_subplot(gs[1, 0:2])
    hd = np.linspace(0.5, 3.5, 50)
    fos_sand = 3.5 * np.exp(-0.4 * hd) + 0.5 + np.random.normal(0, 0.1, 50)
    fos_clay = 2.8 * np.exp(-0.3 * hd) + 0.8 + np.random.normal(0, 0.1, 50)
    ax4.plot(hd, fos_sand, 'o-', color='#DAA520', markersize=3, label='Sand', linewidth=1.5)
    ax4.plot(hd, fos_clay, 's-', color='#8B4513', markersize=3, label='Clay', linewidth=1.5)
    ax4.axhline(y=1.0, color='red', linestyle='--', label='FoS = 1.0 (Failure)')
    ax4.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7, label='FoS = 1.5 (Threshold)')
    ax4.set_xlabel('H/D Ratio'); ax4.set_ylabel('Factor of Safety')
    ax4.set_title('(d) Factor of Safety vs Embedment Ratio'); ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)

    # 4. Pore pressure evolution
    ax5 = fig.add_subplot(gs[1, 2:4])
    time = np.linspace(0, 100, 200)
    pwp_sand = 50 * np.exp(-0.05 * time) * (1 + 0.3 * np.sin(0.2 * time)) + np.random.normal(0, 2, 200)
    pwp_clay = 80 * (1 - np.exp(-0.02 * time)) + 10 * np.sin(0.1 * time) + np.random.normal(0, 2, 200)
    ax5.plot(time, pwp_sand, color='#DAA520', linewidth=1.5, label='Sand (rapid dissipation)')
    ax5.plot(time, pwp_clay, color='#8B4513', linewidth=1.5, label='Clay (slow build-up)')
    ax5.set_xlabel('Time (days)'); ax5.set_ylabel('Pore Water Pressure (kPa)')
    ax5.set_title('(e) Pore Water Pressure Response'); ax5.legend(); ax5.grid(True, alpha=0.3)

    # 5. Displacement vectors
    ax6 = fig.add_subplot(gs[2, 0:2])
    x = np.linspace(-10, 10, 15)
    y = np.linspace(-10, 0, 10)
    X, Y = np.meshgrid(x, y)
    U = -0.5 * X * np.exp(-(X ** 2 + Y ** 2) / 30)
    V = -2 * np.exp(-(X ** 2 + (Y + 3) ** 2) / 20)
    magnitude = np.sqrt(U ** 2 + V ** 2)
    ax6.quiver(X, Y, U, V, magnitude, cmap='Reds', scale=15, alpha=0.8)
    ax6.set_xlabel('Horizontal Distance (m)'); ax6.set_ylabel('Depth (m)')
    ax6.set_title('(f) Displacement Field Around Structure'); ax6.set_aspect('equal')
    ax6.add_patch(plt.Rectangle((-2, -5), 4, 5, fill=True, facecolor='gray', edgecolor='black', linewidth=2, alpha=0.5))
    ax6.annotate('Structure', xy=(0, -2.5), ha='center', fontsize=9, fontweight='bold')

    # 6. Risk matrix
    ax7 = fig.add_subplot(gs[2, 2:4])
    risk_matrix = np.array([
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 5],
        [3, 4, 5, 5, 5],
        [4, 5, 5, 5, 5],
        [5, 5, 5, 5, 5],
    ])
    im = ax7.imshow(risk_matrix, cmap='RdYlGn_r', aspect='auto', origin='lower')
    ax7.set_xticks(range(5)); ax7.set_xticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'], fontsize=8)
    ax7.set_yticks(range(5)); ax7.set_yticklabels(['Very Low', 'Low', 'Medium', 'High', 'Very High'], fontsize=8)
    ax7.set_xlabel('Soil Vulnerability'); ax7.set_ylabel('Structure Vulnerability')
    ax7.set_title('(g) Risk Matrix: Combined Soil-Structure Assessment')
    for i in range(5):
        for j in range(5):
            ax7.text(j, i, str(risk_matrix[i, j]), ha='center', va='center', fontsize=12, fontweight='bold',
                     color='white' if risk_matrix[i, j] >= 4 else 'black')

    fig.suptitle('Failure Mechanisms of Underground Structures in Sandy and Clay Soils\n— Comprehensive Dataset Overview —',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig00_master_overview.png")
    plt.close()


# ============================================================================
# PACKAGE INTO ZIP
# ============================================================================
def create_zip():
    print("\nPackaging CSV files into zip...")
    zip_path = "underground_structure_failure_datasets.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in sorted(os.listdir(DATASETS_DIR)):
            if fname.endswith('.csv'):
                fpath = os.path.join(DATASETS_DIR, fname)
                zf.write(fpath, fname)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Created: {zip_path} ({size_mb:.2f} MB)")
    return zip_path


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Generating Fabricated Datasets:")
    print("Topic: Failure Mechanism of Underground Structures")
    print("       in Sandy and Clay Soils")
    print("=" * 70)

    df1 = generate_sinkhole_settlement()
    df2 = generate_embankment_monitoring()
    df3 = generate_deep_excavation()
    df4 = generate_geohazard()
    df5 = generate_meteorological()
    df6a, df6b = generate_soil_properties()
    df7 = generate_tunnel_risk()
    df8 = generate_centrifuge()
    df9 = generate_fem_parametric()
    df10 = generate_summary_table()
    generate_master_overview()
    zip_path = create_zip()

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nCSV files in: ./{DATASETS_DIR}/")
    print(f"Figures in:   ./{FIGURES_DIR}/")
    print(f"ZIP archive:  ./{zip_path}")
    print("\nDataset summary:")
    for name, df in [("01 Sinkhole Settlement", df1), ("02 Embankment Monitoring", df2),
                     ("03 Deep Excavation", df3), ("04 Geohazard Susceptibility", df4),
                     ("05 Meteorological", df5), ("06a Sand Properties", df6a),
                     ("06b Clay Properties", df6b), ("07 Tunnel Risk", df7),
                     ("08 Centrifuge Modeling", df8), ("09 FEM Parametric", df9),
                     ("10 Summary Table", df10)]:
        print(f"  {name}: {len(df)} rows x {len(df.columns)} columns")
