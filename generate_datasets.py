import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
os.makedirs('underground_structure_failure_datasets/csv_data', exist_ok=True)
os.makedirs('underground_structure_failure_datasets/figures', exist_ok=True)

print("Generating Dataset 1: Physical Model - Pipeline Leakage Sinkholes...")
# Dataset 1: Manmade sinkholes due to pipeline leakage
n_samples = 500
pipeline_data = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Time_hours': np.linspace(0, 240, n_samples),
    'Flow_Rate_L_per_min': np.random.uniform(5, 50, n_samples),
    'Settlement_mm': np.cumsum(np.random.exponential(0.5, n_samples)),
    'Sinkhole_Diameter_m': np.random.uniform(0.5, 5.0, n_samples),
    'Soil_Layer': np.random.choice(['Sandy Clay Top', 'Clay Middle', 'Sandy Clay Bottom'], n_samples),
    'Moisture_Content_percent': np.random.uniform(15, 35, n_samples),
    'Pipeline_Depth_m': np.random.uniform(1.5, 6.0, n_samples),
    'Pressure_kPa': np.random.uniform(50, 300, n_samples),
    'Cavity_Volume_m3': np.random.uniform(0.1, 15.0, n_samples),
    'Failure_Occurred': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
})
pipeline_data.to_csv('underground_structure_failure_datasets/csv_data/01_pipeline_leakage_sinkholes.csv', index=False)

print("Generating Dataset 2: Field Monitoring - Instrumented Embankment...")
# Dataset 2: Instrumented embankment on clay formation
n_samples = 400
embankment_data = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Depth_m': np.random.uniform(0, 30, n_samples),
    'P_Wave_Velocity_m_per_s': np.random.uniform(300, 1800, n_samples),
    'S_Wave_Velocity_m_per_s': np.random.uniform(150, 800, n_samples),
    'Extensometer_Displacement_mm': np.random.uniform(-50, 150, n_samples),
    'Pressure_Cell_kPa': np.random.uniform(0, 500, n_samples),
    'Loading_Stage': np.random.randint(1, 6, n_samples),
    'Clay_Type': np.random.choice(['Mudstone Formation', 'London Clay', 'Keuper Marl'], n_samples),
    'Undrained_Shear_Strength_kPa': np.random.uniform(20, 150, n_samples),
    'Plasticity_Index': np.random.uniform(15, 55, n_samples),
    'Void_Ratio': np.random.uniform(0.4, 1.2, n_samples),
    'Failure_Mode': np.random.choice(['None', 'Heave', 'Settlement', 'Lateral'], n_samples, p=[0.6, 0.15, 0.15, 0.1])
})
embankment_data.to_csv('underground_structure_failure_datasets/csv_data/02_instrumented_embankment.csv', index=False)

print("Generating Dataset 3: Numerical Simulation - Deep Excavation...")
# Dataset 3: Deep excavation in London Clay
n_samples = 600
excavation_data = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Excavation_Depth_m': np.random.uniform(5, 30, n_samples),
    'Excavation_Width_m': np.random.uniform(10, 50, n_samples),
    'Wall_Thickness_m': np.random.uniform(0.3, 1.5, n_samples),
    'Prop_Spacing_m': np.random.uniform(2, 8, n_samples),
    'Horizontal_Displacement_mm': np.random.uniform(0, 150, n_samples),
    'Max_Bending_Moment_kNm': np.random.uniform(50, 2000, n_samples),
    'Basal_Heave_mm': np.random.uniform(-10, 80, n_samples),
    'Ground_Settlement_mm': np.random.uniform(0, 100, n_samples),
    'Clay_Undrained_Strength_kPa': np.random.uniform(50, 200, n_samples),
    'Youngs_Modulus_MPa': np.random.uniform(10, 100, n_samples),
    'Poissons_Ratio': np.random.uniform(0.2, 0.35, n_samples),
    'Safety_Factor': np.random.uniform(0.8, 3.0, n_samples),
    'Failure_Risk': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples, p=[0.5, 0.3, 0.15, 0.05])
})
excavation_data.to_csv('underground_structure_failure_datasets/csv_data/03_deep_excavation_london_clay.csv', index=False)

print("Generating Dataset 4: Geohazard Susceptibility...")
# Dataset 4: BGS Corroded Asset Failure datasets
n_samples = 800
geohazard_data = pd.DataFrame({
    'Location_ID': range(1, n_samples + 1),
    'Latitude': np.random.uniform(50.5, 55.5, n_samples),
    'Longitude': np.random.uniform(-5.0, 2.0, n_samples),
    'Soil_Type': np.random.choice(['Swelling Clay', 'Compressible Ground', 'Running Sand', 'Mixed'], n_samples),
    'Ground_Movement_Potential': np.random.choice(['Very Low', 'Low', 'Moderate', 'High', 'Very High'], n_samples),
    'Shrink_Swell_Index': np.random.uniform(0, 5, n_samples),
    'Compressibility_Index_Cc': np.random.uniform(0.05, 0.5, n_samples),
    'Permeability_m_per_s': 10 ** np.random.uniform(-9, -4, n_samples),
    'Groundwater_Depth_m': np.random.uniform(0.5, 15, n_samples),
    'Asset_Age_years': np.random.randint(10, 100, n_samples),
    'Corrosion_Rate_mm_per_year': np.random.uniform(0.01, 0.5, n_samples),
    'Asset_Type': np.random.choice(['Pipeline', 'Foundation', 'Tunnel', 'Basement'], n_samples),
    'Failure_Probability': np.random.uniform(0, 1, n_samples)
})
geohazard_data.to_csv('underground_structure_failure_datasets/csv_data/04_geohazard_susceptibility.csv', index=False)

print("Generating Dataset 5: Environmental/Meteorological Data...")
# Dataset 5: Meteorological data for clay embankment failure prediction
start_date = datetime(2020, 1, 1)
n_days = 1095  # 3 years of data
met_data = pd.DataFrame({
    'Date': [start_date + timedelta(days=i) for i in range(n_days)],
    'Precipitation_mm': np.random.gamma(2, 2, n_days),
    'Temperature_C': 12 + 8 * np.sin(np.linspace(0, 6*np.pi, n_days)) + np.random.normal(0, 3, n_days),
    'Evapotranspiration_mm': np.random.uniform(0, 5, n_days),
    'Soil_Moisture_percent': np.random.uniform(10, 40, n_days),
    'Pore_Water_Pressure_kPa': np.random.uniform(-50, 150, n_days),
    'Matric_Suction_kPa': np.random.uniform(0, 300, n_days),
    'Embankment_Settlement_mm': np.cumsum(np.random.uniform(-0.1, 0.3, n_days)),
    'Crack_Width_mm': np.maximum(0, np.random.normal(1, 2, n_days)),
    'Alert_Level': np.random.choice(['Green', 'Yellow', 'Orange', 'Red'], n_days, p=[0.7, 0.2, 0.08, 0.02])
})
met_data.to_csv('underground_structure_failure_datasets/csv_data/05_meteorological_embankment.csv', index=False)

print("Generating Dataset 6: Soil Properties - Sandy Soil...")
# Dataset 6a: Sandy Soil Properties
n_samples = 300
sandy_soil = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Relative_Density_percent': np.random.uniform(30, 95, n_samples),
    'D10_mm': np.random.uniform(0.05, 0.5, n_samples),
    'D50_mm': np.random.uniform(0.2, 2.0, n_samples),
    'D90_mm': np.random.uniform(1.0, 5.0, n_samples),
    'Uniformity_Coefficient_Cu': np.random.uniform(1.5, 15, n_samples),
    'Coefficient_of_Curvature_Cc': np.random.uniform(0.8, 3.0, n_samples),
    'Internal_Friction_Angle_degrees': np.random.uniform(28, 42, n_samples),
    'Dilation_Angle_degrees': np.random.uniform(0, 15, n_samples),
    'Dry_Density_kg_per_m3': np.random.uniform(1400, 1900, n_samples),
    'Saturated_Density_kg_per_m3': np.random.uniform(1800, 2200, n_samples),
    'Permeability_m_per_s': 10 ** np.random.uniform(-5, -3, n_samples),
    'Cohesion_kPa': np.random.uniform(0, 5, n_samples),
    'Youngs_Modulus_MPa': np.random.uniform(10, 80, n_samples),
    'Poissons_Ratio': np.random.uniform(0.15, 0.35, n_samples),
    'Classification': np.random.choice(['SP', 'SW', 'SM', 'SC'], n_samples)
})
sandy_soil.to_csv('underground_structure_failure_datasets/csv_data/06a_sandy_soil_properties.csv', index=False)

print("Generating Dataset 6b: Soil Properties - Clay Soil...")
# Dataset 6b: Clay Soil Properties
n_samples = 300
clay_soil = pd.DataFrame({
    'Sample_ID': range(1, n_samples + 1),
    'Liquid_Limit_percent': np.random.uniform(30, 90, n_samples),
    'Plastic_Limit_percent': np.random.uniform(15, 40, n_samples),
    'Plasticity_Index': np.random.uniform(10, 60, n_samples),
    'Clay_Fraction_percent': np.random.uniform(20, 70, n_samples),
    'Undrained_Shear_Strength_kPa': np.random.uniform(15, 200, n_samples),
    'Effective_Cohesion_kPa': np.random.uniform(0, 30, n_samples),
    'Effective_Friction_Angle_degrees': np.random.uniform(18, 32, n_samples),
    'Compression_Index_Cc': np.random.uniform(0.1, 0.6, n_samples),
    'Swelling_Index_Cs': np.random.uniform(0.01, 0.1, n_samples),
    'Coefficient_of_Consolidation_m2_per_year': 10 ** np.random.uniform(-2, 1, n_samples),
    'Overconsolidation_Ratio': np.random.uniform(1.0, 8.0, n_samples),
    'Sensitivity': np.random.uniform(1, 15, n_samples),
    'Natural_Moisture_Content_percent': np.random.uniform(20, 60, n_samples),
    'Dry_Density_kg_per_m3': np.random.uniform(1200, 1700, n_samples),
    'Saturated_Density_kg_per_m3': np.random.uniform(1700, 2100, n_samples),
    'Permeability_m_per_s': 10 ** np.random.uniform(-10, -7, n_samples),
    'Youngs_Modulus_MPa': np.random.uniform(5, 50, n_samples),
    'Poissons_Ratio': np.random.uniform(0.25, 0.45, n_samples),
    'Classification': np.random.choice(['CL', 'CH', 'CI', 'CV'], n_samples)
})
clay_soil.to_csv('underground_structure_failure_datasets/csv_data/06b_clay_soil_properties.csv', index=False)

print("Generating Dataset 7: Structural Monitoring - Tunnel Risk...")
# Dataset 7: Structural & Monitoring Datasets (Tunnel Risk)
n_samples = 1000
tunnel_data = pd.DataFrame({
    'Tunnel_ID': range(1, n_samples + 1),
    'Tunnel_Depth_m': np.random.uniform(5, 50, n_samples),
    'Tunnel_Diameter_m': np.random.uniform(3, 12, n_samples),
    'Soil_Type': np.random.choice(['Clay', 'Sand', 'Mixed Clay-Sand', 'Rock'], n_samples),
    'Cover_Depth_Ratio': np.random.uniform(1.0, 8.0, n_samples),
    'Lining_Thickness_m': np.random.uniform(0.2, 1.0, n_samples),
    'Vertical_Settlement_mm': np.random.uniform(-5, 80, n_samples),
    'Horizontal_Displacement_mm': np.random.uniform(-20, 50, n_samples),
    'Crown_Settlement_mm': np.random.uniform(0, 100, n_samples),
    'Invert_Heave_mm': np.random.uniform(-10, 40, n_samples),
    'Earth_Pressure_kPa': np.random.uniform(50, 800, n_samples),
    'Bending_Moment_kNm': np.random.uniform(0, 1500, n_samples),
    'Axial_Force_kN': np.random.uniform(100, 5000, n_samples),
    'Groundwater_Level_m': np.random.uniform(0, 20, n_samples),
    'Construction_Method': np.random.choice(['TBM', 'NATM', 'Cut-and-Cover', 'Pipe-Jacking'], n_samples),
    'Service_Years': np.random.randint(0, 100, n_samples),
    'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
    'Failure_Mode': np.random.choice(['None', 'Excessive Settlement', 'Lining Crack', 'Water Ingress', 'Collapse'], 
                                     n_samples, p=[0.7, 0.15, 0.08, 0.05, 0.02])
})
tunnel_data.to_csv('underground_structure_failure_datasets/csv_data/07_tunnel_monitoring.csv', index=False)

print("Generating Dataset 8: Experimental Physical Modeling...")
# Dataset 8: Experimental/Physical Modeling (Centrifuge Tests)
n_samples = 250
centrifuge_data = pd.DataFrame({
    'Test_ID': range(1, n_samples + 1),
    'Centrifuge_g_Level': np.random.choice([20, 40, 60, 80, 100], n_samples),
    'Model_Scale': np.random.choice(['1:20', '1:40', '1:60', '1:80', '1:100'], n_samples),
    'Soil_Type': np.random.choice(['Dry Sand', 'Saturated Sand', 'Clay', 'Silty Clay'], n_samples),
    'Structure_Type': np.random.choice(['Tunnel', 'Foundation', 'Retaining Wall', 'Pile'], n_samples),
    'Embedment_Depth_mm': np.random.uniform(50, 300, n_samples),
    'Moisture_Content_percent': np.random.uniform(0, 35, n_samples),
    'Applied_Load_kN': np.random.uniform(0, 50, n_samples),
    'Settlement_mm': np.random.uniform(0, 30, n_samples),
    'Lateral_Displacement_mm': np.random.uniform(-15, 25, n_samples),
    'Tilt_Angle_degrees': np.random.uniform(-5, 5, n_samples),
    'Bearing_Capacity_kPa': np.random.uniform(50, 500, n_samples),
    'Failure_Load_kN': np.random.uniform(10, 60, n_samples),
    'PIV_Max_Shear_Strain_percent': np.random.uniform(0, 15, n_samples),
    'Shear_Band_Thickness_mm': np.random.uniform(2, 20, n_samples),
    'Failure_Mechanism': np.random.choice(['General Shear', 'Local Shear', 'Punching', 'Bearing', 'Piping'], n_samples)
})
centrifuge_data.to_csv('underground_structure_failure_datasets/csv_data/08_centrifuge_physical_modeling.csv', index=False)

print("Generating Dataset 9: Synthetic FEM Parametric Study...")
# Dataset 9: Synthetic Datasets (Numerical Simulation - FEM Parametric)
n_samples = 1500
fem_data = pd.DataFrame({
    'Simulation_ID': range(1, n_samples + 1),
    'Soil_Type': np.random.choice(['Sand', 'Clay', 'Mixed'], n_samples),
    'Structure_Type': np.random.choice(['Tunnel', 'Basement', 'Foundation', 'Retaining Wall'], n_samples),
    'Embedment_Depth_m': np.random.uniform(2, 30, n_samples),
    'Structure_Width_m': np.random.uniform(3, 25, n_samples),
    'HD_Ratio': np.random.uniform(0.5, 5.0, n_samples),
    'Groundwater_Condition': np.random.choice(['Dry', 'Saturated', 'Partially Saturated'], n_samples),
    'Cohesion_kPa': np.random.uniform(0, 100, n_samples),
    'Friction_Angle_degrees': np.random.uniform(0, 40, n_samples),
    'Youngs_Modulus_MPa': np.random.uniform(5, 100, n_samples),
    'Poissons_Ratio': np.random.uniform(0.15, 0.45, n_samples),
    'Unit_Weight_kN_per_m3': np.random.uniform(16, 22, n_samples),
    'Permeability_m_per_s': 10 ** np.random.uniform(-10, -3, n_samples),
    'Applied_Surcharge_kPa': np.random.uniform(0, 100, n_samples),
    'Max_Vertical_Displacement_mm': np.random.uniform(-10, 150, n_samples),
    'Max_Horizontal_Displacement_mm': np.random.uniform(-30, 100, n_samples),
    'Max_Shear_Strain_percent': np.random.uniform(0, 20, n_samples),
    'Max_Principal_Stress_kPa': np.random.uniform(50, 1000, n_samples),
    'Min_Safety_Factor': np.random.uniform(0.5, 4.0, n_samples),
    'Pore_Water_Pressure_kPa': np.random.uniform(-100, 300, n_samples),
    'Failure_Mode': np.random.choice(['None', 'Heave', 'Piping', 'Structural Buckling', 
                                      'Bearing Failure', 'Excessive Settlement'], n_samples, 
                                     p=[0.55, 0.12, 0.08, 0.10, 0.08, 0.07]),
    'Failure_Occurred': np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
})
fem_data.to_csv('underground_structure_failure_datasets/csv_data/09_synthetic_fem_parametric.csv', index=False)

print("Generating Dataset 10: Summary Statistics Table...")
# Dataset 10: Summary table combining all key variables
summary_data = pd.DataFrame({
    'Data_Category': ['Constitutive', 'Constitutive', 'Constitutive', 'Constitutive', 'Constitutive',
                      'Geometric', 'Geometric', 'Geometric', 'Geometric',
                      'Observational', 'Observational', 'Observational', 'Observational',
                      'Risk/Mode', 'Risk/Mode', 'Risk/Mode'],
    'Key_Variable': ['Youngs_Modulus_E_MPa', 'Poissons_Ratio_nu', 'Cohesion_c_kPa', 'Friction_Angle_phi_degrees',
                    'Unit_Weight_kN_per_m3', 'Wall_Thickness_m', 'Tunnel_Diameter_m', 'Embedment_Depth_m',
                    'Excavation_Width_m', 'Settlement_S_mm', 'Horizontal_Displacement_mm', 
                    'Pore_Water_Pressure_u_kPa', 'Earth_Pressure_kPa',
                    'Safety_Factor', 'Failure_Probability', 'Failure_Mode'],
    'Typical_Range': ['5-100', '0.15-0.45', '0-100', '0-42', '16-22',
                     '0.2-1.5', '3-12', '2-30', '10-50',
                     '0-150', '-30-100', '-100-300', '50-800',
                     '0.5-4.0', '0-1', 'Categorical'],
    'Unit': ['MPa', 'dimensionless', 'kPa', 'degrees', 'kN/m³',
            'meters', 'meters', 'meters', 'meters',
            'mm', 'mm', 'kPa', 'kPa',
            'dimensionless', 'probability', 'text'],
    'Typical_Source': ['Lab tests (Triaxial/Oedometer)', 'Lab tests', 'Direct Shear/Triaxial', 
                      'Triaxial Test', 'Lab measurement',
                      'Engineering design specs', 'Design specifications', 'Site survey', 'Design drawings',
                      'Field sensors/Inclinometers', 'Inclinometers/Total Station', 
                      'Piezometers', 'Pressure cells',
                      'Numerical Analysis', 'Statistical/ML Model', 'Field observation/FEM']
})
summary_data.to_csv('underground_structure_failure_datasets/csv_data/10_summary_parameters.csv', index=False)

print("\nAll CSV datasets generated successfully!")
print(f"Total datasets created: 11 CSV files")
print(f"Location: underground_structure_failure_datasets/csv_data/")
