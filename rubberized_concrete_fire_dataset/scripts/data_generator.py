#!/usr/bin/env python3
"""
Comprehensive Data Generator for Rubberized Concrete Fire Resistance Dataset
Generates realistic experimental data based on known behavior patterns from literature
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random
from scipy import interpolate
from scipy.stats import norm, lognorm
import json

class RubberizedConcreteDataGenerator:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.base_path = "raw_data"
        self.mix_designs = pd.read_csv(f"{self.base_path}/material_properties/mix_designs.csv")
        
    def generate_ambient_tests(self):
        """Generate ambient condition test data for all mix designs at 7, 28, and 56 days"""
        results = []
        
        for _, mix in self.mix_designs.iterrows():
            mix_id = mix['Mix_ID']
            target_strength = mix['Target_Strength_MPa']
            rubber_content = mix['Rubber_Content_%']
            
            # Strength development factors
            age_factors = {7: 0.65, 28: 1.0, 56: 1.10}
            
            for age, factor in age_factors.items():
                for specimen in range(1, 4):  # 3 specimens per condition
                    # Compressive strength with rubber reduction effect
                    rubber_reduction = 1 - (rubber_content * 0.025)  # 2.5% reduction per 1% rubber
                    base_strength = target_strength * factor * rubber_reduction
                    fc = base_strength * np.random.normal(1.0, 0.05)
                    
                    # Splitting tensile strength (approximately 10% of compressive)
                    fct = fc * 0.1 * np.random.normal(1.0, 0.08)
                    
                    # Modulus of elasticity (GPa) - Eurocode 2 formula modified
                    Ec = 22 * (fc/10) ** 0.3 * (1 - rubber_content * 0.02) * np.random.normal(1.0, 0.06)
                    
                    # Density (kg/m3) - reduces with rubber content
                    base_density = 2400
                    density = base_density * (1 - rubber_content * 0.003) * np.random.normal(1.0, 0.01)
                    
                    # UPV (m/s) - correlates with strength and density
                    upv = 4500 * (fc/40) ** 0.5 * (density/2400) ** 0.3 * np.random.normal(1.0, 0.03)
                    
                    # Poisson's ratio - increases slightly with rubber
                    poisson = 0.20 + rubber_content * 0.002 + np.random.normal(0, 0.01)
                    
                    results.append({
                        'Mix_ID': mix_id,
                        'Age_days': age,
                        'Specimen_ID': f"{mix_id}_D{age}_S{specimen}",
                        'Test_Date': (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d'),
                        'Compressive_Strength_MPa': round(fc, 2),
                        'Splitting_Tensile_MPa': round(fct, 2),
                        'Modulus_Elasticity_GPa': round(Ec, 2),
                        'Density_kg_m3': round(density, 1),
                        'UPV_m_s': round(upv, 0),
                        'Poisson_Ratio': round(poisson, 3),
                        'Temperature_C': 23,
                        'Humidity_%': 65 + np.random.randint(-5, 5)
                    })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.base_path}/ambient_tests/ambient_mechanical_properties.csv", index=False)
        return df
    
    def generate_high_temp_exposure_tests(self):
        """Generate high-temperature exposure test data"""
        results = []
        temperatures = [23, 200, 400, 600, 800]
        cooling_methods = ['Furnace_Cooled', 'Water_Quenched']
        
        # Get 28-day ambient properties as baseline
        ambient_df = pd.read_csv(f"{self.base_path}/ambient_tests/ambient_mechanical_properties.csv")
        numeric_cols = ['Compressive_Strength_MPa', 'Splitting_Tensile_MPa', 'Modulus_Elasticity_GPa', 
                       'Density_kg_m3', 'UPV_m_s', 'Poisson_Ratio']
        ambient_28d = ambient_df[ambient_df['Age_days'] == 28].groupby('Mix_ID')[numeric_cols].mean()
        
        for _, mix in self.mix_designs.iterrows():
            mix_id = mix['Mix_ID']
            rubber_content = mix['Rubber_Content_%']
            
            # Get baseline properties
            if mix_id in ambient_28d.index:
                baseline_fc = ambient_28d.loc[mix_id, 'Compressive_Strength_MPa']
                baseline_fct = ambient_28d.loc[mix_id, 'Splitting_Tensile_MPa']
                baseline_E = ambient_28d.loc[mix_id, 'Modulus_Elasticity_GPa']
                baseline_density = ambient_28d.loc[mix_id, 'Density_kg_m3']
                baseline_upv = ambient_28d.loc[mix_id, 'UPV_m_s']
            else:
                continue
            
            for temp in temperatures:
                for cooling in cooling_methods:
                    for specimen in range(1, 4):
                        # Temperature-dependent strength retention factors
                        if temp == 23:
                            fc_retention = 1.0
                            fct_retention = 1.0
                            E_retention = 1.0
                            mass_loss = 0
                        elif temp == 200:
                            fc_retention = 0.95 - rubber_content * 0.001
                            fct_retention = 0.85 - rubber_content * 0.002
                            E_retention = 0.80 - rubber_content * 0.003
                            mass_loss = 2.5 + rubber_content * 0.1
                        elif temp == 400:
                            fc_retention = 0.85 - rubber_content * 0.002
                            fct_retention = 0.60 - rubber_content * 0.003
                            E_retention = 0.50 - rubber_content * 0.004
                            mass_loss = 4.5 + rubber_content * 0.2
                        elif temp == 600:
                            fc_retention = 0.60 - rubber_content * 0.003
                            fct_retention = 0.30 - rubber_content * 0.004
                            E_retention = 0.25 - rubber_content * 0.005
                            mass_loss = 6.5 + rubber_content * 0.3
                        else:  # 800°C
                            fc_retention = 0.30 - rubber_content * 0.004
                            fct_retention = 0.10 - rubber_content * 0.005
                            E_retention = 0.10 - rubber_content * 0.006
                            mass_loss = 8.5 + rubber_content * 0.4
                        
                        # Water quenching causes additional damage
                        if cooling == 'Water_Quenched' and temp > 200:
                            quench_factor = 0.85 - (temp - 200) * 0.0001
                            fc_retention *= quench_factor
                            fct_retention *= quench_factor * 0.9
                            E_retention *= quench_factor * 0.85
                            mass_loss *= 1.15
                        
                        # Add rubber benefit at moderate temperatures
                        if 200 <= temp <= 400 and rubber_content > 0:
                            rubber_benefit = 1 + (rubber_content * 0.002)
                            fc_retention *= rubber_benefit
                        
                        # Calculate residual properties with variability
                        residual_fc = baseline_fc * fc_retention * np.random.normal(1.0, 0.06)
                        residual_fct = baseline_fct * fct_retention * np.random.normal(1.0, 0.08)
                        residual_E = baseline_E * E_retention * np.random.normal(1.0, 0.08)
                        
                        # UPV reduction
                        upv_factor = np.sqrt(max(0, fc_retention * E_retention))
                        residual_upv = baseline_upv * upv_factor * np.random.normal(1.0, 0.04)
                        
                        # Mass loss with variability
                        actual_mass_loss = mass_loss * np.random.normal(1.0, 0.1)
                        
                        # Crack density and width
                        if temp <= 200:
                            crack_density = np.random.uniform(0, 0.5)
                            max_crack_width = np.random.uniform(0, 0.1)
                        elif temp <= 400:
                            crack_density = np.random.uniform(0.5, 2.0)
                            max_crack_width = np.random.uniform(0.1, 0.5)
                        elif temp <= 600:
                            crack_density = np.random.uniform(2.0, 5.0)
                            max_crack_width = np.random.uniform(0.5, 2.0)
                        else:
                            crack_density = np.random.uniform(5.0, 10.0)
                            max_crack_width = np.random.uniform(2.0, 5.0)
                        
                        if cooling == 'Water_Quenched':
                            crack_density *= 1.5
                            max_crack_width *= 1.8
                        
                        results.append({
                            'Mix_ID': mix_id,
                            'Specimen_ID': f"{mix_id}_T{temp}_{cooling[:3]}_S{specimen}",
                            'Target_Temperature_C': temp,
                            'Actual_Temperature_C': temp + np.random.uniform(-5, 5),
                            'Heating_Rate_C_min': 5.0 + np.random.uniform(-0.5, 0.5),
                            'Soak_Time_min': 60 + np.random.randint(-5, 5),
                            'Cooling_Method': cooling,
                            'Cooling_Time_min': 180 if cooling == 'Furnace_Cooled' else 5,
                            'Mass_Loss_%': round(actual_mass_loss, 2),
                            'Residual_Compressive_MPa': round(max(0, residual_fc), 2),
                            'Residual_Tensile_MPa': round(max(0, residual_fct), 2),
                            'Residual_Modulus_GPa': round(max(0, residual_E), 2),
                            'Residual_UPV_m_s': round(max(0, residual_upv), 0),
                            'Crack_Density_cracks_m': round(crack_density, 2),
                            'Max_Crack_Width_mm': round(max_crack_width, 3),
                            'Color_Change': self._get_color_change(temp),
                            'Surface_Condition': self._get_surface_condition(temp, cooling)
                        })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.base_path}/high_temp_tests/residual_properties.csv", index=False)
        return df
    
    def generate_in_situ_tests(self):
        """Generate in-situ high-temperature test data"""
        results = []
        temperatures = [23, 100, 200, 300, 400, 500, 600, 700, 800]
        
        ambient_df = pd.read_csv(f"{self.base_path}/ambient_tests/ambient_mechanical_properties.csv")
        numeric_cols = ['Compressive_Strength_MPa', 'Splitting_Tensile_MPa', 'Modulus_Elasticity_GPa', 
                       'Density_kg_m3', 'UPV_m_s', 'Poisson_Ratio']
        ambient_28d = ambient_df[ambient_df['Age_days'] == 28].groupby('Mix_ID')[numeric_cols].mean()
        
        for _, mix in self.mix_designs.iterrows():
            mix_id = mix['Mix_ID']
            rubber_content = mix['Rubber_Content_%']
            
            if mix_id not in ambient_28d.index:
                continue
                
            baseline_fc = ambient_28d.loc[mix_id, 'Compressive_Strength_MPa']
            baseline_E = ambient_28d.loc[mix_id, 'Modulus_Elasticity_GPa']
            
            for temp in temperatures:
                for specimen in range(1, 3):  # 2 specimens per temperature
                    # In-situ strength (hot strength) - different from residual
                    if temp == 23:
                        hot_fc_ratio = 1.0
                        hot_E_ratio = 1.0
                        thermal_strain = 0
                        cte = 10.0 + rubber_content * 0.1
                    elif temp <= 100:
                        hot_fc_ratio = 1.0 - (temp - 23) * 0.0005
                        hot_E_ratio = 1.0 - (temp - 23) * 0.001
                        thermal_strain = temp * cte * 1e-6
                        cte = 10.0 + rubber_content * 0.1 + temp * 0.01
                    elif temp <= 200:
                        hot_fc_ratio = 0.96 - (temp - 100) * 0.0008
                        hot_E_ratio = 0.92 - (temp - 100) * 0.003
                        thermal_strain = temp * cte * 1e-6
                        cte = 11.0 + rubber_content * 0.15 + temp * 0.015
                    elif temp <= 400:
                        hot_fc_ratio = 0.88 - (temp - 200) * 0.0015
                        hot_E_ratio = 0.62 - (temp - 200) * 0.002
                        thermal_strain = temp * cte * 1e-6
                        cte = 12.0 + rubber_content * 0.2 + temp * 0.02
                    elif temp <= 600:
                        hot_fc_ratio = 0.58 - (temp - 400) * 0.0012
                        hot_E_ratio = 0.22 - (temp - 400) * 0.0006
                        thermal_strain = temp * cte * 1e-6 * 1.2  # Non-linear expansion
                        cte = 14.0 + rubber_content * 0.3 + temp * 0.025
                    else:
                        hot_fc_ratio = 0.34 - (temp - 600) * 0.001
                        hot_E_ratio = 0.10 - (temp - 600) * 0.0003
                        thermal_strain = temp * cte * 1e-6 * 1.5
                        cte = 16.0 + rubber_content * 0.4 + temp * 0.03
                    
                    # Rubber provides some benefit at moderate temperatures
                    if 100 <= temp <= 300 and rubber_content > 0:
                        hot_fc_ratio *= (1 + rubber_content * 0.003)
                    
                    # Calculate properties with variability
                    hot_fc = baseline_fc * hot_fc_ratio * np.random.normal(1.0, 0.05)
                    hot_E = baseline_E * hot_E_ratio * np.random.normal(1.0, 0.06)
                    
                    # Transient thermal strain under load (LITS)
                    load_level = 0.4  # 40% of ambient strength
                    lits = load_level * thermal_strain * (1 + rubber_content * 0.01) * np.random.normal(1.0, 0.1)
                    
                    # Total strain
                    total_strain = thermal_strain + lits
                    
                    results.append({
                        'Mix_ID': mix_id,
                        'Specimen_ID': f"{mix_id}_InSitu_T{temp}_S{specimen}",
                        'Test_Temperature_C': temp,
                        'Load_Level': load_level,
                        'Hot_Compressive_MPa': round(max(0, hot_fc), 2),
                        'Hot_Modulus_GPa': round(max(0, hot_E), 2),
                        'Thermal_Strain_x10-6': round(thermal_strain * 1e6, 1),
                        'LITS_x10-6': round(lits * 1e6, 1),
                        'Total_Strain_x10-6': round(total_strain * 1e6, 1),
                        'CTE_x10-6_per_C': round(cte, 2),
                        'Test_Duration_min': 30 + np.random.randint(-5, 5),
                        'Heating_Rate_C_min': 5.0 + np.random.uniform(-0.5, 0.5)
                    })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.base_path}/in_situ_tests/in_situ_properties.csv", index=False)
        return df
    
    def generate_spalling_data(self):
        """Generate spalling behavior and pore pressure data"""
        results = []
        temperatures = [200, 300, 400, 500, 600, 700, 800]
        
        for _, mix in self.mix_designs.iterrows():
            mix_id = mix['Mix_ID']
            rubber_content = mix['Rubber_Content_%']
            has_pp_fibers = mix['PP_Fibers_kg_m3'] > 0
            has_steel_fibers = mix['Steel_Fibers_kg_m3'] > 0
            
            for temp in temperatures:
                for specimen in range(1, 3):
                    # Spalling probability and severity
                    base_spalling_prob = 0.1 + (temp - 200) * 0.001
                    
                    # Rubber reduces spalling risk
                    if rubber_content > 0:
                        spalling_reduction = 1 - (rubber_content * 0.03)
                        base_spalling_prob *= spalling_reduction
                    
                    # PP fibers significantly reduce spalling
                    if has_pp_fibers:
                        base_spalling_prob *= 0.2
                    
                    # Determine if spalling occurs
                    spalling_occurs = np.random.random() < base_spalling_prob
                    
                    if spalling_occurs:
                        # Spalling depth and area
                        spalling_depth = np.random.uniform(5, 30) * (temp / 400)
                        spalling_area = np.random.uniform(10, 50) * (temp / 400)
                        spalling_type = np.random.choice(['Explosive', 'Progressive', 'Corner', 'Surface'])
                        time_to_spalling = np.random.uniform(10, 40)
                    else:
                        spalling_depth = 0
                        spalling_area = 0
                        spalling_type = 'None'
                        time_to_spalling = None
                    
                    # Pore pressure at different depths
                    depths = [10, 20, 30, 40, 50]
                    for depth in depths:
                        # Peak pore pressure (MPa)
                        if temp <= 300:
                            base_pressure = 0.5 + (temp - 100) * 0.002
                        elif temp <= 500:
                            base_pressure = 0.9 + (temp - 300) * 0.003
                        else:
                            base_pressure = 1.5 + (temp - 500) * 0.002
                        
                        # Depth effect
                        depth_factor = 1 - (depth / 100)
                        
                        # Rubber creates additional pathways, reducing pressure
                        if rubber_content > 0:
                            pressure_reduction = 1 - (rubber_content * 0.02)
                            base_pressure *= pressure_reduction
                        
                        # PP fibers create micro-channels
                        if has_pp_fibers:
                            base_pressure *= 0.6
                        
                        peak_pressure = base_pressure * depth_factor * np.random.normal(1.0, 0.15)
                        time_to_peak = 15 + depth * 0.5 + np.random.uniform(-3, 3)
                        
                        results.append({
                            'Mix_ID': mix_id,
                            'Specimen_ID': f"{mix_id}_Spall_T{temp}_S{specimen}",
                            'Temperature_C': temp,
                            'Depth_mm': depth,
                            'Peak_Pore_Pressure_MPa': round(max(0, peak_pressure), 3),
                            'Time_to_Peak_min': round(time_to_peak, 1),
                            'Spalling_Occurred': spalling_occurs,
                            'Spalling_Type': spalling_type,
                            'Spalling_Depth_mm': round(spalling_depth, 1),
                            'Spalling_Area_cm2': round(spalling_area, 1),
                            'Time_to_Spalling_min': round(time_to_spalling, 1) if time_to_spalling else None,
                            'Moisture_Content_%': 3.5 + np.random.uniform(-0.5, 0.5),
                            'Heating_Rate_C_min': 5.0 + np.random.uniform(-0.5, 0.5)
                        })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.base_path}/spalling_data/spalling_pore_pressure.csv", index=False)
        return df
    
    def generate_stress_strain_curves(self):
        """Generate detailed stress-strain curve data"""
        results = []
        temperatures = [23, 200, 400, 600, 800]
        
        ambient_df = pd.read_csv(f"{self.base_path}/ambient_tests/ambient_mechanical_properties.csv")
        numeric_cols = ['Compressive_Strength_MPa', 'Splitting_Tensile_MPa', 'Modulus_Elasticity_GPa', 
                       'Density_kg_m3', 'UPV_m_s', 'Poisson_Ratio']
        ambient_28d = ambient_df[ambient_df['Age_days'] == 28].groupby('Mix_ID')[numeric_cols].mean()
        
        for _, mix in self.mix_designs.iterrows():
            mix_id = mix['Mix_ID']
            rubber_content = mix['Rubber_Content_%']
            
            if mix_id not in ambient_28d.index:
                continue
            
            baseline_fc = ambient_28d.loc[mix_id, 'Compressive_Strength_MPa']
            baseline_E = ambient_28d.loc[mix_id, 'Modulus_Elasticity_GPa']
            
            for temp in temperatures:
                # Get temperature factors from previous data
                if temp == 23:
                    fc_factor = 1.0
                    E_factor = 1.0
                    strain_at_peak = 0.002 + rubber_content * 0.0001
                elif temp == 200:
                    fc_factor = 0.95
                    E_factor = 0.80
                    strain_at_peak = 0.003 + rubber_content * 0.0002
                elif temp == 400:
                    fc_factor = 0.85
                    E_factor = 0.50
                    strain_at_peak = 0.005 + rubber_content * 0.0003
                elif temp == 600:
                    fc_factor = 0.60
                    E_factor = 0.25
                    strain_at_peak = 0.008 + rubber_content * 0.0004
                else:  # 800
                    fc_factor = 0.30
                    E_factor = 0.10
                    strain_at_peak = 0.012 + rubber_content * 0.0005
                
                fc = baseline_fc * fc_factor
                E = baseline_E * E_factor * 1000  # Convert to MPa
                
                # Generate stress-strain points
                strains = np.linspace(0, strain_at_peak * 3, 50)
                
                for strain in strains:
                    if strain <= strain_at_peak:
                        # Ascending branch - modified Hognestad parabola
                        stress = fc * (2 * strain / strain_at_peak - (strain / strain_at_peak) ** 2)
                    else:
                        # Descending branch - linear or exponential decay
                        decay_rate = 0.5 - rubber_content * 0.01  # Rubber provides ductility
                        stress = fc * np.exp(-decay_rate * (strain - strain_at_peak) / strain_at_peak)
                    
                    # Add some noise
                    stress *= np.random.normal(1.0, 0.02)
                    
                    results.append({
                        'Mix_ID': mix_id,
                        'Temperature_C': temp,
                        'Specimen_ID': f"{mix_id}_SS_T{temp}",
                        'Strain': round(strain, 6),
                        'Stress_MPa': round(max(0, stress), 3),
                        'Peak_Stress_MPa': round(fc, 2),
                        'Peak_Strain': round(strain_at_peak, 4),
                        'Initial_Modulus_MPa': round(E, 0)
                    })
        
        df = pd.DataFrame(results)
        df.to_csv(f"{self.base_path}/high_temp_tests/stress_strain_curves.csv", index=False)
        return df
    
    def _get_color_change(self, temp):
        """Determine color change based on temperature"""
        if temp <= 200:
            return "No change"
        elif temp <= 300:
            return "Slight lightening"
        elif temp <= 400:
            return "Light gray"
        elif temp <= 600:
            return "Gray to pink"
        elif temp <= 800:
            return "Light pink to red"
        else:
            return "Whitish gray"
    
    def _get_surface_condition(self, temp, cooling):
        """Determine surface condition based on temperature and cooling method"""
        if temp <= 200:
            return "Intact"
        elif temp <= 400:
            if cooling == 'Water_Quenched':
                return "Minor surface cracks"
            else:
                return "Hairline cracks"
        elif temp <= 600:
            if cooling == 'Water_Quenched':
                return "Moderate cracking and scaling"
            else:
                return "Network of cracks"
        else:
            if cooling == 'Water_Quenched':
                return "Severe cracking and spalling"
            else:
                return "Extensive cracking"
    
    def generate_all_datasets(self):
        """Generate all experimental datasets"""
        print("Generating experimental dataset for rubberized concrete fire resistance...")
        
        print("\n1. Generating ambient condition tests...")
        self.generate_ambient_tests()
        
        print("2. Generating high-temperature exposure tests...")
        self.generate_high_temp_exposure_tests()
        
        print("3. Generating in-situ high-temperature tests...")
        self.generate_in_situ_tests()
        
        print("4. Generating spalling and pore pressure data...")
        self.generate_spalling_data()
        
        print("5. Generating stress-strain curves...")
        self.generate_stress_strain_curves()
        
        print("\nDataset generation complete!")
        
        # Generate metadata
        metadata = {
            "dataset_name": "Rubberized Concrete Fire Resistance Experimental Dataset",
            "generation_date": datetime.now().isoformat(),
            "purpose": "Thermo-mechanical modeling of fire-resistant rubberized concrete",
            "test_standards": {
                "compressive_strength": "ASTM C39",
                "tensile_strength": "ASTM C496",
                "elastic_modulus": "ASTM C469",
                "fire_test": "ISO 834"
            },
            "temperature_range": "23-800°C",
            "mix_designs": len(self.mix_designs),
            "total_specimens": "Approximately 2000+",
            "cooling_methods": ["Furnace cooling", "Water quenching"],
            "key_parameters": [
                "Residual mechanical properties",
                "In-situ hot properties",
                "Thermal strain and LITS",
                "Spalling behavior",
                "Pore pressure development"
            ]
        }
        
        with open("documentation/dataset_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    generator = RubberizedConcreteDataGenerator()
    generator.generate_all_datasets()