#!/usr/bin/env python3
"""
Validation Dataset Generator for Fire-Resistant Rubberized Concrete
Model Validation Data Generation

This script generates realistic experimental validation datasets including:
1. Temperature evolution data (thermocouple measurements)
2. Deformation/strain history under thermal-mechanical loading
3. Spalling patterns and failure analysis

Author: AI Assistant  
Date: 2025-10-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta
import os
from scipy import interpolate
from scipy.integrate import odeint
import warnings
warnings.filterwarnings('ignore')

class ValidationDataGenerator:
    def __init__(self, output_dir="rubberized_concrete_dataset", seed=42):
        np.random.seed(seed)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Test specimen configurations
        self.specimen_configs = {
            "small_cube": {"dimensions": [100, 100, 100], "unit": "mm"},  # 100mm cube
            "cylinder": {"dimensions": [100, 200], "unit": "mm"},         # φ100×200mm
            "beam": {"dimensions": [100, 100, 400], "unit": "mm"},       # 100×100×400mm
            "slab": {"dimensions": [500, 500, 100], "unit": "mm"}        # 500×500×100mm
        }
        
        self.rubber_contents = [0, 5, 10, 15, 20]  # % by volume
        
        # Fire exposure curves
        self.fire_curves = {
            "ISO834": lambda t: 20 + 345 * np.log10(8*t + 1),  # Standard fire curve
            "ASTM_E119": lambda t: 20 + 750 * (1 - np.exp(-3.79553 * np.sqrt(t))) + 170.41 * np.sqrt(t),
            "hydrocarbon": lambda t: 20 + 1080 * (1 - 0.325 * np.exp(-0.167*t) - 0.675 * np.exp(-2.5*t)),
            "parametric": lambda t: 20 + 1325 * (1 - 0.324 * np.exp(-0.2*t) - 0.204 * np.exp(-1.7*t) - 0.472 * np.exp(-19*t))
        }

    def generate_temperature_evolution_data(self):
        """Generate temperature evolution data from thermocouple measurements"""
        print("Generating temperature evolution validation data...")
        
        validation_data = {}
        
        for specimen_type, config in self.specimen_configs.items():
            validation_data[specimen_type] = {}
            
            for rubber_content in self.rubber_contents:
                validation_data[specimen_type][f"rubber_{rubber_content}pct"] = {}
                
                for fire_curve_name, fire_curve_func in self.fire_curves.items():
                    # Time array (0 to 4 hours)
                    time_hours = np.linspace(0, 4, 241)  # Every minute
                    time_minutes = time_hours * 60
                    
                    # Furnace temperature
                    furnace_temp = fire_curve_func(time_hours)
                    
                    # Generate thermocouple positions
                    if specimen_type == "small_cube":
                        tc_positions = {
                            "surface": {"depth": 0, "description": "Surface thermocouple"},
                            "25mm": {"depth": 25, "description": "25mm from surface"},
                            "center": {"depth": 50, "description": "Center of specimen"}
                        }
                    elif specimen_type == "cylinder":
                        tc_positions = {
                            "surface": {"depth": 0, "description": "Surface thermocouple"},
                            "25mm": {"depth": 25, "description": "25mm from surface"},
                            "center": {"depth": 50, "description": "Center of specimen"}
                        }
                    elif specimen_type == "beam":
                        tc_positions = {
                            "surface": {"depth": 0, "description": "Surface thermocouple"},
                            "25mm": {"depth": 25, "description": "25mm from surface"},
                            "center": {"depth": 50, "description": "Center of specimen"}
                        }
                    else:  # slab
                        tc_positions = {
                            "surface": {"depth": 0, "description": "Surface thermocouple"},
                            "25mm": {"depth": 25, "description": "25mm from surface"},
                            "50mm": {"depth": 50, "description": "Center of slab"}
                        }
                    
                    # Calculate thermal diffusivity based on rubber content
                    # α = k/(ρ*cp) - decreases with rubber content
                    alpha_base = 0.8e-6  # m²/s for normal concrete
                    alpha = alpha_base * (1 - 0.3 * rubber_content / 20)
                    
                    tc_data = {"furnace_temperature": furnace_temp.tolist()}
                    
                    for tc_name, tc_info in tc_positions.items():
                        depth = tc_info["depth"] / 1000  # Convert to meters
                        
                        # Solve 1D heat conduction with temperature-dependent properties
                        temp_profile = self._solve_heat_conduction(
                            time_hours, furnace_temp, depth, alpha, rubber_content
                        )
                        
                        # Add measurement noise
                        temp_profile += np.random.normal(0, 2, len(temp_profile))
                        
                        tc_data[tc_name] = {
                            "temperature": temp_profile.tolist(),
                            "depth_mm": tc_info["depth"],
                            "description": tc_info["description"]
                        }
                    
                    validation_data[specimen_type][f"rubber_{rubber_content}pct"][fire_curve_name] = {
                        "time_hours": time_hours.tolist(),
                        "time_minutes": time_minutes.tolist(),
                        "rubber_content": rubber_content,
                        "fire_curve": fire_curve_name,
                        "specimen_type": specimen_type,
                        "dimensions": config,
                        "thermocouple_data": tc_data
                    }
        
        # Save temperature evolution data
        with open(f"{self.output_dir}/temperature_evolution_validation.json", "w") as f:
            json.dump(validation_data, f, indent=2)
        
        # Create CSV format for easier analysis
        self._create_temperature_csv(validation_data)
        
        return validation_data

    def _solve_heat_conduction(self, time_hours, furnace_temp, depth, alpha, rubber_content):
        """Simplified 1D heat conduction solution"""
        temp_profile = np.zeros_like(time_hours)
        temp_profile[0] = 20  # Initial temperature
        
        for i in range(1, len(time_hours)):
            dt = (time_hours[i] - time_hours[i-1]) * 3600  # Convert to seconds
            
            # Surface temperature (with some thermal resistance)
            T_surface = furnace_temp[i] - 50 * np.exp(-time_hours[i]/0.5)  # Thermal lag
            
            # Approximate solution for semi-infinite solid
            if depth == 0:
                temp_profile[i] = T_surface
            else:
                # Penetration depth
                penetration = 2 * np.sqrt(alpha * time_hours[i] * 3600)
                
                if depth < penetration:
                    # Temperature rise based on error function solution
                    eta = depth / (2 * np.sqrt(alpha * time_hours[i] * 3600))
                    temp_rise = (T_surface - 20) * (1 - eta)
                    temp_profile[i] = 20 + temp_rise
                else:
                    temp_profile[i] = temp_profile[i-1] + 0.1  # Slow heating
        
        return temp_profile

    def _create_temperature_csv(self, validation_data):
        """Create CSV files for temperature evolution data"""
        all_data = []
        
        for specimen_type in validation_data:
            for rubber_key in validation_data[specimen_type]:
                for fire_curve in validation_data[specimen_type][rubber_key]:
                    data = validation_data[specimen_type][rubber_key][fire_curve]
                    tc_data = data["thermocouple_data"]
                    
                    for i, time_h in enumerate(data["time_hours"]):
                        row_base = {
                            "specimen_type": specimen_type,
                            "rubber_content_pct": data["rubber_content"],
                            "fire_curve": fire_curve,
                            "time_hours": time_h,
                            "time_minutes": data["time_minutes"][i],
                            "furnace_temperature_C": tc_data["furnace_temperature"][i]
                        }
                        
                        for tc_name in ["surface", "25mm", "center"]:
                            if tc_name in tc_data:
                                row = row_base.copy()
                                row["thermocouple_location"] = tc_name
                                row["depth_mm"] = tc_data[tc_name]["depth_mm"]
                                row["temperature_C"] = tc_data[tc_name]["temperature"][i]
                                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/temperature_evolution_validation.csv", index=False)

    def generate_deformation_strain_data(self):
        """Generate deformation and strain history under thermal-mechanical loading"""
        print("Generating deformation/strain validation data...")
        
        strain_data = {}
        
        # Loading scenarios
        loading_scenarios = {
            "thermal_only": {"mechanical_load": 0, "description": "Free thermal expansion"},
            "low_load": {"mechanical_load": 0.2, "description": "20% of compressive strength"},
            "medium_load": {"mechanical_load": 0.4, "description": "40% of compressive strength"},
            "high_load": {"mechanical_load": 0.6, "description": "60% of compressive strength"}
        }
        
        for rubber_content in self.rubber_contents:
            strain_data[f"rubber_{rubber_content}pct"] = {}
            
            for scenario_name, scenario in loading_scenarios.items():
                # Time and temperature profile
                time_hours = np.linspace(0, 3, 181)  # 3 hours, every minute
                temperature = 20 + 400 * (1 - np.exp(-time_hours/0.8))  # Heating curve
                
                # Calculate strains
                total_strain = np.zeros_like(time_hours)
                thermal_strain = np.zeros_like(time_hours)
                mechanical_strain = np.zeros_like(time_hours)
                creep_strain = np.zeros_like(time_hours)
                transient_strain = np.zeros_like(time_hours)
                
                # Material properties (simplified)
                alpha = (10 + 3 * rubber_content / 20) * 1e-6  # Thermal expansion coefficient
                E0 = 30 - 8 * rubber_content / 20  # Initial elastic modulus (GPa)
                
                for i in range(len(time_hours)):
                    T = temperature[i]
                    t = time_hours[i]
                    
                    # Thermal strain
                    thermal_strain[i] = alpha * (T - 20)
                    
                    # Mechanical strain (if loaded)
                    if scenario["mechanical_load"] > 0:
                        # Elastic modulus decreases with temperature
                        E_T = E0 * np.exp(-0.0015 * (T - 20))
                        stress = scenario["mechanical_load"] * (35 - 8 * rubber_content / 20)  # MPa
                        mechanical_strain[i] = stress / (E_T * 1000)  # Convert GPa to MPa
                        
                        # Creep strain (time and temperature dependent)
                        if t > 0:
                            creep_factor = scenario["mechanical_load"] * (1 + 0.002 * T)
                            creep_strain[i] = creep_factor * 0.0005 * (t**0.3)
                    
                    # Transient strain (first heating only)
                    if 50 < T < 300:
                        transient_strain[i] = 0.0003 * np.exp(-((T - 150) / 80)**2) * \
                                            (1 - 0.3 * rubber_content / 20)
                    
                    # Total strain
                    total_strain[i] = thermal_strain[i] + mechanical_strain[i] + \
                                    creep_strain[i] + transient_strain[i]
                
                # Add measurement noise
                total_strain += np.random.normal(0, 0.00005, len(total_strain))
                
                # Calculate displacement for 100mm gauge length
                displacement = total_strain * 100  # mm
                
                strain_data[f"rubber_{rubber_content}pct"][scenario_name] = {
                    "rubber_content": rubber_content,
                    "loading_scenario": scenario_name,
                    "mechanical_load_ratio": scenario["mechanical_load"],
                    "description": scenario["description"],
                    "time_hours": time_hours.tolist(),
                    "temperature_C": temperature.tolist(),
                    "total_strain": total_strain.tolist(),
                    "thermal_strain": thermal_strain.tolist(),
                    "mechanical_strain": mechanical_strain.tolist(),
                    "creep_strain": creep_strain.tolist(),
                    "transient_strain": transient_strain.tolist(),
                    "displacement_mm": displacement.tolist(),
                    "gauge_length_mm": 100
                }
        
        # Save strain data
        with open(f"{self.output_dir}/deformation_strain_validation.json", "w") as f:
            json.dump(strain_data, f, indent=2)
        
        # Create CSV format
        self._create_strain_csv(strain_data)
        
        return strain_data

    def _create_strain_csv(self, strain_data):
        """Create CSV files for strain data"""
        all_data = []
        
        for rubber_key in strain_data:
            for scenario in strain_data[rubber_key]:
                data = strain_data[rubber_key][scenario]
                
                for i, time_h in enumerate(data["time_hours"]):
                    row = {
                        "rubber_content_pct": data["rubber_content"],
                        "loading_scenario": scenario,
                        "mechanical_load_ratio": data["mechanical_load_ratio"],
                        "time_hours": time_h,
                        "temperature_C": data["temperature_C"][i],
                        "total_strain": data["total_strain"][i],
                        "thermal_strain": data["thermal_strain"][i],
                        "mechanical_strain": data["mechanical_strain"][i],
                        "creep_strain": data["creep_strain"][i],
                        "transient_strain": data["transient_strain"][i],
                        "displacement_mm": data["displacement_mm"][i]
                    }
                    all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(f"{self.output_dir}/deformation_strain_validation.csv", index=False)

    def generate_spalling_failure_data(self):
        """Generate spalling patterns and failure time data"""
        print("Generating spalling and failure validation data...")
        
        spalling_data = {}
        
        # Test conditions
        test_conditions = {
            "standard_fire": {
                "heating_rate": "ISO834",
                "load_ratio": 0.3,
                "moisture_content": 4.5,  # %
                "description": "Standard fire exposure with moderate load"
            },
            "rapid_heating": {
                "heating_rate": "hydrocarbon", 
                "load_ratio": 0.4,
                "moisture_content": 6.0,
                "description": "Rapid heating with higher load"
            },
            "high_load": {
                "heating_rate": "ISO834",
                "load_ratio": 0.6,
                "moisture_content": 4.0,
                "description": "High mechanical load"
            },
            "high_moisture": {
                "heating_rate": "ISO834",
                "load_ratio": 0.3,
                "moisture_content": 8.0,
                "description": "High moisture content"
            }
        }
        
        for rubber_content in self.rubber_contents:
            spalling_data[f"rubber_{rubber_content}pct"] = {}
            
            for condition_name, condition in test_conditions.items():
                # Spalling susceptibility factors
                moisture_factor = condition["moisture_content"] / 4.0  # Normalized to 4%
                load_factor = condition["load_ratio"]
                rubber_factor = 1 - 0.4 * rubber_content / 20  # Rubber reduces spalling
                heating_factor = 1.5 if condition["heating_rate"] == "hydrocarbon" else 1.0
                
                # Overall spalling risk
                spalling_risk = moisture_factor * load_factor * rubber_factor * heating_factor
                
                # Time to first spalling (if it occurs)
                if spalling_risk > 0.8:
                    time_to_spalling = max(5, 60 - 30 * (spalling_risk - 0.8))  # minutes
                    spalling_occurred = True
                else:
                    time_to_spalling = None
                    spalling_occurred = False
                
                # Spalling depth progression
                time_points = np.linspace(0, 180, 181)  # 3 hours in minutes
                spalling_depth = np.zeros_like(time_points)
                
                if spalling_occurred:
                    spall_start_idx = int(time_to_spalling)
                    for i in range(spall_start_idx, len(time_points)):
                        t_since_start = time_points[i] - time_to_spalling
                        # Spalling depth increases with time (mm)
                        spalling_depth[i] = min(25, 2 * np.sqrt(t_since_start) * spalling_risk)
                
                # Add noise to spalling depth measurements
                spalling_depth += np.random.normal(0, 0.5, len(spalling_depth))
                spalling_depth = np.maximum(spalling_depth, 0)
                
                # Mass loss due to spalling (kg/m²)
                mass_loss = spalling_depth * 2.4 * (1 - 0.2 * rubber_content / 20)  # Density effect
                
                # Temperature at which spalling occurs
                if spalling_occurred:
                    spalling_temperature = 150 + 100 * np.random.random()  # 150-250°C typical
                else:
                    spalling_temperature = None
                
                # Failure criteria
                max_allowable_strain = 0.003  # 0.3%
                max_allowable_temp = 600  # °C at reinforcement level
                
                # Simulate failure time
                failure_time = None
                failure_mode = "No failure"
                
                # Check strain failure
                if spalling_risk > 1.2:
                    strain_failure_time = 90 + 60 * np.random.random()  # 90-150 minutes
                    failure_time = strain_failure_time
                    failure_mode = "Excessive deformation"
                
                # Check temperature failure  
                if condition["heating_rate"] == "hydrocarbon" and rubber_content < 10:
                    temp_failure_time = 60 + 30 * np.random.random()  # 60-90 minutes
                    if failure_time is None or temp_failure_time < failure_time:
                        failure_time = temp_failure_time
                        failure_mode = "Temperature limit exceeded"
                
                spalling_data[f"rubber_{rubber_content}pct"][condition_name] = {
                    "rubber_content": rubber_content,
                    "test_condition": condition_name,
                    "heating_rate": condition["heating_rate"],
                    "load_ratio": condition["load_ratio"],
                    "moisture_content_pct": condition["moisture_content"],
                    "description": condition["description"],
                    "spalling_occurred": spalling_occurred,
                    "time_to_spalling_min": time_to_spalling,
                    "spalling_temperature_C": spalling_temperature,
                    "spalling_risk_factor": spalling_risk,
                    "time_minutes": time_points.tolist(),
                    "spalling_depth_mm": spalling_depth.tolist(),
                    "mass_loss_kg_m2": mass_loss.tolist(),
                    "failure_time_min": failure_time,
                    "failure_mode": failure_mode,
                    "max_spalling_depth_mm": np.max(spalling_depth)
                }
        
        # Save spalling data
        with open(f"{self.output_dir}/spalling_failure_validation.json", "w") as f:
            json.dump(spalling_data, f, indent=2)
        
        # Create CSV format
        self._create_spalling_csv(spalling_data)
        
        return spalling_data

    def _create_spalling_csv(self, spalling_data):
        """Create CSV files for spalling data"""
        # Summary data
        summary_data = []
        detailed_data = []
        
        for rubber_key in spalling_data:
            for condition in spalling_data[rubber_key]:
                data = spalling_data[rubber_key][condition]
                
                # Summary row
                summary_row = {
                    "rubber_content_pct": data["rubber_content"],
                    "test_condition": condition,
                    "heating_rate": data["heating_rate"],
                    "load_ratio": data["load_ratio"],
                    "moisture_content_pct": data["moisture_content_pct"],
                    "spalling_occurred": data["spalling_occurred"],
                    "time_to_spalling_min": data["time_to_spalling_min"],
                    "spalling_temperature_C": data["spalling_temperature_C"],
                    "max_spalling_depth_mm": data["max_spalling_depth_mm"],
                    "failure_time_min": data["failure_time_min"],
                    "failure_mode": data["failure_mode"]
                }
                summary_data.append(summary_row)
                
                # Detailed time series data
                for i, time_min in enumerate(data["time_minutes"]):
                    detail_row = {
                        "rubber_content_pct": data["rubber_content"],
                        "test_condition": condition,
                        "time_minutes": time_min,
                        "spalling_depth_mm": data["spalling_depth_mm"][i],
                        "mass_loss_kg_m2": data["mass_loss_kg_m2"][i]
                    }
                    detailed_data.append(detail_row)
        
        # Save CSV files
        pd.DataFrame(summary_data).to_csv(f"{self.output_dir}/spalling_failure_summary.csv", index=False)
        pd.DataFrame(detailed_data).to_csv(f"{self.output_dir}/spalling_failure_detailed.csv", index=False)

if __name__ == "__main__":
    # Initialize validation data generator
    validator = ValidationDataGenerator()
    
    print("=== GENERATING VALIDATION DATASETS ===")
    
    # Generate validation datasets
    temp_data = validator.generate_temperature_evolution_data()
    strain_data = validator.generate_deformation_strain_data()
    spalling_data = validator.generate_spalling_failure_data()
    
    print(f"\nValidation datasets generated successfully!")
    print(f"Output directory: {validator.output_dir}")