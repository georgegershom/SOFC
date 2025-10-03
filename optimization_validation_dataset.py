"""
Optimization and Validation Dataset Generator
For inverse modeling and PSO-based defect identification

This script generates:
1. FEM-predicted vs. experimental stress/strain profiles
2. Crack depth estimates from synchrotron XRD vs. model predictions
3. Optimal sintering parameters
4. Geometric design variations
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Set random seed for reproducibility
np.random.seed(42)

class OptimizationValidationDataGenerator:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_stress_strain_profiles(self, n_samples=100, n_positions=50):
        """
        Generate FEM-predicted vs. experimental stress/strain profiles
        """
        print("Generating stress/strain profiles...")
        
        # Position along the sample (mm)
        positions = np.linspace(0, 100, n_positions)
        
        data = []
        for sample_id in range(1, n_samples + 1):
            # Material properties variation
            youngs_modulus = np.random.uniform(180, 220)  # GPa
            poisson_ratio = np.random.uniform(0.28, 0.32)
            
            # FEM predicted stress profile (MPa)
            # Sinusoidal pattern with noise representing typical residual stress distribution
            base_stress_fem = 50 * np.sin(2 * np.pi * positions / 100) + \
                             30 * np.cos(4 * np.pi * positions / 100) + \
                             np.random.normal(0, 5, n_positions)
            
            # Experimental stress from synchrotron XRD (MPa)
            # Add measurement noise and slight systematic deviation
            base_stress_exp = base_stress_fem + \
                             np.random.normal(0, 8, n_positions) + \
                             np.random.uniform(-3, 3)
            
            # FEM predicted strain (microstrain)
            strain_fem = (base_stress_fem / youngs_modulus) * 1e6
            
            # Experimental strain (microstrain)
            strain_exp = (base_stress_exp / youngs_modulus) * 1e6 + \
                        np.random.normal(0, 50, n_positions)
            
            # Residual calculation
            stress_residual = np.abs(base_stress_fem - base_stress_exp)
            strain_residual = np.abs(strain_fem - strain_exp)
            
            for i, pos in enumerate(positions):
                data.append({
                    'sample_id': sample_id,
                    'position_mm': pos,
                    'fem_stress_MPa': base_stress_fem[i],
                    'experimental_stress_MPa': base_stress_exp[i],
                    'fem_strain_microstrain': strain_fem[i],
                    'experimental_strain_microstrain': strain_exp[i],
                    'stress_residual_MPa': stress_residual[i],
                    'strain_residual_microstrain': strain_residual[i],
                    'youngs_modulus_GPa': youngs_modulus,
                    'poisson_ratio': poisson_ratio,
                    'measurement_method': 'synchrotron_XRD',
                    'fem_mesh_size_mm': np.random.choice([0.5, 1.0, 2.0]),
                    'convergence_achieved': np.random.choice([True, False], p=[0.95, 0.05])
                })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_crack_depth_estimates(self, n_samples=200):
        """
        Generate crack depth estimates from synchrotron XRD vs. model predictions
        """
        print("Generating crack depth estimates...")
        
        data = []
        
        for sample_id in range(1, n_samples + 1):
            # True crack depth (unknown in practice, for validation)
            true_crack_depth = np.random.uniform(0.1, 5.0)  # mm
            
            # Synchrotron XRD measurement
            xrd_measurement_error = np.random.normal(0, 0.15)  # mm
            xrd_crack_depth = max(0.01, true_crack_depth + xrd_measurement_error)
            
            # PSO-based inverse model prediction
            pso_iterations = np.random.randint(50, 200)
            pso_convergence_rate = np.random.uniform(0.85, 0.99)
            
            # Model prediction error decreases with iterations
            model_error_factor = 1.0 - (pso_convergence_rate * (pso_iterations / 200))
            pso_prediction_error = np.random.normal(0, 0.2 * model_error_factor)
            pso_crack_depth = max(0.01, true_crack_depth + pso_prediction_error)
            
            # Fitness function value (lower is better)
            fitness_value = np.random.uniform(0.001, 0.1) * (1 + abs(pso_prediction_error))
            
            # Crack characteristics
            crack_type = np.random.choice(['surface', 'subsurface', 'through-thickness'], 
                                         p=[0.5, 0.3, 0.2])
            crack_orientation = np.random.uniform(0, 90)  # degrees from normal
            crack_width = np.random.uniform(0.01, 0.5)  # mm
            
            # Location on sample
            x_position = np.random.uniform(0, 100)
            y_position = np.random.uniform(0, 50)
            
            # Stress intensity factor (MPa·√m)
            stress_intensity = np.random.uniform(1.0, 10.0) * np.sqrt(true_crack_depth / 1000)
            
            data.append({
                'sample_id': sample_id,
                'xrd_crack_depth_mm': xrd_crack_depth,
                'pso_predicted_depth_mm': pso_crack_depth,
                'true_crack_depth_mm': true_crack_depth,
                'xrd_measurement_error_mm': abs(xrd_crack_depth - true_crack_depth),
                'pso_prediction_error_mm': abs(pso_crack_depth - true_crack_depth),
                'xrd_pso_difference_mm': abs(xrd_crack_depth - pso_crack_depth),
                'pso_iterations': pso_iterations,
                'pso_convergence_rate': pso_convergence_rate,
                'fitness_value': fitness_value,
                'crack_type': crack_type,
                'crack_orientation_deg': crack_orientation,
                'crack_width_mm': crack_width,
                'x_position_mm': x_position,
                'y_position_mm': y_position,
                'stress_intensity_factor_MPa_sqrtm': stress_intensity,
                'detection_confidence': np.random.uniform(0.7, 0.99),
                'material': np.random.choice(['Al2O3', 'ZrO2', 'Si3N4', 'SiC']),
                'sintering_temperature_C': np.random.uniform(1400, 1700)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_sintering_parameters(self, n_experiments=150):
        """
        Generate optimal sintering parameters dataset
        """
        print("Generating optimal sintering parameters...")
        
        data = []
        
        for exp_id in range(1, n_experiments + 1):
            # Sintering parameters
            cooling_rate = np.random.uniform(0.5, 5.0)  # °C/min
            heating_rate = np.random.uniform(2.0, 10.0)  # °C/min
            hold_temperature = np.random.uniform(1400, 1700)  # °C
            hold_time = np.random.uniform(1, 8)  # hours
            
            # Atmospheric conditions
            atmosphere = np.random.choice(['air', 'argon', 'vacuum', 'nitrogen'])
            pressure = np.random.uniform(0.001, 101.325)  # kPa
            
            # Resulting material properties
            # Optimal cooling rate is 1-2°C/min, use this to calculate quality
            cooling_optimality = np.exp(-((cooling_rate - 1.5) ** 2) / 2)
            temp_optimality = np.exp(-((hold_temperature - 1550) ** 2) / 5000)
            
            base_quality = cooling_optimality * temp_optimality * 100
            
            # Density (% of theoretical)
            density = base_quality * np.random.uniform(0.95, 1.0)
            
            # Grain size (μm)
            grain_size = 5.0 + (cooling_rate * 2) + np.random.normal(0, 1)
            
            # Porosity (%)
            porosity = (100 - density) * 0.3 + np.random.uniform(0, 2)
            
            # Mechanical properties
            flexural_strength = density * 5 - grain_size * 10 + np.random.normal(0, 20)  # MPa
            hardness = density * 20 + np.random.normal(0, 50)  # HV
            fracture_toughness = 5 + (density / 20) - (grain_size * 0.1) + np.random.normal(0, 0.5)  # MPa·√m
            
            # Defect metrics
            crack_density = (5 - cooling_rate) * abs(np.random.normal(0.5, 0.2))  # cracks/cm²
            crack_density = max(0, crack_density)
            
            residual_stress = abs(cooling_rate - 1.5) * 30 + np.random.normal(0, 10)  # MPa
            
            # Warpage
            warpage = abs(cooling_rate - 1.5) * 0.5 + np.random.uniform(0, 0.3)  # mm
            
            # Cost and time
            total_cycle_time = (hold_temperature / heating_rate) + hold_time + \
                              (hold_temperature / cooling_rate)  # hours
            
            energy_cost = total_cycle_time * hold_temperature * 0.001  # relative units
            
            # Overall quality score
            quality_score = (density * 0.4 + 
                           (100 - porosity) * 0.2 + 
                           (flexural_strength / 10) * 0.2 + 
                           (5 - crack_density) * 0.1 + 
                           (100 - residual_stress / 5) * 0.1)
            
            data.append({
                'experiment_id': exp_id,
                'cooling_rate_C_per_min': cooling_rate,
                'heating_rate_C_per_min': heating_rate,
                'hold_temperature_C': hold_temperature,
                'hold_time_hours': hold_time,
                'atmosphere': atmosphere,
                'pressure_kPa': pressure,
                'density_percent_theoretical': density,
                'grain_size_um': grain_size,
                'porosity_percent': porosity,
                'flexural_strength_MPa': flexural_strength,
                'hardness_HV': hardness,
                'fracture_toughness_MPa_sqrtm': fracture_toughness,
                'crack_density_per_cm2': crack_density,
                'residual_stress_MPa': residual_stress,
                'warpage_mm': warpage,
                'total_cycle_time_hours': total_cycle_time,
                'energy_cost_relative': energy_cost,
                'quality_score': quality_score,
                'optimal_range_cooling': 1.0 <= cooling_rate <= 2.0,
                'material': np.random.choice(['Al2O3', 'ZrO2', 'Si3N4', 'SiC']),
                'green_density_percent': np.random.uniform(50, 65)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_geometric_designs(self, n_designs=100):
        """
        Generate geometric design variations (bow-shaped vs. rectangular channels)
        """
        print("Generating geometric design variations...")
        
        data = []
        
        for design_id in range(1, n_designs + 1):
            # Design type
            design_type = np.random.choice(['bow_shaped', 'rectangular', 'trapezoidal', 'circular'])
            
            # Common parameters
            length = np.random.uniform(50, 200)  # mm
            width = np.random.uniform(10, 50)  # mm
            height = np.random.uniform(5, 30)  # mm
            wall_thickness = np.random.uniform(1, 5)  # mm
            
            # Design-specific parameters
            if design_type == 'bow_shaped':
                curvature_radius = np.random.uniform(50, 500)  # mm
                arc_angle = np.random.uniform(10, 180)  # degrees
                channel_count = np.random.randint(1, 10)
                specific_param = curvature_radius
            elif design_type == 'rectangular':
                aspect_ratio = width / height
                corner_radius = np.random.uniform(0, 5)  # mm
                channel_count = np.random.randint(1, 20)
                specific_param = aspect_ratio
            elif design_type == 'trapezoidal':
                top_width = width + np.random.uniform(-5, 5)
                taper_angle = np.random.uniform(5, 45)  # degrees
                channel_count = np.random.randint(1, 15)
                specific_param = taper_angle
            else:  # circular
                diameter = np.random.uniform(10, 50)  # mm
                channel_count = np.random.randint(1, 30)
                specific_param = diameter
            
            # Calculate volume and surface area (simplified)
            volume = length * width * height * 0.7  # mm³ (approximate with fill factor)
            surface_area = 2 * (length * width + length * height + width * height)  # mm²
            
            # Hydraulic properties (for channel designs)
            hydraulic_diameter = (4 * volume / surface_area) if surface_area > 0 else 0
            flow_rate = np.random.uniform(0.1, 10.0)  # L/min
            pressure_drop = np.random.uniform(0.1, 5.0)  # kPa
            
            # Mechanical performance
            if design_type == 'bow_shaped':
                stress_concentration = 1.2 + np.random.uniform(0, 0.5)
                thermal_performance = 85 + np.random.uniform(0, 10)
            elif design_type == 'rectangular':
                stress_concentration = 2.5 + np.random.uniform(0, 1.0)
                thermal_performance = 70 + np.random.uniform(0, 15)
            elif design_type == 'trapezoidal':
                stress_concentration = 1.8 + np.random.uniform(0, 0.7)
                thermal_performance = 75 + np.random.uniform(0, 12)
            else:  # circular
                stress_concentration = 1.0 + np.random.uniform(0, 0.3)
                thermal_performance = 90 + np.random.uniform(0, 8)
            
            # FEM results
            max_von_mises_stress = np.random.uniform(50, 300) * stress_concentration  # MPa
            max_displacement = np.random.uniform(0.01, 0.5)  # mm
            max_strain = np.random.uniform(100, 2000)  # microstrain
            
            # Thermal properties
            max_temperature = np.random.uniform(100, 600)  # °C
            thermal_gradient = np.random.uniform(1, 20)  # °C/mm
            heat_transfer_coeff = np.random.uniform(10, 1000)  # W/m²K
            
            # Manufacturing metrics
            manufacturing_difficulty = {
                'bow_shaped': np.random.uniform(6, 9),
                'rectangular': np.random.uniform(2, 5),
                'trapezoidal': np.random.uniform(4, 7),
                'circular': np.random.uniform(3, 6)
            }[design_type]
            
            manufacturing_cost = volume * manufacturing_difficulty * 0.01  # relative units
            
            # Defect susceptibility
            warpage_risk = stress_concentration * 0.3 + np.random.uniform(0, 0.5)  # mm
            crack_risk_score = max_von_mises_stress / 100 + np.random.uniform(0, 2)
            
            # Optimization metrics
            weight = volume * 3.9e-6  # kg (assuming ceramic density ~3.9 g/cm³)
            efficiency_score = thermal_performance / (stress_concentration * manufacturing_difficulty)
            
            data.append({
                'design_id': design_id,
                'design_type': design_type,
                'length_mm': length,
                'width_mm': width,
                'height_mm': height,
                'wall_thickness_mm': wall_thickness,
                'channel_count': channel_count,
                'design_specific_parameter': specific_param,
                'volume_mm3': volume,
                'surface_area_mm2': surface_area,
                'hydraulic_diameter_mm': hydraulic_diameter,
                'flow_rate_L_per_min': flow_rate,
                'pressure_drop_kPa': pressure_drop,
                'stress_concentration_factor': stress_concentration,
                'max_von_mises_stress_MPa': max_von_mises_stress,
                'max_displacement_mm': max_displacement,
                'max_strain_microstrain': max_strain,
                'thermal_performance_score': thermal_performance,
                'max_temperature_C': max_temperature,
                'thermal_gradient_C_per_mm': thermal_gradient,
                'heat_transfer_coefficient_W_m2K': heat_transfer_coeff,
                'manufacturing_difficulty_score': manufacturing_difficulty,
                'manufacturing_cost_relative': manufacturing_cost,
                'warpage_risk_mm': warpage_risk,
                'crack_risk_score': crack_risk_score,
                'weight_kg': weight,
                'efficiency_score': efficiency_score,
                'material': np.random.choice(['Al2O3', 'ZrO2', 'Si3N4', 'SiC']),
                'sintering_temperature_C': np.random.uniform(1400, 1700)
            })
        
        df = pd.DataFrame(data)
        return df
    
    def generate_pso_optimization_history(self, n_runs=50):
        """
        Generate PSO optimization history for defect identification
        """
        print("Generating PSO optimization history...")
        
        data = []
        
        for run_id in range(1, n_runs + 1):
            # PSO parameters
            n_particles = np.random.choice([20, 30, 50, 100])
            n_iterations = np.random.choice([100, 200, 500])
            w = np.random.uniform(0.4, 0.9)  # inertia weight
            c1 = np.random.uniform(1.5, 2.5)  # cognitive parameter
            c2 = np.random.uniform(1.5, 2.5)  # social parameter
            
            # Initial fitness
            initial_fitness = np.random.uniform(0.5, 2.0)
            
            # Convergence characteristics
            convergence_rate = np.random.uniform(0.85, 0.99)
            
            # Generate iteration history
            for iteration in range(0, n_iterations, 10):
                progress = iteration / n_iterations
                
                # Fitness decreases with iteration (exponential decay)
                best_fitness = initial_fitness * np.exp(-convergence_rate * progress * 5)
                avg_fitness = best_fitness * np.random.uniform(1.2, 2.0)
                worst_fitness = best_fitness * np.random.uniform(2.0, 4.0)
                
                # Best parameters found
                crack_depth_param = np.random.uniform(0.1, 5.0)
                crack_width_param = np.random.uniform(0.01, 0.5)
                crack_angle_param = np.random.uniform(0, 90)
                
                # Diversity metrics
                swarm_diversity = (1 - progress) * np.random.uniform(0.5, 1.0)
                
                data.append({
                    'run_id': run_id,
                    'iteration': iteration,
                    'n_particles': n_particles,
                    'inertia_weight': w,
                    'cognitive_param_c1': c1,
                    'social_param_c2': c2,
                    'best_fitness': best_fitness,
                    'average_fitness': avg_fitness,
                    'worst_fitness': worst_fitness,
                    'swarm_diversity': swarm_diversity,
                    'best_crack_depth_mm': crack_depth_param,
                    'best_crack_width_mm': crack_width_param,
                    'best_crack_angle_deg': crack_angle_param,
                    'convergence_achieved': progress > 0.8 and best_fitness < 0.05,
                    'computation_time_sec': np.random.uniform(0.1, 5.0)
                })
        
        df = pd.DataFrame(data)
        return df
    
    def save_datasets(self, output_dir='/workspace/optimization_datasets'):
        """
        Generate and save all datasets
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("OPTIMIZATION AND VALIDATION DATASET GENERATOR")
        print(f"{'='*60}\n")
        
        # Generate all datasets
        stress_strain_df = self.generate_stress_strain_profiles()
        crack_depth_df = self.generate_crack_depth_estimates()
        sintering_df = self.generate_sintering_parameters()
        geometric_df = self.generate_geometric_designs()
        pso_history_df = self.generate_pso_optimization_history()
        
        # Save as CSV
        stress_strain_df.to_csv(f'{output_dir}/stress_strain_profiles.csv', index=False)
        crack_depth_df.to_csv(f'{output_dir}/crack_depth_estimates.csv', index=False)
        sintering_df.to_csv(f'{output_dir}/sintering_parameters.csv', index=False)
        geometric_df.to_csv(f'{output_dir}/geometric_designs.csv', index=False)
        pso_history_df.to_csv(f'{output_dir}/pso_optimization_history.csv', index=False)
        
        # Save summary statistics
        summary = {
            'generation_timestamp': self.timestamp,
            'stress_strain_profiles': {
                'n_samples': len(stress_strain_df['sample_id'].unique()),
                'n_datapoints': len(stress_strain_df),
                'avg_stress_residual_MPa': float(stress_strain_df['stress_residual_MPa'].mean()),
                'avg_strain_residual_microstrain': float(stress_strain_df['strain_residual_microstrain'].mean())
            },
            'crack_depth_estimates': {
                'n_samples': len(crack_depth_df),
                'avg_xrd_error_mm': float(crack_depth_df['xrd_measurement_error_mm'].mean()),
                'avg_pso_error_mm': float(crack_depth_df['pso_prediction_error_mm'].mean()),
                'avg_xrd_pso_difference_mm': float(crack_depth_df['xrd_pso_difference_mm'].mean())
            },
            'sintering_parameters': {
                'n_experiments': len(sintering_df),
                'optimal_cooling_rate_range': '1-2 °C/min',
                'n_optimal_experiments': int(sintering_df['optimal_range_cooling'].sum()),
                'avg_quality_score': float(sintering_df['quality_score'].mean())
            },
            'geometric_designs': {
                'n_designs': len(geometric_df),
                'design_types': geometric_df['design_type'].value_counts().to_dict(),
                'best_design_type': geometric_df.groupby('design_type')['efficiency_score'].mean().idxmax()
            },
            'pso_optimization': {
                'n_runs': len(pso_history_df['run_id'].unique()),
                'n_datapoints': len(pso_history_df),
                'avg_convergence_iterations': int(pso_history_df.groupby('run_id')['iteration'].max().mean())
            }
        }
        
        with open(f'{output_dir}/dataset_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print("DATASET GENERATION COMPLETE")
        print(f"{'='*60}\n")
        
        print("Files generated:")
        print(f"  1. stress_strain_profiles.csv ({len(stress_strain_df)} rows)")
        print(f"  2. crack_depth_estimates.csv ({len(crack_depth_df)} rows)")
        print(f"  3. sintering_parameters.csv ({len(sintering_df)} rows)")
        print(f"  4. geometric_designs.csv ({len(geometric_df)} rows)")
        print(f"  5. pso_optimization_history.csv ({len(pso_history_df)} rows)")
        print(f"  6. dataset_summary.json")
        
        print(f"\nOutput directory: {output_dir}\n")
        
        # Print key insights
        print("Key Insights:")
        print(f"  • Average stress residual (FEM vs Exp): {summary['stress_strain_profiles']['avg_stress_residual_MPa']:.2f} MPa")
        print(f"  • Average PSO prediction error: {summary['crack_depth_estimates']['avg_pso_error_mm']:.3f} mm")
        print(f"  • Experiments in optimal cooling range: {summary['sintering_parameters']['n_optimal_experiments']}/{summary['sintering_parameters']['n_experiments']}")
        print(f"  • Best geometric design type: {summary['geometric_designs']['best_design_type']}")
        
        return {
            'stress_strain': stress_strain_df,
            'crack_depth': crack_depth_df,
            'sintering': sintering_df,
            'geometric': geometric_df,
            'pso_history': pso_history_df
        }


if __name__ == "__main__":
    generator = OptimizationValidationDataGenerator()
    datasets = generator.save_datasets()
