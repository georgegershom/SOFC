#!/usr/bin/env python3
"""
Optimization and Validation Dataset Generator
For inverse modeling and PSO-based defect identification

This script generates synthetic data for:
1. FEM-predicted vs. experimental stress/strain profiles
2. Crack depth estimates from synchrotron XRD vs. model predictions
3. Optimal sintering parameters
4. Geometric design variations (bow-shaped vs. rectangular channels)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import interp1d
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ValidationDatasetGenerator:
    def __init__(self, seed=42):
        """Initialize the dataset generator with reproducible random seed."""
        np.random.seed(seed)
        self.data = {}
        
    def generate_stress_strain_profiles(self, n_samples=100):
        """
        Generate FEM-predicted vs experimental stress/strain profiles.
        Includes realistic material behavior with noise and systematic differences.
        """
        print("Generating stress/strain profiles...")
        
        # Strain range (0 to 0.1 for typical ceramic materials)
        strain_range = np.linspace(0, 0.1, 50)
        
        # Material parameters for different sample types
        materials = {
            'alumina': {'E': 370e9, 'sigma_y': 300e6, 'n': 0.15},
            'zirconia': {'E': 200e9, 'sigma_y': 800e6, 'n': 0.12},
            'silicon_carbide': {'E': 450e9, 'sigma_y': 400e6, 'n': 0.18}
        }
        
        profiles = []
        
        for i in range(n_samples):
            # Random material selection
            material = np.random.choice(list(materials.keys()))
            params = materials[material]
            
            # Generate FEM prediction (idealized)
            E = params['E']
            sigma_y = params['sigma_y']
            n = params['n']
            
            # Ramberg-Osgood model for stress-strain relationship
            stress_fem = E * strain_range * (1 + (E * strain_range / sigma_y) ** (1/n - 1))
            
            # Add FEM modeling uncertainties (systematic bias)
            fem_bias = np.random.normal(1.0, 0.05)  # 5% systematic error
            stress_fem *= fem_bias
            
            # Generate experimental data (with measurement noise)
            noise_level = np.random.uniform(0.02, 0.08)  # 2-8% noise
            stress_exp = stress_fem * (1 + np.random.normal(0, noise_level, len(strain_range)))
            
            # Add experimental artifacts (strain rate effects, temperature variations)
            temp_effect = 1 + 0.01 * np.sin(2 * np.pi * strain_range * 10)  # Temperature fluctuation
            stress_exp *= temp_effect
            
            # Sample information
            sample_info = {
                'sample_id': f'SS_{i:03d}',
                'material': material,
                'temperature': np.random.uniform(20, 200),  # °C
                'strain_rate': np.random.uniform(1e-4, 1e-2),  # s^-1
                'density': np.random.uniform(0.95, 0.99),  # relative density
                'grain_size': np.random.uniform(0.5, 5.0),  # μm
            }
            
            profiles.append({
                'sample_info': sample_info,
                'strain': strain_range,
                'stress_fem': stress_fem,
                'stress_exp': stress_exp,
                'rmse': np.sqrt(np.mean((stress_fem - stress_exp)**2)),
                'r_squared': 1 - np.sum((stress_exp - stress_fem)**2) / np.sum((stress_exp - np.mean(stress_exp))**2)
            })
        
        self.data['stress_strain_profiles'] = profiles
        return profiles
    
    def generate_crack_depth_data(self, n_samples=80):
        """
        Generate crack depth estimates from synchrotron XRD vs model predictions.
        Includes realistic diffraction patterns and crack evolution.
        """
        print("Generating crack depth estimates...")
        
        crack_data = []
        
        for i in range(n_samples):
            # Sample characteristics
            sample_id = f'CD_{i:03d}'
            material = np.random.choice(['alumina', 'zirconia', 'silicon_carbide'])
            
            # True crack depth (unknown in real experiments)
            true_depth = np.random.uniform(0.1, 50.0)  # μm
            
            # Synchrotron XRD measurements (multiple reflections)
            reflections = ['(104)', '(110)', '(113)', '(024)', '(116)']
            xrd_measurements = []
            
            for reflection in reflections:
                # Diffraction peak broadening due to crack
                peak_broadening = true_depth * np.random.uniform(0.8, 1.2)  # Measurement uncertainty
                
                # Instrumental broadening
                instrumental_broadening = np.random.uniform(0.05, 0.15)
                
                # Total broadening
                total_broadening = np.sqrt(peak_broadening**2 + instrumental_broadening**2)
                
                # Intensity reduction due to crack
                intensity_reduction = np.exp(-true_depth / np.random.uniform(10, 30))
                intensity = np.random.uniform(0.7, 1.0) * intensity_reduction
                
                xrd_measurements.append({
                    'reflection': reflection,
                    'peak_broadening': peak_broadening,
                    'total_broadening': total_broadening,
                    'intensity': intensity,
                    'd_spacing': np.random.uniform(1.0, 3.0)  # Å
                })
            
            # Model predictions (PSO-based crack identification)
            # Add systematic errors and noise
            model_bias = np.random.normal(1.0, 0.1)  # 10% systematic error
            model_noise = np.random.normal(0, 0.05)  # 5% random error
            
            predicted_depth = true_depth * model_bias * (1 + model_noise)
            
            # Confidence intervals
            confidence_interval = predicted_depth * np.random.uniform(0.1, 0.3)
            
            # Crack orientation and geometry
            crack_orientation = np.random.uniform(0, 180)  # degrees
            crack_aspect_ratio = np.random.uniform(0.1, 0.5)
            
            crack_data.append({
                'sample_id': sample_id,
                'material': material,
                'true_depth': true_depth,
                'predicted_depth': predicted_depth,
                'confidence_interval': confidence_interval,
                'xrd_measurements': xrd_measurements,
                'crack_orientation': crack_orientation,
                'crack_aspect_ratio': crack_aspect_ratio,
                'measurement_error': abs(true_depth - predicted_depth) / true_depth * 100,
                'sample_temperature': np.random.uniform(25, 300),  # °C
                'beam_energy': np.random.uniform(8, 15),  # keV
                'exposure_time': np.random.uniform(1, 10)  # seconds
            })
        
        self.data['crack_depth_data'] = crack_data
        return crack_data
    
    def generate_sintering_parameters(self, n_samples=60):
        """
        Generate optimal sintering parameters data.
        Includes cooling rates, temperatures, and processing conditions.
        """
        print("Generating sintering parameters...")
        
        sintering_data = []
        
        for i in range(n_samples):
            # Base sintering conditions
            max_temperature = np.random.uniform(1400, 1600)  # °C
            holding_time = np.random.uniform(30, 120)  # minutes
            
            # Optimal cooling rate (1-2°C/min as specified)
            optimal_cooling_rate = np.random.uniform(1.0, 2.0)
            
            # Alternative cooling rates for comparison
            cooling_rates = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]  # °C/min
            
            # Material properties after sintering
            density = np.random.uniform(0.92, 0.99)
            grain_size = np.random.uniform(0.5, 8.0)  # μm
            porosity = 1 - density
            
            # Mechanical properties
            hardness = np.random.uniform(15, 25)  # GPa
            fracture_toughness = np.random.uniform(3, 8)  # MPa·m^0.5
            
            # Processing conditions
            atmosphere = np.random.choice(['air', 'nitrogen', 'argon', 'vacuum'])
            pressure = np.random.uniform(0.1, 10)  # MPa (for hot pressing)
            
            # Quality metrics
            surface_roughness = np.random.uniform(0.1, 2.0)  # μm
            dimensional_accuracy = np.random.uniform(0.1, 1.0)  # %
            
            sintering_data.append({
                'sample_id': f'SP_{i:03d}',
                'max_temperature': max_temperature,
                'holding_time': holding_time,
                'optimal_cooling_rate': optimal_cooling_rate,
                'cooling_rates_tested': cooling_rates,
                'density': density,
                'grain_size': grain_size,
                'porosity': porosity,
                'hardness': hardness,
                'fracture_toughness': fracture_toughness,
                'atmosphere': atmosphere,
                'pressure': pressure,
                'surface_roughness': surface_roughness,
                'dimensional_accuracy': dimensional_accuracy,
                'processing_time': holding_time + (max_temperature - 25) / optimal_cooling_rate,  # total time
                'energy_consumption': max_temperature * holding_time * np.random.uniform(0.8, 1.2)  # arbitrary units
            })
        
        self.data['sintering_parameters'] = sintering_data
        return sintering_data
    
    def generate_geometric_variations(self, n_samples=40):
        """
        Generate geometric design variations data.
        Compares bow-shaped vs rectangular channels.
        """
        print("Generating geometric design variations...")
        
        geometric_data = []
        
        for i in range(n_samples):
            # Channel type
            channel_type = np.random.choice(['bow_shaped', 'rectangular'])
            
            # Basic dimensions
            length = np.random.uniform(10, 100)  # mm
            width = np.random.uniform(1, 5)  # mm
            height = np.random.uniform(0.5, 3)  # mm
            
            if channel_type == 'bow_shaped':
                # Bow-shaped channel parameters
                curvature_radius = np.random.uniform(5, 50)  # mm
                arc_angle = np.random.uniform(30, 180)  # degrees
                channel_area = 0.5 * curvature_radius**2 * np.radians(arc_angle) - 0.5 * curvature_radius**2 * np.sin(np.radians(arc_angle))
            else:
                # Rectangular channel parameters
                channel_area = width * height
                curvature_radius = np.inf
                arc_angle = 0
            
            # Flow characteristics
            flow_rate = np.random.uniform(0.1, 10)  # L/min
            pressure_drop = np.random.uniform(0.1, 5)  # kPa
            
            # Heat transfer properties
            heat_transfer_coefficient = np.random.uniform(50, 500)  # W/m²·K
            nusselt_number = np.random.uniform(1, 10)
            
            # Manufacturing parameters
            surface_finish = np.random.uniform(0.1, 2.0)  # μm Ra
            dimensional_tolerance = np.random.uniform(0.01, 0.1)  # mm
            
            # Performance metrics
            efficiency = np.random.uniform(0.7, 0.95)
            pressure_recovery = np.random.uniform(0.6, 0.9)
            
            # Stress concentration factors
            stress_concentration = np.random.uniform(1.2, 3.0)
            
            # Computational fluid dynamics (CFD) results
            reynolds_number = np.random.uniform(100, 10000)
            friction_factor = np.random.uniform(0.01, 0.1)
            
            geometric_data.append({
                'sample_id': f'GV_{i:03d}',
                'channel_type': channel_type,
                'length': length,
                'width': width,
                'height': height,
                'curvature_radius': curvature_radius,
                'arc_angle': arc_angle,
                'channel_area': channel_area,
                'flow_rate': flow_rate,
                'pressure_drop': pressure_drop,
                'heat_transfer_coefficient': heat_transfer_coefficient,
                'nusselt_number': nusselt_number,
                'surface_finish': surface_finish,
                'dimensional_tolerance': dimensional_tolerance,
                'efficiency': efficiency,
                'pressure_recovery': pressure_recovery,
                'stress_concentration': stress_concentration,
                'reynolds_number': reynolds_number,
                'friction_factor': friction_factor,
                'manufacturing_cost': np.random.uniform(10, 100),  # arbitrary units
                'design_complexity': np.random.uniform(1, 10)  # 1=simple, 10=complex
            })
        
        self.data['geometric_variations'] = geometric_data
        return geometric_data
    
    def generate_combined_dataset(self):
        """Generate all datasets and combine into a comprehensive validation dataset."""
        print("Generating comprehensive validation dataset...")
        
        # Generate all individual datasets
        stress_strain = self.generate_stress_strain_profiles()
        crack_depth = self.generate_crack_depth_data()
        sintering = self.generate_sintering_parameters()
        geometric = self.generate_geometric_variations()
        
        # Create summary statistics
        summary = {
            'dataset_info': {
                'generation_date': datetime.now().isoformat(),
                'total_samples': len(stress_strain) + len(crack_depth) + len(sintering) + len(geometric),
                'stress_strain_samples': len(stress_strain),
                'crack_depth_samples': len(crack_depth),
                'sintering_samples': len(sintering),
                'geometric_samples': len(geometric)
            },
            'data_quality_metrics': {
                'stress_strain_avg_r_squared': np.mean([p['r_squared'] for p in stress_strain]),
                'crack_depth_avg_error': np.mean([c['measurement_error'] for c in crack_depth]),
                'sintering_avg_density': np.mean([s['density'] for s in sintering]),
                'geometric_avg_efficiency': np.mean([g['efficiency'] for g in geometric])
            }
        }
        
        self.data['summary'] = summary
        
        return self.data
    
    def save_datasets(self, output_dir='/workspace'):
        """Save all datasets to files."""
        print("Saving datasets...")
        
        # Save as JSON files
        with open(f'{output_dir}/stress_strain_profiles.json', 'w') as f:
            json.dump(self.data['stress_strain_profiles'], f, indent=2, default=str)
        
        with open(f'{output_dir}/crack_depth_data.json', 'w') as f:
            json.dump(self.data['crack_depth_data'], f, indent=2, default=str)
        
        with open(f'{output_dir}/sintering_parameters.json', 'w') as f:
            json.dump(self.data['sintering_parameters'], f, indent=2, default=str)
        
        with open(f'{output_dir}/geometric_variations.json', 'w') as f:
            json.dump(self.data['geometric_variations'], f, indent=2, default=str)
        
        with open(f'{output_dir}/dataset_summary.json', 'w') as f:
            json.dump(self.data['summary'], f, indent=2, default=str)
        
        # Save as CSV files for easy analysis
        self._save_to_csv(output_dir)
        
        print(f"Datasets saved to {output_dir}")
    
    def _save_to_csv(self, output_dir):
        """Save datasets as CSV files for easy analysis."""
        
        # Stress-strain data
        ss_data = []
        for profile in self.data['stress_strain_profiles']:
            for i, (strain, stress_fem, stress_exp) in enumerate(zip(
                profile['strain'], profile['stress_fem'], profile['stress_exp']
            )):
                ss_data.append({
                    'sample_id': profile['sample_info']['sample_id'],
                    'material': profile['sample_info']['material'],
                    'strain': strain,
                    'stress_fem': stress_fem,
                    'stress_exp': stress_exp,
                    'rmse': profile['rmse'],
                    'r_squared': profile['r_squared']
                })
        
        pd.DataFrame(ss_data).to_csv(f'{output_dir}/stress_strain_data.csv', index=False)
        
        # Crack depth data
        cd_data = []
        for crack in self.data['crack_depth_data']:
            cd_data.append({
                'sample_id': crack['sample_id'],
                'material': crack['material'],
                'true_depth': crack['true_depth'],
                'predicted_depth': crack['predicted_depth'],
                'measurement_error': crack['measurement_error'],
                'confidence_interval': crack['confidence_interval']
            })
        
        pd.DataFrame(cd_data).to_csv(f'{output_dir}/crack_depth_data.csv', index=False)
        
        # Sintering parameters
        sp_data = []
        for sinter in self.data['sintering_parameters']:
            sp_data.append({
                'sample_id': sinter['sample_id'],
                'max_temperature': sinter['max_temperature'],
                'optimal_cooling_rate': sinter['optimal_cooling_rate'],
                'density': sinter['density'],
                'grain_size': sinter['grain_size'],
                'hardness': sinter['hardness'],
                'fracture_toughness': sinter['fracture_toughness']
            })
        
        pd.DataFrame(sp_data).to_csv(f'{output_dir}/sintering_parameters.csv', index=False)
        
        # Geometric variations
        gv_data = []
        for geom in self.data['geometric_variations']:
            gv_data.append({
                'sample_id': geom['sample_id'],
                'channel_type': geom['channel_type'],
                'length': geom['length'],
                'width': geom['width'],
                'height': geom['height'],
                'efficiency': geom['efficiency'],
                'pressure_drop': geom['pressure_drop'],
                'stress_concentration': geom['stress_concentration']
            })
        
        pd.DataFrame(gv_data).to_csv(f'{output_dir}/geometric_variations.csv', index=False)
    
    def create_visualizations(self, output_dir='/workspace'):
        """Create visualization plots for the datasets."""
        print("Creating visualizations...")
        
        # Stress-strain comparison plot
        plt.figure(figsize=(12, 8))
        for i, profile in enumerate(self.data['stress_strain_profiles'][:5]):  # Show first 5 samples
            plt.subplot(2, 3, 1)
            plt.plot(profile['strain'], profile['stress_fem'], 'b-', alpha=0.7, label='FEM' if i == 0 else "")
            plt.plot(profile['strain'], profile['stress_exp'], 'r--', alpha=0.7, label='Experimental' if i == 0 else "")
        
        plt.xlabel('Strain')
        plt.ylabel('Stress (Pa)')
        plt.title('FEM vs Experimental Stress-Strain')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Crack depth comparison
        plt.subplot(2, 3, 2)
        true_depths = [c['true_depth'] for c in self.data['crack_depth_data']]
        pred_depths = [c['predicted_depth'] for c in self.data['crack_depth_data']]
        plt.scatter(true_depths, pred_depths, alpha=0.6)
        plt.plot([min(true_depths), max(true_depths)], [min(true_depths), max(true_depths)], 'r--', label='Perfect prediction')
        plt.xlabel('True Crack Depth (μm)')
        plt.ylabel('Predicted Crack Depth (μm)')
        plt.title('Crack Depth: XRD vs Model')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Sintering parameters
        plt.subplot(2, 3, 3)
        cooling_rates = [s['optimal_cooling_rate'] for s in self.data['sintering_parameters']]
        densities = [s['density'] for s in self.data['sintering_parameters']]
        plt.scatter(cooling_rates, densities, alpha=0.6)
        plt.xlabel('Cooling Rate (°C/min)')
        plt.ylabel('Density')
        plt.title('Sintering: Cooling Rate vs Density')
        plt.grid(True, alpha=0.3)
        
        # Geometric variations
        plt.subplot(2, 3, 4)
        bow_shaped = [g for g in self.data['geometric_variations'] if g['channel_type'] == 'bow_shaped']
        rectangular = [g for g in self.data['geometric_variations'] if g['channel_type'] == 'rectangular']
        
        plt.scatter([g['length'] for g in bow_shaped], [g['efficiency'] for g in bow_shaped], 
                   alpha=0.6, label='Bow-shaped', color='blue')
        plt.scatter([g['length'] for g in rectangular], [g['efficiency'] for g in rectangular], 
                   alpha=0.6, label='Rectangular', color='red')
        plt.xlabel('Channel Length (mm)')
        plt.ylabel('Efficiency')
        plt.title('Geometric Design: Length vs Efficiency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Error distribution
        plt.subplot(2, 3, 5)
        errors = [c['measurement_error'] for c in self.data['crack_depth_data']]
        plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Measurement Error (%)')
        plt.ylabel('Frequency')
        plt.title('Crack Depth Prediction Error Distribution')
        plt.grid(True, alpha=0.3)
        
        # Material comparison
        plt.subplot(2, 3, 6)
        materials = {}
        for profile in self.data['stress_strain_profiles']:
            material = profile['sample_info']['material']
            if material not in materials:
                materials[material] = []
            materials[material].append(profile['r_squared'])
        
        material_names = list(materials.keys())
        r_squared_values = [materials[m] for m in material_names]
        plt.boxplot(r_squared_values, labels=material_names)
        plt.ylabel('R² Value')
        plt.title('Model Accuracy by Material')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/validation_dataset_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}/validation_dataset_plots.png")

def main():
    """Main function to generate the complete validation dataset."""
    print("🔬 Optimization and Validation Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = ValidationDatasetGenerator(seed=42)
    
    # Generate comprehensive dataset
    dataset = generator.generate_combined_dataset()
    
    # Save all datasets
    generator.save_datasets()
    
    # Create visualizations
    generator.create_visualizations()
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"Total samples: {dataset['summary']['dataset_info']['total_samples']}")
    print(f"Stress-strain profiles: {dataset['summary']['dataset_info']['stress_strain_samples']}")
    print(f"Crack depth estimates: {dataset['summary']['dataset_info']['crack_depth_samples']}")
    print(f"Sintering parameters: {dataset['summary']['dataset_info']['sintering_samples']}")
    print(f"Geometric variations: {dataset['summary']['dataset_info']['geometric_samples']}")
    
    print(f"\n📈 Data Quality Metrics:")
    print(f"Average R² (stress-strain): {dataset['summary']['data_quality_metrics']['stress_strain_avg_r_squared']:.3f}")
    print(f"Average crack depth error: {dataset['summary']['data_quality_metrics']['crack_depth_avg_error']:.1f}%")
    print(f"Average sintered density: {dataset['summary']['data_quality_metrics']['sintering_avg_density']:.3f}")
    print(f"Average geometric efficiency: {dataset['summary']['data_quality_metrics']['geometric_avg_efficiency']:.3f}")
    
    print("\n✅ Dataset generation complete!")
    print("Files saved:")
    print("  - stress_strain_profiles.json")
    print("  - crack_depth_data.json") 
    print("  - sintering_parameters.json")
    print("  - geometric_variations.json")
    print("  - dataset_summary.json")
    print("  - *.csv files for analysis")
    print("  - validation_dataset_plots.png")

if __name__ == "__main__":
    main()