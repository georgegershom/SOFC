#!/usr/bin/env python3
"""
Atomic-Scale Simulation Dataset Usage Examples
Demonstrates how to load and use the generated dataset for various applications
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats

class DatasetUsageExamples:
    """Examples of how to use the atomic-scale simulation dataset"""
    
    def __init__(self, data_dir='atomic_simulation_dataset'):
        self.data_dir = data_dir
        self.load_datasets()
    
    def load_datasets(self):
        """Load all dataset components"""
        print("Loading atomic-scale simulation datasets...")
        
        # Load DFT data
        self.dft_formation = pd.read_csv(f'{self.data_dir}/dft_formation_energies.csv')
        self.activation_barriers = pd.read_csv(f'{self.data_dir}/activation_barriers.csv')
        self.surface_energies = pd.read_csv(f'{self.data_dir}/surface_energies.csv')
        
        # Load MD data
        self.gb_sliding = pd.read_csv(f'{self.data_dir}/md_data/grain_boundary_sliding.csv')
        self.disl_mobility = pd.read_csv(f'{self.data_dir}/md_data/dislocation_mobility.csv')
        self.ff_params = pd.read_csv(f'{self.data_dir}/md_data/force_field_parameters.csv')
        
        # Load trajectory data
        with open(f'{self.data_dir}/md_data/sample_trajectories.json', 'r') as f:
            self.trajectories = json.load(f)
        
        print("✅ All datasets loaded successfully!")
    
    def example_1_diffusion_coefficient_calculation(self):
        """Example 1: Calculate diffusion coefficients from activation barriers"""
        print("\n" + "="*60)
        print("EXAMPLE 1: Diffusion Coefficient Calculation")
        print("="*60)
        
        # Constants
        k_B = 8.617e-5  # eV/K (Boltzmann constant)
        
        # Filter for vacancy migration data
        vacancy_data = self.activation_barriers[
            self.activation_barriers['diffusion_mechanism'] == 'vacancy_migration'
        ].copy()
        
        # Calculate diffusion coefficients: D = D0 * exp(-Q/kT)
        # Where D0 is related to attempt frequency and jump distance
        vacancy_data['diffusion_coeff_m2_s'] = (
            vacancy_data['attempt_frequency_Hz'] * 
            (vacancy_data['migration_path_length_A'] * 1e-10)**2 * 
            np.exp(-vacancy_data['activation_barrier_eV'] / 
                   (k_B * vacancy_data['temperature_K']))
        )
        
        # Group by material and calculate average
        diffusion_by_material = vacancy_data.groupby('material').agg({
            'diffusion_coeff_m2_s': ['mean', 'std'],
            'activation_barrier_eV': 'mean',
            'temperature_K': 'mean'
        }).round(6)
        
        print("Calculated Diffusion Coefficients by Material:")
        print(diffusion_by_material)
        
        # Arrhenius plot for aluminum
        al_data = vacancy_data[vacancy_data['material'] == 'Al']
        if len(al_data) > 0:
            plt.figure(figsize=(8, 6))
            plt.semilogy(1000/al_data['temperature_K'], al_data['diffusion_coeff_m2_s'], 'o')
            plt.xlabel('1000/T (K⁻¹)')
            plt.ylabel('Diffusion Coefficient (m²/s)')
            plt.title('Arrhenius Plot for Al Vacancy Diffusion')
            plt.grid(True)
            plt.savefig(f'{self.data_dir}/figures/arrhenius_plot_example.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Arrhenius plot saved as arrhenius_plot_example.png")
    
    def example_2_creep_rate_prediction(self):
        """Example 2: Predict creep rates using grain boundary sliding data"""
        print("\n" + "="*60)
        print("EXAMPLE 2: Creep Rate Prediction Model")
        print("="*60)
        
        # Prepare features for machine learning
        features = ['temperature_K', 'applied_stress_MPa', 'gb_angle_deg', 'gb_energy_J_m2']
        target = 'avg_sliding_rate_A_ps'
        
        # Encode categorical variables
        gb_encoded = pd.get_dummies(self.gb_sliding['material'], prefix='material')
        gb_type_encoded = pd.get_dummies(self.gb_sliding['gb_type'], prefix='gb_type')
        
        X = pd.concat([
            self.gb_sliding[features],
            gb_encoded,
            gb_type_encoded
        ], axis=1)
        
        y = np.log10(self.gb_sliding[target] + 1e-12)  # Log transform for better modeling
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train random forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Model Performance:")
        print(f"  R² Score: {r2:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for i, row in feature_importance.head().iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
        
        # Prediction vs actual plot
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual log₁₀(Sliding Rate)')
        plt.ylabel('Predicted log₁₀(Sliding Rate)')
        plt.title(f'Creep Rate Prediction (R² = {r2:.3f})')
        plt.grid(True)
        plt.savefig(f'{self.data_dir}/figures/creep_prediction_example.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Prediction plot saved as creep_prediction_example.png")
    
    def example_3_phase_field_parameterization(self):
        """Example 3: Extract parameters for phase-field modeling"""
        print("\n" + "="*60)
        print("EXAMPLE 3: Phase-Field Model Parameterization")
        print("="*60)
        
        # Calculate interface energies from surface energy data
        interface_energies = self.surface_energies.groupby('material').agg({
            'surface_energy_J_m2': ['mean', 'std']
        }).round(4)
        
        print("Interface Energies for Phase-Field Models:")
        print(interface_energies)
        
        # Calculate mobility parameters from dislocation data
        mobility_params = self.disl_mobility.groupby('material').agg({
            'base_mobility_m2_Pa_s': ['mean', 'std'],
            'avg_velocity_A_ps': ['mean', 'std']
        }).round(8)
        
        print("\nMobility Parameters:")
        print(mobility_params)
        
        # Generate parameter file for phase-field simulation
        pf_params = {}
        
        for material in self.surface_energies['material'].unique():
            # Interface energy
            gamma = self.surface_energies[
                self.surface_energies['material'] == material
            ]['surface_energy_J_m2'].mean()
            
            # Mobility (from dislocation data)
            mobility_data = self.disl_mobility[
                self.disl_mobility['material'] == material
            ]
            if len(mobility_data) > 0:
                mobility = mobility_data['base_mobility_m2_Pa_s'].mean()
            else:
                mobility = 1e-6  # Default value
            
            # Formation energy (for nucleation)
            formation_data = self.dft_formation[
                (self.dft_formation['material'] == material) &
                (self.dft_formation['defect_type'] == 'vacancy')
            ]
            if len(formation_data) > 0:
                formation_energy = formation_data['formation_energy_eV'].mean()
            else:
                formation_energy = 1.0  # Default value
            
            pf_params[material] = {
                'interface_energy_J_m2': float(gamma),
                'mobility_m2_Pa_s': float(mobility),
                'formation_energy_eV': float(formation_energy),
                'interface_width_m': 1e-9,  # Typical nanoscale width
                'gradient_coefficient': float(gamma * 1e-9)  # γ * interface_width
            }
        
        # Save parameters
        with open(f'{self.data_dir}/phase_field_parameters.json', 'w') as f:
            json.dump(pf_params, f, indent=2)
        
        print(f"\n📄 Phase-field parameters saved to phase_field_parameters.json")
        print("Example parameters for Al:")
        if 'Al' in pf_params:
            for key, value in pf_params['Al'].items():
                print(f"  {key}: {value}")
    
    def example_4_temperature_dependence_analysis(self):
        """Example 4: Analyze temperature dependence of properties"""
        print("\n" + "="*60)
        print("EXAMPLE 4: Temperature Dependence Analysis")
        print("="*60)
        
        # Analyze formation energy temperature dependence
        temp_bins = pd.cut(self.dft_formation['temperature_K'], bins=5)
        temp_analysis = self.dft_formation.groupby([temp_bins, 'material']).agg({
            'formation_energy_eV': ['mean', 'std', 'count']
        }).round(4)
        
        print("Formation Energy vs Temperature (by material):")
        print(temp_analysis.head(10))
        
        # Statistical analysis for aluminum
        al_data = self.dft_formation[self.dft_formation['material'] == 'Al']
        if len(al_data) > 10:
            # Linear regression: E_f = E_f0 + α*T
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                al_data['temperature_K'], al_data['formation_energy_eV']
            )
            
            print(f"\nTemperature Dependence for Al Formation Energy:")
            print(f"  E_f = {intercept:.4f} + {slope:.6f} * T")
            print(f"  R² = {r_value**2:.4f}")
            print(f"  p-value = {p_value:.4f}")
            
            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(al_data['temperature_K'], al_data['formation_energy_eV'], alpha=0.6)
            temp_range = np.linspace(al_data['temperature_K'].min(), al_data['temperature_K'].max(), 100)
            plt.plot(temp_range, intercept + slope * temp_range, 'r-', linewidth=2)
            plt.xlabel('Temperature (K)')
            plt.ylabel('Formation Energy (eV)')
            plt.title(f'Al Formation Energy vs Temperature (R² = {r_value**2:.3f})')
            plt.grid(True)
            plt.savefig(f'{self.data_dir}/figures/temperature_dependence_example.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("📊 Temperature dependence plot saved as temperature_dependence_example.png")
    
    def example_5_trajectory_analysis(self):
        """Example 5: Analyze MD trajectory data"""
        print("\n" + "="*60)
        print("EXAMPLE 5: MD Trajectory Analysis")
        print("="*60)
        
        if not self.trajectories:
            print("No trajectory data available")
            return
        
        # Analyze first trajectory
        traj_id = list(self.trajectories.keys())[0]
        traj = self.trajectories[traj_id]
        
        print(f"Analyzing trajectory: {traj_id}")
        print(f"Material: {traj['metadata']['material']}")
        print(f"Number of atoms: {traj['metadata']['n_atoms']}")
        print(f"Number of steps: {traj['metadata']['n_steps']}")
        print(f"Total time: {traj['metadata']['total_time_ps']:.3f} ps")
        
        # Convert to numpy arrays for analysis
        energies = np.array(traj['total_energy_eV'])
        temperatures = np.array(traj['temperature_K'])
        
        # Calculate energy statistics
        print(f"\nEnergy Statistics:")
        print(f"  Mean total energy: {np.mean(energies):.2f} ± {np.std(energies):.2f} eV")
        print(f"  Energy range: {np.min(energies):.2f} to {np.max(energies):.2f} eV")
        
        # Calculate temperature statistics
        print(f"\nTemperature Statistics:")
        print(f"  Mean temperature: {np.mean(temperatures):.1f} ± {np.std(temperatures):.1f} K")
        
        # Plot energy evolution
        time_steps = np.arange(len(energies)) * traj['metadata']['timestep_fs'] / 1000  # Convert to ps
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_steps, energies)
        plt.xlabel('Time (ps)')
        plt.ylabel('Total Energy (eV)')
        plt.title('Energy Evolution')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(time_steps, temperatures)
        plt.xlabel('Time (ps)')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature Evolution')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.data_dir}/figures/trajectory_analysis_example.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Trajectory analysis plot saved as trajectory_analysis_example.png")
    
    def run_all_examples(self):
        """Run all usage examples"""
        print("🚀 Running Atomic-Scale Simulation Dataset Usage Examples")
        print("="*80)
        
        self.example_1_diffusion_coefficient_calculation()
        self.example_2_creep_rate_prediction()
        self.example_3_phase_field_parameterization()
        self.example_4_temperature_dependence_analysis()
        self.example_5_trajectory_analysis()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("📊 Check the figures/ directory for generated plots")
        print("📄 Check phase_field_parameters.json for extracted parameters")

def main():
    """Main function to run usage examples"""
    examples = DatasetUsageExamples()
    examples.run_all_examples()

if __name__ == "__main__":
    main()