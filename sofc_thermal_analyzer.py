#!/usr/bin/env python3
"""
SOFC Thermal Data Analyzer and Visualizer
Analyzes thermal history data for residual stress calculation and delamination risk assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import json
import os
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class SOFCThermalAnalyzer:
    def __init__(self, data_directory='sofc_thermal_data'):
        """Initialize the thermal analyzer with data directory."""
        self.data_dir = data_directory
        self.load_data()
        
        # Material properties for stress calculations
        self.material_properties = {
            'anode': {
                'youngs_modulus': 45e9,  # Pa
                'thermal_expansion': 12.5e-6,  # /K
                'poisson_ratio': 0.3
            },
            'cathode': {
                'youngs_modulus': 50e9,  # Pa
                'thermal_expansion': 13.2e-6,  # /K
                'poisson_ratio': 0.32
            },
            'electrolyte': {
                'youngs_modulus': 200e9,  # Pa
                'thermal_expansion': 10.5e-6,  # /K
                'poisson_ratio': 0.25
            },
            'interconnect': {
                'youngs_modulus': 210e9,  # Pa
                'thermal_expansion': 11.8e-6,  # /K
                'poisson_ratio': 0.28
            }
        }
    
    def load_data(self):
        """Load all thermal data files."""
        print("Loading thermal data...")
        
        try:
            self.sintering_data = pd.read_csv(f'{self.data_dir}/sintering_thermal_data.csv')
            self.cycling_data = pd.read_csv(f'{self.data_dir}/thermal_cycling_data.csv')
            self.steady_data = pd.read_csv(f'{self.data_dir}/steady_state_thermal_data.csv')
            
            # Load spatial data
            spatial_data = np.load(f'{self.data_dir}/spatial_thermal_data.npz', allow_pickle=True)
            self.spatial_data = {key: spatial_data[key] for key in spatial_data.files}
            
            # Load metadata
            with open(f'{self.data_dir}/metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            print("Data loaded successfully!")
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please run the data generator first.")
            raise
    
    def analyze_sintering_stresses(self):
        """Analyze residual stresses from sintering process."""
        print("Analyzing sintering-induced residual stresses...")
        
        # Calculate thermal gradients and stress indicators
        sintering_analysis = self.sintering_data.copy()
        
        # Calculate stress indicators
        sintering_analysis['thermal_stress_mpa'] = self._calculate_thermal_stress_mpa(
            sintering_analysis['temperature_gradient_C']
        )
        
        # Identify critical phases
        critical_phases = sintering_analysis.groupby('process_stage').agg({
            'temperature_gradient_C': ['mean', 'max'],
            'thermal_stress_mpa': ['mean', 'max'],
            'center_temperature_C': ['mean', 'max']
        }).round(2)
        
        # Calculate residual stress accumulation
        sintering_analysis['residual_stress_accumulation'] = self._calculate_residual_stress_accumulation(
            sintering_analysis
        )
        
        return sintering_analysis, critical_phases
    
    def analyze_thermal_cycling(self):
        """Analyze thermal cycling effects on delamination risk."""
        print("Analyzing thermal cycling effects...")
        
        cycling_analysis = self.cycling_data.copy()
        
        # Calculate cycle-specific metrics
        cycle_metrics = []
        
        for cycle in cycling_analysis['cycle_number'].unique():
            cycle_data = cycling_analysis[cycling_analysis['cycle_number'] == cycle]
            
            # Calculate thermal shock indicators
            startup_data = cycle_data[cycle_data['phase'] == 'startup']
            shutdown_data = cycle_data[cycle_data['phase'] == 'shutdown']
            
            if not startup_data.empty and not shutdown_data.empty:
                max_heating_rate = np.max(np.diff(startup_data['center_temperature_C']) / 
                                        np.diff(startup_data['time_hours']))
                max_cooling_rate = np.min(np.diff(shutdown_data['center_temperature_C']) / 
                                        np.diff(shutdown_data['time_hours']))
                
                max_gradient = np.max(cycle_data['temperature_gradient_C'])
                avg_stress = np.mean(cycle_data['thermal_stress_indicator'])
                
                cycle_metrics.append({
                    'cycle_number': cycle,
                    'max_heating_rate_C_per_hour': max_heating_rate * 60,  # Convert to C/hour
                    'max_cooling_rate_C_per_hour': abs(max_cooling_rate) * 60,
                    'max_temperature_gradient_C': max_gradient,
                    'average_thermal_stress': avg_stress,
                    'delamination_risk_score': self._calculate_delamination_risk(
                        max_heating_rate * 60, abs(max_cooling_rate) * 60, max_gradient
                    )
                })
        
        cycle_metrics_df = pd.DataFrame(cycle_metrics)
        
        # Analyze cumulative damage
        cycling_analysis['cumulative_damage'] = self._calculate_cumulative_damage(cycling_analysis)
        
        return cycling_analysis, cycle_metrics_df
    
    def analyze_steady_state_gradients(self):
        """Analyze steady-state temperature gradients."""
        print("Analyzing steady-state temperature gradients...")
        
        steady_analysis = self.steady_data.copy()
        
        # Statistical analysis of temperature gradients
        gradient_stats = {
            'mean_gradient': np.mean(steady_analysis['temperature_gradient_C']),
            'std_gradient': np.std(steady_analysis['temperature_gradient_C']),
            'max_gradient': np.max(steady_analysis['temperature_gradient_C']),
            'min_gradient': np.min(steady_analysis['temperature_gradient_C']),
            'gradient_95th_percentile': np.percentile(steady_analysis['temperature_gradient_C'], 95)
        }
        
        # Identify periods of high thermal stress
        high_stress_threshold = gradient_stats['mean_gradient'] + 2 * gradient_stats['std_gradient']
        high_stress_periods = steady_analysis[
            steady_analysis['temperature_gradient_C'] > high_stress_threshold
        ]
        
        # Calculate long-term degradation indicators
        steady_analysis['degradation_indicator'] = self._calculate_degradation_indicator(steady_analysis)
        
        # Analyze correlation with operating parameters
        correlations = steady_analysis[['temperature_gradient_C', 'current_density_A_cm2', 
                                      'fuel_utilization', 'thermal_stress_indicator']].corr()
        
        return steady_analysis, gradient_stats, high_stress_periods, correlations
    
    def _calculate_thermal_stress_mpa(self, temperature_gradients):
        """Calculate thermal stress in MPa from temperature gradients."""
        # Use average material properties
        avg_youngs = np.mean([props['youngs_modulus'] for props in self.material_properties.values()])
        avg_alpha = np.mean([props['thermal_expansion'] for props in self.material_properties.values()])
        avg_poisson = np.mean([props['poisson_ratio'] for props in self.material_properties.values()])
        
        # Thermal stress formula: σ = E * α * ΔT / (1 - ν)
        thermal_stress_pa = (avg_youngs * avg_alpha * temperature_gradients) / (1 - avg_poisson)
        
        # Convert to MPa
        return thermal_stress_pa / 1e6
    
    def _calculate_residual_stress_accumulation(self, sintering_data):
        """Calculate cumulative residual stress accumulation during sintering."""
        # Simplified model based on temperature history
        stress_accumulation = np.zeros(len(sintering_data))
        
        for i in range(1, len(sintering_data)):
            temp_change = abs(sintering_data.iloc[i]['center_temperature_C'] - 
                            sintering_data.iloc[i-1]['center_temperature_C'])
            gradient = sintering_data.iloc[i]['temperature_gradient_C']
            
            # Accumulate stress based on temperature changes and gradients
            stress_increment = 0.1 * temp_change + 0.05 * gradient
            stress_accumulation[i] = stress_accumulation[i-1] + stress_increment
        
        return stress_accumulation
    
    def _calculate_delamination_risk(self, heating_rate, cooling_rate, max_gradient):
        """Calculate delamination risk score based on thermal cycling parameters."""
        # Empirical risk model (0-100 scale)
        heating_risk = min(heating_rate / 100, 1.0) * 30  # Max 30 points
        cooling_risk = min(cooling_rate / 150, 1.0) * 40  # Max 40 points
        gradient_risk = min(max_gradient / 50, 1.0) * 30   # Max 30 points
        
        return heating_risk + cooling_risk + gradient_risk
    
    def _calculate_cumulative_damage(self, cycling_data):
        """Calculate cumulative damage from thermal cycling."""
        damage = np.zeros(len(cycling_data))
        
        for i in range(1, len(cycling_data)):
            stress_indicator = cycling_data.iloc[i]['thermal_stress_indicator']
            # Simple damage accumulation model
            damage_increment = stress_indicator / 1000  # Normalize
            damage[i] = damage[i-1] + damage_increment
        
        return damage
    
    def _calculate_degradation_indicator(self, steady_data):
        """Calculate long-term degradation indicator."""
        # Based on accumulated thermal stress over time
        degradation = np.zeros(len(steady_data))
        
        for i in range(1, len(steady_data)):
            stress = steady_data.iloc[i]['thermal_stress_indicator']
            time_factor = steady_data.iloc[i]['time_hours'] / 1000  # Normalize time
            
            degradation_increment = stress * time_factor * 0.01
            degradation[i] = degradation[i-1] + degradation_increment
        
        return degradation
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of thermal data."""
        print("Creating comprehensive visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create output directory for plots
        os.makedirs('thermal_analysis_plots', exist_ok=True)
        
        # 1. Sintering Process Visualization
        self._plot_sintering_analysis()
        
        # 2. Thermal Cycling Analysis
        self._plot_thermal_cycling_analysis()
        
        # 3. Steady-State Analysis
        self._plot_steady_state_analysis()
        
        # 4. Spatial Temperature Distributions
        self._plot_spatial_distributions()
        
        # 5. Comprehensive Summary Dashboard
        self._create_summary_dashboard()
        
        print("Visualizations saved to 'thermal_analysis_plots/' directory")
    
    def _plot_sintering_analysis(self):
        """Create sintering process analysis plots."""
        sintering_analysis, critical_phases = self.analyze_sintering_stresses()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SOFC Sintering Process Thermal Analysis', fontsize=16, fontweight='bold')
        
        # Temperature profile
        axes[0,0].plot(sintering_analysis['time_hours'], sintering_analysis['center_temperature_C'], 
                      'b-', linewidth=2, label='Center Temperature')
        axes[0,0].fill_between(sintering_analysis['time_hours'], 
                              sintering_analysis['min_temperature_C'],
                              sintering_analysis['max_temperature_C'], 
                              alpha=0.3, label='Temperature Range')
        axes[0,0].set_xlabel('Time (hours)')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].set_title('Temperature Profile During Sintering')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Temperature gradients
        axes[0,1].plot(sintering_analysis['time_hours'], sintering_analysis['temperature_gradient_C'], 
                      'r-', linewidth=2)
        axes[0,1].set_xlabel('Time (hours)')
        axes[0,1].set_ylabel('Temperature Gradient (°C)')
        axes[0,1].set_title('Temperature Gradients')
        axes[0,1].grid(True, alpha=0.3)
        
        # Thermal stress
        axes[1,0].plot(sintering_analysis['time_hours'], sintering_analysis['thermal_stress_mpa'], 
                      'g-', linewidth=2)
        axes[1,0].set_xlabel('Time (hours)')
        axes[1,0].set_ylabel('Thermal Stress (MPa)')
        axes[1,0].set_title('Calculated Thermal Stress')
        axes[1,0].grid(True, alpha=0.3)
        
        # Residual stress accumulation
        axes[1,1].plot(sintering_analysis['time_hours'], sintering_analysis['residual_stress_accumulation'], 
                      'purple', linewidth=2)
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].set_ylabel('Cumulative Residual Stress')
        axes[1,1].set_title('Residual Stress Accumulation')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('thermal_analysis_plots/sintering_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Critical phases summary
        fig, ax = plt.subplots(figsize=(12, 8))
        critical_phases_plot = critical_phases['thermal_stress_mpa']['max'].sort_values(ascending=True)
        critical_phases_plot.plot(kind='barh', ax=ax, color='coral')
        ax.set_xlabel('Maximum Thermal Stress (MPa)')
        ax.set_title('Maximum Thermal Stress by Sintering Stage')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('thermal_analysis_plots/sintering_critical_phases.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_thermal_cycling_analysis(self):
        """Create thermal cycling analysis plots."""
        cycling_analysis, cycle_metrics = self.analyze_thermal_cycling()
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('SOFC Thermal Cycling Analysis', fontsize=16, fontweight='bold')
        
        # Temperature profiles for first 3 cycles
        for cycle in range(1, 4):
            cycle_data = cycling_analysis[cycling_analysis['cycle_number'] == cycle]
            axes[0,cycle-1].plot(cycle_data['time_hours'], cycle_data['center_temperature_C'], 
                               'b-', linewidth=2)
            axes[0,cycle-1].set_xlabel('Time (hours)')
            axes[0,cycle-1].set_ylabel('Temperature (°C)')
            axes[0,cycle-1].set_title(f'Cycle {cycle} Temperature Profile')
            axes[0,cycle-1].grid(True, alpha=0.3)
        
        # Delamination risk scores
        axes[1,0].bar(cycle_metrics['cycle_number'], cycle_metrics['delamination_risk_score'], 
                     color='red', alpha=0.7)
        axes[1,0].set_xlabel('Cycle Number')
        axes[1,0].set_ylabel('Delamination Risk Score')
        axes[1,0].set_title('Delamination Risk by Cycle')
        axes[1,0].grid(True, alpha=0.3)
        
        # Heating/cooling rates
        axes[1,1].plot(cycle_metrics['cycle_number'], cycle_metrics['max_heating_rate_C_per_hour'], 
                      'o-', label='Heating Rate', linewidth=2)
        axes[1,1].plot(cycle_metrics['cycle_number'], cycle_metrics['max_cooling_rate_C_per_hour'], 
                      's-', label='Cooling Rate', linewidth=2)
        axes[1,1].set_xlabel('Cycle Number')
        axes[1,1].set_ylabel('Rate (°C/hour)')
        axes[1,1].set_title('Maximum Heating/Cooling Rates')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Cumulative damage
        axes[1,2].plot(cycling_analysis['time_hours'], cycling_analysis['cumulative_damage'], 
                      'purple', linewidth=2)
        axes[1,2].set_xlabel('Time (hours)')
        axes[1,2].set_ylabel('Cumulative Damage')
        axes[1,2].set_title('Cumulative Thermal Damage')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('thermal_analysis_plots/thermal_cycling_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_steady_state_analysis(self):
        """Create steady-state analysis plots."""
        steady_analysis, gradient_stats, high_stress_periods, correlations = self.analyze_steady_state_gradients()
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SOFC Steady-State Operation Analysis', fontsize=16, fontweight='bold')
        
        # Temperature over time
        axes[0,0].plot(steady_analysis['time_hours'], steady_analysis['center_temperature_C'], 
                      'b-', linewidth=1, alpha=0.7)
        axes[0,0].set_xlabel('Time (hours)')
        axes[0,0].set_ylabel('Temperature (°C)')
        axes[0,0].set_title('Steady-State Temperature Profile')
        axes[0,0].grid(True, alpha=0.3)
        
        # Temperature gradient distribution
        axes[0,1].hist(steady_analysis['temperature_gradient_C'], bins=50, alpha=0.7, color='green')
        axes[0,1].axvline(gradient_stats['mean_gradient'], color='red', linestyle='--', 
                         label=f"Mean: {gradient_stats['mean_gradient']:.1f}°C")
        axes[0,1].set_xlabel('Temperature Gradient (°C)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Temperature Gradient Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Correlation heatmap
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Parameter Correlations')
        
        # Degradation indicator
        axes[1,1].plot(steady_analysis['time_hours'], steady_analysis['degradation_indicator'], 
                      'orange', linewidth=2)
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].set_ylabel('Degradation Indicator')
        axes[1,1].set_title('Long-term Degradation Trend')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('thermal_analysis_plots/steady_state_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spatial_distributions(self):
        """Create spatial temperature distribution plots."""
        # Load spatial coordinates
        x_coords = self.spatial_data['x_coordinates']
        y_coords = self.spatial_data['y_coordinates']
        
        # Plot spatial distributions for key time points
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SOFC Spatial Temperature Distributions', fontsize=16, fontweight='bold')
        
        # Sintering spatial data (select a few key points)
        sintering_spatial = self.spatial_data['sintering_spatial']
        if len(sintering_spatial) > 0:
            # Plot at different sintering stages
            indices = [0, len(sintering_spatial)//4, len(sintering_spatial)//2, 
                      3*len(sintering_spatial)//4, len(sintering_spatial)-1]
            
            for i, idx in enumerate(indices[:3]):
                if idx < len(sintering_spatial):
                    spatial_temp = sintering_spatial[idx]['spatial_temp']
                    im = axes[0,i].contourf(x_coords, y_coords, spatial_temp, levels=20, cmap='hot')
                    axes[0,i].set_title(f'Sintering: t={sintering_spatial[idx]["time_hours"]:.1f}h')
                    axes[0,i].set_xlabel('X (mm)')
                    axes[0,i].set_ylabel('Y (mm)')
                    plt.colorbar(im, ax=axes[0,i], label='Temperature (°C)')
        
        # Steady-state spatial data
        steady_spatial = self.spatial_data['steady_spatial']
        if len(steady_spatial) > 0:
            indices = [0, len(steady_spatial)//3, len(steady_spatial)-1]
            
            for i, idx in enumerate(indices):
                if idx < len(steady_spatial):
                    spatial_temp = steady_spatial[idx]['spatial_temp']
                    im = axes[1,i].contourf(x_coords, y_coords, spatial_temp, levels=20, cmap='viridis')
                    axes[1,i].set_title(f'Steady-State: t={steady_spatial[idx]["time_hours"]:.1f}h')
                    axes[1,i].set_xlabel('X (mm)')
                    axes[1,i].set_ylabel('Y (mm)')
                    plt.colorbar(im, ax=axes[1,i], label='Temperature (°C)')
        
        plt.tight_layout()
        plt.savefig('thermal_analysis_plots/spatial_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_summary_dashboard(self):
        """Create a comprehensive summary dashboard."""
        # Analyze all data
        sintering_analysis, critical_phases = self.analyze_sintering_stresses()
        cycling_analysis, cycle_metrics = self.analyze_thermal_cycling()
        steady_analysis, gradient_stats, high_stress_periods, correlations = self.analyze_steady_state_gradients()
        
        # Create summary statistics
        summary_stats = {
            'Sintering Process': {
                'Max Temperature': f"{sintering_analysis['center_temperature_C'].max():.0f}°C",
                'Max Gradient': f"{sintering_analysis['temperature_gradient_C'].max():.1f}°C",
                'Max Thermal Stress': f"{sintering_analysis['thermal_stress_mpa'].max():.1f} MPa",
                'Final Residual Stress': f"{sintering_analysis['residual_stress_accumulation'].iloc[-1]:.1f}"
            },
            'Thermal Cycling': {
                'Number of Cycles': len(cycle_metrics),
                'Avg Delamination Risk': f"{cycle_metrics['delamination_risk_score'].mean():.1f}",
                'Max Heating Rate': f"{cycle_metrics['max_heating_rate_C_per_hour'].max():.0f}°C/h",
                'Max Cooling Rate': f"{cycle_metrics['max_cooling_rate_C_per_hour'].max():.0f}°C/h"
            },
            'Steady-State Operation': {
                'Operating Duration': f"{steady_analysis['time_hours'].max():.0f} hours",
                'Avg Temperature': f"{steady_analysis['center_temperature_C'].mean():.0f}°C",
                'Avg Gradient': f"{gradient_stats['mean_gradient']:.1f}°C",
                'High Stress Events': len(high_stress_periods)
            }
        }
        
        # Save summary to JSON
        with open('thermal_analysis_plots/analysis_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Create text summary
        with open('thermal_analysis_plots/analysis_summary.txt', 'w') as f:
            f.write("SOFC THERMAL HISTORY ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for category, stats in summary_stats.items():
                f.write(f"{category}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("CRITICAL FINDINGS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"• Maximum thermal stress during sintering: {sintering_analysis['thermal_stress_mpa'].max():.1f} MPa\n")
            f.write(f"• Average delamination risk score: {cycle_metrics['delamination_risk_score'].mean():.1f}/100\n")
            f.write(f"• Steady-state temperature stability: ±{steady_analysis['center_temperature_C'].std():.1f}°C\n")
            f.write(f"• High thermal stress events: {len(high_stress_periods)} occurrences\n")
        
        print("Analysis summary saved to 'thermal_analysis_plots/analysis_summary.json' and '.txt'")

def main():
    """Main function to run thermal analysis."""
    print("SOFC Thermal Data Analyzer")
    print("=" * 40)
    
    try:
        analyzer = SOFCThermalAnalyzer()
        
        # Run comprehensive analysis
        analyzer.create_comprehensive_visualizations()
        
        print("\nAnalysis complete!")
        print("Check 'thermal_analysis_plots/' directory for all visualizations and summaries.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure to run the data generator first.")

if __name__ == "__main__":
    main()