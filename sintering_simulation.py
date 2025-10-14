"""
Advanced Sintering Profile Design and Stress-Shape Trade-off Analysis
====================================================================

This module implements a comprehensive simulation for SOFC sintering processes,
including thermal profile generation, stress analysis, and Pareto optimization
for process parameter selection.

Author: Advanced Materials Simulation Lab
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

@dataclass
class SinteringProfile:
    """Data class for sintering profile parameters"""
    name: str
    ramp_rate: float  # °C/min
    soak_temp: float  # °C
    soak_duration: float  # min
    cool_rate: float  # °C/min
    ambient_temp: float = 25.0  # °C

@dataclass
class MaterialProperties:
    """Material properties for SOFC simulation"""
    # Thermal properties
    thermal_conductivity: float = 2.5  # W/m·K
    specific_heat: float = 800.0  # J/kg·K
    density: float = 3200.0  # kg/m³
    
    # Mechanical properties
    youngs_modulus: float = 200e9  # Pa
    poisson_ratio: float = 0.25
    thermal_expansion_coeff: float = 12e-6  # 1/K
    
    # Sintering properties
    activation_energy: float = 450e3  # J/mol
    pre_exponential: float = 1e12  # 1/s
    gas_constant: float = 8.314  # J/mol·K

class SinteringSimulator:
    """Advanced sintering process simulator with thermal and mechanical analysis"""
    
    def __init__(self, material: MaterialProperties):
        self.material = material
        self.profiles = []
        self.results = {}
        
    def add_profile(self, profile: SinteringProfile):
        """Add a sintering profile to the simulation"""
        self.profiles.append(profile)
        
    def generate_temperature_profile(self, profile: SinteringProfile, time_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate temperature profile for given sintering parameters"""
        # Calculate time segments
        ramp_time = (profile.soak_temp - profile.ambient_temp) / profile.ramp_rate
        cool_time = (profile.soak_temp - profile.ambient_temp) / profile.cool_rate
        total_time = ramp_time + profile.soak_duration + cool_time
        
        # Generate time array
        t = np.linspace(0, total_time, time_points)
        T = np.zeros_like(t)
        
        # Ramp up phase
        ramp_mask = t <= ramp_time
        T[ramp_mask] = profile.ambient_temp + profile.ramp_rate * t[ramp_mask]
        
        # Soak phase
        soak_start = ramp_time
        soak_end = ramp_time + profile.soak_duration
        soak_mask = (t > ramp_time) & (t <= soak_end)
        T[soak_mask] = profile.soak_temp
        
        # Cool down phase
        cool_mask = t > soak_end
        cool_times = t[cool_mask] - soak_end
        T[cool_mask] = profile.soak_temp - profile.cool_rate * cool_times
        
        return t, T
    
    def calculate_thermal_stress(self, T: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate thermal stress using simplified thermoelastic model"""
        # Temperature gradient approximation (simplified)
        dT_dt = np.gradient(T, t)
        dT_dt = np.abs(dT_dt)  # Magnitude of temperature change rate
        
        # Thermal stress calculation (simplified model)
        alpha = self.material.thermal_expansion_coeff
        E = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        
        # Thermal stress due to temperature gradients
        thermal_stress = (alpha * E * dT_dt) / (1 - nu)
        
        # Add temperature-dependent stress relaxation
        stress_relaxation = np.exp(-self.material.activation_energy / (self.material.gas_constant * (T + 273.15)))
        effective_stress = thermal_stress * (1 - stress_relaxation)
        
        return effective_stress
    
    def calculate_warpage(self, T: np.ndarray, t: np.ndarray, profile: SinteringProfile) -> float:
        """Calculate out-of-plane warpage based on thermal gradients and TEC mismatch"""
        # Temperature gradient across thickness (simplified)
        max_temp_grad = np.max(np.gradient(T, t)) * 0.1  # Simplified gradient
        
        # Warpage calculation (simplified beam theory)
        alpha = self.material.thermal_expansion_coeff
        thickness = 0.5e-3  # 0.5 mm typical thickness
        length = 10e-3  # 10 mm typical length
        
        # Curvature due to thermal gradient
        curvature = alpha * max_temp_grad / thickness
        
        # Warpage (deflection at center)
        warpage = curvature * (length**2) / 8
        
        # Scale by profile aggressiveness
        aggressiveness_factor = (profile.ramp_rate / 1.0) * (profile.soak_temp / 1000.0)
        warpage *= (1 + 0.2 * aggressiveness_factor)
        
        return warpage * 1e6  # Convert to micrometers
    
    def calculate_residual_strain(self, T: np.ndarray, t: np.ndarray, profile: SinteringProfile) -> float:
        """Calculate residual Lagrangian strain after full cycle"""
        # Temperature history effect on strain
        alpha = self.material.thermal_expansion_coeff
        
        # Thermal strain from temperature cycling
        thermal_strain = alpha * (np.max(T) - np.min(T))
        
        # Creep relaxation effect (higher temp, longer soak = more relaxation)
        creep_factor = np.exp(-self.material.activation_energy / (self.material.gas_constant * (profile.soak_temp + 273.15)))
        creep_relaxation = 1 - creep_factor * (profile.soak_duration / 60.0)  # Scale by soak time
        
        # Residual strain after relaxation
        residual_strain = thermal_strain * creep_relaxation
        
        # Add profile-specific effects
        if profile.ramp_rate > 1.5:
            residual_strain *= 0.8  # Faster ramps reduce residual strain
        if profile.soak_temp > 1000:
            residual_strain *= 0.7  # Higher temps increase relaxation
            
        return residual_strain * 1e6  # Convert to microstrain
    
    def simulate_profile(self, profile: SinteringProfile) -> Dict:
        """Run complete simulation for a single profile"""
        # Generate temperature profile
        t, T = self.generate_temperature_profile(profile)
        
        # Calculate thermal stress
        thermal_stress = self.calculate_thermal_stress(T, t)
        
        # Calculate warpage
        warpage = self.calculate_warpage(T, t, profile)
        
        # Calculate residual strain
        residual_strain = self.calculate_residual_strain(T, t, profile)
        
        # Store results
        results = {
            'profile': profile,
            'time': t,
            'temperature': T,
            'thermal_stress': thermal_stress,
            'warpage': warpage,
            'residual_strain': residual_strain,
            'max_stress': np.max(thermal_stress),
            'max_temp': np.max(T)
        }
        
        self.results[profile.name] = results
        return results
    
    def run_all_simulations(self):
        """Run simulations for all profiles"""
        for profile in self.profiles:
            self.simulate_profile(profile)
    
    def find_pareto_frontier(self) -> List[Tuple[float, float]]:
        """Find Pareto-efficient frontier for strain vs warpage trade-off"""
        if not self.results:
            self.run_all_simulations()
        
        # Extract data points
        points = []
        for name, result in self.results.items():
            points.append((result['residual_strain'], result['warpage']))
        
        points = np.array(points)
        
        # Find Pareto frontier (minimize both strain and warpage)
        pareto_indices = []
        for i, point in enumerate(points):
            is_pareto = True
            for j, other_point in enumerate(points):
                if i != j:
                    # Check if other point dominates this one
                    if (other_point[0] <= point[0] and other_point[1] <= point[1] and 
                        (other_point[0] < point[0] or other_point[1] < point[1])):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        return points[pareto_indices]

def create_advanced_visualization(simulator: SinteringSimulator):
    """Create professional visualization similar to Abaqus results"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1], 
                         hspace=0.3, wspace=0.3)
    
    # Panel A: Temperature profiles
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Color scheme for professional look
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        t = result['time']
        T = result['temperature']
        
        # Plot temperature profile
        ax1.plot(t, T, linewidth=2.5, color=colors[i % len(colors)], 
                label=f"P{i+1}: $\\dot{{T}} \\approx {profile.ramp_rate:.1f}°C/min, T_{{soak}}={profile.soak_temp:.0f}°C$")
        
        # Add soak duration annotation
        soak_start = (profile.soak_temp - profile.ambient_temp) / profile.ramp_rate
        soak_end = soak_start + profile.soak_duration
        ax1.axhspan(profile.soak_temp-5, profile.soak_temp+5, 
                   xmin=soak_start/t[-1], xmax=soak_end/t[-1], 
                   alpha=0.2, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax1.set_title('Panel A: Staged Sintering Temperature Profiles', fontsize=16, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, None)
    
    # Add professional styling
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel B: Pareto map
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract data for Pareto plot
    strain_data = []
    warpage_data = []
    labels = []
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        strain_data.append(result['residual_strain'])
        warpage_data.append(result['warpage'])
        labels.append(f"P{i+1}")
    
    # Plot data points
    scatter = ax2.scatter(strain_data, warpage_data, c=colors[:len(strain_data)], 
                         s=200, alpha=0.8, edgecolors='black', linewidth=2)
    
    # Add point labels
    for i, (x, y, label) in enumerate(zip(strain_data, warpage_data, labels)):
        ax2.annotate(label, (x, y), xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Find and plot Pareto frontier
    pareto_points = simulator.find_pareto_frontier()
    if len(pareto_points) > 1:
        # Sort by strain for proper line plotting
        pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
        ax2.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'k--', linewidth=2, 
                alpha=0.7, label='Pareto Frontier')
    
    ax2.set_xlabel('Residual Lagrangian Strain (µε)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Out-of-plane Warpage (µm)', fontsize=14, fontweight='bold')
    ax2.set_title('Panel B: Pareto Map - Strain vs. Warpage Trade-off', fontsize=16, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Add professional styling
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Panel C: Thermal stress evolution (additional analysis)
    ax3 = fig.add_subplot(gs[1, :])
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        t = result['time']
        stress = result['thermal_stress']
        
        ax3.plot(t, stress/1e6, linewidth=2, color=colors[i % len(colors)], 
                label=f"P{i+1}: Max Stress = {result['max_stress']/1e6:.1f} MPa")
    
    ax3.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Thermal Stress (MPa)', fontsize=14, fontweight='bold')
    ax3.set_title('Panel C: Thermal Stress Evolution During Sintering', fontsize=16, fontweight='bold', pad=20)
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    
    # Add professional styling
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Add overall title and annotations
    fig.suptitle('Advanced SOFC Sintering Process Optimization\nThermal Profile Design and Stress-Shape Trade-off Analysis', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Add simulation parameters box
    param_text = f"""Simulation Parameters:
Material: SOFC Ceramic (YSZ)
Thermal Conductivity: {simulator.material.thermal_conductivity} W/m·K
Young's Modulus: {simulator.material.youngs_modulus/1e9:.0f} GPa
TEC: {simulator.material.thermal_expansion_coeff*1e6:.1f} ppm/K
Activation Energy: {simulator.material.activation_energy/1e3:.0f} kJ/mol"""
    
    fig.text(0.02, 0.02, param_text, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("Advanced SOFC Sintering Process Simulation")
    print("=" * 50)
    
    # Initialize material properties
    material = MaterialProperties()
    
    # Create simulator
    simulator = SinteringSimulator(material)
    
    # Define sintering profiles (P1, P2, P3)
    profiles = [
        SinteringProfile("P1", ramp_rate=1.0, soak_temp=900, soak_duration=120, cool_rate=1.0),
        SinteringProfile("P2", ramp_rate=1.5, soak_temp=1000, soak_duration=90, cool_rate=1.5),
        SinteringProfile("P3", ramp_rate=2.0, soak_temp=1050, soak_duration=60, cool_rate=2.0),
    ]
    
    # Add profiles to simulator
    for profile in profiles:
        simulator.add_profile(profile)
        print(f"Added profile: {profile.name} - {profile.ramp_rate}°C/min, {profile.soak_temp}°C")
    
    # Run simulations
    print("\nRunning thermal and mechanical simulations...")
    simulator.run_all_simulations()
    
    # Print results summary
    print("\nSimulation Results Summary:")
    print("-" * 40)
    for name, result in simulator.results.items():
        print(f"{name}: Strain = {result['residual_strain']:.1f} µε, "
              f"Warpage = {result['warpage']:.1f} µm, "
              f"Max Stress = {result['max_stress']/1e6:.1f} MPa")
    
    # Find Pareto frontier
    pareto_points = simulator.find_pareto_frontier()
    print(f"\nPareto-efficient profiles: {len(pareto_points)}")
    
    # Create visualization
    print("\nGenerating professional visualization...")
    fig = create_advanced_visualization(simulator)
    
    # Save figure
    fig.savefig('sintering_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("Figure saved as 'sintering_analysis.png'")
    
    # Show plot
    plt.show()
    
    # Create data export
    print("\nExporting simulation data...")
    export_data(simulator)
    
    print("\nSimulation completed successfully!")

def export_data(simulator: SinteringSimulator):
    """Export simulation data to CSV files"""
    # Export temperature profiles
    temp_data = {}
    for name, result in simulator.results.items():
        temp_data[f'{name}_time'] = result['time']
        temp_data[f'{name}_temperature'] = result['temperature']
        temp_data[f'{name}_stress'] = result['thermal_stress']
    
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv('temperature_profiles.csv', index=False)
    
    # Export summary results
    summary_data = []
    for name, result in simulator.results.items():
        summary_data.append({
            'Profile': name,
            'Ramp_Rate_C_min': result['profile'].ramp_rate,
            'Soak_Temp_C': result['profile'].soak_temp,
            'Soak_Duration_min': result['profile'].soak_duration,
            'Residual_Strain_mue': result['residual_strain'],
            'Warpage_um': result['warpage'],
            'Max_Stress_MPa': result['max_stress'] / 1e6,
            'Max_Temp_C': result['max_temp']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('simulation_summary.csv', index=False)
    
    print("Data exported to 'temperature_profiles.csv' and 'simulation_summary.csv'")

if __name__ == "__main__":
    main()