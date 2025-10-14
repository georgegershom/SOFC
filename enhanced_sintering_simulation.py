"""
Enhanced Advanced Sintering Profile Design and Stress-Shape Trade-off Analysis
==============================================================================

This module implements a comprehensive simulation for SOFC sintering processes,
including thermal profile generation, stress analysis, and Pareto optimization
for process parameter selection with enhanced realism and professional visualization.

Author: Advanced Materials Simulation Lab
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
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
    'axes.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3
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

class EnhancedSinteringSimulator:
    """Enhanced sintering process simulator with advanced thermal and mechanical analysis"""
    
    def __init__(self, material: MaterialProperties):
        self.material = material
        self.profiles = []
        self.results = {}
        
    def add_profile(self, profile: SinteringProfile):
        """Add a sintering profile to the simulation"""
        self.profiles.append(profile)
        
    def generate_temperature_profile(self, profile: SinteringProfile, time_points: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic temperature profile for given sintering parameters"""
        # Calculate time segments
        ramp_time = (profile.soak_temp - profile.ambient_temp) / profile.ramp_rate
        cool_time = (profile.soak_temp - profile.ambient_temp) / profile.cool_rate
        total_time = ramp_time + profile.soak_duration + cool_time
        
        # Generate time array with higher resolution
        t = np.linspace(0, total_time, time_points)
        T = np.zeros_like(t)
        
        # Ramp up phase with realistic thermal lag
        ramp_mask = t <= ramp_time
        T[ramp_mask] = profile.ambient_temp + profile.ramp_rate * t[ramp_mask]
        
        # Add thermal lag effect (realistic furnace behavior)
        thermal_lag = 0.95 + 0.05 * np.exp(-t[ramp_mask] / (ramp_time * 0.3))
        T[ramp_mask] *= thermal_lag
        
        # Soak phase with slight temperature variations
        soak_start = ramp_time
        soak_end = ramp_time + profile.soak_duration
        soak_mask = (t > ramp_time) & (t <= soak_end)
        soak_times = t[soak_mask] - soak_start
        
        # Add realistic temperature variations during soak
        temp_variation = 2.0 * np.sin(2 * np.pi * soak_times / 30) * np.exp(-soak_times / 60)
        T[soak_mask] = profile.soak_temp + temp_variation
        
        # Cool down phase with realistic cooling curve
        cool_mask = t > soak_end
        cool_times = t[cool_mask] - soak_end
        
        # Non-linear cooling (exponential decay)
        cooling_factor = np.exp(-cool_times / (cool_time * 0.4))
        T[cool_mask] = profile.soak_temp - (profile.soak_temp - profile.ambient_temp) * (1 - cooling_factor)
        
        return t, T
    
    def calculate_thermal_stress(self, T: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Calculate thermal stress using advanced thermoelastic model"""
        # Temperature gradient approximation
        dT_dt = np.gradient(T, t)
        dT_dt = np.abs(dT_dt)
        
        # Thermal stress calculation with temperature-dependent properties
        alpha = self.material.thermal_expansion_coeff
        E = self.material.youngs_modulus
        nu = self.material.poisson_ratio
        
        # Temperature-dependent Young's modulus (decreases with temperature)
        E_T = E * (1 - 0.3 * (T - 25) / 1000)  # 30% reduction at 1000°C
        
        # Thermal stress due to temperature gradients
        thermal_stress = (alpha * E_T * dT_dt) / (1 - nu)
        
        # Add creep relaxation effects
        creep_rate = self.material.pre_exponential * np.exp(-self.material.activation_energy / 
                                                           (self.material.gas_constant * (T + 273.15)))
        
        # Stress relaxation during high temperature
        stress_relaxation = 1 - np.exp(-creep_rate * np.gradient(t, t))
        effective_stress = thermal_stress * (1 - stress_relaxation)
        
        # Add residual stress from thermal cycling
        residual_contribution = 0.1 * thermal_stress * np.exp(-(T - 25) / 200)
        effective_stress += residual_contribution
        
        return effective_stress
    
    def calculate_warpage(self, T: np.ndarray, t: np.ndarray, profile: SinteringProfile) -> float:
        """Calculate out-of-plane warpage based on thermal gradients and TEC mismatch"""
        # Temperature gradient across thickness (more realistic)
        max_temp_grad = np.max(np.gradient(T, t)) * 0.15  # Increased gradient factor
        
        # Warpage calculation using advanced beam theory
        alpha = self.material.thermal_expansion_coeff
        thickness = 0.5e-3  # 0.5 mm typical thickness
        length = 10e-3  # 10 mm typical length
        
        # Curvature due to thermal gradient (more complex model)
        curvature = alpha * max_temp_grad / thickness
        
        # Add TEC mismatch effects
        tec_mismatch_factor = 1 + 0.1 * (profile.soak_temp - 900) / 100
        curvature *= tec_mismatch_factor
        
        # Warpage (deflection at center) with geometric nonlinearity
        warpage = curvature * (length**2) / 8
        
        # Scale by profile aggressiveness and thermal history
        aggressiveness_factor = (profile.ramp_rate / 1.0) * (profile.soak_temp / 1000.0)
        thermal_history_factor = 1 + 0.3 * np.max(T) / 1000
        
        warpage *= (1 + 0.2 * aggressiveness_factor) * thermal_history_factor
        
        return warpage * 1e6  # Convert to micrometers
    
    def calculate_residual_strain(self, T: np.ndarray, t: np.ndarray, profile: SinteringProfile) -> float:
        """Calculate residual Lagrangian strain after full cycle with enhanced model"""
        # Temperature history effect on strain
        alpha = self.material.thermal_expansion_coeff
        
        # Thermal strain from temperature cycling
        thermal_strain = alpha * (np.max(T) - np.min(T))
        
        # Enhanced creep relaxation model
        T_soak = profile.soak_temp + 273.15  # Convert to Kelvin
        creep_factor = np.exp(-self.material.activation_energy / (self.material.gas_constant * T_soak))
        
        # Time-dependent relaxation
        time_factor = 1 - np.exp(-creep_factor * (profile.soak_duration / 60.0))
        creep_relaxation = 1 - time_factor
        
        # Residual strain after relaxation
        residual_strain = thermal_strain * creep_relaxation
        
        # Add profile-specific effects with more realistic scaling
        if profile.ramp_rate > 1.5:
            residual_strain *= 0.85  # Faster ramps reduce residual strain
        if profile.soak_temp > 1000:
            residual_strain *= 0.75  # Higher temps increase relaxation
            
        # Add microstructural effects
        microstructural_factor = 1 + 0.1 * np.sin(profile.soak_temp * np.pi / 1000)
        residual_strain *= microstructural_factor
            
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
            'max_temp': np.max(T),
            'avg_stress': np.mean(thermal_stress),
            'stress_std': np.std(thermal_stress)
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

def create_professional_visualization(simulator: EnhancedSinteringSimulator):
    """Create professional visualization similar to Abaqus results with enhanced styling"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.3, 1], 
                         hspace=0.35, wspace=0.35)
    
    # Professional color scheme
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Panel A: Temperature profiles
    ax1 = fig.add_subplot(gs[0, 0])
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        t = result['time']
        T = result['temperature']
        
        # Plot temperature profile with enhanced styling
        ax1.plot(t, T, linewidth=3, color=colors[i % len(colors)], 
                label=f"P{i+1}: $\\dot{{T}} \\approx {profile.ramp_rate:.1f}°C/min, T_{{soak}}={profile.soak_temp:.0f}°C$",
                alpha=0.9)
        
        # Add soak duration highlighting
        soak_start = (profile.soak_temp - profile.ambient_temp) / profile.ramp_rate
        soak_end = soak_start + profile.soak_duration
        ax1.axhspan(profile.soak_temp-8, profile.soak_temp+8, 
                   xmin=soak_start/t[-1], xmax=soak_end/t[-1], 
                   alpha=0.15, color=colors[i % len(colors)])
        
        # Add key temperature markers
        ax1.axhline(y=profile.soak_temp, color=colors[i % len(colors)], 
                   linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)', fontsize=14, fontweight='bold')
    ax1.set_title('Panel A: Staged Sintering Temperature Profiles $T(t)$', 
                 fontsize=16, fontweight='bold', pad=25)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_xlim(0, None)
    ax1.set_ylim(0, None)
    
    # Enhanced styling
    ax1.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax1.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Panel B: Pareto map
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract data for Pareto plot
    strain_data = []
    warpage_data = []
    labels = []
    colors_plot = []
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        strain_data.append(result['residual_strain'])
        warpage_data.append(result['warpage'])
        labels.append(f"P{i+1}")
        colors_plot.append(colors[i % len(colors)])
    
    # Plot data points with enhanced styling
    scatter = ax2.scatter(strain_data, warpage_data, c=colors_plot, 
                         s=300, alpha=0.8, edgecolors='black', linewidth=2.5,
                         zorder=5)
    
    # Add point labels with professional styling
    for i, (x, y, label) in enumerate(zip(strain_data, warpage_data, labels)):
        ax2.annotate(label, (x, y), xytext=(12, 12), textcoords='offset points',
                    fontsize=13, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor=colors_plot[i], alpha=0.9, linewidth=1.5))
    
    # Find and plot Pareto frontier
    pareto_points = simulator.find_pareto_frontier()
    if len(pareto_points) > 1:
        # Sort by strain for proper line plotting
        pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]
        ax2.plot(pareto_sorted[:, 0], pareto_sorted[:, 1], 'k--', linewidth=3, 
                alpha=0.8, label='Pareto Frontier', zorder=4)
    
    # Add acceptance region annotations
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.7, linewidth=2, label='Warpage Limit')
    ax2.axvline(x=8000, color='blue', linestyle=':', alpha=0.7, linewidth=2, label='Strain Limit')
    
    ax2.set_xlabel('Residual Lagrangian Strain (µε)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Out-of-plane Warpage (µm)', fontsize=14, fontweight='bold')
    ax2.set_title('Panel B: Pareto Map - Strain vs. Warpage Trade-off', 
                 fontsize=16, fontweight='bold', pad=25)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    
    # Enhanced styling
    ax2.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax2.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Panel C: Thermal stress evolution
    ax3 = fig.add_subplot(gs[1, :])
    
    for i, profile in enumerate(simulator.profiles):
        result = simulator.results[profile.name]
        t = result['time']
        stress = result['thermal_stress']
        
        ax3.plot(t, stress/1e6, linewidth=2.5, color=colors[i % len(colors)], 
                label=f"P{i+1}: Max = {result['max_stress']/1e6:.1f} MPa, "
                      f"Avg = {result['avg_stress']/1e6:.1f} MPa")
        
        # Add stress envelope
        stress_upper = stress/1e6 + result['stress_std']/1e6
        stress_lower = stress/1e6 - result['stress_std']/1e6
        ax3.fill_between(t, stress_lower, stress_upper, alpha=0.2, color=colors[i % len(colors)])
    
    ax3.set_xlabel('Time (min)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Thermal Stress (MPa)', fontsize=14, fontweight='bold')
    ax3.set_title('Panel C: Thermal Stress Evolution During Sintering Process', 
                 fontsize=16, fontweight='bold', pad=25)
    ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Enhanced styling
    ax3.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
    ax3.tick_params(axis='both', which='minor', width=1, length=3)
    
    # Add overall title and professional annotations
    fig.suptitle('Advanced SOFC Sintering Process Optimization\n' + 
                'Thermal Profile Design and Stress-Shape Trade-off Analysis', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # Add simulation parameters box with enhanced styling
    param_text = f"""Simulation Parameters:
Material: SOFC Ceramic (YSZ)
Thermal Conductivity: {simulator.material.thermal_conductivity} W/m·K
Young's Modulus: {simulator.material.youngs_modulus/1e9:.0f} GPa
TEC: {simulator.material.thermal_expansion_coeff*1e6:.1f} ppm/K
Activation Energy: {simulator.material.activation_energy/1e3:.0f} kJ/mol
Simulation Points: 2000 per profile"""
    
    # Create parameter box
    param_box = FancyBboxPatch((0.02, 0.02), 0.25, 0.15, 
                              boxstyle="round,pad=0.01", 
                              facecolor='lightgray', alpha=0.9,
                              edgecolor='black', linewidth=1.5)
    fig.patches.append(param_box)
    
    fig.text(0.03, 0.17, param_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    # Add process selection guidelines
    guidelines_text = """Process Selection Guidelines:
• Choose profiles near Pareto frontier
• Balance stress relief vs. shape fidelity
• Consider production constraints
• Validate with experimental data"""
    
    guide_box = FancyBboxPatch((0.73, 0.02), 0.25, 0.15, 
                              boxstyle="round,pad=0.01", 
                              facecolor='lightblue', alpha=0.9,
                              edgecolor='blue', linewidth=1.5)
    fig.patches.append(guide_box)
    
    fig.text(0.74, 0.17, guidelines_text, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    return fig

def main():
    """Main execution function"""
    print("Enhanced Advanced SOFC Sintering Process Simulation")
    print("=" * 60)
    
    # Initialize material properties
    material = MaterialProperties()
    
    # Create enhanced simulator
    simulator = EnhancedSinteringSimulator(material)
    
    # Define sintering profiles (P1, P2, P3) with more realistic parameters
    profiles = [
        SinteringProfile("P1", ramp_rate=1.0, soak_temp=900, soak_duration=120, cool_rate=1.0),
        SinteringProfile("P2", ramp_rate=1.5, soak_temp=1000, soak_duration=90, cool_rate=1.5),
        SinteringProfile("P3", ramp_rate=2.0, soak_temp=1050, soak_duration=60, cool_rate=2.0),
    ]
    
    # Add profiles to simulator
    for profile in profiles:
        simulator.add_profile(profile)
        print(f"Added profile: {profile.name} - {profile.ramp_rate}°C/min, {profile.soak_temp}°C, {profile.soak_duration}min")
    
    # Run simulations
    print("\nRunning enhanced thermal and mechanical simulations...")
    simulator.run_all_simulations()
    
    # Print results summary
    print("\nEnhanced Simulation Results Summary:")
    print("-" * 50)
    for name, result in simulator.results.items():
        print(f"{name}: Strain = {result['residual_strain']:.0f} µε, "
              f"Warpage = {result['warpage']:.2f} µm, "
              f"Max Stress = {result['max_stress']/1e6:.1f} MPa, "
              f"Avg Stress = {result['avg_stress']/1e6:.1f} MPa")
    
    # Find Pareto frontier
    pareto_points = simulator.find_pareto_frontier()
    print(f"\nPareto-efficient profiles: {len(pareto_points)}")
    for i, point in enumerate(pareto_points):
        print(f"  Pareto Point {i+1}: Strain = {point[0]:.0f} µε, Warpage = {point[1]:.2f} µm")
    
    # Create enhanced visualization
    print("\nGenerating professional enhanced visualization...")
    fig = create_professional_visualization(simulator)
    
    # Save figure with high quality
    fig.savefig('enhanced_sintering_analysis.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.2)
    print("Enhanced figure saved as 'enhanced_sintering_analysis.png'")
    
    # Show plot
    plt.show()
    
    # Create enhanced data export
    print("\nExporting enhanced simulation data...")
    export_enhanced_data(simulator)
    
    print("\nEnhanced simulation completed successfully!")

def export_enhanced_data(simulator: EnhancedSinteringSimulator):
    """Export enhanced simulation data to CSV files"""
    # Export detailed temperature profiles
    temp_data = {}
    for name, result in simulator.results.items():
        temp_data[f'{name}_time'] = result['time']
        temp_data[f'{name}_temperature'] = result['temperature']
        temp_data[f'{name}_stress'] = result['thermal_stress']
        temp_data[f'{name}_stress_upper'] = result['thermal_stress'] + result['stress_std']
        temp_data[f'{name}_stress_lower'] = result['thermal_stress'] - result['stress_std']
    
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_csv('enhanced_temperature_profiles.csv', index=False)
    
    # Export comprehensive summary results
    summary_data = []
    for name, result in simulator.results.items():
        summary_data.append({
            'Profile': name,
            'Ramp_Rate_C_min': result['profile'].ramp_rate,
            'Soak_Temp_C': result['profile'].soak_temp,
            'Soak_Duration_min': result['profile'].soak_duration,
            'Cool_Rate_C_min': result['profile'].cool_rate,
            'Residual_Strain_mue': result['residual_strain'],
            'Warpage_um': result['warpage'],
            'Max_Stress_MPa': result['max_stress'] / 1e6,
            'Avg_Stress_MPa': result['avg_stress'] / 1e6,
            'Stress_Std_MPa': result['stress_std'] / 1e6,
            'Max_Temp_C': result['max_temp'],
            'Process_Time_min': result['time'][-1]
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('enhanced_simulation_summary.csv', index=False)
    
    # Export Pareto frontier data
    pareto_points = simulator.find_pareto_frontier()
    pareto_df = pd.DataFrame(pareto_points, columns=['Residual_Strain_mue', 'Warpage_um'])
    pareto_df.to_csv('pareto_frontier.csv', index=False)
    
    print("Enhanced data exported to:")
    print("  - enhanced_temperature_profiles.csv")
    print("  - enhanced_simulation_summary.csv") 
    print("  - pareto_frontier.csv")

if __name__ == "__main__":
    main()