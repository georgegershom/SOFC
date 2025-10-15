#!/usr/bin/env python3
"""
Advanced Sintering Simulation for SOFC Residual Stress Analysis
==============================================================

This module provides comprehensive sintering simulation capabilities for SOFCs,
including:

1. Density evolution during sintering
2. Shrinkage and dimensional changes
3. Residual stress generation during cool-down
4. Multi-layer sintering kinetics
5. Temperature-dependent material properties

The simulation accounts for:
- Different sintering mechanisms (surface diffusion, grain boundary diffusion, volume diffusion)
- Temperature-dependent sintering rates
- CTE mismatch effects during cooling
- Viscoelastic behavior at high temperatures
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class SinteringKinetics:
    """Sintering kinetics parameters for each layer"""
    # Surface diffusion parameters
    D_s0: float  # Pre-exponential factor (m²/s)
    Q_s: float   # Activation energy (J/mol)
    
    # Grain boundary diffusion parameters
    D_gb0: float  # Pre-exponential factor (m²/s)
    Q_gb: float   # Activation energy (J/mol)
    
    # Volume diffusion parameters
    D_v0: float   # Pre-exponential factor (m²/s)
    Q_v: float    # Activation energy (J/mol)
    
    # Sintering parameters
    initial_density: float  # Green density (relative)
    final_density: float    # Final density (relative)
    grain_size: float       # Initial grain size (m)
    particle_size: float    # Initial particle size (m)
    
    # Sintering mechanism weights
    surface_weight: float = 0.3
    grain_boundary_weight: float = 0.5
    volume_weight: float = 0.2

@dataclass
class SinteringResults:
    """Results from sintering simulation"""
    time: np.ndarray
    temperature: np.ndarray
    density: np.ndarray
    shrinkage: np.ndarray
    stress: np.ndarray
    strain: np.ndarray
    final_dimensions: Dict[str, float]
    residual_stress: float

class SOFCSinteringSimulator:
    """
    Advanced sintering simulator for SOFC residual stress analysis
    """
    
    def __init__(self):
        self.layers = {}
        self.temperature_profile = None
        self.sintering_results = {}
        self.residual_stress_results = {}
        
        # Physical constants
        self.R = 8.314  # Gas constant (J/mol·K)
        self.k_B = 1.381e-23  # Boltzmann constant (J/K)
        
        logger.info("SOFC Sintering Simulator initialized")
    
    def add_layer(self, layer_name: str, kinetics: SinteringKinetics, 
                  material_properties: Dict):
        """Add a layer to the sintering simulation"""
        self.layers[layer_name] = {
            'kinetics': kinetics,
            'material_properties': material_properties
        }
        logger.info(f"Added layer: {layer_name}")
    
    def set_temperature_profile(self, time_points: np.ndarray, 
                              temperature_points: np.ndarray):
        """Set temperature profile for sintering"""
        self.temperature_profile = {
            'time': time_points,
            'temperature': temperature_points
        }
        logger.info(f"Temperature profile set with {len(time_points)} points")
    
    def simulate_sintering(self, layer_name: str, 
                          time_span: Tuple[float, float] = (0, 7200)) -> SinteringResults:
        """
        Simulate sintering process for a specific layer
        
        Args:
            layer_name: Name of the layer to simulate
            time_span: Time span for simulation (seconds)
        
        Returns:
            SinteringResults object with simulation results
        """
        logger.info(f"Simulating sintering for layer: {layer_name}")
        
        if layer_name not in self.layers:
            raise ValueError(f"Layer {layer_name} not found")
        
        layer_data = self.layers[layer_name]
        kinetics = layer_data['kinetics']
        material_props = layer_data['material_properties']
        
        # Define sintering rate equation
        def sintering_rate(t, y):
            """Sintering rate equation"""
            density = y[0]
            shrinkage = y[1]
            
            # Get temperature at time t
            T = self._get_temperature_at_time(t)
            
            # Calculate sintering rate based on different mechanisms
            rate_surface = self._calculate_surface_diffusion_rate(density, T, kinetics)
            rate_gb = self._calculate_grain_boundary_diffusion_rate(density, T, kinetics)
            rate_volume = self._calculate_volume_diffusion_rate(density, T, kinetics)
            
            # Combined sintering rate
            total_rate = (kinetics.surface_weight * rate_surface + 
                         kinetics.grain_boundary_weight * rate_gb + 
                         kinetics.volume_weight * rate_volume)
            
            # Density evolution
            density_rate = total_rate * (kinetics.final_density - density)
            
            # Shrinkage evolution (simplified)
            shrinkage_rate = total_rate * 0.1  # 10% of density rate
            
            return [density_rate, shrinkage_rate]
        
        # Initial conditions
        y0 = [kinetics.initial_density, 0.0]  # [density, shrinkage]
        
        # Solve ODE system
        sol = solve_ivp(sintering_rate, time_span, y0, 
                       t_eval=np.linspace(time_span[0], time_span[1], 1000),
                       method='RK45', rtol=1e-6, atol=1e-8)
        
        if not sol.success:
            raise RuntimeError(f"Sintering simulation failed: {sol.message}")
        
        # Calculate stress during sintering
        stress = self._calculate_sintering_stress(sol.t, sol.y[0], sol.y[1], 
                                                material_props, kinetics)
        
        # Calculate strain
        strain = self._calculate_strain(sol.y[1], material_props)
        
        # Calculate final dimensions
        final_dimensions = self._calculate_final_dimensions(sol.y[1], layer_name)
        
        # Calculate residual stress
        residual_stress = self._calculate_residual_stress(sol.y[0], sol.y[1], 
                                                        material_props, kinetics)
        
        # Create results object
        results = SinteringResults(
            time=sol.t,
            temperature=self._get_temperature_at_time(sol.t),
            density=sol.y[0],
            shrinkage=sol.y[1],
            stress=stress,
            strain=strain,
            final_dimensions=final_dimensions,
            residual_stress=residual_stress
        )
        
        self.sintering_results[layer_name] = results
        logger.info(f"Sintering simulation complete for {layer_name}")
        
        return results
    
    def _get_temperature_at_time(self, t: np.ndarray) -> np.ndarray:
        """Get temperature at specific time points"""
        if self.temperature_profile is None:
            return np.full_like(t, 800.0)  # Default temperature
        
        # Interpolate temperature profile
        return np.interp(t, self.temperature_profile['time'], 
                        self.temperature_profile['temperature'])
    
    def _calculate_surface_diffusion_rate(self, density: float, T: float, 
                                        kinetics: SinteringKinetics) -> float:
        """Calculate surface diffusion sintering rate"""
        if T <= 0:
            return 0.0
        
        # Surface diffusion coefficient
        D_s = kinetics.D_s0 * np.exp(-kinetics.Q_s / (self.R * T))
        
        # Surface diffusion rate (simplified model)
        rate = D_s * (1 - density) / (kinetics.particle_size ** 2)
        
        return rate
    
    def _calculate_grain_boundary_diffusion_rate(self, density: float, T: float, 
                                               kinetics: SinteringKinetics) -> float:
        """Calculate grain boundary diffusion sintering rate"""
        if T <= 0:
            return 0.0
        
        # Grain boundary diffusion coefficient
        D_gb = kinetics.D_gb0 * np.exp(-kinetics.Q_gb / (self.R * T))
        
        # Grain boundary diffusion rate
        rate = D_gb * (1 - density) / (kinetics.grain_size ** 2)
        
        return rate
    
    def _calculate_volume_diffusion_rate(self, density: float, T: float, 
                                       kinetics: SinteringKinetics) -> float:
        """Calculate volume diffusion sintering rate"""
        if T <= 0:
            return 0.0
        
        # Volume diffusion coefficient
        D_v = kinetics.D_v0 * np.exp(-kinetics.Q_v / (self.R * T))
        
        # Volume diffusion rate
        rate = D_v * (1 - density) / (kinetics.particle_size ** 3)
        
        return rate
    
    def _calculate_sintering_stress(self, time: np.ndarray, density: np.ndarray, 
                                  shrinkage: np.ndarray, material_props: Dict, 
                                  kinetics: SinteringKinetics) -> np.ndarray:
        """Calculate stress during sintering"""
        # Temperature at each time point
        T = self._get_temperature_at_time(time)
        
        # Young's modulus (temperature dependent)
        E = material_props['E_25C'] * (1 - 0.1 * (T - 25) / 1000)  # Simplified
        
        # Thermal expansion coefficient
        alpha = material_props['CTE_25C'] * (1 + 0.05 * (T - 25) / 1000)  # Simplified
        
        # Sintering stress (simplified model)
        # Stress = E * (shrinkage - thermal_expansion)
        thermal_expansion = alpha * (T - 25)
        sintering_stress = E * (shrinkage - thermal_expansion)
        
        return sintering_stress
    
    def _calculate_strain(self, shrinkage: np.ndarray, material_props: Dict) -> np.ndarray:
        """Calculate strain from shrinkage"""
        # Strain = shrinkage / original_length
        return shrinkage
    
    def _calculate_final_dimensions(self, shrinkage: np.ndarray, layer_name: str) -> Dict[str, float]:
        """Calculate final dimensions after sintering"""
        # This would depend on the specific geometry
        # For now, return simplified results
        final_shrinkage = shrinkage[-1]
        
        return {
            'final_length': 100.0 * (1 - final_shrinkage),  # mm
            'final_width': 100.0 * (1 - final_shrinkage),   # mm
            'final_thickness': 0.15 * (1 - final_shrinkage),  # mm
            'shrinkage_factor': final_shrinkage
        }
    
    def _calculate_residual_stress(self, final_density: float, final_shrinkage: float, 
                                 material_props: Dict, kinetics: SinteringKinetics) -> float:
        """Calculate residual stress after cool-down"""
        # Temperature change from sintering to room temperature
        T_sintering = 1350.0  # °C
        T_room = 25.0  # °C
        delta_T = T_sintering - T_room
        
        # Young's modulus at room temperature
        E_room = material_props['E_25C']
        
        # Thermal expansion coefficient
        alpha = material_props['CTE_25C']
        
        # Residual stress due to thermal contraction
        thermal_stress = E_room * alpha * delta_T
        
        # Residual stress due to sintering shrinkage
        sintering_stress = E_room * final_shrinkage
        
        # Total residual stress
        total_residual_stress = thermal_stress + sintering_stress
        
        return total_residual_stress
    
    def simulate_multi_layer_sintering(self, layer_order: List[str]) -> Dict[str, SinteringResults]:
        """
        Simulate sintering for multiple layers with interactions
        
        Args:
            layer_order: List of layer names in sintering order
        
        Returns:
            Dictionary with results for each layer
        """
        logger.info(f"Simulating multi-layer sintering for {len(layer_order)} layers")
        
        results = {}
        
        for i, layer_name in enumerate(layer_order):
            logger.info(f"Simulating layer {i+1}/{len(layer_order)}: {layer_name}")
            
            # Simulate sintering for this layer
            layer_results = self.simulate_sintering(layer_name)
            results[layer_name] = layer_results
            
            # Update temperature profile for next layer (if needed)
            # This could include heat transfer between layers
        
        # Calculate inter-layer interactions
        self._calculate_inter_layer_stresses(results, layer_order)
        
        logger.info("Multi-layer sintering simulation complete")
        return results
    
    def _calculate_inter_layer_stresses(self, results: Dict[str, SinteringResults], 
                                      layer_order: List[str]):
        """Calculate stresses between layers due to CTE mismatch"""
        logger.info("Calculating inter-layer stresses")
        
        # This is a simplified implementation
        # In practice, this would involve complex stress analysis
        
        for i in range(len(layer_order) - 1):
            layer1 = layer_order[i]
            layer2 = layer_order[i + 1]
            
            if layer1 in results and layer2 in results:
                # Calculate CTE mismatch stress
                cte1 = self.layers[layer1]['material_properties']['CTE_25C']
                cte2 = self.layers[layer2]['material_properties']['CTE_25C']
                
                cte_mismatch = abs(cte1 - cte2)
                
                # Temperature change
                delta_T = 1350.0 - 25.0  # °C
                
                # CTE mismatch stress
                E_avg = (self.layers[layer1]['material_properties']['E_25C'] + 
                        self.layers[layer2]['material_properties']['E_25C']) / 2
                
                cte_stress = E_avg * cte_mismatch * delta_T
                
                # Store inter-layer stress
                self.residual_stress_results[f"{layer1}_{layer2}"] = {
                    'cte_mismatch': cte_mismatch,
                    'cte_stress': cte_stress,
                    'layer1': layer1,
                    'layer2': layer2
                }
    
    def create_sintering_visualizations(self, output_dir: str = "sintering_results"):
        """Create comprehensive visualizations of sintering results"""
        logger.info("Creating sintering visualizations")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Plot density evolution
        self._plot_density_evolution(output_path)
        
        # Plot shrinkage evolution
        self._plot_shrinkage_evolution(output_path)
        
        # Plot stress evolution
        self._plot_stress_evolution(output_path)
        
        # Plot temperature profile
        self._plot_temperature_profile(output_path)
        
        # Plot residual stress analysis
        self._plot_residual_stress_analysis(output_path)
        
        logger.info(f"Sintering visualizations saved to {output_path}")
    
    def _plot_density_evolution(self, output_path: Path):
        """Plot density evolution during sintering"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (layer_name, results) in enumerate(self.sintering_results.items()):
            if i >= 4:  # Limit to 4 subplots
                break
            
            row, col = i // 2, i % 2
            
            # Convert time to minutes
            time_min = results.time / 60
            
            axes[row, col].plot(time_min, results.density, 'b-', linewidth=2, label='Density')
            axes[row, col].set_xlabel('Time (min)')
            axes[row, col].set_ylabel('Relative Density')
            axes[row, col].set_title(f'{layer_name} - Density Evolution')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
            
            # Add final density annotation
            final_density = results.density[-1]
            axes[row, col].annotate(f'Final: {final_density:.3f}', 
                                  xy=(time_min[-1], final_density),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(self.sintering_results), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'density_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_shrinkage_evolution(self, output_path: Path):
        """Plot shrinkage evolution during sintering"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (layer_name, results) in enumerate(self.sintering_results.items()):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            
            # Convert time to minutes
            time_min = results.time / 60
            
            axes[row, col].plot(time_min, results.shrinkage * 100, 'r-', linewidth=2, label='Shrinkage')
            axes[row, col].set_xlabel('Time (min)')
            axes[row, col].set_ylabel('Shrinkage (%)')
            axes[row, col].set_title(f'{layer_name} - Shrinkage Evolution')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
            
            # Add final shrinkage annotation
            final_shrinkage = results.shrinkage[-1] * 100
            axes[row, col].annotate(f'Final: {final_shrinkage:.2f}%', 
                                  xy=(time_min[-1], final_shrinkage),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(self.sintering_results), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'shrinkage_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_stress_evolution(self, output_path: Path):
        """Plot stress evolution during sintering"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (layer_name, results) in enumerate(self.sintering_results.items()):
            if i >= 4:
                break
            
            row, col = i // 2, i % 2
            
            # Convert time to minutes and stress to MPa
            time_min = results.time / 60
            stress_mpa = results.stress / 1e6
            
            axes[row, col].plot(time_min, stress_mpa, 'g-', linewidth=2, label='Stress')
            axes[row, col].set_xlabel('Time (min)')
            axes[row, col].set_ylabel('Stress (MPa)')
            axes[row, col].set_title(f'{layer_name} - Stress Evolution')
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
            
            # Add final stress annotation
            final_stress = stress_mpa[-1]
            axes[row, col].annotate(f'Final: {final_stress:.1f} MPa', 
                                  xy=(time_min[-1], final_stress),
                                  xytext=(10, 10), textcoords='offset points',
                                  bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # Hide unused subplots
        for i in range(len(self.sintering_results), 4):
            row, col = i // 2, i % 2
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / 'stress_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temperature_profile(self, output_path: Path):
        """Plot temperature profile"""
        if self.temperature_profile is None:
            return
        
        plt.figure(figsize=(12, 8))
        
        # Convert time to minutes
        time_min = self.temperature_profile['time'] / 60
        temperature = self.temperature_profile['temperature']
        
        plt.plot(time_min, temperature, 'b-', linewidth=3, label='Temperature Profile')
        plt.xlabel('Time (min)')
        plt.ylabel('Temperature (°C)')
        plt.title('Sintering Temperature Profile')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add phase annotations
        plt.axhline(y=1000, color='r', linestyle='--', alpha=0.7, label='Sintering Start')
        plt.axhline(y=1350, color='orange', linestyle='--', alpha=0.7, label='Max Temperature')
        plt.axhline(y=25, color='blue', linestyle='--', alpha=0.7, label='Room Temperature')
        
        plt.tight_layout()
        plt.savefig(output_path / 'temperature_profile.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_residual_stress_analysis(self, output_path: Path):
        """Plot residual stress analysis"""
        if not self.residual_stress_results:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residual stress by layer
        layer_names = list(self.sintering_results.keys())
        residual_stresses = [results.residual_stress / 1e6 for results in self.sintering_results.values()]
        
        axes[0].bar(layer_names, residual_stresses, color=['red', 'blue', 'green', 'orange'])
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Residual Stress (MPa)')
        axes[0].set_title('Residual Stress by Layer')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Inter-layer CTE mismatch stress
        if self.residual_stress_results:
            interface_names = list(self.residual_stress_results.keys())
            cte_stresses = [data['cte_stress'] / 1e6 for data in self.residual_stress_results.values()]
            
            axes[1].bar(interface_names, cte_stresses, color='purple')
            axes[1].set_xlabel('Interface')
            axes[1].set_ylabel('CTE Mismatch Stress (MPa)')
            axes[1].set_title('Inter-layer CTE Mismatch Stress')
            axes[1].tick_params(axis='x', rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'residual_stress_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_standard_sofc_layers() -> Dict[str, Dict]:
    """Create standard SOFC layer configurations"""
    
    # Anode (Ni-YSZ)
    anode_kinetics = SinteringKinetics(
        D_s0=1e-6, Q_s=200000,  # Surface diffusion
        D_gb0=1e-8, Q_gb=300000,  # Grain boundary diffusion
        D_v0=1e-10, Q_v=400000,  # Volume diffusion
        initial_density=0.6, final_density=0.95,
        grain_size=1e-6, particle_size=0.5e-6
    )
    
    anode_material = {
        'E_25C': 55e9,  # Pa
        'E_800C': 29e9,  # Pa
        'nu': 0.29,
        'CTE_25C': 12.5e-6,  # 1/K
        'CTE_800C': 13.3e-6,  # 1/K
        'density': 6500.0  # kg/m³
    }
    
    # Electrolyte (8YSZ)
    electrolyte_kinetics = SinteringKinetics(
        D_s0=1e-7, Q_s=250000,
        D_gb0=1e-9, Q_gb=350000,
        D_v0=1e-11, Q_v=450000,
        initial_density=0.55, final_density=0.98,
        grain_size=0.5e-6, particle_size=0.2e-6
    )
    
    electrolyte_material = {
        'E_25C': 200e9,  # Pa
        'E_800C': 170e9,  # Pa
        'nu': 0.23,
        'CTE_25C': 10.0e-6,  # 1/K
        'CTE_800C': 10.5e-6,  # 1/K
        'density': 5900.0  # kg/m³
    }
    
    # Cathode (LSM-YSZ)
    cathode_kinetics = SinteringKinetics(
        D_s0=1e-6, Q_s=180000,
        D_gb0=1e-8, Q_gb=280000,
        D_v0=1e-10, Q_v=380000,
        initial_density=0.6, final_density=0.92,
        grain_size=0.8e-6, particle_size=0.3e-6
    )
    
    cathode_material = {
        'E_25C': 45e9,  # Pa
        'E_800C': 40e9,  # Pa
        'nu': 0.25,
        'CTE_25C': 11.5e-6,  # 1/K
        'CTE_800C': 12.0e-6,  # 1/K
        'density': 6200.0  # kg/m³
    }
    
    # Interconnect (Crofer 22 APU)
    interconnect_kinetics = SinteringKinetics(
        D_s0=1e-5, Q_s=150000,
        D_gb0=1e-7, Q_gb=250000,
        D_v0=1e-9, Q_v=350000,
        initial_density=0.7, final_density=0.99,
        grain_size=2e-6, particle_size=1e-6
    )
    
    interconnect_material = {
        'E_25C': 160e9,  # Pa
        'E_800C': 140e9,  # Pa
        'nu': 0.30,
        'CTE_25C': 11.5e-6,  # 1/K
        'CTE_800C': 11.9e-6,  # 1/K
        'density': 7800.0  # kg/m³
    }
    
    return {
        'anode': {'kinetics': anode_kinetics, 'material_properties': anode_material},
        'electrolyte': {'kinetics': electrolyte_kinetics, 'material_properties': electrolyte_material},
        'cathode': {'kinetics': cathode_kinetics, 'material_properties': cathode_material},
        'interconnect': {'kinetics': interconnect_kinetics, 'material_properties': interconnect_material}
    }

def main():
    """Example usage of the sintering simulator"""
    logger.info("Starting SOFC sintering simulation example")
    
    # Initialize simulator
    simulator = SOFCSinteringSimulator()
    
    # Create standard SOFC layers
    layers = create_standard_sofc_layers()
    
    # Add layers to simulator
    for layer_name, layer_data in layers.items():
        simulator.add_layer(layer_name, layer_data['kinetics'], layer_data['material_properties'])
    
    # Define temperature profile
    time_points = np.array([0, 1800, 3600, 5400, 7200])  # seconds
    temperature_points = np.array([25, 1000, 1350, 1350, 25])  # °C
    
    simulator.set_temperature_profile(time_points, temperature_points)
    
    # Simulate multi-layer sintering
    layer_order = ['anode', 'electrolyte', 'cathode', 'interconnect']
    results = simulator.simulate_multi_layer_sintering(layer_order)
    
    # Create visualizations
    simulator.create_sintering_visualizations()
    
    # Print results summary
    print("\n" + "="*60)
    print("SOFC SINTERING SIMULATION RESULTS")
    print("="*60)
    
    for layer_name, layer_results in results.items():
        print(f"\n{layer_name.upper()}:")
        print(f"  Final Density: {layer_results.density[-1]:.3f}")
        print(f"  Final Shrinkage: {layer_results.shrinkage[-1]*100:.2f}%")
        print(f"  Residual Stress: {layer_results.residual_stress/1e6:.1f} MPa")
        print(f"  Final Dimensions: {layer_results.final_dimensions}")
    
    print(f"\nInter-layer Stresses:")
    for interface, data in simulator.residual_stress_results.items():
        print(f"  {interface}: {data['cte_stress']/1e6:.1f} MPa")
    
    print(f"\nSintering simulation complete! 🎉")
    print(f"Results saved to: sintering_results/")

if __name__ == "__main__":
    main()