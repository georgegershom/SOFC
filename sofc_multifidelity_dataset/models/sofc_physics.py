"""
SOFC Physics Models for Multi-Fidelity Dataset Generation
Implements electrochemical, thermal, and mechanical models at different fidelity levels
"""

import numpy as np
from scipy import integrate, interpolate
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import warnings

# Physical Constants
R = 8.314  # Universal gas constant [J/(mol·K)]
F = 96485  # Faraday constant [C/mol]


@dataclass
class SOFCOperatingConditions:
    """Operating conditions for SOFC simulation"""
    temperature: float  # K
    pressure: float  # Pa
    current_density: float  # A/m²
    fuel_utilization: float  # fraction
    air_utilization: float  # fraction
    inlet_fuel_composition: Dict[str, float]  # mole fractions
    inlet_air_composition: Dict[str, float]  # mole fractions
    time: float = 0.0  # hours


class ElectrochemicalModel:
    """SOFC Electrochemical Model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mat = config['material_properties']
        self.geom = config['geometry']
        
    def nernst_voltage(self, T: float, p_H2: float, p_H2O: float, p_O2: float) -> float:
        """Calculate Nernst voltage"""
        E0 = 1.253 - 2.4516e-4 * (T - 298.15)  # Standard potential
        E_nernst = E0 + (R * T) / (2 * F) * np.log(p_H2 * np.sqrt(p_O2) / p_H2O)
        return E_nernst
    
    def activation_overpotential(self, i: float, T: float, electrode: str) -> float:
        """Butler-Volmer activation overpotential"""
        if electrode == 'anode':
            i0 = 5000 * np.exp(-100000 / (R * T))  # Exchange current density
            alpha = 0.5  # Transfer coefficient
        else:  # cathode
            i0 = 2000 * np.exp(-120000 / (R * T))
            alpha = 0.5
            
        if i == 0:
            return 0
        
        # Simplified Butler-Volmer
        eta_act = (R * T) / (alpha * 2 * F) * np.arcsinh(i / (2 * i0))
        return abs(eta_act)
    
    def ohmic_overpotential(self, i: float, T: float) -> float:
        """Ohmic losses in electrolyte"""
        # Temperature-dependent ionic conductivity
        sigma = self.mat['electrolyte']['ionic_conductivity_0'] * \
                np.exp(-self.mat['electrolyte']['activation_energy'] / (R * T))
        R_ohm = self.geom['thickness']['electrolyte'] / sigma
        return i * R_ohm
    
    def concentration_overpotential(self, i: float, T: float, i_lim: float) -> float:
        """Concentration overpotential"""
        if i >= i_lim:
            return np.inf
        if i == 0:
            return 0
        eta_conc = (R * T) / (2 * F) * np.log(1 - i/i_lim)
        return abs(eta_conc)
    
    def limiting_current_density(self, T: float, p: float, y_H2: float) -> float:
        """Calculate limiting current density"""
        # Simplified - based on H2 diffusion
        D_H2 = 1.1e-4 * (T/273)**1.75 * (101325/p)  # Binary diffusion coefficient
        delta = self.geom['thickness']['anode']
        porosity = self.mat['anode']['porosity']
        tortuosity = self.mat['anode']['tortuosity']
        
        D_eff = D_H2 * porosity / tortuosity
        i_lim = 2 * F * D_eff * p * y_H2 / (R * T * delta)
        return i_lim
    
    def cell_voltage(self, conditions: SOFCOperatingConditions) -> Tuple[float, Dict]:
        """Calculate cell voltage and overpotentials"""
        T = conditions.temperature
        p = conditions.pressure
        i = conditions.current_density
        
        # Partial pressures (simplified)
        y_H2 = conditions.inlet_fuel_composition.get('H2', 0.97)
        y_H2O = conditions.inlet_fuel_composition.get('H2O', 0.03)
        y_O2 = conditions.inlet_air_composition.get('O2', 0.21)
        
        # Account for fuel utilization
        Uf = conditions.fuel_utilization
        y_H2_out = y_H2 * (1 - Uf)
        y_H2O_out = y_H2O + y_H2 * Uf
        
        # Average compositions
        y_H2_avg = (y_H2 + y_H2_out) / 2
        y_H2O_avg = (y_H2O + y_H2O_out) / 2
        
        # Partial pressures
        p_H2 = p * y_H2_avg
        p_H2O = p * y_H2O_avg
        p_O2 = p * y_O2
        
        # Calculate voltage components
        E_nernst = self.nernst_voltage(T, p_H2, p_H2O, p_O2)
        eta_act_a = self.activation_overpotential(i, T, 'anode')
        eta_act_c = self.activation_overpotential(i, T, 'cathode')
        eta_ohm = self.ohmic_overpotential(i, T)
        
        i_lim = self.limiting_current_density(T, p, y_H2_avg)
        eta_conc = self.concentration_overpotential(i, T, i_lim)
        
        # Cell voltage
        V_cell = E_nernst - eta_act_a - eta_act_c - eta_ohm - eta_conc
        
        # Power density
        P = V_cell * i  # W/m²
        
        overpotentials = {
            'E_nernst': E_nernst,
            'eta_act_anode': eta_act_a,
            'eta_act_cathode': eta_act_c,
            'eta_ohmic': eta_ohm,
            'eta_concentration': eta_conc,
            'power_density': P
        }
        
        return V_cell, overpotentials


class ThermalModel:
    """SOFC Thermal Model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mat = config['material_properties']
        self.geom = config['geometry']
        
    def heat_generation_rate(self, i: float, V: float, E_nernst: float) -> float:
        """Calculate volumetric heat generation rate"""
        # Joule heating + entropic heat
        q_joule = i * (E_nernst - V)  # W/m²
        
        # Entropic heat (simplified)
        T = 1073  # Nominal temperature
        Delta_S = -0.178  # J/(mol·K) for H2 oxidation
        q_entropic = i * T * abs(Delta_S) / (2 * F)
        
        q_total = q_joule + q_entropic
        return q_total  # W/m²
    
    def temperature_distribution_1D(self, conditions: SOFCOperatingConditions, 
                                   V: float, E_nernst: float) -> np.ndarray:
        """1D temperature distribution (simplified)"""
        i = conditions.current_density
        T_inlet = conditions.temperature
        
        # Heat generation
        q = self.heat_generation_rate(i, V, E_nernst)
        
        # Simple 1D heat conduction with generation
        # Assuming uniform properties
        k_avg = 5.0  # Average thermal conductivity W/(m·K)
        L = sum(self.geom['thickness'].values())
        
        # Temperature rise
        Delta_T = q * L / (2 * k_avg)
        
        # Linear profile (simplified)
        x = np.linspace(0, L, 10)
        T = T_inlet + Delta_T * (x/L) * (2 - x/L)
        
        return T
    
    def temperature_distribution_2D(self, conditions: SOFCOperatingConditions,
                                   V: float, E_nernst: float, 
                                   mesh_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
        """2D temperature distribution"""
        nx, ny = mesh_size
        i = conditions.current_density
        T_inlet = conditions.temperature
        
        # Heat generation
        q = self.heat_generation_rate(i, V, E_nernst)
        
        # Create 2D field
        T_field = np.zeros((nx, ny))
        
        # Simple 2D distribution with hot spots
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        
        # Base temperature with gradient
        T_base = T_inlet + 50 * (1 - Y)  # Flow direction gradient
        
        # Add hot spots near current collectors
        for i in range(3):
            x_center = 0.2 + i * 0.3
            y_center = 0.5
            hot_spot = 30 * np.exp(-((X - x_center)**2 + (Y - y_center)**2) / 0.05)
            T_base += hot_spot
        
        # Add heat generation effect
        T_field = T_base + q * 1e-5  # Scaling factor
        
        return T_field
    
    def temperature_distribution_3D(self, conditions: SOFCOperatingConditions,
                                   V: float, E_nernst: float,
                                   mesh_size: Tuple[int, int, int] = (100, 100, 20)) -> np.ndarray:
        """3D temperature distribution"""
        nx, ny, nz = mesh_size
        i = conditions.current_density
        T_inlet = conditions.temperature
        
        # Heat generation
        q = self.heat_generation_rate(i, V, E_nernst)
        
        # Create 3D field
        T_field = np.zeros((nx, ny, nz))
        
        # 3D distribution
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        z = np.linspace(0, 1, nz)
        
        for k in range(nz):
            # Each layer has different thermal properties
            layer_factor = 1.0
            if k < nz//3:  # Anode
                layer_factor = 1.2
            elif k > 2*nz//3:  # Cathode
                layer_factor = 1.1
            
            # 2D pattern for this layer
            T_layer = self.temperature_distribution_2D(conditions, V, E_nernst, (nx, ny))
            
            # Through-thickness variation
            z_factor = 1 + 0.1 * np.sin(np.pi * z[k])
            
            T_field[:, :, k] = T_layer * layer_factor * z_factor
            
        return T_field


class MechanicalModel:
    """SOFC Mechanical/Structural Model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.mat = config['material_properties']
        self.geom = config['geometry']
        
    def thermal_stress(self, T_field: np.ndarray, T_ref: float = 298.15) -> np.ndarray:
        """Calculate thermal stress from temperature field"""
        # Simplified thermal stress calculation
        if T_field.ndim == 1:
            # 1D case - return scalar stress
            Delta_T = np.mean(T_field) - T_ref
            alpha = 11e-6  # Average CTE
            E = 100e9  # Average Young's modulus
            sigma = alpha * E * Delta_T
            return np.array([sigma])
            
        elif T_field.ndim == 2:
            # 2D case
            nx, ny = T_field.shape
            sigma = np.zeros((nx, ny, 3))  # sigma_xx, sigma_yy, sigma_xy
            
            # Temperature gradient stresses
            grad_T_x = np.gradient(T_field, axis=0)
            grad_T_y = np.gradient(T_field, axis=1)
            
            alpha = 11e-6
            E = 100e9
            nu = 0.3
            
            # Plane stress assumption
            factor = E * alpha / (1 - nu)
            sigma[:, :, 0] = factor * (T_field - T_ref)  # sigma_xx
            sigma[:, :, 1] = factor * (T_field - T_ref)  # sigma_yy
            sigma[:, :, 2] = 0.5 * factor * (grad_T_x + grad_T_y) * 10  # sigma_xy (scaled)
            
            return sigma
            
        else:  # 3D
            nx, ny, nz = T_field.shape
            sigma = np.zeros((nx, ny, nz, 6))  # Full stress tensor
            
            # Simplified 3D thermal stress
            for i in range(nz):
                if i < nz//3:  # Anode
                    alpha = self.mat['anode']['thermal_expansion']
                    E = self.mat['anode']['youngs_modulus']
                    nu = self.mat['anode']['poissons_ratio']
                elif i > 2*nz//3:  # Cathode
                    alpha = self.mat['cathode']['thermal_expansion']
                    E = self.mat['cathode']['youngs_modulus']
                    nu = self.mat['cathode']['poissons_ratio']
                else:  # Electrolyte
                    alpha = self.mat['electrolyte']['thermal_expansion']
                    E = self.mat['electrolyte']['youngs_modulus']
                    nu = self.mat['electrolyte']['poissons_ratio']
                
                factor = E * alpha / (1 - 2*nu)
                Delta_T = T_field[:, :, i] - T_ref
                
                # Normal stresses
                sigma[:, :, i, 0] = factor * Delta_T  # sigma_xx
                sigma[:, :, i, 1] = factor * Delta_T  # sigma_yy
                sigma[:, :, i, 2] = factor * Delta_T  # sigma_zz
                
                # Shear stresses (simplified)
                if i > 0:
                    Delta_T_interface = T_field[:, :, i] - T_field[:, :, i-1]
                    sigma[:, :, i, 3] = 0.1 * factor * Delta_T_interface  # sigma_xy
                    sigma[:, :, i, 4] = 0.1 * factor * Delta_T_interface  # sigma_xz
                    sigma[:, :, i, 5] = 0.1 * factor * Delta_T_interface  # sigma_yz
                    
            return sigma
    
    def von_mises_stress(self, sigma: np.ndarray) -> np.ndarray:
        """Calculate von Mises equivalent stress"""
        if sigma.ndim == 1:
            return np.abs(sigma)
            
        elif sigma.shape[-1] == 3:  # 2D plane stress
            s11, s22, s12 = sigma[..., 0], sigma[..., 1], sigma[..., 2]
            vm = np.sqrt(s11**2 + s22**2 - s11*s22 + 3*s12**2)
            return vm
            
        else:  # 3D full tensor
            s11, s22, s33 = sigma[..., 0], sigma[..., 1], sigma[..., 2]
            s12, s13, s23 = sigma[..., 3], sigma[..., 4], sigma[..., 5]
            
            vm = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 
                               6*(s12**2 + s13**2 + s23**2)))
            return vm
    
    def creep_strain_rate(self, sigma: float, T: float, t: float) -> float:
        """Norton-Bailey creep model"""
        # Material-specific creep parameters
        A = 1e-10  # Creep coefficient
        n = 3.0    # Stress exponent
        Q = 200000  # Activation energy J/mol
        
        if T < 773:  # Below 500°C, negligible creep
            return 0
            
        eps_dot = A * (sigma**n) * np.exp(-Q / (R * T))
        return eps_dot
    
    def damage_accumulation(self, sigma: np.ndarray, T: np.ndarray, 
                           cycles: int, time: float) -> np.ndarray:
        """Calculate damage field based on stress and temperature history"""
        # Simplified damage model
        if isinstance(sigma, (int, float)) or sigma.ndim == 1:
            # Scalar or 1D
            sigma_vm = np.abs(sigma) if isinstance(sigma, (int, float)) else np.abs(sigma[0])
            T_avg = np.mean(T)
            
            # Fatigue damage (Coffin-Manson)
            N_f = 1e6 * (300e6 / sigma_vm)**3  # Cycles to failure
            D_fatigue = cycles / N_f if N_f > 0 else 1.0
            
            # Creep damage
            eps_creep = self.creep_strain_rate(sigma_vm, T_avg, time) * time * 3600
            D_creep = eps_creep / 0.01  # 1% creep strain as failure
            
            # Total damage
            D_total = min(D_fatigue + D_creep, 1.0)
            return np.array([D_total])
            
        else:
            # Multi-dimensional
            sigma_vm = self.von_mises_stress(sigma)
            
            # Initialize damage field
            D = np.zeros_like(sigma_vm)
            
            # Calculate damage at each point
            for idx in np.ndindex(sigma_vm.shape):
                s = sigma_vm[idx]
                T_local = T[idx] if T.shape == sigma_vm.shape else np.mean(T)
                
                # Fatigue damage
                if s > 1e6:  # Threshold stress
                    N_f = 1e6 * (300e6 / s)**3
                    D_fatigue = cycles / N_f
                else:
                    D_fatigue = 0
                    
                # Creep damage
                eps_creep = self.creep_strain_rate(s, T_local, time) * time * 3600
                D_creep = eps_creep / 0.01
                
                # Combined damage
                D[idx] = min(D_fatigue + D_creep, 1.0)
                
            return D


class DegradationModel:
    """SOFC Degradation Mechanisms Model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.deg = config['degradation']
        
    def nickel_coarsening(self, T: float, t: float, overpotential: float) -> float:
        """Ni particle coarsening in anode"""
        # Ostwald ripening model
        Q = self.deg['ni_coarsening']['activation_energy']
        k0 = self.deg['ni_coarsening']['pre_exponential']
        
        # Temperature and overpotential accelerated coarsening
        k = k0 * np.exp(-Q / (R * T)) * (1 + abs(overpotential))
        
        # Particle size growth
        r0 = 0.5e-6  # Initial particle radius (m)
        r = r0 * (1 + k * t)**0.33
        
        # Degradation metric (relative increase)
        degradation = (r - r0) / r0
        return min(degradation, 1.0)
    
    def chromium_poisoning(self, T: float, t: float, i: float) -> float:
        """Cathode Cr poisoning from interconnect"""
        # Cr evaporation and deposition model
        rate = self.deg['cr_poisoning']['deposition_rate']
        
        # Temperature and current dependent
        rate_actual = rate * np.exp(-80000 / (R * T)) * (i / 5000)
        
        # Coverage fraction
        coverage = rate_actual * t * 3600 / 1e-6  # Normalized by monolayer
        coverage = min(coverage, self.deg['cr_poisoning']['coverage_threshold'])
        
        return coverage
    
    def crack_density(self, sigma_history: np.ndarray, cycles: int) -> float:
        """Calculate crack density from stress history"""
        # Paris law for crack growth
        K_IC = self.deg['crack']['critical_stress_intensity']
        C = self.deg['crack']['paris_law_C']
        m = self.deg['crack']['paris_law_m']
        
        # Maximum stress
        sigma_max = np.max(np.abs(sigma_history))
        
        # Initial crack size
        a0 = 1e-6  # 1 μm
        
        # Stress intensity factor (simplified)
        K = sigma_max * np.sqrt(np.pi * a0)
        
        if K < K_IC:
            # Subcritical crack growth
            da_dN = C * (K**m)
            a_final = a0 + da_dN * cycles
            
            # Crack density (cracks per unit area)
            rho = a_final / a0 - 1
            return min(rho, 1.0)
        else:
            # Critical crack - failure
            return 1.0
    
    def delamination_risk(self, T_gradient: float, thermal_cycles: int) -> float:
        """Interface delamination risk assessment"""
        # Energy release rate
        G_c = self.deg['delamination']['critical_energy_release_rate']
        
        # Thermal mismatch stress (simplified)
        Delta_alpha = 6e-6  # CTE mismatch
        E = 100e9  # Modulus
        h = 10e-6  # Layer thickness
        
        # Energy release rate
        G = 0.5 * (Delta_alpha * T_gradient)**2 * E * h
        
        # Cyclic loading factor
        cycle_factor = (thermal_cycles / 1000)**0.5
        
        # Delamination risk
        risk = (G / G_c) * cycle_factor
        return min(risk, 1.0)
    
    def total_degradation(self, conditions: SOFCOperatingConditions,
                         overpotentials: Dict, stress: np.ndarray,
                         cycles: int) -> Dict[str, float]:
        """Calculate all degradation metrics"""
        T = conditions.temperature
        t = conditions.time
        i = conditions.current_density
        
        # Individual degradation mechanisms
        ni_coars = self.nickel_coarsening(T, t, overpotentials.get('eta_act_anode', 0))
        cr_pois = self.chromium_poisoning(T, t, i)
        
        # Mechanical degradation
        if isinstance(stress, np.ndarray) and stress.size > 0:
            crack_dens = self.crack_density(stress.flatten(), cycles)
        else:
            crack_dens = 0
            
        delam_risk = self.delamination_risk(50, cycles)  # Assumed gradient
        
        # Voltage degradation rate (mV/1000h)
        deg_rate = 2 + 5*ni_coars + 10*cr_pois + 20*crack_dens + 15*delam_risk
        
        return {
            'nickel_coarsening': ni_coars,
            'chromium_poisoning': cr_pois,
            'crack_density': crack_dens,
            'delamination_risk': delam_risk,
            'voltage_degradation_rate': deg_rate,
            'estimated_lifetime': 50000 / deg_rate if deg_rate > 0 else 50000  # hours
        }