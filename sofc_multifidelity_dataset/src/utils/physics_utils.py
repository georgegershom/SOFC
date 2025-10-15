"""Physics utility functions for SOFC modeling."""

import numpy as np
from scipy import constants
from typing import Dict, Tuple, Optional, Union


class SOFCPhysics:
    """Physics models for Solid Oxide Fuel Cell calculations."""
    
    def __init__(self, config: Dict):
        """Initialize with configuration parameters."""
        self.config = config
        self.R = constants.R  # Universal gas constant
        self.F = constants.physical_constants['Faraday constant'][0]
        
    def nernst_voltage(self, T: float, pH2: float, pH2O: float, pO2: float) -> float:
        """
        Calculate Nernst voltage for SOFC.
        
        Args:
            T: Temperature [K]
            pH2: Partial pressure of H2 [Pa]
            pH2O: Partial pressure of H2O [Pa]
            pO2: Partial pressure of O2 [Pa]
            
        Returns:
            Nernst voltage [V]
        """
        # Standard Gibbs free energy change for H2 + 0.5*O2 -> H2O
        dG0 = -241830 + 44.43 * T  # J/mol (temperature dependent)
        
        # Nernst equation
        E0 = -dG0 / (2 * self.F)
        E_nernst = E0 + (self.R * T) / (2 * self.F) * np.log(pH2 * np.sqrt(pO2) / pH2O)
        
        return E_nernst
    
    def butler_volmer_current(self, eta: float, T: float, i0: float, 
                             alpha: float = 0.5, n: int = 2) -> float:
        """
        Calculate current density using Butler-Volmer equation.
        
        Args:
            eta: Overpotential [V]
            T: Temperature [K]
            i0: Exchange current density [A/m²]
            alpha: Transfer coefficient [-]
            n: Number of electrons
            
        Returns:
            Current density [A/m²]
        """
        exp_term = n * self.F * eta / (self.R * T)
        i = i0 * (np.exp(alpha * exp_term) - np.exp(-(1 - alpha) * exp_term))
        return i
    
    def ohmic_resistance(self, T: float, thickness: float, 
                        conductivity_params: Dict) -> float:
        """
        Calculate ohmic resistance with temperature dependence.
        
        Args:
            T: Temperature [K]
            thickness: Layer thickness [m]
            conductivity_params: Dictionary with 'sigma_0' and 'E_act'
            
        Returns:
            Area-specific resistance [Ohm*m²]
        """
        sigma_0 = conductivity_params.get('sigma_0', 3.34e4)
        E_act = conductivity_params.get('E_act', 80000)
        
        # Arrhenius temperature dependence
        sigma = sigma_0 * np.exp(-E_act / (self.R * T))
        R_ohm = thickness / sigma
        
        return R_ohm
    
    def concentration_overpotential(self, i: float, i_lim: float, T: float) -> float:
        """
        Calculate concentration overpotential.
        
        Args:
            i: Current density [A/m²]
            i_lim: Limiting current density [A/m²]
            T: Temperature [K]
            
        Returns:
            Concentration overpotential [V]
        """
        if abs(i) >= abs(i_lim):
            return np.sign(i) * np.inf
        
        eta_conc = (self.R * T) / (2 * self.F) * np.log(1 - i / i_lim)
        return eta_conc
    
    def thermal_stress(self, T: float, T_ref: float, CTE_diff: float, 
                      E: float, nu: float) -> float:
        """
        Calculate thermal stress from temperature change.
        
        Args:
            T: Current temperature [K]
            T_ref: Reference temperature [K]
            CTE_diff: CTE mismatch [1/K]
            E: Elastic modulus [Pa]
            nu: Poisson's ratio [-]
            
        Returns:
            Thermal stress [Pa]
        """
        delta_T = T - T_ref
        sigma = E * CTE_diff * delta_T / (1 - nu)
        return sigma
    
    def creep_strain_rate(self, sigma: float, T: float, 
                         creep_params: Dict) -> float:
        """
        Calculate creep strain rate using Norton's law.
        
        Args:
            sigma: Stress [Pa]
            T: Temperature [K]
            creep_params: Dictionary with 'A', 'n', and 'Q'
            
        Returns:
            Creep strain rate [1/s]
        """
        A = creep_params.get('A', 1e-25)
        n = creep_params.get('n', 3.5)
        Q = creep_params.get('Q', 3e5)
        
        eps_dot = A * (sigma ** n) * np.exp(-Q / (self.R * T))
        return eps_dot
    
    def ni_coarsening_rate(self, T: float, t: float, 
                          coarsening_params: Dict) -> float:
        """
        Calculate Ni particle coarsening rate.
        
        Args:
            T: Temperature [K]
            t: Time [s]
            coarsening_params: Dictionary with parameters
            
        Returns:
            Particle size increase rate [m/s]
        """
        A = coarsening_params.get('pre_exponential', 1e-18)
        Q = coarsening_params.get('activation_energy', 2e5)
        n = coarsening_params.get('time_exponent', 0.33)
        
        # Power law coarsening
        d_dt = A * np.exp(-Q / (self.R * T)) * (t ** (n - 1))
        return d_dt
    
    def crack_growth_rate(self, K: float, K_th: float, K_c: float, 
                         paris_params: Dict) -> float:
        """
        Calculate crack growth rate using Paris law.
        
        Args:
            K: Stress intensity factor [Pa*sqrt(m)]
            K_th: Threshold SIF [Pa*sqrt(m)]
            K_c: Critical SIF [Pa*sqrt(m)]
            paris_params: Dictionary with 'C' and 'm'
            
        Returns:
            Crack growth rate [m/cycle]
        """
        if K < K_th or K >= K_c:
            return 0.0
        
        C = paris_params.get('C', 1e-12)
        m = paris_params.get('m', 3)
        
        da_dN = C * ((K - K_th) ** m)
        return da_dN
    
    def heat_generation_rate(self, i: float, V: float, E_tn: float) -> float:
        """
        Calculate volumetric heat generation rate in SOFC.
        
        Args:
            i: Current density [A/m²]
            V: Operating voltage [V]
            E_tn: Thermoneutral voltage [V]
            
        Returns:
            Heat generation rate [W/m³]
        """
        q_gen = i * (E_tn - V)
        return q_gen
    
    def limiting_current_density(self, T: float, p: float, 
                                 x_fuel: float, D_eff: float, 
                                 L: float) -> float:
        """
        Calculate limiting current density for mass transport.
        
        Args:
            T: Temperature [K]
            p: Pressure [Pa]
            x_fuel: Fuel mole fraction [-]
            D_eff: Effective diffusivity [m²/s]
            L: Diffusion length [m]
            
        Returns:
            Limiting current density [A/m²]
        """
        C_fuel = p * x_fuel / (self.R * T)  # Concentration [mol/m³]
        i_lim = 2 * self.F * D_eff * C_fuel / L
        return i_lim


class MaterialProperties:
    """Temperature-dependent material properties for SOFC components."""
    
    @staticmethod
    def elastic_modulus_YSZ(T: float) -> float:
        """Young's modulus of YSZ as function of temperature."""
        E_0 = 200e9  # Pa at room temperature
        return E_0 * (1 - 0.0004 * (T - 298))
    
    @staticmethod
    def elastic_modulus_Ni(T: float) -> float:
        """Young's modulus of Ni as function of temperature."""
        E_0 = 200e9  # Pa at room temperature
        return E_0 * (1 - 0.0003 * (T - 298))
    
    @staticmethod
    def CTE_YSZ(T: float) -> float:
        """Coefficient of thermal expansion of YSZ."""
        return 10.5e-6 * (1 + 0.0001 * (T - 298))  # 1/K
    
    @staticmethod
    def CTE_Ni(T: float) -> float:
        """Coefficient of thermal expansion of Ni."""
        return 13.4e-6 * (1 + 0.00015 * (T - 298))  # 1/K
    
    @staticmethod
    def thermal_conductivity_YSZ(T: float) -> float:
        """Thermal conductivity of YSZ."""
        return 2.0 + 0.0001 * (T - 1073)  # W/m/K
    
    @staticmethod
    def ionic_conductivity_YSZ(T: float) -> float:
        """Ionic conductivity of YSZ."""
        sigma_0 = 3.34e4  # S/m
        E_act = 80000  # J/mol
        R = constants.R
        return sigma_0 * np.exp(-E_act / (R * T))  # S/m


class DegradationModels:
    """Degradation models for SOFC components."""
    
    @staticmethod
    def voltage_degradation_rate(t: float, T: float, i: float, 
                                 deg_params: Dict) -> float:
        """
        Calculate voltage degradation rate.
        
        Args:
            t: Time [hours]
            T: Temperature [K]
            i: Current density [A/m²]
            deg_params: Degradation parameters
            
        Returns:
            Degradation rate [%/1000h]
        """
        # Empirical degradation model
        A = deg_params.get('A', 0.1)
        E_act = deg_params.get('E_act', 50000)
        n = deg_params.get('current_exp', 0.5)
        m = deg_params.get('time_exp', 0.25)
        
        R = constants.R
        rate = A * np.exp(-E_act / (R * T)) * (i / 5000) ** n * (t / 1000) ** m
        
        return rate * 1000  # Convert to %/1000h
    
    @staticmethod
    def anode_reoxidation_damage(redox_cycles: int, 
                                 volume_change: float = 0.24) -> float:
        """
        Calculate cumulative damage from redox cycling.
        
        Args:
            redox_cycles: Number of redox cycles
            volume_change: Fractional volume change Ni->NiO
            
        Returns:
            Damage parameter [0-1]
        """
        # Empirical damage accumulation
        damage = 1 - np.exp(-0.01 * redox_cycles * volume_change)
        return min(damage, 1.0)
    
    @staticmethod
    def chromium_poisoning_rate(T: float, t: float, 
                               cr_params: Dict) -> float:
        """
        Calculate chromium deposition rate at cathode.
        
        Args:
            T: Temperature [K]
            t: Time [s]
            cr_params: Chromium poisoning parameters
            
        Returns:
            Deposition rate [kg/m²/s]
        """
        k_0 = cr_params.get('deposition_rate', 1e-9)
        E_act = cr_params.get('activation_energy', 1.5e5)
        R = constants.R
        
        k = k_0 * np.exp(-E_act / (R * T))
        return k
    
    @staticmethod
    def interface_delamination_criterion(G: float, G_c: float, 
                                        mode_mixity: float = 0.5) -> bool:
        """
        Check interface delamination using energy criterion.
        
        Args:
            G: Energy release rate [J/m²]
            G_c: Critical energy release rate [J/m²]
            mode_mixity: Mode I/II mixity parameter
            
        Returns:
            True if delamination occurs
        """
        # Mixed-mode criterion
        G_eff = G * (1 + mode_mixity)
        return G_eff >= G_c