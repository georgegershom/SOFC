"""
Physics-Based and Data-Driven Degradation Models for SOFCs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class SOFCDegradationModels:
    """Collection of degradation models at different fidelity levels"""
    
    def __init__(self):
        self.gas_constant = 8.314  # J/mol/K
        self.faraday = 96485  # C/mol
        
    def compute_voltage_degradation(self,
                                   params: Dict[str, float],
                                   time_hours: np.ndarray,
                                   fidelity: str = 'HF') -> Dict[str, np.ndarray]:
        """
        Compute voltage degradation over time
        
        Args:
            params: Dictionary of input parameters
            time_hours: Time vector in hours
            fidelity: Model fidelity level
        
        Returns:
            Dictionary with degradation results
        """
        if fidelity == 'LF':
            return self._lf_voltage_degradation(params, time_hours)
        elif fidelity == 'MF':
            return self._mf_voltage_degradation(params, time_hours)
        else:
            return self._hf_voltage_degradation(params, time_hours)
    
    def _lf_voltage_degradation(self, 
                               params: Dict[str, float], 
                               time_hours: np.ndarray) -> Dict[str, np.ndarray]:
        """Low-fidelity lumped parameter model"""
        
        # Extract key parameters
        T = params.get('system.temperature', 1023)
        j = params.get('system.current_density', 0.5)
        p_O2 = params.get('system.pressure', 1.0) * 0.21  # Partial pressure of O2
        
        # Initial voltage (Nernst + activation + ohmic)
        E_nernst = 1.23 - 2.304e-4 * (T - 298)
        
        # Activation overpotential (simplified Butler-Volmer)
        j0 = 0.1 * np.exp(-80000 / (self.gas_constant * T))
        if j > 0:
            eta_act = (self.gas_constant * T / (2 * self.faraday)) * np.log(j / j0)
        else:
            eta_act = 0
        
        # Ohmic losses
        R_ohmic = 0.15  # Ω·cm²
        eta_ohmic = j * R_ohmic
        
        # Initial voltage
        V_initial = E_nernst - eta_act - eta_ohmic
        
        # Simple linear degradation model
        degradation_rate = 0.5e-5 * j * np.exp(-100000 / (self.gas_constant * T))  # V/h
        
        # Add cyclic degradation
        n_cycles = params.get('transient.thermal_cycles', 0)
        cycle_damage = n_cycles * 1e-5  # V per cycle
        
        # Voltage over time
        voltage = V_initial - degradation_rate * time_hours - cycle_damage
        
        # ASR increase
        ASR_initial = R_ohmic
        ASR_degradation_rate = 1e-6 * np.exp(-80000 / (self.gas_constant * T))
        ASR = ASR_initial + ASR_degradation_rate * time_hours
        
        return {
            'voltage': voltage,
            'ASR': ASR,
            'power_density': voltage * j,
            'degradation_rate': np.ones_like(time_hours) * degradation_rate * 1000,  # mV/1000h
            'efficiency': voltage / 1.48,  # Assuming HHV of H2
        }
    
    def _mf_voltage_degradation(self, 
                               params: Dict[str, float], 
                               time_hours: np.ndarray) -> Dict[str, np.ndarray]:
        """Medium-fidelity model with multiple degradation mechanisms"""
        
        # Extract parameters
        T = params.get('system.temperature', 1023)
        j = params.get('system.current_density', 0.5)
        p_H2 = params.get('system.pressure', 1.0) * 0.97
        p_H2O = params.get('system.pressure', 1.0) * 0.03
        p_O2 = params.get('system.pressure', 1.0) * 0.21
        
        # Material parameters
        porosity_anode = params.get('anode_material.anode_porosity', 0.35)
        porosity_cathode = params.get('cathode_material.cathode_porosity', 0.40)
        ni_particle_size = params.get('anode_material.ni_particle_size', 1.5)
        
        # Degradation parameters
        Cr_rate = params.get('degradation.chromium_poisoning_rate', 1e-9)
        C_rate = params.get('degradation.carbon_deposition_rate', 1e-8)
        S_level = params.get('degradation.sulfur_poisoning_level', 1)
        
        def degradation_odes(y, t):
            """ODEs for degradation mechanisms"""
            V, R_ohmic, porosity_a, d_Ni, theta_Cr, theta_C, theta_S = y
            
            # Ni coarsening (Ostwald ripening)
            if t > 0:
                dd_Ni_dt = 1e-6 * np.exp(-150000 / (self.gas_constant * T)) * t**(-1/3)
            else:
                dd_Ni_dt = 0
            
            # Porosity reduction due to sintering
            dporosity_dt = -1e-8 * np.exp(-120000 / (self.gas_constant * T))
            
            # Chromium poisoning (cathode)
            dtheta_Cr_dt = Cr_rate * (1 - theta_Cr)
            
            # Carbon deposition (anode)
            dtheta_C_dt = C_rate * (1 - theta_C) * (p_H2 / p_H2O - 1)
            
            # Sulfur poisoning (anode)
            dtheta_S_dt = 1e-7 * S_level * (1 - theta_S)
            
            # Ohmic resistance increase
            dR_ohmic_dt = 1e-7 * (1 + theta_Cr + theta_C) * np.exp(-90000 / (self.gas_constant * T))
            
            # Voltage degradation
            dV_dt = -j * dR_ohmic_dt - 0.1 * (dtheta_Cr_dt + dtheta_C_dt + dtheta_S_dt)
            
            return [dV_dt, dR_ohmic_dt, dporosity_dt, dd_Ni_dt, 
                   dtheta_Cr_dt, dtheta_C_dt, dtheta_S_dt]
        
        # Initial conditions
        E_nernst = 1.23 - 2.304e-4 * (T - 298) + \
                  (self.gas_constant * T / (4 * self.faraday)) * np.log(p_O2 * (p_H2 / p_H2O)**2)
        
        # Initial activation overpotentials
        j0_anode = 0.5 * np.exp(-100000 / (self.gas_constant * T))
        j0_cathode = 0.05 * np.exp(-120000 / (self.gas_constant * T))
        
        if j > 0:
            eta_act_anode = (self.gas_constant * T / (2 * self.faraday)) * np.log(j / j0_anode)
            eta_act_cathode = (self.gas_constant * T / (2 * self.faraday)) * np.log(j / j0_cathode)
        else:
            eta_act_anode = 0
            eta_act_cathode = 0
        
        # Initial ohmic resistance
        R_ohmic_initial = 0.1 + 0.05 / porosity_anode + 0.03 / porosity_cathode
        
        # Initial voltage
        V_initial = E_nernst - eta_act_anode - eta_act_cathode - j * R_ohmic_initial
        
        # Initial state vector
        y0 = [V_initial, R_ohmic_initial, porosity_anode, ni_particle_size, 0, 0, 0]
        
        # Solve ODEs
        solution = odeint(degradation_odes, y0, time_hours)
        
        voltage = solution[:, 0]
        R_ohmic = solution[:, 1]
        porosity = solution[:, 2]
        particle_size = solution[:, 3]
        Cr_coverage = solution[:, 4]
        C_coverage = solution[:, 5]
        S_coverage = solution[:, 6]
        
        # Calculate ASR
        if j > 0:
            ASR = R_ohmic + eta_act_anode / j + eta_act_cathode / j
        else:
            ASR = R_ohmic
        
        # Power density
        power_density = voltage * j
        
        # Degradation rate (mV/1000h)
        degradation_rate = -np.gradient(voltage) / np.gradient(time_hours) * 1000
        
        return {
            'voltage': voltage,
            'ASR': ASR,
            'power_density': power_density,
            'degradation_rate': degradation_rate,
            'porosity': porosity,
            'ni_particle_size': particle_size,
            'cr_poisoning': Cr_coverage,
            'carbon_deposition': C_coverage,
            'sulfur_poisoning': S_coverage,
            'efficiency': voltage / 1.48,
        }
    
    def _hf_voltage_degradation(self, 
                               params: Dict[str, float], 
                               time_hours: np.ndarray) -> Dict[str, np.ndarray]:
        """High-fidelity model with microstructure evolution"""
        
        # All MF mechanisms plus microstructural evolution
        mf_results = self._mf_voltage_degradation(params, time_hours)
        
        # Additional HF calculations
        T = params.get('system.temperature', 1023)
        j = params.get('system.current_density', 0.5)
        
        # Microstructural parameters
        tpb_density = params.get('anode_material.tpb_density', 1e7)
        connectivity_ni = params.get('microstructure.connectivity_ni', 0.95)
        connectivity_ysz = params.get('microstructure.connectivity_ysz', 0.95)
        specific_surface = params.get('microstructure.specific_surface_area', 3e6)
        
        # TPB density evolution
        tpb_degradation_rate = 1e-4 * np.exp(-130000 / (self.gas_constant * T))
        tpb_evolution = tpb_density * np.exp(-tpb_degradation_rate * time_hours)
        
        # Connectivity evolution (percolation)
        connectivity_loss_rate = 1e-5 * j * np.exp(-110000 / (self.gas_constant * T))
        connectivity_ni_evolution = connectivity_ni * np.exp(-connectivity_loss_rate * time_hours)
        connectivity_ysz_evolution = connectivity_ysz * np.exp(-connectivity_loss_rate * time_hours * 0.5)
        
        # Surface area evolution (sintering)
        surface_reduction_rate = 5e-5 * np.exp(-140000 / (self.gas_constant * T))
        surface_evolution = specific_surface * np.exp(-surface_reduction_rate * time_hours)
        
        # Crack propagation model
        stress_intensity = j * 10 + params.get('transient.thermal_cycles', 0) * 0.1
        K_IC = params.get('electrolyte_material.fracture_toughness', 2.2)
        crack_growth_rate = 1e-10 * (stress_intensity / K_IC)**4
        crack_length = crack_growth_rate * time_hours
        
        # Creep strain accumulation
        stress = j * 50  # MPa
        Q_creep = params.get('interconnect_material.creep_activation_energy', 300) * 1000
        n_creep = params.get('interconnect_material.creep_stress_exponent', 5)
        creep_rate = 1e-20 * stress**n_creep * np.exp(-Q_creep / (self.gas_constant * T))
        creep_strain = creep_rate * time_hours
        
        # Oxide scale growth (parabolic)
        k_p = params.get('interconnect_material.oxide_growth_rate', 1e-13)
        oxide_thickness = np.sqrt(2 * k_p * time_hours * 3600)  # Convert to seconds
        
        # Update voltage with microstructural effects
        microstructure_factor = (tpb_evolution / tpb_density) * \
                               np.sqrt(connectivity_ni_evolution * connectivity_ysz_evolution) * \
                               (surface_evolution / specific_surface)**0.3
        
        voltage_hf = mf_results['voltage'] * microstructure_factor
        
        # Update ASR with oxide scale
        ASR_oxide = oxide_thickness * 1e4  # Assuming resistivity of oxide
        ASR_hf = mf_results['ASR'] + ASR_oxide
        
        # Mechanical failure probability (Weibull)
        m_weibull = params.get('electrolyte_material.weibull_modulus', 10)
        sigma_max = stress + 10 * np.sqrt(crack_length)
        sigma_0 = params.get('electrolyte_material.yield_strength', 300)
        failure_probability = 1 - np.exp(-(sigma_max / sigma_0)**m_weibull)
        
        # Compile HF results
        hf_results = mf_results.copy()
        hf_results.update({
            'voltage': voltage_hf,
            'ASR': ASR_hf,
            'power_density': voltage_hf * j,
            'tpb_density': tpb_evolution,
            'connectivity_ni': connectivity_ni_evolution,
            'connectivity_ysz': connectivity_ysz_evolution,
            'specific_surface_area': surface_evolution,
            'crack_length': crack_length,
            'creep_strain': creep_strain,
            'oxide_thickness': oxide_thickness,
            'failure_probability': failure_probability,
            'microstructure_factor': microstructure_factor,
        })
        
        return hf_results
    
    def compute_thermal_stress(self,
                               params: Dict[str, float],
                               temperature_profile: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute thermal stress distribution
        
        Args:
            params: Dictionary of input parameters
            temperature_profile: Temperature distribution array
        
        Returns:
            Dictionary with stress results
        """
        # Material properties
        CTE_anode = 12.5e-6  # 1/K for Ni-YSZ
        CTE_electrolyte = params.get('electrolyte_material.thermal_expansion_ysz', 10.5e-6)
        CTE_cathode = 11.8e-6  # 1/K for LSCF
        CTE_interconnect = params.get('interconnect_material.thermal_expansion_ic', 11.8e-6)
        
        E_electrolyte = params.get('electrolyte_material.youngs_modulus', 200) * 1e9  # Pa
        nu_electrolyte = params.get('electrolyte_material.poissons_ratio', 0.30)
        
        # Temperature gradient
        T_ref = 298  # K
        delta_T = temperature_profile - T_ref
        
        # Thermal strain mismatch
        strain_mismatch_anode = (CTE_anode - CTE_electrolyte) * delta_T
        strain_mismatch_cathode = (CTE_cathode - CTE_electrolyte) * delta_T
        strain_mismatch_ic = (CTE_interconnect - CTE_electrolyte) * delta_T
        
        # Stress calculation (plane stress)
        stress_anode = E_electrolyte / (1 - nu_electrolyte) * strain_mismatch_anode
        stress_cathode = E_electrolyte / (1 - nu_electrolyte) * strain_mismatch_cathode
        stress_ic = E_electrolyte / (1 - nu_electrolyte) * strain_mismatch_ic
        
        # Von Mises stress
        von_mises = np.sqrt(stress_anode**2 + stress_cathode**2 + stress_ic**2)
        
        # Curvature calculation
        thickness_total = params.get('geometry.anode_thickness', 500) + \
                         params.get('geometry.electrolyte_thickness', 10) + \
                         params.get('geometry.cathode_thickness', 50)
        
        curvature = 6 * (CTE_anode - CTE_cathode) * delta_T / (thickness_total * 1e-6)
        
        return {
            'stress_anode': stress_anode,
            'stress_cathode': stress_cathode,
            'stress_interconnect': stress_ic,
            'von_mises_stress': von_mises,
            'curvature': curvature,
            'max_stress': np.maximum.reduce([np.abs(stress_anode), 
                                            np.abs(stress_cathode), 
                                            np.abs(stress_ic)])
        }
    
    def compute_electrochemical_response(self,
                                        params: Dict[str, float],
                                        current_range: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute IV curves and impedance
        
        Args:
            params: Dictionary of input parameters
            current_range: Current density array for IV curve
        
        Returns:
            Dictionary with electrochemical responses
        """
        T = params.get('system.temperature', 1023)
        p_H2 = params.get('system.pressure', 1.0) * 0.97
        p_H2O = params.get('system.pressure', 1.0) * 0.03
        p_O2 = params.get('system.pressure', 1.0) * 0.21
        
        # Nernst voltage
        E_nernst = 1.23 - 2.304e-4 * (T - 298) + \
                  (self.gas_constant * T / (4 * self.faraday)) * np.log(p_O2 * (p_H2 / p_H2O)**2)
        
        # Exchange current densities
        j0_anode = 0.5 * np.exp(-100000 / (self.gas_constant * T))
        j0_cathode = 0.05 * np.exp(-120000 / (self.gas_constant * T))
        
        # Ohmic resistance
        R_ohmic = 0.15  # Ω·cm²
        
        # IV curve calculation
        voltage = []
        for j in current_range:
            if j == 0:
                V = E_nernst
            else:
                # Butler-Volmer for activation
                eta_act_anode = (self.gas_constant * T / (2 * self.faraday)) * np.log(j / j0_anode)
                eta_act_cathode = (self.gas_constant * T / (2 * self.faraday)) * np.log(j / j0_cathode)
                
                # Concentration overpotential (simplified)
                j_lim = 2.0  # A/cm²
                eta_conc = (self.gas_constant * T / (2 * self.faraday)) * np.log(1 - j / j_lim)
                
                V = E_nernst - eta_act_anode - eta_act_cathode - j * R_ohmic - eta_conc
            
            voltage.append(V)
        
        voltage = np.array(voltage)
        power_density = voltage * current_range
        
        # Impedance calculation (simplified Randles circuit)
        frequencies = np.logspace(-2, 5, 100)  # Hz
        omega = 2 * np.pi * frequencies
        
        # Ohmic resistance
        R_ohm = R_ohmic
        
        # Charge transfer resistance
        R_ct_anode = (self.gas_constant * T) / (2 * self.faraday * j0_anode * 1)
        R_ct_cathode = (self.gas_constant * T) / (2 * self.faraday * j0_cathode * 1)
        
        # Double layer capacitance
        C_dl = 20e-6  # F/cm²
        
        # Warburg element (diffusion)
        sigma = 0.01  # Warburg coefficient
        
        # Total impedance
        Z_real = R_ohm + R_ct_anode / (1 + (omega * R_ct_anode * C_dl)**2) + \
                R_ct_cathode / (1 + (omega * R_ct_cathode * C_dl)**2) + \
                sigma / np.sqrt(omega)
        
        Z_imag = -omega * R_ct_anode**2 * C_dl / (1 + (omega * R_ct_anode * C_dl)**2) - \
                -omega * R_ct_cathode**2 * C_dl / (1 + (omega * R_ct_cathode * C_dl)**2) - \
                sigma / np.sqrt(omega)
        
        return {
            'current_density': current_range,
            'voltage': voltage,
            'power_density': power_density,
            'efficiency': voltage / 1.48,
            'frequencies': frequencies,
            'impedance_real': Z_real,
            'impedance_imag': Z_imag,
            'OCV': E_nernst,
            'ASR_total': R_ohm + R_ct_anode + R_ct_cathode
        }
    
    def generate_synthetic_microstructure(self,
                                         params: Dict[str, float],
                                         grid_size: Tuple[int, int, int] = (50, 50, 50),
                                         seed: int = 42) -> np.ndarray:
        """
        Generate synthetic 3D microstructure
        
        Args:
            params: Dictionary of input parameters
            grid_size: 3D grid dimensions
            seed: Random seed
        
        Returns:
            3D array with phase labels (0: pore, 1: Ni, 2: YSZ)
        """
        np.random.seed(seed)
        
        # Get microstructural parameters
        porosity = params.get('anode_material.anode_porosity', 0.35)
        ni_fraction = params.get('anode_material.ni_volume_fraction', 0.40)
        ysz_fraction = params.get('anode_material.ysz_volume_fraction', 0.35)
        particle_size = params.get('anode_material.ni_particle_size', 1.5)
        
        # Normalize fractions
        total = porosity + ni_fraction + ysz_fraction
        porosity /= total
        ni_fraction /= total
        ysz_fraction /= total
        
        # Initialize grid
        microstructure = np.zeros(grid_size, dtype=np.uint8)
        
        # Generate random seeds for each phase
        n_particles_ni = int(ni_fraction * np.prod(grid_size) / (particle_size**3))
        n_particles_ysz = int(ysz_fraction * np.prod(grid_size) / ((particle_size * 0.7)**3))
        
        # Place Ni particles
        for _ in range(n_particles_ni):
            center = np.random.rand(3) * np.array(grid_size)
            radius = np.random.normal(particle_size, particle_size * 0.2)
            
            # Create sphere
            x, y, z = np.ogrid[:grid_size[0], :grid_size[1], :grid_size[2]]
            mask = ((x - center[0])**2 + (y - center[1])**2 + 
                   (z - center[2])**2) <= radius**2
            microstructure[mask] = 1
        
        # Place YSZ particles
        for _ in range(n_particles_ysz):
            center = np.random.rand(3) * np.array(grid_size)
            radius = np.random.normal(particle_size * 0.7, particle_size * 0.1)
            
            # Create sphere
            x, y, z = np.ogrid[:grid_size[0], :grid_size[1], :grid_size[2]]
            mask = ((x - center[0])**2 + (y - center[1])**2 + 
                   (z - center[2])**2) <= radius**2
            
            # Only place YSZ where there's no Ni
            microstructure[mask & (microstructure == 0)] = 2
        
        return microstructure