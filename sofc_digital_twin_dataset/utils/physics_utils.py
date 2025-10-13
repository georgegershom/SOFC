"""
Physics utility functions for SOFC digital twin dataset.
"""

import numpy as np
from scipy import constants
from typing import Dict, Any, Tuple, List
import warnings


class PhysicsUtils:
    """Utility class for physics calculations and conversions."""
    
    # Physical constants
    R = constants.R  # Universal gas constant (J/mol/K)
    F = constants.physical_constants['Faraday constant'][0]  # Faraday constant (C/mol)
    k_B = constants.k  # Boltzmann constant (J/K)
    N_A = constants.N_A  # Avogadro's number (1/mol)
    
    @staticmethod
    def nernst_voltage(T: float, p_H2: float, p_H2O: float, p_O2: float) -> float:
        """
        Calculate Nernst voltage for SOFC.
        
        Args:
            T: Temperature (K)
            p_H2: Partial pressure of H2 (Pa)
            p_H2O: Partial pressure of H2O (Pa)
            p_O2: Partial pressure of O2 (Pa)
        
        Returns:
            Nernst voltage (V)
        """
        return (PhysicsUtils.R * T / (2 * PhysicsUtils.F)) * np.log(p_H2 * np.sqrt(p_O2) / p_H2O)
    
    @staticmethod
    def butler_volmer_current_density(eta: float, T: float, i_0: float, alpha: float = 0.5) -> float:
        """
        Calculate current density using Butler-Volmer equation.
        
        Args:
            eta: Overpotential (V)
            T: Temperature (K)
            i_0: Exchange current density (A/m²)
            alpha: Transfer coefficient
        
        Returns:
            Current density (A/m²)
        """
        return i_0 * (np.exp(alpha * PhysicsUtils.F * eta / (PhysicsUtils.R * T)) - 
                     np.exp(-(1 - alpha) * PhysicsUtils.F * eta / (PhysicsUtils.R * T)))
    
    @staticmethod
    def activation_overpotential(i: float, T: float, i_0: float, alpha: float = 0.5) -> float:
        """
        Calculate activation overpotential.
        
        Args:
            i: Current density (A/m²)
            T: Temperature (K)
            i_0: Exchange current density (A/m²)
            alpha: Transfer coefficient
        
        Returns:
            Activation overpotential (V)
        """
        return (PhysicsUtils.R * T / (alpha * PhysicsUtils.F)) * np.arcsinh(i / (2 * i_0))
    
    @staticmethod
    def ohmic_overpotential(i: float, R_ohmic: float) -> float:
        """
        Calculate ohmic overpotential.
        
        Args:
            i: Current density (A/m²)
            R_ohmic: Ohmic resistance (Ω·m²)
        
        Returns:
            Ohmic overpotential (V)
        """
        return i * R_ohmic
    
    @staticmethod
    def concentration_overpotential(i: float, T: float, i_L: float) -> float:
        """
        Calculate concentration overpotential.
        
        Args:
            i: Current density (A/m²)
            T: Temperature (K)
            i_L: Limiting current density (A/m²)
        
        Returns:
            Concentration overpotential (V)
        """
        return (PhysicsUtils.R * T / PhysicsUtils.F) * np.log(1 - i / i_L)
    
    @staticmethod
    def thermal_expansion_strain(T: float, T_ref: float, alpha: float) -> float:
        """
        Calculate thermal expansion strain.
        
        Args:
            T: Current temperature (K)
            T_ref: Reference temperature (K)
            alpha: Thermal expansion coefficient (1/K)
        
        Returns:
            Thermal expansion strain
        """
        return alpha * (T - T_ref)
    
    @staticmethod
    def thermal_stress(E: float, nu: float, alpha: float, delta_T: float) -> float:
        """
        Calculate thermal stress.
        
        Args:
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            alpha: Thermal expansion coefficient (1/K)
            delta_T: Temperature difference (K)
        
        Returns:
            Thermal stress (Pa)
        """
        return E * alpha * delta_T / (1 - nu)
    
    @staticmethod
    def heat_generation_rate(i: float, V_cell: float, V_ocv: float) -> float:
        """
        Calculate heat generation rate.
        
        Args:
            i: Current density (A/m²)
            V_cell: Cell voltage (V)
            V_ocv: Open circuit voltage (V)
        
        Returns:
            Heat generation rate (W/m²)
        """
        return i * (V_ocv - V_cell)
    
    @staticmethod
    def fuel_utilization(n_fuel_in: float, n_fuel_out: float) -> float:
        """
        Calculate fuel utilization.
        
        Args:
            n_fuel_in: Fuel flow rate in (mol/s)
            n_fuel_out: Fuel flow rate out (mol/s)
        
        Returns:
            Fuel utilization (fraction)
        """
        return (n_fuel_in - n_fuel_out) / n_fuel_in if n_fuel_in > 0 else 0
    
    @staticmethod
    def efficiency(V_cell: float, V_ocv: float) -> float:
        """
        Calculate cell efficiency.
        
        Args:
            V_cell: Cell voltage (V)
            V_ocv: Open circuit voltage (V)
        
        Returns:
            Cell efficiency (fraction)
        """
        return V_cell / V_ocv if V_ocv > 0 else 0
    
    @staticmethod
    def arrhenius_rate(k_0: float, E_a: float, T: float) -> float:
        """
        Calculate reaction rate using Arrhenius equation.
        
        Args:
            k_0: Pre-exponential factor
            E_a: Activation energy (J/mol)
            T: Temperature (K)
        
        Returns:
            Reaction rate
        """
        return k_0 * np.exp(-E_a / (PhysicsUtils.R * T))
    
    @staticmethod
    def diffusion_coefficient(D_0: float, E_a: float, T: float) -> float:
        """
        Calculate diffusion coefficient.
        
        Args:
            D_0: Pre-exponential factor (m²/s)
            E_a: Activation energy (J/mol)
            T: Temperature (K)
        
        Returns:
            Diffusion coefficient (m²/s)
        """
        return D_0 * np.exp(-E_a / (PhysicsUtils.R * T))
    
    @staticmethod
    def electrical_conductivity(sigma_0: float, E_a: float, T: float) -> float:
        """
        Calculate electrical conductivity.
        
        Args:
            sigma_0: Pre-exponential factor (S/m)
            E_a: Activation energy (J/mol)
            T: Temperature (K)
        
        Returns:
            Electrical conductivity (S/m)
        """
        return sigma_0 * np.exp(-E_a / (PhysicsUtils.R * T))
    
    @staticmethod
    def thermal_conductivity(k_0: float, T: float, n: float = -0.5) -> float:
        """
        Calculate thermal conductivity.
        
        Args:
            k_0: Reference thermal conductivity (W/m/K)
            T: Temperature (K)
            n: Temperature exponent
        
        Returns:
            Thermal conductivity (W/m/K)
        """
        return k_0 * (T / 298.15) ** n
    
    @staticmethod
    def youngs_modulus(E_0: float, T: float, T_ref: float = 298.15) -> float:
        """
        Calculate temperature-dependent Young's modulus.
        
        Args:
            E_0: Reference Young's modulus (Pa)
            T: Temperature (K)
            T_ref: Reference temperature (K)
        
        Returns:
            Young's modulus (Pa)
        """
        return E_0 * (1 - 0.1 * (T - T_ref) / T_ref)
    
    @staticmethod
    def poisson_ratio(nu_0: float, T: float, T_ref: float = 298.15) -> float:
        """
        Calculate temperature-dependent Poisson's ratio.
        
        Args:
            nu_0: Reference Poisson's ratio
            T: Temperature (K)
            T_ref: Reference temperature (K)
        
        Returns:
            Poisson's ratio
        """
        return nu_0 * (1 + 0.05 * (T - T_ref) / T_ref)
    
    @staticmethod
    def creep_strain_rate(A: float, n: float, sigma: float, T: float, Q: float) -> float:
        """
        Calculate creep strain rate.
        
        Args:
            A: Material constant
            n: Stress exponent
            sigma: Stress (Pa)
            T: Temperature (K)
            Q: Activation energy (J/mol)
        
        Returns:
            Creep strain rate (1/s)
        """
        return A * (sigma ** n) * np.exp(-Q / (PhysicsUtils.R * T))
    
    @staticmethod
    def fatigue_life(N_f: float, sigma_a: float, sigma_f: float, b: float) -> float:
        """
        Calculate fatigue life using Basquin equation.
        
        Args:
            N_f: Number of cycles to failure
            sigma_a: Stress amplitude (Pa)
            sigma_f: Fatigue strength coefficient (Pa)
            b: Fatigue strength exponent
        
        Returns:
            Fatigue life (cycles)
        """
        return N_f * (sigma_a / sigma_f) ** (1 / b)
    
    @staticmethod
    def crack_growth_rate(da_dN: float, Delta_K: float, C: float, m: float) -> float:
        """
        Calculate crack growth rate using Paris equation.
        
        Args:
            da_dN: Crack growth rate (m/cycle)
            Delta_K: Stress intensity factor range (Pa·m^0.5)
            C: Material constant
            m: Paris exponent
        
        Returns:
            Crack growth rate (m/cycle)
        """
        return C * (Delta_K ** m)
    
    @staticmethod
    def weibull_failure_probability(t: float, eta: float, beta: float) -> float:
        """
        Calculate Weibull failure probability.
        
        Args:
            t: Time (hours)
            eta: Scale parameter (hours)
            beta: Shape parameter
        
        Returns:
            Failure probability (fraction)
        """
        return 1 - np.exp(-(t / eta) ** beta)
    
    @staticmethod
    def remaining_useful_life(t: float, eta: float, beta: float) -> float:
        """
        Calculate remaining useful life.
        
        Args:
            t: Current time (hours)
            eta: Scale parameter (hours)
            beta: Shape parameter
        
        Returns:
            Remaining useful life (hours)
        """
        return eta * (1 - (t / eta) ** beta) ** (1 / beta)
    
    @staticmethod
    def calculate_effective_properties(microstructure: np.ndarray, 
                                    solid_properties: Dict[str, float],
                                    pore_properties: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate effective properties from microstructure.
        
        Args:
            microstructure: 3D binary microstructure (0=solid, 1=pore)
            solid_properties: Properties of solid phase
            pore_properties: Properties of pore phase
        
        Returns:
            Effective properties
        """
        porosity = np.sum(microstructure) / microstructure.size
        
        # Bruggeman effective medium theory
        effective_properties = {}
        
        for prop in solid_properties:
            if prop in pore_properties:
                # Bruggeman equation
                phi = porosity
                k_s = solid_properties[prop]
                k_p = pore_properties[prop]
                
                # Solve: phi * (k_p - k_eff) / (k_p + 2*k_eff) + (1-phi) * (k_s - k_eff) / (k_s + 2*k_eff) = 0
                # Simplified for small porosity
                k_eff = k_s * (1 - 1.5 * phi) if phi < 0.3 else k_s * (1 - phi) ** 1.5
                effective_properties[prop] = k_eff
        
        return effective_properties
    
    @staticmethod
    def calculate_tortuosity(microstructure: np.ndarray) -> float:
        """
        Calculate tortuosity from microstructure.
        
        Args:
            microstructure: 3D binary microstructure (0=solid, 1=pore)
        
        Returns:
            Tortuosity
        """
        # Simplified tortuosity calculation
        porosity = np.sum(microstructure) / microstructure.size
        
        # Archie's law
        tortuosity = 1.0 + 0.5 * (1 - porosity) / porosity if porosity > 0 else 1.0
        
        return tortuosity
    
    @staticmethod
    def calculate_specific_surface_area(microstructure: np.ndarray, voxel_size: float) -> float:
        """
        Calculate specific surface area from microstructure.
        
        Args:
            microstructure: 3D binary microstructure (0=solid, 1=pore)
            voxel_size: Size of each voxel (m)
        
        Returns:
            Specific surface area (m²/m³)
        """
        # Count surface voxels (solid-pore interfaces)
        surface_voxels = 0
        for i in range(1, microstructure.shape[0]-1):
            for j in range(1, microstructure.shape[1]-1):
                for k in range(1, microstructure.shape[2]-1):
                    if microstructure[i,j,k] == 0:  # Solid voxel
                        # Check if any neighbor is pore
                        neighbors = microstructure[i-1:i+2, j-1:j+2, k-1:k+2]
                        if np.any(neighbors == 1):
                            surface_voxels += 1
        
        voxel_volume = voxel_size ** 3
        total_volume = microstructure.size * voxel_volume
        specific_surface_area = surface_voxels * (voxel_size ** 2) / total_volume
        
        return specific_surface_area
    
    @staticmethod
    def convert_temperature(T: float, from_unit: str, to_unit: str) -> float:
        """
        Convert temperature between units.
        
        Args:
            T: Temperature value
            from_unit: Source unit ('C', 'K', 'F')
            to_unit: Target unit ('C', 'K', 'F')
        
        Returns:
            Converted temperature
        """
        # Convert to Kelvin first
        if from_unit == 'C':
            T_K = T + 273.15
        elif from_unit == 'F':
            T_K = (T - 32) * 5/9 + 273.15
        elif from_unit == 'K':
            T_K = T
        else:
            raise ValueError(f"Unknown temperature unit: {from_unit}")
        
        # Convert from Kelvin to target unit
        if to_unit == 'C':
            return T_K - 273.15
        elif to_unit == 'F':
            return (T_K - 273.15) * 9/5 + 32
        elif to_unit == 'K':
            return T_K
        else:
            raise ValueError(f"Unknown temperature unit: {to_unit}")
    
    @staticmethod
    def convert_pressure(p: float, from_unit: str, to_unit: str) -> float:
        """
        Convert pressure between units.
        
        Args:
            p: Pressure value
            from_unit: Source unit ('Pa', 'bar', 'atm', 'psi')
            to_unit: Target unit ('Pa', 'bar', 'atm', 'psi')
        
        Returns:
            Converted pressure
        """
        # Convert to Pa first
        if from_unit == 'Pa':
            p_Pa = p
        elif from_unit == 'bar':
            p_Pa = p * 1e5
        elif from_unit == 'atm':
            p_Pa = p * 101325
        elif from_unit == 'psi':
            p_Pa = p * 6894.76
        else:
            raise ValueError(f"Unknown pressure unit: {from_unit}")
        
        # Convert from Pa to target unit
        if to_unit == 'Pa':
            return p_Pa
        elif to_unit == 'bar':
            return p_Pa / 1e5
        elif to_unit == 'atm':
            return p_Pa / 101325
        elif to_unit == 'psi':
            return p_Pa / 6894.76
        else:
            raise ValueError(f"Unknown pressure unit: {to_unit}")
    
    @staticmethod
    def validate_physical_quantities(data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Validate physical quantities for reasonableness.
        
        Args:
            data: Dictionary containing physical quantities
        
        Returns:
            Dictionary of validation warnings
        """
        warnings_dict = {}
        
        for key, value in data.items():
            warnings_list = []
            
            if isinstance(value, (int, float, np.number)):
                # Temperature validation
                if 'temp' in key.lower() or 'temperature' in key.lower():
                    if value < 200 or value > 1500:  # Reasonable temperature range
                        warnings_list.append(f"Temperature {value} seems unreasonable")
                
                # Pressure validation
                elif 'pressure' in key.lower() or 'p_' in key:
                    if value < 0 or value > 1e7:  # Reasonable pressure range
                        warnings_list.append(f"Pressure {value} seems unreasonable")
                
                # Voltage validation
                elif 'voltage' in key.lower() or 'V_' in key:
                    if value < 0 or value > 2:  # Reasonable voltage range
                        warnings_list.append(f"Voltage {value} seems unreasonable")
                
                # Current density validation
                elif 'current' in key.lower() or 'i_' in key:
                    if value < 0 or value > 1e6:  # Reasonable current density range
                        warnings_list.append(f"Current density {value} seems unreasonable")
            
            if warnings_list:
                warnings_dict[key] = warnings_list
        
        return warnings_dict