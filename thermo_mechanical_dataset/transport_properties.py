"""
Transport and poro-mechanical property generation module
Provides temperature-dependent transport properties and pore pressure evolution
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

class TransportPropertyGenerator:
    """
    Generator for transport and poro-mechanical properties
    Includes permeability, porosity, moisture transport, and pore pressure
    """
    
    def __init__(self, microstructure_params: Dict):
        """
        Initialize transport property generator
        
        Parameters:
        -----------
        microstructure_params : Dict
            Microstructural parameters for each mix
        """
        self.microstructure = microstructure_params
        
    def generate(self, mix_id: str, data_type: str) -> Dict:
        """
        Generate complete transport property dataset for a mix
        
        Parameters:
        -----------
        mix_id : str
            Mix identification
        data_type : str
            'Calibration' or 'Validation'
        
        Returns:
        --------
        Dict : Transport properties with temperature dependence
        """
        micro = self.microstructure[mix_id]
        rubber_content = micro['rubber_content_percent']
        initial_porosity = micro['initial_porosity']
        
        # Temperature array
        temps = np.linspace(20, 800, 40)
        
        # Generate porosity evolution
        porosity = {
            'temperature': temps.tolist(),
            'total_porosity': self._total_porosity(temps, rubber_content, initial_porosity),
            'connected_porosity': self._connected_porosity(temps, rubber_content, initial_porosity),
            'capillary_porosity': self._capillary_porosity(temps, rubber_content, initial_porosity),
            'gel_porosity': self._gel_porosity(temps, rubber_content, initial_porosity),
            'crack_porosity': self._crack_porosity(temps, rubber_content),
        }
        
        # Generate permeability
        permeability = {
            'temperature': temps.tolist(),
            'intrinsic_permeability': self._intrinsic_permeability(temps, rubber_content, initial_porosity),
            'gas_permeability': self._gas_permeability(temps, rubber_content),
            'liquid_permeability': self._liquid_permeability(temps, rubber_content),
            'relative_permeability_gas': self._relative_permeability_gas(temps),
            'relative_permeability_liquid': self._relative_permeability_liquid(temps),
        }
        
        # Generate moisture transport
        moisture = {
            'temperature': temps.tolist(),
            'moisture_content': self._moisture_content(temps, rubber_content),
            'moisture_diffusivity': self._moisture_diffusivity(temps, rubber_content),
            'vapor_diffusivity': self._vapor_diffusivity(temps, rubber_content),
            'sorption_isotherm': self._sorption_isotherm(rubber_content),
            'desorption_rate': self._desorption_rate(temps, rubber_content),
        }
        
        # Generate pore pressure
        pore_pressure = {
            'temperature': temps.tolist(),
            'vapor_pressure': self._vapor_pressure(temps),
            'capillary_pressure': self._capillary_pressure(temps, rubber_content),
            'gas_pressure_buildup': self._gas_pressure_buildup(temps, rubber_content),
            'effective_stress_coefficient': self._effective_stress_coefficient(temps, rubber_content),
        }
        
        # Generate mass transport
        mass_transport = {
            'temperature': temps.tolist(),
            'chloride_diffusivity': self._chloride_diffusivity(temps, rubber_content),
            'carbonation_coefficient': self._carbonation_coefficient(temps, rubber_content),
            'oxygen_diffusivity': self._oxygen_diffusivity(temps, rubber_content),
            'tortuosity': self._tortuosity(rubber_content, initial_porosity),
        }
        
        # Generate thermal spalling risk
        spalling = {
            'temperature': temps.tolist(),
            'spalling_risk_index': self._spalling_risk(temps, rubber_content),
            'critical_pore_pressure': self._critical_pore_pressure(temps, rubber_content),
            'moisture_clog_depth': self._moisture_clog_depth(temps, rubber_content),
        }
        
        # Combine all properties
        properties = {
            'porosity': porosity,
            'permeability': permeability,
            'moisture': moisture,
            'pore_pressure': pore_pressure,
            'mass_transport': mass_transport,
            'spalling': spalling,
            'poro_mechanical_coupling': self._poro_mechanical_coupling(rubber_content)
        }
        
        # Add stochastic variation for validation set
        if data_type == 'Validation':
            properties = self._add_stochastic_variation(properties)
        
        return properties
    
    def _total_porosity(self, T: np.ndarray, rubber: float, initial: float) -> List[float]:
        """
        Temperature-dependent total porosity evolution
        Fraction (0-1)
        """
        # Initial porosity increases with rubber content
        phi_0 = initial
        
        # Porosity evolution with temperature
        phi = phi_0 * np.ones_like(T)
        
        # Increase due to dehydration and decomposition
        mask_100_200 = (T >= 100) & (T < 200)
        mask_200_400 = (T >= 200) & (T < 400)
        mask_400_600 = (T >= 400) & (T < 600)
        mask_600 = T >= 600
        
        # Moisture loss creates porosity
        phi[mask_100_200] = phi_0 * (1 + 0.1 * (T[mask_100_200] - 100) / 100)
        
        # Dehydration
        phi[mask_200_400] = phi_0 * (1.1 + 0.3 * (T[mask_200_400] - 200) / 200)
        
        # Decomposition and cracking
        phi[mask_400_600] = phi_0 * (1.4 + 0.5 * (T[mask_400_600] - 400) / 200)
        
        # Significant damage
        phi[mask_600] = phi_0 * (1.9 + 0.3 * (T[mask_600] - 600) / 200)
        
        # Rubber decomposition adds porosity
        if rubber > 0:
            rubber_decomp = self._rubber_decomposition_porosity(T, rubber)
            phi = phi + rubber_decomp
        
        # Cap at reasonable maximum
        phi = np.minimum(phi, 0.6)
        
        return phi.tolist()
    
    def _connected_porosity(self, T: np.ndarray, rubber: float, initial: float) -> List[float]:
        """
        Connected (effective) porosity
        Fraction (0-1)
        """
        total = np.array(self._total_porosity(T, rubber, initial))
        
        # Connectivity factor increases with temperature due to cracking
        connectivity = 0.3 + 0.0005 * T
        connectivity = np.minimum(connectivity, 0.9)
        
        # Rubber slightly reduces connectivity at low temps but helps at high temps
        if rubber > 0:
            rubber_effect = 1 - 0.01 * rubber  # Reduces connectivity
            mask_high = T > 400
            rubber_effect_high = 1 + 0.02 * rubber  # Improves connectivity
            connectivity = connectivity * rubber_effect
            connectivity[mask_high] = connectivity[mask_high] * rubber_effect_high / rubber_effect
        
        connected = total * connectivity
        
        return connected.tolist()
    
    def _capillary_porosity(self, T: np.ndarray, rubber: float, initial: float) -> List[float]:
        """
        Capillary pore volume fraction
        Fraction (0-1)
        """
        total = np.array(self._total_porosity(T, rubber, initial))
        
        # Capillary porosity fraction of total
        cap_fraction = 0.7 * np.ones_like(T)
        
        # Decreases with temperature as pores coarsen
        mask_200 = T > 200
        cap_fraction[mask_200] = 0.7 - 0.0005 * (T[mask_200] - 200)
        cap_fraction = np.maximum(cap_fraction, 0.3)
        
        capillary = total * cap_fraction
        
        return capillary.tolist()
    
    def _gel_porosity(self, T: np.ndarray, rubber: float, initial: float) -> List[float]:
        """
        Gel pore volume fraction
        Fraction (0-1)
        """
        # Gel porosity independent of total porosity
        gel_0 = 0.01  # Initial gel porosity
        
        gel = gel_0 * np.ones_like(T)
        
        # Gel pores collapse with temperature
        mask_100 = T > 100
        gel[mask_100] = gel_0 * np.exp(-0.005 * (T[mask_100] - 100))
        
        return gel.tolist()
    
    def _crack_porosity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Crack-induced porosity
        Fraction (0-1)
        """
        # No cracks initially
        cracks = np.zeros_like(T)
        
        # Thermal cracking starts around 300°C
        mask_300_500 = (T >= 300) & (T < 500)
        mask_500 = T >= 500
        
        cracks[mask_300_500] = 0.001 * (T[mask_300_500] - 300) / 200
        cracks[mask_500] = 0.001 + 0.002 * (T[mask_500] - 500) / 300
        
        # Rubber delays cracking
        if rubber > 0:
            delay_factor = 1 - 0.3 * rubber / 20  # Up to 30% delay
            cracks = cracks * delay_factor
        
        return cracks.tolist()
    
    def _rubber_decomposition_porosity(self, T: np.ndarray, rubber: float) -> np.ndarray:
        """
        Additional porosity from rubber decomposition
        """
        # Rubber decomposes between 250-500°C
        decomp = np.zeros_like(T)
        
        mask_250_500 = (T >= 250) & (T <= 500)
        mask_500 = T > 500
        
        # Sigmoid decomposition
        decomp[mask_250_500] = (rubber / 100) * 0.1 * (1 / (1 + np.exp(-0.02 * (T[mask_250_500] - 375))))
        decomp[mask_500] = (rubber / 100) * 0.1
        
        return decomp
    
    def _intrinsic_permeability(self, T: np.ndarray, rubber: float, initial_porosity: float) -> List[float]:
        """
        Intrinsic permeability
        m²
        """
        # Kozeny-Carman relation: k = phi³ / (c * S²)
        phi = np.array(self._connected_porosity(T, rubber, initial_porosity))
        
        # Specific surface area effect
        S_0 = 1e4  # m²/m³
        c = 180  # Kozeny constant
        
        # Permeability calculation
        k = (phi ** 3) / (c * S_0 ** 2 * (1 - phi) ** 2)
        
        # Temperature effect on pore structure
        temp_factor = np.ones_like(T)
        mask_300 = T > 300
        temp_factor[mask_300] = 1 + 0.01 * (T[mask_300] - 300)  # Coarsening increases permeability
        
        k = k * temp_factor
        
        # Rubber effect
        if rubber > 0:
            # Rubber initially reduces permeability but decomposition increases it
            rubber_factor = np.ones_like(T)
            mask_low = T < 250
            mask_decomp = T >= 250
            
            rubber_factor[mask_low] = 1 - 0.3 * rubber / 20
            rubber_factor[mask_decomp] = 1 + 0.5 * rubber / 20
            
            k = k * rubber_factor
        
        # Convert to reasonable range (1e-20 to 1e-15 m²)
        k = np.maximum(k, 1e-20)
        k = np.minimum(k, 1e-15)
        
        return (k * 1e18).tolist()  # Convert to 10^-18 m² for readability
    
    def _gas_permeability(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Gas permeability (Klinkenberg effect included)
        m²
        """
        # Intrinsic permeability
        k_int = np.array(self._intrinsic_permeability(T, rubber, 
                                                      self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity'])) * 1e-18
        
        # Klinkenberg factor
        b = 1e5  # Pa
        P_mean = 1e5  # Mean pressure, Pa
        
        # Gas permeability with slip flow
        k_gas = k_int * (1 + b / P_mean)
        
        # Temperature effect on gas viscosity
        mu_ratio = np.sqrt(T / 293)  # Viscosity ratio
        k_gas = k_gas * mu_ratio
        
        return (k_gas * 1e18).tolist()
    
    def _liquid_permeability(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Liquid water permeability
        m²
        """
        # Similar to intrinsic but affected by saturation
        k_int = np.array(self._intrinsic_permeability(T, rubber,
                                                      self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity'])) * 1e-18
        
        # Saturation effect
        S_w = np.array(self._moisture_content(T, rubber)) / 100  # Convert to fraction
        
        # Relative permeability
        k_rel = S_w ** 3  # Cubic law
        
        k_liquid = k_int * k_rel
        
        return (k_liquid * 1e18).tolist()
    
    def _relative_permeability_gas(self, T: np.ndarray) -> List[float]:
        """
        Relative permeability for gas phase
        Dimensionless (0-1)
        """
        # Depends on liquid saturation
        S_w = np.array(self._moisture_content(T, 0)) / 100  # Base saturation
        
        # Gas relative permeability (Corey model)
        S_gr = 0.05  # Residual gas saturation
        k_rg = ((1 - S_w - S_gr) / (1 - S_gr)) ** 2
        k_rg = np.maximum(k_rg, 0)
        k_rg = np.minimum(k_rg, 1)
        
        return k_rg.tolist()
    
    def _relative_permeability_liquid(self, T: np.ndarray) -> List[float]:
        """
        Relative permeability for liquid phase
        Dimensionless (0-1)
        """
        # Depends on liquid saturation
        S_w = np.array(self._moisture_content(T, 0)) / 100  # Base saturation
        
        # Liquid relative permeability (Corey model)
        S_wr = 0.1  # Residual water saturation
        k_rw = ((S_w - S_wr) / (1 - S_wr)) ** 3
        k_rw = np.maximum(k_rw, 0)
        k_rw = np.minimum(k_rw, 1)
        
        return k_rw.tolist()
    
    def _moisture_content(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Moisture content by weight
        Percentage
        """
        # Initial moisture content
        w_0 = 4.0  # 4% by weight
        
        # Temperature-dependent evaporation
        w = w_0 * np.ones_like(T)
        
        # Free water evaporation (20-105°C)
        mask_20_105 = T <= 105
        mask_105_200 = (T > 105) & (T <= 200)
        mask_200_400 = (T > 200) & (T <= 400)
        mask_400 = T > 400
        
        w[mask_20_105] = w_0 * (1 - 0.7 * (T[mask_20_105] - 20) / 85)
        w[mask_105_200] = 0.3 * w_0 * (1 - 0.8 * (T[mask_105_200] - 105) / 95)
        w[mask_200_400] = 0.06 * w_0 * (1 - 0.9 * (T[mask_200_400] - 200) / 200)
        w[mask_400] = 0.006 * w_0
        
        # Rubber reduces initial moisture but retains more at high temp
        if rubber > 0:
            w = w * (1 - 0.01 * rubber)  # Less initial moisture
            mask_high = T > 200
            w[mask_high] = w[mask_high] * (1 + 0.1 * rubber / 20)  # Better retention
        
        return w.tolist()
    
    def _moisture_diffusivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Moisture diffusivity
        m²/s
        """
        # Base diffusivity at 20°C
        D_0 = 1e-9  # m²/s
        
        # Arrhenius temperature dependence
        E_a = 25000  # J/mol
        R = 8.314  # J/(mol·K)
        T_kelvin = T + 273.15
        
        D = D_0 * np.exp(E_a / R * (1/293 - 1/T_kelvin))
        
        # Increase significantly near 100°C due to vapor transport
        mask_80_120 = (T >= 80) & (T <= 120)
        D[mask_80_120] = D[mask_80_120] * (1 + 10 * np.exp(-((T[mask_80_120] - 100) / 20) ** 2))
        
        # Rubber effect
        if rubber > 0:
            D = D * (1 - 0.2 * rubber / 20)  # Reduces diffusivity
        
        return (D * 1e6).tolist()  # Convert to mm²/s
    
    def _vapor_diffusivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Water vapor diffusivity in air
        m²/s
        """
        # Temperature-dependent vapor diffusivity
        D_v0 = 2.5e-5  # m²/s at 25°C
        
        # Temperature dependence (T^1.75 law)
        T_kelvin = T + 273.15
        D_v = D_v0 * (T_kelvin / 298) ** 1.75
        
        # Porosity and tortuosity effects
        phi = np.array(self._connected_porosity(T, rubber, 
                                               self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity']))
        tau = 3.0  # Tortuosity
        
        D_eff = D_v * phi / tau
        
        return (D_eff * 1e6).tolist()  # Convert to mm²/s
    
    def _sorption_isotherm(self, rubber: float) -> Dict:
        """
        Sorption isotherm parameters (BET model)
        """
        # BET parameters
        w_m = 0.05 - 0.001 * rubber  # Monolayer water content
        C = 50 - rubber  # BET constant
        
        # Generate isotherm curve
        RH = np.linspace(0, 1, 21)  # Relative humidity
        
        # BET equation
        w = w_m * C * RH / ((1 - RH) * (1 - RH + C * RH))
        
        return {
            'relative_humidity': RH.tolist(),
            'moisture_content': w.tolist(),
            'monolayer_content': w_m,
            'BET_constant': C,
            'hysteresis_factor': 0.15  # Desorption vs adsorption
        }
    
    def _desorption_rate(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Moisture desorption rate constant
        1/s
        """
        # Arrhenius kinetics
        k_0 = 1e-3  # 1/s
        E_a = 40000  # J/mol
        R = 8.314  # J/(mol·K)
        T_kelvin = T + 273.15
        
        k = k_0 * np.exp(-E_a / (R * T_kelvin))
        
        # Rubber slows desorption
        if rubber > 0:
            k = k * (1 - 0.2 * rubber / 20)
        
        return k.tolist()
    
    def _vapor_pressure(self, T: np.ndarray) -> List[float]:
        """
        Saturated vapor pressure
        Pa
        """
        # Clausius-Clapeyron equation
        T_kelvin = T + 273.15
        
        # Antoine equation for water
        A = 8.07131
        B = 1730.63
        C = 233.426
        
        # Pressure in mmHg
        P_mmHg = 10 ** (A - B / (C + T))
        
        # Convert to Pa
        P_Pa = P_mmHg * 133.322
        
        return P_Pa.tolist()
    
    def _capillary_pressure(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Capillary pressure
        Pa
        """
        # Young-Laplace equation: P_c = 2*sigma*cos(theta)/r
        
        # Surface tension of water (temperature dependent)
        sigma = 0.0728 * (1 - 0.0002 * T)  # N/m
        
        # Contact angle (assumed)
        theta = np.pi / 6  # 30 degrees
        
        # Pore radius (temperature and rubber dependent)
        r_0 = 1e-7  # m (100 nm)
        r = r_0 * (1 + 0.001 * T)  # Pores expand with temperature
        
        if rubber > 0:
            r = r * (1 + 0.1 * rubber / 20)  # Rubber creates larger pores
        
        P_c = 2 * sigma * np.cos(theta) / r
        
        return (P_c / 1e6).tolist()  # Convert to MPa
    
    def _gas_pressure_buildup(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Gas pressure buildup in pores
        Pa
        """
        # Ideal gas law for vapor pressure buildup
        P_vapor = np.array(self._vapor_pressure(T))
        
        # Additional pressure from decomposition gases
        P_decomp = np.zeros_like(T)
        
        mask_200_400 = (T >= 200) & (T < 400)
        mask_400 = T >= 400
        
        P_decomp[mask_200_400] = 1e5 * (T[mask_200_400] - 200) / 200
        P_decomp[mask_400] = 1e5 + 2e5 * (T[mask_400] - 400) / 400
        
        # Rubber decomposition adds gas pressure
        if rubber > 0:
            mask_rubber = (T >= 250) & (T <= 500)
            P_decomp[mask_rubber] = P_decomp[mask_rubber] + 0.5e5 * rubber / 20
        
        P_total = P_vapor + P_decomp
        
        return (P_total / 1e6).tolist()  # Convert to MPa
    
    def _effective_stress_coefficient(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Biot effective stress coefficient
        Dimensionless (0-1)
        """
        # Related to porosity and stiffness
        phi = np.array(self._total_porosity(T, rubber, 
                                           self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity']))
        
        # Biot coefficient
        alpha = 0.6 + 0.3 * phi
        
        # Temperature effect
        mask_300 = T > 300
        alpha[mask_300] = alpha[mask_300] * (1 + 0.0001 * (T[mask_300] - 300))
        
        # Cap at 1
        alpha = np.minimum(alpha, 1.0)
        
        return alpha.tolist()
    
    def _chloride_diffusivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Chloride ion diffusivity
        m²/s
        """
        # Base diffusivity
        D_0 = 1e-12  # m²/s
        
        # Temperature dependence
        E_a = 35000  # J/mol
        R = 8.314  # J/(mol·K)
        T_kelvin = T + 273.15
        
        D = D_0 * np.exp(E_a / R * (1/293 - 1/T_kelvin))
        
        # Porosity effect
        phi = np.array(self._connected_porosity(T, rubber,
                                               self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity']))
        D = D * phi
        
        # Rubber reduces diffusivity at low temps
        if rubber > 0:
            mask_low = T < 200
            D[mask_low] = D[mask_low] * (1 - 0.3 * rubber / 20)
        
        return (D * 1e12).tolist()  # Convert to 10^-12 m²/s
    
    def _carbonation_coefficient(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Carbonation rate coefficient
        mm/√year
        """
        # Base coefficient
        k_0 = 3.0  # mm/√year
        
        # Temperature effect (optimal around 20-40°C)
        k = k_0 * np.ones_like(T)
        
        mask_40_100 = (T > 40) & (T <= 100)
        mask_100 = T > 100
        
        k[mask_40_100] = k_0 * (1 - 0.005 * (T[mask_40_100] - 40))
        k[mask_100] = k_0 * 0.7 * np.exp(-0.01 * (T[mask_100] - 100))
        
        # Rubber effect
        if rubber > 0:
            k = k * (1 + 0.1 * rubber / 20)  # Increases carbonation
        
        return k.tolist()
    
    def _oxygen_diffusivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Oxygen diffusivity
        m²/s
        """
        # Similar to gas diffusivity
        D_O2_air = 2e-5  # m²/s in air at 20°C
        
        # Temperature dependence
        T_kelvin = T + 273.15
        D_O2 = D_O2_air * (T_kelvin / 293) ** 1.75
        
        # Porosity effect
        phi = np.array(self._connected_porosity(T, rubber,
                                               self.microstructure[f'R{int(rubber)}S' if rubber > 0 else 'C']['initial_porosity']))
        tau = 3.0
        
        D_eff = D_O2 * phi / tau
        
        return (D_eff * 1e6).tolist()  # Convert to mm²/s
    
    def _tortuosity(self, rubber: float, initial_porosity: float) -> float:
        """
        Pore network tortuosity
        Dimensionless
        """
        # Base tortuosity
        tau = 3.0
        
        # Rubber effect
        if rubber > 0:
            tau = tau * (1 + 0.1 * rubber / 20)
        
        # Porosity effect (lower porosity = higher tortuosity)
        tau = tau / (initial_porosity * 10)
        
        return tau
    
    def _spalling_risk(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Thermal spalling risk index
        Dimensionless (0-1)
        """
        # Based on pore pressure and tensile strength
        P_pore = np.array(self._gas_pressure_buildup(T, rubber))  # MPa
        
        # Approximate tensile strength (degrades with temperature)
        f_t = 4.0 * np.ones_like(T)  # MPa
        mask_200 = T > 200
        f_t[mask_200] = 4.0 * np.exp(-0.003 * (T[mask_200] - 200))
        
        # Risk index
        risk = P_pore / (f_t + 0.1)  # Add small value to avoid division by zero
        
        # Rubber reduces spalling risk
        if rubber > 0:
            risk = risk * (1 - 0.3 * rubber / 20)
        
        # Normalize to 0-1
        risk = np.minimum(risk, 1.0)
        
        return risk.tolist()
    
    def _critical_pore_pressure(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Critical pore pressure for spalling
        MPa
        """
        # Related to tensile strength
        f_t = 4.0 * np.ones_like(T)  # MPa
        mask_200 = T > 200
        f_t[mask_200] = 4.0 * np.exp(-0.003 * (T[mask_200] - 200))
        
        # Critical pressure is fraction of tensile strength
        P_crit = 0.7 * f_t
        
        # Rubber increases critical pressure (more ductile)
        if rubber > 0:
            P_crit = P_crit * (1 + 0.2 * rubber / 20)
        
        return P_crit.tolist()
    
    def _moisture_clog_depth(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Moisture clog formation depth
        mm
        """
        # Depth where moisture accumulates during heating
        depth = np.zeros_like(T)
        
        # Clog forms around 100°C isotherm
        mask_80_120 = (T >= 80) & (T <= 120)
        mask_120 = T > 120
        
        # Peak clog depth at 100°C
        depth[mask_80_120] = 10 * np.exp(-((T[mask_80_120] - 100) / 20) ** 2)
        depth[mask_120] = 10 * np.exp(-((120 - 100) / 20) ** 2) * np.exp(-0.01 * (T[mask_120] - 120))
        
        # Rubber reduces clog depth (better permeability)
        if rubber > 0:
            depth = depth * (1 - 0.2 * rubber / 20)
        
        return depth.tolist()
    
    def _poro_mechanical_coupling(self, rubber: float) -> Dict:
        """
        Poro-mechanical coupling parameters
        """
        return {
            'biot_willis_coefficient': 0.6 + 0.01 * rubber,
            'skempton_coefficient_B': 0.95 - 0.01 * rubber,
            'skempton_coefficient_A': 0.5,
            'undrained_poisson_ratio': 0.35 + 0.002 * rubber,
            'specific_storage': 1e-6,  # 1/Pa
            'hydraulic_conductivity': 1e-10 * (1 + 0.1 * rubber),  # m/s
            'threshold_gradient': 0.1,
            'coupling_type': 'fully_coupled',
            'consolidation_coefficient': 1e-7  # m²/s
        }
    
    def _add_stochastic_variation(self, properties: Dict, cv: float = 0.05) -> Dict:
        """
        Add stochastic variation to properties
        """
        varied_props = {}
        
        for key, value in properties.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                # Add normal variation
                base = np.array(value)
                std = base * cv
                varied = np.random.normal(base, std)
                # Ensure positive values
                varied = np.maximum(varied, base * 0.1)  # At least 10% of base value
                varied_props[key] = varied.tolist()
            elif isinstance(value, dict):
                # Recursively apply to nested dicts
                varied_props[key] = self._add_stochastic_variation(value, cv)
            else:
                # Keep as is
                varied_props[key] = value
        
        return varied_props