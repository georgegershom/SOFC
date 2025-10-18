"""
Thermal property generation module
Provides temperature-dependent thermal properties with physical consistency
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import interpolate

class ThermalPropertyGenerator:
    """
    Generator for temperature-dependent thermal properties
    Incorporates degradation models and rubber content effects
    """
    
    def __init__(self, microstructure_params: Dict):
        """
        Initialize thermal property generator
        
        Parameters:
        -----------
        microstructure_params : Dict
            Microstructural parameters for each mix
        """
        self.microstructure = microstructure_params
        
    def generate(self, mix_id: str, data_type: str) -> Dict:
        """
        Generate complete thermal property dataset for a mix
        
        Parameters:
        -----------
        mix_id : str
            Mix identification
        data_type : str
            'Calibration' or 'Validation'
        
        Returns:
        --------
        Dict : Thermal properties with temperature dependence
        """
        micro = self.microstructure[mix_id]
        rubber_content = micro['rubber_content_percent']
        
        # Temperature array
        temps = np.linspace(20, 800, 40)
        
        # Generate properties
        properties = {
            'temperature': temps.tolist(),
            'thermal_conductivity': self._thermal_conductivity(temps, rubber_content),
            'specific_heat_capacity': self._specific_heat(temps, rubber_content),
            'thermal_diffusivity': self._thermal_diffusivity(temps, rubber_content),
            'density': self._density(temps, rubber_content),
            'thermal_expansion': self._thermal_expansion(temps, rubber_content),
            'emissivity': self._emissivity(temps, rubber_content),
            'stefan_boltzmann': 5.67e-8,  # W/(m²·K⁴)
        }
        
        # Add degradation functions
        properties['degradation'] = {
            'conductivity_reduction': self._conductivity_degradation(temps, rubber_content),
            'density_reduction': self._density_degradation(temps, rubber_content),
            'phase_change_temps': self._phase_change_temperatures(rubber_content),
            'latent_heats': self._latent_heats(rubber_content)
        }
        
        # Add stochastic variation for validation set
        if data_type == 'Validation':
            properties = self._add_stochastic_variation(properties)
        
        return properties
    
    def _thermal_conductivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent thermal conductivity
        W/(m·K)
        """
        # Base conductivity for control concrete
        k_base = 1.95 - 0.0011 * (T - 20)
        
        # Rubber effect: reduces conductivity
        rubber_factor = 1 - 0.025 * rubber  # 2.5% reduction per 1% rubber
        
        # High temperature degradation
        degradation = np.ones_like(T)
        mask_100_400 = (T >= 100) & (T <= 400)
        mask_400_800 = T > 400
        
        degradation[mask_100_400] = 1 - 0.0005 * (T[mask_100_400] - 100)
        degradation[mask_400_800] = 0.85 - 0.0008 * (T[mask_400_800] - 400)
        
        k = k_base * rubber_factor * degradation
        
        # Ensure positive values
        k = np.maximum(k, 0.1)
        
        return k.tolist()
    
    def _specific_heat(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent specific heat capacity
        J/(kg·K)
        """
        # Base specific heat
        c_base = np.ones_like(T) * 900
        
        # Temperature effects
        mask_100_200 = (T >= 100) & (T <= 200)
        mask_200_400 = (T > 200) & (T <= 400)
        mask_400_600 = (T > 400) & (T <= 600)
        mask_600_800 = T > 600
        
        # Peak at 100-200°C due to moisture evaporation
        c_base[mask_100_200] = 900 + 1100 * np.exp(-((T[mask_100_200] - 150) / 50) ** 2)
        
        # Gradual increase
        c_base[mask_200_400] = 1000 + 0.5 * (T[mask_200_400] - 200)
        
        # Further increase
        c_base[mask_400_600] = 1100 + 0.8 * (T[mask_400_600] - 400)
        
        # Stabilization
        c_base[mask_600_800] = 1260 + 0.2 * (T[mask_600_800] - 600)
        
        # Rubber effect: increases heat capacity
        rubber_factor = 1 + 0.015 * rubber  # 1.5% increase per 1% rubber
        
        c = c_base * rubber_factor
        
        return c.tolist()
    
    def _thermal_diffusivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent thermal diffusivity
        m²/s
        """
        # Get conductivity and heat capacity
        k = np.array(self._thermal_conductivity(T, rubber))
        c = np.array(self._specific_heat(T, rubber))
        rho = np.array(self._density(T, rubber))
        
        # Thermal diffusivity: α = k / (ρ * c)
        alpha = k / (rho * c)
        
        return (alpha * 1e6).tolist()  # Convert to mm²/s for convenience
    
    def _density(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent density
        kg/m³
        """
        # Base density
        rho_base = 2400 - 12 * rubber  # Rubber reduces density
        
        # Temperature-dependent reduction
        reduction = np.ones_like(T)
        
        # Moisture loss (20-200°C)
        mask_20_200 = T <= 200
        reduction[mask_20_200] = 1 - 0.00015 * (T[mask_20_200] - 20)
        
        # Dehydration (200-600°C)
        mask_200_600 = (T > 200) & (T <= 600)
        reduction[mask_200_600] = 0.973 - 0.00025 * (T[mask_200_600] - 200)
        
        # Decomposition (>600°C)
        mask_600 = T > 600
        reduction[mask_600] = 0.873 - 0.0001 * (T[mask_600] - 600)
        
        rho = rho_base * reduction
        
        return rho.tolist()
    
    def _thermal_expansion(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent thermal expansion coefficient
        1/K
        """
        # Base coefficient
        alpha_base = 8e-6  # 1/K
        
        # Temperature dependence
        alpha = alpha_base * np.ones_like(T)
        
        # Increases with temperature
        mask_200 = T > 200
        alpha[mask_200] = alpha_base * (1 + 0.001 * (T[mask_200] - 200))
        
        # Rubber effect: increases expansion
        rubber_factor = 1 + 0.05 * rubber  # 5% increase per 1% rubber
        
        alpha = alpha * rubber_factor
        
        return (alpha * 1e6).tolist()  # Convert to μstrain/K
    
    def _emissivity(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent surface emissivity
        Dimensionless
        """
        # Base emissivity
        epsilon = 0.85 * np.ones_like(T)
        
        # Increases with temperature due to surface changes
        mask_300 = T > 300
        epsilon[mask_300] = 0.85 + 0.0001 * (T[mask_300] - 300)
        
        # Rubber slightly increases emissivity
        rubber_factor = 1 + 0.002 * rubber
        
        epsilon = epsilon * rubber_factor
        
        # Cap at 0.95
        epsilon = np.minimum(epsilon, 0.95)
        
        return epsilon.tolist()
    
    def _conductivity_degradation(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Conductivity degradation factor
        Dimensionless (0-1)
        """
        degradation = np.ones_like(T)
        
        # Degradation starts at 100°C
        mask = T > 100
        degradation[mask] = np.exp(-0.002 * (T[mask] - 100))
        
        # Rubber provides some protection
        protection_factor = 1 + 0.1 * rubber / 20  # Max 10% protection at 20% rubber
        degradation = 1 - (1 - degradation) / protection_factor
        
        return degradation.tolist()
    
    def _density_degradation(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Density degradation factor
        Dimensionless (0-1)
        """
        degradation = np.ones_like(T)
        
        # Multi-stage degradation
        mask_100_200 = (T >= 100) & (T < 200)
        mask_200_600 = (T >= 200) & (T < 600)
        mask_600 = T >= 600
        
        # Moisture loss
        degradation[mask_100_200] = 1 - 0.03 * (T[mask_100_200] - 100) / 100
        
        # Dehydration
        degradation[mask_200_600] = 0.97 - 0.10 * (T[mask_200_600] - 200) / 400
        
        # Decomposition
        degradation[mask_600] = 0.87 - 0.05 * (T[mask_600] - 600) / 200
        
        return degradation.tolist()
    
    def _phase_change_temperatures(self, rubber: float) -> Dict[str, float]:
        """
        Critical temperatures for phase changes
        °C
        """
        return {
            'free_water_evaporation': 100,
            'bound_water_release_start': 105 + rubber * 0.5,
            'bound_water_release_end': 200 + rubber * 2,
            'portlandite_decomposition': 450 - rubber * 5,
            'calcium_carbonate_decomposition': 700 - rubber * 10,
            'rubber_decomposition_start': 250,
            'rubber_decomposition_peak': 380,
            'rubber_decomposition_end': 500
        }
    
    def _latent_heats(self, rubber: float) -> Dict[str, float]:
        """
        Latent heats for phase changes
        J/kg
        """
        return {
            'water_evaporation': 2257000,  # J/kg
            'bound_water': 500000 * (1 - rubber * 0.01),
            'portlandite': 100000,
            'calcium_carbonate': 150000,
            'rubber_decomposition': 800000 * rubber / 100
        }
    
    def _add_stochastic_variation(self, properties: Dict, cv: float = 0.05) -> Dict:
        """
        Add stochastic variation to properties
        
        Parameters:
        -----------
        properties : Dict
            Base properties
        cv : float
            Coefficient of variation
        
        Returns:
        --------
        Dict : Properties with stochastic variation
        """
        varied_props = {}
        
        for key, value in properties.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], (int, float)):
                # Add normal variation
                base = np.array(value)
                std = base * cv
                varied = np.random.normal(base, std)
                varied_props[key] = varied.tolist()
            elif isinstance(value, dict):
                # Recursively apply to nested dicts
                varied_props[key] = self._add_stochastic_variation(value, cv)
            else:
                # Keep as is
                varied_props[key] = value
        
        return varied_props