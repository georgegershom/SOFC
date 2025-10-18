"""
Mechanical property generation module
Provides temperature-dependent mechanical properties with constitutive models
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import interpolate
from scipy.special import erf

class MechanicalPropertyGenerator:
    """
    Generator for temperature-dependent mechanical properties
    Includes elastic, plastic, and damage evolution models
    """
    
    def __init__(self, microstructure_params: Dict):
        """
        Initialize mechanical property generator
        
        Parameters:
        -----------
        microstructure_params : Dict
            Microstructural parameters for each mix
        """
        self.microstructure = microstructure_params
        
    def generate(self, mix_id: str, data_type: str) -> Dict:
        """
        Generate complete mechanical property dataset for a mix
        
        Parameters:
        -----------
        mix_id : str
            Mix identification
        data_type : str
            'Calibration' or 'Validation'
        
        Returns:
        --------
        Dict : Mechanical properties with temperature dependence
        """
        micro = self.microstructure[mix_id]
        rubber_content = micro['rubber_content_percent']
        
        # Temperature array
        temps = np.linspace(20, 800, 40)
        
        # Generate elastic properties
        elastic = {
            'temperature': temps.tolist(),
            'elastic_modulus': self._elastic_modulus(temps, rubber_content),
            'poisson_ratio': self._poisson_ratio(temps, rubber_content),
            'shear_modulus': self._shear_modulus(temps, rubber_content),
            'bulk_modulus': self._bulk_modulus(temps, rubber_content),
        }
        
        # Generate strength properties
        strength = {
            'temperature': temps.tolist(),
            'compressive_strength': self._compressive_strength(temps, rubber_content),
            'tensile_strength': self._tensile_strength(temps, rubber_content),
            'flexural_strength': self._flexural_strength(temps, rubber_content),
            'fracture_energy': self._fracture_energy(temps, rubber_content),
            'cohesion': self._cohesion(temps, rubber_content),
            'friction_angle': self._friction_angle(temps, rubber_content),
        }
        
        # Generate plastic properties
        plastic = {
            'temperature': temps.tolist(),
            'yield_stress': self._yield_stress(temps, rubber_content),
            'hardening_modulus': self._hardening_modulus(temps, rubber_content),
            'plastic_strain_at_peak': self._plastic_strain_peak(temps, rubber_content),
            'dilation_angle': self._dilation_angle(temps, rubber_content),
            'eccentricity': 0.1,  # Flow potential eccentricity
            'fb0_fc0_ratio': 1.16,  # Biaxial to uniaxial strength ratio
            'K': 0.667,  # Viscosity parameter
        }
        
        # Generate damage evolution
        damage = {
            'temperature': temps.tolist(),
            'damage_initiation_strain': self._damage_initiation(temps, rubber_content),
            'damage_evolution': self._damage_evolution(temps, rubber_content),
            'stiffness_degradation': self._stiffness_degradation(temps, rubber_content),
            'compression_damage': self._compression_damage(temps, rubber_content),
            'tension_damage': self._tension_damage(temps, rubber_content),
        }
        
        # Generate creep properties
        creep = {
            'temperature': temps.tolist(),
            'creep_compliance': self._creep_compliance(temps, rubber_content),
            'activation_energy': self._activation_energy(rubber_content),
            'creep_exponent': self._creep_exponent(temps, rubber_content),
            'reference_stress': 10.0,  # MPa
        }
        
        # Combine all properties
        properties = {
            'elastic': elastic,
            'strength': strength,
            'plastic': plastic,
            'damage': damage,
            'creep': creep,
            'constitutive_model': self._get_constitutive_parameters(rubber_content)
        }
        
        # Add stochastic variation for validation set
        if data_type == 'Validation':
            properties = self._add_stochastic_variation(properties)
        
        return properties
    
    def _elastic_modulus(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent elastic modulus
        GPa
        """
        # Base modulus at room temperature
        E_base = 35 - 0.8 * rubber  # GPa, decreases with rubber content
        
        # Temperature degradation
        degradation = np.ones_like(T)
        
        # Different degradation stages
        mask_20_100 = T <= 100
        mask_100_300 = (T > 100) & (T <= 300)
        mask_300_500 = (T > 300) & (T <= 500)
        mask_500_800 = T > 500
        
        # Minimal degradation up to 100°C
        degradation[mask_20_100] = 1 - 0.0001 * (T[mask_20_100] - 20)
        
        # Moderate degradation 100-300°C
        degradation[mask_100_300] = 0.992 - 0.001 * (T[mask_100_300] - 100)
        
        # Significant degradation 300-500°C
        degradation[mask_300_500] = 0.792 - 0.002 * (T[mask_300_500] - 300)
        
        # Severe degradation >500°C
        degradation[mask_500_800] = 0.392 * np.exp(-0.002 * (T[mask_500_800] - 500))
        
        # Rubber provides some ductility benefit
        rubber_benefit = 1 + 0.05 * rubber / 20  # Max 5% benefit at 20% rubber
        degradation = degradation * rubber_benefit
        
        E = E_base * degradation
        
        # Ensure minimum value
        E = np.maximum(E, 0.5)
        
        return E.tolist()
    
    def _poisson_ratio(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent Poisson's ratio
        Dimensionless
        """
        # Base Poisson's ratio
        nu_base = 0.20 + 0.002 * rubber  # Slightly increases with rubber
        
        # Temperature effect - increases with temperature due to softening
        nu = nu_base * np.ones_like(T)
        
        mask_200 = T > 200
        nu[mask_200] = nu_base + 0.00005 * (T[mask_200] - 200)
        
        # Cap at 0.45 to maintain stability
        nu = np.minimum(nu, 0.45)
        
        return nu.tolist()
    
    def _shear_modulus(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent shear modulus
        GPa
        """
        # Calculate from E and nu
        E = np.array(self._elastic_modulus(T, rubber))
        nu = np.array(self._poisson_ratio(T, rubber))
        
        G = E / (2 * (1 + nu))
        
        return G.tolist()
    
    def _bulk_modulus(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent bulk modulus
        GPa
        """
        # Calculate from E and nu
        E = np.array(self._elastic_modulus(T, rubber))
        nu = np.array(self._poisson_ratio(T, rubber))
        
        K = E / (3 * (1 - 2 * nu))
        
        return K.tolist()
    
    def _compressive_strength(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent compressive strength
        MPa
        """
        # Base strength at room temperature
        fc_base = 45 - 0.5 * rubber  # MPa, decreases with rubber
        
        # Temperature degradation following experimental data
        degradation = np.ones_like(T)
        
        # Strength evolution stages
        mask_20_100 = T <= 100
        mask_100_200 = (T > 100) & (T <= 200)
        mask_200_400 = (T > 200) & (T <= 400)
        mask_400_600 = (T > 400) & (T <= 600)
        mask_600_800 = T > 600
        
        # Slight increase up to 100°C due to moisture loss
        degradation[mask_20_100] = 1 + 0.0002 * (T[mask_20_100] - 20)
        
        # Gradual decrease 100-200°C
        degradation[mask_100_200] = 1.016 - 0.0015 * (T[mask_100_200] - 100)
        
        # Moderate decrease 200-400°C
        degradation[mask_200_400] = 0.866 - 0.0025 * (T[mask_200_400] - 200)
        
        # Significant decrease 400-600°C
        degradation[mask_400_600] = 0.366 - 0.001 * (T[mask_400_600] - 400)
        
        # Severe degradation >600°C
        degradation[mask_600_800] = 0.166 * np.exp(-0.003 * (T[mask_600_800] - 600))
        
        fc = fc_base * degradation
        
        # Ensure minimum value
        fc = np.maximum(fc, 1.0)
        
        return fc.tolist()
    
    def _tensile_strength(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent tensile strength
        MPa
        """
        # Tensile strength as fraction of compressive strength
        fc = np.array(self._compressive_strength(T, rubber))
        
        # Base ratio increases with rubber content (rubber improves tensile behavior)
        ratio_base = 0.10 + 0.002 * rubber
        
        # Ratio degrades with temperature
        ratio = ratio_base * np.ones_like(T)
        mask_200 = T > 200
        ratio[mask_200] = ratio_base * (1 - 0.001 * (T[mask_200] - 200))
        ratio = np.maximum(ratio, 0.05)
        
        ft = fc * ratio
        
        return ft.tolist()
    
    def _flexural_strength(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent flexural strength
        MPa
        """
        # Flexural strength related to tensile strength
        ft = np.array(self._tensile_strength(T, rubber))
        
        # Flexural is typically 1.5-2x tensile
        factor = 1.8 + 0.01 * rubber  # Rubber improves flexural behavior
        
        ff = ft * factor
        
        return ff.tolist()
    
    def _fracture_energy(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent fracture energy
        N/m
        """
        # Base fracture energy
        Gf_base = 120 + 5 * rubber  # N/m, increases with rubber (ductility)
        
        # Temperature effect
        Gf = Gf_base * np.ones_like(T)
        
        # Increases initially due to microcracking, then decreases
        mask_100_300 = (T >= 100) & (T <= 300)
        mask_300 = T > 300
        
        Gf[mask_100_300] = Gf_base * (1 + 0.0005 * (T[mask_100_300] - 100))
        Gf[mask_300] = Gf_base * 1.1 * np.exp(-0.001 * (T[mask_300] - 300))
        
        return Gf.tolist()
    
    def _cohesion(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent cohesion (Drucker-Prager model)
        MPa
        """
        # Related to compressive strength
        fc = np.array(self._compressive_strength(T, rubber))
        
        # Cohesion approximately fc/10
        c = fc / 10
        
        return c.tolist()
    
    def _friction_angle(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent internal friction angle
        Degrees
        """
        # Base friction angle
        phi_base = 37 - 0.3 * rubber  # Degrees, slightly decreases with rubber
        
        # Temperature degradation
        phi = phi_base * np.ones_like(T)
        
        mask_300 = T > 300
        phi[mask_300] = phi_base - 0.01 * (T[mask_300] - 300)
        
        # Minimum value
        phi = np.maximum(phi, 20)
        
        return phi.tolist()
    
    def _yield_stress(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent yield stress
        MPa
        """
        # Approximately 0.4-0.5 of compressive strength
        fc = np.array(self._compressive_strength(T, rubber))
        
        sigma_y = 0.45 * fc
        
        return sigma_y.tolist()
    
    def _hardening_modulus(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Temperature-dependent hardening modulus
        MPa
        """
        # Related to elastic modulus
        E = np.array(self._elastic_modulus(T, rubber)) * 1000  # Convert to MPa
        
        # Hardening modulus as fraction of elastic modulus
        H = E * 0.01  # 1% of elastic modulus
        
        return H.tolist()
    
    def _plastic_strain_peak(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Plastic strain at peak stress
        Dimensionless
        """
        # Base strain at peak
        eps_base = 0.002 + 0.0002 * rubber  # Increases with rubber
        
        # Temperature effect - increases with temperature
        eps = eps_base * np.ones_like(T)
        
        mask_200 = T > 200
        eps[mask_200] = eps_base * (1 + 0.001 * (T[mask_200] - 200))
        
        return eps.tolist()
    
    def _dilation_angle(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Dilation angle for plasticity model
        Degrees
        """
        # Base dilation angle
        psi_base = 36 - 0.5 * rubber  # Degrees
        
        # Temperature effect
        psi = psi_base * np.ones_like(T)
        
        mask_300 = T > 300
        psi[mask_300] = psi_base - 0.02 * (T[mask_300] - 300)
        
        # Minimum value
        psi = np.maximum(psi, 10)
        
        return psi.tolist()
    
    def _damage_initiation(self, T: np.ndarray, rubber: float) -> Dict:
        """
        Damage initiation parameters
        """
        # Strain at damage initiation
        eps_tension = 0.0001 + 0.00002 * rubber
        eps_compression = 0.001 + 0.0001 * rubber
        
        # Temperature modification
        temp_factor = 1 + 0.001 * np.maximum(T - 200, 0)
        
        return {
            'tension_strain': (eps_tension * temp_factor).tolist(),
            'compression_strain': (eps_compression * temp_factor).tolist(),
            'tension_stress': self._tensile_strength(T, rubber),
            'compression_stress': self._compressive_strength(T, rubber)
        }
    
    def _damage_evolution(self, T: np.ndarray, rubber: float) -> Dict:
        """
        Damage evolution parameters
        """
        # Exponential damage evolution
        # d = 1 - exp(-alpha * plastic_strain)
        
        alpha_tension_base = 500 - 20 * rubber  # Rubber delays damage
        alpha_compression_base = 200 - 10 * rubber
        
        # Temperature effect
        temp_factor = 1 - 0.0005 * np.maximum(T - 200, 0)
        
        alpha_tension = alpha_tension_base * temp_factor
        alpha_compression = alpha_compression_base * temp_factor
        
        return {
            'tension_alpha': alpha_tension.tolist(),
            'compression_alpha': alpha_compression.tolist(),
            'type': 'exponential',
            'mesh_sensitivity': 'characteristic_length'
        }
    
    def _stiffness_degradation(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Stiffness degradation factor
        Dimensionless (0-1)
        """
        # Related to elastic modulus degradation
        E = np.array(self._elastic_modulus(T, rubber))
        E_initial = E[0]
        
        degradation = E / E_initial
        
        return degradation.tolist()
    
    def _compression_damage(self, T: np.ndarray, rubber: float) -> Dict:
        """
        Compression damage parameters
        """
        # Damage variable evolution with plastic strain
        plastic_strains = np.linspace(0, 0.01, 20)
        
        damage_values = []
        for temp in T:
            # Temperature-dependent damage evolution
            if temp < 200:
                alpha = 100
            elif temp < 400:
                alpha = 150
            else:
                alpha = 200
            
            # Rubber effect
            alpha = alpha * (1 - 0.1 * rubber / 20)
            
            d = 1 - np.exp(-alpha * plastic_strains)
            damage_values.append(d.tolist())
        
        return {
            'plastic_strain': plastic_strains.tolist(),
            'damage_variable': damage_values
        }
    
    def _tension_damage(self, T: np.ndarray, rubber: float) -> Dict:
        """
        Tension damage parameters
        """
        # More severe damage in tension
        plastic_strains = np.linspace(0, 0.005, 20)
        
        damage_values = []
        for temp in T:
            # Temperature-dependent damage evolution
            if temp < 200:
                alpha = 500
            elif temp < 400:
                alpha = 700
            else:
                alpha = 1000
            
            # Rubber effect - delays damage
            alpha = alpha * (1 - 0.2 * rubber / 20)
            
            d = 1 - np.exp(-alpha * plastic_strains)
            damage_values.append(d.tolist())
        
        return {
            'plastic_strain': plastic_strains.tolist(),
            'damage_variable': damage_values
        }
    
    def _creep_compliance(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Creep compliance function
        1/MPa
        """
        # Power law creep model
        # J(t) = J0 + J1 * t^n
        
        E = np.array(self._elastic_modulus(T, rubber)) * 1000  # Convert to MPa
        
        # Instantaneous compliance
        J0 = 1 / E
        
        # Temperature-dependent creep compliance
        J1_base = 1e-7  # 1/MPa
        
        # Arrhenius temperature dependence
        Q = self._activation_energy(rubber)
        R = 8.314  # J/(mol·K)
        T_kelvin = T + 273.15
        
        J1 = J1_base * np.exp(-Q / (R * T_kelvin))
        
        # Rubber increases creep
        rubber_factor = 1 + 0.1 * rubber
        J1 = J1 * rubber_factor
        
        # Total initial compliance
        J_total = J0 + J1
        
        return (J_total * 1e6).tolist()  # Convert to 1/GPa
    
    def _activation_energy(self, rubber: float) -> float:
        """
        Activation energy for creep
        J/mol
        """
        # Base activation energy
        Q_base = 45000  # J/mol
        
        # Rubber reduces activation energy (easier creep)
        Q = Q_base - 500 * rubber
        
        return Q
    
    def _creep_exponent(self, T: np.ndarray, rubber: float) -> List[float]:
        """
        Creep power law exponent
        Dimensionless
        """
        # Base exponent
        n_base = 0.3 + 0.01 * rubber
        
        # Temperature effect
        n = n_base * np.ones_like(T)
        
        mask_300 = T > 300
        n[mask_300] = n_base + 0.0001 * (T[mask_300] - 300)
        
        return n.tolist()
    
    def _get_constitutive_parameters(self, rubber: float) -> Dict:
        """
        Get constitutive model parameters
        """
        return {
            'model_type': 'Concrete_Damaged_Plasticity',
            'plasticity': {
                'eccentricity': 0.1,
                'fb0_fc0_ratio': 1.16,
                'K': 0.667,
                'viscosity': 0.0001
            },
            'damage': {
                'compression_recovery': 0.9 - 0.01 * rubber,
                'tension_recovery': 0.1 + 0.01 * rubber,
                'compression_stiffness_recovery': 0.9,
                'tension_stiffness_recovery': 0.0
            },
            'regularization': {
                'type': 'characteristic_length',
                'mesh_dependency': True
            },
            'rubber_modified_parameters': {
                'ductility_enhancement': 1 + 0.1 * rubber / 20,
                'energy_absorption': 1 + 0.2 * rubber / 20,
                'damping_ratio': 0.05 + 0.002 * rubber
            }
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
                # Ensure positive values for most properties
                if key not in ['poisson_ratio', 'damage_variable']:
                    varied = np.maximum(varied, 0.01)
                varied_props[key] = varied.tolist()
            elif isinstance(value, dict):
                # Recursively apply to nested dicts
                varied_props[key] = self._add_stochastic_variation(value, cv)
            else:
                # Keep as is
                varied_props[key] = value
        
        return varied_props