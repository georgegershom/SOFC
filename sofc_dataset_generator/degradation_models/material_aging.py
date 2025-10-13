"""
Material aging model for SOFC degradation analysis.
Implements various aging mechanisms including creep, oxidation, and phase changes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class MaterialAgingModel:
    """
    Material aging model for SOFC degradation analysis.
    
    Implements:
    - Creep deformation and damage accumulation
    - Oxidation and corrosion
    - Phase changes and microstructural evolution
    - Property degradation over time
    """
    
    def __init__(self, config: Dict):
        """
        Initialize material aging model.
        
        Args:
            config: Configuration dictionary with material properties and aging parameters
        """
        self.config = config
        self.logger = logging.getLogger('MaterialAgingModel')
        
        # Default material properties
        self.material_props = {
            'anode': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'density': 3000,  # kg/m³
                'thermal_expansion': 12e-6,  # 1/K
                'creep_coefficient': 1e-15,  # s⁻¹
                'creep_exponent': 3.0,
                'activation_energy': 100000,  # J/mol
                'oxidation_rate': 1e-12,  # m/s
                'phase_change_temp': 1273.15,  # K
            },
            'electrolyte': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'density': 6000,  # kg/m³
                'thermal_expansion': 10e-6,  # 1/K
                'creep_coefficient': 1e-16,  # s⁻¹
                'creep_exponent': 3.5,
                'activation_energy': 120000,  # J/mol
                'oxidation_rate': 1e-13,  # m/s
                'phase_change_temp': 1373.15,  # K
            },
            'cathode': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'density': 3000,  # kg/m³
                'thermal_expansion': 12e-6,  # 1/K
                'creep_coefficient': 1e-15,  # s⁻¹
                'creep_exponent': 3.0,
                'activation_energy': 100000,  # J/mol
                'oxidation_rate': 1e-12,  # m/s
                'phase_change_temp': 1273.15,  # K
            },
            'interconnect': {
                'youngs_modulus': 200e9,  # Pa
                'poisson_ratio': 0.3,
                'density': 8000,  # kg/m³
                'thermal_expansion': 12e-6,  # 1/K
                'creep_coefficient': 1e-14,  # s⁻¹
                'creep_exponent': 2.5,
                'activation_energy': 80000,  # J/mol
                'oxidation_rate': 1e-11,  # m/s
                'phase_change_temp': 1173.15,  # K
            }
        }
        
        # Update with config values
        self.material_props.update(config.get('material_properties', {}))
        
        # Aging state tracking
        self.aging_state = {}
        self.property_history = {}
        
    def initialize_aging_state(
        self,
        material_type: str,
        initial_properties: Optional[Dict] = None
    ) -> None:
        """
        Initialize aging state for a material.
        
        Args:
            material_type: Type of material
            initial_properties: Optional initial property values
        """
        if material_type not in self.material_props:
            raise ValueError(f"Unknown material type: {material_type}")
        
        # Initialize aging state
        self.aging_state[material_type] = {
            'creep_damage': 0.0,
            'oxidation_thickness': 0.0,
            'phase_fraction': 0.0,
            'porosity_change': 0.0,
            'grain_size_change': 0.0,
            'residual_stress': 0.0,
            'property_degradation': {
                'youngs_modulus': 1.0,  # Relative to initial
                'strength': 1.0,
                'toughness': 1.0,
                'conductivity': 1.0
            }
        }
        
        # Initialize property history
        self.property_history[material_type] = {
            'time': [],
            'creep_damage': [],
            'oxidation_thickness': [],
            'phase_fraction': [],
            'porosity_change': [],
            'grain_size_change': [],
            'residual_stress': [],
            'property_degradation': {
                'youngs_modulus': [],
                'strength': [],
                'toughness': [],
                'conductivity': []
            }
        }
        
        self.logger.info(f"Initialized aging state for {material_type}")
    
    def simulate_aging(
        self,
        material_type: str,
        stress_history: np.ndarray,
        temperature_history: np.ndarray,
        time_points: np.ndarray,
        environmental_conditions: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Simulate material aging over time.
        
        Args:
            material_type: Type of material
            stress_history: Stress history (Pa)
            temperature_history: Temperature history (K)
            time_points: Time points (s)
            environmental_conditions: Optional environmental conditions
            
        Returns:
            Dictionary containing aging simulation results
        """
        if material_type not in self.aging_state:
            self.initialize_aging_state(material_type)
        
        self.logger.info(f"Starting aging simulation for {material_type}...")
        
        n_time = len(time_points)
        material_props = self.material_props[material_type]
        
        # Initialize results arrays
        creep_damage = np.zeros(n_time)
        oxidation_thickness = np.zeros(n_time)
        phase_fraction = np.zeros(n_time)
        porosity_change = np.zeros(n_time)
        grain_size_change = np.zeros(n_time)
        residual_stress = np.zeros(n_time)
        
        property_degradation = {
            'youngs_modulus': np.ones(n_time),
            'strength': np.ones(n_time),
            'toughness': np.ones(n_time),
            'conductivity': np.ones(n_time)
        }
        
        # Simulate aging processes
        for i in range(1, n_time):
            dt = time_points[i] - time_points[i-1]
            current_stress = stress_history[i]
            current_temp = temperature_history[i]
            
            # Creep damage accumulation
            creep_damage[i] = self._calculate_creep_damage(
                current_stress, current_temp, dt, material_props, creep_damage[i-1]
            )
            
            # Oxidation growth
            oxidation_thickness[i] = self._calculate_oxidation_growth(
                current_temp, dt, material_props, oxidation_thickness[i-1], environmental_conditions
            )
            
            # Phase changes
            phase_fraction[i] = self._calculate_phase_change(
                current_temp, dt, material_props, phase_fraction[i-1]
            )
            
            # Porosity evolution
            porosity_change[i] = self._calculate_porosity_evolution(
                current_stress, current_temp, dt, material_props, porosity_change[i-1]
            )
            
            # Grain size evolution
            grain_size_change[i] = self._calculate_grain_size_evolution(
                current_temp, dt, material_props, grain_size_change[i-1]
            )
            
            # Residual stress accumulation
            residual_stress[i] = self._calculate_residual_stress(
                current_stress, current_temp, dt, material_props, residual_stress[i-1]
            )
            
            # Property degradation
            property_degradation['youngs_modulus'][i] = self._calculate_property_degradation(
                'youngs_modulus', creep_damage[i], oxidation_thickness[i], phase_fraction[i]
            )
            property_degradation['strength'][i] = self._calculate_property_degradation(
                'strength', creep_damage[i], oxidation_thickness[i], phase_fraction[i]
            )
            property_degradation['toughness'][i] = self._calculate_property_degradation(
                'toughness', creep_damage[i], oxidation_thickness[i], phase_fraction[i]
            )
            property_degradation['conductivity'][i] = self._calculate_property_degradation(
                'conductivity', creep_damage[i], oxidation_thickness[i], phase_fraction[i]
            )
            
            # Update aging state
            self.aging_state[material_type]['creep_damage'] = creep_damage[i]
            self.aging_state[material_type]['oxidation_thickness'] = oxidation_thickness[i]
            self.aging_state[material_type]['phase_fraction'] = phase_fraction[i]
            self.aging_state[material_type]['porosity_change'] = porosity_change[i]
            self.aging_state[material_type]['grain_size_change'] = grain_size_change[i]
            self.aging_state[material_type]['residual_stress'] = residual_stress[i]
            
            # Update property history
            self._update_property_history(material_type, time_points[i], {
                'creep_damage': creep_damage[i],
                'oxidation_thickness': oxidation_thickness[i],
                'phase_fraction': phase_fraction[i],
                'porosity_change': porosity_change[i],
                'grain_size_change': grain_size_change[i],
                'residual_stress': residual_stress[i],
                'property_degradation': {k: v[i] for k, v in property_degradation.items()}
            })
        
        results = {
            'time_points': time_points,
            'creep_damage': creep_damage,
            'oxidation_thickness': oxidation_thickness,
            'phase_fraction': phase_fraction,
            'porosity_change': porosity_change,
            'grain_size_change': grain_size_change,
            'residual_stress': residual_stress,
            'property_degradation': property_degradation,
            'aging_state': self.aging_state[material_type].copy()
        }
        
        self.logger.info(f"Aging simulation completed for {material_type}")
        return results
    
    def _calculate_creep_damage(
        self,
        stress: float,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_damage: float
    ) -> float:
        """Calculate creep damage accumulation."""
        # Norton's creep law: ε̇ = A * σ^n * exp(-Q/RT)
        R = 8.314  # Gas constant
        A = material_props['creep_coefficient']
        n = material_props['creep_exponent']
        Q = material_props['activation_energy']
        
        # Creep strain rate
        creep_rate = A * (stress ** n) * np.exp(-Q / (R * temperature))
        
        # Damage accumulation (simplified Kachanov model)
        damage_rate = creep_rate * (1 - previous_damage) ** (-1)
        
        # Update damage
        new_damage = previous_damage + damage_rate * dt
        
        # Cap at 1.0 (complete failure)
        return min(new_damage, 1.0)
    
    def _calculate_oxidation_growth(
        self,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_thickness: float,
        environmental_conditions: Optional[Dict]
    ) -> float:
        """Calculate oxidation layer growth."""
        # Parabolic oxidation law: x² = k_p * t
        k_p = material_props['oxidation_rate'] * np.exp(-5000 / temperature)
        
        # Environmental effects
        if environmental_conditions:
            oxygen_partial_pressure = environmental_conditions.get('oxygen_partial_pressure', 0.21)
            k_p *= oxygen_partial_pressure ** 0.5
        
        # Calculate new thickness
        new_thickness_squared = previous_thickness ** 2 + k_p * dt
        new_thickness = np.sqrt(new_thickness_squared)
        
        return new_thickness
    
    def _calculate_phase_change(
        self,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_fraction: float
    ) -> float:
        """Calculate phase change fraction."""
        phase_change_temp = material_props['phase_change_temp']
        
        # Simplified phase change model
        if temperature > phase_change_temp:
            # Heating: increase phase fraction
            phase_rate = 0.01 * (temperature - phase_change_temp) / 100  # 1% per 100K
            new_fraction = min(previous_fraction + phase_rate * dt, 1.0)
        else:
            # Cooling: decrease phase fraction
            phase_rate = 0.005 * (phase_change_temp - temperature) / 100  # 0.5% per 100K
            new_fraction = max(previous_fraction - phase_rate * dt, 0.0)
        
        return new_fraction
    
    def _calculate_porosity_evolution(
        self,
        stress: float,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_porosity: float
    ) -> float:
        """Calculate porosity evolution."""
        # Porosity change due to creep and sintering
        creep_porosity_rate = 0.001 * stress / 1e6  # Increase with stress
        sintering_rate = 0.0001 * np.exp(-1000 / temperature)  # Decrease with temperature
        
        net_porosity_rate = creep_porosity_rate - sintering_rate
        new_porosity = previous_porosity + net_porosity_rate * dt
        
        # Bound between 0 and 0.5
        return np.clip(new_porosity, 0.0, 0.5)
    
    def _calculate_grain_size_evolution(
        self,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_grain_size: float
    ) -> float:
        """Calculate grain size evolution."""
        # Grain growth: d = d₀ + k * t^0.5
        k = 1e-12 * np.exp(-5000 / temperature)  # Growth rate constant
        
        new_grain_size = previous_grain_size + k * np.sqrt(dt)
        
        return new_grain_size
    
    def _calculate_residual_stress(
        self,
        stress: float,
        temperature: float,
        dt: float,
        material_props: Dict,
        previous_residual_stress: float
    ) -> float:
        """Calculate residual stress accumulation."""
        # Thermal stress due to temperature cycling
        thermal_expansion = material_props['thermal_expansion']
        youngs_modulus = material_props['youngs_modulus']
        
        # Simplified thermal stress calculation
        temp_change = temperature - 1073.15  # Reference temperature
        thermal_stress = youngs_modulus * thermal_expansion * temp_change
        
        # Residual stress accumulation (simplified)
        residual_stress_rate = 0.1 * thermal_stress / 1e6  # 0.1 MPa/s
        new_residual_stress = previous_residual_stress + residual_stress_rate * dt
        
        return new_residual_stress
    
    def _calculate_property_degradation(
        self,
        property_name: str,
        creep_damage: float,
        oxidation_thickness: float,
        phase_fraction: float
    ) -> float:
        """Calculate property degradation based on aging mechanisms."""
        # Base degradation from creep damage
        base_degradation = 1.0 - creep_damage
        
        # Additional degradation from oxidation
        oxidation_degradation = 1.0 - 0.1 * oxidation_thickness / 1e-6  # 10% per μm
        
        # Additional degradation from phase changes
        phase_degradation = 1.0 - 0.05 * phase_fraction  # 5% per 10% phase change
        
        # Combined degradation
        total_degradation = base_degradation * oxidation_degradation * phase_degradation
        
        # Property-specific adjustments
        if property_name == 'youngs_modulus':
            return total_degradation
        elif property_name == 'strength':
            return total_degradation ** 1.5  # More sensitive to damage
        elif property_name == 'toughness':
            return total_degradation ** 2.0  # Most sensitive to damage
        elif property_name == 'conductivity':
            return total_degradation ** 0.5  # Less sensitive to damage
        else:
            return total_degradation
    
    def _update_property_history(
        self,
        material_type: str,
        time: float,
        properties: Dict
    ) -> None:
        """Update property history for a material."""
        history = self.property_history[material_type]
        history['time'].append(time)
        
        for key, value in properties.items():
            if key == 'property_degradation':
                for prop_name, prop_value in value.items():
                    history['property_degradation'][prop_name].append(prop_value)
            else:
                history[key].append(value)
    
    def predict_remaining_useful_life(
        self,
        material_type: str,
        current_aging_state: Dict,
        stress_history: np.ndarray,
        temperature_history: np.ndarray,
        time_points: np.ndarray,
        failure_criteria: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Predict remaining useful life based on current aging state.
        
        Args:
            material_type: Type of material
            current_aging_state: Current aging state
            stress_history: Stress history (Pa)
            temperature_history: Temperature history (K)
            time_points: Time points (s)
            failure_criteria: Optional failure criteria
            
        Returns:
            Dictionary containing RUL predictions
        """
        if failure_criteria is None:
            failure_criteria = {
                'creep_damage': 0.8,
                'oxidation_thickness': 10e-6,  # 10 μm
                'property_degradation': 0.5
            }
        
        # Update aging state
        self.aging_state[material_type] = current_aging_state
        
        # Simulate future aging
        future_time = np.linspace(time_points[-1], time_points[-1] + 30*24*3600, 1000)  # 30 days
        future_stress = np.full_like(future_time, stress_history[-1])
        future_temp = np.full_like(future_time, temperature_history[-1])
        
        results = self.simulate_aging(
            material_type, future_stress, future_temp, future_time
        )
        
        # Find failure times
        failure_times = {}
        
        # Creep damage failure
        creep_failure_idx = np.where(results['creep_damage'] >= failure_criteria['creep_damage'])[0]
        if len(creep_failure_idx) > 0:
            failure_times['creep'] = future_time[creep_failure_idx[0]] - time_points[-1]
        else:
            failure_times['creep'] = np.inf
        
        # Oxidation failure
        ox_failure_idx = np.where(results['oxidation_thickness'] >= failure_criteria['oxidation_thickness'])[0]
        if len(ox_failure_idx) > 0:
            failure_times['oxidation'] = future_time[ox_failure_idx[0]] - time_points[-1]
        else:
            failure_times['oxidation'] = np.inf
        
        # Property degradation failure
        prop_failure_idx = np.where(
            results['property_degradation']['strength'] <= failure_criteria['property_degradation']
        )[0]
        if len(prop_failure_idx) > 0:
            failure_times['property'] = future_time[prop_failure_idx[0]] - time_points[-1]
        else:
            failure_times['property'] = np.inf
        
        # Overall RUL (minimum of all failure modes)
        rul_time = min(failure_times.values())
        
        return {
            'rul_time': rul_time,
            'rul_hours': rul_time / 3600,
            'rul_days': rul_time / (24 * 3600),
            'failure_times': failure_times,
            'failure_criteria': failure_criteria,
            'aging_curves': results
        }
    
    def get_aging_statistics(self, material_type: str) -> Dict[str, Any]:
        """Get current aging statistics for a material."""
        if material_type not in self.aging_state:
            return {}
        
        state = self.aging_state[material_type]
        history = self.property_history[material_type]
        
        return {
            'current_state': state,
            'total_aging_time': history['time'][-1] if history['time'] else 0,
            'max_creep_damage': max(history['creep_damage']) if history['creep_damage'] else 0,
            'max_oxidation_thickness': max(history['oxidation_thickness']) if history['oxidation_thickness'] else 0,
            'max_phase_fraction': max(history['phase_fraction']) if history['phase_fraction'] else 0,
            'property_degradation_summary': {
                prop: {
                    'current': state['property_degradation'][prop],
                    'min': min(history['property_degradation'][prop]) if history['property_degradation'][prop] else 1.0,
                    'max': max(history['property_degradation'][prop]) if history['property_degradation'][prop] else 1.0
                }
                for prop in state['property_degradation'].keys()
            }
        }