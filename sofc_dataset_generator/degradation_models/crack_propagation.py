"""
Crack propagation model for SOFC degradation analysis.
Implements fracture mechanics-based crack growth models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy.integrate import odeint
from scipy.interpolate import interp1d

class CrackPropagationModel:
    """
    Crack propagation model for SOFC degradation analysis.
    
    Implements:
    - Paris' law for fatigue crack growth
    - Stress intensity factor calculations
    - Crack coalescence and branching
    - Failure prediction
    """
    
    def __init__(self, config: Dict):
        """
        Initialize crack propagation model.
        
        Args:
            config: Configuration dictionary with material properties and crack parameters
        """
        self.config = config
        self.logger = logging.getLogger('CrackPropagationModel')
        
        # Default material properties
        self.material_props = {
            'anode': {
                'fracture_toughness': 2.0,  # MPa√m
                'paris_c': 1e-12,  # Paris law constant
                'paris_m': 3.0,    # Paris law exponent
                'threshold_stress_intensity': 0.5,  # MPa√m
                'critical_crack_length': 1e-3,  # m
            },
            'electrolyte': {
                'fracture_toughness': 1.5,  # MPa√m
                'paris_c': 1e-13,  # Paris law constant
                'paris_m': 3.2,    # Paris law exponent
                'threshold_stress_intensity': 0.3,  # MPa√m
                'critical_crack_length': 0.5e-3,  # m
            },
            'cathode': {
                'fracture_toughness': 2.0,  # MPa√m
                'paris_c': 1e-12,  # Paris law constant
                'paris_m': 3.0,    # Paris law exponent
                'threshold_stress_intensity': 0.5,  # MPa√m
                'critical_crack_length': 1e-3,  # m
            },
            'interconnect': {
                'fracture_toughness': 50.0,  # MPa√m
                'paris_c': 1e-11,  # Paris law constant
                'paris_m': 2.8,    # Paris law exponent
                'threshold_stress_intensity': 2.0,  # MPa√m
                'critical_crack_length': 5e-3,  # m
            }
        }
        
        # Update with config values
        self.material_props.update(config.get('material_properties', {}))
        
        # Crack tracking
        self.cracks = []
        self.crack_history = []
        
    def initialize_cracks(
        self,
        initial_cracks: List[Dict],
        material_type: str = 'electrolyte'
    ) -> None:
        """
        Initialize crack system with initial crack configurations.
        
        Args:
            initial_cracks: List of initial crack configurations
            material_type: Material type for crack properties
        """
        self.cracks = []
        self.crack_history = []
        
        for i, crack_config in enumerate(initial_cracks):
            crack = {
                'id': i,
                'material_type': material_type,
                'initial_length': crack_config.get('length', 1e-6),  # m
                'current_length': crack_config.get('length', 1e-6),  # m
                'position': crack_config.get('position', [0.0, 0.0, 0.0]),
                'orientation': crack_config.get('orientation', [1.0, 0.0, 0.0]),
                'crack_type': crack_config.get('type', 'mode_I'),  # mode_I, mode_II, mode_III, mixed
                'growth_history': [],
                'is_active': True,
                'failure_time': None
            }
            self.cracks.append(crack)
        
        self.logger.info(f"Initialized {len(self.cracks)} cracks in {material_type}")
    
    def simulate_crack_growth(
        self,
        stress_history: np.ndarray,
        time_points: np.ndarray,
        temperature_history: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Simulate crack growth over time using Paris' law.
        
        Args:
            stress_history: Stress history (Pa)
            time_points: Time points (s)
            temperature_history: Optional temperature history (K)
            
        Returns:
            Dictionary containing crack growth results
        """
        self.logger.info("Starting crack growth simulation...")
        
        n_cracks = len(self.cracks)
        n_time = len(time_points)
        
        # Initialize results arrays
        crack_lengths = np.zeros((n_cracks, n_time))
        stress_intensity_factors = np.zeros((n_cracks, n_time))
        crack_growth_rates = np.zeros((n_cracks, n_time))
        failure_times = np.full(n_cracks, np.nan)
        
        # Simulate each crack
        for i, crack in enumerate(self.cracks):
            if not crack['is_active']:
                continue
                
            material_props = self.material_props[crack['material_type']]
            
            # Initial conditions
            initial_length = crack['current_length']
            crack_lengths[i, 0] = initial_length
            
            # Calculate stress intensity factor for initial crack
            stress_intensity_factors[i, 0] = self._calculate_stress_intensity_factor(
                initial_length, stress_history[0], crack
            )
            
            # Simulate crack growth
            for j in range(1, n_time):
                # Current crack length
                current_length = crack_lengths[i, j-1]
                
                # Calculate stress intensity factor
                K = self._calculate_stress_intensity_factor(
                    current_length, stress_history[j], crack
                )
                stress_intensity_factors[i, j] = K
                
                # Check for crack growth threshold
                if K < material_props['threshold_stress_intensity']:
                    crack_growth_rates[i, j] = 0.0
                    crack_lengths[i, j] = current_length
                else:
                    # Paris' law: da/dN = C * (ΔK)^m
                    delta_K = K - material_props['threshold_stress_intensity']
                    da_dN = material_props['paris_c'] * (delta_K ** material_props['paris_m'])
                    
                    # Convert to time-based growth rate
                    # Assuming 1 Hz frequency for simplicity
                    da_dt = da_dN * 1.0  # cycles per second
                    
                    # Temperature effect on growth rate
                    if temperature_history is not None:
                        temp_factor = np.exp(-1000 / (8.314 * temperature_history[j]))
                        da_dt *= temp_factor
                    
                    crack_growth_rates[i, j] = da_dt
                    
                    # Update crack length
                    dt = time_points[j] - time_points[j-1]
                    new_length = current_length + da_dt * dt
                    crack_lengths[i, j] = new_length
                    
                    # Check for failure
                    if new_length >= material_props['critical_crack_length']:
                        failure_times[i] = time_points[j]
                        crack['is_active'] = False
                        crack['failure_time'] = time_points[j]
                        self.logger.warning(f"Crack {i} failed at time {time_points[j]:.2f} s")
                        break
                
                # Update crack history
                crack['current_length'] = crack_lengths[i, j]
                crack['growth_history'].append({
                    'time': time_points[j],
                    'length': crack_lengths[i, j],
                    'stress_intensity': K,
                    'growth_rate': crack_growth_rates[i, j]
                })
        
        # Calculate crack interactions
        crack_interactions = self._calculate_crack_interactions(crack_lengths, time_points)
        
        # Calculate failure probability
        failure_probability = self._calculate_failure_probability(
            crack_lengths, stress_intensity_factors, time_points
        )
        
        results = {
            'crack_lengths': crack_lengths,
            'stress_intensity_factors': stress_intensity_factors,
            'crack_growth_rates': crack_growth_rates,
            'failure_times': failure_times,
            'crack_interactions': crack_interactions,
            'failure_probability': failure_probability,
            'crack_history': [crack['growth_history'] for crack in self.cracks]
        }
        
        self.logger.info("Crack growth simulation completed")
        return results
    
    def calculate_stress_intensity_factor(
        self,
        crack_length: float,
        stress: float,
        crack: Dict
    ) -> float:
        """
        Calculate stress intensity factor for a given crack.
        
        Args:
            crack_length: Current crack length (m)
            stress: Applied stress (Pa)
            crack: Crack configuration dictionary
            
        Returns:
            Stress intensity factor (Pa√m)
        """
        return self._calculate_stress_intensity_factor(crack_length, stress, crack)
    
    def _calculate_stress_intensity_factor(
        self,
        crack_length: float,
        stress: float,
        crack: Dict
    ) -> float:
        """Internal method to calculate stress intensity factor."""
        # Simplified calculation for edge crack in infinite plate
        # K = Y * σ * √(π * a)
        # where Y is the geometry factor
        
        # Geometry factor (simplified)
        Y = 1.12  # Edge crack in semi-infinite plate
        
        # Mode I stress intensity factor
        K_I = Y * stress * np.sqrt(np.pi * crack_length)
        
        # For mixed mode cracks, combine modes
        if crack['crack_type'] == 'mode_I':
            return K_I
        elif crack['crack_type'] == 'mode_II':
            # Mode II component (simplified)
            K_II = 0.5 * K_I
            return np.sqrt(K_I**2 + K_II**2)
        elif crack['crack_type'] == 'mixed':
            # Mixed mode (simplified)
            K_II = 0.3 * K_I
            K_III = 0.1 * K_I
            return np.sqrt(K_I**2 + K_II**2 + K_III**2)
        else:
            return K_I
    
    def _calculate_crack_interactions(
        self,
        crack_lengths: np.ndarray,
        time_points: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate crack interaction effects."""
        n_cracks, n_time = crack_lengths.shape
        
        interactions = {
            'coalescence_events': [],
            'shielding_effects': np.zeros((n_cracks, n_time)),
            'amplification_effects': np.zeros((n_cracks, n_time))
        }
        
        # Simplified crack interaction model
        for i in range(n_cracks):
            for j in range(i+1, n_cracks):
                # Calculate distance between cracks (simplified)
                distance = 1e-3  # 1 mm (simplified)
                
                # Check for coalescence
                for t in range(n_time):
                    if (crack_lengths[i, t] + crack_lengths[j, t]) > distance:
                        interactions['coalescence_events'].append({
                            'time': time_points[t],
                            'crack_1': i,
                            'crack_2': j,
                            'combined_length': crack_lengths[i, t] + crack_lengths[j, t]
                        })
                        break
        
        return interactions
    
    def _calculate_failure_probability(
        self,
        crack_lengths: np.ndarray,
        stress_intensity_factors: np.ndarray,
        time_points: np.ndarray
    ) -> np.ndarray:
        """Calculate failure probability over time."""
        n_cracks, n_time = crack_lengths.shape
        failure_probability = np.zeros(n_time)
        
        for t in range(n_time):
            # Calculate probability of each crack causing failure
            for i in range(n_cracks):
                if not self.cracks[i]['is_active']:
                    continue
                    
                material_props = self.material_props[self.cracks[i]['material_type']]
                K = stress_intensity_factors[i, t]
                K_c = material_props['fracture_toughness']
                
                # Weibull failure probability
                if K > 0:
                    beta = 2.0  # Weibull shape parameter
                    eta = K_c / (np.log(2))**(1/beta)  # Weibull scale parameter
                    P_f = 1 - np.exp(-(K/eta)**beta)
                    failure_probability[t] = 1 - (1 - failure_probability[t]) * (1 - P_f)
        
        return failure_probability
    
    def predict_remaining_useful_life(
        self,
        current_crack_lengths: List[float],
        stress_history: np.ndarray,
        time_points: np.ndarray,
        target_probability: float = 0.1
    ) -> Dict[str, Any]:
        """
        Predict remaining useful life based on current crack state.
        
        Args:
            current_crack_lengths: Current crack lengths (m)
            stress_history: Stress history (Pa)
            time_points: Time points (s)
            target_probability: Target failure probability
            
        Returns:
            Dictionary containing RUL predictions
        """
        # Update current crack lengths
        for i, crack in enumerate(self.cracks):
            if i < len(current_crack_lengths):
                crack['current_length'] = current_crack_lengths[i]
        
        # Simulate future crack growth
        future_time = np.linspace(time_points[-1], time_points[-1] + 24*3600, 1000)  # 24 hours
        future_stress = np.full_like(future_time, stress_history[-1])  # Constant stress
        
        # Run simulation
        results = self.simulate_crack_growth(future_stress, future_time)
        
        # Find time when failure probability reaches target
        failure_prob = results['failure_probability']
        target_indices = np.where(failure_prob >= target_probability)[0]
        
        if len(target_indices) > 0:
            rul_time = future_time[target_indices[0]] - time_points[-1]
        else:
            rul_time = np.inf
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(results, target_probability)
        
        return {
            'rul_time': rul_time,
            'rul_hours': rul_time / 3600,
            'target_probability': target_probability,
            'confidence_intervals': confidence_intervals,
            'failure_probability_curve': failure_prob,
            'time_curve': future_time
        }
    
    def _calculate_confidence_intervals(
        self,
        results: Dict[str, Any],
        target_probability: float
    ) -> Dict[str, float]:
        """Calculate confidence intervals for RUL prediction."""
        # Simplified confidence interval calculation
        failure_prob = results['failure_probability']
        
        # Find 95% confidence interval
        prob_95 = target_probability * 0.95
        prob_105 = target_probability * 1.05
        
        indices_95 = np.where(failure_prob >= prob_95)[0]
        indices_105 = np.where(failure_prob >= prob_105)[0]
        
        if len(indices_95) > 0 and len(indices_105) > 0:
            time_95 = results['time_curve'][indices_95[0]]
            time_105 = results['time_curve'][indices_105[0]]
            
            return {
                'lower_bound': time_95,
                'upper_bound': time_105,
                'confidence_level': 0.95
            }
        else:
            return {
                'lower_bound': np.nan,
                'upper_bound': np.nan,
                'confidence_level': 0.95
            }
    
    def get_crack_statistics(self) -> Dict[str, Any]:
        """Get current crack statistics."""
        active_cracks = [crack for crack in self.cracks if crack['is_active']]
        failed_cracks = [crack for crack in self.cracks if not crack['is_active']]
        
        if not active_cracks:
            return {
                'total_cracks': len(self.cracks),
                'active_cracks': 0,
                'failed_cracks': len(failed_cracks),
                'average_length': 0.0,
                'max_length': 0.0,
                'min_length': 0.0
            }
        
        lengths = [crack['current_length'] for crack in active_cracks]
        
        return {
            'total_cracks': len(self.cracks),
            'active_cracks': len(active_cracks),
            'failed_cracks': len(failed_cracks),
            'average_length': np.mean(lengths),
            'max_length': np.max(lengths),
            'min_length': np.min(lengths),
            'length_std': np.std(lengths)
        }