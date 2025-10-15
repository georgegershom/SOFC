"""
Creep Models for SOFC Materials

Implements various creep constitutive models for time-dependent deformation analysis.
Based on the research article data for 8YSZ creep behavior.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class CreepParameters:
    """Container for creep model parameters"""
    B: float  # Pre-exponential factor
    n: float  # Stress exponent
    Q: float  # Activation energy (J/mol)
    R: float = 8.314  # Gas constant (J/mol·K)


class CreepModel(ABC):
    """Abstract base class for creep models"""
    
    @abstractmethod
    def strain_rate(self, stress: float, temperature: float, time: float = 0.0) -> float:
        """Calculate creep strain rate"""
        pass
    
    @abstractmethod
    def accumulated_strain(self, stress: float, temperature: float, time: float) -> float:
        """Calculate accumulated creep strain"""
        pass


class NortonBaileyCreep(CreepModel):
    """
    Norton-Bailey creep model for 8YSZ
    
    Based on experimental data from the research article:
    B = 8.5e-12 s⁻¹ MPa⁻ⁿ
    n = 1.8
    Q = 385 kJ/mol
    """
    
    def __init__(self, parameters: Optional[CreepParameters] = None):
        if parameters is None:
            # Default parameters for 8YSZ from research article
            self.params = CreepParameters(
                B=8.5e-12,  # s⁻¹ MPa⁻ⁿ
                n=1.8,
                Q=385e3  # J/mol
            )
        else:
            self.params = parameters
    
    def strain_rate(self, stress: float, temperature: float, time: float = 0.0) -> float:
        """
        Calculate creep strain rate using Norton-Bailey law
        
        ε̇_cr = B * σ^n * exp(-Q/RT)
        
        Args:
            stress: Equivalent stress (MPa)
            temperature: Absolute temperature (K)
            time: Time (s) - not used in this model
            
        Returns:
            Creep strain rate (s⁻¹)
        """
        if stress <= 0 or temperature <= 0:
            return 0.0
        
        # Convert temperature to Kelvin if in Celsius
        if temperature < 1000:
            temperature += 273.15
        
        # Norton-Bailey equation
        strain_rate = (self.params.B * 
                      (stress ** self.params.n) * 
                      np.exp(-self.params.Q / (self.params.R * temperature)))
        
        return strain_rate
    
    def accumulated_strain(self, stress: float, temperature: float, time: float) -> float:
        """
        Calculate accumulated creep strain assuming constant stress
        
        ε_cr = ε̇_cr * t
        
        Args:
            stress: Equivalent stress (MPa)
            temperature: Absolute temperature (K)
            time: Time (s)
            
        Returns:
            Accumulated creep strain
        """
        strain_rate = self.strain_rate(stress, temperature, time)
        return strain_rate * time
    
    def stress_relaxation_factor(self, stress: float, temperature: float, time: float) -> float:
        """
        Calculate stress relaxation factor due to creep
        
        This is a simplified approach - in reality, stress relaxation
        requires solving the coupled creep-stress problem
        
        Args:
            stress: Initial stress (MPa)
            temperature: Absolute temperature (K)
            time: Time (s)
            
        Returns:
            Stress relaxation factor (0-1)
        """
        # Simplified stress relaxation based on accumulated strain
        accumulated_strain = self.accumulated_strain(stress, temperature, time)
        
        # Assume linear stress-strain relationship for relaxation
        # This is a simplification - actual relaxation is nonlinear
        relaxation_factor = 1.0 - min(accumulated_strain * 0.1, 0.3)  # Max 30% relaxation
        
        return max(relaxation_factor, 0.7)  # Minimum 70% of original stress


class PowerLawCreep(CreepModel):
    """
    General power law creep model
    
    ε̇_cr = A * σ^n * exp(-Q/RT)
    """
    
    def __init__(self, A: float, n: float, Q: float):
        self.A = A
        self.n = n
        self.Q = Q
        self.R = 8.314  # Gas constant
    
    def strain_rate(self, stress: float, temperature: float, time: float = 0.0) -> float:
        """Calculate creep strain rate"""
        if stress <= 0 or temperature <= 0:
            return 0.0
        
        # Convert temperature to Kelvin if in Celsius
        if temperature < 1000:
            temperature += 273.15
        
        strain_rate = (self.A * 
                      (stress ** self.n) * 
                      np.exp(-self.Q / (self.R * temperature)))
        
        return strain_rate
    
    def accumulated_strain(self, stress: float, temperature: float, time: float) -> float:
        """Calculate accumulated creep strain"""
        strain_rate = self.strain_rate(stress, temperature, time)
        return strain_rate * time


class ViscoelasticModel:
    """
    Viscoelastic model combining elastic and creep behavior
    
    Total strain rate: ε̇_total = ε̇_elastic + ε̇_thermal + ε̇_creep
    """
    
    def __init__(self, youngs_modulus: float, poisson_ratio: float, 
                 cte: float, creep_model: CreepModel):
        self.E = youngs_modulus
        self.nu = poisson_ratio
        self.cte = cte
        self.creep_model = creep_model
    
    def elastic_strain_rate(self, stress_rate: float) -> float:
        """Calculate elastic strain rate"""
        return stress_rate / self.E
    
    def thermal_strain_rate(self, temperature_rate: float) -> float:
        """Calculate thermal strain rate"""
        return self.cte * temperature_rate
    
    def creep_strain_rate(self, stress: float, temperature: float, time: float) -> float:
        """Calculate creep strain rate"""
        return self.creep_model.strain_rate(stress, temperature, time)
    
    def total_strain_rate(self, stress: float, stress_rate: float, 
                         temperature: float, temperature_rate: float, 
                         time: float) -> float:
        """Calculate total strain rate"""
        elastic = self.elastic_strain_rate(stress_rate)
        thermal = self.thermal_strain_rate(temperature_rate)
        creep = self.creep_strain_rate(stress, temperature, time)
        
        return elastic + thermal + creep


def create_8ysz_creep_model() -> NortonBaileyCreep:
    """Create 8YSZ creep model with literature parameters"""
    return NortonBaileyCreep()


def create_viscoelastic_8ysz(youngs_modulus: float, poisson_ratio: float, 
                           cte: float) -> ViscoelasticModel:
    """Create viscoelastic model for 8YSZ"""
    creep_model = create_8ysz_creep_model()
    return ViscoelasticModel(youngs_modulus, poisson_ratio, cte, creep_model)


if __name__ == "__main__":
    # Example usage
    creep_model = create_8ysz_creep_model()
    
    # Test at different stress levels and temperatures
    stress_levels = [50, 100, 150]  # MPa
    temperatures = [700, 750, 800]  # °C
    time = 3600  # 1 hour in seconds
    
    print("8YSZ Creep Analysis:")
    print("Stress (MPa) | Temperature (°C) | Strain Rate (s⁻¹) | Accumulated Strain")
    print("-" * 70)
    
    for stress in stress_levels:
        for temp in temperatures:
            strain_rate = creep_model.strain_rate(stress, temp)
            accumulated = creep_model.accumulated_strain(stress, temp, time)
            print(f"{stress:11.0f} | {temp:13.0f} | {strain_rate:15.2e} | {accumulated:16.2e}")