#!/usr/bin/env python3
"""
Acoustic Models for Stratified Flow Analysis
Implements various acoustic propagation models for stratified media
"""

import numpy as np
from scipy import special
from scipy.integrate import quad
import matplotlib.pyplot as plt

class AcousticPropagationModels:
    """
    Collection of acoustic propagation models for stratified flows
    """
    
    def __init__(self, fluid_properties):
        """
        Initialize with fluid properties
        
        Parameters:
        -----------
        fluid_properties : dict
            Dictionary containing fluid properties (density, sound speed, viscosity)
        """
        self.rho1 = fluid_properties['density_1']
        self.rho2 = fluid_properties['density_2']
        self.c1 = fluid_properties['sound_speed_1']
        self.c2 = fluid_properties['sound_speed_2']
        self.mu1 = fluid_properties['viscosity_1']
        self.mu2 = fluid_properties['viscosity_2']
    
    def woods_equation(self, volume_fraction_1):
        """
        Wood's equation for effective sound speed in two-phase media
        
        Parameters:
        -----------
        volume_fraction_1 : float or array
            Volume fraction of fluid 1
            
        Returns:
        --------
        effective_sound_speed : float or array
            Effective sound speed
        """
        alpha1 = volume_fraction_1
        alpha2 = 1 - alpha1
        
        # Wood's equation: 1/c_eff² = α₁/c₁² + α₂/c₂²
        c_eff_squared = 1.0 / (alpha1/self.c1**2 + alpha2/self.c2**2)
        return np.sqrt(c_eff_squared)
    
    def modified_woods_equation(self, volume_fraction_1, frequency, interface_thickness=0.01):
        """
        Modified Wood's equation accounting for interface effects
        
        Parameters:
        -----------
        volume_fraction_1 : float or array
            Volume fraction of fluid 1
        frequency : float
            Acoustic frequency
        interface_thickness : float
            Thickness of the interface layer
            
        Returns:
        --------
        effective_sound_speed : float or array
            Effective sound speed with interface effects
        """
        # Base Wood's equation
        c_eff_base = self.woods_equation(volume_fraction_1)
        
        # Interface correction factor
        k = 2 * np.pi * frequency / c_eff_base
        interface_correction = 1.0 / (1.0 + 1j * k * interface_thickness)
        
        return c_eff_base * np.real(interface_correction)
    
    def attenuation_coefficient_interface(self, frequency, volume_fraction_1, interface_roughness=1e-6):
        """
        Calculate attenuation coefficient due to interface scattering
        
        Parameters:
        -----------
        frequency : float
            Acoustic frequency
        volume_fraction_1 : float
            Volume fraction of fluid 1
        interface_roughness : float
            RMS roughness of the interface
            
        Returns:
        --------
        attenuation : float
            Attenuation coefficient in Np/m
        """
        # Interface scattering attenuation
        k = 2 * np.pi * frequency / self.woods_equation(volume_fraction_1)
        sigma = interface_roughness
        
        # Scattering coefficient (simplified model)
        scattering_coeff = 0.5 * (k * sigma)**2
        
        return scattering_coeff
    
    def attenuation_coefficient_turbulence(self, frequency, turbulence_intensity, length_scale):
        """
        Calculate attenuation coefficient due to turbulence
        
        Parameters:
        -----------
        frequency : float
            Acoustic frequency
        turbulence_intensity : float
            Turbulence intensity (0-1)
        length_scale : float
            Turbulence length scale
            
        Returns:
        --------
        attenuation : float
            Attenuation coefficient in Np/m
        """
        # Turbulence scattering attenuation
        k = 2 * np.pi * frequency / self.c1
        Tu = turbulence_intensity
        L = length_scale
        
        # Scattering coefficient for turbulence
        scattering_coeff = 0.1 * Tu**2 * (k * L)**(4/3)
        
        return scattering_coeff
    
    def attenuation_coefficient_viscous(self, frequency, temperature=293.15):
        """
        Calculate viscous attenuation coefficient
        
        Parameters:
        -----------
        frequency : float
            Acoustic frequency
        temperature : float
            Temperature in Kelvin
            
        Returns:
        --------
        attenuation : float
            Viscous attenuation coefficient in Np/m
        """
        # Viscous attenuation (Stokes-Kirchhoff theory)
        omega = 2 * np.pi * frequency
        c = self.c1
        
        # Viscosity effects
        mu_eff = 0.5 * (self.mu1 + self.mu2)
        alpha_viscous = (omega**2 * mu_eff) / (2 * self.rho1 * c**3)
        
        return alpha_viscous
    
    def total_attenuation_coefficient(self, frequency, volume_fraction_1, 
                                    turbulence_intensity=0.05, length_scale=0.01,
                                    interface_roughness=1e-6, temperature=293.15):
        """
        Calculate total attenuation coefficient combining all mechanisms
        
        Parameters:
        -----------
        frequency : float
            Acoustic frequency
        volume_fraction_1 : float
            Volume fraction of fluid 1
        turbulence_intensity : float
            Turbulence intensity
        length_scale : float
            Turbulence length scale
        interface_roughness : float
            Interface roughness
        temperature : float
            Temperature
            
        Returns:
        --------
        total_attenuation : float
            Total attenuation coefficient in Np/m
        """
        # Individual attenuation mechanisms
        alpha_interface = self.attenuation_coefficient_interface(
            frequency, volume_fraction_1, interface_roughness)
        alpha_turbulence = self.attenuation_coefficient_turbulence(
            frequency, turbulence_intensity, length_scale)
        alpha_viscous = self.attenuation_coefficient_viscous(frequency, temperature)
        
        # Total attenuation (additive)
        total_alpha = alpha_interface + alpha_turbulence + alpha_viscous
        
        return total_alpha
    
    def transfer_matrix_method(self, frequency, layer_thicknesses, layer_properties):
        """
        Transfer matrix method for wave propagation through stratified layers
        
        Parameters:
        -----------
        frequency : float
            Acoustic frequency
        layer_thicknesses : array
            Thickness of each layer
        layer_properties : list of dict
            Properties of each layer (density, sound speed, viscosity)
            
        Returns:
        --------
        transmission_coefficient : complex
            Complex transmission coefficient
        reflection_coefficient : complex
            Complex reflection coefficient
        """
        omega = 2 * np.pi * frequency
        
        # Initialize transfer matrix as identity
        T = np.eye(2, dtype=complex)
        
        for i, (thickness, props) in enumerate(zip(layer_thicknesses, layer_properties)):
            rho = props['density']
            c = props['sound_speed']
            mu = props.get('viscosity', 0)
            
            # Wave number
            k = omega / c
            
            # Impedance
            Z = rho * c
            
            # Viscous correction
            if mu > 0:
                k = k * (1 + 1j * mu * omega / (2 * rho * c**2))
            
            # Layer transfer matrix
            T_layer = np.array([
                [np.cos(k * thickness), 1j * np.sin(k * thickness) / Z],
                [1j * Z * np.sin(k * thickness), np.cos(k * thickness)]
            ])
            
            T = T @ T_layer
        
        # Calculate reflection and transmission coefficients
        Z0 = layer_properties[0]['density'] * layer_properties[0]['sound_speed']
        Zf = layer_properties[-1]['density'] * layer_properties[-1]['sound_speed']
        
        # Transmission coefficient
        T_coeff = 2 * Zf / (T[0, 0] * Zf + T[0, 1] + T[1, 0] * Z0 * Zf + T[1, 1] * Z0)
        
        # Reflection coefficient
        R_coeff = (T[0, 0] * Zf + T[0, 1] - T[1, 0] * Z0 * Zf - T[1, 1] * Z0) / \
                  (T[0, 0] * Zf + T[0, 1] + T[1, 0] * Z0 * Zf + T[1, 1] * Z0)
        
        return T_coeff, R_coeff
    
    def lighthill_analogy(self, velocity_field, density_field, source_position, receiver_position):
        """
        Lighthill acoustic analogy for sound generation in flows
        
        Parameters:
        -----------
        velocity_field : array
            3D velocity field
        density_field : array
            3D density field
        source_position : array
            Position of acoustic source
        receiver_position : array
            Position of receiver
            
        Returns:
        --------
        acoustic_pressure : float
            Acoustic pressure at receiver
        """
        # Lighthill stress tensor T_ij = ρ u_i u_j + p δ_ij - c² ρ δ_ij
        # Simplified implementation
        
        # Distance from source to receiver
        r = np.linalg.norm(np.array(receiver_position) - np.array(source_position))
        
        # Lighthill source strength (simplified)
        source_strength = 1.0  # Placeholder - would be calculated from flow field
        
        # Acoustic pressure (simplified Green's function solution)
        c = self.c1
        acoustic_pressure = source_strength / (4 * np.pi * r)
        
        return acoustic_pressure
    
    def fwh_equation(self, surface_data, observer_position, time):
        """
        Ffowcs Williams-Hawkings equation for moving surfaces
        
        Parameters:
        -----------
        surface_data : dict
            Surface data including position, velocity, pressure
        observer_position : array
            Position of observer
        time : float
            Time instant
            
        Returns:
        --------
        acoustic_pressure : float
            Acoustic pressure at observer
        """
        # FWH equation implementation (simplified)
        # This would typically involve surface integration
        
        # Distance from surface to observer
        r = np.linalg.norm(np.array(observer_position) - np.array(surface_data['position']))
        
        # Retarded time
        t_ret = time - r / self.c1
        
        # FWH source terms (simplified)
        monopole_term = 0.0  # Would be calculated from surface motion
        dipole_term = 0.0    # Would be calculated from surface forces
        quadrupole_term = 0.0  # Would be calculated from volume sources
        
        acoustic_pressure = monopole_term + dipole_term + quadrupole_term
        
        return acoustic_pressure

def generate_acoustic_analysis_data():
    """
    Generate comprehensive acoustic analysis data for the thesis
    """
    # Fluid properties
    fluid_props = {
        'density_1': 1000.0,    # kg/m³ (water)
        'density_2': 1.2,       # kg/m³ (air)
        'sound_speed_1': 1500.0, # m/s
        'sound_speed_2': 343.0,  # m/s
        'viscosity_1': 1e-3,    # Pa·s
        'viscosity_2': 1.8e-5   # Pa·s
    }
    
    # Initialize acoustic models
    acoustic = AcousticPropagationModels(fluid_props)
    
    # Generate frequency sweep data
    frequencies = np.logspace(2, 4, 100)  # 100 Hz to 10 kHz
    volume_fractions = np.linspace(0, 1, 50)  # 0 to 1
    
    # Generate data arrays
    effective_sound_speeds = np.zeros((len(frequencies), len(volume_fractions)))
    attenuation_coefficients = np.zeros((len(frequencies), len(volume_fractions)))
    
    for i, freq in enumerate(frequencies):
        for j, vf in enumerate(volume_fractions):
            # Effective sound speed
            effective_sound_speeds[i, j] = acoustic.woods_equation(vf)
            
            # Attenuation coefficient
            attenuation_coefficients[i, j] = acoustic.total_attenuation_coefficient(
                freq, vf, turbulence_intensity=0.05, length_scale=0.01)
    
    # Generate transfer matrix data
    layer_properties = [
        {'density': 1000.0, 'sound_speed': 1500.0, 'viscosity': 1e-3},
        {'density': 1.2, 'sound_speed': 343.0, 'viscosity': 1.8e-5}
    ]
    layer_thicknesses = [0.5, 0.5]  # meters
    
    transmission_data = []
    reflection_data = []
    
    for freq in frequencies:
        T_coeff, R_coeff = acoustic.transfer_matrix_method(
            freq, layer_thicknesses, layer_properties)
        transmission_data.append(abs(T_coeff)**2)
        reflection_data.append(abs(R_coeff)**2)
    
    return {
        'frequencies': frequencies,
        'volume_fractions': volume_fractions,
        'effective_sound_speeds': effective_sound_speeds,
        'attenuation_coefficients': attenuation_coefficients,
        'transmission_coefficients': transmission_data,
        'reflection_coefficients': reflection_data,
        'acoustic_models': acoustic
    }

if __name__ == "__main__":
    # Generate acoustic analysis data
    data = generate_acoustic_analysis_data()
    
    # Save data
    np.savez('acoustic_analysis_data.npz', **data)
    print("Acoustic analysis data generated and saved to acoustic_analysis_data.npz")