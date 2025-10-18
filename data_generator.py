"""
Main data generator for rubberized concrete experimental dataset
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from config import *

class ConcreteDataGenerator:
    def __init__(self, seed: int = 42):
        """Initialize the data generator with random seed for reproducibility"""
        np.random.seed(seed)
        random.seed(seed)
        self.data = []
        
    def generate_ambient_data(self) -> pd.DataFrame:
        """Generate ambient condition test data for all mix designs and curing ages"""
        ambient_data = []
        
        for mix_name, mix_props in MIX_DESIGNS.items():
            for age in CURING_AGES:
                for specimen_id in range(SPECIMENS_PER_CONDITION):
                    # Calculate age-dependent strength development
                    age_factor = self._calculate_age_factor(age)
                    
                    # Calculate rubber content effects
                    rubber_factor = self._calculate_rubber_effects(mix_props['rubber_content'])
                    
                    # Generate base properties
                    base_compressive = BASE_PROPERTIES['compressive_strength_28d'] * age_factor * rubber_factor['compressive']
                    base_tensile = BASE_PROPERTIES['tensile_strength_28d'] * age_factor * rubber_factor['tensile']
                    base_flexural = BASE_PROPERTIES['flexural_strength_28d'] * age_factor * rubber_factor['tensile']
                    base_modulus = BASE_PROPERTIES['modulus_elasticity_28d'] * age_factor * rubber_factor['modulus']
                    
                    # Add statistical variation
                    compressive_strength = self._add_variation(base_compressive, 'compressive_strength')
                    tensile_strength = self._add_variation(base_tensile, 'tensile_strength')
                    flexural_strength = self._add_variation(base_flexural, 'flexural_strength')
                    modulus_elasticity = self._add_variation(base_modulus, 'modulus_elasticity')
                    
                    # Calculate density and UPV
                    density = self._calculate_density(mix_props['rubber_content'])
                    upv = self._calculate_upv(density, compressive_strength)
                    
                    # Calculate Poisson's ratio (slightly affected by rubber content)
                    poisson_ratio = BASE_PROPERTIES['poisson_ratio'] * (1 + 0.1 * mix_props['rubber_content'] / 100)
                    
                    ambient_data.append({
                        'test_type': 'ambient',
                        'mix_design': mix_name,
                        'curing_age_days': age,
                        'specimen_id': specimen_id + 1,
                        'temperature_c': 23,
                        'cooling_regime': 'N/A',
                        'compressive_strength_mpa': compressive_strength,
                        'tensile_strength_mpa': tensile_strength,
                        'flexural_strength_mpa': flexural_strength,
                        'modulus_elasticity_mpa': modulus_elasticity,
                        'density_kg_m3': density,
                        'upv_m_s': upv,
                        'poisson_ratio': poisson_ratio,
                        'mass_loss_percent': 0.0,
                        'spalling_depth_mm': 0.0,
                        'thermal_expansion_coeff': BASE_PROPERTIES['thermal_expansion_coeff'] * (1 + rubber_factor['thermal_expansion']),
                        'w_c_ratio': mix_props['w_c_ratio'],
                        'cement_content_kg_m3': mix_props['cement_content'],
                        'rubber_content_percent': mix_props['rubber_content'],
                        'rubber_size_mm': mix_props['rubber_size'],
                        'admixtures': mix_props['admixtures']
                    })
        
        return pd.DataFrame(ambient_data)
    
    def generate_thermal_data(self) -> pd.DataFrame:
        """Generate high-temperature exposure test data"""
        thermal_data = []
        
        for mix_name, mix_props in MIX_DESIGNS.items():
            for temp in TEMPERATURE_LEVELS:
                for cooling in COOLING_REGIMES:
                    for specimen_id in range(SPECIMENS_PER_CONDITION):
                        # Calculate temperature effects
                        temp_factor = self._calculate_temperature_effects(temp, cooling)
                        
                        # Calculate rubber content effects
                        rubber_factor = self._calculate_rubber_effects(mix_props['rubber_content'])
                        
                        # Generate residual properties after thermal exposure
                        base_compressive = BASE_PROPERTIES['compressive_strength_28d'] * temp_factor['compressive'] * rubber_factor['compressive']
                        base_tensile = BASE_PROPERTIES['tensile_strength_28d'] * temp_factor['tensile'] * rubber_factor['tensile']
                        base_flexural = BASE_PROPERTIES['flexural_strength_28d'] * temp_factor['tensile'] * rubber_factor['tensile']
                        base_modulus = BASE_PROPERTIES['modulus_elasticity_28d'] * temp_factor['modulus'] * rubber_factor['modulus']
                        
                        # Add statistical variation
                        compressive_strength = self._add_variation(base_compressive, 'compressive_strength')
                        tensile_strength = self._add_variation(base_tensile, 'tensile_strength')
                        flexural_strength = self._add_variation(base_flexural, 'flexural_strength')
                        modulus_elasticity = self._add_variation(base_modulus, 'modulus_elasticity')
                        
                        # Calculate post-thermal properties
                        density = self._calculate_density(mix_props['rubber_content']) * temp_factor['density']
                        upv = self._calculate_upv(density, compressive_strength) * temp_factor['upv']
                        mass_loss = self._calculate_mass_loss(temp, cooling, mix_props['rubber_content'])
                        spalling_depth = self._calculate_spalling(temp, cooling, mix_props['rubber_content'])
                        
                        # Calculate thermal expansion coefficient
                        thermal_expansion = self._calculate_thermal_expansion(temp, mix_props['rubber_content'])
                        
                        thermal_data.append({
                            'test_type': 'thermal_exposure',
                            'mix_design': mix_name,
                            'curing_age_days': 28,  # Standard age for thermal tests
                            'specimen_id': specimen_id + 1,
                            'temperature_c': temp,
                            'cooling_regime': cooling,
                            'compressive_strength_mpa': compressive_strength,
                            'tensile_strength_mpa': tensile_strength,
                            'flexural_strength_mpa': flexural_strength,
                            'modulus_elasticity_mpa': modulus_elasticity,
                            'density_kg_m3': density,
                            'upv_m_s': upv,
                            'poisson_ratio': BASE_PROPERTIES['poisson_ratio'] * temp_factor['poisson'],
                            'mass_loss_percent': mass_loss,
                            'spalling_depth_mm': spalling_depth,
                            'thermal_expansion_coeff': thermal_expansion,
                            'w_c_ratio': mix_props['w_c_ratio'],
                            'cement_content_kg_m3': mix_props['cement_content'],
                            'rubber_content_percent': mix_props['rubber_content'],
                            'rubber_size_mm': mix_props['rubber_size'],
                            'admixtures': mix_props['admixtures']
                        })
        
        return pd.DataFrame(thermal_data)
    
    def generate_insitu_data(self) -> pd.DataFrame:
        """Generate in-situ high-temperature test data"""
        insitu_data = []
        
        for mix_name, mix_props in MIX_DESIGNS.items():
            for temp in TEMPERATURE_LEVELS[1:]:  # Skip ambient temperature
                for specimen_id in range(SPECIMENS_PER_CONDITION):
                    # In-situ properties (tested at temperature)
                    temp_factor = self._calculate_insitu_temperature_effects(temp)
                    rubber_factor = self._calculate_rubber_effects(mix_props['rubber_content'])
                    
                    # Generate in-situ properties
                    insitu_compressive = BASE_PROPERTIES['compressive_strength_28d'] * temp_factor['compressive'] * rubber_factor['compressive']
                    insitu_modulus = BASE_PROPERTIES['modulus_elasticity_28d'] * temp_factor['modulus'] * rubber_factor['modulus']
                    
                    # Add statistical variation
                    compressive_strength = self._add_variation(insitu_compressive, 'compressive_strength')
                    modulus_elasticity = self._add_variation(insitu_modulus, 'modulus_elasticity')
                    
                    # Calculate transient thermal strain
                    thermal_strain = self._calculate_thermal_strain(temp, mix_props['rubber_content'])
                    
                    # Calculate pore pressure (simulated)
                    pore_pressure = self._calculate_pore_pressure(temp, mix_props['rubber_content'])
                    
                    insitu_data.append({
                        'test_type': 'insitu_thermal',
                        'mix_design': mix_name,
                        'curing_age_days': 28,
                        'specimen_id': specimen_id + 1,
                        'temperature_c': temp,
                        'cooling_regime': 'N/A',
                        'compressive_strength_mpa': compressive_strength,
                        'tensile_strength_mpa': np.nan,
                        'flexural_strength_mpa': np.nan,
                        'modulus_elasticity_mpa': modulus_elasticity,
                        'density_kg_m3': np.nan,
                        'upv_m_s': np.nan,
                        'poisson_ratio': BASE_PROPERTIES['poisson_ratio'] * temp_factor['poisson'],
                        'mass_loss_percent': np.nan,
                        'spalling_depth_mm': np.nan,
                        'thermal_expansion_coeff': np.nan,
                        'thermal_strain_microstrain': thermal_strain,
                        'pore_pressure_mpa': pore_pressure,
                        'w_c_ratio': mix_props['w_c_ratio'],
                        'cement_content_kg_m3': mix_props['cement_content'],
                        'rubber_content_percent': mix_props['rubber_content'],
                        'rubber_size_mm': mix_props['rubber_size'],
                        'admixtures': mix_props['admixtures']
                    })
        
        return pd.DataFrame(insitu_data)
    
    def _calculate_age_factor(self, age: int) -> float:
        """Calculate strength development factor based on curing age"""
        if age <= 28:
            return 0.4 + 0.6 * (age / 28)
        else:
            return 1.0 + 0.1 * np.log(age / 28)
    
    def _calculate_rubber_effects(self, rubber_content: float) -> Dict[str, float]:
        """Calculate the effects of rubber content on material properties"""
        if rubber_content == 0:
            return {key: 1.0 for key in ['compressive', 'tensile', 'modulus', 'thermal_expansion']}
        
        return {
            'compressive': 1.0 + (rubber_content / 100) * RUBBER_EFFECTS['compressive_strength'],
            'tensile': 1.0 + (rubber_content / 100) * RUBBER_EFFECTS['tensile_strength'],
            'modulus': 1.0 + (rubber_content / 100) * RUBBER_EFFECTS['modulus_elasticity'],
            'thermal_expansion': 1.0 + (rubber_content / 100) * RUBBER_EFFECTS['thermal_expansion']
        }
    
    def _calculate_temperature_effects(self, temp: float, cooling: str) -> Dict[str, float]:
        """Calculate temperature effects on material properties"""
        if temp == 23:
            return {key: 1.0 for key in ['compressive', 'tensile', 'modulus', 'density', 'upv', 'poisson']}
        
        # Base temperature effects
        comp_factor = TEMP_REDUCTION_FACTORS['compressive_strength'].get(temp, 0.1)
        tensile_factor = TEMP_REDUCTION_FACTORS['tensile_strength'].get(temp, 0.05)
        modulus_factor = TEMP_REDUCTION_FACTORS['modulus_elasticity'].get(temp, 0.05)
        
        # Cooling regime effects
        if cooling == 'water_quenching':
            comp_factor *= 0.8  # Additional damage from thermal shock
            tensile_factor *= 0.7
            modulus_factor *= 0.8
        
        return {
            'compressive': comp_factor,
            'tensile': tensile_factor,
            'modulus': modulus_factor,
            'density': 1.0 - (temp - 23) * 0.0001,  # Slight density reduction
            'upv': comp_factor ** 0.5,  # UPV correlates with strength
            'poisson': 1.0 + (temp - 23) * 0.0001  # Slight increase in Poisson's ratio
        }
    
    def _calculate_insitu_temperature_effects(self, temp: float) -> Dict[str, float]:
        """Calculate in-situ temperature effects (different from residual effects)"""
        if temp == 23:
            return {key: 1.0 for key in ['compressive', 'modulus', 'poisson']}
        
        # In-situ properties are generally higher than residual properties
        comp_factor = min(1.0, 0.9 - (temp - 23) * 0.0008)
        modulus_factor = min(1.0, 0.95 - (temp - 23) * 0.0006)
        
        return {
            'compressive': comp_factor,
            'modulus': modulus_factor,
            'poisson': 1.0 + (temp - 23) * 0.0002
        }
    
    def _add_variation(self, value: float, property_type: str) -> float:
        """Add statistical variation to a property value"""
        cov = COEFFICIENT_OF_VARIATION.get(property_type, 0.05)
        variation = np.random.normal(0, cov * value)
        return max(0, value + variation)
    
    def _calculate_density(self, rubber_content: float) -> float:
        """Calculate concrete density considering rubber content"""
        base_density = BASE_PROPERTIES['density']
        rubber_density = 1200  # kg/m³ (typical rubber density)
        aggregate_density = 2650  # kg/m³ (typical aggregate density)
        
        # Calculate density based on volume replacement
        rubber_vol = rubber_content / 100
        aggregate_vol = 1 - rubber_vol
        
        density = base_density * (rubber_vol * rubber_density + aggregate_vol * aggregate_density) / base_density
        return self._add_variation(density, 'density')
    
    def _calculate_upv(self, density: float, strength: float) -> float:
        """Calculate ultrasonic pulse velocity based on density and strength"""
        # UPV correlates with both density and strength
        base_upv = BASE_PROPERTIES['upv_ambient']
        density_factor = (density / BASE_PROPERTIES['density']) ** 0.5
        strength_factor = (strength / BASE_PROPERTIES['compressive_strength_28d']) ** 0.3
        
        upv = base_upv * density_factor * strength_factor
        return self._add_variation(upv, 'upv')
    
    def _calculate_mass_loss(self, temp: float, cooling: str, rubber_content: float) -> float:
        """Calculate mass loss due to thermal exposure"""
        if temp == 23:
            return 0.0
        
        # Base mass loss from dehydration
        base_loss = (temp - 23) * 0.02  # 2% per 100°C
        
        # Additional loss from rubber degradation
        rubber_loss = rubber_content * (temp - 23) * 0.001
        
        # Additional loss from thermal shock
        shock_loss = 0.02 if cooling == 'water_quenching' else 0.0
        
        total_loss = base_loss + rubber_loss + shock_loss
        return min(15.0, total_loss)  # Cap at 15% mass loss
    
    def _calculate_spalling(self, temp: float, cooling: str, rubber_content: float) -> float:
        """Calculate spalling depth due to thermal exposure"""
        if temp < 400:
            return 0.0
        
        # Spalling increases with temperature
        base_spalling = (temp - 400) * 0.1  # 0.1 mm per 100°C above 400°C
        
        # Rubber content reduces spalling
        rubber_protection = rubber_content * 0.05  # 0.05 mm reduction per 1% rubber
        
        # Thermal shock increases spalling
        shock_spalling = 2.0 if cooling == 'water_quenching' else 0.0
        
        spalling = max(0, base_spalling - rubber_protection + shock_spalling)
        return spalling
    
    def _calculate_thermal_expansion(self, temp: float, rubber_content: float) -> float:
        """Calculate thermal expansion coefficient"""
        base_cte = BASE_PROPERTIES['thermal_expansion_coeff']
        rubber_effect = 1.0 + (rubber_content / 100) * RUBBER_EFFECTS['thermal_expansion']
        temp_effect = 1.0 + (temp - 23) * 0.0001  # Slight increase with temperature
        
        return base_cte * rubber_effect * temp_effect
    
    def _calculate_thermal_strain(self, temp: float, rubber_content: float) -> float:
        """Calculate transient thermal strain"""
        base_cte = self._calculate_thermal_expansion(temp, rubber_content)
        thermal_strain = base_cte * (temp - 23) * 1e6  # Convert to microstrain
        return thermal_strain + np.random.normal(0, 50)  # Add some variation
    
    def _calculate_pore_pressure(self, temp: float, rubber_content: float) -> float:
        """Calculate pore pressure during heating (simulated)"""
        if temp < 100:
            return 0.0
        
        # Pore pressure increases with temperature due to steam generation
        base_pressure = (temp - 100) * 0.01  # 0.01 MPa per 100°C above 100°C
        
        # Rubber content affects pore pressure (rubber can act as pressure relief)
        rubber_effect = 1.0 - (rubber_content / 100) * 0.3
        
        return base_pressure * rubber_effect
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """Generate the complete experimental dataset"""
        print("Generating ambient condition test data...")
        ambient_data = self.generate_ambient_data()
        
        print("Generating thermal exposure test data...")
        thermal_data = self.generate_thermal_data()
        
        print("Generating in-situ thermal test data...")
        insitu_data = self.generate_insitu_data()
        
        # Combine all data
        complete_data = pd.concat([ambient_data, thermal_data, insitu_data], ignore_index=True)
        
        print(f"Generated complete dataset with {len(complete_data)} records")
        return complete_data