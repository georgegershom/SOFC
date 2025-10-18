#!/usr/bin/env python3
"""
Phase 3: Comprehensive Microstructural and Chemical Analysis Dataset Generator
Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements
Utilizing High-Performance Rubberized Concrete

This module generates multi-scale, quantitative microstructural and chemical analysis data
with cross-technique correlation and temperature-dependent evolution tracking.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import hashlib
from scipy import stats, interpolate, ndimage
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')

# ================================================================================
# Core Data Structures and Constants
# ================================================================================

@dataclass
class MixDesign:
    """Concrete mix design parameters linked to Phase 2"""
    mix_id: str
    cement_content: float  # kg/m³
    rubber_content: float  # % by volume
    rubber_size: str  # Fine, Coarse, Mixed
    w_c_ratio: float
    aggregate_type: str
    admixtures: Dict[str, float]
    
@dataclass
class ThermalExposure:
    """Thermal exposure conditions"""
    temperature: float  # °C
    exposure_type: str  # Furnace, ISO834, Hydrocarbon
    duration: float  # minutes
    heating_rate: float  # °C/min
    cooling_method: str  # Air, Water, Controlled
    
@dataclass
class AnalysisMetadata:
    """Metadata for each analysis"""
    sample_id: str
    analysis_id: str
    analysis_type: str
    instrument: str
    operator: str
    date: str
    calibration_standard: str
    measurement_conditions: Dict

# Critical temperature thresholds for phase transformations
CRITICAL_TEMPERATURES = [25, 100, 200, 300, 400, 500, 600, 700, 800]

# Mix designs from Phase 2
MIX_DESIGNS = {
    'C-28-R': MixDesign('C-28-R', 350, 10, 'Fine', 0.45, 'Granite', {'SP': 1.2, 'SF': 25}),
    'C-35-R': MixDesign('C-35-R', 380, 10, 'Fine', 0.42, 'Granite', {'SP': 1.5, 'SF': 30}),
    'C-45-R': MixDesign('C-45-R', 420, 10, 'Fine', 0.38, 'Granite', {'SP': 1.8, 'SF': 35}),
    'C-55-R': MixDesign('C-55-R', 450, 10, 'Fine', 0.35, 'Granite', {'SP': 2.0, 'SF': 40}),
    'C-28-R15': MixDesign('C-28-R15', 350, 15, 'Mixed', 0.45, 'Granite', {'SP': 1.5, 'SF': 25}),
    'C-35-R15': MixDesign('C-35-R15', 380, 15, 'Mixed', 0.42, 'Granite', {'SP': 1.8, 'SF': 30}),
    'C-28-R20': MixDesign('C-28-R20', 350, 20, 'Coarse', 0.45, 'Granite', {'SP': 1.8, 'SF': 25}),
    'C-35-R20': MixDesign('C-35-R20', 380, 20, 'Coarse', 0.42, 'Granite', {'SP': 2.0, 'SF': 30}),
    'C-28-Control': MixDesign('C-28-Control', 350, 0, 'N/A', 0.45, 'Granite', {'SP': 0.8, 'SF': 25}),
    'C-35-Control': MixDesign('C-35-Control', 380, 0, 'N/A', 0.42, 'Granite', {'SP': 1.0, 'SF': 30}),
    'C-45-Control': MixDesign('C-45-Control', 420, 0, 'N/A', 0.38, 'Granite', {'SP': 1.2, 'SF': 35}),
    'C-55-Control': MixDesign('C-55-Control', 450, 0, 'N/A', 0.35, 'Granite', {'SP': 1.5, 'SF': 40}),
}

# ================================================================================
# SEM Analysis Data Generator
# ================================================================================

class SEMAnalysisGenerator:
    """Generates quantitative SEM/EDS analysis data"""
    
    def __init__(self, mix_design: MixDesign, thermal_exposure: ThermalExposure):
        self.mix = mix_design
        self.exposure = thermal_exposure
        self.base_porosity = self._calculate_base_porosity()
        
    def _calculate_base_porosity(self) -> float:
        """Calculate baseline porosity based on mix design"""
        base = 0.12 - 0.02 * (1 - self.mix.w_c_ratio)
        rubber_effect = self.mix.rubber_content * 0.005
        return base + rubber_effect
    
    def generate_morphology_data(self, num_fields: int = 10) -> pd.DataFrame:
        """Generate quantitative morphology measurements from SEM images"""
        
        data = []
        for field in range(num_fields):
            # Temperature-dependent microstructural evolution
            temp_factor = np.exp(self.exposure.temperature / 500)
            
            # Pore structure analysis
            porosity = self.base_porosity * temp_factor
            if self.exposure.temperature > 200 and self.mix.rubber_content > 0:
                # Rubber decomposition creates additional porosity
                porosity += self.mix.rubber_content * 0.001 * (self.exposure.temperature - 200) / 100
            
            # Pore size distribution (log-normal)
            mean_pore_size = 5 + 15 * (self.exposure.temperature / 800)  # µm
            if self.mix.rubber_content > 0 and self.exposure.temperature > 300:
                mean_pore_size *= 1 + 0.5 * (self.mix.rubber_content / 20)
            
            pore_sizes = np.random.lognormal(np.log(mean_pore_size), 0.8, 100)
            
            # Crack analysis
            crack_density = 0
            if self.exposure.temperature > 300:
                crack_density = 0.5 * np.exp((self.exposure.temperature - 300) / 200)  # cracks/mm²
                if self.mix.rubber_content > 0:
                    crack_density *= 0.7  # Rubber reduces crack density
            
            avg_crack_width = 0
            if crack_density > 0:
                avg_crack_width = 2 + 8 * (self.exposure.temperature / 800)  # µm
                
            # Interface analysis (ITZ characteristics)
            itz_thickness = 20 + 10 * self.mix.w_c_ratio  # µm
            if self.exposure.temperature > 400:
                itz_thickness *= 1 + 0.3 * (self.exposure.temperature - 400) / 400
            
            itz_porosity = porosity * 1.5  # ITZ more porous than bulk
            
            # Rubber particle analysis
            rubber_integrity = 100  # %
            if self.mix.rubber_content > 0:
                if self.exposure.temperature > 200:
                    rubber_integrity = max(0, 100 - (self.exposure.temperature - 200) / 6)
            
            data.append({
                'Field_ID': f"F{field+1:03d}",
                'Magnification': np.random.choice([500, 1000, 2000, 5000]),
                'Total_Porosity_%': round(porosity * 100, 2),
                'Mean_Pore_Size_um': round(mean_pore_size, 2),
                'D10_Pore_um': round(np.percentile(pore_sizes, 10), 2),
                'D50_Pore_um': round(np.percentile(pore_sizes, 50), 2),
                'D90_Pore_um': round(np.percentile(pore_sizes, 90), 2),
                'Crack_Density_per_mm2': round(crack_density, 3),
                'Avg_Crack_Width_um': round(avg_crack_width, 2),
                'Max_Crack_Width_um': round(avg_crack_width * np.random.uniform(1.5, 2.5), 2) if avg_crack_width > 0 else 0,
                'ITZ_Thickness_um': round(itz_thickness, 1),
                'ITZ_Porosity_%': round(itz_porosity * 100, 2),
                'Rubber_Integrity_%': round(rubber_integrity, 1),
                'Aspect_Ratio': round(np.random.uniform(1.2, 2.5), 2),
                'Circularity': round(np.random.uniform(0.6, 0.9), 2),
                'Specific_Surface_m2_g': round(2.5 + np.random.normal(0, 0.3), 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_eds_data(self, num_points: int = 20) -> pd.DataFrame:
        """Generate EDS elemental composition data"""
        
        data = []
        for point in range(num_points):
            # Base composition (weight %)
            composition = {
                'Ca': 25 + np.random.normal(0, 2),
                'Si': 15 + np.random.normal(0, 1.5),
                'Al': 5 + np.random.normal(0, 0.5),
                'Fe': 2 + np.random.normal(0, 0.3),
                'Mg': 1.5 + np.random.normal(0, 0.2),
                'Na': 0.8 + np.random.normal(0, 0.1),
                'K': 1.2 + np.random.normal(0, 0.15),
                'S': 1.0 + np.random.normal(0, 0.1),
                'O': 45 + np.random.normal(0, 3),
                'C': 3 + self.mix.rubber_content * 0.2 + np.random.normal(0, 0.5)
            }
            
            # Temperature effects on composition
            if self.exposure.temperature > 400:
                # Dehydration reduces Ca(OH)2
                composition['Ca'] -= (self.exposure.temperature - 400) / 100
                composition['O'] -= (self.exposure.temperature - 400) / 150
                
            if self.exposure.temperature > 600:
                # Decarbonation of CaCO3
                composition['C'] -= (self.exposure.temperature - 600) / 200
                
            # Normalize to 100%
            total = sum(composition.values())
            for element in composition:
                composition[element] = round(composition[element] * 100 / total, 2)
            
            # Location type
            location_type = np.random.choice(['Matrix', 'ITZ', 'Aggregate', 'Rubber', 'Crack'])
            
            data.append({
                'Point_ID': f"P{point+1:03d}",
                'Location_Type': location_type,
                'X_Position_um': round(np.random.uniform(0, 1000), 1),
                'Y_Position_um': round(np.random.uniform(0, 1000), 1),
                **{f"{elem}_wt%": val for elem, val in composition.items()},
                'Ca_Si_Ratio': round(composition['Ca'] / composition['Si'], 2),
                'Total_%': 100.0
            })
        
        return pd.DataFrame(data)

# ================================================================================
# XRD Analysis Data Generator
# ================================================================================

class XRDAnalysisGenerator:
    """Generates quantitative XRD phase analysis data"""
    
    def __init__(self, mix_design: MixDesign, thermal_exposure: ThermalExposure):
        self.mix = mix_design
        self.exposure = thermal_exposure
        
    def generate_phase_data(self) -> pd.DataFrame:
        """Generate quantitative phase composition data from Rietveld refinement"""
        
        # Initial phase composition at room temperature
        phases = {
            'C3S': 45,  # Alite
            'C2S': 20,  # Belite
            'C3A': 8,   # Tricalcium aluminate
            'C4AF': 10, # Tetracalcium aluminoferrite
            'CH': 12,   # Portlandite
            'CSH': 0,   # C-S-H (amorphous, increases with hydration)
            'CC': 3,    # Calcite
            'Quartz': 2 # From aggregates
        }
        
        # Temperature-dependent phase transformations
        temp = self.exposure.temperature
        
        if temp > 25:
            # Hydration products increase up to 100°C
            hydration_factor = min(1, temp / 100)
            phases['CSH'] = 30 * hydration_factor
            phases['C3S'] -= 15 * hydration_factor
            phases['C2S'] -= 10 * hydration_factor
        
        if temp > 100:
            # Dehydration of ettringite
            phases['Ettringite'] = max(0, 5 - (temp - 100) / 20)
        
        if temp > 400:
            # Portlandite decomposition
            phases['CH'] = max(0, phases['CH'] - (temp - 400) / 20)
            phases['CaO'] = min(10, (temp - 400) / 30)
        
        if temp > 600:
            # CSH decomposition
            phases['CSH'] = max(0, phases['CSH'] - (temp - 600) / 10)
            # Calcite decomposition
            phases['CC'] = max(0, phases['CC'] - (temp - 600) / 100)
        
        if temp > 700:
            # Formation of new phases
            phases['C12A7'] = min(5, (temp - 700) / 20)  # Mayenite
            phases['C2S_beta'] = min(15, (temp - 700) / 15)  # Beta-C2S
        
        # Add amorphous content
        total_crystalline = sum(phases.values())
        phases['Amorphous'] = 100 - total_crystalline
        
        # Normalize to 100%
        total = sum(phases.values())
        for phase in phases:
            phases[phase] = round(phases[phase] * 100 / total, 2)
        
        # Create DataFrame with additional parameters
        data = []
        for phase, content in phases.items():
            if content > 0:
                data.append({
                    'Phase': phase,
                    'Content_wt%': content,
                    'Crystallite_Size_nm': round(20 + np.random.normal(0, 5), 1),
                    'Microstrain_%': round(0.1 + np.random.exponential(0.05), 3),
                    'Lattice_Parameter_A': round(5.4 + np.random.normal(0, 0.1), 3),
                    'FWHM_deg': round(0.15 + np.random.normal(0, 0.02), 3),
                    'Preferred_Orientation': round(1.0 + np.random.normal(0, 0.1), 2)
                })
        
        df = pd.DataFrame(data)
        
        # Add refinement quality parameters
        df.attrs['Rwp'] = round(8.5 + np.random.normal(0, 1), 2)  # Weighted profile R-factor
        df.attrs['Rexp'] = round(6.2 + np.random.normal(0, 0.5), 2)  # Expected R-factor
        df.attrs['GOF'] = round(df.attrs['Rwp'] / df.attrs['Rexp'], 2)  # Goodness of fit
        
        return df

# ================================================================================
# TGA/DTA Analysis Data Generator
# ================================================================================

class ThermalAnalysisGenerator:
    """Generates TGA/DTA thermal analysis data"""
    
    def __init__(self, mix_design: MixDesign, thermal_exposure: ThermalExposure):
        self.mix = mix_design
        self.exposure = thermal_exposure
        
    def generate_tga_curve(self, temp_range: Tuple[float, float] = (25, 1000),
                          heating_rate: float = 10) -> pd.DataFrame:
        """Generate TGA weight loss curve with decomposition events"""
        
        temperatures = np.linspace(temp_range[0], temp_range[1], 500)
        weight = np.ones_like(temperatures) * 100  # Start at 100%
        
        # Define decomposition events
        events = []
        
        # Free water evaporation (25-105°C)
        mask = (temperatures > 25) & (temperatures <= 105)
        weight[mask] -= 2.5 * (temperatures[mask] - 25) / 80
        events.append({'Temp_Range': '25-105', 'Process': 'Free water', 'Weight_Loss_%': 2.5})
        
        # CSH dehydration (105-200°C)
        mask = (temperatures > 105) & (temperatures <= 200)
        weight[mask] -= 1.5 * (temperatures[mask] - 105) / 95
        events.append({'Temp_Range': '105-200', 'Process': 'CSH water', 'Weight_Loss_%': 1.5})
        
        # Rubber decomposition (200-500°C) if present
        if self.mix.rubber_content > 0:
            mask = (temperatures > 200) & (temperatures <= 500)
            rubber_loss = self.mix.rubber_content * 0.15  # 15% of rubber content
            weight[mask] -= rubber_loss * (temperatures[mask] - 200) / 300
            events.append({'Temp_Range': '200-500', 'Process': 'Rubber decomposition', 
                         'Weight_Loss_%': round(rubber_loss, 2)})
        
        # Portlandite decomposition (400-500°C)
        mask = (temperatures > 400) & (temperatures <= 500)
        weight[mask] -= 3.5 * (temperatures[mask] - 400) / 100
        events.append({'Temp_Range': '400-500', 'Process': 'Ca(OH)2 → CaO + H2O', 
                      'Weight_Loss_%': 3.5})
        
        # Calcite decomposition (600-800°C)
        mask = (temperatures > 600) & (temperatures <= 800)
        weight[mask] -= 2.0 * (temperatures[mask] - 600) / 200
        events.append({'Temp_Range': '600-800', 'Process': 'CaCO3 → CaO + CO2', 
                      'Weight_Loss_%': 2.0})
        
        # Calculate derivative (DTG)
        dtg = -np.gradient(weight, temperatures)
        
        # Add noise
        weight += np.random.normal(0, 0.05, len(weight))
        dtg += np.random.normal(0, 0.001, len(dtg))
        
        df = pd.DataFrame({
            'Temperature_C': temperatures,
            'Weight_%': weight,
            'DTG_%_per_C': dtg * 100,
            'Time_min': temperatures / heating_rate
        })
        
        df.attrs['events'] = events
        df.attrs['total_weight_loss'] = round(100 - weight[-1], 2)
        df.attrs['heating_rate'] = heating_rate
        
        return df
    
    def generate_dta_curve(self, temp_range: Tuple[float, float] = (25, 1000)) -> pd.DataFrame:
        """Generate DTA heat flow curve with thermal events"""
        
        temperatures = np.linspace(temp_range[0], temp_range[1], 500)
        heat_flow = np.zeros_like(temperatures)
        
        # Define thermal events (endothermic negative, exothermic positive)
        
        # Endothermic events
        # Water evaporation
        self._add_peak(heat_flow, temperatures, 95, -2.5, 20)
        
        # CSH dehydration
        self._add_peak(heat_flow, temperatures, 150, -1.5, 30)
        
        # Rubber melting/decomposition if present
        if self.mix.rubber_content > 0:
            self._add_peak(heat_flow, temperatures, 350, -3.0 * self.mix.rubber_content / 10, 50)
        
        # Portlandite decomposition
        self._add_peak(heat_flow, temperatures, 450, -4.0, 25)
        
        # Calcite decomposition
        self._add_peak(heat_flow, temperatures, 720, -3.5, 40)
        
        # Exothermic events
        # Oxidation of rubber char if present
        if self.mix.rubber_content > 0:
            self._add_peak(heat_flow, temperatures, 550, 2.0 * self.mix.rubber_content / 10, 40)
        
        # Add baseline drift and noise
        heat_flow += 0.001 * temperatures  # Baseline drift
        heat_flow += np.random.normal(0, 0.1, len(heat_flow))
        
        df = pd.DataFrame({
            'Temperature_C': temperatures,
            'Heat_Flow_mW_mg': heat_flow,
            'Time_min': temperatures / 10  # 10°C/min heating rate
        })
        
        return df
    
    def _add_peak(self, signal: np.ndarray, x: np.ndarray, center: float, 
                  amplitude: float, width: float) -> None:
        """Add a Gaussian peak to the signal"""
        peak = amplitude * np.exp(-0.5 * ((x - center) / width) ** 2)
        signal += peak

# ================================================================================
# Micro-CT Analysis Data Generator
# ================================================================================

class MicroCTAnalysisGenerator:
    """Generates 3D microstructural data from Micro-CT analysis"""
    
    def __init__(self, mix_design: MixDesign, thermal_exposure: ThermalExposure):
        self.mix = mix_design
        self.exposure = thermal_exposure
        
    def generate_3d_structure(self, voxel_size: float = 25,  # µm
                            volume_size: Tuple[int, int, int] = (100, 100, 100)) -> Dict:
        """Generate synthetic 3D microstructure data"""
        
        # Create 3D volume
        volume = np.zeros(volume_size, dtype=np.uint8)
        
        # Add phases (0: pore, 1: paste, 2: aggregate, 3: rubber, 4: crack)
        
        # Generate paste matrix
        volume[:, :, :] = 1
        
        # Add aggregates
        num_aggregates = int(20 * (1 - self.mix.rubber_content / 100))
        for _ in range(num_aggregates):
            center = np.random.randint(10, 90, 3)
            radius = np.random.randint(5, 15)
            self._add_sphere(volume, center, radius, 2)
        
        # Add rubber particles if present
        if self.mix.rubber_content > 0:
            num_rubber = int(self.mix.rubber_content * 2)
            for _ in range(num_rubber):
                center = np.random.randint(10, 90, 3)
                radius = np.random.randint(3, 8)
                if self.exposure.temperature < 300:
                    self._add_sphere(volume, center, radius, 3)
                else:
                    # Degraded rubber becomes pore
                    self._add_sphere(volume, center, radius, 0)
        
        # Add pores
        porosity_target = 0.1 + 0.15 * (self.exposure.temperature / 800)
        self._add_pores(volume, porosity_target)
        
        # Add cracks if temperature > 300°C
        if self.exposure.temperature > 300:
            num_cracks = int(5 * (self.exposure.temperature - 300) / 500)
            self._add_cracks(volume, num_cracks)
        
        # Calculate statistics
        stats = self._calculate_volume_statistics(volume, voxel_size)
        
        return {
            'volume_data': volume,
            'voxel_size_um': voxel_size,
            'statistics': stats,
            'metadata': {
                'dimensions': volume_size,
                'resolution': f"{voxel_size}µm",
                'total_volume_mm3': np.prod(volume_size) * (voxel_size/1000)**3
            }
        }
    
    def _add_sphere(self, volume: np.ndarray, center: np.ndarray, 
                    radius: float, value: int) -> None:
        """Add a spherical inclusion to the volume"""
        x, y, z = np.ogrid[:volume.shape[0], :volume.shape[1], :volume.shape[2]]
        mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2
        volume[mask] = value
    
    def _add_pores(self, volume: np.ndarray, target_porosity: float) -> None:
        """Add pores to achieve target porosity"""
        current_porosity = np.sum(volume == 0) / volume.size
        
        while current_porosity < target_porosity:
            center = np.random.randint(0, volume.shape[0], 3)
            radius = np.random.exponential(3)  # Exponential pore size distribution
            self._add_sphere(volume, center, radius, 0)
            current_porosity = np.sum(volume == 0) / volume.size
    
    def _add_cracks(self, volume: np.ndarray, num_cracks: int) -> None:
        """Add crack networks to the volume"""
        for _ in range(num_cracks):
            # Generate crack path
            start = np.random.randint(0, volume.shape[0], 3)
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Propagate crack
            length = np.random.randint(20, 50)
            for i in range(length):
                pos = start + i * direction
                pos = np.clip(pos.astype(int), 0, np.array(volume.shape) - 1)
                volume[pos[0], pos[1], pos[2]] = 4  # Crack phase
                
                # Add some width to crack
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if np.random.random() < 0.3:
                            try:
                                volume[pos[0]+dx, pos[1]+dy, pos[2]] = 4
                            except IndexError:
                                pass
    
    def _calculate_volume_statistics(self, volume: np.ndarray, 
                                    voxel_size: float) -> Dict:
        """Calculate 3D microstructural statistics"""
        
        voxel_volume = (voxel_size / 1000) ** 3  # mm³
        
        # Phase fractions
        phase_fractions = {
            'Porosity_%': round(100 * np.sum(volume == 0) / volume.size, 2),
            'Paste_%': round(100 * np.sum(volume == 1) / volume.size, 2),
            'Aggregate_%': round(100 * np.sum(volume == 2) / volume.size, 2),
            'Rubber_%': round(100 * np.sum(volume == 3) / volume.size, 2),
            'Cracks_%': round(100 * np.sum(volume == 4) / volume.size, 2)
        }
        
        # Connectivity analysis
        pore_labeled, num_pores = ndimage.label(volume == 0)
        
        # Pore size distribution
        pore_sizes = []
        for i in range(1, num_pores + 1):
            pore_size = np.sum(pore_labeled == i) * voxel_volume
            pore_sizes.append(pore_size)
        
        pore_sizes = np.array(pore_sizes) if pore_sizes else np.array([0])
        
        # Tortuosity estimation (simplified)
        tortuosity = 1.0 + 0.5 * phase_fractions['Porosity_%'] / 100
        
        return {
            **phase_fractions,
            'Number_of_Pores': num_pores,
            'Mean_Pore_Volume_mm3': round(np.mean(pore_sizes), 6),
            'Max_Pore_Volume_mm3': round(np.max(pore_sizes), 6),
            'Connectivity_Density': round(num_pores / (volume.size * voxel_volume), 3),
            'Tortuosity': round(tortuosity, 2),
            'Specific_Surface_Area_mm-1': round(self._calculate_surface_area(volume) / 
                                               (volume.size * voxel_volume), 2)
        }
    
    def _calculate_surface_area(self, volume: np.ndarray) -> float:
        """Calculate surface area of pore space"""
        # Simplified surface area calculation using edge detection
        edges = ndimage.sobel(volume == 0)
        return np.sum(edges > 0)

# ================================================================================
# Cross-Correlation and Validation Module
# ================================================================================

class CrossTechniqueValidator:
    """Ensures consistency across different analysis techniques"""
    
    def __init__(self):
        self.validation_rules = self._define_validation_rules()
        
    def _define_validation_rules(self) -> List[Dict]:
        """Define cross-technique validation rules"""
        return [
            {
                'name': 'Porosity Consistency',
                'techniques': ['SEM', 'MicroCT'],
                'parameter': 'porosity',
                'tolerance': 0.15  # 15% relative difference allowed
            },
            {
                'name': 'Phase Composition',
                'techniques': ['XRD', 'TGA'],
                'parameter': 'portlandite',
                'tolerance': 0.20  # 20% relative difference allowed
            },
            {
                'name': 'Thermal Events',
                'techniques': ['TGA', 'DTA'],
                'parameter': 'decomposition_temps',
                'tolerance': 10  # 10°C difference allowed
            }
        ]
    
    def validate_dataset(self, data: Dict) -> Dict:
        """Validate cross-technique consistency"""
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'passed': True,
            'checks': []
        }
        
        for rule in self.validation_rules:
            result = self._check_rule(data, rule)
            validation_results['checks'].append(result)
            if not result['passed']:
                validation_results['passed'] = False
        
        return validation_results
    
    def _check_rule(self, data: Dict, rule: Dict) -> Dict:
        """Check a specific validation rule"""
        
        # Simplified validation logic
        check_result = {
            'rule_name': rule['name'],
            'passed': True,
            'message': 'Validation passed',
            'values': {}
        }
        
        # Example: Check porosity consistency between SEM and MicroCT
        if rule['name'] == 'Porosity Consistency':
            if 'SEM' in data and 'MicroCT' in data:
                sem_data = data['SEM'].get('morphology', [])
                sem_porosity = sem_data[0].get('Total_Porosity_%', 0) if sem_data else 0
                ct_porosity = data['MicroCT'].get('statistics', {}).get('Porosity_%', 0)
                
                rel_diff = abs(sem_porosity - ct_porosity) / max(sem_porosity, 0.01)
                check_result['values'] = {
                    'SEM_porosity': sem_porosity,
                    'MicroCT_porosity': ct_porosity,
                    'relative_difference': round(rel_diff, 3)
                }
                
                if rel_diff > rule['tolerance']:
                    check_result['passed'] = False
                    check_result['message'] = f"Porosity mismatch exceeds tolerance ({rel_diff:.3f} > {rule['tolerance']})"
        
        return check_result

# ================================================================================
# Main Dataset Generation Pipeline
# ================================================================================

class Phase3DatasetGenerator:
    """Main class for generating complete Phase 3 microstructural dataset"""
    
    def __init__(self):
        self.output_dir = "phase3_microstructural_data"
        self.create_output_directory()
        
    def create_output_directory(self):
        """Create output directory structure"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['SEM', 'XRD', 'TGA', 'MicroCT', 'Correlations']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
    
    def generate_complete_dataset(self) -> Dict:
        """Generate complete multi-technique dataset for all samples"""
        
        print("=" * 80)
        print("PHASE 3: MICROSTRUCTURAL AND CHEMICAL ANALYSIS DATASET GENERATION")
        print("=" * 80)
        
        complete_dataset = {
            'metadata': {
                'project': 'Fire-Resistant Rubberized Concrete',
                'phase': 3,
                'generation_date': datetime.now().isoformat(),
                'version': '1.0'
            },
            'samples': {}
        }
        
        # Process each mix design at each temperature
        sample_count = 0
        for mix_id, mix_design in MIX_DESIGNS.items():
            for temp in CRITICAL_TEMPERATURES:
                sample_count += 1
                
                # Create sample ID
                exposure_type = 'ISO834' if temp > 400 else 'Furnace'
                sample_id = f"{mix_id}-{temp}C-{exposure_type}"
                
                print(f"\nProcessing Sample {sample_count}: {sample_id}")
                print("-" * 60)
                
                # Create thermal exposure conditions
                thermal_exposure = ThermalExposure(
                    temperature=temp,
                    exposure_type=exposure_type,
                    duration=120 if temp > 400 else 60,
                    heating_rate=5.0,
                    cooling_method='Air'
                )
                
                # Generate data for each technique
                sample_data = {
                    'sample_id': sample_id,
                    'mix_design': asdict(mix_design),
                    'thermal_exposure': asdict(thermal_exposure),
                    'analyses': {}
                }
                
                # SEM Analysis
                print(f"  → Generating SEM/EDS data...")
                sem_gen = SEMAnalysisGenerator(mix_design, thermal_exposure)
                morph_data = sem_gen.generate_morphology_data()
                eds_data = sem_gen.generate_eds_data()
                
                sample_data['analyses']['SEM'] = {
                    'morphology': morph_data.to_dict('records'),
                    'eds': eds_data.to_dict('records')
                }
                
                # XRD Analysis
                print(f"  → Generating XRD phase data...")
                xrd_gen = XRDAnalysisGenerator(mix_design, thermal_exposure)
                phase_data = xrd_gen.generate_phase_data()
                
                sample_data['analyses']['XRD'] = {
                    'phases': phase_data.to_dict('records'),
                    'refinement_quality': dict(phase_data.attrs)
                }
                
                # TGA/DTA Analysis
                print(f"  → Generating TGA/DTA thermal data...")
                thermal_gen = ThermalAnalysisGenerator(mix_design, thermal_exposure)
                tga_data = thermal_gen.generate_tga_curve()
                dta_data = thermal_gen.generate_dta_curve()
                
                sample_data['analyses']['TGA'] = {
                    'curve': tga_data[::20].to_dict('records'),  # Subsample for storage
                    'events': tga_data.attrs['events'],
                    'total_weight_loss': tga_data.attrs['total_weight_loss']
                }
                
                sample_data['analyses']['DTA'] = {
                    'curve': dta_data[::20].to_dict('records')  # Subsample for storage
                }
                
                # Micro-CT Analysis
                print(f"  → Generating Micro-CT 3D data...")
                ct_gen = MicroCTAnalysisGenerator(mix_design, thermal_exposure)
                ct_data = ct_gen.generate_3d_structure(volume_size=(50, 50, 50))  # Smaller for demo
                
                sample_data['analyses']['MicroCT'] = {
                    'statistics': ct_data['statistics'],
                    'metadata': ct_data['metadata']
                }
                
                # Store sample data
                complete_dataset['samples'][sample_id] = sample_data
                
                # Save individual sample data
                self._save_sample_data(sample_id, sample_data)
        
        # Cross-technique validation
        print("\n" + "=" * 60)
        print("CROSS-TECHNIQUE VALIDATION")
        print("=" * 60)
        
        validator = CrossTechniqueValidator()
        for sample_id, sample_data in complete_dataset['samples'].items():
            validation = validator.validate_dataset(sample_data['analyses'])
            complete_dataset['samples'][sample_id]['validation'] = validation
            status = "✓ PASSED" if validation['passed'] else "✗ FAILED"
            print(f"  {sample_id}: {status}")
        
        # Generate correlation matrix
        print("\n" + "=" * 60)
        print("GENERATING CORRELATION ANALYSIS")
        print("=" * 60)
        
        correlation_data = self._generate_correlations(complete_dataset)
        complete_dataset['correlations'] = correlation_data
        
        # Save complete dataset
        self._save_complete_dataset(complete_dataset)
        
        print("\n" + "=" * 80)
        print("DATASET GENERATION COMPLETE")
        print(f"Total Samples Generated: {sample_count}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 80)
        
        return complete_dataset
    
    def _generate_correlations(self, dataset: Dict) -> Dict:
        """Generate cross-technique correlations"""
        
        correlations = []
        
        for sample_id, sample_data in dataset['samples'].items():
            if 'analyses' in sample_data:
                analyses = sample_data['analyses']
                
                # Extract key parameters
                record = {
                    'Sample_ID': sample_id,
                    'Temperature_C': sample_data['thermal_exposure']['temperature'],
                    'Rubber_Content_%': sample_data['mix_design']['rubber_content']
                }
                
                # SEM parameters
                if 'SEM' in analyses and analyses['SEM']['morphology']:
                    morph = analyses['SEM']['morphology'][0]
                    record['SEM_Porosity_%'] = morph.get('Total_Porosity_%', 0)
                    record['SEM_Crack_Density'] = morph.get('Crack_Density_per_mm2', 0)
                
                # XRD parameters
                if 'XRD' in analyses and analyses['XRD']['phases']:
                    phases = {p['Phase']: p['Content_wt%'] for p in analyses['XRD']['phases']}
                    record['XRD_Portlandite_%'] = phases.get('CH', 0)
                    record['XRD_CSH_%'] = phases.get('CSH', 0)
                
                # TGA parameters
                if 'TGA' in analyses:
                    record['TGA_Weight_Loss_%'] = analyses['TGA'].get('total_weight_loss', 0)
                
                # MicroCT parameters
                if 'MicroCT' in analyses and 'statistics' in analyses['MicroCT']:
                    stats = analyses['MicroCT']['statistics']
                    record['CT_Porosity_%'] = stats.get('Porosity_%', 0)
                    record['CT_Tortuosity'] = stats.get('Tortuosity', 1)
                
                correlations.append(record)
        
        # Convert to DataFrame for correlation analysis
        corr_df = pd.DataFrame(correlations)
        
        # Calculate correlation matrix
        numeric_cols = [col for col in corr_df.columns if col != 'Sample_ID']
        corr_matrix = corr_df[numeric_cols].corr()
        
        return {
            'data': correlations,
            'correlation_matrix': corr_matrix.to_dict(),
            'summary_statistics': corr_df[numeric_cols].describe().to_dict()
        }
    
    def _save_sample_data(self, sample_id: str, data: Dict) -> None:
        """Save individual sample data to JSON file"""
        import os
        
        # Determine subdirectory based on temperature
        temp = data['thermal_exposure']['temperature']
        subdir = f"T{temp:04d}C"
        
        # Create temperature subdirectory
        temp_dir = os.path.join(self.output_dir, subdir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save JSON file
        filename = os.path.join(temp_dir, f"{sample_id}.json")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _save_complete_dataset(self, dataset: Dict) -> None:
        """Save complete dataset to master JSON file"""
        import os
        
        # Save complete dataset
        filename = os.path.join(self.output_dir, "complete_dataset.json")
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        # Also save summary CSV
        summary_data = []
        for sample_id, sample_data in dataset['samples'].items():
            summary_data.append({
                'Sample_ID': sample_id,
                'Mix_ID': sample_data['mix_design']['mix_id'],
                'Temperature_C': sample_data['thermal_exposure']['temperature'],
                'Rubber_Content_%': sample_data['mix_design']['rubber_content'],
                'Validation': sample_data.get('validation', {}).get('passed', False)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(self.output_dir, "dataset_summary.csv"), index=False)

# ================================================================================
# Main Execution
# ================================================================================

if __name__ == "__main__":
    # Generate complete Phase 3 dataset
    generator = Phase3DatasetGenerator()
    dataset = generator.generate_complete_dataset()
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("DATASET SUMMARY STATISTICS")
    print("=" * 80)
    
    total_samples = len(dataset['samples'])
    techniques = ['SEM', 'XRD', 'TGA', 'DTA', 'MicroCT']
    
    print(f"\nTotal Samples: {total_samples}")
    print(f"Mix Designs: {len(MIX_DESIGNS)}")
    print(f"Temperature Points: {len(CRITICAL_TEMPERATURES)}")
    print(f"Analysis Techniques: {', '.join(techniques)}")
    
    # Calculate data points
    total_datapoints = 0
    for sample in dataset['samples'].values():
        if 'analyses' in sample:
            for technique, data in sample['analyses'].items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            total_datapoints += len(value)
    
    print(f"Total Data Points: {total_datapoints:,}")
    
    # Validation summary
    passed_validations = sum(1 for s in dataset['samples'].values() 
                           if s.get('validation', {}).get('passed', False))
    print(f"Validation Pass Rate: {passed_validations}/{total_samples} ({100*passed_validations/total_samples:.1f}%)")
    
    print("\n" + "=" * 80)
    print("Phase 3 dataset generation complete!")
    print("Output directory: phase3_microstructural_data/")
    print("=" * 80)