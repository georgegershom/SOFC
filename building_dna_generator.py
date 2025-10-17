#!/usr/bin/env python3
"""
Building DNA Dataset Generator
A comprehensive tool for generating realistic building static and fabric data
for Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization
"""

import json
import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

@dataclass
class MaterialProperties:
    """Material properties for building components"""
    name: str
    density: float  # kg/m³
    thermal_conductivity: float  # W/m·K
    specific_heat: float  # J/kg·K
    u_value: float  # W/m²·K
    r_value: float  # m²·K/W
    thermal_mass: float  # J/m²·K
    emissivity: float
    solar_absorptance: float
    visible_transmittance: float
    shgc: float  # Solar Heat Gain Coefficient
    air_permeability: float  # m³/h·m² at 50 Pa
    fire_resistance: int  # minutes
    acoustic_rating: float  # dB
    embodied_carbon: float  # kg CO2/m²
    cost_per_m2: float  # USD/m²

@dataclass
class WallAssembly:
    """Wall assembly with layer-by-layer composition"""
    name: str
    layers: List[Dict[str, Any]]
    total_thickness: float  # mm
    total_u_value: float  # W/m²·K
    total_r_value: float  # m²·K/W
    total_thermal_mass: float  # J/m²·K
    air_tightness: float  # m³/h·m² at 50 Pa
    structural_load: float  # kN/m²
    fire_rating: int  # minutes
    acoustic_rating: float  # dB
    embodied_carbon: float  # kg CO2/m²
    cost_per_m2: float  # USD/m²

@dataclass
class WindowSpecification:
    """Window and door specifications"""
    id: str
    type: str  # window, door, skylight
    material: str  # aluminum, wood, vinyl, fiberglass
    frame_type: str
    glazing_type: str  # single, double, triple, low-e
    gas_fill: str  # air, argon, krypton
    u_value: float  # W/m²·K
    shgc: float
    visible_transmittance: float
    air_leakage: float  # m³/h·m² at 75 Pa
    water_penetration: float  # Pa
    structural_load: float  # Pa
    age: int  # years
    condition: str  # excellent, good, fair, poor
    maintenance_history: List[Dict[str, Any]]
    cost: float  # USD

@dataclass
class HVACSystem:
    """HVAC system specifications"""
    id: str
    system_type: str  # split, packaged, VRF, chiller, boiler
    make: str
    model: str
    fuel_type: str  # electric, gas, oil, heat pump
    age: int  # years
    efficiency_rating: Dict[str, float]  # SEER, AFUE, COP, EER
    rated_capacity: float  # kW or BTU/h
    actual_capacity: float  # kW or BTU/h
    maintenance_history: List[Dict[str, Any]]
    control_system: str
    zoning: int  # number of zones
    ductwork_condition: str
    filter_type: str
    cost: float  # USD
    replacement_cost: float  # USD

@dataclass
class LightingSystem:
    """Lighting system inventory"""
    id: str
    fixture_type: str
    lamp_type: str  # LED, CFL, incandescent, halogen
    wattage: float  # W
    lumens: float
    color_temperature: int  # K
    cri: int  # Color Rendering Index
    control_type: str  # switch, dimmer, occupancy, daylight
    quantity: int
    age: int  # years
    condition: str
    maintenance_schedule: str
    cost: float  # USD

@dataclass
class RenewableEnergySystem:
    """Renewable energy systems"""
    id: str
    system_type: str  # solar_pv, solar_thermal, wind, geothermal
    capacity: float  # kW or m²
    efficiency: float  # %
    age: int  # years
    orientation: float  # degrees from south
    tilt_angle: float  # degrees
    shading_factor: float  # 0-1
    inverter_type: str
    battery_storage: bool
    battery_capacity: float  # kWh
    maintenance_history: List[Dict[str, Any]]
    performance_data: List[Dict[str, Any]]
    cost: float  # USD

@dataclass
class BuildingGeometry:
    """3D building geometry and spatial data"""
    building_id: str
    building_type: str  # residential, commercial, office, industrial
    total_area: float  # m²
    total_volume: float  # m³
    number_of_floors: int
    floor_heights: List[float]  # m
    floor_areas: List[float]  # m²
    wall_areas: Dict[str, float]  # orientation -> area
    window_areas: Dict[str, float]  # orientation -> area
    roof_area: float  # m²
    roof_type: str  # flat, pitched, gable, hip
    roof_pitch: float  # degrees
    building_height: float  # m
    aspect_ratio: float
    window_to_wall_ratio: float
    floor_plan_coordinates: List[List[Tuple[float, float]]]  # 2D coordinates for each floor
    roof_coordinates: List[Tuple[float, float, float]]  # 3D roof coordinates
    surrounding_buildings: List[Dict[str, Any]]
    solar_exposure: Dict[str, float]  # orientation -> solar exposure factor

@dataclass
class AirTightnessData:
    """Air tightness and infiltration data"""
    test_date: str
    test_method: str  # blower_door, tracer_gas, pressurization
    air_changes_per_hour: float  # ACH at 50 Pa
    air_leakage_rate: float  # m³/h·m² at 50 Pa
    pressure_difference: float  # Pa
    temperature: float  # °C
    humidity: float  # %
    wind_speed: float  # m/s
    test_conditions: Dict[str, Any]
    leakage_locations: List[Dict[str, Any]]
    recommendations: List[str]

class BuildingDNAGenerator:
    """Main class for generating comprehensive building DNA datasets"""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with random seed"""
        np.random.seed(seed)
        random.seed(seed)
        self.materials_db = self._initialize_materials_database()
        self.hvac_systems_db = self._initialize_hvac_database()
        self.lighting_db = self._initialize_lighting_database()
        self.renewable_db = self._initialize_renewable_database()
        
    def _initialize_materials_database(self) -> Dict[str, MaterialProperties]:
        """Initialize comprehensive materials database"""
        materials = {
            # Wall Materials
            'brick_common': MaterialProperties(
                name='Common Brick', density=1800, thermal_conductivity=0.77,
                specific_heat=840, u_value=1.3, r_value=0.77, thermal_mass=1512,
                emissivity=0.93, solar_absorptance=0.65, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.1, fire_resistance=240, acoustic_rating=45,
                embodied_carbon=0.2, cost_per_m2=45
            ),
            'concrete_block': MaterialProperties(
                name='Concrete Block', density=2000, thermal_conductivity=1.1,
                specific_heat=1000, u_value=2.0, r_value=0.5, thermal_mass=2000,
                emissivity=0.9, solar_absorptance=0.7, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.05, fire_resistance=180, acoustic_rating=50,
                embodied_carbon=0.15, cost_per_m2=35
            ),
            'insulation_fiberglass': MaterialProperties(
                name='Fiberglass Insulation', density=12, thermal_conductivity=0.04,
                specific_heat=840, u_value=0.25, r_value=4.0, thermal_mass=10,
                emissivity=0.9, solar_absorptance=0.8, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=25,
                embodied_carbon=0.05, cost_per_m2=8
            ),
            'insulation_rockwool': MaterialProperties(
                name='Rockwool Insulation', density=30, thermal_conductivity=0.035,
                specific_heat=1000, u_value=0.22, r_value=4.5, thermal_mass=30,
                emissivity=0.9, solar_absorptance=0.8, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=30,
                embodied_carbon=0.08, cost_per_m2=12
            ),
            'insulation_xps': MaterialProperties(
                name='XPS Insulation', density=35, thermal_conductivity=0.03,
                specific_heat=1500, u_value=0.19, r_value=5.3, thermal_mass=53,
                emissivity=0.9, solar_absorptance=0.8, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=20,
                embodied_carbon=0.12, cost_per_m2=15
            ),
            'gypsum_board': MaterialProperties(
                name='Gypsum Board', density=800, thermal_conductivity=0.17,
                specific_heat=1090, u_value=1.0, r_value=1.0, thermal_mass=872,
                emissivity=0.9, solar_absorptance=0.6, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=30, acoustic_rating=35,
                embodied_carbon=0.08, cost_per_m2=8
            ),
            'aluminum_siding': MaterialProperties(
                name='Aluminum Siding', density=2700, thermal_conductivity=237,
                specific_heat=900, u_value=5.0, r_value=0.2, thermal_mass=2430,
                emissivity=0.1, solar_absorptance=0.3, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=15,
                embodied_carbon=0.25, cost_per_m2=25
            ),
            'vinyl_siding': MaterialProperties(
                name='Vinyl Siding', density=1400, thermal_conductivity=0.16,
                specific_heat=1200, u_value=0.8, r_value=1.25, thermal_mass=1680,
                emissivity=0.9, solar_absorptance=0.7, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=20,
                embodied_carbon=0.15, cost_per_m2=18
            ),
            
            # Roof Materials
            'asphalt_shingles': MaterialProperties(
                name='Asphalt Shingles', density=1200, thermal_conductivity=0.19,
                specific_heat=1000, u_value=0.5, r_value=2.0, thermal_mass=1200,
                emissivity=0.9, solar_absorptance=0.8, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=60, acoustic_rating=25,
                embodied_carbon=0.1, cost_per_m2=12
            ),
            'metal_roofing': MaterialProperties(
                name='Metal Roofing', density=7800, thermal_conductivity=50,
                specific_heat=460, u_value=2.0, r_value=0.5, thermal_mass=3588,
                emissivity=0.1, solar_absorptance=0.3, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=120, acoustic_rating=40,
                embodied_carbon=0.3, cost_per_m2=35
            ),
            'membrane_roofing': MaterialProperties(
                name='Membrane Roofing', density=1200, thermal_conductivity=0.16,
                specific_heat=1000, u_value=0.4, r_value=2.5, thermal_mass=1200,
                emissivity=0.9, solar_absorptance=0.7, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=90, acoustic_rating=30,
                embodied_carbon=0.2, cost_per_m2=20
            ),
            
            # Floor Materials
            'concrete_slab': MaterialProperties(
                name='Concrete Slab', density=2400, thermal_conductivity=1.7,
                specific_heat=1000, u_value=1.5, r_value=0.67, thermal_mass=2400,
                emissivity=0.9, solar_absorptance=0.6, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=240, acoustic_rating=50,
                embodied_carbon=0.2, cost_per_m2=30
            ),
            'wood_flooring': MaterialProperties(
                name='Wood Flooring', density=600, thermal_conductivity=0.14,
                specific_heat=1200, u_value=0.7, r_value=1.43, thermal_mass=720,
                emissivity=0.9, solar_absorptance=0.5, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=30, acoustic_rating=25,
                embodied_carbon=0.1, cost_per_m2=25
            ),
            'carpet': MaterialProperties(
                name='Carpet', density=200, thermal_conductivity=0.06,
                specific_heat=1200, u_value=0.3, r_value=3.33, thermal_mass=240,
                emissivity=0.9, solar_absorptance=0.6, visible_transmittance=0.0,
                shgc=0.0, air_permeability=0.0, fire_resistance=0, acoustic_rating=35,
                embodied_carbon=0.05, cost_per_m2=15
            ),
        }
        return materials
    
    def _initialize_hvac_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize HVAC systems database"""
        return {
            'residential_split': {
                'system_type': 'Split System',
                'fuel_type': 'Electric',
                'efficiency_range': {'SEER': (14, 21), 'EER': (11, 13)},
                'capacity_range': (1.5, 5.0),  # tons
                'age_range': (0, 25),
                'cost_range': (3000, 8000)
            },
            'commercial_packaged': {
                'system_type': 'Packaged Unit',
                'fuel_type': 'Electric',
                'efficiency_range': {'EER': (8, 12), 'COP': (2.5, 4.0)},
                'capacity_range': (5, 50),  # tons
                'age_range': (0, 30),
                'cost_range': (10000, 50000)
            },
            'heat_pump': {
                'system_type': 'Heat Pump',
                'fuel_type': 'Electric',
                'efficiency_range': {'HSPF': (8, 13), 'SEER': (14, 20)},
                'capacity_range': (1.5, 5.0),  # tons
                'age_range': (0, 20),
                'cost_range': (4000, 12000)
            },
            'boiler': {
                'system_type': 'Boiler',
                'fuel_type': 'Natural Gas',
                'efficiency_range': {'AFUE': (80, 95)},
                'capacity_range': (50, 500),  # MBH
                'age_range': (0, 35),
                'cost_range': (5000, 25000)
            },
            'chiller': {
                'system_type': 'Chiller',
                'fuel_type': 'Electric',
                'efficiency_range': {'COP': (4, 7), 'EER': (12, 18)},
                'capacity_range': (50, 1000),  # tons
                'age_range': (0, 30),
                'cost_range': (50000, 500000)
            }
        }
    
    def _initialize_lighting_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize lighting systems database"""
        return {
            'led_recessed': {
                'fixture_type': 'Recessed LED',
                'lamp_type': 'LED',
                'wattage_range': (8, 15),
                'lumens_range': (800, 1200),
                'color_temp_range': (2700, 6500),
                'cri_range': (80, 95),
                'control_types': ['switch', 'dimmer', 'occupancy', 'daylight'],
                'cost_range': (25, 75)
            },
            'led_linear': {
                'fixture_type': 'Linear LED',
                'lamp_type': 'LED',
                'wattage_range': (15, 40),
                'lumens_range': (1500, 4000),
                'color_temp_range': (2700, 6500),
                'cri_range': (80, 95),
                'control_types': ['switch', 'dimmer', 'occupancy', 'daylight'],
                'cost_range': (50, 150)
            },
            'fluorescent_t8': {
                'fixture_type': 'Fluorescent T8',
                'lamp_type': 'Fluorescent',
                'wattage_range': (25, 35),
                'lumens_range': (2000, 3000),
                'color_temp_range': (3000, 6500),
                'cri_range': (70, 85),
                'control_types': ['switch', 'dimmer'],
                'cost_range': (30, 80)
            },
            'incandescent': {
                'fixture_type': 'Incandescent',
                'lamp_type': 'Incandescent',
                'wattage_range': (40, 100),
                'lumens_range': (400, 1600),
                'color_temp_range': (2700, 3000),
                'cri_range': (95, 100),
                'control_types': ['switch', 'dimmer'],
                'cost_range': (5, 15)
            }
        }
    
    def _initialize_renewable_database(self) -> Dict[str, Dict[str, Any]]:
        """Initialize renewable energy systems database"""
        return {
            'solar_pv': {
                'system_type': 'Solar PV',
                'efficiency_range': (15, 22),  # %
                'capacity_range': (1, 1000),  # kW
                'age_range': (0, 25),
                'orientation_range': (0, 360),  # degrees
                'tilt_range': (0, 60),  # degrees
                'cost_range': (2000, 3000)  # USD/kW
            },
            'solar_thermal': {
                'system_type': 'Solar Thermal',
                'efficiency_range': (40, 80),  # %
                'capacity_range': (2, 200),  # m²
                'age_range': (0, 20),
                'orientation_range': (0, 360),  # degrees
                'tilt_range': (0, 60),  # degrees
                'cost_range': (300, 600)  # USD/m²
            },
            'wind_turbine': {
                'system_type': 'Wind Turbine',
                'efficiency_range': (25, 45),  # %
                'capacity_range': (1, 100),  # kW
                'age_range': (0, 20),
                'cost_range': (3000, 5000)  # USD/kW
            },
            'geothermal': {
                'system_type': 'Geothermal Heat Pump',
                'efficiency_range': (300, 500),  # COP
                'capacity_range': (2, 20),  # tons
                'age_range': (0, 30),
                'cost_range': (10000, 25000)  # USD/ton
            }
        }
    
    def generate_building_geometry(self, building_type: str = 'residential') -> BuildingGeometry:
        """Generate realistic building geometry data"""
        # Building type specific parameters
        type_params = {
            'residential': {
                'area_range': (100, 400),  # m²
                'floor_range': (1, 3),
                'height_range': (2.4, 3.0),  # m per floor
                'aspect_ratio_range': (0.5, 2.0),
                'wwr_range': (0.15, 0.25)
            },
            'commercial': {
                'area_range': (500, 5000),  # m²
                'floor_range': (2, 20),
                'height_range': (2.7, 3.5),  # m per floor
                'aspect_ratio_range': (0.3, 3.0),
                'wwr_range': (0.3, 0.6)
            },
            'office': {
                'area_range': (1000, 10000),  # m²
                'floor_range': (3, 50),
                'height_range': (2.8, 3.2),  # m per floor
                'aspect_ratio_range': (0.5, 4.0),
                'wwr_range': (0.4, 0.7)
            },
            'industrial': {
                'area_range': (2000, 50000),  # m²
                'floor_range': (1, 5),
                'height_range': (3.0, 8.0),  # m per floor
                'aspect_ratio_range': (0.2, 5.0),
                'wwr_range': (0.1, 0.3)
            }
        }
        
        params = type_params.get(building_type, type_params['residential'])
        
        # Generate basic dimensions
        total_area = np.random.uniform(*params['area_range'])
        num_floors = np.random.randint(*params['floor_range'])
        floor_height = np.random.uniform(*params['height_range'])
        
        # Calculate floor area
        floor_area = total_area / num_floors
        
        # Generate aspect ratio and dimensions
        aspect_ratio = np.random.uniform(*params['aspect_ratio_range'])
        length = math.sqrt(floor_area * aspect_ratio)
        width = floor_area / length
        
        # Generate floor heights (slight variation)
        floor_heights = [floor_height + np.random.normal(0, 0.1) for _ in range(num_floors)]
        floor_areas = [floor_area + np.random.normal(0, floor_area * 0.05) for _ in range(num_floors)]
        
        # Calculate wall areas by orientation
        wall_areas = {
            'north': length * sum(floor_heights),
            'south': length * sum(floor_heights),
            'east': width * sum(floor_heights),
            'west': width * sum(floor_heights)
        }
        
        # Generate window areas
        wwr = np.random.uniform(*params['wwr_range'])
        window_areas = {orient: area * wwr for orient, area in wall_areas.items()}
        
        # Generate roof data
        roof_area = length * width
        roof_types = ['flat', 'pitched', 'gable', 'hip']
        roof_type = np.random.choice(roof_types)
        roof_pitch = np.random.uniform(0, 45) if roof_type != 'flat' else 0
        
        # Generate floor plan coordinates (simplified rectangular)
        floor_plan_coordinates = []
        for i in range(num_floors):
            coords = [
                (0, 0), (length, 0), (length, width), (0, width), (0, 0)
            ]
            floor_plan_coordinates.append(coords)
        
        # Generate roof coordinates (3D)
        roof_coordinates = []
        if roof_type == 'flat':
            roof_coordinates = [(0, 0, sum(floor_heights)), (length, 0, sum(floor_heights)),
                               (length, width, sum(floor_heights)), (0, width, sum(floor_heights))]
        else:
            # Simplified pitched roof
            peak_height = sum(floor_heights) + length * math.tan(math.radians(roof_pitch)) / 2
            roof_coordinates = [(0, 0, sum(floor_heights)), (length, 0, sum(floor_heights)),
                               (length/2, width/2, peak_height), (0, width, sum(floor_heights))]
        
        # Generate surrounding buildings data
        num_surrounding = np.random.randint(0, 10)
        surrounding_buildings = []
        for _ in range(num_surrounding):
            dist = np.random.uniform(5, 100)  # meters
            height = np.random.uniform(3, 50)  # meters
            angle = np.random.uniform(0, 360)  # degrees
            surrounding_buildings.append({
                'distance': dist,
                'height': height,
                'angle': angle,
                'shading_factor': min(1.0, height / (dist * 0.5))
            })
        
        # Generate solar exposure data
        solar_exposure = {}
        for orient in ['north', 'south', 'east', 'west']:
            base_exposure = 1.0 if orient == 'south' else 0.5
            shading_factor = sum(b['shading_factor'] for b in surrounding_buildings) / max(1, len(surrounding_buildings))
            solar_exposure[orient] = base_exposure * (1 - shading_factor * 0.3)
        
        return BuildingGeometry(
            building_id=str(uuid.uuid4()),
            building_type=building_type,
            total_area=total_area,
            total_volume=total_area * sum(floor_heights),
            number_of_floors=num_floors,
            floor_heights=floor_heights,
            floor_areas=floor_areas,
            wall_areas=wall_areas,
            window_areas=window_areas,
            roof_area=roof_area,
            roof_type=roof_type,
            roof_pitch=roof_pitch,
            building_height=sum(floor_heights),
            aspect_ratio=aspect_ratio,
            window_to_wall_ratio=wwr,
            floor_plan_coordinates=floor_plan_coordinates,
            roof_coordinates=roof_coordinates,
            surrounding_buildings=surrounding_buildings,
            solar_exposure=solar_exposure
        )
    
    def generate_wall_assembly(self, building_type: str = 'residential') -> WallAssembly:
        """Generate realistic wall assembly with layer-by-layer composition"""
        # Common wall assembly types by building type
        assembly_templates = {
            'residential': [
                {
                    'name': 'Brick Veneer Wall',
                    'layers': [
                        {'material': 'brick_common', 'thickness': 100, 'position': 'exterior'},
                        {'material': 'insulation_fiberglass', 'thickness': 90, 'position': 'cavity'},
                        {'material': 'concrete_block', 'thickness': 200, 'position': 'structural'},
                        {'material': 'gypsum_board', 'thickness': 13, 'position': 'interior'}
                    ]
                },
                {
                    'name': 'Vinyl Siding Wall',
                    'layers': [
                        {'material': 'vinyl_siding', 'thickness': 10, 'position': 'exterior'},
                        {'material': 'insulation_xps', 'thickness': 50, 'position': 'exterior_insulation'},
                        {'material': 'concrete_block', 'thickness': 200, 'position': 'structural'},
                        {'material': 'gypsum_board', 'thickness': 13, 'position': 'interior'}
                    ]
                }
            ],
            'commercial': [
                {
                    'name': 'Curtain Wall System',
                    'layers': [
                        {'material': 'aluminum_siding', 'thickness': 5, 'position': 'exterior'},
                        {'material': 'insulation_rockwool', 'thickness': 100, 'position': 'cavity'},
                        {'material': 'concrete_block', 'thickness': 200, 'position': 'structural'},
                        {'material': 'gypsum_board', 'thickness': 13, 'position': 'interior'}
                    ]
                }
            ]
        }
        
        # Select random assembly template
        templates = assembly_templates.get(building_type, assembly_templates['residential'])
        template = np.random.choice(templates)
        
        # Calculate properties for each layer
        layers = []
        total_thickness = 0
        total_r_value = 0
        total_thermal_mass = 0
        total_embodied_carbon = 0
        total_cost = 0
        
        for layer_data in template['layers']:
            material = self.materials_db[layer_data['material']]
            thickness = layer_data['thickness'] / 1000  # convert to meters
            
            layer_props = {
                'material_name': material.name,
                'thickness_mm': layer_data['thickness'],
                'thickness_m': thickness,
                'position': layer_data['position'],
                'density': material.density,
                'thermal_conductivity': material.thermal_conductivity,
                'specific_heat': material.specific_heat,
                'r_value': thickness / material.thermal_conductivity,
                'thermal_mass': material.density * material.specific_heat * thickness,
                'embodied_carbon': material.embodied_carbon * thickness * 1000,  # per m²
                'cost': material.cost_per_m2
            }
            
            layers.append(layer_props)
            total_thickness += layer_data['thickness']
            total_r_value += layer_props['r_value']
            total_thermal_mass += layer_props['thermal_mass']
            total_embodied_carbon += layer_props['embodied_carbon']
            total_cost += layer_props['cost']
        
        # Calculate overall properties
        total_u_value = 1.0 / total_r_value if total_r_value > 0 else 0
        
        # Generate air tightness data
        air_tightness = np.random.uniform(0.1, 2.0)  # m³/h·m² at 50 Pa
        
        # Generate structural and performance data
        structural_load = np.random.uniform(2.0, 10.0)  # kN/m²
        fire_rating = min(240, int(total_thickness / 10))  # minutes
        acoustic_rating = np.random.uniform(30, 60)  # dB
        
        return WallAssembly(
            name=template['name'],
            layers=layers,
            total_thickness=total_thickness,
            total_u_value=total_u_value,
            total_r_value=total_r_value,
            total_thermal_mass=total_thermal_mass,
            air_tightness=air_tightness,
            structural_load=structural_load,
            fire_rating=fire_rating,
            acoustic_rating=acoustic_rating,
            embodied_carbon=total_embodied_carbon,
            cost_per_m2=total_cost
        )
    
    def generate_window_specification(self, building_type: str = 'residential') -> WindowSpecification:
        """Generate realistic window specifications"""
        # Window types by building type
        window_types = {
            'residential': ['single_hung', 'double_hung', 'casement', 'sliding', 'fixed'],
            'commercial': ['fixed', 'operable', 'curtain_wall', 'storefront'],
            'office': ['fixed', 'operable', 'curtain_wall', 'punch_window'],
            'industrial': ['fixed', 'operable', 'high_performance']
        }
        
        window_type = np.random.choice(window_types.get(building_type, window_types['residential']))
        
        # Material and frame types
        materials = ['aluminum', 'wood', 'vinyl', 'fiberglass', 'steel']
        frame_types = ['thermal_break', 'non_thermal_break', 'composite']
        glazing_types = ['single', 'double', 'triple', 'low_e_double', 'low_e_triple']
        gas_fills = ['air', 'argon', 'krypton', 'xenon']
        
        material = np.random.choice(materials)
        frame_type = np.random.choice(frame_types)
        glazing_type = np.random.choice(glazing_types)
        gas_fill = np.random.choice(gas_fills)
        
        # Generate performance properties based on glazing type
        if glazing_type == 'single':
            u_value = np.random.uniform(4.0, 6.0)
            shgc = np.random.uniform(0.7, 0.9)
            vt = np.random.uniform(0.8, 0.9)
        elif glazing_type == 'double':
            u_value = np.random.uniform(2.0, 3.0)
            shgc = np.random.uniform(0.5, 0.7)
            vt = np.random.uniform(0.6, 0.8)
        elif glazing_type == 'triple':
            u_value = np.random.uniform(1.0, 2.0)
            shgc = np.random.uniform(0.3, 0.5)
            vt = np.random.uniform(0.4, 0.6)
        elif 'low_e' in glazing_type:
            u_value = np.random.uniform(0.8, 2.5)
            shgc = np.random.uniform(0.2, 0.6)
            vt = np.random.uniform(0.4, 0.8)
        else:
            u_value = np.random.uniform(1.5, 3.5)
            shgc = np.random.uniform(0.4, 0.7)
            vt = np.random.uniform(0.5, 0.8)
        
        # Generate other properties
        air_leakage = np.random.uniform(0.1, 1.0)  # m³/h·m² at 75 Pa
        water_penetration = np.random.uniform(100, 500)  # Pa
        structural_load = np.random.uniform(1000, 5000)  # Pa
        age = np.random.randint(0, 30)  # years
        
        # Condition based on age
        if age < 5:
            condition = 'excellent'
        elif age < 15:
            condition = 'good'
        elif age < 25:
            condition = 'fair'
        else:
            condition = 'poor'
        
        # Generate maintenance history
        maintenance_history = []
        if age > 0:
            num_maintenance = max(1, age // 5)
            for i in range(num_maintenance):
                maintenance_history.append({
                    'date': (datetime.now() - timedelta(days=np.random.randint(30, age*365))).strftime('%Y-%m-%d'),
                    'type': np.random.choice(['cleaning', 'repair', 'replacement', 'inspection']),
                    'cost': np.random.uniform(50, 500),
                    'description': f'Maintenance event {i+1}'
                })
        
        # Generate cost
        base_cost = np.random.uniform(200, 2000)
        cost = base_cost * (1 + age * 0.05)  # Cost increases with age
        
        return WindowSpecification(
            id=str(uuid.uuid4()),
            type='window',
            material=material,
            frame_type=frame_type,
            glazing_type=glazing_type,
            gas_fill=gas_fill,
            u_value=u_value,
            shgc=shgc,
            visible_transmittance=vt,
            air_leakage=air_leakage,
            water_penetration=water_penetration,
            structural_load=structural_load,
            age=age,
            condition=condition,
            maintenance_history=maintenance_history,
            cost=cost
        )
    
    def generate_hvac_system(self, building_type: str = 'residential') -> HVACSystem:
        """Generate realistic HVAC system specifications"""
        # Select system type based on building type
        if building_type == 'residential':
            system_types = ['residential_split', 'heat_pump']
        elif building_type in ['commercial', 'office']:
            system_types = ['commercial_packaged', 'chiller', 'heat_pump']
        else:  # industrial
            system_types = ['chiller', 'boiler']
        
        system_type = np.random.choice(system_types)
        system_data = self.hvac_systems_db[system_type]
        
        # Generate system properties
        efficiency_rating = {}
        for rating_type, (min_val, max_val) in system_data['efficiency_range'].items():
            efficiency_rating[rating_type] = np.random.uniform(min_val, max_val)
        
        capacity = np.random.uniform(*system_data['capacity_range'])
        actual_capacity = capacity * np.random.uniform(0.8, 1.0)  # Some degradation
        age = np.random.randint(*system_data['age_range'])
        
        # Generate maintenance history
        maintenance_history = []
        if age > 0:
            num_maintenance = max(1, age // 3)
            for i in range(num_maintenance):
                maintenance_history.append({
                    'date': (datetime.now() - timedelta(days=np.random.randint(30, age*365))).strftime('%Y-%m-%d'),
                    'type': np.random.choice(['filter_change', 'coil_cleaning', 'refrigerant_check', 'inspection', 'repair']),
                    'cost': np.random.uniform(100, 2000),
                    'description': f'Maintenance event {i+1}'
                })
        
        # Generate control system
        control_systems = ['basic_thermostat', 'programmable_thermostat', 'smart_thermostat', 'building_automation']
        control_system = np.random.choice(control_systems)
        
        # Generate other properties
        zoning = np.random.randint(1, 10) if building_type != 'residential' else 1
        ductwork_condition = np.random.choice(['excellent', 'good', 'fair', 'poor'])
        filter_types = ['fiberglass', 'pleated', 'hepa', 'electrostatic']
        filter_type = np.random.choice(filter_types)
        
        # Generate costs
        base_cost = np.random.uniform(*system_data['cost_range'])
        cost = base_cost * (1 + age * 0.02)  # Slight cost increase with age
        replacement_cost = base_cost * 1.2  # 20% more for replacement
        
        return HVACSystem(
            id=str(uuid.uuid4()),
            system_type=system_data['system_type'],
            make=np.random.choice(['Carrier', 'Trane', 'Lennox', 'Rheem', 'Goodman', 'York', 'Daikin']),
            model=f"Model-{np.random.randint(1000, 9999)}",
            fuel_type=system_data['fuel_type'],
            age=age,
            efficiency_rating=efficiency_rating,
            rated_capacity=capacity,
            actual_capacity=actual_capacity,
            maintenance_history=maintenance_history,
            control_system=control_system,
            zoning=zoning,
            ductwork_condition=ductwork_condition,
            filter_type=filter_type,
            cost=cost,
            replacement_cost=replacement_cost
        )
    
    def generate_lighting_system(self, building_type: str = 'residential') -> List[LightingSystem]:
        """Generate realistic lighting system inventory"""
        lighting_systems = []
        
        # Room types and their lighting requirements
        room_types = {
            'residential': ['living_room', 'bedroom', 'kitchen', 'bathroom', 'hallway', 'garage'],
            'commercial': ['office', 'conference_room', 'lobby', 'restroom', 'storage', 'mechanical'],
            'office': ['open_office', 'private_office', 'conference_room', 'lobby', 'break_room', 'restroom'],
            'industrial': ['warehouse', 'production', 'office', 'break_room', 'restroom', 'loading_dock']
        }
        
        rooms = room_types.get(building_type, room_types['residential'])
        
        for room in rooms:
            # Select lighting type based on room and building type
            if building_type == 'residential':
                lighting_types = ['led_recessed', 'led_linear', 'incandescent']
            else:
                lighting_types = ['led_recessed', 'led_linear', 'fluorescent_t8']
            
            lighting_type = np.random.choice(lighting_types)
            lighting_data = self.lighting_db[lighting_type]
            
            # Generate properties
            wattage = np.random.uniform(*lighting_data['wattage_range'])
            lumens = np.random.uniform(*lighting_data['lumens_range'])
            color_temp = np.random.randint(*lighting_data['color_temp_range'])
            cri = np.random.randint(*lighting_data['cri_range'])
            control_type = np.random.choice(lighting_data['control_types'])
            quantity = np.random.randint(1, 20)  # Number of fixtures
            age = np.random.randint(0, 15)
            
            # Condition based on age
            if age < 3:
                condition = 'excellent'
            elif age < 8:
                condition = 'good'
            elif age < 12:
                condition = 'fair'
            else:
                condition = 'poor'
            
            # Maintenance schedule
            maintenance_schedules = ['monthly', 'quarterly', 'semi_annual', 'annual']
            maintenance_schedule = np.random.choice(maintenance_schedules)
            
            # Cost
            cost = np.random.uniform(*lighting_data['cost_range']) * quantity
            
            lighting_system = LightingSystem(
                id=str(uuid.uuid4()),
                fixture_type=lighting_data['fixture_type'],
                lamp_type=lighting_data['lamp_type'],
                wattage=wattage,
                lumens=lumens,
                color_temperature=color_temp,
                cri=cri,
                control_type=control_type,
                quantity=quantity,
                age=age,
                condition=condition,
                maintenance_schedule=maintenance_schedule,
                cost=cost
            )
            
            lighting_systems.append(lighting_system)
        
        return lighting_systems
    
    def generate_renewable_energy_system(self, building_type: str = 'residential') -> List[RenewableEnergySystem]:
        """Generate realistic renewable energy systems"""
        renewable_systems = []
        
        # Probability of having renewable systems by building type
        renewable_probabilities = {
            'residential': {'solar_pv': 0.3, 'solar_thermal': 0.1, 'geothermal': 0.05},
            'commercial': {'solar_pv': 0.4, 'solar_thermal': 0.2, 'wind_turbine': 0.1},
            'office': {'solar_pv': 0.5, 'solar_thermal': 0.15, 'wind_turbine': 0.05},
            'industrial': {'solar_pv': 0.6, 'wind_turbine': 0.3, 'geothermal': 0.1}
        }
        
        probs = renewable_probabilities.get(building_type, renewable_probabilities['residential'])
        
        for system_type, probability in probs.items():
            if np.random.random() < probability:
                system_data = self.renewable_db[system_type]
                
                # Generate properties
                capacity = np.random.uniform(*system_data['capacity_range'])
                efficiency = np.random.uniform(*system_data['efficiency_range'])
                age = np.random.randint(*system_data['age_range'])
                
                # Generate orientation and tilt for solar systems
                if 'solar' in system_type:
                    orientation = np.random.uniform(*system_data['orientation_range'])
                    tilt_angle = np.random.uniform(*system_data['tilt_range'])
                    shading_factor = np.random.uniform(0.0, 0.3)
                else:
                    orientation = 0
                    tilt_angle = 0
                    shading_factor = 0
                
                # Generate inverter type for PV
                inverter_types = ['string', 'micro', 'central', 'power_optimizer']
                inverter_type = np.random.choice(inverter_types) if system_type == 'solar_pv' else 'N/A'
                
                # Generate battery storage
                battery_storage = np.random.random() < 0.2  # 20% chance
                battery_capacity = np.random.uniform(5, 50) if battery_storage else 0
                
                # Generate maintenance history
                maintenance_history = []
                if age > 0:
                    num_maintenance = max(1, age // 5)
                    for i in range(num_maintenance):
                        maintenance_history.append({
                            'date': (datetime.now() - timedelta(days=np.random.randint(30, age*365))).strftime('%Y-%m-%d'),
                            'type': np.random.choice(['cleaning', 'inspection', 'repair', 'replacement']),
                            'cost': np.random.uniform(100, 2000),
                            'description': f'Maintenance event {i+1}'
                        })
                
                # Generate performance data
                performance_data = []
                for month in range(12):
                    performance_data.append({
                        'month': month + 1,
                        'energy_production': np.random.uniform(0.7, 1.3) * capacity * efficiency / 12,
                        'efficiency': efficiency * np.random.uniform(0.9, 1.1),
                        'availability': np.random.uniform(0.95, 1.0)
                    })
                
                # Generate cost
                cost = np.random.uniform(*system_data['cost_range']) * capacity
                
                renewable_system = RenewableEnergySystem(
                    id=str(uuid.uuid4()),
                    system_type=system_data['system_type'],
                    capacity=capacity,
                    efficiency=efficiency,
                    age=age,
                    orientation=orientation,
                    tilt_angle=tilt_angle,
                    shading_factor=shading_factor,
                    inverter_type=inverter_type,
                    battery_storage=battery_storage,
                    battery_capacity=battery_capacity,
                    maintenance_history=maintenance_history,
                    performance_data=performance_data,
                    cost=cost
                )
                
                renewable_systems.append(renewable_system)
        
        return renewable_systems
    
    def generate_air_tightness_data(self) -> AirTightnessData:
        """Generate realistic air tightness test data"""
        # Generate test parameters
        test_date = (datetime.now() - timedelta(days=np.random.randint(0, 365))).strftime('%Y-%m-%d')
        test_methods = ['blower_door', 'tracer_gas', 'pressurization']
        test_method = np.random.choice(test_methods)
        
        # Generate air tightness values
        ach_50 = np.random.uniform(0.5, 8.0)  # Air changes per hour at 50 Pa
        air_leakage_rate = ach_50 * 0.6  # Convert to m³/h·m² (approximate)
        
        # Generate test conditions
        pressure_difference = 50  # Pa
        temperature = np.random.uniform(15, 25)  # °C
        humidity = np.random.uniform(30, 70)  # %
        wind_speed = np.random.uniform(0, 10)  # m/s
        
        test_conditions = {
            'pressure_difference': pressure_difference,
            'temperature': temperature,
            'humidity': humidity,
            'wind_speed': wind_speed,
            'test_duration': np.random.uniform(30, 120),  # minutes
            'equipment': np.random.choice(['Minneapolis Blower Door', 'Retrotec', 'TEC'])
        }
        
        # Generate leakage locations
        leakage_locations = []
        num_locations = np.random.randint(1, 8)
        for i in range(num_locations):
            leakage_locations.append({
                'location': np.random.choice(['window_frame', 'door_frame', 'electrical_outlet', 'plumbing_penetration', 'attic_access', 'basement_joint']),
                'severity': np.random.choice(['minor', 'moderate', 'major']),
                'air_flow_rate': np.random.uniform(0.1, 2.0),  # m³/h
                'description': f'Leakage location {i+1}'
            })
        
        # Generate recommendations
        recommendations = []
        if ach_50 > 3.0:
            recommendations.append('Seal air leaks around windows and doors')
        if ach_50 > 5.0:
            recommendations.append('Improve air sealing at building envelope')
        if ach_50 > 7.0:
            recommendations.append('Consider comprehensive air sealing retrofit')
        
        recommendations.extend([
            'Install weatherstripping on doors',
            'Seal electrical and plumbing penetrations',
            'Improve attic and basement air sealing'
        ])
        
        return AirTightnessData(
            test_date=test_date,
            test_method=test_method,
            air_changes_per_hour=ach_50,
            air_leakage_rate=air_leakage_rate,
            pressure_difference=pressure_difference,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            test_conditions=test_conditions,
            leakage_locations=leakage_locations,
            recommendations=recommendations
        )
    
    def generate_complete_building_dna(self, building_type: str = 'residential') -> Dict[str, Any]:
        """Generate complete building DNA dataset"""
        print(f"Generating building DNA for {building_type} building...")
        
        # Generate all components
        geometry = self.generate_building_geometry(building_type)
        wall_assembly = self.generate_wall_assembly(building_type)
        window_specs = [self.generate_window_specification(building_type) for _ in range(np.random.randint(5, 20))]
        hvac_system = self.generate_hvac_system(building_type)
        lighting_systems = self.generate_lighting_system(building_type)
        renewable_systems = self.generate_renewable_energy_system(building_type)
        air_tightness = self.generate_air_tightness_data()
        
        # Compile complete dataset
        building_id = str(uuid.uuid4())
        building_dna = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'building_type': building_type,
                'generator_version': '1.0.0',
                'dataset_id': str(uuid.uuid4()),
                'building_id': building_id
            },
            'geometry': asdict(geometry),
            'construction_materials': {
                'wall_assembly': asdict(wall_assembly),
                'materials_database': {k: asdict(v) for k, v in self.materials_db.items()}
            },
            'openings': {
                'windows': [asdict(ws) for ws in window_specs],
                'doors': [asdict(self.generate_window_specification(building_type)) for _ in range(np.random.randint(2, 8))]
            },
            'systems': {
                'hvac': asdict(hvac_system),
                'lighting': [asdict(ls) for ls in lighting_systems],
                'renewable_energy': [asdict(rs) for rs in renewable_systems]
            },
            'performance_data': {
                'air_tightness': asdict(air_tightness),
                'thermal_performance': {
                    'overall_u_value': wall_assembly.total_u_value,
                    'overall_r_value': wall_assembly.total_r_value,
                    'thermal_mass': wall_assembly.total_thermal_mass,
                    'air_changes_per_hour': air_tightness.air_changes_per_hour
                }
            }
        }
        
        return building_dna
    
    def save_building_dna(self, building_dna: Dict[str, Any], filename: str = None) -> str:
        """Save building DNA dataset to JSON file"""
        if filename is None:
            building_type = building_dna['metadata']['building_type']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"building_dna_{building_type}_{timestamp}.json"
        
        filepath = Path(filename)
        with open(filepath, 'w') as f:
            json.dump(building_dna, f, indent=2, default=str)
        
        print(f"Building DNA dataset saved to: {filepath}")
        return str(filepath)
    
    def generate_multiple_buildings(self, building_types: List[str], num_buildings: int = 5) -> List[Dict[str, Any]]:
        """Generate multiple building DNA datasets"""
        all_buildings = []
        
        for building_type in building_types:
            for i in range(num_buildings):
                print(f"Generating building {i+1}/{num_buildings} of type {building_type}")
                building_dna = self.generate_complete_building_dna(building_type)
                all_buildings.append(building_dna)
        
        return all_buildings

def main():
    """Main function to demonstrate the building DNA generator"""
    print("Building DNA Dataset Generator")
    print("=" * 50)
    
    # Initialize generator
    generator = BuildingDNAGenerator(seed=42)
    
    # Generate sample buildings
    building_types = ['residential', 'commercial', 'office', 'industrial']
    all_buildings = generator.generate_multiple_buildings(building_types, num_buildings=2)
    
    # Save individual building datasets
    for i, building in enumerate(all_buildings):
        filename = f"building_dna_{i+1}_{building['metadata']['building_type']}.json"
        generator.save_building_dna(building, filename)
    
    # Create summary dataset
    summary_data = {
        'metadata': {
            'generated_at': datetime.now().isoformat(),
            'total_buildings': len(all_buildings),
            'building_types': building_types,
            'generator_version': '1.0.0'
        },
        'buildings': all_buildings
    }
    
    # Save summary dataset
    with open('building_dna_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    print(f"\nGenerated {len(all_buildings)} building DNA datasets")
    print("Summary dataset saved to: building_dna_summary.json")
    
    # Generate statistics
    print("\nDataset Statistics:")
    print("-" * 30)
    for building_type in building_types:
        count = sum(1 for b in all_buildings if b['metadata']['building_type'] == building_type)
        print(f"{building_type.capitalize()}: {count} buildings")

if __name__ == "__main__":
    main()