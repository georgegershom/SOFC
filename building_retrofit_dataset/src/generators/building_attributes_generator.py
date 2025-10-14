"""
Building Attributes & Fabric Data Generator
Generates detailed building characteristics including:
- Geometric & structural data
- Construction materials and thermal properties
- Building function and architectural style
- Envelope characteristics (U-values, R-values)
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any
from faker import Faker

fake = Faker()

class BuildingAttributesGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        Faker.seed(seed)
        
        # Define building characteristics distributions
        self.building_types = ['office', 'residential', 'retail', 'educational', 
                               'healthcare', 'industrial', 'mixed_use']
        
        self.construction_periods = {
            'pre_1920': (1850, 1920),
            '1920_1945': (1920, 1945),
            '1946_1979': (1946, 1979),
            '1980_1999': (1980, 1999),
            '2000_2010': (2000, 2010),
            '2011_present': (2011, 2024)
        }
        
        self.architectural_styles = {
            'pre_1920': ['Victorian', 'Edwardian', 'Georgian', 'Art Nouveau'],
            '1920_1945': ['Art Deco', 'Modernist', 'International', 'Bauhaus'],
            '1946_1979': ['Brutalist', 'Mid-Century Modern', 'Post-Modern'],
            '1980_1999': ['High-Tech', 'Deconstructivist', 'Neo-Modern'],
            '2000_2010': ['Contemporary', 'Sustainable', 'Minimalist'],
            '2011_present': ['Parametric', 'Biophilic', 'Smart', 'Net-Zero']
        }
        
        self.construction_materials = {
            'structure': ['concrete', 'steel', 'timber', 'masonry', 'composite'],
            'facade': ['brick', 'stone', 'glass', 'metal_panel', 'composite_panel', 'stucco'],
            'insulation': ['mineral_wool', 'fiberglass', 'polyurethane', 'cellulose', 
                          'eps', 'xps', 'aerogel'],
            'windows': ['single_glazed', 'double_glazed', 'triple_glazed', 
                       'low_e_double', 'low_e_triple', 'smart_glass'],
            'roof': ['flat_membrane', 'pitched_tile', 'metal_standing_seam', 
                    'green_roof', 'solar_integrated']
        }
        
    def generate_building(self, building_id: str) -> Dict[str, Any]:
        """Generate comprehensive building attributes"""
        
        # Basic information
        building_type = random.choice(self.building_types)
        construction_period = random.choice(list(self.construction_periods.keys()))
        year_range = self.construction_periods[construction_period]
        construction_year = random.randint(*year_range)
        
        # Geometric properties
        geometry = self._generate_geometry(building_type)
        
        # Materials and construction
        materials = self._generate_materials(construction_period, building_type)
        
        # Thermal properties
        thermal = self._generate_thermal_properties(construction_period, materials)
        
        # Building quality and condition
        quality = self._generate_quality_assessment(construction_year)
        
        # Compile all attributes
        building_data = {
            'building_id': building_id,
            'building_name': f"{fake.company()} {random.choice(['Building', 'Tower', 'Center', 'House', 'Complex'])}",
            'address': fake.address().replace('\n', ', '),
            'latitude': round(fake.latitude(), 6),
            'longitude': round(fake.longitude(), 6),
            
            # Type and function
            'building_type': building_type,
            'primary_use': self._get_primary_use(building_type),
            'occupancy_class': self._get_occupancy_class(building_type),
            
            # Age and style
            'construction_year': construction_year,
            'construction_period': construction_period,
            'architectural_style': random.choice(self.architectural_styles.get(
                construction_period, ['Contemporary'])),
            'heritage_status': 'listed' if construction_year < 1950 and random.random() < 0.2 else 'none',
            
            # Geometry
            **geometry,
            
            # Materials
            **{f'material_{k}': v for k, v in materials.items()},
            
            # Thermal properties
            **{f'thermal_{k}': v for k, v in thermal.items()},
            
            # Quality assessment
            **{f'quality_{k}': v for k, v in quality.items()},
            
            # Certifications
            'energy_rating': self._generate_energy_rating(construction_year, thermal),
            'leed_certification': self._generate_leed_cert(construction_year),
            'breeam_rating': self._generate_breeam_rating(construction_year),
            
            # Renovation history
            'last_major_renovation': self._generate_renovation_year(construction_year),
            'retrofit_potential': self._assess_retrofit_potential(construction_year, quality, thermal)
        }
        
        return building_data
    
    def _generate_geometry(self, building_type: str) -> Dict[str, Any]:
        """Generate building geometric properties"""
        
        # Size distributions by type
        size_ranges = {
            'residential': (500, 5000, 2, 5),  # area_min, area_max, floors_min, floors_max
            'office': (2000, 50000, 3, 30),
            'retail': (1000, 20000, 1, 4),
            'educational': (3000, 30000, 2, 5),
            'healthcare': (5000, 50000, 2, 10),
            'industrial': (2000, 40000, 1, 3),
            'mixed_use': (3000, 40000, 3, 25)
        }
        
        area_min, area_max, floors_min, floors_max = size_ranges.get(
            building_type, (1000, 10000, 2, 10))
        
        # Generate dimensions
        gross_floor_area = round(random.uniform(area_min, area_max), 2)
        number_of_floors = random.randint(floors_min, floors_max)
        
        # Footprint and dimensions
        footprint_area = gross_floor_area / number_of_floors
        
        # Assume roughly rectangular buildings
        aspect_ratio = random.uniform(1.2, 2.5)
        building_width = np.sqrt(footprint_area / aspect_ratio)
        building_length = footprint_area / building_width
        
        # Heights
        floor_height = random.uniform(3.0, 4.5) if building_type != 'industrial' else random.uniform(4.5, 8.0)
        total_height = number_of_floors * floor_height
        
        # Envelope areas
        wall_area = 2 * total_height * (building_width + building_length)
        window_wall_ratio = random.uniform(0.2, 0.7) if building_type == 'office' else random.uniform(0.15, 0.35)
        window_area = wall_area * window_wall_ratio
        
        # Roof area (considering potential for solar/green roof)
        roof_area = footprint_area * random.uniform(1.0, 1.3)  # Account for roof shape
        usable_roof_area = roof_area * random.uniform(0.5, 0.9)  # Usable for solar/green
        
        return {
            'gross_floor_area_m2': gross_floor_area,
            'footprint_area_m2': round(footprint_area, 2),
            'number_of_floors': number_of_floors,
            'building_height_m': round(total_height, 1),
            'floor_to_ceiling_height_m': round(floor_height, 1),
            'building_length_m': round(building_length, 1),
            'building_width_m': round(building_width, 1),
            'total_wall_area_m2': round(wall_area, 2),
            'total_window_area_m2': round(window_area, 2),
            'window_wall_ratio': round(window_wall_ratio, 3),
            'roof_area_m2': round(roof_area, 2),
            'usable_roof_area_m2': round(usable_roof_area, 2),
            'building_volume_m3': round(gross_floor_area * floor_height, 2),
            'shape_factor': round(wall_area / gross_floor_area, 3),  # Compactness indicator
            'has_basement': random.choice([True, False]),
            'number_of_units': random.randint(1, 100) if building_type == 'residential' else 1
        }
    
    def _generate_materials(self, construction_period: str, building_type: str) -> Dict[str, str]:
        """Generate construction materials based on period and type"""
        
        # Period-specific material preferences
        period_materials = {
            'pre_1920': {
                'structure': ['masonry', 'timber'],
                'facade': ['brick', 'stone'],
                'insulation': ['none', 'mineral_wool'],
                'windows': ['single_glazed'],
                'roof': ['pitched_tile', 'flat_membrane']
            },
            '1920_1945': {
                'structure': ['masonry', 'concrete', 'steel'],
                'facade': ['brick', 'stucco'],
                'insulation': ['mineral_wool', 'none'],
                'windows': ['single_glazed'],
                'roof': ['pitched_tile', 'flat_membrane']
            },
            '1946_1979': {
                'structure': ['concrete', 'steel'],
                'facade': ['brick', 'concrete', 'metal_panel'],
                'insulation': ['fiberglass', 'mineral_wool'],
                'windows': ['single_glazed', 'double_glazed'],
                'roof': ['flat_membrane', 'metal_standing_seam']
            },
            '1980_1999': {
                'structure': ['concrete', 'steel'],
                'facade': ['glass', 'metal_panel', 'composite_panel'],
                'insulation': ['fiberglass', 'polyurethane'],
                'windows': ['double_glazed', 'low_e_double'],
                'roof': ['flat_membrane', 'metal_standing_seam']
            },
            '2000_2010': {
                'structure': ['concrete', 'steel', 'composite'],
                'facade': ['glass', 'composite_panel', 'metal_panel'],
                'insulation': ['polyurethane', 'eps', 'xps'],
                'windows': ['low_e_double', 'triple_glazed'],
                'roof': ['flat_membrane', 'green_roof', 'metal_standing_seam']
            },
            '2011_present': {
                'structure': ['steel', 'composite', 'timber'],
                'facade': ['glass', 'composite_panel', 'metal_panel'],
                'insulation': ['polyurethane', 'xps', 'aerogel'],
                'windows': ['low_e_triple', 'triple_glazed', 'smart_glass'],
                'roof': ['green_roof', 'solar_integrated', 'flat_membrane']
            }
        }
        
        available_materials = period_materials.get(construction_period, self.construction_materials)
        
        materials = {}
        for component in ['structure', 'facade', 'insulation', 'windows', 'roof']:
            if component in available_materials:
                materials[component] = random.choice(available_materials[component])
            else:
                materials[component] = random.choice(self.construction_materials[component])
        
        return materials
    
    def _generate_thermal_properties(self, construction_period: str, 
                                    materials: Dict[str, str]) -> Dict[str, float]:
        """Generate thermal properties based on construction period and materials"""
        
        # U-value ranges by period (W/m²K)
        u_value_ranges = {
            'pre_1920': {'wall': (1.5, 2.5), 'roof': (1.8, 2.8), 'floor': (1.2, 2.0), 'window': (4.5, 5.8)},
            '1920_1945': {'wall': (1.3, 2.2), 'roof': (1.5, 2.5), 'floor': (1.0, 1.8), 'window': (4.5, 5.8)},
            '1946_1979': {'wall': (0.8, 1.8), 'roof': (0.8, 2.0), 'floor': (0.7, 1.5), 'window': (2.8, 5.8)},
            '1980_1999': {'wall': (0.4, 1.0), 'roof': (0.4, 1.0), 'floor': (0.4, 0.8), 'window': (2.0, 3.5)},
            '2000_2010': {'wall': (0.25, 0.5), 'roof': (0.2, 0.4), 'floor': (0.25, 0.5), 'window': (1.4, 2.2)},
            '2011_present': {'wall': (0.15, 0.35), 'roof': (0.13, 0.25), 'floor': (0.15, 0.3), 'window': (0.8, 1.6)}
        }
        
        ranges = u_value_ranges.get(construction_period, u_value_ranges['1980_1999'])
        
        # Generate U-values
        u_values = {}
        for component in ['wall', 'roof', 'floor', 'window']:
            u_values[f'u_value_{component}'] = round(random.uniform(*ranges[component]), 3)
        
        # Calculate R-values (thermal resistance = 1/U)
        r_values = {}
        for component in ['wall', 'roof', 'floor', 'window']:
            r_values[f'r_value_{component}'] = round(1 / u_values[f'u_value_{component}'], 3)
        
        # Additional thermal properties
        thermal_props = {
            **u_values,
            **r_values,
            'thermal_mass_kj_k_m2': round(random.uniform(50, 500), 1),  # Thermal mass
            'air_leakage_m3_h_m2': round(random.uniform(1, 15), 2),  # Air permeability
            'thermal_bridge_coefficient': round(random.uniform(0.05, 0.3), 3),
            'solar_heat_gain_coefficient': round(random.uniform(0.3, 0.7), 2),
            'infiltration_rate_ach': round(random.uniform(0.1, 1.5), 2)  # Air changes per hour
        }
        
        return thermal_props
    
    def _generate_quality_assessment(self, construction_year: int) -> Dict[str, Any]:
        """Generate building quality and condition metrics"""
        
        building_age = 2024 - construction_year
        
        # Condition deteriorates with age (with some randomness)
        base_condition = max(1, 5 - building_age // 30)
        condition_score = min(5, max(1, base_condition + random.randint(-1, 1)))
        
        conditions = {
            1: 'poor',
            2: 'fair',
            3: 'good',
            4: 'very_good',
            5: 'excellent'
        }
        
        # Detailed component conditions
        components = ['structure', 'envelope', 'roof', 'windows', 'hvac', 'electrical', 'plumbing']
        component_conditions = {}
        
        for component in components:
            # Components deteriorate at different rates
            deterioration_rate = random.uniform(0.8, 1.2)
            comp_condition = min(5, max(1, condition_score + random.randint(-1, 1)))
            component_conditions[f'{component}_condition'] = conditions[comp_condition]
            component_conditions[f'{component}_remaining_life_years'] = max(
                0, round((6 - comp_condition) * 10 * deterioration_rate))
        
        return {
            'overall_condition': conditions[condition_score],
            'condition_score': condition_score,
            'maintenance_backlog_cost_eur': round(building_age * random.uniform(50, 200) * 
                                                  (6 - condition_score) * 100),
            **component_conditions
        }
    
    def _get_primary_use(self, building_type: str) -> str:
        """Get primary use category for building type"""
        uses = {
            'residential': random.choice(['single_family', 'multi_family', 'apartment', 'condo']),
            'office': random.choice(['corporate', 'government', 'professional', 'coworking']),
            'retail': random.choice(['shopping_mall', 'department_store', 'supermarket', 'boutique']),
            'educational': random.choice(['primary_school', 'high_school', 'university', 'training_center']),
            'healthcare': random.choice(['hospital', 'clinic', 'medical_office', 'care_facility']),
            'industrial': random.choice(['manufacturing', 'warehouse', 'distribution', 'workshop']),
            'mixed_use': 'mixed_commercial_residential'
        }
        return uses.get(building_type, 'general')
    
    def _get_occupancy_class(self, building_type: str) -> str:
        """Get occupancy classification"""
        classes = {
            'residential': 'R',
            'office': 'B',
            'retail': 'M',
            'educational': 'E',
            'healthcare': 'I',
            'industrial': 'F',
            'mixed_use': 'Mixed'
        }
        return classes.get(building_type, 'U')
    
    def _generate_energy_rating(self, construction_year: int, thermal: Dict) -> str:
        """Generate EU energy rating (A-G)"""
        # Better ratings for newer buildings and better thermal properties
        avg_u_value = np.mean([thermal[k] for k in thermal if k.startswith('u_value')])
        
        if construction_year > 2010 and avg_u_value < 1.0:
            return random.choice(['A', 'B'])
        elif construction_year > 2000 and avg_u_value < 1.5:
            return random.choice(['B', 'C'])
        elif construction_year > 1990 and avg_u_value < 2.0:
            return random.choice(['C', 'D'])
        elif construction_year > 1980 and avg_u_value < 2.5:
            return random.choice(['D', 'E'])
        elif avg_u_value < 3.0:
            return random.choice(['E', 'F'])
        else:
            return random.choice(['F', 'G'])
    
    def _generate_leed_cert(self, construction_year: int) -> str:
        """Generate LEED certification status"""
        if construction_year > 2000 and random.random() < 0.3:
            return random.choice(['Certified', 'Silver', 'Gold', 'Platinum'])
        return 'None'
    
    def _generate_breeam_rating(self, construction_year: int) -> str:
        """Generate BREEAM rating"""
        if construction_year > 2005 and random.random() < 0.25:
            return random.choice(['Pass', 'Good', 'Very Good', 'Excellent', 'Outstanding'])
        return 'None'
    
    def _generate_renovation_year(self, construction_year: int) -> int:
        """Generate last major renovation year"""
        if random.random() < 0.4:  # 40% chance of having been renovated
            earliest_renovation = min(max(construction_year + 20, 1990), 2023)
            if earliest_renovation < 2023:
                return random.randint(earliest_renovation, 2023)
        return 0  # No renovation
    
    def _assess_retrofit_potential(self, construction_year: int, 
                                  quality: Dict, thermal: Dict) -> str:
        """Assess retrofit potential"""
        avg_u_value = np.mean([thermal[k] for k in thermal if k.startswith('u_value')])
        condition_score = quality['condition_score']
        
        if avg_u_value > 2.0 and condition_score < 4:
            return 'high'
        elif avg_u_value > 1.5 or condition_score < 3:
            return 'medium'
        elif construction_year < 2000:
            return 'low'
        else:
            return 'minimal'
    
    def generate_building_dataset(self, n_buildings: int = 100) -> pd.DataFrame:
        """Generate dataset for multiple buildings"""
        buildings = []
        
        for i in range(n_buildings):
            building_id = f"BLD_{str(i+1).zfill(5)}"
            building_data = self.generate_building(building_id)
            buildings.append(building_data)
        
        return pd.DataFrame(buildings)