#!/usr/bin/env python3
"""
Generate building attributes and fabric data for building retrofit research.
Creates detailed building characteristics including geometric, structural,
construction, and thermal properties.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import random

class BuildingAttributesGenerator:
    def __init__(self):
        self.building_types = ['residential', 'office', 'retail', 'educational', 'healthcare', 'industrial']
        self.construction_periods = ['pre-1970', '1970-1980', '1981-1990', '1991-2000', '2001-2010', '2011-2020', 'post-2020']
        self.construction_qualities = ['poor', 'fair', 'good', 'excellent']
        self.architectural_styles = ['traditional', 'modern', 'brutalist', 'postmodern', 'contemporary', 'vernacular']
        
    def generate_geometric_data(self, building_id: str, building_type: str) -> Dict:
        """Generate geometric and structural data."""
        
        # Base dimensions by building type
        base_dimensions = {
            'residential': {'min_area': 50, 'max_area': 200, 'min_floors': 1, 'max_floors': 6},
            'office': {'min_area': 500, 'max_area': 5000, 'min_floors': 2, 'max_floors': 20},
            'retail': {'min_area': 200, 'max_area': 2000, 'min_floors': 1, 'max_floors': 3},
            'educational': {'min_area': 1000, 'max_area': 10000, 'min_floors': 1, 'max_floors': 4},
            'healthcare': {'min_area': 2000, 'max_area': 15000, 'min_floors': 2, 'max_floors': 8},
            'industrial': {'min_area': 1000, 'max_area': 20000, 'min_floors': 1, 'max_floors': 3}
        }
        
        config = base_dimensions.get(building_type, base_dimensions['office'])
        
        # Generate random dimensions
        floor_area = random.uniform(config['min_area'], config['max_area'])
        num_floors = random.randint(config['min_floors'], config['max_floors'])
        floor_height = random.uniform(2.5, 4.0)  # meters
        
        # Calculate derived metrics
        total_floor_area = floor_area * num_floors
        building_height = num_floors * floor_height
        
        # Assume roughly square footprint
        footprint_length = np.sqrt(floor_area)
        footprint_width = floor_area / footprint_length
        
        # Rooftop area (slightly less than footprint due to overhangs)
        rooftop_area = floor_area * random.uniform(0.85, 0.95)
        
        # Volume
        volume = total_floor_area * floor_height
        
        # Window-to-wall ratio
        wwr = random.uniform(0.15, 0.45)
        window_area = (2 * (footprint_length + footprint_width) * building_height * wwr)
        
        return {
            'building_id': building_id,
            'floor_area_m2': round(floor_area, 2),
            'total_floor_area_m2': round(total_floor_area, 2),
            'num_floors': num_floors,
            'floor_height_m': round(floor_height, 2),
            'building_height_m': round(building_height, 2),
            'footprint_length_m': round(footprint_length, 2),
            'footprint_width_m': round(footprint_width, 2),
            'rooftop_area_m2': round(rooftop_area, 2),
            'volume_m3': round(volume, 2),
            'window_area_m2': round(window_area, 2),
            'window_to_wall_ratio': round(wwr, 3)
        }
    
    def generate_construction_data(self, building_id: str, building_type: str) -> Dict:
        """Generate construction year, materials, and quality data."""
        
        # Construction year based on building type and random selection
        construction_year = random.randint(1950, 2023)
        construction_period = self.get_construction_period(construction_year)
        
        # Building age
        current_year = 2023
        building_age = current_year - construction_year
        
        # Construction quality (older buildings tend to have lower quality)
        if construction_year < 1980:
            quality_weights = [0.4, 0.4, 0.2, 0.0]  # More likely to be poor/fair
        elif construction_year < 2000:
            quality_weights = [0.2, 0.4, 0.3, 0.1]
        else:
            quality_weights = [0.1, 0.2, 0.4, 0.3]  # More likely to be good/excellent
        
        construction_quality = np.random.choice(self.construction_qualities, p=quality_weights)
        
        # Architectural style (correlated with construction year)
        if construction_year < 1970:
            style_weights = [0.3, 0.1, 0.2, 0.1, 0.1, 0.2]  # More traditional/vernacular
        elif construction_year < 1990:
            style_weights = [0.1, 0.2, 0.3, 0.2, 0.1, 0.1]  # More brutalist/postmodern
        else:
            style_weights = [0.05, 0.1, 0.1, 0.2, 0.4, 0.15]  # More contemporary/modern
        
        architectural_style = np.random.choice(self.architectural_styles, p=style_weights)
        
        # Primary construction materials
        materials = self.generate_construction_materials(construction_year, building_type)
        
        return {
            'building_id': building_id,
            'construction_year': construction_year,
            'construction_period': construction_period,
            'building_age_years': building_age,
            'construction_quality': construction_quality,
            'architectural_style': architectural_style,
            'primary_material': materials['primary'],
            'secondary_material': materials['secondary'],
            'roof_material': materials['roof'],
            'window_material': materials['window'],
            'insulation_present': materials['insulation'],
            'insulation_type': materials['insulation_type']
        }
    
    def generate_construction_materials(self, construction_year: int, building_type: str) -> Dict:
        """Generate construction materials based on year and building type."""
        
        # Primary structural materials by era
        if construction_year < 1970:
            primary_options = ['brick', 'concrete', 'stone', 'wood']
            primary_weights = [0.4, 0.3, 0.2, 0.1]
        elif construction_year < 1990:
            primary_options = ['concrete', 'steel', 'brick', 'wood']
            primary_weights = [0.5, 0.3, 0.15, 0.05]
        else:
            primary_options = ['steel', 'concrete', 'composite', 'wood']
            primary_weights = [0.4, 0.3, 0.2, 0.1]
        
        primary_material = np.random.choice(primary_options, p=primary_weights)
        
        # Secondary materials
        secondary_options = ['brick', 'concrete', 'glass', 'metal_cladding', 'wood', 'composite']
        secondary_weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
        secondary_material = np.random.choice(secondary_options, p=secondary_weights)
        
        # Roof materials
        roof_options = ['asphalt_shingle', 'metal', 'tile', 'membrane', 'slate', 'green_roof']
        roof_weights = [0.3, 0.2, 0.2, 0.15, 0.1, 0.05]
        roof_material = np.random.choice(roof_options, p=roof_weights)
        
        # Window materials
        window_options = ['aluminum', 'wood', 'vinyl', 'fiberglass', 'composite']
        window_weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        window_material = np.random.choice(window_options, p=window_weights)
        
        # Insulation (more likely in newer buildings)
        insulation_prob = min(0.9, 0.3 + (construction_year - 1950) * 0.01)
        insulation_present = np.random.random() < insulation_prob
        
        insulation_types = ['fiberglass', 'cellulose', 'foam', 'mineral_wool', 'none']
        if insulation_present:
            insulation_weights = [0.3, 0.2, 0.2, 0.3, 0.0]
        else:
            insulation_weights = [0.0, 0.0, 0.0, 0.0, 1.0]
        
        insulation_type = np.random.choice(insulation_types, p=insulation_weights)
        
        return {
            'primary': primary_material,
            'secondary': secondary_material,
            'roof': roof_material,
            'window': window_material,
            'insulation': insulation_present,
            'insulation_type': insulation_type
        }
    
    def generate_thermal_properties(self, building_id: str, construction_data: Dict) -> Dict:
        """Generate thermal properties of the building envelope."""
        
        construction_year = construction_data['construction_year']
        construction_quality = construction_data['construction_quality']
        insulation_present = construction_data['insulation_present']
        
        # Base U-values by construction period (W/m²K)
        base_u_values = {
            'pre-1970': {'wall': 2.5, 'roof': 3.0, 'floor': 2.0, 'window': 5.0},
            '1970-1980': {'wall': 1.8, 'roof': 2.2, 'floor': 1.5, 'window': 4.0},
            '1981-1990': {'wall': 1.2, 'roof': 1.5, 'floor': 1.0, 'window': 3.0},
            '1991-2000': {'wall': 0.8, 'roof': 1.0, 'floor': 0.8, 'window': 2.5},
            '2001-2010': {'wall': 0.6, 'roof': 0.8, 'floor': 0.6, 'window': 2.0},
            '2011-2020': {'wall': 0.4, 'roof': 0.6, 'floor': 0.4, 'window': 1.5},
            'post-2020': {'wall': 0.3, 'roof': 0.4, 'floor': 0.3, 'window': 1.2}
        }
        
        period = construction_data['construction_period']
        base_values = base_u_values.get(period, base_u_values['1991-2000'])
        
        # Quality adjustment
        quality_multipliers = {'poor': 1.3, 'fair': 1.1, 'good': 0.9, 'excellent': 0.7}
        quality_mult = quality_multipliers[construction_quality]
        
        # Insulation adjustment
        insulation_mult = 0.6 if insulation_present else 1.0
        
        # Calculate U-values
        u_wall = base_values['wall'] * quality_mult * insulation_mult
        u_roof = base_values['roof'] * quality_mult * insulation_mult
        u_floor = base_values['floor'] * quality_mult * insulation_mult
        u_window = base_values['window'] * quality_mult
        
        # Calculate R-values (R = 1/U)
        r_wall = 1 / u_wall if u_wall > 0 else 0
        r_roof = 1 / u_roof if u_roof > 0 else 0
        r_floor = 1 / u_floor if u_floor > 0 else 0
        r_window = 1 / u_window if u_window > 0 else 0
        
        # Air tightness (ACH at 50 Pa)
        if construction_year < 1980:
            air_tightness = random.uniform(8, 15)
        elif construction_year < 2000:
            air_tightness = random.uniform(4, 8)
        else:
            air_tightness = random.uniform(1, 4)
        
        # Thermal mass (kJ/m²K)
        thermal_mass = random.uniform(50, 200)
        
        return {
            'building_id': building_id,
            'u_value_wall_w_m2k': round(u_wall, 3),
            'u_value_roof_w_m2k': round(u_roof, 3),
            'u_value_floor_w_m2k': round(u_floor, 3),
            'u_value_window_w_m2k': round(u_window, 3),
            'r_value_wall_m2k_w': round(r_wall, 3),
            'r_value_roof_m2k_w': round(r_roof, 3),
            'r_value_floor_m2k_w': round(r_floor, 3),
            'r_value_window_m2k_w': round(r_window, 3),
            'air_tightness_ach50': round(air_tightness, 2),
            'thermal_mass_kj_m2k': round(thermal_mass, 1)
        }
    
    def get_construction_period(self, year: int) -> str:
        """Get construction period based on year."""
        if year < 1970:
            return 'pre-1970'
        elif year < 1981:
            return '1970-1980'
        elif year < 1991:
            return '1981-1990'
        elif year < 2001:
            return '1991-2000'
        elif year < 2011:
            return '2001-2010'
        elif year < 2021:
            return '2011-2020'
        else:
            return 'post-2020'
    
    def generate_building_attributes(self, buildings: List[Dict]) -> pd.DataFrame:
        """Generate complete building attributes for all buildings."""
        
        all_attributes = []
        
        for building in buildings:
            building_id = building['id']
            building_type = building['type']
            
            print(f"Generating attributes for building {building_id} ({building_type})...")
            
            # Generate each component
            geometric_data = self.generate_geometric_data(building_id, building_type)
            construction_data = self.generate_construction_data(building_id, building_type)
            thermal_data = self.generate_thermal_properties(building_id, construction_data)
            
            # Combine all data
            building_attributes = {**geometric_data, **construction_data, **thermal_data}
            all_attributes.append(building_attributes)
        
        return pd.DataFrame(all_attributes)

def main():
    """Generate building attributes data for the building retrofit dataset."""
    
    # Sample buildings
    buildings = [
        {'id': 'B001', 'type': 'residential'},
        {'id': 'B002', 'type': 'office'},
        {'id': 'B003', 'type': 'retail'},
        {'id': 'B004', 'type': 'educational'},
        {'id': 'B005', 'type': 'residential'},
        {'id': 'B006', 'type': 'office'},
        {'id': 'B007', 'type': 'retail'},
        {'id': 'B008', 'type': 'educational'},
        {'id': 'B009', 'type': 'residential'},
        {'id': 'B010', 'type': 'office'},
        {'id': 'B011', 'type': 'healthcare'},
        {'id': 'B012', 'type': 'industrial'},
        {'id': 'B013', 'type': 'residential'},
        {'id': 'B014', 'type': 'office'},
        {'id': 'B015', 'type': 'retail'}
    ]
    
    # Initialize generator
    generator = BuildingAttributesGenerator()
    
    # Generate building attributes
    print("Generating building attributes data...")
    attributes_df = generator.generate_building_attributes(buildings)
    
    # Save data
    output_dir = '../raw_data/building_attributes'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{output_dir}/building_attributes.csv"
    attributes_df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(attributes_df)} buildings")
    
        # Create summary statistics
        summary = {
            'total_buildings': len(attributes_df),
            'building_types': attributes_df['building_type'].value_counts().to_dict() if 'building_type' in attributes_df.columns else {},
            'construction_periods': attributes_df['construction_period'].value_counts().to_dict(),
            'construction_qualities': attributes_df['construction_quality'].value_counts().to_dict(),
            'architectural_styles': attributes_df['architectural_style'].value_counts().to_dict(),
            'primary_materials': attributes_df['primary_material'].value_counts().to_dict(),
            'insulation_coverage': attributes_df['insulation_present'].value_counts().to_dict()
        }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nBuilding attributes generation complete!")
    print(f"Generated attributes for {len(attributes_df)} buildings")
    print(f"Building types: {list(summary['building_types'].keys())}")
    print(f"Construction periods: {list(summary['construction_periods'].keys())}")

if __name__ == "__main__":
    main()