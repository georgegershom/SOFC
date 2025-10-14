"""
Building Attributes Dataset Generator for Building Retrofit Research
Generates detailed building fabric and structural attributes.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

np.random.seed(42)


class BuildingAttributesGenerator:
    """Generate comprehensive building attributes including geometric, structural, and thermal properties."""
    
    def __init__(self, num_buildings=50):
        self.num_buildings = num_buildings
        
        # Material properties database
        self.wall_materials = {
            'brick_uninsulated': {'u_value': 2.1, 'r_value': 0.48, 'cost_per_m2': 85, 'carbon_kgco2_m2': 45},
            'brick_insulated': {'u_value': 0.35, 'r_value': 2.86, 'cost_per_m2': 145, 'carbon_kgco2_m2': 55},
            'concrete_block': {'u_value': 1.8, 'r_value': 0.56, 'cost_per_m2': 95, 'carbon_kgco2_m2': 65},
            'cavity_wall_insulated': {'u_value': 0.28, 'r_value': 3.57, 'cost_per_m2': 165, 'carbon_kgco2_m2': 48},
            'timber_frame': {'u_value': 0.25, 'r_value': 4.0, 'cost_per_m2': 125, 'carbon_kgco2_m2': 28},
            'steel_frame': {'u_value': 0.30, 'r_value': 3.33, 'cost_per_m2': 185, 'carbon_kgco2_m2': 95}
        }
        
        self.roof_materials = {
            'pitched_uninsulated': {'u_value': 2.3, 'r_value': 0.43, 'cost_per_m2': 75, 'carbon_kgco2_m2': 35},
            'pitched_insulated': {'u_value': 0.16, 'r_value': 6.25, 'cost_per_m2': 135, 'carbon_kgco2_m2': 42},
            'flat_uninsulated': {'u_value': 1.8, 'r_value': 0.56, 'cost_per_m2': 85, 'carbon_kgco2_m2': 38},
            'flat_insulated': {'u_value': 0.18, 'r_value': 5.56, 'cost_per_m2': 155, 'carbon_kgco2_m2': 45},
            'green_roof': {'u_value': 0.15, 'r_value': 6.67, 'cost_per_m2': 245, 'carbon_kgco2_m2': 32}
        }
        
        self.window_types = {
            'single_glazed': {'u_value': 5.8, 'r_value': 0.17, 'cost_per_m2': 185, 'carbon_kgco2_m2': 55, 'shgc': 0.86},
            'double_glazed': {'u_value': 2.8, 'r_value': 0.36, 'cost_per_m2': 285, 'carbon_kgco2_m2': 75, 'shgc': 0.76},
            'double_glazed_low_e': {'u_value': 1.8, 'r_value': 0.56, 'cost_per_m2': 385, 'carbon_kgco2_m2': 82, 'shgc': 0.70},
            'triple_glazed': {'u_value': 0.8, 'r_value': 1.25, 'cost_per_m2': 485, 'carbon_kgco2_m2': 95, 'shgc': 0.50}
        }
        
        self.hvac_systems = {
            'gas_boiler_old': {'efficiency': 0.65, 'cop': None, 'cost': 3500, 'carbon_kgco2_year': 2500},
            'gas_boiler_condensing': {'efficiency': 0.90, 'cop': None, 'cost': 5500, 'carbon_kgco2_year': 1800},
            'air_source_heat_pump': {'efficiency': None, 'cop': 3.2, 'cost': 12000, 'carbon_kgco2_year': 800},
            'ground_source_heat_pump': {'efficiency': None, 'cop': 4.5, 'cost': 22000, 'carbon_kgco2_year': 600},
            'district_heating': {'efficiency': 0.85, 'cop': None, 'cost': 8000, 'carbon_kgco2_year': 1200},
            'electric_resistance': {'efficiency': 1.0, 'cop': None, 'cost': 2500, 'carbon_kgco2_year': 3500}
        }
    
    def generate_geometric_data(self, building_type):
        """Generate geometric and structural data."""
        if building_type == 'residential':
            num_floors = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.35, 0.2, 0.1, 0.05])
            floor_area_per_floor = np.random.uniform(80, 250)
        elif building_type == 'commercial':
            num_floors = np.random.choice([1, 2, 3, 4, 5, 6, 8, 10], p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05])
            floor_area_per_floor = np.random.uniform(300, 1200)
        elif building_type == 'industrial':
            num_floors = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            floor_area_per_floor = np.random.uniform(500, 3000)
        else:  # educational
            num_floors = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
            floor_area_per_floor = np.random.uniform(400, 1500)
        
        total_floor_area = num_floors * floor_area_per_floor
        
        # Estimate building footprint (assuming rectangular)
        aspect_ratio = np.random.uniform(1.2, 2.5)
        footprint_area = floor_area_per_floor
        building_width = np.sqrt(footprint_area / aspect_ratio)
        building_length = footprint_area / building_width
        
        # Height
        floor_height = np.random.uniform(2.7, 3.5)
        total_height = num_floors * floor_height
        
        # Volume
        volume = total_floor_area * floor_height
        
        # Rooftop area
        rooftop_area = footprint_area
        
        # Envelope area (walls)
        perimeter = 2 * (building_length + building_width)
        wall_area = perimeter * total_height
        
        # Window to wall ratio
        if building_type == 'commercial':
            window_wall_ratio = np.random.uniform(0.35, 0.60)
        elif building_type == 'educational':
            window_wall_ratio = np.random.uniform(0.25, 0.45)
        else:
            window_wall_ratio = np.random.uniform(0.15, 0.35)
        
        window_area = wall_area * window_wall_ratio
        
        return {
            'num_floors': num_floors,
            'floor_height_m': round(floor_height, 2),
            'total_height_m': round(total_height, 2),
            'footprint_area_m2': round(footprint_area, 2),
            'total_floor_area_m2': round(total_floor_area, 2),
            'building_width_m': round(building_width, 2),
            'building_length_m': round(building_length, 2),
            'rooftop_area_m2': round(rooftop_area, 2),
            'wall_area_m2': round(wall_area, 2),
            'window_area_m2': round(window_area, 2),
            'window_wall_ratio': round(window_wall_ratio, 3),
            'volume_m3': round(volume, 2)
        }
    
    def generate_construction_attributes(self, construction_year):
        """Generate construction materials and quality based on construction year."""
        # Material selection based on construction period
        if construction_year < 1950:
            wall_material = np.random.choice(['brick_uninsulated', 'concrete_block'], p=[0.7, 0.3])
            roof_material = np.random.choice(['pitched_uninsulated', 'flat_uninsulated'], p=[0.8, 0.2])
            window_type = 'single_glazed'
            hvac_system = np.random.choice(['gas_boiler_old', 'electric_resistance'], p=[0.7, 0.3])
            quality = np.random.choice(['poor', 'fair', 'good'], p=[0.4, 0.4, 0.2])
        elif construction_year < 1980:
            wall_material = np.random.choice(['brick_uninsulated', 'concrete_block', 'cavity_wall_insulated'], p=[0.4, 0.4, 0.2])
            roof_material = np.random.choice(['pitched_uninsulated', 'flat_uninsulated', 'pitched_insulated'], p=[0.4, 0.3, 0.3])
            window_type = np.random.choice(['single_glazed', 'double_glazed'], p=[0.7, 0.3])
            hvac_system = np.random.choice(['gas_boiler_old', 'gas_boiler_condensing'], p=[0.6, 0.4])
            quality = np.random.choice(['fair', 'good'], p=[0.6, 0.4])
        elif construction_year < 2010:
            wall_material = np.random.choice(['brick_insulated', 'cavity_wall_insulated', 'timber_frame'], p=[0.4, 0.4, 0.2])
            roof_material = np.random.choice(['pitched_insulated', 'flat_insulated'], p=[0.6, 0.4])
            window_type = np.random.choice(['double_glazed', 'double_glazed_low_e'], p=[0.6, 0.4])
            hvac_system = np.random.choice(['gas_boiler_condensing', 'district_heating', 'air_source_heat_pump'], p=[0.6, 0.2, 0.2])
            quality = np.random.choice(['good', 'excellent'], p=[0.6, 0.4])
        else:  # Modern buildings
            wall_material = np.random.choice(['cavity_wall_insulated', 'timber_frame', 'steel_frame'], p=[0.3, 0.4, 0.3])
            roof_material = np.random.choice(['pitched_insulated', 'flat_insulated', 'green_roof'], p=[0.4, 0.4, 0.2])
            window_type = np.random.choice(['double_glazed_low_e', 'triple_glazed'], p=[0.6, 0.4])
            hvac_system = np.random.choice(['air_source_heat_pump', 'ground_source_heat_pump', 'district_heating'], p=[0.5, 0.3, 0.2])
            quality = 'excellent'
        
        return {
            'wall_material': wall_material,
            'roof_material': roof_material,
            'window_type': window_type,
            'hvac_system': hvac_system,
            'quality': quality
        }
    
    def generate_thermal_properties(self, materials, geometry):
        """Calculate thermal properties based on materials and geometry."""
        wall_props = self.wall_materials[materials['wall_material']]
        roof_props = self.roof_materials[materials['roof_material']]
        window_props = self.window_types[materials['window_type']]
        
        # Calculate weighted average U-value for entire envelope
        wall_area = geometry['wall_area_m2']
        window_area = geometry['window_area_m2']
        roof_area = geometry['rooftop_area_m2']
        
        actual_wall_area = wall_area - window_area
        total_envelope_area = actual_wall_area + window_area + roof_area
        
        weighted_u_value = (
            (actual_wall_area * wall_props['u_value'] +
             window_area * window_props['u_value'] +
             roof_area * roof_props['u_value']) / total_envelope_area
        )
        
        # Heat loss coefficient (W/K)
        heat_loss_coefficient = (
            actual_wall_area * wall_props['u_value'] +
            window_area * window_props['u_value'] +
            roof_area * roof_props['u_value']
        )
        
        # Thermal mass (kJ/K) - simplified estimation
        volume = geometry['volume_m3']
        thermal_mass = volume * np.random.uniform(400, 800)  # kJ/K
        
        return {
            'wall_u_value': wall_props['u_value'],
            'wall_r_value': wall_props['r_value'],
            'roof_u_value': roof_props['u_value'],
            'roof_r_value': roof_props['r_value'],
            'window_u_value': window_props['u_value'],
            'window_shgc': window_props['shgc'],
            'envelope_avg_u_value': round(weighted_u_value, 3),
            'heat_loss_coefficient_w_k': round(heat_loss_coefficient, 2),
            'thermal_mass_kj_k': round(thermal_mass, 2)
        }
    
    def estimate_epc_rating(self, thermal_props, hvac_system, construction_year):
        """Estimate Energy Performance Certificate rating (EU A-G scale)."""
        # Simplified EPC calculation based on thermal properties and HVAC
        u_value = thermal_props['envelope_avg_u_value']
        
        hvac_props = self.hvac_systems[hvac_system]
        hvac_efficiency = hvac_props.get('cop', hvac_props.get('efficiency', 1.0))
        
        # Energy performance score (lower is better)
        base_score = u_value * 100
        hvac_factor = 1.0 / hvac_efficiency if hvac_efficiency else 1.0
        age_penalty = max(0, (2024 - construction_year) / 100)
        
        performance_score = base_score * hvac_factor * (1 + age_penalty)
        
        # Convert to rating
        if performance_score < 25:
            return 'A'
        elif performance_score < 50:
            return 'B'
        elif performance_score < 75:
            return 'C'
        elif performance_score < 100:
            return 'D'
        elif performance_score < 130:
            return 'E'
        elif performance_score < 160:
            return 'F'
        else:
            return 'G'
    
    def generate_building_style(self, building_type, construction_year):
        """Generate architectural style based on type and construction year."""
        styles = {
            'residential': {
                'pre_1950': ['Victorian', 'Edwardian', 'Georgian', 'Traditional'],
                '1950_1980': ['Post-war', 'Modernist', 'Brutalist'],
                '1980_2010': ['Contemporary', 'Neo-traditional', 'Postmodern'],
                'post_2010': ['Modern', 'Sustainable', 'Smart building']
            },
            'commercial': {
                'pre_1950': ['Classical', 'Art Deco', 'Traditional'],
                '1950_1980': ['International Style', 'Brutalist', 'Modernist'],
                '1980_2010': ['High-tech', 'Postmodern', 'Corporate'],
                'post_2010': ['Green building', 'Smart building', 'Contemporary']
            }
        }
        
        style_dict = styles.get(building_type, styles['residential'])
        
        if construction_year < 1950:
            period = 'pre_1950'
        elif construction_year < 1980:
            period = '1950_1980'
        elif construction_year < 2010:
            period = '1980_2010'
        else:
            period = 'post_2010'
        
        return np.random.choice(style_dict[period])
    
    def generate_dataset(self):
        """Generate complete building attributes dataset."""
        print(f"Generating building attributes for {self.num_buildings} buildings...")
        
        building_types = ['residential', 'commercial', 'industrial', 'educational']
        building_functions = {
            'residential': ['Single-family', 'Multi-family', 'Apartment', 'Townhouse'],
            'commercial': ['Office', 'Retail', 'Hotel', 'Restaurant', 'Mixed-use'],
            'industrial': ['Manufacturing', 'Warehouse', 'Factory', 'Workshop'],
            'educational': ['School', 'University', 'Training center', 'Library']
        }
        
        buildings = []
        
        for i in range(self.num_buildings):
            building_id = f'BLD_{i+1:03d}'
            building_type = np.random.choice(building_types)
            construction_year = int(np.random.uniform(1920, 2023))
            
            # Generate all attributes
            geometry = self.generate_geometric_data(building_type)
            materials = self.generate_construction_attributes(construction_year)
            thermal = self.generate_thermal_properties(materials, geometry)
            
            epc_rating = self.estimate_epc_rating(thermal, materials['hvac_system'], construction_year)
            style = self.generate_building_style(building_type, construction_year)
            function = np.random.choice(building_functions[building_type])
            
            # Get material costs and carbon
            wall_props = self.wall_materials[materials['wall_material']]
            roof_props = self.roof_materials[materials['roof_material']]
            window_props = self.window_types[materials['window_type']]
            hvac_props = self.hvac_systems[materials['hvac_system']]
            
            building = {
                'building_id': building_id,
                'building_type': building_type,
                'building_function': function,
                'construction_year': construction_year,
                'building_age_years': 2024 - construction_year,
                'architectural_style': style,
                'quality_rating': materials['quality'],
                'epc_rating': epc_rating,
                
                # Geometric properties
                **geometry,
                
                # Materials
                'wall_material': materials['wall_material'],
                'roof_material': materials['roof_material'],
                'window_type': materials['window_type'],
                'hvac_system': materials['hvac_system'],
                
                # Thermal properties
                **thermal,
                
                # Cost estimates
                'wall_cost_per_m2': wall_props['cost_per_m2'],
                'roof_cost_per_m2': roof_props['cost_per_m2'],
                'window_cost_per_m2': window_props['cost_per_m2'],
                'hvac_system_cost': hvac_props['cost'],
                
                # Carbon footprint
                'wall_carbon_kgco2_m2': wall_props['carbon_kgco2_m2'],
                'roof_carbon_kgco2_m2': roof_props['carbon_kgco2_m2'],
                'window_carbon_kgco2_m2': window_props['carbon_kgco2_m2'],
                'hvac_carbon_kgco2_year': hvac_props['carbon_kgco2_year']
            }
            
            buildings.append(building)
        
        return pd.DataFrame(buildings)


def main():
    """Main function to generate and save building attributes data."""
    print("=" * 80)
    print("BUILDING ATTRIBUTES DATA GENERATOR FOR RETROFIT RESEARCH")
    print("=" * 80)
    
    # Generate data
    generator = BuildingAttributesGenerator(num_buildings=50)
    df = generator.generate_dataset()
    
    # Save datasets
    print("\nSaving datasets...")
    df.to_csv('../data/building_attributes.csv', index=False)
    df.to_excel('../data/building_attributes.xlsx', index=False)
    
    # Save material properties as reference
    material_db = {
        'wall_materials': generator.wall_materials,
        'roof_materials': generator.roof_materials,
        'window_types': generator.window_types,
        'hvac_systems': generator.hvac_systems
    }
    
    with open('../data/material_properties_database.json', 'w') as f:
        json.dump(material_db, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total buildings: {len(df)}")
    print(f"\nColumns: {len(df.columns)}")
    
    print("\n" + "-" * 80)
    print("BUILDING TYPE DISTRIBUTION:")
    print("-" * 80)
    print(df['building_type'].value_counts())
    
    print("\n" + "-" * 80)
    print("EPC RATING DISTRIBUTION:")
    print("-" * 80)
    print(df['epc_rating'].value_counts().sort_index())
    
    print("\n" + "-" * 80)
    print("CONSTRUCTION ERA:")
    print("-" * 80)
    print(f"Oldest building: {df['construction_year'].min()}")
    print(f"Newest building: {df['construction_year'].max()}")
    print(f"Average age: {df['building_age_years'].mean():.1f} years")
    
    print("\n" + "-" * 80)
    print("THERMAL PERFORMANCE:")
    print("-" * 80)
    print(f"Average envelope U-value: {df['envelope_avg_u_value'].mean():.3f} W/m²K")
    print(f"Best U-value: {df['envelope_avg_u_value'].min():.3f} W/m²K")
    print(f"Worst U-value: {df['envelope_avg_u_value'].max():.3f} W/m²K")
    
    print("\n✅ Building attributes data generation complete!")
    print(f"📁 Saved to: ../data/building_attributes.csv")
    print(f"📁 Saved to: ../data/building_attributes.xlsx")
    print(f"📁 Material database: ../data/material_properties_database.json")


if __name__ == "__main__":
    main()
