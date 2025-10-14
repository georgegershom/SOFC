"""
Lifecycle Assessment (LCA) Data Generator
Generates comprehensive LCA data including:
- Environmental Product Declarations (EPDs) for materials
- Carbon footprint data for construction materials and processes
- Embodied carbon calculations
- Whole lifecycle environmental impacts
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, List, Tuple

class LCADataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Material carbon intensity database (kgCO2e/kg or kgCO2e/m³)
        self.material_carbon_data = {
            'concrete': {
                'embodied_carbon_kg_co2_m3': (200, 400),
                'density_kg_m3': 2400,
                'recyclability_pct': (20, 40),
                'lifespan_years': (50, 100)
            },
            'steel': {
                'embodied_carbon_kg_co2_kg': (1.5, 2.5),
                'density_kg_m3': 7850,
                'recyclability_pct': (90, 95),
                'lifespan_years': (50, 100)
            },
            'timber': {
                'embodied_carbon_kg_co2_m3': (-500, -200),  # Carbon negative
                'density_kg_m3': 500,
                'recyclability_pct': (60, 80),
                'lifespan_years': (30, 80)
            },
            'brick': {
                'embodied_carbon_kg_co2_kg': (0.2, 0.3),
                'density_kg_m3': 1900,
                'recyclability_pct': (80, 90),
                'lifespan_years': (100, 150)
            },
            'glass': {
                'embodied_carbon_kg_co2_kg': (0.8, 1.2),
                'density_kg_m3': 2500,
                'recyclability_pct': (90, 100),
                'lifespan_years': (30, 50)
            },
            'insulation_mineral_wool': {
                'embodied_carbon_kg_co2_kg': (1.0, 1.5),
                'density_kg_m3': 100,
                'recyclability_pct': (50, 70),
                'lifespan_years': (30, 50)
            },
            'insulation_polyurethane': {
                'embodied_carbon_kg_co2_kg': (3.0, 5.0),
                'density_kg_m3': 30,
                'recyclability_pct': (10, 30),
                'lifespan_years': (25, 40)
            },
            'aluminum': {
                'embodied_carbon_kg_co2_kg': (8.0, 12.0),
                'density_kg_m3': 2700,
                'recyclability_pct': (95, 100),
                'lifespan_years': (40, 60)
            }
        }
        
        # LCA impact categories
        self.impact_categories = [
            'global_warming_potential',
            'ozone_depletion_potential',
            'acidification_potential',
            'eutrophication_potential',
            'photochemical_oxidation_potential',
            'abiotic_depletion_potential',
            'water_consumption',
            'primary_energy_demand'
        ]
        
    def generate_material_epd(self, material_type: str, quantity: float, 
                             unit: str = 'kg') -> Dict:
        """Generate Environmental Product Declaration for a material"""
        
        if material_type not in self.material_carbon_data:
            material_type = random.choice(list(self.material_carbon_data.keys()))
        
        material_data = self.material_carbon_data[material_type]
        
        # Calculate embodied carbon
        if 'embodied_carbon_kg_co2_kg' in material_data:
            carbon_per_unit = random.uniform(*material_data['embodied_carbon_kg_co2_kg'])
            total_embodied_carbon = carbon_per_unit * quantity
        else:
            carbon_per_unit = random.uniform(*material_data['embodied_carbon_kg_co2_m3'])
            total_embodied_carbon = carbon_per_unit * quantity / 1000  # Convert to tonnes
        
        # Generate other environmental impacts
        epd_data = {
            'material_type': material_type,
            'quantity': quantity,
            'unit': unit,
            'epd_number': f"EPD-{random.randint(1000, 9999)}-{material_type.upper()[:3]}",
            'manufacturer': f"{material_type.capitalize()} Corp {random.randint(1, 100)}",
            'production_location': random.choice(['Europe', 'Asia', 'North America', 'Local']),
            'transport_distance_km': random.randint(50, 2000),
            
            # A1-A3: Product stage
            'a1_a3_embodied_carbon_kg_co2': round(total_embodied_carbon, 2),
            'a1_raw_material_kg_co2': round(total_embodied_carbon * 0.3, 2),
            'a2_transport_kg_co2': round(total_embodied_carbon * 0.1, 2),
            'a3_manufacturing_kg_co2': round(total_embodied_carbon * 0.6, 2),
            
            # A4-A5: Construction stage
            'a4_transport_to_site_kg_co2': round(quantity * 0.05, 2),
            'a5_installation_kg_co2': round(quantity * 0.02, 2),
            
            # B1-B7: Use stage
            'b1_b7_use_stage_kg_co2_per_year': round(quantity * 0.001, 3),
            
            # C1-C4: End of life
            'c1_deconstruction_kg_co2': round(quantity * 0.01, 2),
            'c2_transport_eol_kg_co2': round(quantity * 0.03, 2),
            'c3_waste_processing_kg_co2': round(quantity * 0.02, 2),
            'c4_disposal_kg_co2': round(quantity * 0.05, 2),
            
            # D: Benefits beyond system
            'd_recycling_potential_kg_co2': round(-total_embodied_carbon * 
                                                  random.uniform(*material_data['recyclability_pct']) / 100 * 0.5, 2),
            
            # Other properties
            'density_kg_m3': material_data['density_kg_m3'],
            'recyclability_pct': round(random.uniform(*material_data['recyclability_pct']), 1),
            'expected_lifespan_years': random.randint(*material_data['lifespan_years']),
            'renewable_content_pct': random.uniform(0, 100) if 'timber' in material_type else random.uniform(0, 20),
            'recycled_content_pct': random.uniform(0, 50),
            
            # Certification
            'certification_standard': random.choice(['EN 15804', 'ISO 14025', 'ISO 21930']),
            'valid_from': '2023-01-01',
            'valid_until': '2028-01-01'
        }
        
        # Add other impact categories
        for impact in self.impact_categories[1:]:  # Skip GWP as we already have it
            epd_data[f'{impact}_value'] = round(random.uniform(0, 10) * quantity, 3)
            epd_data[f'{impact}_unit'] = self._get_impact_unit(impact)
        
        return epd_data
    
    def generate_building_lca(self, building_id: str, floor_area: float,
                             materials: Dict[str, str]) -> pd.DataFrame:
        """Generate complete building LCA data"""
        
        lca_components = []
        
        # Estimate material quantities based on floor area
        material_quantities = self._estimate_material_quantities(floor_area, materials)
        
        for component, (material, quantity, unit) in material_quantities.items():
            epd = self.generate_material_epd(material, quantity, unit)
            epd['building_id'] = building_id
            epd['component'] = component
            lca_components.append(epd)
        
        return pd.DataFrame(lca_components)
    
    def _estimate_material_quantities(self, floor_area: float, 
                                     materials: Dict[str, str]) -> Dict:
        """Estimate material quantities based on building size"""
        
        quantities = {}
        
        # Structure
        if materials.get('structure') == 'concrete':
            quantities['structure'] = ('concrete', floor_area * 0.5, 'm3')  # 0.5 m³/m²
        elif materials.get('structure') == 'steel':
            quantities['structure'] = ('steel', floor_area * 50, 'kg')  # 50 kg/m²
        elif materials.get('structure') == 'timber':
            quantities['structure'] = ('timber', floor_area * 0.3, 'm3')
        else:
            quantities['structure'] = ('concrete', floor_area * 0.4, 'm3')
        
        # Facade
        wall_area = floor_area * 1.5  # Approximate wall to floor ratio
        if materials.get('facade') == 'brick':
            quantities['facade'] = ('brick', wall_area * 100, 'kg')  # 100 kg/m²
        elif materials.get('facade') == 'glass':
            quantities['facade'] = ('glass', wall_area * 25, 'kg')
        else:
            quantities['facade'] = ('concrete', wall_area * 0.2, 'm3')
        
        # Insulation
        if 'mineral_wool' in materials.get('insulation', ''):
            quantities['insulation'] = ('insulation_mineral_wool', wall_area * 10, 'kg')
        elif 'polyurethane' in materials.get('insulation', ''):
            quantities['insulation'] = ('insulation_polyurethane', wall_area * 3, 'kg')
        else:
            quantities['insulation'] = ('insulation_mineral_wool', wall_area * 8, 'kg')
        
        # Windows (assume 30% of wall area)
        window_area = wall_area * 0.3
        quantities['windows'] = ('glass', window_area * 20, 'kg')  # Double glazing
        quantities['window_frames'] = ('aluminum', window_area * 5, 'kg')
        
        return quantities
    
    def generate_retrofit_lca(self, building_id: str, retrofit_measures: List[str],
                            floor_area: float) -> pd.DataFrame:
        """Generate LCA data for retrofit measures"""
        
        retrofit_lca = []
        
        # Material requirements for common retrofit measures
        retrofit_materials = {
            'wall_insulation': [
                ('insulation_polyurethane', floor_area * 1.5 * 5, 'kg'),  # Wall area * thickness
                ('aluminum', floor_area * 0.5, 'kg')  # Fixings
            ],
            'roof_insulation': [
                ('insulation_mineral_wool', floor_area * 0.3 * 15, 'kg'),
            ],
            'window_upgrade': [
                ('glass', floor_area * 0.3 * 30, 'kg'),  # Triple glazing
                ('aluminum', floor_area * 0.3 * 8, 'kg')
            ],
            'solar_panels': [
                ('glass', floor_area * 0.2 * 15, 'kg'),  # Panel area
                ('aluminum', floor_area * 0.2 * 10, 'kg'),
                ('steel', floor_area * 0.2 * 5, 'kg')  # Mounting
            ],
            'heat_pump': [
                ('steel', 200, 'kg'),
                ('aluminum', 50, 'kg')
            ]
        }
        
        for measure in retrofit_measures:
            if measure in retrofit_materials:
                for material, quantity, unit in retrofit_materials[measure]:
                    epd = self.generate_material_epd(material, quantity, unit)
                    epd['building_id'] = building_id
                    epd['retrofit_measure'] = measure
                    retrofit_lca.append(epd)
        
        return pd.DataFrame(retrofit_lca)
    
    def generate_lifecycle_impacts(self, building_id: str, building_lca: pd.DataFrame,
                                  lifespan_years: int = 50) -> Dict:
        """Calculate whole lifecycle environmental impacts"""
        
        # Sum impacts by lifecycle stage
        stages = ['a1_a3', 'a4', 'a5', 'b1_b7', 'c1', 'c2', 'c3', 'c4', 'd']
        
        lifecycle_impacts = {
            'building_id': building_id,
            'assessment_period_years': lifespan_years
        }
        
        for stage in stages:
            stage_columns = [col for col in building_lca.columns if col.startswith(stage)]
            for col in stage_columns:
                if 'kg_co2' in col:
                    if 'per_year' in col:
                        lifecycle_impacts[col.replace('_per_year', '_total')] = \
                            round(building_lca[col].sum() * lifespan_years, 2)
                    else:
                        lifecycle_impacts[col + '_total'] = round(building_lca[col].sum(), 2)
        
        # Calculate total carbon footprint
        total_embodied = building_lca['a1_a3_embodied_carbon_kg_co2'].sum()
        total_operational = building_lca['b1_b7_use_stage_kg_co2_per_year'].sum() * lifespan_years
        total_eol = sum(building_lca[f'{stage}_kg_co2'].sum() 
                       for stage in ['c1_deconstruction', 'c2_transport_eol', 
                                    'c3_waste_processing', 'c4_disposal'] 
                       if f'{stage}_kg_co2' in building_lca.columns)
        recycling_credit = building_lca['d_recycling_potential_kg_co2'].sum()
        
        lifecycle_impacts.update({
            'total_embodied_carbon_kg_co2': round(total_embodied, 2),
            'total_operational_carbon_kg_co2': round(total_operational, 2),
            'total_end_of_life_carbon_kg_co2': round(total_eol, 2),
            'total_recycling_credit_kg_co2': round(recycling_credit, 2),
            'net_lifecycle_carbon_kg_co2': round(total_embodied + total_operational + 
                                                 total_eol + recycling_credit, 2),
            'carbon_per_m2_per_year_kg_co2': round((total_embodied + total_operational + 
                                                    total_eol + recycling_credit) / 
                                                   (building_lca['quantity'].sum() * lifespan_years), 3)
        })
        
        # Add other impact categories
        for impact in self.impact_categories[1:]:
            col_name = f'{impact}_value'
            if col_name in building_lca.columns:
                lifecycle_impacts[f'total_{impact}'] = round(building_lca[col_name].sum(), 2)
        
        return lifecycle_impacts
    
    def _get_impact_unit(self, impact_category: str) -> str:
        """Get unit for impact category"""
        units = {
            'ozone_depletion_potential': 'kg CFC-11 eq',
            'acidification_potential': 'kg SO2 eq',
            'eutrophication_potential': 'kg PO4 eq',
            'photochemical_oxidation_potential': 'kg C2H4 eq',
            'abiotic_depletion_potential': 'kg Sb eq',
            'water_consumption': 'm3',
            'primary_energy_demand': 'MJ'
        }
        return units.get(impact_category, 'unit')
    
    def generate_carbon_offset_potential(self, building_id: str, 
                                        retrofit_measures: List[str],
                                        floor_area: float) -> Dict:
        """Calculate carbon offset potential from retrofits"""
        
        # Carbon savings potential per measure (kgCO2/m²/year)
        savings_potential = {
            'wall_insulation': (10, 25),
            'roof_insulation': (8, 20),
            'window_upgrade': (5, 15),
            'solar_panels': (30, 60),
            'heat_pump': (40, 80),
            'green_roof': (2, 5),
            'led_lighting': (3, 8),
            'building_controls': (5, 15)
        }
        
        total_savings = 0
        measure_savings = {}
        
        for measure in retrofit_measures:
            if measure in savings_potential:
                savings = random.uniform(*savings_potential[measure]) * floor_area
                measure_savings[measure] = round(savings, 2)
                total_savings += savings
        
        return {
            'building_id': building_id,
            'annual_carbon_savings_kg_co2': round(total_savings, 2),
            'lifetime_carbon_savings_kg_co2': round(total_savings * 25, 2),  # 25-year period
            'carbon_payback_period_years': round(random.uniform(3, 15), 1),
            'measure_savings': measure_savings,
            'carbon_credits_potential': round(total_savings / 1000, 3),  # Tonnes CO2
            'monetary_value_carbon_credits_eur': round(total_savings / 1000 * 
                                                       random.uniform(20, 50), 2)
        }