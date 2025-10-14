#!/usr/bin/env python3
"""
Generate Lifecycle Assessment (LCA) data for building retrofit research.
Creates Environmental Product Declarations (EPDs), carbon footprint data,
and LCA databases for building materials and processes.
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple
import random

class LCADataGenerator:
    def __init__(self):
        self.material_categories = [
            'concrete', 'steel', 'brick', 'wood', 'glass', 'insulation',
            'roofing', 'flooring', 'paint', 'adhesive', 'sealant'
        ]
        
        self.lca_phases = [
            'raw_material_extraction', 'material_processing', 'transportation',
            'construction', 'use_phase', 'maintenance', 'end_of_life'
        ]
        
        self.epd_programs = [
            'EN 15804', 'ISO 14025', 'ASTM E2129', 'BRE Global',
            'UL Environment', 'IBU', 'EcoPlatform'
        ]
        
    def generate_material_epds(self) -> pd.DataFrame:
        """Generate Environmental Product Declarations for building materials."""
        
        epd_data = []
        
        # Material-specific EPD data
        material_properties = {
            'concrete': {
                'density_kg_m3': 2400,
                'gwp_kg_co2e_m3': 200,
                'odp_kg_cfc11e_m3': 0.0001,
                'pocp_kg_ethene_m3': 0.5,
                'ap_kg_so2e_m3': 1.2,
                'ep_kg_po4e_m3': 0.3,
                'adp_kg_sbe_m3': 0.01
            },
            'steel': {
                'density_kg_m3': 7850,
                'gwp_kg_co2e_m3': 1200,
                'odp_kg_cfc11e_m3': 0.0005,
                'pocp_kg_ethene_m3': 2.0,
                'ap_kg_so2e_m3': 8.0,
                'ep_kg_po4e_m3': 1.5,
                'adp_kg_sbe_m3': 0.05
            },
            'brick': {
                'density_kg_m3': 1800,
                'gwp_kg_co2e_m3': 150,
                'odp_kg_cfc11e_m3': 0.00005,
                'pocp_kg_ethene_m3': 0.3,
                'ap_kg_so2e_m3': 0.8,
                'ep_kg_po4e_m3': 0.2,
                'adp_kg_sbe_m3': 0.005
            },
            'wood': {
                'density_kg_m3': 500,
                'gwp_kg_co2e_m3': -50,  # Negative due to carbon storage
                'odp_kg_cfc11e_m3': 0.00001,
                'pocp_kg_ethene_m3': 0.1,
                'ap_kg_so2e_m3': 0.3,
                'ep_kg_po4e_m3': 0.05,
                'adp_kg_sbe_m3': 0.001
            },
            'glass': {
                'density_kg_m3': 2500,
                'gwp_kg_co2e_m3': 400,
                'odp_kg_cfc11e_m3': 0.0002,
                'pocp_kg_ethene_m3': 1.0,
                'ap_kg_so2e_m3': 2.0,
                'ep_kg_po4e_m3': 0.4,
                'adp_kg_sbe_m3': 0.02
            },
            'insulation': {
                'density_kg_m3': 50,
                'gwp_kg_co2e_m3': 20,
                'odp_kg_cfc11e_m3': 0.00001,
                'pocp_kg_ethene_m3': 0.05,
                'ap_kg_so2e_m3': 0.1,
                'ep_kg_po4e_m3': 0.02,
                'adp_kg_sbe_m3': 0.001
            }
        }
        
        for material in self.material_categories:
            # Get base properties or use defaults
            base_props = material_properties.get(material, {
                'density_kg_m3': 1000,
                'gwp_kg_co2e_m3': 100,
                'odp_kg_cfc11e_m3': 0.0001,
                'pocp_kg_ethene_m3': 0.5,
                'ap_kg_so2e_m3': 1.0,
                'ep_kg_po4e_m3': 0.2,
                'adp_kg_sbe_m3': 0.01
            })
            
            # Add variation to base values
            for prop, base_value in base_props.items():
                if prop == 'density_kg_m3':
                    variation = random.uniform(0.9, 1.1)
                else:
                    variation = random.uniform(0.8, 1.2)
                
                base_props[prop] = base_value * variation
            
            # Generate EPD entry
            epd_entry = {
                'material_id': f"EPD_{material.upper()}_{random.randint(1000, 9999)}",
                'material_name': material,
                'material_category': material,
                'epd_program': random.choice(self.epd_programs),
                'epd_valid_from': f"2020-{random.randint(1, 12):02d}-01",
                'epd_valid_until': f"2025-{random.randint(1, 12):02d}-01",
                'functional_unit': '1 m³',
                'density_kg_m3': round(base_props['density_kg_m3'], 2),
                'gwp_kg_co2e_m3': round(base_props['gwp_kg_co2e_m3'], 3),
                'odp_kg_cfc11e_m3': round(base_props['odp_kg_cfc11e_m3'], 6),
                'pocp_kg_ethene_m3': round(base_props['pocp_kg_ethene_m3'], 3),
                'ap_kg_so2e_m3': round(base_props['ap_kg_so2e_m3'], 3),
                'ep_kg_po4e_m3': round(base_props['ep_kg_po4e_m3'], 3),
                'adp_kg_sbe_m3': round(base_props['adp_kg_sbe_m3'], 4),
                'renewable_energy_percent': round(random.uniform(10, 80), 1),
                'recycled_content_percent': round(random.uniform(0, 50), 1),
                'recyclability_percent': round(random.uniform(60, 95), 1)
            }
            
            epd_data.append(epd_entry)
        
        return pd.DataFrame(epd_data)
    
    def generate_construction_process_lca(self) -> pd.DataFrame:
        """Generate LCA data for construction processes."""
        
        processes = [
            'excavation', 'foundation_pouring', 'wall_construction', 'roof_installation',
            'window_installation', 'insulation_installation', 'flooring_installation',
            'painting', 'electrical_installation', 'plumbing_installation',
            'hvac_installation', 'demolition', 'waste_disposal'
        ]
        
        process_data = []
        
        for process in processes:
            # Base environmental impact per unit process
            base_impacts = {
                'excavation': {'gwp': 5, 'energy': 50, 'water': 10},
                'foundation_pouring': {'gwp': 20, 'energy': 200, 'water': 50},
                'wall_construction': {'gwp': 15, 'energy': 150, 'water': 30},
                'roof_installation': {'gwp': 10, 'energy': 100, 'water': 20},
                'window_installation': {'gwp': 8, 'energy': 80, 'water': 15},
                'insulation_installation': {'gwp': 3, 'energy': 30, 'water': 5},
                'flooring_installation': {'gwp': 12, 'energy': 120, 'water': 25},
                'painting': {'gwp': 2, 'energy': 20, 'water': 10},
                'electrical_installation': {'gwp': 4, 'energy': 40, 'water': 8},
                'plumbing_installation': {'gwp': 6, 'energy': 60, 'water': 12},
                'hvac_installation': {'gwp': 25, 'energy': 250, 'water': 40},
                'demolition': {'gwp': 8, 'energy': 80, 'water': 15},
                'waste_disposal': {'gwp': 3, 'energy': 30, 'water': 5}
            }
            
            base = base_impacts.get(process, {'gwp': 10, 'energy': 100, 'water': 20})
            
            # Add variation
            gwp = base['gwp'] * random.uniform(0.8, 1.2)
            energy = base['energy'] * random.uniform(0.8, 1.2)
            water = base['water'] * random.uniform(0.8, 1.2)
            
            process_entry = {
                'process_id': f"PROC_{process.upper()}_{random.randint(100, 999)}",
                'process_name': process,
                'process_category': self.categorize_process(process),
                'unit': 'per m²',
                'gwp_kg_co2e_m2': round(gwp, 3),
                'energy_consumption_mj_m2': round(energy, 2),
                'water_consumption_l_m2': round(water, 2),
                'waste_generation_kg_m2': round(random.uniform(0.5, 5.0), 2),
                'noise_level_db': round(random.uniform(60, 90), 1),
                'dust_emission_kg_m2': round(random.uniform(0.1, 2.0), 3)
            }
            
            process_data.append(process_entry)
        
        return pd.DataFrame(process_data)
    
    def categorize_process(self, process: str) -> str:
        """Categorize construction process."""
        if process in ['excavation', 'foundation_pouring']:
            return 'foundation'
        elif process in ['wall_construction', 'roof_installation']:
            return 'structure'
        elif process in ['window_installation', 'insulation_installation']:
            return 'envelope'
        elif process in ['flooring_installation', 'painting']:
            return 'finishing'
        elif process in ['electrical_installation', 'plumbing_installation', 'hvac_installation']:
            return 'mep'
        else:
            return 'other'
    
    def generate_building_lca_database(self, buildings: List[Dict]) -> pd.DataFrame:
        """Generate LCA data for complete buildings."""
        
        building_lca_data = []
        
        for building in buildings:
            building_id = building['id']
            building_type = building['type']
            construction_year = building['construction_year']
            floor_area = building['floor_area_m2']
            
            # Calculate building-level LCA impacts
            # These would typically be calculated from material quantities and EPDs
            # Here we use simplified calculations
            
            # Base impacts per m² by building type
            base_impacts = {
                'residential': {'gwp': 300, 'energy': 3000, 'water': 500},
                'office': {'gwp': 400, 'energy': 4000, 'water': 600},
                'retail': {'gwp': 350, 'energy': 3500, 'water': 550},
                'educational': {'gwp': 380, 'energy': 3800, 'water': 580},
                'healthcare': {'gwp': 450, 'energy': 4500, 'water': 700},
                'industrial': {'gwp': 500, 'energy': 5000, 'water': 800}
            }
            
            config = base_impacts.get(building_type, base_impacts['office'])
            
            # Age factor (older buildings may have different impacts)
            age_factor = 1.0 + (2023 - construction_year) * 0.01
            
            # Calculate total impacts
            total_gwp = config['gwp'] * floor_area * age_factor * random.uniform(0.8, 1.2)
            total_energy = config['energy'] * floor_area * age_factor * random.uniform(0.8, 1.2)
            total_water = config['water'] * floor_area * age_factor * random.uniform(0.8, 1.2)
            
            # Breakdown by life cycle phase
            phase_breakdown = {
                'raw_material_extraction': 0.15,
                'material_processing': 0.25,
                'transportation': 0.05,
                'construction': 0.10,
                'use_phase': 0.35,
                'maintenance': 0.05,
                'end_of_life': 0.05
            }
            
            building_entry = {
                'building_id': building_id,
                'building_type': building_type,
                'construction_year': construction_year,
                'floor_area_m2': floor_area,
                'total_gwp_kg_co2e': round(total_gwp, 2),
                'total_energy_mj': round(total_energy, 2),
                'total_water_l': round(total_water, 2),
                'gwp_per_m2_kg_co2e': round(total_gwp / floor_area, 2),
                'energy_per_m2_mj': round(total_energy / floor_area, 2),
                'water_per_m2_l': round(total_water / floor_area, 2)
            }
            
            # Add phase breakdown
            for phase, fraction in phase_breakdown.items():
                building_entry[f'{phase}_gwp'] = round(total_gwp * fraction, 2)
                building_entry[f'{phase}_energy'] = round(total_energy * fraction, 2)
                building_entry[f'{phase}_water'] = round(total_water * fraction, 2)
            
            # Add additional LCA indicators
            building_entry['renewable_energy_percent'] = round(random.uniform(10, 40), 1)
            building_entry['recycled_materials_percent'] = round(random.uniform(5, 30), 1)
            building_entry['waste_generation_kg'] = round(random.uniform(100, 1000), 2)
            building_entry['hazardous_waste_kg'] = round(random.uniform(10, 100), 2)
            
            building_lca_data.append(building_entry)
        
        return pd.DataFrame(building_lca_data)
    
    def generate_retrofit_lca_data(self, buildings: List[Dict]) -> pd.DataFrame:
        """Generate LCA data for retrofit scenarios."""
        
        retrofit_lca_data = []
        
        for building in buildings:
            building_id = building['id']
            building_type = building['type']
            
            # Determine retrofit scenarios
            retrofit_scenarios = [
                'no_retrofit', 'light_retrofit', 'deep_retrofit', 'comprehensive_retrofit'
            ]
            
            for scenario in retrofit_scenarios:
                # Retrofit impact factors
                impact_factors = {
                    'no_retrofit': {'gwp_reduction': 0, 'energy_reduction': 0, 'cost': 0},
                    'light_retrofit': {'gwp_reduction': 15, 'energy_reduction': 20, 'cost': 50},
                    'deep_retrofit': {'gwp_reduction': 35, 'energy_reduction': 45, 'cost': 150},
                    'comprehensive_retrofit': {'gwp_reduction': 60, 'energy_reduction': 70, 'cost': 300}
                }
                
                factors = impact_factors[scenario]
                
                # Calculate retrofit impacts
                retrofit_gwp = factors['gwp_reduction'] * random.uniform(0.8, 1.2)
                retrofit_energy = factors['energy_reduction'] * random.uniform(0.8, 1.2)
                retrofit_cost = factors['cost'] * random.uniform(0.8, 1.2)
                
                # Payback period calculation
                annual_savings = retrofit_energy * 0.12  # €0.12/kWh
                payback_period = retrofit_cost / annual_savings if annual_savings > 0 else None
                
                retrofit_entry = {
                    'building_id': building_id,
                    'retrofit_scenario': scenario,
                    'gwp_reduction_percent': round(retrofit_gwp, 2),
                    'energy_reduction_percent': round(retrofit_energy, 2),
                    'retrofit_cost_eur_m2': round(retrofit_cost, 2),
                    'payback_period_years': round(payback_period, 1) if payback_period else None,
                    'co2_savings_kg_co2e_m2': round(retrofit_gwp * 10, 2),  # Rough conversion
                    'energy_savings_kwh_m2': round(retrofit_energy * 2, 2),  # Rough conversion
                    'renewable_energy_addition_percent': round(random.uniform(0, 50), 1),
                    'material_recycling_percent': round(random.uniform(60, 90), 1)
                }
                
                retrofit_lca_data.append(retrofit_entry)
        
        return pd.DataFrame(retrofit_lca_data)

def main():
    """Generate LCA data for the building retrofit dataset."""
    
    # Sample buildings
    buildings = [
        {'id': 'B001', 'type': 'residential', 'construction_year': 1985, 'floor_area_m2': 120},
        {'id': 'B002', 'type': 'office', 'construction_year': 1995, 'floor_area_m2': 2500},
        {'id': 'B003', 'type': 'retail', 'construction_year': 2005, 'floor_area_m2': 800},
        {'id': 'B004', 'type': 'educational', 'construction_year': 1970, 'floor_area_m2': 3000},
        {'id': 'B005', 'type': 'residential', 'construction_year': 2010, 'floor_area_m2': 150},
        {'id': 'B006', 'type': 'office', 'construction_year': 1980, 'floor_area_m2': 1800},
        {'id': 'B007', 'type': 'retail', 'construction_year': 1990, 'floor_area_m2': 1200},
        {'id': 'B008', 'type': 'educational', 'construction_year': 2000, 'floor_area_m2': 4000},
        {'id': 'B009', 'type': 'residential', 'construction_year': 1965, 'floor_area_m2': 100},
        {'id': 'B010', 'type': 'office', 'construction_year': 2015, 'floor_area_m2': 3200}
    ]
    
    # Initialize generator
    generator = LCADataGenerator()
    
    # Generate LCA data
    print("Generating LCA data...")
    
    # Generate material EPDs
    print("Generating material EPDs...")
    epd_df = generator.generate_material_epds()
    
    # Generate construction process LCA
    print("Generating construction process LCA...")
    process_df = generator.generate_construction_process_lca()
    
    # Generate building LCA database
    print("Generating building LCA database...")
    building_lca_df = generator.generate_building_lca_database(buildings)
    
    # Generate retrofit LCA data
    print("Generating retrofit LCA data...")
    retrofit_lca_df = generator.generate_retrofit_lca_data(buildings)
    
    # Save data
    output_dir = '../raw_data/lca_data'
    os.makedirs(output_dir, exist_ok=True)
    
    epd_df.to_csv(f"{output_dir}/material_epds.csv", index=False)
    process_df.to_csv(f"{output_dir}/construction_process_lca.csv", index=False)
    building_lca_df.to_csv(f"{output_dir}/building_lca_database.csv", index=False)
    retrofit_lca_df.to_csv(f"{output_dir}/retrofit_lca_scenarios.csv", index=False)
    
    print(f"Saved material EPDs: {len(epd_df)} records")
    print(f"Saved construction process LCA: {len(process_df)} records")
    print(f"Saved building LCA database: {len(building_lca_df)} records")
    print(f"Saved retrofit LCA scenarios: {len(retrofit_lca_df)} records")
    
    # Create summary
    summary = {
        'total_materials': len(epd_df),
        'total_processes': len(process_df),
        'total_buildings': len(building_lca_df),
        'total_retrofit_scenarios': len(retrofit_lca_df),
        'material_categories': epd_df['material_category'].value_counts().to_dict(),
        'epd_programs': epd_df['epd_program'].value_counts().to_dict(),
        'retrofit_scenarios': retrofit_lca_df['retrofit_scenario'].value_counts().to_dict()
    }
    
    with open(f"{output_dir}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nLCA data generation complete!")
    print(f"Generated EPDs for {len(epd_df)} materials")
    print(f"Generated LCA data for {len(process_df)} construction processes")
    print(f"Generated LCA data for {len(building_lca_df)} buildings")
    print(f"Generated {len(retrofit_lca_df)} retrofit scenarios")

if __name__ == "__main__":
    main()