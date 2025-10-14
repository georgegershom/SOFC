"""
Lifecycle Assessment (LCA) Dataset Generator for Building Retrofit Research
Generates comprehensive LCA data including EPDs, carbon footprint, and environmental impacts.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

np.random.seed(42)


class LCADataGenerator:
    """Generate lifecycle assessment data for building materials and retrofit measures."""
    
    def __init__(self):
        # Environmental Product Declaration (EPD) database for construction materials
        self.epd_database = {
            # Wall materials
            'brick_clay': {
                'gwp_kgco2eq_kg': 0.24,
                'embodied_energy_mj_kg': 3.0,
                'water_usage_l_kg': 1.5,
                'recyclability_pct': 75,
                'density_kg_m3': 1920,
                'thermal_conductivity': 0.77,
                'lifespan_years': 100
            },
            'concrete_block': {
                'gwp_kgco2eq_kg': 0.16,
                'embodied_energy_mj_kg': 1.5,
                'water_usage_l_kg': 2.0,
                'recyclability_pct': 80,
                'density_kg_m3': 2000,
                'thermal_conductivity': 1.4,
                'lifespan_years': 100
            },
            'mineral_wool_insulation': {
                'gwp_kgco2eq_kg': 1.28,
                'embodied_energy_mj_kg': 16.8,
                'water_usage_l_kg': 0.5,
                'recyclability_pct': 60,
                'density_kg_m3': 40,
                'thermal_conductivity': 0.035,
                'lifespan_years': 50
            },
            'eps_insulation': {
                'gwp_kgco2eq_kg': 3.48,
                'embodied_energy_mj_kg': 88.6,
                'water_usage_l_kg': 0.2,
                'recyclability_pct': 95,
                'density_kg_m3': 25,
                'thermal_conductivity': 0.033,
                'lifespan_years': 50
            },
            'wood_timber': {
                'gwp_kgco2eq_kg': -0.71,  # Carbon sequestration
                'embodied_energy_mj_kg': 7.4,
                'water_usage_l_kg': 0.8,
                'recyclability_pct': 90,
                'density_kg_m3': 500,
                'thermal_conductivity': 0.13,
                'lifespan_years': 60
            },
            'steel_structural': {
                'gwp_kgco2eq_kg': 1.77,
                'embodied_energy_mj_kg': 24.4,
                'water_usage_l_kg': 0.3,
                'recyclability_pct': 98,
                'density_kg_m3': 7850,
                'thermal_conductivity': 50,
                'lifespan_years': 80
            },
            'aluminum': {
                'gwp_kgco2eq_kg': 8.24,
                'embodied_energy_mj_kg': 170,
                'water_usage_l_kg': 0.4,
                'recyclability_pct': 95,
                'density_kg_m3': 2700,
                'thermal_conductivity': 237,
                'lifespan_years': 50
            },
            'glass_double_glazed': {
                'gwp_kgco2eq_kg': 0.85,
                'embodied_energy_mj_kg': 15.9,
                'water_usage_l_kg': 0.3,
                'recyclability_pct': 100,
                'density_kg_m3': 2500,
                'thermal_conductivity': 1.0,
                'lifespan_years': 35
            },
            'glass_triple_glazed': {
                'gwp_kgco2eq_kg': 1.15,
                'embodied_energy_mj_kg': 23.8,
                'water_usage_l_kg': 0.4,
                'recyclability_pct': 100,
                'density_kg_m3': 2500,
                'thermal_conductivity': 1.0,
                'lifespan_years': 35
            },
            'concrete_reinforced': {
                'gwp_kgco2eq_kg': 0.32,
                'embodied_energy_mj_kg': 2.0,
                'water_usage_l_kg': 3.5,
                'recyclability_pct': 70,
                'density_kg_m3': 2400,
                'thermal_conductivity': 2.5,
                'lifespan_years': 100
            },
            'gypsum_board': {
                'gwp_kgco2eq_kg': 0.38,
                'embodied_energy_mj_kg': 6.1,
                'water_usage_l_kg': 0.6,
                'recyclability_pct': 100,
                'density_kg_m3': 900,
                'thermal_conductivity': 0.25,
                'lifespan_years': 30
            },
            'ceramic_tiles': {
                'gwp_kgco2eq_kg': 0.62,
                'embodied_energy_mj_kg': 12.5,
                'water_usage_l_kg': 1.2,
                'recyclability_pct': 50,
                'density_kg_m3': 2000,
                'thermal_conductivity': 1.3,
                'lifespan_years': 50
            },
            'paint_coating': {
                'gwp_kgco2eq_kg': 2.91,
                'embodied_energy_mj_kg': 61.5,
                'water_usage_l_kg': 0.8,
                'recyclability_pct': 10,
                'density_kg_m3': 1400,
                'thermal_conductivity': 0.3,
                'lifespan_years': 10
            }
        }
        
        # HVAC systems LCA data
        self.hvac_lca = {
            'gas_boiler': {
                'manufacturing_gwp_kgco2eq': 450,
                'manufacturing_energy_mj': 8500,
                'operational_gwp_kgco2eq_year': 2500,
                'operational_energy_kwh_year': 12000,
                'end_of_life_gwp_kgco2eq': -50,
                'recyclability_pct': 75,
                'lifespan_years': 15
            },
            'condensing_boiler': {
                'manufacturing_gwp_kgco2eq': 520,
                'manufacturing_energy_mj': 9200,
                'operational_gwp_kgco2eq_year': 1800,
                'operational_energy_kwh_year': 9000,
                'end_of_life_gwp_kgco2eq': -60,
                'recyclability_pct': 80,
                'lifespan_years': 18
            },
            'air_source_heat_pump': {
                'manufacturing_gwp_kgco2eq': 1200,
                'manufacturing_energy_mj': 15500,
                'operational_gwp_kgco2eq_year': 800,
                'operational_energy_kwh_year': 4500,
                'end_of_life_gwp_kgco2eq': -150,
                'recyclability_pct': 85,
                'lifespan_years': 20
            },
            'ground_source_heat_pump': {
                'manufacturing_gwp_kgco2eq': 2800,
                'manufacturing_energy_mj': 28000,
                'operational_gwp_kgco2eq_year': 600,
                'operational_energy_kwh_year': 3500,
                'end_of_life_gwp_kgco2eq': -200,
                'recyclability_pct': 80,
                'lifespan_years': 25
            },
            'solar_pv': {
                'manufacturing_gwp_kgco2eq_kwp': 1800,
                'manufacturing_energy_mj_kwp': 25000,
                'operational_gwp_kgco2eq_year': -500,  # Negative = avoidance
                'operational_energy_kwh_year_kwp': 1200,
                'end_of_life_gwp_kgco2eq_kwp': -180,
                'recyclability_pct': 90,
                'lifespan_years': 25
            }
        }
        
        # Construction processes impact
        self.construction_processes = {
            'demolition': {
                'gwp_kgco2eq_m2': 5.2,
                'energy_mj_m2': 45,
                'waste_generation_kg_m2': 120,
                'dust_emissions_kg_m2': 0.8
            },
            'transportation_materials': {
                'gwp_kgco2eq_tkm': 0.062,  # per tonne-km
                'energy_mj_tkm': 0.92
            },
            'installation_labor': {
                'gwp_kgco2eq_hour': 0.5,
                'energy_mj_hour': 8.5
            },
            'waste_disposal': {
                'gwp_kgco2eq_tonne': 45,
                'recycling_benefit_kgco2eq_tonne': -120
            }
        }
    
    def calculate_material_lca(self, material_name, quantity_kg, area_m2):
        """Calculate LCA metrics for a specific material quantity."""
        if material_name not in self.epd_database:
            return None
        
        epd = self.epd_database[material_name]
        
        return {
            'material': material_name,
            'quantity_kg': round(quantity_kg, 2),
            'area_m2': round(area_m2, 2),
            'gwp_total_kgco2eq': round(quantity_kg * epd['gwp_kgco2eq_kg'], 2),
            'gwp_per_m2_kgco2eq': round((quantity_kg * epd['gwp_kgco2eq_kg']) / area_m2, 2),
            'embodied_energy_total_mj': round(quantity_kg * epd['embodied_energy_mj_kg'], 2),
            'embodied_energy_per_m2_mj': round((quantity_kg * epd['embodied_energy_mj_kg']) / area_m2, 2),
            'water_usage_total_l': round(quantity_kg * epd['water_usage_l_kg'], 2),
            'recyclability_pct': epd['recyclability_pct'],
            'lifespan_years': epd['lifespan_years']
        }
    
    def calculate_building_lca(self, building_attrs):
        """Calculate whole-building lifecycle assessment."""
        building_id = building_attrs['building_id']
        floor_area = building_attrs['total_floor_area_m2']
        wall_area = building_attrs['wall_area_m2']
        roof_area = building_attrs['rooftop_area_m2']
        window_area = building_attrs['window_area_m2']
        volume = building_attrs['volume_m3']
        
        # Estimate material quantities based on building type and construction
        wall_material = building_attrs['wall_material']
        roof_material = building_attrs['roof_material']
        window_type = building_attrs['window_type']
        
        materials_lca = []
        
        # Wall materials calculation
        wall_thickness = 0.30  # 30cm average
        if 'brick' in wall_material:
            brick_qty = wall_area * wall_thickness * self.epd_database['brick_clay']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('brick_clay', brick_qty, wall_area))
        elif 'concrete' in wall_material:
            concrete_qty = wall_area * wall_thickness * self.epd_database['concrete_block']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('concrete_block', concrete_qty, wall_area))
        
        # Insulation (if insulated walls)
        if 'insulated' in wall_material or 'cavity' in wall_material:
            insulation_thickness = 0.10
            insulation_qty = wall_area * insulation_thickness * self.epd_database['mineral_wool_insulation']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('mineral_wool_insulation', insulation_qty, wall_area))
        
        # Roof materials
        roof_thickness = 0.25
        concrete_roof_qty = roof_area * roof_thickness * self.epd_database['concrete_reinforced']['density_kg_m3']
        materials_lca.append(self.calculate_material_lca('concrete_reinforced', concrete_roof_qty, roof_area))
        
        if 'insulated' in roof_material:
            roof_insulation_thickness = 0.15
            roof_insulation_qty = roof_area * roof_insulation_thickness * self.epd_database['mineral_wool_insulation']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('mineral_wool_insulation', roof_insulation_qty, roof_area))
        
        # Windows
        window_thickness = 0.025 if 'single' in window_type else 0.040
        if 'triple' in window_type:
            window_qty = window_area * window_thickness * self.epd_database['glass_triple_glazed']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('glass_triple_glazed', window_qty, window_area))
        else:
            window_qty = window_area * window_thickness * self.epd_database['glass_double_glazed']['density_kg_m3']
            materials_lca.append(self.calculate_material_lca('glass_double_glazed', window_qty, window_area))
        
        # Aluminum frames for windows
        frame_qty = window_area * 5  # 5 kg/m2 for frames
        materials_lca.append(self.calculate_material_lca('aluminum', frame_qty, window_area))
        
        # Interior finishes
        gypsum_qty = floor_area * 2 * 0.0125 * self.epd_database['gypsum_board']['density_kg_m3']
        materials_lca.append(self.calculate_material_lca('gypsum_board', gypsum_qty, floor_area * 2))
        
        # Calculate totals
        total_gwp = sum([m['gwp_total_kgco2eq'] for m in materials_lca if m])
        total_embodied_energy = sum([m['embodied_energy_total_mj'] for m in materials_lca if m])
        total_water = sum([m['water_usage_total_l'] for m in materials_lca if m])
        
        # Construction process impacts
        construction_gwp = floor_area * 15  # Simplified estimate
        transport_distance_km = np.random.uniform(50, 200)
        total_material_weight = sum([m['quantity_kg'] for m in materials_lca if m])
        transport_gwp = (total_material_weight / 1000) * transport_distance_km * self.construction_processes['transportation_materials']['gwp_kgco2eq_tkm']
        
        # HVAC system LCA
        hvac_type = building_attrs.get('hvac_system', 'gas_boiler_old')
        hvac_key = 'gas_boiler'
        if 'heat_pump' in hvac_type:
            if 'ground' in hvac_type:
                hvac_key = 'ground_source_heat_pump'
            else:
                hvac_key = 'air_source_heat_pump'
        elif 'condensing' in hvac_type:
            hvac_key = 'condensing_boiler'
        
        hvac_data = self.hvac_lca.get(hvac_key, self.hvac_lca['gas_boiler'])
        
        return {
            'building_id': building_id,
            'materials_breakdown': materials_lca,
            'total_gwp_materials_kgco2eq': round(total_gwp, 2),
            'total_embodied_energy_mj': round(total_embodied_energy, 2),
            'total_water_usage_l': round(total_water, 2),
            'gwp_per_m2_kgco2eq': round(total_gwp / floor_area, 2),
            'embodied_energy_per_m2_mj': round(total_embodied_energy / floor_area, 2),
            'construction_gwp_kgco2eq': round(construction_gwp, 2),
            'transport_gwp_kgco2eq': round(transport_gwp, 2),
            'transport_distance_km': round(transport_distance_km, 2),
            'hvac_manufacturing_gwp_kgco2eq': hvac_data['manufacturing_gwp_kgco2eq'],
            'hvac_operational_gwp_kgco2eq_year': hvac_data['operational_gwp_kgco2eq_year'],
            'hvac_lifespan_years': hvac_data['lifespan_years'],
            'total_embodied_carbon_kgco2eq': round(total_gwp + construction_gwp + transport_gwp + hvac_data['manufacturing_gwp_kgco2eq'], 2)
        }
    
    def calculate_retrofit_lca(self, building_attrs, retrofit_measures):
        """Calculate LCA for retrofit measures."""
        floor_area = building_attrs['total_floor_area_m2']
        wall_area = building_attrs['wall_area_m2']
        roof_area = building_attrs['rooftop_area_m2']
        window_area = building_attrs['window_area_m2']
        
        retrofit_lca_data = []
        
        for measure in retrofit_measures:
            if measure == 'wall_insulation':
                insulation_thickness = 0.10
                insulation_qty = wall_area * insulation_thickness * self.epd_database['eps_insulation']['density_kg_m3']
                lca = self.calculate_material_lca('eps_insulation', insulation_qty, wall_area)
                
                # Add installation impact
                installation_hours = wall_area / 10  # 10 m2 per hour
                installation_gwp = installation_hours * self.construction_processes['installation_labor']['gwp_kgco2eq_hour']
                
                if lca:
                    lca['installation_gwp_kgco2eq'] = round(installation_gwp, 2)
                    lca['total_gwp_with_installation'] = round(lca['gwp_total_kgco2eq'] + installation_gwp, 2)
                    lca['measure'] = measure
                    retrofit_lca_data.append(lca)
            
            elif measure == 'roof_insulation':
                insulation_thickness = 0.15
                insulation_qty = roof_area * insulation_thickness * self.epd_database['mineral_wool_insulation']['density_kg_m3']
                lca = self.calculate_material_lca('mineral_wool_insulation', insulation_qty, roof_area)
                
                installation_hours = roof_area / 12
                installation_gwp = installation_hours * self.construction_processes['installation_labor']['gwp_kgco2eq_hour']
                
                if lca:
                    lca['installation_gwp_kgco2eq'] = round(installation_gwp, 2)
                    lca['total_gwp_with_installation'] = round(lca['gwp_total_kgco2eq'] + installation_gwp, 2)
                    lca['measure'] = measure
                    retrofit_lca_data.append(lca)
            
            elif measure == 'window_replacement':
                window_thickness = 0.040
                window_qty = window_area * window_thickness * self.epd_database['glass_double_glazed']['density_kg_m3']
                lca = self.calculate_material_lca('glass_double_glazed', window_qty, window_area)
                
                # Add demolition of old windows
                demolition_gwp = window_area * self.construction_processes['demolition']['gwp_kgco2eq_m2']
                installation_hours = window_area / 5
                installation_gwp = installation_hours * self.construction_processes['installation_labor']['gwp_kgco2eq_hour']
                
                if lca:
                    lca['demolition_gwp_kgco2eq'] = round(demolition_gwp, 2)
                    lca['installation_gwp_kgco2eq'] = round(installation_gwp, 2)
                    lca['total_gwp_with_installation'] = round(lca['gwp_total_kgco2eq'] + installation_gwp + demolition_gwp, 2)
                    lca['measure'] = measure
                    retrofit_lca_data.append(lca)
            
            elif measure == 'hvac_upgrade_heat_pump':
                hvac_data = self.hvac_lca['air_source_heat_pump']
                retrofit_lca_data.append({
                    'measure': measure,
                    'material': 'air_source_heat_pump',
                    'gwp_total_kgco2eq': hvac_data['manufacturing_gwp_kgco2eq'],
                    'embodied_energy_total_mj': hvac_data['manufacturing_energy_mj'],
                    'operational_gwp_savings_kgco2eq_year': 1700,  # Savings vs old boiler
                    'lifespan_years': hvac_data['lifespan_years'],
                    'installation_gwp_kgco2eq': 150,
                    'total_gwp_with_installation': hvac_data['manufacturing_gwp_kgco2eq'] + 150
                })
            
            elif measure == 'solar_pv':
                pv_capacity = (roof_area * 0.1) * 0.15  # 10% of roof, 150W/m2
                pv_data = self.hvac_lca['solar_pv']
                retrofit_lca_data.append({
                    'measure': measure,
                    'material': 'solar_pv',
                    'capacity_kwp': round(pv_capacity, 2),
                    'gwp_total_kgco2eq': round(pv_capacity * pv_data['manufacturing_gwp_kgco2eq_kwp'], 2),
                    'embodied_energy_total_mj': round(pv_capacity * pv_data['manufacturing_energy_mj_kwp'], 2),
                    'operational_gwp_savings_kgco2eq_year': round(pv_capacity * abs(pv_data['operational_gwp_kgco2eq_year']), 2),
                    'lifespan_years': pv_data['lifespan_years'],
                    'installation_gwp_kgco2eq': round(pv_capacity * 100, 2),
                    'total_gwp_with_installation': round((pv_capacity * pv_data['manufacturing_gwp_kgco2eq_kwp']) + (pv_capacity * 100), 2)
                })
        
        return retrofit_lca_data
    
    def calculate_carbon_payback(self, retrofit_lca, annual_carbon_savings):
        """Calculate carbon payback period for retrofit measures."""
        results = []
        
        for measure_lca in retrofit_lca:
            embodied_carbon = measure_lca.get('total_gwp_with_installation', measure_lca.get('gwp_total_kgco2eq', 0))
            
            # Use measure-specific savings if available
            if 'operational_gwp_savings_kgco2eq_year' in measure_lca:
                savings_per_year = measure_lca['operational_gwp_savings_kgco2eq_year']
            else:
                savings_per_year = annual_carbon_savings * 0.25  # Assume each measure contributes 25%
            
            if savings_per_year > 0:
                payback_years = embodied_carbon / savings_per_year
            else:
                payback_years = 999
            
            lifespan = measure_lca.get('lifespan_years', 50)
            lifetime_carbon_benefit = (savings_per_year * lifespan) - embodied_carbon
            
            results.append({
                'measure': measure_lca.get('measure', measure_lca.get('material', 'unknown')),
                'embodied_carbon_kgco2eq': round(embodied_carbon, 2),
                'annual_carbon_savings_kgco2eq': round(savings_per_year, 2),
                'carbon_payback_years': round(payback_years, 2),
                'lifespan_years': lifespan,
                'lifetime_carbon_benefit_kgco2eq': round(lifetime_carbon_benefit, 2),
                'benefit_to_impact_ratio': round(lifetime_carbon_benefit / embodied_carbon, 2) if embodied_carbon > 0 else 0
            })
        
        return results
    
    def generate_dataset(self, building_attributes_df, retrofit_scenarios_df):
        """Generate complete LCA dataset."""
        print(f"Generating LCA data for {len(building_attributes_df)} buildings...")
        
        all_building_lca = []
        all_retrofit_lca = []
        all_carbon_payback = []
        
        for idx, building in building_attributes_df.iterrows():
            building_id = building['building_id']
            print(f"Processing {building_id}...")
            
            # Building LCA
            building_lca = self.calculate_building_lca(building)
            all_building_lca.append(building_lca)
            
            # Retrofit LCA for this building
            building_retrofits = retrofit_scenarios_df[retrofit_scenarios_df['building_id'] == building_id]
            
            for _, retrofit in building_retrofits.iterrows():
                measures = retrofit['measure_list'].split(', ')
                retrofit_lca = self.calculate_retrofit_lca(building, measures)
                
                # Calculate carbon payback
                annual_savings = retrofit['annual_carbon_saving_kgco2']
                carbon_payback = self.calculate_carbon_payback(retrofit_lca, annual_savings)
                
                for measure_lca in retrofit_lca:
                    measure_lca['building_id'] = building_id
                    measure_lca['scenario_name'] = retrofit['scenario_name']
                    all_retrofit_lca.append(measure_lca)
                
                for payback in carbon_payback:
                    payback['building_id'] = building_id
                    payback['scenario_name'] = retrofit['scenario_name']
                    all_carbon_payback.append(payback)
        
        building_lca_df = pd.DataFrame(all_building_lca)
        retrofit_lca_df = pd.DataFrame(all_retrofit_lca)
        carbon_payback_df = pd.DataFrame(all_carbon_payback)
        
        return building_lca_df, retrofit_lca_df, carbon_payback_df


def main():
    """Main function to generate and save LCA data."""
    print("=" * 80)
    print("LIFECYCLE ASSESSMENT (LCA) DATA GENERATOR FOR RETROFIT RESEARCH")
    print("=" * 80)
    
    # Load required data
    print("\nLoading building attributes and retrofit scenarios...")
    building_attrs = pd.read_csv('../data/building_attributes.csv')
    retrofit_scenarios = pd.read_csv('../data/retrofit_scenarios.csv')
    
    # Generate LCA data
    generator = LCADataGenerator()
    building_lca_df, retrofit_lca_df, carbon_payback_df = generator.generate_dataset(
        building_attrs, retrofit_scenarios
    )
    
    # Save datasets
    print("\nSaving datasets...")
    building_lca_df.to_csv('../data/lca_building_baseline.csv', index=False)
    retrofit_lca_df.to_csv('../data/lca_retrofit_measures.csv', index=False)
    carbon_payback_df.to_csv('../data/lca_carbon_payback.csv', index=False)
    
    # Save EPD database
    with open('../data/epd_database.json', 'w') as f:
        json.dump(generator.epd_database, f, indent=2)
    
    with open('../data/hvac_lca_database.json', 'w') as f:
        json.dump(generator.hvac_lca, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Building LCA records: {len(building_lca_df)}")
    print(f"Retrofit measure LCA records: {len(retrofit_lca_df)}")
    print(f"Carbon payback analyses: {len(carbon_payback_df)}")
    
    print("\n" + "-" * 80)
    print("BUILDING BASELINE LCA STATISTICS:")
    print("-" * 80)
    print(f"Average embodied carbon: {building_lca_df['total_embodied_carbon_kgco2eq'].mean():,.0f} kg CO2eq")
    print(f"Average embodied carbon per m²: {building_lca_df['gwp_per_m2_kgco2eq'].mean():.1f} kg CO2eq/m²")
    print(f"Average embodied energy: {building_lca_df['total_embodied_energy_mj'].mean():,.0f} MJ")
    
    print("\n" + "-" * 80)
    print("RETROFIT MEASURES LCA:")
    print("-" * 80)
    print(retrofit_lca_df.groupby('measure')['gwp_total_kgco2eq'].agg(['count', 'mean', 'sum']))
    
    print("\n" + "-" * 80)
    print("CARBON PAYBACK ANALYSIS:")
    print("-" * 80)
    print(f"Average carbon payback: {carbon_payback_df['carbon_payback_years'].mean():.1f} years")
    print(f"Average benefit-to-impact ratio: {carbon_payback_df['benefit_to_impact_ratio'].mean():.1f}")
    print("\nBy measure:")
    print(carbon_payback_df.groupby('measure')['carbon_payback_years'].mean().sort_values())
    
    print("\n✅ LCA data generation complete!")
    print(f"📁 Building LCA: ../data/lca_building_baseline.csv")
    print(f"📁 Retrofit LCA: ../data/lca_retrofit_measures.csv")
    print(f"📁 Carbon payback: ../data/lca_carbon_payback.csv")
    print(f"📁 EPD database: ../data/epd_database.json")
    print(f"📁 HVAC LCA database: ../data/hvac_lca_database.json")


if __name__ == "__main__":
    main()
