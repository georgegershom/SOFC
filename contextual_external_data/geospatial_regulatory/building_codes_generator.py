#!/usr/bin/env python3
"""
Building Codes & Standards Data Generator
Generates comprehensive building code requirements and emissions targets
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

class BuildingCodesGenerator:
    def __init__(self):
        # IECC Climate Zones and their characteristics
        self.climate_zones = {
            '1A': {'name': 'Very Hot-Humid', 'hdd65': 0, 'cdd50': 9000, 'states': ['FL', 'HI']},
            '2A': {'name': 'Hot-Humid', 'hdd65': 1500, 'cdd50': 6000, 'states': ['TX', 'FL', 'LA', 'MS', 'AL', 'GA', 'SC']},
            '2B': {'name': 'Hot-Dry', 'hdd65': 1500, 'cdd50': 6000, 'states': ['AZ', 'CA', 'NV', 'TX']},
            '3A': {'name': 'Warm-Humid', 'hdd65': 3000, 'cdd50': 4000, 'states': ['TX', 'OK', 'AR', 'LA', 'MS', 'AL', 'GA', 'SC', 'NC', 'TN', 'KY']},
            '3B': {'name': 'Warm-Dry', 'hdd65': 3000, 'cdd50': 4000, 'states': ['CA', 'NV', 'AZ', 'NM', 'TX']},
            '3C': {'name': 'Warm-Marine', 'hdd65': 3000, 'cdd50': 1000, 'states': ['CA', 'OR', 'WA']},
            '4A': {'name': 'Mixed-Humid', 'hdd65': 5000, 'cdd50': 2500, 'states': ['NY', 'PA', 'NJ', 'CT', 'MA', 'RI', 'VT', 'NH', 'ME', 'MD', 'DE', 'DC', 'VA', 'WV', 'KY', 'TN', 'NC', 'OH', 'IN', 'IL', 'MO', 'KS']},
            '4B': {'name': 'Mixed-Dry', 'hdd65': 5000, 'cdd50': 2500, 'states': ['NV', 'UT', 'CO', 'KS', 'OK', 'NM']},
            '4C': {'name': 'Mixed-Marine', 'hdd65': 5000, 'cdd50': 1000, 'states': ['WA', 'OR']},
            '5A': {'name': 'Cool-Humid', 'hdd65': 7000, 'cdd50': 1500, 'states': ['NY', 'VT', 'NH', 'ME', 'MA', 'CT', 'RI', 'PA', 'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'NE', 'SD', 'ND']},
            '5B': {'name': 'Cool-Dry', 'hdd65': 7000, 'cdd50': 1500, 'states': ['CO', 'WY', 'MT', 'ID', 'UT', 'NV']},
            '5C': {'name': 'Cool-Marine', 'hdd65': 7000, 'cdd50': 500, 'states': ['WA', 'OR']},
            '6A': {'name': 'Cold-Humid', 'hdd65': 9000, 'cdd50': 1000, 'states': ['NY', 'VT', 'NH', 'ME', 'MN', 'WI', 'MI', 'ND', 'SD', 'MT']},
            '6B': {'name': 'Cold-Dry', 'hdd65': 9000, 'cdd50': 1000, 'states': ['MT', 'WY', 'CO', 'ID', 'UT']},
            '7': {'name': 'Very Cold', 'hdd65': 12000, 'cdd50': 500, 'states': ['MN', 'WI', 'MI', 'NY', 'VT', 'NH', 'ME', 'MT', 'ND', 'SD', 'AK']},
            '8': {'name': 'Subarctic', 'hdd65': 15000, 'cdd50': 0, 'states': ['AK']}
        }
        
        # IECC 2021 Requirements by Climate Zone
        self.iecc_2021_requirements = {
            'residential': {
                '1A': {'wall_r': 13, 'ceiling_r': 30, 'floor_r': 13, 'window_u': 1.20, 'window_shgc': 0.25, 'air_leakage': 3.0},
                '2A': {'wall_r': 13, 'ceiling_r': 38, 'floor_r': 13, 'window_u': 0.65, 'window_shgc': 0.25, 'air_leakage': 3.0},
                '2B': {'wall_r': 13, 'ceiling_r': 38, 'floor_r': 13, 'window_u': 0.65, 'window_shgc': 0.25, 'air_leakage': 3.0},
                '3A': {'wall_r': 20, 'ceiling_r': 38, 'floor_r': 25, 'window_u': 0.50, 'window_shgc': 0.25, 'air_leakage': 3.0},
                '3B': {'wall_r': 20, 'ceiling_r': 38, 'floor_r': 25, 'window_u': 0.50, 'window_shgc': 0.25, 'air_leakage': 3.0},
                '3C': {'wall_r': 20, 'ceiling_r': 38, 'floor_r': 25, 'window_u': 0.50, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '4A': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.35, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '4B': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.35, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '4C': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.35, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '5A': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.32, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '5B': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.32, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '5C': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.32, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '6A': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.30, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '6B': {'wall_r': 20, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.30, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '7': {'wall_r': 21, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.30, 'window_shgc': 0.40, 'air_leakage': 3.0},
                '8': {'wall_r': 21, 'ceiling_r': 49, 'floor_r': 30, 'window_u': 0.30, 'window_shgc': 0.40, 'air_leakage': 3.0}
            },
            'commercial': {
                '1A': {'wall_u': 0.124, 'roof_u': 0.063, 'window_u': 1.20, 'window_shgc': 0.25, 'lighting_lpd': 0.6},
                '2A': {'wall_u': 0.124, 'roof_u': 0.063, 'window_u': 0.65, 'window_shgc': 0.25, 'lighting_lpd': 0.6},
                '2B': {'wall_u': 0.124, 'roof_u': 0.063, 'window_u': 0.65, 'window_shgc': 0.25, 'lighting_lpd': 0.6},
                '3A': {'wall_u': 0.090, 'roof_u': 0.063, 'window_u': 0.50, 'window_shgc': 0.25, 'lighting_lpd': 0.6},
                '3B': {'wall_u': 0.090, 'roof_u': 0.063, 'window_u': 0.50, 'window_shgc': 0.25, 'lighting_lpd': 0.6},
                '3C': {'wall_u': 0.090, 'roof_u': 0.063, 'window_u': 0.50, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '4A': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.40, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '4B': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.40, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '4C': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.40, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '5A': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.38, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '5B': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.38, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '5C': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.38, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '6A': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.36, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '6B': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.36, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '7': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.36, 'window_shgc': 0.40, 'lighting_lpd': 0.6},
                '8': {'wall_u': 0.090, 'roof_u': 0.048, 'window_u': 0.36, 'window_shgc': 0.40, 'lighting_lpd': 0.6}
            }
        }
        
        # State-specific building codes and amendments
        self.state_codes = {
            'CA': {
                'base_code': 'Title 24 (California Energy Code)',
                'version': '2022',
                'stringency': 'very_high',
                'solar_requirement': True,
                'battery_ready': True,
                'electric_ready': True,
                'zero_net_energy_target': 2030
            },
            'NY': {
                'base_code': 'IECC 2021 with amendments',
                'version': '2021',
                'stringency': 'high',
                'solar_requirement': False,
                'battery_ready': True,
                'electric_ready': True,
                'zero_net_energy_target': 2035
            },
            'MA': {
                'base_code': 'IECC 2021 with amendments',
                'version': '2021',
                'stringency': 'high',
                'solar_requirement': False,
                'battery_ready': True,
                'electric_ready': True,
                'zero_net_energy_target': 2035
            },
            'WA': {
                'base_code': 'IECC 2021 with amendments',
                'version': '2021',
                'stringency': 'high',
                'solar_requirement': False,
                'battery_ready': False,
                'electric_ready': True,
                'zero_net_energy_target': 2040
            },
            'TX': {
                'base_code': 'IECC 2021',
                'version': '2021',
                'stringency': 'medium',
                'solar_requirement': False,
                'battery_ready': False,
                'electric_ready': False,
                'zero_net_energy_target': None
            },
            'FL': {
                'base_code': 'Florida Building Code',
                'version': '2020',
                'stringency': 'medium',
                'solar_requirement': False,
                'battery_ready': False,
                'electric_ready': False,
                'zero_net_energy_target': None
            }
        }
        
        # Local emissions targets and building performance standards
        self.local_emissions_targets = {
            'New York City': {
                'law': 'Local Law 97',
                'implementation_year': 2024,
                'building_size_threshold': 25000,  # sq ft
                'emissions_limits': {
                    2024: {'office': 0.00885, 'multifamily': 0.00453, 'hotel': 0.00398},  # tCO2e/sq ft
                    2030: {'office': 0.00531, 'multifamily': 0.00272, 'hotel': 0.00239}
                },
                'penalties': {'per_ton_co2e': 268},
                'compliance_pathways': ['direct_emissions', 'renewable_energy_credits', 'carbon_offsets']
            },
            'Boston': {
                'law': 'Building Emissions Reduction and Disclosure Ordinance (BERDO)',
                'implementation_year': 2025,
                'building_size_threshold': 20000,
                'emissions_limits': {
                    2025: {'office': 0.0089, 'multifamily': 0.0048, 'retail': 0.0057},
                    2030: {'office': 0.0050, 'multifamily': 0.0027, 'retail': 0.0032}
                },
                'penalties': {'per_ton_co2e': 234},
                'compliance_pathways': ['direct_emissions', 'alternative_compliance_payments']
            },
            'Washington DC': {
                'law': 'Building Energy Performance Standards (BEPS)',
                'implementation_year': 2026,
                'building_size_threshold': 50000,
                'energy_limits': {
                    2026: {'office': 38.4, 'multifamily': 35.8, 'hotel': 48.3},  # kBtu/sq ft
                    2033: {'office': 30.7, 'multifamily': 28.6, 'hotel': 38.6}
                },
                'penalties': {'per_sq_ft': 7.50},
                'compliance_pathways': ['energy_performance', 'alternative_compliance']
            },
            'Seattle': {
                'law': 'Building Tune-Ups',
                'implementation_year': 2021,
                'building_size_threshold': 50000,
                'requirements': 'energy_audit_and_retro_commissioning',
                'frequency': 'every_5_years',
                'penalties': {'daily_fine': 150}
            }
        }
    
    def generate_iecc_requirements_database(self):
        """Generate comprehensive IECC requirements database"""
        
        requirements_data = []
        
        for climate_zone, zone_info in self.climate_zones.items():
            for building_type in ['residential', 'commercial']:
                if building_type in self.iecc_2021_requirements:
                    reqs = self.iecc_2021_requirements[building_type].get(climate_zone, {})
                    
                    requirements_data.append({
                        'climate_zone': climate_zone,
                        'climate_zone_name': zone_info['name'],
                        'building_type': building_type,
                        'code_version': 'IECC 2021',
                        'hdd65': zone_info['hdd65'],
                        'cdd50': zone_info['cdd50'],
                        'applicable_states': ', '.join(zone_info['states']),
                        
                        # Envelope requirements
                        'wall_r_value': reqs.get('wall_r'),
                        'wall_u_factor': reqs.get('wall_u'),
                        'ceiling_r_value': reqs.get('ceiling_r'),
                        'roof_u_factor': reqs.get('roof_u'),
                        'floor_r_value': reqs.get('floor_r'),
                        'window_u_factor': reqs.get('window_u'),
                        'window_shgc': reqs.get('window_shgc'),
                        
                        # Air tightness
                        'air_leakage_ach50': reqs.get('air_leakage'),
                        
                        # Lighting (commercial only)
                        'lighting_power_density': reqs.get('lighting_lpd'),
                        
                        # Calculated performance targets
                        'estimated_energy_savings_vs_2018': self._calculate_energy_savings(climate_zone, building_type),
                        'compliance_cost_premium': self._estimate_compliance_cost(climate_zone, building_type),
                        
                        'data_source': 'IECC 2021 Standard'
                    })
        
        return pd.DataFrame(requirements_data)
    
    def _calculate_energy_savings(self, climate_zone, building_type):
        """Estimate energy savings vs previous code version"""
        
        # Simplified energy savings estimation
        base_savings = 0.10  # 10% base improvement
        
        # Climate zone adjustments
        zone_multipliers = {
            '1A': 0.8, '2A': 0.9, '2B': 0.9, '3A': 1.0, '3B': 1.0, '3C': 1.0,
            '4A': 1.1, '4B': 1.1, '4C': 1.1, '5A': 1.2, '5B': 1.2, '5C': 1.2,
            '6A': 1.3, '6B': 1.3, '7': 1.4, '8': 1.5
        }
        
        # Building type adjustments
        type_multipliers = {'residential': 1.0, 'commercial': 1.2}
        
        savings = base_savings * zone_multipliers.get(climate_zone, 1.0) * type_multipliers.get(building_type, 1.0)
        
        return round(savings, 3)
    
    def _estimate_compliance_cost(self, climate_zone, building_type):
        """Estimate cost premium for code compliance"""
        
        # Base cost premium (% of construction cost)
        base_premium = 0.02  # 2% base premium
        
        # Climate zone adjustments (colder = higher premium)
        zone_multipliers = {
            '1A': 0.5, '2A': 0.7, '2B': 0.7, '3A': 0.8, '3B': 0.8, '3C': 0.8,
            '4A': 1.0, '4B': 1.0, '4C': 1.0, '5A': 1.2, '5B': 1.2, '5C': 1.2,
            '6A': 1.4, '6B': 1.4, '7': 1.6, '8': 1.8
        }
        
        # Building type adjustments
        type_multipliers = {'residential': 1.0, 'commercial': 0.8}
        
        premium = base_premium * zone_multipliers.get(climate_zone, 1.0) * type_multipliers.get(building_type, 1.0)
        
        return round(premium, 4)
    
    def generate_state_code_database(self):
        """Generate state-specific building code database"""
        
        state_data = []
        
        for state, code_info in self.state_codes.items():
            # Get climate zones for this state
            state_zones = []
            for zone, zone_info in self.climate_zones.items():
                if state in zone_info['states']:
                    state_zones.append(zone)
            
            state_data.append({
                'state': state,
                'base_code': code_info['base_code'],
                'code_version': code_info['version'],
                'stringency_level': code_info['stringency'],
                'climate_zones': ', '.join(state_zones),
                
                # Special requirements
                'solar_requirement': code_info['solar_requirement'],
                'battery_ready_requirement': code_info['battery_ready'],
                'electric_ready_requirement': code_info['electric_ready'],
                'zero_net_energy_target_year': code_info['zero_net_energy_target'],
                
                # Calculated metrics
                'estimated_above_iecc_baseline': self._calculate_above_baseline(code_info['stringency']),
                'renewable_energy_requirements': self._get_renewable_requirements(state),
                'electrification_requirements': self._get_electrification_requirements(state),
                
                'last_updated': datetime.now().isoformat(),
                'data_source': 'State Building Code Analysis'
            })
        
        return pd.DataFrame(state_data)
    
    def _calculate_above_baseline(self, stringency):
        """Calculate how much more stringent than IECC baseline"""
        
        stringency_multipliers = {
            'very_high': 0.25,  # 25% more stringent
            'high': 0.15,       # 15% more stringent
            'medium': 0.05,     # 5% more stringent
            'low': 0.0          # Same as IECC
        }
        
        return stringency_multipliers.get(stringency, 0.0)
    
    def _get_renewable_requirements(self, state):
        """Get renewable energy requirements for state"""
        
        renewable_reqs = {
            'CA': 'Solar PV required on new residential (2020+), commercial (2023+)',
            'NY': 'No statewide requirement, but incentives available',
            'MA': 'Solar-ready requirements, stretch code options',
            'WA': 'No statewide requirement',
            'TX': 'No statewide requirement',
            'FL': 'Solar-ready requirements in some jurisdictions'
        }
        
        return renewable_reqs.get(state, 'No specific requirements')
    
    def _get_electrification_requirements(self, state):
        """Get electrification requirements for state"""
        
        electrification_reqs = {
            'CA': 'Electric-ready requirements, gas appliance restrictions in some cities',
            'NY': 'Electric-ready requirements, heat pump incentives',
            'MA': 'Electric-ready requirements, heat pump stretch code',
            'WA': 'Heat pump requirements in some jurisdictions',
            'TX': 'No specific requirements',
            'FL': 'No specific requirements'
        }
        
        return electrification_reqs.get(state, 'No specific requirements')
    
    def generate_local_emissions_standards(self):
        """Generate local emissions standards database"""
        
        emissions_data = []
        
        for city, standard_info in self.local_emissions_targets.items():
            # Process emissions limits by year and building type
            if 'emissions_limits' in standard_info:
                for year, limits in standard_info['emissions_limits'].items():
                    for building_type, limit in limits.items():
                        emissions_data.append({
                            'city': city,
                            'law_name': standard_info['law'],
                            'implementation_year': standard_info['implementation_year'],
                            'building_size_threshold_sqft': standard_info['building_size_threshold'],
                            'compliance_year': year,
                            'building_type': building_type,
                            'emissions_limit_tco2e_per_sqft': limit,
                            'penalty_per_ton_co2e': standard_info.get('penalties', {}).get('per_ton_co2e'),
                            'compliance_pathways': ', '.join(standard_info.get('compliance_pathways', [])),
                            'standard_type': 'emissions_based',
                            'data_source': f'{city} Building Performance Standard'
                        })
            
            # Process energy limits
            elif 'energy_limits' in standard_info:
                for year, limits in standard_info['energy_limits'].items():
                    for building_type, limit in limits.items():
                        emissions_data.append({
                            'city': city,
                            'law_name': standard_info['law'],
                            'implementation_year': standard_info['implementation_year'],
                            'building_size_threshold_sqft': standard_info['building_size_threshold'],
                            'compliance_year': year,
                            'building_type': building_type,
                            'energy_limit_kbtu_per_sqft': limit,
                            'penalty_per_sq_ft': standard_info.get('penalties', {}).get('per_sq_ft'),
                            'compliance_pathways': ', '.join(standard_info.get('compliance_pathways', [])),
                            'standard_type': 'energy_based',
                            'data_source': f'{city} Building Performance Standard'
                        })
            
            # Process other requirements
            else:
                emissions_data.append({
                    'city': city,
                    'law_name': standard_info['law'],
                    'implementation_year': standard_info['implementation_year'],
                    'building_size_threshold_sqft': standard_info['building_size_threshold'],
                    'requirements': standard_info.get('requirements'),
                    'frequency': standard_info.get('frequency'),
                    'daily_penalty': standard_info.get('penalties', {}).get('daily_fine'),
                    'standard_type': 'procedural',
                    'data_source': f'{city} Building Performance Standard'
                })
        
        return pd.DataFrame(emissions_data)
    
    def generate_future_code_projections(self, start_year=2024, end_year=2040):
        """Generate projections for future building code stringency"""
        
        projection_data = []
        
        # Define code evolution scenarios
        scenarios = {
            'conservative': {'annual_stringency_increase': 0.02},  # 2% per year
            'moderate': {'annual_stringency_increase': 0.04},     # 4% per year
            'aggressive': {'annual_stringency_increase': 0.06}    # 6% per year
        }
        
        for scenario_name, scenario_params in scenarios.items():
            for year in range(start_year, end_year + 1):
                years_from_base = year - 2023
                
                # Calculate stringency increase
                stringency_multiplier = (1 + scenario_params['annual_stringency_increase']) ** years_from_base
                
                for climate_zone in self.climate_zones.keys():
                    # Project envelope requirements
                    base_reqs = self.iecc_2021_requirements['residential'].get(climate_zone, {})
                    
                    if base_reqs:
                        projected_wall_r = base_reqs.get('wall_r', 20) * stringency_multiplier
                        projected_ceiling_r = base_reqs.get('ceiling_r', 49) * stringency_multiplier
                        projected_window_u = base_reqs.get('window_u', 0.35) / stringency_multiplier
                        
                        projection_data.append({
                            'year': year,
                            'scenario': scenario_name,
                            'climate_zone': climate_zone,
                            'projected_wall_r_value': min(60, projected_wall_r),  # Cap at R-60
                            'projected_ceiling_r_value': min(80, projected_ceiling_r),  # Cap at R-80
                            'projected_window_u_factor': max(0.15, projected_window_u),  # Floor at 0.15
                            'projected_air_leakage_ach50': max(1.0, 3.0 / stringency_multiplier),  # Floor at 1.0 ACH50
                            'stringency_multiplier': stringency_multiplier,
                            'estimated_energy_savings_vs_2023': min(0.5, (stringency_multiplier - 1) * 0.8),
                            'estimated_cost_premium_vs_2023': (stringency_multiplier - 1) * 0.15
                        })
        
        return pd.DataFrame(projection_data)
    
    def generate_compliance_cost_analysis(self):
        """Generate building code compliance cost analysis"""
        
        cost_data = []
        
        # Cost categories for code compliance
        cost_categories = {
            'insulation_upgrade': {'base_cost_per_sqft': 2.50, 'r_value_factor': 0.15},
            'window_upgrade': {'base_cost_per_sqft': 25.0, 'u_factor_improvement': 0.8},
            'air_sealing': {'base_cost_per_sqft': 1.25, 'ach50_factor': 0.75},
            'hvac_efficiency': {'base_cost_per_sqft': 3.50, 'efficiency_factor': 1.2},
            'lighting_upgrade': {'base_cost_per_sqft': 1.80, 'lpd_factor': 0.9},
            'commissioning': {'base_cost_per_sqft': 0.85, 'fixed_cost': True}
        }
        
        for climate_zone in self.climate_zones.keys():
            for building_type in ['residential', 'commercial']:
                base_reqs = self.iecc_2021_requirements[building_type].get(climate_zone, {})
                
                if base_reqs:
                    total_cost = 0
                    cost_breakdown = {}
                    
                    # Calculate costs for each category
                    for category, cost_info in cost_categories.items():
                        if category == 'insulation_upgrade' and 'wall_r' in base_reqs:
                            cost = cost_info['base_cost_per_sqft'] * (1 + base_reqs['wall_r'] * cost_info['r_value_factor'] / 20)
                        elif category == 'window_upgrade' and 'window_u' in base_reqs:
                            cost = cost_info['base_cost_per_sqft'] * (1 - base_reqs['window_u'] * cost_info['u_factor_improvement'])
                        elif category == 'air_sealing' and 'air_leakage' in base_reqs:
                            cost = cost_info['base_cost_per_sqft'] * (4 - base_reqs['air_leakage']) * cost_info['ach50_factor']
                        elif building_type == 'commercial':
                            cost = cost_info['base_cost_per_sqft']
                        else:
                            cost = cost_info['base_cost_per_sqft'] * 0.5  # Residential adjustment
                        
                        cost_breakdown[category] = max(0, cost)
                        total_cost += cost_breakdown[category]
                    
                    cost_data.append({
                        'climate_zone': climate_zone,
                        'building_type': building_type,
                        'total_compliance_cost_per_sqft': total_cost,
                        'insulation_cost_per_sqft': cost_breakdown['insulation_upgrade'],
                        'window_cost_per_sqft': cost_breakdown['window_upgrade'],
                        'air_sealing_cost_per_sqft': cost_breakdown['air_sealing'],
                        'hvac_cost_per_sqft': cost_breakdown['hvac_efficiency'],
                        'lighting_cost_per_sqft': cost_breakdown.get('lighting_upgrade', 0),
                        'commissioning_cost_per_sqft': cost_breakdown['commissioning'],
                        'payback_period_years': self._calculate_payback_period(total_cost, climate_zone, building_type),
                        'data_source': 'Building Code Compliance Cost Analysis'
                    })
        
        return pd.DataFrame(cost_data)
    
    def _calculate_payback_period(self, total_cost, climate_zone, building_type):
        """Calculate simple payback period for code compliance"""
        
        # Estimate annual energy savings ($/sq ft/year)
        base_savings = 0.50  # Base savings
        
        # Climate zone adjustments
        zone_multipliers = {
            '1A': 0.6, '2A': 0.8, '2B': 0.8, '3A': 1.0, '3B': 1.0, '3C': 0.9,
            '4A': 1.2, '4B': 1.2, '4C': 1.1, '5A': 1.4, '5B': 1.4, '5C': 1.3,
            '6A': 1.6, '6B': 1.6, '7': 1.8, '8': 2.0
        }
        
        annual_savings = base_savings * zone_multipliers.get(climate_zone, 1.0)
        
        if building_type == 'commercial':
            annual_savings *= 1.5  # Commercial buildings typically have higher savings
        
        payback_period = total_cost / annual_savings if annual_savings > 0 else 999
        
        return min(50, round(payback_period, 1))  # Cap at 50 years

def main():
    """Generate comprehensive building codes and standards data"""
    
    print("Generating building codes and standards data...")
    
    generator = BuildingCodesGenerator()
    
    # Generate all building code datasets
    iecc_requirements = generator.generate_iecc_requirements_database()
    state_codes = generator.generate_state_code_database()
    local_emissions = generator.generate_local_emissions_standards()
    future_projections = generator.generate_future_code_projections()
    compliance_costs = generator.generate_compliance_cost_analysis()
    
    # Create output directory
    output_dir = "geospatial_regulatory/building_codes"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    iecc_requirements.to_csv(f"{output_dir}/iecc_2021_requirements.csv", index=False)
    state_codes.to_csv(f"{output_dir}/state_building_codes.csv", index=False)
    local_emissions.to_csv(f"{output_dir}/local_emissions_standards.csv", index=False)
    future_projections.to_csv(f"{output_dir}/future_code_projections_2024_2040.csv", index=False)
    compliance_costs.to_csv(f"{output_dir}/compliance_cost_analysis.csv", index=False)
    
    # Generate comprehensive summary
    building_codes_summary = {
        'generation_date': datetime.now().isoformat(),
        'iecc_version': '2021',
        'climate_zones_covered': len(generator.climate_zones),
        'states_analyzed': len(generator.state_codes),
        'local_jurisdictions': len(generator.local_emissions_targets),
        'projection_years': '2024-2040',
        'data_categories': [
            'iecc_envelope_requirements', 'state_code_amendments',
            'local_emissions_standards', 'future_stringency_projections',
            'compliance_cost_analysis'
        ],
        'key_insights': {
            'most_stringent_climate_zone': '8 (Subarctic)',
            'states_with_solar_requirements': ['CA'],
            'cities_with_emissions_limits': ['New York City', 'Boston', 'Washington DC'],
            'average_compliance_cost_premium': '2-6% of construction cost'
        },
        'compliance_pathways': {
            'prescriptive': 'Meet specific component requirements',
            'performance': 'Meet overall energy performance targets',
            'outcome_based': 'Meet actual measured performance'
        }
    }
    
    with open(f"{output_dir}/building_codes_summary.json", 'w') as f:
        json.dump(building_codes_summary, f, indent=2)
    
    print("Building codes and standards data generation completed!")

if __name__ == "__main__":
    main()