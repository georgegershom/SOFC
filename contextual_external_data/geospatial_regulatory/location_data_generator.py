#!/usr/bin/env python3
"""
Location Data Generator
Generates comprehensive geospatial data for building retrofit analysis
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os
import math

class LocationDataGenerator:
    def __init__(self):
        # Major US cities with detailed location data
        self.cities_data = {
            'New York City': {
                'latitude': 40.7128, 'longitude': -74.0060, 'elevation_m': 10,
                'climate_zone': '4A', 'state': 'NY', 'county': 'New York',
                'population': 8336817, 'urban_density': 'very_high',
                'building_density': 'very_high', 'average_building_height': 6.2
            },
            'Los Angeles': {
                'latitude': 34.0522, 'longitude': -118.2437, 'elevation_m': 71,
                'climate_zone': '3B', 'state': 'CA', 'county': 'Los Angeles',
                'population': 3898747, 'urban_density': 'high',
                'building_density': 'high', 'average_building_height': 2.8
            },
            'Chicago': {
                'latitude': 41.8781, 'longitude': -87.6298, 'elevation_m': 182,
                'climate_zone': '5A', 'state': 'IL', 'county': 'Cook',
                'population': 2746388, 'urban_density': 'high',
                'building_density': 'high', 'average_building_height': 3.1
            },
            'Houston': {
                'latitude': 29.7604, 'longitude': -95.3698, 'elevation_m': 13,
                'climate_zone': '2A', 'state': 'TX', 'county': 'Harris',
                'population': 2304580, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.2
            },
            'Phoenix': {
                'latitude': 33.4484, 'longitude': -112.0740, 'elevation_m': 331,
                'climate_zone': '2B', 'state': 'AZ', 'county': 'Maricopa',
                'population': 1608139, 'urban_density': 'medium',
                'building_density': 'low', 'average_building_height': 1.8
            },
            'Philadelphia': {
                'latitude': 39.9526, 'longitude': -75.1652, 'elevation_m': 12,
                'climate_zone': '4A', 'state': 'PA', 'county': 'Philadelphia',
                'population': 1603797, 'urban_density': 'high',
                'building_density': 'high', 'average_building_height': 2.9
            },
            'San Antonio': {
                'latitude': 29.4241, 'longitude': -98.4936, 'elevation_m': 198,
                'climate_zone': '2A', 'state': 'TX', 'county': 'Bexar',
                'population': 1434625, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 1.9
            },
            'San Diego': {
                'latitude': 32.7157, 'longitude': -117.1611, 'elevation_m': 19,
                'climate_zone': '3B', 'state': 'CA', 'county': 'San Diego',
                'population': 1386932, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.1
            },
            'Dallas': {
                'latitude': 32.7767, 'longitude': -96.7970, 'elevation_m': 131,
                'climate_zone': '3A', 'state': 'TX', 'county': 'Dallas',
                'population': 1304379, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.4
            },
            'San Jose': {
                'latitude': 37.3382, 'longitude': -121.8863, 'elevation_m': 25,
                'climate_zone': '3C', 'state': 'CA', 'county': 'Santa Clara',
                'population': 1013240, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.0
            },
            'Austin': {
                'latitude': 30.2672, 'longitude': -97.7431, 'elevation_m': 149,
                'climate_zone': '2A', 'state': 'TX', 'county': 'Travis',
                'population': 978908, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.3
            },
            'Jacksonville': {
                'latitude': 30.3322, 'longitude': -81.6557, 'elevation_m': 5,
                'climate_zone': '2A', 'state': 'FL', 'county': 'Duval',
                'population': 949611, 'urban_density': 'low',
                'building_density': 'low', 'average_building_height': 1.7
            },
            'Fort Worth': {
                'latitude': 32.7555, 'longitude': -97.3308, 'elevation_m': 203,
                'climate_zone': '3A', 'state': 'TX', 'county': 'Tarrant',
                'population': 918915, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.0
            },
            'Columbus': {
                'latitude': 39.9612, 'longitude': -82.9988, 'elevation_m': 254,
                'climate_zone': '5A', 'state': 'OH', 'county': 'Franklin',
                'population': 905748, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.1
            },
            'Charlotte': {
                'latitude': 35.2271, 'longitude': -80.8431, 'elevation_m': 229,
                'climate_zone': '3A', 'state': 'NC', 'county': 'Mecklenburg',
                'population': 885708, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.2
            },
            'San Francisco': {
                'latitude': 37.7749, 'longitude': -122.4194, 'elevation_m': 16,
                'climate_zone': '3C', 'state': 'CA', 'county': 'San Francisco',
                'population': 873965, 'urban_density': 'very_high',
                'building_density': 'very_high', 'average_building_height': 4.1
            },
            'Indianapolis': {
                'latitude': 39.7684, 'longitude': -86.1581, 'elevation_m': 223,
                'climate_zone': '5A', 'state': 'IN', 'county': 'Marion',
                'population': 876384, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.0
            },
            'Seattle': {
                'latitude': 47.6062, 'longitude': -122.3321, 'elevation_m': 56,
                'climate_zone': '4C', 'state': 'WA', 'county': 'King',
                'population': 749256, 'urban_density': 'high',
                'building_density': 'high', 'average_building_height': 3.2
            },
            'Denver': {
                'latitude': 39.7392, 'longitude': -104.9903, 'elevation_m': 1609,
                'climate_zone': '5B', 'state': 'CO', 'county': 'Denver',
                'population': 715522, 'urban_density': 'medium',
                'building_density': 'medium', 'average_building_height': 2.4
            },
            'Boston': {
                'latitude': 42.3601, 'longitude': -71.0589, 'elevation_m': 43,
                'climate_zone': '5A', 'state': 'MA', 'county': 'Suffolk',
                'population': 695506, 'urban_density': 'high',
                'building_density': 'high', 'average_building_height': 3.8
            }
        }
    
    def generate_detailed_location_data(self):
        """Generate comprehensive location data for all cities"""
        
        location_data = []
        
        for city_name, city_info in self.cities_data.items():
            # Calculate solar geometry parameters
            solar_data = self._calculate_solar_geometry(city_info['latitude'])
            
            # Calculate urban heat island effect
            uhi_effect = self._calculate_uhi_effect(city_info['urban_density'], city_info['population'])
            
            # Generate shading analysis data
            shading_data = self._generate_shading_analysis(city_info)
            
            # Calculate wind patterns
            wind_data = self._calculate_wind_patterns(city_info)
            
            location_entry = {
                'city_name': city_name,
                'latitude': city_info['latitude'],
                'longitude': city_info['longitude'],
                'elevation_m': city_info['elevation_m'],
                'elevation_ft': city_info['elevation_m'] * 3.28084,
                'climate_zone': city_info['climate_zone'],
                'state': city_info['state'],
                'county': city_info['county'],
                'population': city_info['population'],
                'urban_density': city_info['urban_density'],
                'building_density': city_info['building_density'],
                'average_building_height': city_info['average_building_height'],
                
                # Solar geometry
                'solar_declination_max': solar_data['declination_max'],
                'solar_declination_min': solar_data['declination_min'],
                'solar_noon_elevation_summer': solar_data['noon_elevation_summer'],
                'solar_noon_elevation_winter': solar_data['noon_elevation_winter'],
                'daylight_hours_summer': solar_data['daylight_hours_summer'],
                'daylight_hours_winter': solar_data['daylight_hours_winter'],
                
                # Urban heat island
                'uhi_intensity_max': uhi_effect['intensity_max'],
                'uhi_intensity_avg': uhi_effect['intensity_avg'],
                'uhi_seasonal_variation': uhi_effect['seasonal_variation'],
                
                # Shading analysis
                'avg_sky_view_factor': shading_data['sky_view_factor'],
                'building_shading_factor': shading_data['building_shading_factor'],
                'horizon_angle_avg': shading_data['horizon_angle_avg'],
                
                # Wind patterns
                'prevailing_wind_direction': wind_data['prevailing_direction'],
                'wind_speed_reduction_factor': wind_data['speed_reduction_factor'],
                'wind_turbulence_factor': wind_data['turbulence_factor'],
                
                # Geographic context
                'time_zone': self._get_time_zone(city_info['longitude']),
                'magnetic_declination': self._calculate_magnetic_declination(city_info['latitude'], city_info['longitude']),
                'nearest_airport_code': self._get_nearest_airport(city_name),
                'distance_to_coast_km': self._calculate_distance_to_coast(city_info),
                
                # Data generation metadata
                'data_generation_date': datetime.now().isoformat(),
                'data_source': 'Synthetic Location Data Generator'
            }
            
            location_data.append(location_entry)
        
        return pd.DataFrame(location_data)
    
    def _calculate_solar_geometry(self, latitude):
        """Calculate solar geometry parameters for a location"""
        
        lat_rad = math.radians(latitude)
        
        # Solar declination angles
        declination_max = 23.45  # Summer solstice
        declination_min = -23.45  # Winter solstice
        
        # Solar elevation at solar noon
        noon_elevation_summer = 90 - abs(latitude - declination_max)
        noon_elevation_winter = 90 - abs(latitude - declination_min)
        
        # Daylight hours calculation (simplified)
        summer_hour_angle = math.degrees(math.acos(-math.tan(lat_rad) * math.tan(math.radians(declination_max))))
        winter_hour_angle = math.degrees(math.acos(-math.tan(lat_rad) * math.tan(math.radians(declination_min))))
        
        daylight_hours_summer = 2 * summer_hour_angle / 15
        daylight_hours_winter = 2 * winter_hour_angle / 15
        
        return {
            'declination_max': declination_max,
            'declination_min': declination_min,
            'noon_elevation_summer': max(0, noon_elevation_summer),
            'noon_elevation_winter': max(0, noon_elevation_winter),
            'daylight_hours_summer': min(24, daylight_hours_summer),
            'daylight_hours_winter': max(0, daylight_hours_winter)
        }
    
    def _calculate_uhi_effect(self, urban_density, population):
        """Calculate urban heat island effect parameters"""
        
        # UHI intensity based on urban density and population
        density_factors = {
            'very_high': 4.5,
            'high': 3.2,
            'medium': 2.1,
            'low': 1.0
        }
        
        base_intensity = density_factors.get(urban_density, 2.0)
        
        # Population adjustment
        pop_factor = min(2.0, 1 + math.log10(population / 100000) * 0.3)
        
        intensity_max = base_intensity * pop_factor
        intensity_avg = intensity_max * 0.6
        seasonal_variation = intensity_max * 0.3
        
        return {
            'intensity_max': round(intensity_max, 1),
            'intensity_avg': round(intensity_avg, 1),
            'seasonal_variation': round(seasonal_variation, 1)
        }
    
    def _generate_shading_analysis(self, city_info):
        """Generate shading analysis data based on urban context"""
        
        density_factors = {
            'very_high': {'svf': 0.3, 'shading': 0.7, 'horizon': 45},
            'high': {'svf': 0.5, 'shading': 0.5, 'horizon': 30},
            'medium': {'svf': 0.7, 'shading': 0.3, 'horizon': 20},
            'low': {'svf': 0.9, 'shading': 0.1, 'horizon': 10}
        }
        
        factors = density_factors.get(city_info['building_density'], density_factors['medium'])
        
        # Adjust based on average building height
        height_adjustment = min(0.3, city_info['average_building_height'] / 10 * 0.2)
        
        return {
            'sky_view_factor': max(0.1, factors['svf'] - height_adjustment),
            'building_shading_factor': min(0.9, factors['shading'] + height_adjustment),
            'horizon_angle_avg': factors['horizon'] + city_info['average_building_height'] * 2
        }
    
    def _calculate_wind_patterns(self, city_info):
        """Calculate wind pattern modifications due to urban context"""
        
        # Wind speed reduction in urban areas
        density_reductions = {
            'very_high': 0.4,  # 60% reduction
            'high': 0.3,       # 70% of original
            'medium': 0.15,    # 85% of original
            'low': 0.05        # 95% of original
        }
        
        reduction = density_reductions.get(city_info['urban_density'], 0.15)
        speed_reduction_factor = 1 - reduction
        
        # Turbulence increase in urban areas
        turbulence_increases = {
            'very_high': 2.5,
            'high': 2.0,
            'medium': 1.5,
            'low': 1.1
        }
        
        turbulence_factor = turbulence_increases.get(city_info['urban_density'], 1.5)
        
        # Simplified prevailing wind direction (based on geographic location)
        if city_info['longitude'] < -100:  # Western US
            prevailing_direction = 270  # West
        elif city_info['latitude'] > 40:  # Northern US
            prevailing_direction = 225  # Southwest
        else:  # Southern/Eastern US
            prevailing_direction = 180  # South
        
        return {
            'prevailing_direction': prevailing_direction,
            'speed_reduction_factor': speed_reduction_factor,
            'turbulence_factor': turbulence_factor
        }
    
    def _get_time_zone(self, longitude):
        """Determine time zone based on longitude"""
        
        # Simplified time zone calculation
        tz_offset = round(longitude / 15)
        
        if tz_offset == -8:
            return "Pacific"
        elif tz_offset == -7:
            return "Mountain"
        elif tz_offset == -6:
            return "Central"
        elif tz_offset == -5:
            return "Eastern"
        else:
            return f"UTC{tz_offset:+d}"
    
    def _calculate_magnetic_declination(self, latitude, longitude):
        """Calculate approximate magnetic declination"""
        
        # Simplified magnetic declination calculation (2023)
        # This is a rough approximation
        base_declination = (longitude + 95) * 0.2
        lat_adjustment = (latitude - 40) * 0.1
        
        return round(base_declination + lat_adjustment, 1)
    
    def _get_nearest_airport(self, city_name):
        """Get nearest major airport code"""
        
        airport_codes = {
            'New York City': 'JFK',
            'Los Angeles': 'LAX',
            'Chicago': 'ORD',
            'Houston': 'IAH',
            'Phoenix': 'PHX',
            'Philadelphia': 'PHL',
            'San Antonio': 'SAT',
            'San Diego': 'SAN',
            'Dallas': 'DFW',
            'San Jose': 'SJC',
            'Austin': 'AUS',
            'Jacksonville': 'JAX',
            'Fort Worth': 'DFW',
            'Columbus': 'CMH',
            'Charlotte': 'CLT',
            'San Francisco': 'SFO',
            'Indianapolis': 'IND',
            'Seattle': 'SEA',
            'Denver': 'DEN',
            'Boston': 'BOS'
        }
        
        return airport_codes.get(city_name, 'N/A')
    
    def _calculate_distance_to_coast(self, city_info):
        """Calculate approximate distance to nearest coast"""
        
        # Simplified distance calculation
        lat, lon = city_info['latitude'], city_info['longitude']
        
        # Coastal cities
        if city_info['state'] in ['CA', 'FL', 'WA', 'NY'] and abs(lon) > 70:
            return 25  # Near coast
        elif city_info['state'] in ['TX', 'NC', 'MA']:
            return 150  # Moderate distance
        else:
            return 800  # Inland
    
    def generate_neighborhood_context_data(self):
        """Generate neighborhood-level context data for shading analysis"""
        
        neighborhood_data = []
        
        for city_name, city_info in self.cities_data.items():
            # Generate multiple neighborhood types per city
            neighborhood_types = [
                'downtown_core', 'urban_residential', 'suburban_residential',
                'industrial', 'mixed_use', 'commercial_strip'
            ]
            
            for neighborhood_type in neighborhood_types:
                context_data = self._generate_neighborhood_characteristics(
                    city_name, city_info, neighborhood_type
                )
                neighborhood_data.append(context_data)
        
        return pd.DataFrame(neighborhood_data)
    
    def _generate_neighborhood_characteristics(self, city_name, city_info, neighborhood_type):
        """Generate characteristics for a specific neighborhood type"""
        
        # Neighborhood type characteristics
        type_characteristics = {
            'downtown_core': {
                'building_height_avg': 15.0, 'building_height_std': 8.0,
                'building_coverage': 0.8, 'street_width_avg': 20,
                'tree_coverage': 0.1, 'setback_avg': 2
            },
            'urban_residential': {
                'building_height_avg': 3.5, 'building_height_std': 1.5,
                'building_coverage': 0.6, 'street_width_avg': 15,
                'tree_coverage': 0.3, 'setback_avg': 8
            },
            'suburban_residential': {
                'building_height_avg': 2.2, 'building_height_std': 0.8,
                'building_coverage': 0.3, 'street_width_avg': 12,
                'tree_coverage': 0.5, 'setback_avg': 15
            },
            'industrial': {
                'building_height_avg': 6.0, 'building_height_std': 3.0,
                'building_coverage': 0.5, 'street_width_avg': 25,
                'tree_coverage': 0.05, 'setback_avg': 20
            },
            'mixed_use': {
                'building_height_avg': 8.0, 'building_height_std': 4.0,
                'building_coverage': 0.7, 'street_width_avg': 18,
                'tree_coverage': 0.2, 'setback_avg': 5
            },
            'commercial_strip': {
                'building_height_avg': 4.5, 'building_height_std': 2.0,
                'building_coverage': 0.4, 'street_width_avg': 22,
                'tree_coverage': 0.15, 'setback_avg': 25
            }
        }
        
        chars = type_characteristics[neighborhood_type]
        
        return {
            'city_name': city_name,
            'neighborhood_type': neighborhood_type,
            'latitude': city_info['latitude'],
            'longitude': city_info['longitude'],
            'climate_zone': city_info['climate_zone'],
            
            # Building characteristics
            'building_height_avg_m': chars['building_height_avg'],
            'building_height_std_m': chars['building_height_std'],
            'building_coverage_ratio': chars['building_coverage'],
            'building_setback_avg_m': chars['setback_avg'],
            
            # Street characteristics
            'street_width_avg_m': chars['street_width_avg'],
            'street_orientation_primary': np.random.choice(['N-S', 'E-W', 'NE-SW', 'NW-SE']),
            
            # Vegetation
            'tree_coverage_ratio': chars['tree_coverage'],
            'tree_height_avg_m': chars['tree_coverage'] * 12 + 3,  # Taller trees in greener areas
            
            # Shading calculations
            'sky_view_factor': self._calculate_neighborhood_svf(chars),
            'solar_access_morning': self._calculate_solar_access(chars, 'morning'),
            'solar_access_midday': self._calculate_solar_access(chars, 'midday'),
            'solar_access_afternoon': self._calculate_solar_access(chars, 'afternoon'),
            
            # Urban heat island
            'surface_albedo': self._calculate_surface_albedo(neighborhood_type),
            'thermal_mass_factor': self._calculate_thermal_mass(chars),
            
            'data_generation_date': datetime.now().isoformat()
        }
    
    def _calculate_neighborhood_svf(self, characteristics):
        """Calculate sky view factor for neighborhood"""
        
        # Simplified SVF calculation based on building height and coverage
        height_factor = min(0.8, characteristics['building_height_avg'] / 20)
        coverage_factor = characteristics['building_coverage']
        
        svf = 1 - (height_factor * coverage_factor * 0.7)
        return max(0.1, svf)
    
    def _calculate_solar_access(self, characteristics, time_period):
        """Calculate solar access for different times of day"""
        
        base_access = 1 - characteristics['building_coverage'] * 0.6
        
        # Time-specific adjustments
        if time_period == 'morning':
            # Eastern shading more important
            access = base_access * 0.9
        elif time_period == 'afternoon':
            # Western shading more important
            access = base_access * 0.9
        else:  # midday
            # Overhead shading most important
            height_reduction = min(0.4, characteristics['building_height_avg'] / 25)
            access = base_access * (1 - height_reduction)
        
        return max(0.1, access)
    
    def _calculate_surface_albedo(self, neighborhood_type):
        """Calculate average surface albedo for neighborhood type"""
        
        albedo_values = {
            'downtown_core': 0.15,      # Dark surfaces, concrete
            'urban_residential': 0.25,   # Mixed surfaces
            'suburban_residential': 0.35, # More vegetation, lighter surfaces
            'industrial': 0.20,         # Mixed industrial surfaces
            'mixed_use': 0.22,          # Urban mix
            'commercial_strip': 0.18    # Parking lots, commercial
        }
        
        return albedo_values.get(neighborhood_type, 0.25)
    
    def _calculate_thermal_mass(self, characteristics):
        """Calculate thermal mass factor for neighborhood"""
        
        # Higher building coverage and height = more thermal mass
        mass_factor = (
            characteristics['building_coverage'] * 0.5 +
            min(1.0, characteristics['building_height_avg'] / 20) * 0.5
        )
        
        return mass_factor

def main():
    """Generate comprehensive location data"""
    
    print("Generating comprehensive location data...")
    
    generator = LocationDataGenerator()
    
    # Generate location datasets
    location_data = generator.generate_detailed_location_data()
    neighborhood_data = generator.generate_neighborhood_context_data()
    
    # Create output directory
    output_dir = "geospatial_regulatory/location_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    location_data.to_csv(f"{output_dir}/city_location_data.csv", index=False)
    neighborhood_data.to_csv(f"{output_dir}/neighborhood_context_data.csv", index=False)
    
    # Generate location summary
    location_summary = {
        'generation_date': datetime.now().isoformat(),
        'total_cities': len(location_data),
        'total_neighborhoods': len(neighborhood_data),
        'climate_zones_covered': sorted(location_data['climate_zone'].unique().tolist()),
        'states_covered': sorted(location_data['state'].unique().tolist()),
        'data_categories': [
            'solar_geometry', 'urban_heat_island', 'shading_analysis',
            'wind_patterns', 'geographic_context', 'neighborhood_characteristics'
        ],
        'coordinate_bounds': {
            'latitude_min': float(location_data['latitude'].min()),
            'latitude_max': float(location_data['latitude'].max()),
            'longitude_min': float(location_data['longitude'].min()),
            'longitude_max': float(location_data['longitude'].max())
        }
    }
    
    with open(f"{output_dir}/location_data_summary.json", 'w') as f:
        json.dump(location_summary, f, indent=2)
    
    print("Location data generation completed!")

if __name__ == "__main__":
    main()