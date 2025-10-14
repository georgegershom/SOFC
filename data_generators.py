"""
Data Generators for Building Retrofit Dataset
Generates synthetic data for IoT sensors, building attributes, energy performance, and LCA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import json
from dataset_schema import schema

class IoTDataGenerator:
    """Generates synthetic IoT sensor data"""
    
    def __init__(self, num_buildings: int = 100, start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
        self.num_buildings = num_buildings
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.building_ids = [f"B_{i:04d}" for i in range(1, num_buildings + 1)]
    
    def generate_energy_consumption_data(self) -> pd.DataFrame:
        """Generate energy consumption data with realistic patterns"""
        data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            for building_id in self.building_ids:
                # Base consumption varies by building type and season
                month = current_date.month
                is_weekend = current_date.weekday() >= 5
                
                # Seasonal patterns
                if month in [12, 1, 2]:  # Winter
                    base_consumption = np.random.normal(150, 30)
                    heating_ratio = 0.4
                elif month in [6, 7, 8]:  # Summer
                    base_consumption = np.random.normal(180, 35)
                    heating_ratio = 0.1
                else:  # Spring/Fall
                    base_consumption = np.random.normal(120, 25)
                    heating_ratio = 0.2
                
                # Weekend vs weekday patterns
                if is_weekend:
                    base_consumption *= 0.8
                
                # Add some noise and trends
                noise = np.random.normal(0, 10)
                total_consumption = max(0, base_consumption + noise)
                
                # Distribute across end uses
                heating = total_consumption * heating_ratio * np.random.uniform(0.8, 1.2)
                cooling = total_consumption * (1 - heating_ratio) * 0.3 * np.random.uniform(0.8, 1.2)
                lighting = total_consumption * 0.15 * np.random.uniform(0.8, 1.2)
                appliances = total_consumption * 0.25 * np.random.uniform(0.8, 1.2)
                hvac = total_consumption * 0.2 * np.random.uniform(0.8, 1.2)
                other = total_consumption - (heating + cooling + lighting + appliances + hvac)
                
                data.append({
                    'building_id': building_id,
                    'timestamp': current_date,
                    'total_consumption_kwh': round(total_consumption, 2),
                    'heating_kwh': round(heating, 2),
                    'cooling_kwh': round(cooling, 2),
                    'lighting_kwh': round(lighting, 2),
                    'appliances_kwh': round(appliances, 2),
                    'hvac_kwh': round(hvac, 2),
                    'other_kwh': round(max(0, other), 2)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def generate_environmental_data(self) -> pd.DataFrame:
        """Generate indoor environmental parameters"""
        data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            for building_id in self.building_ids:
                # CO2 levels (400-2000 ppm)
                co2 = np.random.normal(800, 200)
                co2 = max(400, min(2000, co2))
                
                # TVOC levels (0-1000 ppb)
                tvoc = np.random.exponential(50)
                tvoc = min(1000, tvoc)
                
                # PM2.5 levels (0-50 μg/m³)
                pm25 = np.random.exponential(10)
                pm25 = min(50, pm25)
                
                # Temperature (18-26°C)
                temp = np.random.normal(22, 2)
                temp = max(18, min(26, temp))
                
                # Humidity (30-70%)
                humidity = np.random.normal(50, 10)
                humidity = max(30, min(70, humidity))
                
                # Air Quality Index (0-500)
                aqi = min(500, max(0, co2/4 + tvoc/2 + pm25*2))
                
                data.append({
                    'building_id': building_id,
                    'timestamp': current_date,
                    'co2_ppm': round(co2, 1),
                    'tvoc_ppb': round(tvoc, 1),
                    'pm25_ugm3': round(pm25, 1),
                    'temperature_c': round(temp, 1),
                    'humidity_percent': round(humidity, 1),
                    'air_quality_index': round(aqi, 1)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def generate_weather_data(self) -> pd.DataFrame:
        """Generate outdoor weather conditions"""
        data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            for building_id in self.building_ids:
                month = current_date.month
                
                # Temperature varies by season
                if month in [12, 1, 2]:  # Winter
                    temp = np.random.normal(-5, 5)
                elif month in [6, 7, 8]:  # Summer
                    temp = np.random.normal(25, 5)
                else:  # Spring/Fall
                    temp = np.random.normal(15, 5)
                
                # Humidity (30-90%)
                humidity = np.random.normal(60, 15)
                humidity = max(30, min(90, humidity))
                
                # Wind speed (0-20 m/s)
                wind_speed = np.random.exponential(3)
                wind_speed = min(20, wind_speed)
                
                # Wind direction (0-360°)
                wind_direction = np.random.uniform(0, 360)
                
                # Solar irradiance (0-1000 W/m²)
                if month in [6, 7, 8]:  # Summer
                    solar = np.random.normal(600, 200)
                elif month in [12, 1, 2]:  # Winter
                    solar = np.random.normal(200, 100)
                else:
                    solar = np.random.normal(400, 150)
                solar = max(0, min(1000, solar))
                
                # Precipitation (0-50 mm)
                precipitation = np.random.exponential(2)
                precipitation = min(50, precipitation)
                
                # Atmospheric pressure (950-1050 hPa)
                pressure = np.random.normal(1013, 20)
                pressure = max(950, min(1050, pressure))
                
                data.append({
                    'building_id': building_id,
                    'timestamp': current_date,
                    'outdoor_temp_c': round(temp, 1),
                    'outdoor_humidity_percent': round(humidity, 1),
                    'wind_speed_ms': round(wind_speed, 1),
                    'wind_direction_deg': round(wind_direction, 1),
                    'solar_irradiance_wm2': round(solar, 1),
                    'precipitation_mm': round(precipitation, 1),
                    'atmospheric_pressure_hpa': round(pressure, 1)
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)
    
    def generate_occupancy_data(self) -> pd.DataFrame:
        """Generate occupancy patterns"""
        data = []
        current_date = self.start_date
        
        while current_date <= self.end_date:
            for building_id in self.building_ids:
                hour = current_date.hour
                is_weekend = current_date.weekday() >= 5
                
                # Occupancy patterns vary by time of day
                if 6 <= hour <= 8:  # Morning peak
                    base_occupancy = 0.8
                elif 9 <= hour <= 17:  # Daytime
                    base_occupancy = 0.9
                elif 18 <= hour <= 22:  # Evening
                    base_occupancy = 0.7
                else:  # Night
                    base_occupancy = 0.1
                
                # Weekend patterns
                if is_weekend:
                    base_occupancy *= 0.6
                
                # Add noise
                occupancy_ratio = max(0, min(1, base_occupancy + np.random.normal(0, 0.1)))
                
                # Calculate occupancy count (assuming 50-200 people capacity)
                max_occupancy = np.random.randint(50, 201)
                occupancy_count = int(occupancy_ratio * max_occupancy)
                
                # Occupancy density (people per m²)
                floor_area = np.random.uniform(500, 2000)  # m²
                occupancy_density = occupancy_count / floor_area
                
                # Activity level
                if occupancy_ratio > 0.7:
                    activity_level = "high"
                elif occupancy_ratio > 0.3:
                    activity_level = "medium"
                else:
                    activity_level = "low"
                
                # Occupancy type
                occupancy_type = random.choice(["residential", "commercial", "mixed"])
                
                data.append({
                    'building_id': building_id,
                    'timestamp': current_date,
                    'occupancy_count': occupancy_count,
                    'occupancy_density_per_m2': round(occupancy_density, 4),
                    'activity_level': activity_level,
                    'occupancy_type': occupancy_type
                })
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(data)

class BuildingAttributesGenerator:
    """Generates building attributes and fabric data"""
    
    def __init__(self, num_buildings: int = 100):
        self.num_buildings = num_buildings
        self.building_ids = [f"B_{i:04d}" for i in range(1, num_buildings + 1)]
    
    def generate_basic_info(self) -> pd.DataFrame:
        """Generate basic building information"""
        data = []
        
        building_types = ["residential", "office", "retail", "educational", "healthcare", "industrial"]
        architectural_styles = ["modern", "traditional", "contemporary", "brutalist", "art_deco", "gothic"]
        quality_ratings = ["poor", "fair", "good", "excellent"]
        
        for i, building_id in enumerate(self.building_ids):
            # Construction year (1950-2020)
            construction_year = np.random.randint(1950, 2021)
            
            # Last renovation (if any)
            if np.random.random() < 0.7:  # 70% chance of renovation
                min_renovation_year = construction_year + 5
                max_renovation_year = 2021
                if min_renovation_year < max_renovation_year:
                    last_renovation = np.random.randint(min_renovation_year, max_renovation_year)
                else:
                    last_renovation = construction_year + 5
            else:
                last_renovation = None
            
            data.append({
                'building_id': building_id,
                'name': f"Building {i+1}",
                'address': f"{np.random.randint(1, 999)} Main Street, City {i//10 + 1}",
                'latitude': round(np.random.uniform(40.0, 50.0), 6),
                'longitude': round(np.random.uniform(-5.0, 10.0), 6),
                'construction_year': construction_year,
                'last_renovation_year': last_renovation,
                'building_type': random.choice(building_types),
                'architectural_style': random.choice(architectural_styles),
                'quality_rating': random.choice(quality_ratings)
            })
        
        return pd.DataFrame(data)
    
    def generate_geometric_data(self) -> pd.DataFrame:
        """Generate geometric and structural data"""
        data = []
        
        for building_id in self.building_ids:
            # Floor area (100-5000 m²)
            total_floor_area = np.random.uniform(100, 5000)
            
            # Height (3-50 m)
            height = np.random.uniform(3, 50)
            
            # Floor count
            floor_count = max(1, int(height / 3.5))
            
            # Volume
            volume = total_floor_area * height * 0.8  # 80% efficiency
            
            # Rooftop area (80-120% of floor area)
            rooftop_area = total_floor_area * np.random.uniform(0.8, 1.2)
            
            # Room count (varies with floor area)
            rooms_count = int(total_floor_area / np.random.uniform(15, 25))
            
            # Window area (15-30% of wall area)
            wall_area = total_floor_area * 2.5  # Rough estimate
            window_area = wall_area * np.random.uniform(0.15, 0.30)
            
            # Aspect ratio (length/width)
            aspect_ratio = np.random.uniform(1.0, 3.0)
            
            data.append({
                'building_id': building_id,
                'total_floor_area_m2': round(total_floor_area, 1),
                'rooftop_area_m2': round(rooftop_area, 1),
                'height_m': round(height, 1),
                'volume_m3': round(volume, 1),
                'floor_count': floor_count,
                'rooms_count': rooms_count,
                'window_area_m2': round(window_area, 1),
                'wall_area_m2': round(wall_area, 1),
                'aspect_ratio': round(aspect_ratio, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_thermal_properties(self) -> pd.DataFrame:
        """Generate thermal properties of building envelope"""
        data = []
        
        for building_id in self.building_ids:
            # U-values (W/m²K) - lower is better
            wall_u = np.random.uniform(0.2, 2.0)
            roof_u = np.random.uniform(0.15, 1.5)
            floor_u = np.random.uniform(0.2, 1.8)
            window_u = np.random.uniform(1.0, 3.0)
            
            # R-values (m²K/W) - higher is better
            wall_r = 1 / wall_u
            roof_r = 1 / roof_u
            floor_r = 1 / floor_u
            window_r = 1 / window_u
            
            # Thermal mass (kg)
            thermal_mass = np.random.uniform(50000, 500000)
            
            # Air tightness (air changes per hour)
            air_tightness = np.random.uniform(0.5, 5.0)
            
            data.append({
                'building_id': building_id,
                'wall_u_value_wm2k': round(wall_u, 3),
                'roof_u_value_wm2k': round(roof_u, 3),
                'floor_u_value_wm2k': round(floor_u, 3),
                'window_u_value_wm2k': round(window_u, 3),
                'wall_r_value_m2kw': round(wall_r, 3),
                'roof_r_value_m2kw': round(roof_r, 3),
                'floor_r_value_m2kw': round(floor_r, 3),
                'window_r_value_m2kw': round(window_r, 3),
                'thermal_mass_kg': round(thermal_mass, 0),
                'air_tightness_ach': round(air_tightness, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_construction_materials(self) -> pd.DataFrame:
        """Generate construction materials data"""
        data = []
        
        wall_materials = ["brick", "concrete", "steel", "wood", "stone", "composite"]
        roof_materials = ["tile", "metal", "membrane", "slate", "shingle", "green_roof"]
        floor_materials = ["concrete", "wood", "tile", "carpet", "vinyl", "stone"]
        window_materials = ["aluminum", "wood", "vinyl", "fiberglass", "steel"]
        insulation_types = ["fiberglass", "mineral_wool", "cellulose", "foam", "natural_fiber"]
        
        for building_id in self.building_ids:
            data.append({
                'building_id': building_id,
                'wall_material': random.choice(wall_materials),
                'roof_material': random.choice(roof_materials),
                'floor_material': random.choice(floor_materials),
                'window_material': random.choice(window_materials),
                'insulation_type': random.choice(insulation_types),
                'insulation_thickness_mm': round(np.random.uniform(50, 300), 1),
                'concrete_volume_m3': round(np.random.uniform(10, 500), 1),
                'steel_volume_m3': round(np.random.uniform(1, 100), 1),
                'wood_volume_m3': round(np.random.uniform(5, 200), 1)
            })
        
        return pd.DataFrame(data)

class EnergyPerformanceGenerator:
    """Generates energy performance data"""
    
    def __init__(self, num_buildings: int = 100, years: List[int] = [2018, 2019, 2020, 2021, 2022]):
        self.num_buildings = num_buildings
        self.years = years
        self.building_ids = [f"B_{i:04d}" for i in range(1, num_buildings + 1)]
    
    def generate_historical_consumption(self) -> pd.DataFrame:
        """Generate historical energy consumption data"""
        data = []
        
        for building_id in self.building_ids:
            for year in self.years:
                for month in range(1, 13):
                    # Base consumption varies by building age and type
                    base_consumption = np.random.uniform(2000, 8000)
                    
                    # Seasonal variation
                    if month in [12, 1, 2]:  # Winter
                        multiplier = 1.3
                    elif month in [6, 7, 8]:  # Summer
                        multiplier = 1.2
                    else:
                        multiplier = 1.0
                    
                    total_energy = base_consumption * multiplier * np.random.uniform(0.8, 1.2)
                    
                    # Distribute across end uses
                    heating = total_energy * np.random.uniform(0.3, 0.6)
                    cooling = total_energy * np.random.uniform(0.1, 0.3)
                    lighting = total_energy * np.random.uniform(0.1, 0.2)
                    appliances = total_energy * np.random.uniform(0.2, 0.4)
                    
                    # Energy intensity (kWh/m²)
                    floor_area = np.random.uniform(500, 2000)
                    energy_intensity = total_energy / floor_area
                    
                    data.append({
                        'building_id': building_id,
                        'year': year,
                        'month': month,
                        'total_energy_kwh': round(total_energy, 1),
                        'heating_energy_kwh': round(heating, 1),
                        'cooling_energy_kwh': round(cooling, 1),
                        'lighting_energy_kwh': round(lighting, 1),
                        'appliances_energy_kwh': round(appliances, 1),
                        'energy_intensity_kwhm2': round(energy_intensity, 2)
                    })
        
        return pd.DataFrame(data)
    
    def generate_efficiency_ratings(self) -> pd.DataFrame:
        """Generate energy efficiency ratings"""
        data = []
        
        eu_ratings = ["A", "B", "C", "D", "E", "F", "G"]
        rating_weights = [0.05, 0.15, 0.25, 0.25, 0.20, 0.08, 0.02]  # More buildings in middle ratings
        
        for building_id in self.building_ids:
            rating_year = np.random.randint(2018, 2023)
            eu_rating = np.random.choice(eu_ratings, p=rating_weights)
            
            # Energy performance index (0-200)
            if eu_rating == "A":
                epi = np.random.uniform(0, 50)
            elif eu_rating == "B":
                epi = np.random.uniform(50, 75)
            elif eu_rating == "C":
                epi = np.random.uniform(75, 100)
            elif eu_rating == "D":
                epi = np.random.uniform(100, 125)
            elif eu_rating == "E":
                epi = np.random.uniform(125, 150)
            elif eu_rating == "F":
                epi = np.random.uniform(150, 175)
            else:  # G
                epi = np.random.uniform(175, 200)
            
            # CO2 emissions (kg/m²)
            co2_emissions = epi * np.random.uniform(0.8, 1.2)
            
            # Primary energy demand (kWh/m²)
            primary_energy = epi * np.random.uniform(0.6, 1.0)
            
            # Renewable energy percentage
            renewable_percent = np.random.uniform(0, 100)
            
            data.append({
                'building_id': building_id,
                'rating_year': rating_year,
                'eu_energy_rating': eu_rating,
                'energy_performance_index': round(epi, 1),
                'co2_emissions_kgm2': round(co2_emissions, 1),
                'primary_energy_demand_kwhm2': round(primary_energy, 1),
                'renewable_energy_percent': round(renewable_percent, 1)
            })
        
        return pd.DataFrame(data)
    
    def generate_retrofit_impact(self) -> pd.DataFrame:
        """Generate retrofit impact data"""
        data = []
        
        retrofit_types = ["insulation", "windows", "hvac", "lighting", "renewable", "comprehensive"]
        
        for building_id in self.building_ids:
            # Only 40% of buildings have retrofit data
            if np.random.random() < 0.4:
                retrofit_year = np.random.randint(2018, 2023)
                retrofit_type = random.choice(retrofit_types)
                
                # Energy savings vary by retrofit type
                if retrofit_type == "comprehensive":
                    energy_savings = np.random.uniform(30, 60)
                elif retrofit_type == "insulation":
                    energy_savings = np.random.uniform(15, 35)
                elif retrofit_type == "windows":
                    energy_savings = np.random.uniform(10, 25)
                elif retrofit_type == "hvac":
                    energy_savings = np.random.uniform(20, 40)
                elif retrofit_type == "lighting":
                    energy_savings = np.random.uniform(5, 15)
                else:  # renewable
                    energy_savings = np.random.uniform(25, 50)
                
                co2_reduction = energy_savings * np.random.uniform(0.8, 1.2)
                
                # Cost varies by retrofit type and building size
                base_cost = np.random.uniform(10000, 100000)
                if retrofit_type == "comprehensive":
                    cost = base_cost * np.random.uniform(2, 4)
                else:
                    cost = base_cost * np.random.uniform(0.5, 2)
                
                # Payback period (years)
                payback_period = cost / (base_cost * energy_savings / 100) * np.random.uniform(0.5, 2)
                
                # Lifetime energy savings
                lifetime_years = 20
                annual_savings = base_cost * energy_savings / 100
                lifetime_savings = annual_savings * lifetime_years
                
                data.append({
                    'building_id': building_id,
                    'retrofit_year': retrofit_year,
                    'retrofit_type': retrofit_type,
                    'energy_savings_percent': round(energy_savings, 1),
                    'co2_reduction_percent': round(co2_reduction, 1),
                    'cost_euro': round(cost, 0),
                    'payback_period_years': round(payback_period, 1),
                    'lifetime_energy_savings_kwh': round(lifetime_savings, 0)
                })
        
        return pd.DataFrame(data)

class LCADataGenerator:
    """Generates Lifecycle Assessment data"""
    
    def __init__(self):
        self.material_categories = ["concrete", "steel", "wood", "glass", "insulation", "brick", "tile", "membrane"]
    
    def generate_material_epds(self) -> pd.DataFrame:
        """Generate Environmental Product Declarations for materials"""
        data = []
        
        for i, category in enumerate(self.material_categories):
            # Global Warming Potential (kg CO2 eq)
            if category == "concrete":
                gwp = np.random.uniform(100, 400)
            elif category == "steel":
                gwp = np.random.uniform(200, 800)
            elif category == "wood":
                gwp = np.random.uniform(50, 200)
            elif category == "glass":
                gwp = np.random.uniform(150, 300)
            elif category == "insulation":
                gwp = np.random.uniform(20, 100)
            else:
                gwp = np.random.uniform(50, 300)
            
            # Other impact categories
            acidification = gwp * np.random.uniform(0.1, 0.3)
            eutrophication = gwp * np.random.uniform(0.05, 0.15)
            ozone_depletion = gwp * np.random.uniform(0.001, 0.01)
            primary_energy = gwp * np.random.uniform(5, 15)
            water_consumption = gwp * np.random.uniform(10, 50)
            waste_generated = gwp * np.random.uniform(0.5, 2.0)
            
            data.append({
                'material_id': f"MAT_{i+1:03d}",
                'material_name': f"{category.title()} Material",
                'category': category,
                'global_warming_potential_kgco2eq': round(gwp, 2),
                'acidification_potential_kgso2eq': round(acidification, 2),
                'eutrophication_potential_kgpo4eq': round(eutrophication, 2),
                'ozone_depletion_potential_kgcfc11eq': round(ozone_depletion, 4),
                'primary_energy_demand_mj': round(primary_energy, 1),
                'water_consumption_l': round(water_consumption, 1),
                'waste_generated_kg': round(waste_generated, 2)
            })
        
        return pd.DataFrame(data)
    
    def generate_building_lca(self, building_ids: List[str]) -> pd.DataFrame:
        """Generate building-level LCA data"""
        data = []
        
        lifecycle_stages = ["production", "construction", "use", "end_of_life"]
        
        for building_id in building_ids:
            for stage in lifecycle_stages:
                # Total GWP varies by lifecycle stage
                if stage == "production":
                    total_gwp = np.random.uniform(50000, 200000)
                elif stage == "construction":
                    total_gwp = np.random.uniform(10000, 50000)
                elif stage == "use":
                    total_gwp = np.random.uniform(100000, 500000)
                else:  # end_of_life
                    total_gwp = np.random.uniform(5000, 25000)
                
                # Other impacts scale with GWP
                total_pe = total_gwp * np.random.uniform(8, 12)
                total_water = total_gwp * np.random.uniform(20, 80)
                total_waste = total_gwp * np.random.uniform(1, 3)
                
                # Recycling potential
                recycling_potential = np.random.uniform(20, 80)
                
                data.append({
                    'building_id': building_id,
                    'lifecycle_stage': stage,
                    'total_gwp_kgco2eq': round(total_gwp, 0),
                    'total_pe_mj': round(total_pe, 0),
                    'total_water_l': round(total_water, 0),
                    'total_waste_kg': round(total_waste, 0),
                    'recycling_potential_percent': round(recycling_potential, 1)
                })
        
        return pd.DataFrame(data)