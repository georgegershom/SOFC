"""
Dataset Schema for AI- and IoT-driven Building Retrofit Optimization
PhD Thesis Dataset Structure

This module defines the comprehensive schema for the multi-faceted dataset
integrating IoT sensor data, building attributes, energy performance, and LCA data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json

class BuildingRetrofitDatasetSchema:
    """Schema definition for the building retrofit optimization dataset"""
    
    def __init__(self):
        self.schema = {
            "iot_sensor_data": {
                "energy_consumption": {
                    "building_id": "string",
                    "timestamp": "datetime",
                    "total_consumption_kwh": "float",
                    "heating_kwh": "float",
                    "cooling_kwh": "float",
                    "lighting_kwh": "float",
                    "appliances_kwh": "float",
                    "hvac_kwh": "float",
                    "other_kwh": "float"
                },
                "environmental_parameters": {
                    "building_id": "string",
                    "timestamp": "datetime",
                    "co2_ppm": "float",
                    "tvoc_ppb": "float",
                    "pm25_ugm3": "float",
                    "temperature_c": "float",
                    "humidity_percent": "float",
                    "air_quality_index": "float"
                },
                "weather_conditions": {
                    "building_id": "string",
                    "timestamp": "datetime",
                    "outdoor_temp_c": "float",
                    "outdoor_humidity_percent": "float",
                    "wind_speed_ms": "float",
                    "wind_direction_deg": "float",
                    "solar_irradiance_wm2": "float",
                    "precipitation_mm": "float",
                    "atmospheric_pressure_hpa": "float"
                },
                "occupancy_patterns": {
                    "building_id": "string",
                    "timestamp": "datetime",
                    "occupancy_count": "integer",
                    "occupancy_density_per_m2": "float",
                    "activity_level": "string",  # low, medium, high
                    "occupancy_type": "string"  # residential, commercial, mixed
                }
            },
            "building_attributes": {
                "basic_info": {
                    "building_id": "string",
                    "name": "string",
                    "address": "string",
                    "latitude": "float",
                    "longitude": "float",
                    "construction_year": "integer",
                    "last_renovation_year": "integer",
                    "building_type": "string",
                    "architectural_style": "string",
                    "quality_rating": "string"
                },
                "geometric_data": {
                    "building_id": "string",
                    "total_floor_area_m2": "float",
                    "rooftop_area_m2": "float",
                    "height_m": "float",
                    "volume_m3": "float",
                    "floor_count": "integer",
                    "rooms_count": "integer",
                    "window_area_m2": "float",
                    "wall_area_m2": "float",
                    "aspect_ratio": "float"
                },
                "thermal_properties": {
                    "building_id": "string",
                    "wall_u_value_wm2k": "float",
                    "roof_u_value_wm2k": "float",
                    "floor_u_value_wm2k": "float",
                    "window_u_value_wm2k": "float",
                    "wall_r_value_m2kw": "float",
                    "roof_r_value_m2kw": "float",
                    "floor_r_value_m2kw": "float",
                    "window_r_value_m2kw": "float",
                    "thermal_mass_kg": "float",
                    "air_tightness_ach": "float"
                },
                "construction_materials": {
                    "building_id": "string",
                    "wall_material": "string",
                    "roof_material": "string",
                    "floor_material": "string",
                    "window_material": "string",
                    "insulation_type": "string",
                    "insulation_thickness_mm": "float",
                    "concrete_volume_m3": "float",
                    "steel_volume_m3": "float",
                    "wood_volume_m3": "float"
                }
            },
            "energy_performance": {
                "historical_consumption": {
                    "building_id": "string",
                    "year": "integer",
                    "month": "integer",
                    "total_energy_kwh": "float",
                    "heating_energy_kwh": "float",
                    "cooling_energy_kwh": "float",
                    "lighting_energy_kwh": "float",
                    "appliances_energy_kwh": "float",
                    "energy_intensity_kwhm2": "float"
                },
                "efficiency_ratings": {
                    "building_id": "string",
                    "rating_year": "integer",
                    "eu_energy_rating": "string",  # A, B, C, D, E, F, G
                    "energy_performance_index": "float",
                    "co2_emissions_kgm2": "float",
                    "primary_energy_demand_kwhm2": "float",
                    "renewable_energy_percent": "float"
                },
                "retrofit_impact": {
                    "building_id": "string",
                    "retrofit_year": "integer",
                    "retrofit_type": "string",
                    "energy_savings_percent": "float",
                    "co2_reduction_percent": "float",
                    "cost_euro": "float",
                    "payback_period_years": "float",
                    "lifetime_energy_savings_kwh": "float"
                }
            },
            "lifecycle_assessment": {
                "material_epds": {
                    "material_id": "string",
                    "material_name": "string",
                    "category": "string",
                    "global_warming_potential_kgco2eq": "float",
                    "acidification_potential_kgso2eq": "float",
                    "eutrophication_potential_kgpo4eq": "float",
                    "ozone_depletion_potential_kgcfc11eq": "float",
                    "primary_energy_demand_mj": "float",
                    "water_consumption_l": "float",
                    "waste_generated_kg": "float"
                },
                "building_lca": {
                    "building_id": "string",
                    "lifecycle_stage": "string",  # production, construction, use, end_of_life
                    "total_gwp_kgco2eq": "float",
                    "total_pe_mj": "float",
                    "total_water_l": "float",
                    "total_waste_kg": "float",
                    "recycling_potential_percent": "float"
                }
            }
        }
    
    def get_schema(self) -> Dict[str, Any]:
        """Return the complete schema definition"""
        return self.schema
    
    def validate_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Validate a dataframe against the schema"""
        if table_name not in self.schema:
            return False
        
        expected_columns = set(self.schema[table_name].keys())
        actual_columns = set(df.columns)
        
        return expected_columns.issubset(actual_columns)
    
    def create_empty_dataframe(self, table_name: str) -> pd.DataFrame:
        """Create an empty dataframe with the correct schema"""
        if table_name not in self.schema:
            raise ValueError(f"Unknown table: {table_name}")
        
        columns = list(self.schema[table_name].keys())
        return pd.DataFrame(columns=columns)

# Initialize schema
schema = BuildingRetrofitDatasetSchema()