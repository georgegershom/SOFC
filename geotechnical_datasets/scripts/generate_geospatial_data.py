#!/usr/bin/env python3
"""
Generate synthetic geospatial datasets for regional geotechnical analysis
Focus: Large-scale soil property mapping and hazard assessment
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_regional_soil_grid(region_name, bounds, grid_resolution=0.01):
    """Generate gridded soil property data for a region"""
    
    lon_min, lat_min, lon_max, lat_max = bounds
    
    # Create coordinate grid
    longitudes = np.arange(lon_min, lon_max, grid_resolution)
    latitudes = np.arange(lat_min, lat_max, grid_resolution)
    
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)
    
    # Flatten for DataFrame
    lon_flat = lon_grid.flatten()
    lat_flat = lat_grid.flatten()
    n_points = len(lon_flat)
    
    # Generate spatial patterns using multiple sine waves for realistic variation
    x_norm = (lon_flat - lon_min) / (lon_max - lon_min)
    y_norm = (lat_flat - lat_min) / (lat_max - lat_min)
    
    # Base geological patterns
    pattern1 = np.sin(2 * np.pi * x_norm * 3) * np.cos(2 * np.pi * y_norm * 2)
    pattern2 = np.sin(2 * np.pi * x_norm * 1.5) * np.sin(2 * np.pi * y_norm * 1.8)
    pattern3 = np.cos(2 * np.pi * x_norm * 0.8) * np.cos(2 * np.pi * y_norm * 1.2)
    
    # Sand content (%) - varies with geological patterns
    sand_base = 50 + 25 * pattern1 + 15 * pattern2
    sand_content = sand_base + np.random.normal(0, 8, n_points)
    sand_content = np.clip(sand_content, 10, 90)
    
    # Clay content (%) - inversely correlated with sand
    clay_base = 80 - sand_content + 10 * pattern3
    clay_content = clay_base + np.random.normal(0, 5, n_points)
    clay_content = np.clip(clay_content, 5, 70)
    
    # Silt content (remainder)
    silt_content = 100 - sand_content - clay_content
    silt_content = np.clip(silt_content, 5, 60)
    
    # Normalize to 100%
    total = sand_content + clay_content + silt_content
    sand_content = sand_content / total * 100
    clay_content = clay_content / total * 100
    silt_content = silt_content / total * 100
    
    # Plasticity index - correlated with clay content
    plasticity_index = np.where(
        clay_content > 20,
        15 + 0.8 * clay_content + np.random.normal(0, 5, n_points),
        np.random.exponential(5, n_points)
    )
    plasticity_index = np.clip(plasticity_index, 0, 80)
    
    # Liquid limit
    liquid_limit = plasticity_index + np.random.uniform(15, 35, n_points)
    
    # N-SPT values (blow counts)
    n_spt_base = 15 + 20 * (sand_content / 100) + 10 * pattern2
    n_spt = n_spt_base + np.random.normal(0, 5, n_points)
    n_spt = np.clip(n_spt, 2, 50)
    
    # Relative density (for sandy areas)
    relative_density = np.where(
        sand_content > 50,
        30 + 1.2 * n_spt + np.random.normal(0, 10, n_points),
        np.random.uniform(20, 60, n_points)
    )
    relative_density = np.clip(relative_density, 15, 95)
    
    # Undrained shear strength (for clayey areas)
    undrained_strength = np.where(
        clay_content > 30,
        20 + 2 * n_spt + 0.5 * plasticity_index + np.random.normal(0, 15, n_points),
        np.random.uniform(50, 150, n_points)
    )
    undrained_strength = np.clip(undrained_strength, 10, 300)
    
    # Depth to bedrock (m)
    bedrock_depth_base = 10 + 15 * pattern1 + 20 * pattern3
    bedrock_depth = bedrock_depth_base + np.random.normal(0, 5, n_points)
    bedrock_depth = np.clip(bedrock_depth, 2, 50)
    
    # Groundwater depth (m)
    groundwater_depth_base = 3 + 8 * pattern2 + 5 * pattern3
    groundwater_depth = groundwater_depth_base + np.random.normal(0, 2, n_points)
    groundwater_depth = np.clip(groundwater_depth, 0.5, 20)
    
    # Ensure groundwater is above bedrock
    groundwater_depth = np.minimum(groundwater_depth, bedrock_depth - 1)
    
    # Slope angle (degrees)
    # Create elevation pattern
    elevation_base = 100 + 200 * pattern1 + 150 * pattern2
    elevation = elevation_base + np.random.normal(0, 20, n_points)
    
    # Calculate slope (simplified gradient)
    slope_angle = np.abs(np.gradient(elevation.reshape(lat_grid.shape), axis=0).flatten()) * 100
    slope_angle = np.clip(slope_angle, 0, 45)
    
    # Seismic hazard parameters
    peak_ground_acceleration = 0.1 + 0.3 * np.abs(pattern1) + np.random.normal(0, 0.05, n_points)
    peak_ground_acceleration = np.clip(peak_ground_acceleration, 0.05, 0.8)
    
    # Liquefaction susceptibility (for sandy areas)
    liquefaction_susceptibility = np.where(
        (sand_content > 60) & (groundwater_depth < 10) & (relative_density < 70),
        'High',
        np.where(
            (sand_content > 40) & (groundwater_depth < 15),
            'Medium',
            'Low'
        )
    )
    
    # Landslide susceptibility (for clayey/steep areas)
    landslide_susceptibility = np.where(
        (clay_content > 40) & (slope_angle > 20) & (plasticity_index > 25),
        'High',
        np.where(
            (slope_angle > 15) & (clay_content > 25),
            'Medium',
            'Low'
        )
    )
    
    return pd.DataFrame({
        'longitude': lon_flat,
        'latitude': lat_flat,
        'region': region_name,
        'sand_content_pct': sand_content,
        'clay_content_pct': clay_content,
        'silt_content_pct': silt_content,
        'plasticity_index': plasticity_index,
        'liquid_limit': liquid_limit,
        'n_spt_blows': n_spt,
        'relative_density_pct': relative_density,
        'undrained_shear_strength_kPa': undrained_strength,
        'depth_to_bedrock_m': bedrock_depth,
        'groundwater_depth_m': groundwater_depth,
        'elevation_m': elevation,
        'slope_angle_deg': slope_angle,
        'peak_ground_acceleration_g': peak_ground_acceleration,
        'liquefaction_susceptibility': liquefaction_susceptibility,
        'landslide_susceptibility': landslide_susceptibility
    })

def generate_borehole_data(regional_grid, n_boreholes=200):
    """Generate detailed borehole data at selected locations"""
    
    # Sample locations from the regional grid
    sampled_locations = regional_grid.sample(n=n_boreholes)
    
    borehole_data = []
    
    for idx, location in sampled_locations.iterrows():
        borehole_id = f"BH_{idx:04d}"
        
        # Generate depth profile (layers)
        max_depth = min(location['depth_to_bedrock_m'], 30)  # Maximum 30m depth
        n_layers = np.random.randint(3, 8)
        
        layer_boundaries = np.sort(np.random.uniform(0, max_depth, n_layers-1))
        layer_boundaries = np.concatenate([[0], layer_boundaries, [max_depth]])
        
        for i in range(len(layer_boundaries)-1):
            depth_top = layer_boundaries[i]
            depth_bottom = layer_boundaries[i+1]
            depth_mid = (depth_top + depth_bottom) / 2
            
            # Vary properties with depth
            depth_factor = depth_mid / max_depth
            
            # Soil type variation with depth
            if depth_factor < 0.3:  # Shallow layers - more variable
                sand_var = np.random.normal(0, 15)
                clay_var = np.random.normal(0, 10)
            else:  # Deeper layers - more consistent
                sand_var = np.random.normal(0, 8)
                clay_var = np.random.normal(0, 5)
            
            sand_content = location['sand_content_pct'] + sand_var
            clay_content = location['clay_content_pct'] + clay_var
            
            # Normalize
            sand_content = np.clip(sand_content, 5, 90)
            clay_content = np.clip(clay_content, 5, 85)
            silt_content = 100 - sand_content - clay_content
            
            if silt_content < 5:
                total = sand_content + clay_content + 5
                sand_content = sand_content / total * 95
                clay_content = clay_content / total * 95
                silt_content = 5
            
            # Classify soil type
            if sand_content > 50:
                if silt_content + clay_content < 12:
                    soil_type = "Sand (SP/SW)"
                else:
                    soil_type = "Silty/Clayey Sand (SM/SC)"
            elif clay_content > 50:
                if location['plasticity_index'] > 25:
                    soil_type = "Fat Clay (CH)"
                else:
                    soil_type = "Lean Clay (CL)"
            else:
                soil_type = "Silt (ML/MH)"
            
            # Mechanical properties
            n_spt = location['n_spt_blows'] + np.random.normal(0, 3)
            n_spt = np.clip(n_spt, 1, 50)
            
            # Increase strength with depth
            strength_increase = 1 + 0.5 * depth_factor
            
            if sand_content > 50:
                friction_angle = 28 + 0.3 * n_spt + np.random.normal(0, 2)
                friction_angle = np.clip(friction_angle, 25, 45)
                undrained_strength = np.nan
            else:
                undrained_strength = location['undrained_shear_strength_kPa'] * strength_increase
                undrained_strength += np.random.normal(0, 10)
                undrained_strength = np.clip(undrained_strength, 15, 400)
                friction_angle = np.nan
            
            # Unit weight
            if sand_content > 50:
                unit_weight = 17 + 0.1 * n_spt + np.random.normal(0, 1)
            else:
                unit_weight = 16 + 0.05 * location['plasticity_index'] + np.random.normal(0, 1)
            unit_weight = np.clip(unit_weight, 14, 22)
            
            # Permeability
            if sand_content > 60:
                permeability = np.random.lognormal(np.log(1e-4), 1.0)
            elif clay_content > 40:
                permeability = np.random.lognormal(np.log(1e-9), 1.5)
            else:
                permeability = np.random.lognormal(np.log(1e-7), 1.2)
            
            borehole_data.append({
                'borehole_id': borehole_id,
                'longitude': location['longitude'],
                'latitude': location['latitude'],
                'depth_top_m': depth_top,
                'depth_bottom_m': depth_bottom,
                'layer_thickness_m': depth_bottom - depth_top,
                'soil_type': soil_type,
                'sand_content_pct': sand_content,
                'clay_content_pct': clay_content,
                'silt_content_pct': silt_content,
                'n_spt_blows': n_spt,
                'friction_angle_deg': friction_angle,
                'undrained_shear_strength_kPa': undrained_strength,
                'unit_weight_kN_m3': unit_weight,
                'permeability_m_s': permeability,
                'plasticity_index': location['plasticity_index'] + np.random.normal(0, 3),
                'liquid_limit': location['liquid_limit'] + np.random.normal(0, 5)
            })
    
    return pd.DataFrame(borehole_data)

def generate_hazard_maps(regional_grid):
    """Generate hazard assessment maps"""
    
    # Liquefaction hazard mapping
    liquefaction_hazard = []
    
    for _, point in regional_grid.iterrows():
        # Liquefaction potential index calculation
        if point['sand_content_pct'] > 40:
            # Simplified LPI calculation
            depth_factor = min(point['groundwater_depth_m'], 20) / 20
            density_factor = (100 - point['relative_density_pct']) / 100
            seismic_factor = point['peak_ground_acceleration_g'] / 0.4
            
            lpi = (1 - depth_factor) * density_factor * seismic_factor * 15
            lpi = np.clip(lpi, 0, 20)
            
            if lpi > 10:
                hazard_level = "Very High"
            elif lpi > 5:
                hazard_level = "High"
            elif lpi > 2:
                hazard_level = "Moderate"
            else:
                hazard_level = "Low"
        else:
            lpi = 0
            hazard_level = "Very Low"
        
        liquefaction_hazard.append({
            'longitude': point['longitude'],
            'latitude': point['latitude'],
            'liquefaction_potential_index': lpi,
            'liquefaction_hazard_level': hazard_level
        })
    
    # Landslide hazard mapping
    landslide_hazard = []
    
    for _, point in regional_grid.iterrows():
        # Simplified slope stability analysis
        if point['clay_content_pct'] > 30:
            # Factor of safety estimation
            cohesion = point['undrained_shear_strength_kPa']
            slope_rad = np.radians(point['slope_angle_deg'])
            unit_weight = 18  # Assumed
            
            if point['slope_angle_deg'] > 5:
                fs = cohesion / (unit_weight * point['groundwater_depth_m'] * np.sin(slope_rad))
                fs = np.clip(fs, 0.5, 5.0)
            else:
                fs = 5.0  # Very stable for flat areas
            
            if fs < 1.2:
                hazard_level = "Very High"
            elif fs < 1.5:
                hazard_level = "High"
            elif fs < 2.0:
                hazard_level = "Moderate"
            else:
                hazard_level = "Low"
        else:
            fs = 3.0
            hazard_level = "Low"
        
        landslide_hazard.append({
            'longitude': point['longitude'],
            'latitude': point['latitude'],
            'factor_of_safety': fs,
            'landslide_hazard_level': hazard_level
        })
    
    return pd.DataFrame(liquefaction_hazard), pd.DataFrame(landslide_hazard)

def main():
    """Generate all geospatial datasets"""
    
    print("Generating geospatial datasets...")
    
    # Define study regions
    regions = {
        'San_Francisco_Bay': (-122.5, 37.2, -121.8, 38.0),
        'Los_Angeles_Basin': (-118.8, 33.7, -117.6, 34.3),
        'Puget_Sound': (-122.8, 47.0, -121.5, 47.8)
    }
    
    all_regional_data = []
    all_borehole_data = []
    all_liquefaction_hazard = []
    all_landslide_hazard = []
    
    for region_name, bounds in regions.items():
        print(f"Processing region: {region_name}")
        
        # Generate regional grid
        regional_grid = generate_regional_soil_grid(region_name, bounds, grid_resolution=0.005)
        all_regional_data.append(regional_grid)
        
        # Generate borehole data
        borehole_data = generate_borehole_data(regional_grid, n_boreholes=100)
        all_borehole_data.append(borehole_data)
        
        # Generate hazard maps
        liq_hazard, land_hazard = generate_hazard_maps(regional_grid)
        all_liquefaction_hazard.append(liq_hazard)
        all_landslide_hazard.append(land_hazard)
    
    # Combine all data
    combined_regional = pd.concat(all_regional_data, ignore_index=True)
    combined_boreholes = pd.concat(all_borehole_data, ignore_index=True)
    combined_liq_hazard = pd.concat(all_liquefaction_hazard, ignore_index=True)
    combined_land_hazard = pd.concat(all_landslide_hazard, ignore_index=True)
    
    # Save datasets
    os.makedirs('geotechnical_datasets/geospatial_data', exist_ok=True)
    
    combined_regional.to_csv('geotechnical_datasets/geospatial_data/regional_soil_properties_grid.csv', index=False)
    combined_boreholes.to_csv('geotechnical_datasets/geospatial_data/borehole_data_detailed.csv', index=False)
    combined_liq_hazard.to_csv('geotechnical_datasets/geospatial_data/liquefaction_hazard_map.csv', index=False)
    combined_land_hazard.to_csv('geotechnical_datasets/geospatial_data/landslide_hazard_map.csv', index=False)
    
    # Create metadata
    metadata = {
        'dataset_info': {
            'title': 'Regional Geotechnical Properties and Hazard Assessment',
            'description': 'Synthetic geospatial datasets for geotechnical analysis',
            'regions_covered': list(regions.keys()),
            'coordinate_system': 'WGS84 (EPSG:4326)',
            'grid_resolution': '0.005 degrees (~500m)',
            'generation_date': datetime.now().isoformat()
        },
        'regional_grid': {
            'total_points': len(combined_regional),
            'parameters': list(combined_regional.columns),
            'spatial_extent': {
                'longitude_range': [float(combined_regional['longitude'].min()), 
                                  float(combined_regional['longitude'].max())],
                'latitude_range': [float(combined_regional['latitude'].min()), 
                                 float(combined_regional['latitude'].max())]
            }
        },
        'borehole_data': {
            'total_boreholes': len(combined_boreholes['borehole_id'].unique()),
            'total_layers': len(combined_boreholes),
            'depth_range': [float(combined_boreholes['depth_bottom_m'].min()),
                           float(combined_boreholes['depth_bottom_m'].max())]
        },
        'hazard_maps': {
            'liquefaction_hazard_levels': combined_liq_hazard['liquefaction_hazard_level'].value_counts().to_dict(),
            'landslide_hazard_levels': combined_land_hazard['landslide_hazard_level'].value_counts().to_dict()
        }
    }
    
    with open('geotechnical_datasets/geospatial_data/geospatial_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Generated regional grid with {len(combined_regional)} points")
    print(f"Generated {len(combined_boreholes['borehole_id'].unique())} boreholes with {len(combined_boreholes)} layers")
    print("Files created:")
    print("- regional_soil_properties_grid.csv")
    print("- borehole_data_detailed.csv")
    print("- liquefaction_hazard_map.csv")
    print("- landslide_hazard_map.csv")
    print("- geospatial_metadata.json")
    
    return combined_regional, combined_boreholes, combined_liq_hazard, combined_land_hazard

if __name__ == "__main__":
    regional_data, borehole_data, liq_hazard, land_hazard = main()