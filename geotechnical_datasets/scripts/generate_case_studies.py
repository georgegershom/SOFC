#!/usr/bin/env python3
"""
Generate case study datasets for geotechnical failure mechanisms
Focus: Real-world failure scenarios in sandy and clay soils
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_sandy_soil_case_studies(n_cases=50):
    """Generate case studies of sandy soil failures"""
    
    # Case study metadata
    case_ids = [f"SANDY_CASE_{i+1:03d}" for i in range(n_cases)]
    
    # Geographic locations (major seismic regions)
    locations = [
        "San Francisco Bay Area, CA", "Los Angeles Basin, CA", "Seattle, WA",
        "Charleston, SC", "New Madrid, MO", "Anchorage, AK", "Memphis, TN",
        "Boston, MA", "Salt Lake City, UT", "Portland, OR"
    ]
    
    case_locations = np.random.choice(locations, n_cases)
    
    # Failure types
    failure_types = [
        "Liquefaction-induced settlement", "Liquefaction-induced lateral spreading",
        "Liquefaction-induced uplift", "Cyclic mobility", "Flow liquefaction",
        "Foundation bearing capacity failure", "Slope instability in loose sand"
    ]
    
    failure_type = np.random.choice(failure_types, n_cases)
    
    # Structure types affected
    structure_types = [
        "Residential building", "Commercial building", "Bridge foundation",
        "Tunnel", "Pipeline", "Retaining wall", "Embankment", "Port structure"
    ]
    
    structure_type = np.random.choice(structure_types, n_cases)
    
    # Earthquake parameters (for seismic cases)
    earthquake_magnitude = np.random.uniform(5.5, 8.5, n_cases)
    peak_ground_acceleration = np.random.uniform(0.1, 0.8, n_cases)  # g
    duration_strong_motion = np.random.uniform(10, 120, n_cases)  # seconds
    
    # Soil conditions at failure
    depth_to_groundwater = np.random.uniform(0.5, 8.0, n_cases)  # m
    relative_density = np.random.uniform(20, 70, n_cases)  # %
    fines_content = np.random.uniform(5, 35, n_cases)  # %
    
    # Liquefaction manifestations
    settlement_mm = np.where(
        np.isin(failure_type, ["Liquefaction-induced settlement", "Flow liquefaction"]),
        np.random.lognormal(np.log(150), 0.8, n_cases),
        np.random.exponential(20, n_cases)
    )
    settlement_mm = np.clip(settlement_mm, 10, 2000)
    
    lateral_displacement_m = np.where(
        failure_type == "Liquefaction-induced lateral spreading",
        np.random.lognormal(np.log(2.0), 0.6, n_cases),
        np.random.exponential(0.3, n_cases)
    )
    lateral_displacement_m = np.clip(lateral_displacement_m, 0.1, 15)
    
    # Pore pressure measurements
    initial_pore_pressure = 9.81 * depth_to_groundwater  # kPa
    peak_excess_pore_pressure_ratio = np.random.uniform(0.3, 1.0, n_cases)
    
    # Time to liquefaction
    time_to_liquefaction = np.random.lognormal(np.log(15), 0.5, n_cases)  # seconds
    time_to_liquefaction = np.clip(time_to_liquefaction, 5, 60)
    
    # Recovery time
    recovery_time_days = np.random.lognormal(np.log(30), 0.8, n_cases)
    recovery_time_days = np.clip(recovery_time_days, 7, 365)
    
    # Economic impact
    damage_cost_usd = np.random.lognormal(np.log(500000), 1.2, n_cases)
    damage_cost_usd = np.clip(damage_cost_usd, 10000, 50000000)
    
    # Mitigation measures implemented
    mitigation_measures = [
        "Ground improvement (stone columns)", "Deep foundations", "Soil densification",
        "Dewatering", "Soil replacement", "Ground freezing", "Grouting", "None"
    ]
    
    mitigation = np.random.choice(mitigation_measures, n_cases)
    
    # Case study dates
    base_date = datetime(1990, 1, 1)
    case_dates = [base_date + timedelta(days=np.random.randint(0, 12000)) for _ in range(n_cases)]
    
    return pd.DataFrame({
        'case_id': case_ids,
        'case_date': case_dates,
        'location': case_locations,
        'failure_type': failure_type,
        'structure_type': structure_type,
        'earthquake_magnitude': earthquake_magnitude,
        'peak_ground_acceleration_g': peak_ground_acceleration,
        'duration_strong_motion_s': duration_strong_motion,
        'depth_to_groundwater_m': depth_to_groundwater,
        'relative_density_pct': relative_density,
        'fines_content_pct': fines_content,
        'settlement_mm': settlement_mm,
        'lateral_displacement_m': lateral_displacement_m,
        'initial_pore_pressure_kPa': initial_pore_pressure,
        'peak_excess_pore_pressure_ratio': peak_excess_pore_pressure_ratio,
        'time_to_liquefaction_s': time_to_liquefaction,
        'recovery_time_days': recovery_time_days,
        'damage_cost_usd': damage_cost_usd,
        'mitigation_measures': mitigation
    })

def generate_clay_soil_case_studies(n_cases=40):
    """Generate case studies of clay soil failures"""
    
    # Case study metadata
    case_ids = [f"CLAY_CASE_{i+1:03d}" for i in range(n_cases)]
    
    # Geographic locations (areas with problematic clays)
    locations = [
        "Houston, TX", "Denver, CO", "London, UK", "Paris, France",
        "Mexico City, Mexico", "Istanbul, Turkey", "Beijing, China",
        "Montreal, Canada", "Oslo, Norway", "Adelaide, Australia"
    ]
    
    case_locations = np.random.choice(locations, n_cases)
    
    # Failure types
    failure_types = [
        "Progressive slope failure", "Circular slip failure", "Retrogressive landslide",
        "Foundation heave", "Tunnel face instability", "Excavation collapse",
        "Embankment failure", "Quick clay landslide"
    ]
    
    failure_type = np.random.choice(failure_types, n_cases)
    
    # Structure types affected
    structure_types = [
        "Residential building", "Highway embankment", "Railway cutting",
        "Tunnel", "Excavation", "Slope", "Dam", "Retaining structure"
    ]
    
    structure_type = np.random.choice(structure_types, n_cases)
    
    # Triggering events
    triggers = [
        "Heavy rainfall", "Rapid drawdown", "Excavation", "Loading",
        "Freeze-thaw cycles", "Earthquake", "Groundwater change", "Construction activity"
    ]
    
    trigger_event = np.random.choice(triggers, n_cases)
    
    # Clay properties at failure site
    plasticity_index = np.random.uniform(15, 80, n_cases)
    liquid_limit = plasticity_index + np.random.uniform(20, 40, n_cases)
    sensitivity = np.random.lognormal(np.log(3), 0.8, n_cases)
    sensitivity = np.clip(sensitivity, 1.5, 50)
    
    # Strength parameters
    undrained_shear_strength = np.random.uniform(20, 150, n_cases)  # kPa
    effective_friction_angle = np.random.uniform(18, 32, n_cases)  # degrees
    
    # Failure geometry
    slope_height_m = np.random.uniform(5, 50, n_cases)
    slope_angle_deg = np.random.uniform(15, 45, n_cases)
    failure_depth_m = np.random.uniform(2, 25, n_cases)
    
    # Slip surface characteristics
    slip_surface_types = ["Circular", "Translational", "Compound", "Wedge"]
    slip_surface_type = np.random.choice(slip_surface_types, n_cases)
    
    # Factor of safety at failure (should be close to 1.0)
    factor_of_safety = np.random.normal(1.0, 0.15, n_cases)
    factor_of_safety = np.clip(factor_of_safety, 0.7, 1.3)
    
    # Groundwater conditions
    groundwater_depth_m = np.random.uniform(1, 15, n_cases)
    pore_pressure_ratio = np.random.uniform(0.2, 0.8, n_cases)
    
    # Failure kinematics
    displacement_m = np.random.lognormal(np.log(5), 1.0, n_cases)
    displacement_m = np.clip(displacement_m, 0.5, 100)
    
    velocity_m_day = np.random.lognormal(np.log(0.1), 2.0, n_cases)
    velocity_m_day = np.clip(velocity_m_day, 0.001, 50)
    
    # Volume of failed material
    failure_volume_m3 = np.random.lognormal(np.log(10000), 1.5, n_cases)
    failure_volume_m3 = np.clip(failure_volume_m3, 100, 1000000)
    
    # Time factors
    time_to_failure_hours = np.random.lognormal(np.log(24), 1.5, n_cases)
    time_to_failure_hours = np.clip(time_to_failure_hours, 1, 8760)  # up to 1 year
    
    # Economic and social impact
    damage_cost_usd = np.random.lognormal(np.log(1000000), 1.0, n_cases)
    damage_cost_usd = np.clip(damage_cost_usd, 50000, 100000000)
    
    casualties = np.random.poisson(2, n_cases)
    people_evacuated = np.random.poisson(50, n_cases)
    
    # Remedial measures
    remedial_measures = [
        "Slope regrading", "Drainage installation", "Soil nailing", "Retaining wall",
        "Ground anchors", "Lime stabilization", "Cement stabilization", "Relocation"
    ]
    
    remedial_action = np.random.choice(remedial_measures, n_cases)
    
    # Case study dates
    base_date = datetime(1985, 1, 1)
    case_dates = [base_date + timedelta(days=np.random.randint(0, 14000)) for _ in range(n_cases)]
    
    return pd.DataFrame({
        'case_id': case_ids,
        'case_date': case_dates,
        'location': case_locations,
        'failure_type': failure_type,
        'structure_type': structure_type,
        'trigger_event': trigger_event,
        'plasticity_index': plasticity_index,
        'liquid_limit': liquid_limit,
        'sensitivity': sensitivity,
        'undrained_shear_strength_kPa': undrained_shear_strength,
        'effective_friction_angle_deg': effective_friction_angle,
        'slope_height_m': slope_height_m,
        'slope_angle_deg': slope_angle_deg,
        'failure_depth_m': failure_depth_m,
        'slip_surface_type': slip_surface_type,
        'factor_of_safety': factor_of_safety,
        'groundwater_depth_m': groundwater_depth_m,
        'pore_pressure_ratio': pore_pressure_ratio,
        'displacement_m': displacement_m,
        'velocity_m_day': velocity_m_day,
        'failure_volume_m3': failure_volume_m3,
        'time_to_failure_hours': time_to_failure_hours,
        'damage_cost_usd': damage_cost_usd,
        'casualties': casualties,
        'people_evacuated': people_evacuated,
        'remedial_action': remedial_action
    })

def generate_monitoring_data(case_studies_df, soil_type='sandy'):
    """Generate time-series monitoring data for selected case studies"""
    
    # Select subset of cases with monitoring data
    monitored_cases = case_studies_df.sample(n=min(10, len(case_studies_df)))
    
    monitoring_data = []
    
    for _, case in monitored_cases.iterrows():
        case_id = case['case_id']
        
        # Generate time series data leading up to failure
        if soil_type == 'sandy':
            # Monitoring duration (days before failure)
            monitoring_days = np.random.randint(30, 180)
            
            # Time points
            time_points = np.linspace(-monitoring_days, 0, monitoring_days + 1)
            
            # Pore pressure evolution
            base_pore_pressure = case['initial_pore_pressure_kPa']
            pore_pressure = base_pore_pressure + np.random.normal(0, 5, len(time_points))
            
            # Add earthquake event (sudden spike)
            earthquake_time = -np.random.randint(1, 5)  # days before failure
            earthquake_idx = np.argmin(np.abs(time_points - earthquake_time))
            
            # Sudden increase during earthquake
            excess_pore_pressure = np.zeros_like(time_points)
            excess_pore_pressure[earthquake_idx:] = (
                case['peak_excess_pore_pressure_ratio'] * base_pore_pressure *
                np.exp(-(time_points[earthquake_idx:] - earthquake_time) / 2)
            )
            
            total_pore_pressure = pore_pressure + excess_pore_pressure
            
            # Settlement measurements
            settlement = np.zeros_like(time_points)
            settlement[earthquake_idx:] = (
                case['settlement_mm'] * 
                (1 - np.exp(-(time_points[earthquake_idx:] - earthquake_time) / 5))
            )
            
            # Lateral displacement
            lateral_disp = np.zeros_like(time_points)
            lateral_disp[earthquake_idx:] = (
                case['lateral_displacement_m'] * 1000 *  # convert to mm
                (1 - np.exp(-(time_points[earthquake_idx:] - earthquake_time) / 3))
            )
            
        else:  # clay
            # Monitoring duration (days before failure)
            monitoring_days = np.random.randint(60, 365)
            
            # Time points
            time_points = np.linspace(-monitoring_days, 0, monitoring_days + 1)
            
            # Gradual pore pressure increase
            base_pore_pressure = case['groundwater_depth_m'] * 9.81
            pore_pressure_increase = case['pore_pressure_ratio'] * base_pore_pressure
            
            # Sigmoid increase (gradual then accelerating)
            pore_pressure = (base_pore_pressure + 
                           pore_pressure_increase * 
                           (1 / (1 + np.exp(-0.1 * (time_points + monitoring_days/2)))))
            
            # Settlement/displacement evolution
            total_displacement = case['displacement_m'] * 1000  # convert to mm
            displacement = total_displacement * (1 / (1 + np.exp(-0.05 * (time_points + monitoring_days/3))))
            
            # Slope movement velocity
            velocity = np.gradient(displacement) / np.gradient(time_points)  # mm/day
            
            # Set lateral displacement and settlement for compatibility
            lateral_disp = displacement * 0.7  # assume 70% is lateral
            settlement = displacement * 0.3    # assume 30% is vertical
        
        # Create monitoring records
        for i, t in enumerate(time_points):
            monitoring_data.append({
                'case_id': case_id,
                'days_before_failure': -t,
                'pore_pressure_kPa': total_pore_pressure[i] if soil_type == 'sandy' else pore_pressure[i],
                'settlement_mm': settlement[i],
                'lateral_displacement_mm': lateral_disp[i],
                'velocity_mm_day': velocity[i] if soil_type == 'clay' else np.gradient(settlement)[i],
                'soil_type': soil_type
            })
    
    return pd.DataFrame(monitoring_data)

def main():
    """Generate all case study datasets"""
    
    print("Generating case study datasets...")
    
    # Generate sandy soil case studies
    sandy_cases = generate_sandy_soil_case_studies(50)
    
    # Generate clay soil case studies
    clay_cases = generate_clay_soil_case_studies(40)
    
    # Generate monitoring data
    sandy_monitoring = generate_monitoring_data(sandy_cases, 'sandy')
    clay_monitoring = generate_monitoring_data(clay_cases, 'clay')
    
    # Combine monitoring data
    all_monitoring = pd.concat([sandy_monitoring, clay_monitoring], ignore_index=True)
    
    # Save datasets
    os.makedirs('geotechnical_datasets/case_studies', exist_ok=True)
    
    sandy_cases.to_csv('geotechnical_datasets/case_studies/sandy_soil_failure_cases.csv', index=False)
    clay_cases.to_csv('geotechnical_datasets/case_studies/clay_soil_failure_cases.csv', index=False)
    all_monitoring.to_csv('geotechnical_datasets/case_studies/monitoring_data_time_series.csv', index=False)
    
    # Create summary statistics
    summary_stats = {
        'sandy_soil_cases': {
            'total_cases': len(sandy_cases),
            'failure_types': sandy_cases['failure_type'].value_counts().to_dict(),
            'average_damage_cost': float(sandy_cases['damage_cost_usd'].mean()),
            'total_damage_cost': float(sandy_cases['damage_cost_usd'].sum())
        },
        'clay_soil_cases': {
            'total_cases': len(clay_cases),
            'failure_types': clay_cases['failure_type'].value_counts().to_dict(),
            'average_damage_cost': float(clay_cases['damage_cost_usd'].mean()),
            'total_damage_cost': float(clay_cases['damage_cost_usd'].sum()),
            'total_casualties': int(clay_cases['casualties'].sum()),
            'total_evacuated': int(clay_cases['people_evacuated'].sum())
        },
        'monitoring_data': {
            'total_records': len(all_monitoring),
            'monitored_cases': len(all_monitoring['case_id'].unique()),
            'sandy_records': len(sandy_monitoring),
            'clay_records': len(clay_monitoring)
        }
    }
    
    with open('geotechnical_datasets/case_studies/case_study_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Generated {len(sandy_cases)} sandy soil failure cases")
    print(f"Generated {len(clay_cases)} clay soil failure cases")
    print(f"Generated {len(all_monitoring)} monitoring data records")
    print("Files created:")
    print("- sandy_soil_failure_cases.csv")
    print("- clay_soil_failure_cases.csv")
    print("- monitoring_data_time_series.csv")
    print("- case_study_summary.json")
    
    return sandy_cases, clay_cases, all_monitoring

if __name__ == "__main__":
    sandy_cases, clay_cases, monitoring_data = main()