# Building DNA Dataset - Usage Examples

## Overview

This document provides practical examples of how to use the Building DNA Dataset for various applications including digital twin development, energy modeling, retrofit optimization, and machine learning applications.

## Table of Contents

1. [Basic Data Loading](#basic-data-loading)
2. [Energy Analysis](#energy-analysis)
3. [Thermal Performance Calculations](#thermal-performance-calculations)
4. [HVAC System Analysis](#hvac-system-analysis)
5. [Solar Potential Assessment](#solar-potential-assessment)
6. [Digital Twin Integration](#digital-twin-integration)
7. [Machine Learning Applications](#machine-learning-applications)
8. [Retrofit Optimization](#retrofit-optimization)
9. [Data Validation and Quality Checks](#data-validation-and-quality-checks)

## Basic Data Loading

### Python Example: Loading and Exploring the Dataset

```python
import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

class BuildingDNALoader:
    """Utility class for loading and accessing building DNA data"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.data = {}
        self.load_all_data()
    
    def load_all_data(self):
        """Load all JSON files into memory"""
        json_files = {
            'metadata': 'building_metadata.json',
            'geometry': 'geometric_data/bim_geometry.json',
            'floor_plans': 'geometric_data/floor_plans.json',
            'wall_assemblies': 'construction_materials/wall_assemblies.json',
            'roof_floor': 'construction_materials/roof_floor_assemblies.json',
            'hvac': 'system_data/hvac_systems.json',
            'dhw': 'system_data/dhw_systems.json',
            'lighting': 'system_data/lighting_systems.json',
            'renewable': 'system_data/renewable_energy_systems.json',
            'lidar': 'environmental_data/lidar_aerial_data.json',
            'air_tightness': 'environmental_data/air_tightness_data.json'
        }
        
        for key, file_path in json_files.items():
            full_path = self.dataset_path / file_path
            if full_path.exists():
                with open(full_path, 'r', encoding='utf-8') as f:
                    self.data[key] = json.load(f)
                print(f"Loaded {key}: {full_path}")
            else:
                print(f"Warning: {file_path} not found")
    
    def get_building_summary(self):
        """Get basic building information"""
        metadata = self.data.get('metadata', {})
        characteristics = metadata.get('building_characteristics', {})
        location = metadata.get('location', {})
        
        return {
            'building_id': metadata.get('building_id'),
            'building_type': metadata.get('building_type'),
            'location': location.get('address'),
            'climate_zone': location.get('climate_zone'),
            'year_built': characteristics.get('year_built'),
            'total_area_m2': characteristics.get('total_floor_area_m2'),
            'number_of_floors': characteristics.get('number_of_floors'),
            'max_occupancy': characteristics.get('max_occupancy')
        }
    
    def get_system_summary(self):
        """Get summary of building systems"""
        systems = {}
        
        # HVAC systems
        hvac_data = self.data.get('hvac', {}).get('hvac_systems', [])
        systems['hvac'] = {
            'count': len(hvac_data),
            'total_cooling_capacity_kw': sum(s.get('capacity_data', {}).get('cooling_capacity_kw', 0) for s in hvac_data),
            'total_heating_capacity_kw': sum(s.get('capacity_data', {}).get('heating_capacity_kw', 0) for s in hvac_data)
        }
        
        # Lighting systems
        lighting_data = self.data.get('lighting', {}).get('lighting_systems', [])
        systems['lighting'] = {
            'count': len(lighting_data),
            'total_power_w': sum(s.get('total_system_power', 0) for s in lighting_data),
            'annual_consumption_kwh': sum(s.get('annual_energy_consumption_kwh', 0) for s in lighting_data)
        }
        
        # Renewable energy
        renewable_data = self.data.get('renewable', {}).get('renewable_energy_systems', [])
        pv_systems = [s for s in renewable_data if 'PV' in s.get('system_type', '')]
        if pv_systems:
            systems['solar_pv'] = {
                'count': len(pv_systems),
                'total_capacity_kw': sum(s.get('system_capacity', {}).get('dc_capacity_kw', 0) for s in pv_systems),
                'annual_production_kwh': sum(s.get('performance_data', {}).get('actual_annual_production_kwh', 0) for s in pv_systems)
            }
        
        return systems

# Usage example
loader = BuildingDNALoader('/path/to/building_dna_dataset')

# Get building summary
building_info = loader.get_building_summary()
print("Building Summary:")
for key, value in building_info.items():
    print(f"  {key}: {value}")

# Get systems summary
systems_info = loader.get_system_summary()
print("\nSystems Summary:")
for system, data in systems_info.items():
    print(f"  {system}: {data}")
```

### JavaScript Example: Web Application Integration

```javascript
class BuildingDNAClient {
    constructor(datasetUrl) {
        this.datasetUrl = datasetUrl;
        this.data = {};
    }
    
    async loadData() {
        const files = [
            'building_metadata.json',
            'geometric_data/bim_geometry.json',
            'system_data/hvac_systems.json',
            'system_data/renewable_energy_systems.json'
        ];
        
        for (const file of files) {
            try {
                const response = await fetch(`${this.datasetUrl}/${file}`);
                const data = await response.json();
                const key = file.split('/').pop().replace('.json', '');
                this.data[key] = data;
                console.log(`Loaded ${key}`);
            } catch (error) {
                console.error(`Error loading ${file}:`, error);
            }
        }
    }
    
    getBuildingOverview() {
        const metadata = this.data.building_metadata || {};
        const characteristics = metadata.building_characteristics || {};
        
        return {
            id: metadata.building_id,
            type: metadata.building_type,
            area: characteristics.total_floor_area_m2,
            floors: characteristics.number_of_floors,
            occupancy: characteristics.max_occupancy
        };
    }
    
    getEnergySystemsOverview() {
        const hvac = this.data.hvac_systems?.hvac_systems || [];
        const renewable = this.data.renewable_energy_systems?.renewable_energy_systems || [];
        
        return {
            hvac_count: hvac.length,
            total_cooling_capacity: hvac.reduce((sum, sys) => 
                sum + (sys.capacity_data?.cooling_capacity_kw || 0), 0),
            renewable_count: renewable.length,
            total_pv_capacity: renewable
                .filter(sys => sys.system_type?.includes('PV'))
                .reduce((sum, sys) => sum + (sys.system_capacity?.dc_capacity_kw || 0), 0)
        };
    }
}

// Usage
const client = new BuildingDNAClient('/api/building-data');
await client.loadData();

const overview = client.getBuildingOverview();
const systems = client.getEnergySystemsOverview();

console.log('Building Overview:', overview);
console.log('Energy Systems:', systems);
```

## Energy Analysis

### Calculate Building Energy Use Intensity (EUI)

```python
def calculate_eui(loader):
    """Calculate Energy Use Intensity for the building"""
    
    # Get building area
    metadata = loader.data.get('metadata', {})
    floor_area = metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0)
    
    if floor_area == 0:
        return None
    
    # Calculate total annual energy consumption
    total_energy_kwh = 0
    
    # HVAC energy
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    for system in hvac_systems:
        performance = system.get('performance_data', {})
        total_energy_kwh += performance.get('annual_energy_consumption_kwh', 0)
    
    # Lighting energy
    lighting_systems = loader.data.get('lighting', {}).get('lighting_systems', [])
    for system in lighting_systems:
        total_energy_kwh += system.get('annual_energy_consumption_kwh', 0)
    
    # DHW energy (convert from gas to kWh equivalent)
    dhw_systems = loader.data.get('dhw', {}).get('domestic_hot_water_systems', [])
    for system in dhw_systems:
        performance = system.get('performance_data', {})
        # Convert gas consumption to kWh (1 m³ natural gas ≈ 10.55 kWh)
        gas_m3 = performance.get('annual_gas_consumption_m3', 0)
        total_energy_kwh += gas_m3 * 10.55
    
    # Calculate EUI (kWh/m²/year)
    eui = total_energy_kwh / floor_area
    
    return {
        'total_energy_kwh': total_energy_kwh,
        'floor_area_m2': floor_area,
        'eui_kwh_m2_year': eui,
        'eui_kbtu_sqft_year': eui * 0.316998  # Convert to imperial units
    }

# Usage
loader = BuildingDNALoader('/path/to/dataset')
eui_results = calculate_eui(loader)
print(f"Building EUI: {eui_results['eui_kwh_m2_year']:.1f} kWh/m²/year")
print(f"Building EUI: {eui_results['eui_kbtu_sqft_year']:.1f} kBtu/ft²/year")
```

### Energy Breakdown Analysis

```python
def analyze_energy_breakdown(loader):
    """Analyze energy consumption by end use"""
    
    breakdown = {
        'hvac_cooling_kwh': 0,
        'hvac_heating_kwh': 0,
        'lighting_kwh': 0,
        'dhw_kwh': 0,
        'other_kwh': 0
    }
    
    # HVAC energy breakdown
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    for system in hvac_systems:
        performance = system.get('performance_data', {})
        total_hvac = performance.get('annual_energy_consumption_kwh', 0)
        
        # Estimate cooling vs heating split (simplified approach)
        cooling_fraction = 0.6  # Assume 60% cooling, 40% heating
        breakdown['hvac_cooling_kwh'] += total_hvac * cooling_fraction
        breakdown['hvac_heating_kwh'] += total_hvac * (1 - cooling_fraction)
    
    # Lighting energy
    lighting_systems = loader.data.get('lighting', {}).get('lighting_systems', [])
    for system in lighting_systems:
        breakdown['lighting_kwh'] += system.get('annual_energy_consumption_kwh', 0)
    
    # DHW energy
    dhw_systems = loader.data.get('dhw', {}).get('domestic_hot_water_systems', [])
    for system in dhw_systems:
        performance = system.get('performance_data', {})
        breakdown['dhw_kwh'] += performance.get('annual_water_heating_load_kwh', 0)
    
    # Calculate percentages
    total_energy = sum(breakdown.values())
    percentages = {k: (v / total_energy * 100) if total_energy > 0 else 0 
                  for k, v in breakdown.items()}
    
    return {
        'absolute_kwh': breakdown,
        'percentages': percentages,
        'total_kwh': total_energy
    }

# Visualization
import matplotlib.pyplot as plt

def plot_energy_breakdown(breakdown_data):
    """Create pie chart of energy breakdown"""
    
    labels = ['HVAC Cooling', 'HVAC Heating', 'Lighting', 'DHW', 'Other']
    sizes = [
        breakdown_data['percentages']['hvac_cooling_kwh'],
        breakdown_data['percentages']['hvac_heating_kwh'],
        breakdown_data['percentages']['lighting_kwh'],
        breakdown_data['percentages']['dhw_kwh'],
        breakdown_data['percentages']['other_kwh']
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    plt.figure(figsize=(10, 8))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Building Energy Consumption Breakdown')
    plt.axis('equal')
    plt.show()

# Usage
breakdown = analyze_energy_breakdown(loader)
plot_energy_breakdown(breakdown)
```

## Thermal Performance Calculations

### Wall Assembly Thermal Analysis

```python
def analyze_wall_thermal_performance(loader):
    """Analyze thermal performance of wall assemblies"""
    
    wall_data = loader.data.get('wall_assemblies', {}).get('wall_assemblies', [])
    
    analysis = []
    for assembly in wall_data:
        assembly_id = assembly.get('assembly_id')
        
        # Basic thermal properties
        r_value = assembly.get('total_r_value_m2k_w', 0)
        u_value = assembly.get('total_u_value_w_m2k', 0)
        thermal_mass = assembly.get('thermal_mass_kg_m2', 0)
        
        # Calculate heat loss potential
        # Assume 20°C temperature difference and 1 m² area
        heat_loss_w_m2 = u_value * 20  # W/m² for 20°C difference
        
        # Thermal mass classification
        if thermal_mass < 100:
            mass_class = 'Light'
        elif thermal_mass < 300:
            mass_class = 'Medium'
        else:
            mass_class = 'Heavy'
        
        # Performance rating
        if u_value < 0.3:
            performance = 'Excellent'
        elif u_value < 0.5:
            performance = 'Good'
        elif u_value < 0.8:
            performance = 'Fair'
        else:
            performance = 'Poor'
        
        analysis.append({
            'assembly_id': assembly_id,
            'assembly_name': assembly.get('assembly_name'),
            'r_value_m2k_w': r_value,
            'u_value_w_m2k': u_value,
            'thermal_mass_kg_m2': thermal_mass,
            'mass_classification': mass_class,
            'heat_loss_w_m2_20k': heat_loss_w_m2,
            'performance_rating': performance
        })
    
    return analysis

def calculate_building_envelope_performance(loader):
    """Calculate overall building envelope performance"""
    
    # Get wall assemblies
    wall_analysis = analyze_wall_thermal_performance(loader)
    
    # Get geometry data for wall areas
    geometry = loader.data.get('geometry', {}).get('building_envelope', {})
    exterior_walls = geometry.get('exterior_walls', [])
    
    # Calculate area-weighted average U-value
    total_area = 0
    weighted_u_sum = 0
    
    for wall in exterior_walls:
        wall_area = wall.get('area_m2', 0)
        wall_type = wall.get('wall_type', '')
        
        # Find matching assembly
        matching_assembly = None
        for assembly in wall_analysis:
            if wall_type.lower() in assembly['assembly_name'].lower():
                matching_assembly = assembly
                break
        
        if matching_assembly:
            u_value = matching_assembly['u_value_w_m2k']
            total_area += wall_area
            weighted_u_sum += u_value * wall_area
    
    # Calculate envelope performance
    if total_area > 0:
        avg_u_value = weighted_u_sum / total_area
        avg_r_value = 1 / avg_u_value if avg_u_value > 0 else 0
        
        return {
            'total_envelope_area_m2': total_area,
            'area_weighted_u_value': avg_u_value,
            'area_weighted_r_value': avg_r_value,
            'annual_heat_loss_potential_kwh': (avg_u_value * total_area * 20 * 8760) / 1000  # Simplified calculation
        }
    
    return None

# Usage
wall_performance = analyze_wall_thermal_performance(loader)
envelope_performance = calculate_building_envelope_performance(loader)

print("Wall Assembly Performance:")
for wall in wall_performance:
    print(f"  {wall['assembly_name']}: U={wall['u_value_w_m2k']:.3f} W/m²K, Rating: {wall['performance_rating']}")

if envelope_performance:
    print(f"\nOverall Envelope Performance:")
    print(f"  Area-weighted U-value: {envelope_performance['area_weighted_u_value']:.3f} W/m²K")
    print(f"  Area-weighted R-value: {envelope_performance['area_weighted_r_value']:.1f} m²K/W")
```

## HVAC System Analysis

### System Efficiency Analysis

```python
def analyze_hvac_efficiency(loader):
    """Analyze HVAC system efficiency and performance"""
    
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    
    analysis = []
    for system in hvac_systems:
        system_id = system.get('system_id')
        system_name = system.get('system_name')
        
        # Get capacity and efficiency data
        capacity_data = system.get('capacity_data', {})
        efficiency_data = system.get('efficiency_ratings', {})
        performance_data = system.get('performance_data', {})
        
        cooling_capacity = capacity_data.get('cooling_capacity_kw', 0)
        heating_capacity = capacity_data.get('heating_capacity_kw', 0)
        
        cooling_cop = efficiency_data.get('cooling_cop', 0)
        heating_cop = efficiency_data.get('heating_cop', 0)
        
        annual_energy = performance_data.get('annual_energy_consumption_kwh', 0)
        operating_hours = performance_data.get('operating_hours_annual', 0)
        
        # Calculate performance metrics
        if operating_hours > 0:
            avg_load_factor = annual_energy / (operating_hours * max(cooling_capacity, heating_capacity))
        else:
            avg_load_factor = 0
        
        # Efficiency rating
        if cooling_cop >= 4.0:
            efficiency_rating = 'Excellent'
        elif cooling_cop >= 3.0:
            efficiency_rating = 'Good'
        elif cooling_cop >= 2.5:
            efficiency_rating = 'Fair'
        else:
            efficiency_rating = 'Poor'
        
        analysis.append({
            'system_id': system_id,
            'system_name': system_name,
            'cooling_capacity_kw': cooling_capacity,
            'heating_capacity_kw': heating_capacity,
            'cooling_cop': cooling_cop,
            'heating_cop': heating_cop,
            'annual_energy_kwh': annual_energy,
            'operating_hours': operating_hours,
            'avg_load_factor': avg_load_factor,
            'efficiency_rating': efficiency_rating
        })
    
    return analysis

def calculate_hvac_sizing_adequacy(loader):
    """Check if HVAC systems are properly sized for the building"""
    
    # Get building characteristics
    metadata = loader.data.get('metadata', {})
    characteristics = metadata.get('building_characteristics', {})
    floor_area = characteristics.get('conditioned_floor_area_m2', 0)
    climate_zone = metadata.get('location', {}).get('climate_zone', '')
    
    # Get HVAC capacity
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    total_cooling_capacity = sum(s.get('capacity_data', {}).get('cooling_capacity_kw', 0) for s in hvac_systems)
    total_heating_capacity = sum(s.get('capacity_data', {}).get('heating_capacity_kw', 0) for s in hvac_systems)
    
    if floor_area > 0:
        # Calculate capacity per unit area
        cooling_w_m2 = (total_cooling_capacity * 1000) / floor_area
        heating_w_m2 = (total_heating_capacity * 1000) / floor_area
        
        # Typical ranges for office buildings (W/m²)
        typical_cooling_range = (80, 150)  # Varies by climate zone
        typical_heating_range = (60, 120)
        
        # Sizing assessment
        cooling_adequate = typical_cooling_range[0] <= cooling_w_m2 <= typical_cooling_range[1] * 1.2
        heating_adequate = typical_heating_range[0] <= heating_w_m2 <= typical_heating_range[1] * 1.2
        
        return {
            'floor_area_m2': floor_area,
            'total_cooling_capacity_kw': total_cooling_capacity,
            'total_heating_capacity_kw': total_heating_capacity,
            'cooling_w_m2': cooling_w_m2,
            'heating_w_m2': heating_w_m2,
            'cooling_adequate': cooling_adequate,
            'heating_adequate': heating_adequate,
            'typical_cooling_range': typical_cooling_range,
            'typical_heating_range': typical_heating_range
        }
    
    return None

# Usage
hvac_analysis = analyze_hvac_efficiency(loader)
sizing_analysis = calculate_hvac_sizing_adequacy(loader)

print("HVAC System Efficiency Analysis:")
for system in hvac_analysis:
    print(f"  {system['system_name']}:")
    print(f"    Cooling COP: {system['cooling_cop']:.2f}")
    print(f"    Efficiency Rating: {system['efficiency_rating']}")
    print(f"    Load Factor: {system['avg_load_factor']:.2f}")

if sizing_analysis:
    print(f"\nHVAC Sizing Analysis:")
    print(f"  Cooling: {sizing_analysis['cooling_w_m2']:.1f} W/m² ({'Adequate' if sizing_analysis['cooling_adequate'] else 'Check sizing'})")
    print(f"  Heating: {sizing_analysis['heating_w_m2']:.1f} W/m² ({'Adequate' if sizing_analysis['heating_adequate'] else 'Check sizing'})")
```

## Solar Potential Assessment

### Solar PV Analysis

```python
def analyze_solar_potential(loader):
    """Analyze solar PV potential and performance"""
    
    # Get existing solar systems
    renewable_data = loader.data.get('renewable', {}).get('renewable_energy_systems', [])
    pv_systems = [s for s in renewable_data if 'PV' in s.get('system_type', '')]
    
    # Get solar analysis data
    lidar_data = loader.data.get('lidar', {})
    solar_analysis = lidar_data.get('solar_exposure_analysis', {})
    roof_zones = solar_analysis.get('roof_zones', [])
    
    analysis = {
        'existing_systems': [],
        'roof_potential': [],
        'total_potential': {}
    }
    
    # Analyze existing PV systems
    for system in pv_systems:
        system_data = {
            'system_id': system.get('system_id'),
            'dc_capacity_kw': system.get('system_capacity', {}).get('dc_capacity_kw', 0),
            'annual_production_kwh': system.get('performance_data', {}).get('actual_annual_production_kwh', 0),
            'performance_ratio': system.get('performance_data', {}).get('performance_ratio', 0),
            'capacity_factor': system.get('performance_data', {}).get('capacity_factor_percent', 0),
            'co2_avoided_kg': system.get('environmental_impact', {}).get('annual_co2_avoided_kg', 0)
        }
        
        # Calculate specific yield
        if system_data['dc_capacity_kw'] > 0:
            system_data['specific_yield_kwh_kw'] = system_data['annual_production_kwh'] / system_data['dc_capacity_kw']
        
        analysis['existing_systems'].append(system_data)
    
    # Analyze roof potential
    total_roof_area = 0
    total_suitable_area = 0
    total_potential_kwh = 0
    
    for zone in roof_zones:
        zone_area = zone.get('area_m2', 0)
        annual_irradiation = zone.get('annual_irradiation_kwh_m2', 0)
        shading_factor = zone.get('shading_factor', 0)
        solar_potential = zone.get('solar_potential', '')
        
        # Calculate usable area (assume 70% of roof area is usable for PV)
        usable_area = zone_area * 0.7 * (1 - shading_factor)
        
        # Estimate PV potential (assume 200W/m² panel density, 18% system efficiency)
        potential_capacity_kw = usable_area * 0.2  # 200W/m²
        potential_production_kwh = potential_capacity_kw * annual_irradiation * 0.18 / 1000  # System efficiency
        
        zone_analysis = {
            'zone_id': zone.get('zone_id'),
            'zone_name': zone.get('zone_name'),
            'total_area_m2': zone_area,
            'usable_area_m2': usable_area,
            'annual_irradiation_kwh_m2': annual_irradiation,
            'shading_factor': shading_factor,
            'solar_potential': solar_potential,
            'potential_capacity_kw': potential_capacity_kw,
            'potential_production_kwh': potential_production_kwh
        }
        
        analysis['roof_potential'].append(zone_analysis)
        
        total_roof_area += zone_area
        total_suitable_area += usable_area
        total_potential_kwh += potential_production_kwh
    
    # Calculate total potential
    analysis['total_potential'] = {
        'total_roof_area_m2': total_roof_area,
        'total_suitable_area_m2': total_suitable_area,
        'total_potential_capacity_kw': sum(z['potential_capacity_kw'] for z in analysis['roof_potential']),
        'total_potential_production_kwh': total_potential_kwh,
        'roof_utilization_percent': (total_suitable_area / total_roof_area * 100) if total_roof_area > 0 else 0
    }
    
    return analysis

def calculate_solar_economics(solar_analysis, electricity_rate_kwh=0.12):
    """Calculate economic benefits of solar PV systems"""
    
    economics = {}
    
    # Existing systems economics
    for system in solar_analysis['existing_systems']:
        annual_production = system['annual_production_kwh']
        annual_savings = annual_production * electricity_rate_kwh
        
        system_economics = {
            'annual_energy_savings_usd': annual_savings,
            'co2_avoided_kg': system['co2_avoided_kg'],
            'co2_avoided_tons': system['co2_avoided_kg'] / 1000
        }
        
        economics[system['system_id']] = system_economics
    
    # Potential additional systems economics
    total_potential = solar_analysis['total_potential']
    potential_production = total_potential['total_potential_production_kwh']
    potential_savings = potential_production * electricity_rate_kwh
    
    # Estimate CO2 savings (assume 0.5 kg CO2/kWh grid electricity)
    potential_co2_savings = potential_production * 0.5
    
    economics['potential_additional'] = {
        'potential_capacity_kw': total_potential['total_potential_capacity_kw'],
        'potential_production_kwh': potential_production,
        'potential_annual_savings_usd': potential_savings,
        'potential_co2_savings_kg': potential_co2_savings,
        'potential_co2_savings_tons': potential_co2_savings / 1000
    }
    
    return economics

# Usage
solar_analysis = analyze_solar_potential(loader)
solar_economics = calculate_solar_economics(solar_analysis)

print("Solar PV Analysis:")
print(f"Existing Systems: {len(solar_analysis['existing_systems'])}")
for system in solar_analysis['existing_systems']:
    print(f"  {system['system_id']}: {system['dc_capacity_kw']:.1f} kW, {system['annual_production_kwh']:,.0f} kWh/year")

print(f"\nRoof Potential:")
print(f"  Total roof area: {solar_analysis['total_potential']['total_roof_area_m2']:.0f} m²")
print(f"  Suitable area: {solar_analysis['total_potential']['total_suitable_area_m2']:.0f} m²")
print(f"  Additional potential: {solar_analysis['total_potential']['total_potential_capacity_kw']:.1f} kW")
print(f"  Additional production: {solar_analysis['total_potential']['total_potential_production_kwh']:,.0f} kWh/year")

print(f"\nEconomic Potential:")
print(f"  Additional annual savings: ${solar_economics['potential_additional']['potential_annual_savings_usd']:,.0f}")
print(f"  Additional CO2 savings: {solar_economics['potential_additional']['potential_co2_savings_tons']:.1f} tons/year")
```

## Digital Twin Integration

### IoT Sensor Integration Points

```python
def identify_sensor_integration_points(loader):
    """Identify optimal locations for IoT sensors in the digital twin"""
    
    integration_points = {
        'hvac_sensors': [],
        'environmental_sensors': [],
        'energy_meters': [],
        'occupancy_sensors': [],
        'equipment_monitoring': []
    }
    
    # HVAC system sensor points
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    for system in hvac_systems:
        system_id = system.get('system_id')
        system_name = system.get('system_name')
        zones_served = system.get('serves_zones', [])
        
        # Temperature and humidity sensors
        for zone in zones_served:
            integration_points['hvac_sensors'].append({
                'sensor_type': 'Temperature_Humidity',
                'location': f'Zone_{zone}',
                'system_id': system_id,
                'data_points': ['temperature_c', 'humidity_percent'],
                'sampling_frequency': '1_minute',
                'communication': 'BACnet_IP'
            })
        
        # System performance monitoring
        integration_points['equipment_monitoring'].append({
            'sensor_type': 'HVAC_Performance',
            'location': f'Equipment_{system_id}',
            'system_id': system_id,
            'data_points': [
                'supply_air_temp_c',
                'return_air_temp_c',
                'supply_air_flow_m3s',
                'power_consumption_kw',
                'efficiency_cop'
            ],
            'sampling_frequency': '5_minutes',
            'communication': 'Modbus_TCP'
        })
    
    # Energy monitoring points
    renewable_systems = loader.data.get('renewable', {}).get('renewable_energy_systems', [])
    for system in renewable_systems:
        if 'PV' in system.get('system_type', ''):
            integration_points['energy_meters'].append({
                'sensor_type': 'Solar_Production_Meter',
                'location': f"PV_System_{system.get('system_id')}",
                'system_id': system.get('system_id'),
                'data_points': [
                    'dc_power_kw',
                    'ac_power_kw',
                    'energy_production_kwh',
                    'irradiance_w_m2',
                    'module_temperature_c'
                ],
                'sampling_frequency': '1_minute',
                'communication': 'Ethernet_Modbus'
            })
    
    # Occupancy sensors for spaces
    geometry = loader.data.get('geometry', {}).get('interior_spaces', [])
    for space in geometry:
        space_id = space.get('space_id')
        space_name = space.get('space_name')
        occupancy_type = space.get('occupancy_type')
        
        if occupancy_type in ['Office', 'Meeting', 'Lobby']:
            integration_points['occupancy_sensors'].append({
                'sensor_type': 'Occupancy_Counter',
                'location': space_name,
                'space_id': space_id,
                'data_points': [
                    'occupant_count',
                    'motion_detected',
                    'co2_ppm',
                    'light_level_lux'
                ],
                'sampling_frequency': '30_seconds',
                'communication': 'Wireless_LoRaWAN'
            })
    
    # Environmental monitoring
    integration_points['environmental_sensors'].extend([
        {
            'sensor_type': 'Weather_Station',
            'location': 'Roof_Top',
            'data_points': [
                'outdoor_temperature_c',
                'humidity_percent',
                'wind_speed_ms',
                'wind_direction_degrees',
                'solar_irradiance_w_m2',
                'precipitation_mm'
            ],
            'sampling_frequency': '1_minute',
            'communication': 'Cellular_4G'
        },
        {
            'sensor_type': 'Indoor_Air_Quality',
            'location': 'Representative_Spaces',
            'data_points': [
                'co2_ppm',
                'voc_ppb',
                'pm25_ug_m3',
                'pm10_ug_m3',
                'temperature_c',
                'humidity_percent'
            ],
            'sampling_frequency': '5_minutes',
            'communication': 'WiFi'
        }
    ])
    
    return integration_points

def generate_digital_twin_data_model(loader):
    """Generate data model for digital twin implementation"""
    
    # Get building metadata
    metadata = loader.data.get('metadata', {})
    building_id = metadata.get('building_id')
    
    data_model = {
        'building_id': building_id,
        'timestamp': '2025-10-16T00:00:00Z',
        'static_data': {
            'geometry': 'Reference to BIM model',
            'systems': 'Reference to system specifications',
            'materials': 'Reference to construction data'
        },
        'dynamic_data': {
            'real_time_sensors': {},
            'calculated_metrics': {},
            'predictions': {},
            'alerts': []
        },
        'data_streams': []
    }
    
    # Add sensor data streams
    sensor_points = identify_sensor_integration_points(loader)
    
    for category, sensors in sensor_points.items():
        for sensor in sensors:
            stream_id = f"{sensor['location']}_{sensor['sensor_type']}"
            
            data_model['data_streams'].append({
                'stream_id': stream_id,
                'sensor_type': sensor['sensor_type'],
                'location': sensor['location'],
                'data_points': sensor['data_points'],
                'sampling_frequency': sensor['sampling_frequency'],
                'communication_protocol': sensor['communication'],
                'data_format': 'JSON',
                'retention_period_days': 365,
                'real_time_processing': True,
                'historical_analysis': True
            })
    
    # Add calculated metrics
    data_model['dynamic_data']['calculated_metrics'] = {
        'energy_performance': {
            'real_time_eui_kwh_m2': 'Calculated from energy meters',
            'system_efficiency': 'Calculated from HVAC performance',
            'renewable_fraction': 'Solar production / total consumption'
        },
        'comfort_metrics': {
            'thermal_comfort_index': 'PMV/PPD calculation',
            'air_quality_index': 'Weighted IAQ score',
            'lighting_adequacy': 'Illuminance vs requirements'
        },
        'operational_metrics': {
            'occupancy_utilization': 'Actual vs design occupancy',
            'space_utilization': 'Active spaces / total spaces',
            'equipment_runtime': 'Operating hours tracking'
        }
    }
    
    return data_model

# Usage
sensor_points = identify_sensor_integration_points(loader)
digital_twin_model = generate_digital_twin_data_model(loader)

print("Digital Twin Integration Points:")
for category, sensors in sensor_points.items():
    print(f"  {category}: {len(sensors)} sensors")

print(f"\nTotal data streams: {len(digital_twin_model['data_streams'])}")
print("Sample data streams:")
for stream in digital_twin_model['data_streams'][:3]:
    print(f"  {stream['stream_id']}: {stream['data_points']}")
```

## Machine Learning Applications

### Feature Engineering for Building Performance Prediction

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def extract_ml_features(loader):
    """Extract features for machine learning models"""
    
    features = {}
    
    # Building characteristics features
    metadata = loader.data.get('metadata', {})
    characteristics = metadata.get('building_characteristics', {})
    location = metadata.get('location', {})
    
    features.update({
        'floor_area_m2': characteristics.get('total_floor_area_m2', 0),
        'number_of_floors': characteristics.get('number_of_floors', 0),
        'building_height_m': characteristics.get('building_height_m', 0),
        'year_built': characteristics.get('year_built', 0),
        'max_occupancy': characteristics.get('max_occupancy', 0),
        'elevation_m': location.get('elevation_m', 0),
        'orientation_degrees': location.get('orientation', 0)
    })
    
    # Thermal performance features
    wall_assemblies = loader.data.get('wall_assemblies', {}).get('wall_assemblies', [])
    if wall_assemblies:
        avg_u_value = np.mean([w.get('total_u_value_w_m2k', 0) for w in wall_assemblies])
        avg_thermal_mass = np.mean([w.get('thermal_mass_kg_m2', 0) for w in wall_assemblies])
        
        features.update({
            'avg_wall_u_value': avg_u_value,
            'avg_thermal_mass': avg_thermal_mass
        })
    
    # HVAC system features
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    if hvac_systems:
        total_cooling_capacity = sum(s.get('capacity_data', {}).get('cooling_capacity_kw', 0) for s in hvac_systems)
        total_heating_capacity = sum(s.get('capacity_data', {}).get('heating_capacity_kw', 0) for s in hvac_systems)
        avg_cooling_cop = np.mean([s.get('efficiency_ratings', {}).get('cooling_cop', 0) for s in hvac_systems if s.get('efficiency_ratings', {}).get('cooling_cop', 0) > 0])
        
        features.update({
            'total_cooling_capacity_kw': total_cooling_capacity,
            'total_heating_capacity_kw': total_heating_capacity,
            'cooling_capacity_w_m2': total_cooling_capacity * 1000 / max(features['floor_area_m2'], 1),
            'heating_capacity_w_m2': total_heating_capacity * 1000 / max(features['floor_area_m2'], 1),
            'avg_hvac_efficiency': avg_cooling_cop
        })
    
    # Lighting features
    lighting_systems = loader.data.get('lighting', {}).get('lighting_systems', [])
    if lighting_systems:
        total_lighting_power = sum(s.get('total_system_power', 0) for s in lighting_systems)
        
        features.update({
            'total_lighting_power_w': total_lighting_power,
            'lighting_power_density_w_m2': total_lighting_power / max(features['floor_area_m2'], 1)
        })
    
    # Renewable energy features
    renewable_systems = loader.data.get('renewable', {}).get('renewable_energy_systems', [])
    pv_systems = [s for s in renewable_systems if 'PV' in s.get('system_type', '')]
    
    if pv_systems:
        total_pv_capacity = sum(s.get('system_capacity', {}).get('dc_capacity_kw', 0) for s in pv_systems)
        features.update({
            'pv_capacity_kw': total_pv_capacity,
            'pv_capacity_w_m2': total_pv_capacity * 1000 / max(features['floor_area_m2'], 1),
            'has_renewable_energy': 1
        })
    else:
        features.update({
            'pv_capacity_kw': 0,
            'pv_capacity_w_m2': 0,
            'has_renewable_energy': 0
        })
    
    # Air tightness features
    air_tightness = loader.data.get('air_tightness', {}).get('blower_door_tests', [])
    if air_tightness:
        latest_test = air_tightness[0]  # Assume first is most recent
        test_results = latest_test.get('test_results', {})
        
        features.update({
            'air_changes_per_hour_50pa': test_results.get('air_changes_per_hour_50pa', 0),
            'air_leakage_rate_m3h': test_results.get('air_leakage_rate_50pa_m3h', 0)
        })
    
    return features

def create_energy_prediction_model(building_features_list, energy_targets):
    """Create machine learning model for energy prediction"""
    
    # Convert to DataFrame
    df = pd.DataFrame(building_features_list)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    # Prepare features and targets
    X = df.drop(['building_id'], axis=1, errors='ignore')
    y = np.array(energy_targets)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'scaler': scaler,
        'performance': {
            'mae': mae,
            'r2_score': r2,
            'rmse': np.sqrt(np.mean((y_test - y_pred) ** 2))
        },
        'feature_importance': feature_importance,
        'feature_names': list(X.columns)
    }

def predict_retrofit_impact(model_data, current_features, retrofit_scenarios):
    """Predict energy impact of retrofit scenarios"""
    
    model = model_data['model']
    scaler = model_data['scaler']
    
    predictions = {}
    
    # Baseline prediction
    baseline_features = pd.DataFrame([current_features])
    baseline_features = baseline_features.reindex(columns=model_data['feature_names'], fill_value=0)
    baseline_scaled = scaler.transform(baseline_features)
    baseline_prediction = model.predict(baseline_scaled)[0]
    
    predictions['baseline'] = {
        'predicted_eui_kwh_m2': baseline_prediction,
        'scenario': 'Current building'
    }
    
    # Retrofit scenario predictions
    for scenario_name, scenario_changes in retrofit_scenarios.items():
        # Apply changes to features
        modified_features = current_features.copy()
        modified_features.update(scenario_changes)
        
        # Predict
        scenario_df = pd.DataFrame([modified_features])
        scenario_df = scenario_df.reindex(columns=model_data['feature_names'], fill_value=0)
        scenario_scaled = scaler.transform(scenario_df)
        scenario_prediction = model.predict(scenario_scaled)[0]
        
        # Calculate savings
        energy_savings = baseline_prediction - scenario_prediction
        savings_percent = (energy_savings / baseline_prediction) * 100
        
        predictions[scenario_name] = {
            'predicted_eui_kwh_m2': scenario_prediction,
            'energy_savings_kwh_m2': energy_savings,
            'savings_percent': savings_percent,
            'scenario': scenario_name
        }
    
    return predictions

# Example usage (would require multiple buildings for training)
current_building_features = extract_ml_features(loader)

# Define retrofit scenarios
retrofit_scenarios = {
    'improved_envelope': {
        'avg_wall_u_value': 0.25,  # Improved insulation
        'air_changes_per_hour_50pa': 2.0  # Better air sealing
    },
    'hvac_upgrade': {
        'avg_hvac_efficiency': 4.5,  # High-efficiency equipment
        'cooling_capacity_w_m2': 100  # Right-sized equipment
    },
    'comprehensive_retrofit': {
        'avg_wall_u_value': 0.25,
        'air_changes_per_hour_50pa': 2.0,
        'avg_hvac_efficiency': 4.5,
        'lighting_power_density_w_m2': 6.0,  # LED upgrade
        'pv_capacity_w_m2': 50  # Add solar PV
    }
}

print("Building Features for ML:")
for key, value in current_building_features.items():
    print(f"  {key}: {value}")

print("\nRetrofit Scenarios Defined:")
for scenario, changes in retrofit_scenarios.items():
    print(f"  {scenario}: {len(changes)} parameter changes")
```

## Data Validation and Quality Checks

### Automated Quality Assurance

```python
def run_comprehensive_validation(dataset_path):
    """Run comprehensive validation and quality checks"""
    
    # Import validation tools
    import sys
    sys.path.append(str(Path(dataset_path) / 'validation_tools'))
    
    from data_validator import BuildingDNAValidator
    from data_quality_metrics import DataQualityMetrics
    
    # Run validation
    validator = BuildingDNAValidator(dataset_path)
    validation_results = validator.validate_dataset()
    
    # Calculate quality metrics
    quality_calculator = DataQualityMetrics(dataset_path)
    quality_metrics = quality_calculator.calculate_all_metrics()
    
    # Generate summary report
    report = {
        'validation_status': validation_results['overall_status'],
        'quality_score': quality_metrics['overall_quality_index']['quality_index'],
        'quality_grade': quality_metrics['overall_quality_index']['quality_grade'],
        'total_errors': len(validation_results['errors']),
        'total_warnings': len(validation_results['warnings']),
        'completeness_score': quality_metrics['completeness_metrics']['summary']['overall_completeness_score'],
        'consistency_score': quality_metrics['consistency_metrics']['summary']['overall_consistency_score'],
        'recommendations': validation_results['recommendations']
    }
    
    return report, validation_results, quality_metrics

def custom_data_checks(loader):
    """Run custom data quality checks specific to building data"""
    
    checks = {
        'passed': [],
        'warnings': [],
        'errors': []
    }
    
    # Check 1: Building area consistency
    metadata = loader.data.get('metadata', {})
    geometry = loader.data.get('geometry', {})
    
    metadata_area = metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0)
    
    if 'interior_spaces' in geometry:
        geometry_area = sum(space.get('area_m2', 0) for space in geometry['interior_spaces'])
        
        if abs(metadata_area - geometry_area) / max(metadata_area, 1) > 0.1:  # 10% tolerance
            checks['warnings'].append(f"Building area mismatch: metadata={metadata_area}m², geometry={geometry_area}m²")
        else:
            checks['passed'].append("Building area consistency check passed")
    
    # Check 2: HVAC capacity vs building size
    hvac_systems = loader.data.get('hvac', {}).get('hvac_systems', [])
    total_cooling = sum(s.get('capacity_data', {}).get('cooling_capacity_kw', 0) for s in hvac_systems)
    
    if metadata_area > 0:
        cooling_w_m2 = (total_cooling * 1000) / metadata_area
        
        if cooling_w_m2 < 50:
            checks['warnings'].append(f"HVAC cooling capacity may be undersized: {cooling_w_m2:.1f} W/m²")
        elif cooling_w_m2 > 200:
            checks['warnings'].append(f"HVAC cooling capacity may be oversized: {cooling_w_m2:.1f} W/m²")
        else:
            checks['passed'].append("HVAC capacity sizing check passed")
    
    # Check 3: Thermal properties validation
    wall_assemblies = loader.data.get('wall_assemblies', {}).get('wall_assemblies', [])
    
    for assembly in wall_assemblies:
        r_value = assembly.get('total_r_value_m2k_w', 0)
        u_value = assembly.get('total_u_value_w_m2k', 0)
        
        if r_value > 0 and u_value > 0:
            calculated_u = 1.0 / r_value
            if abs(calculated_u - u_value) / u_value > 0.1:  # 10% tolerance
                checks['errors'].append(f"R-value/U-value inconsistency in {assembly.get('assembly_id')}")
            else:
                checks['passed'].append(f"Thermal properties consistent for {assembly.get('assembly_id')}")
    
    # Check 4: Energy consumption reasonableness
    hvac_energy = sum(s.get('performance_data', {}).get('annual_energy_consumption_kwh', 0) for s in hvac_systems)
    lighting_systems = loader.data.get('lighting', {}).get('lighting_systems', [])
    lighting_energy = sum(s.get('annual_energy_consumption_kwh', 0) for s in lighting_systems)
    
    total_energy = hvac_energy + lighting_energy
    
    if metadata_area > 0:
        eui = total_energy / metadata_area
        
        # Typical office building EUI range: 100-300 kWh/m²/year
        if eui < 50:
            checks['warnings'].append(f"Energy consumption may be too low: {eui:.1f} kWh/m²/year")
        elif eui > 400:
            checks['warnings'].append(f"Energy consumption may be too high: {eui:.1f} kWh/m²/year")
        else:
            checks['passed'].append("Energy consumption within reasonable range")
    
    return checks

# Usage
dataset_path = '/path/to/building_dna_dataset'
loader = BuildingDNALoader(dataset_path)

# Run comprehensive validation
validation_report, detailed_validation, quality_metrics = run_comprehensive_validation(dataset_path)

print("Validation Summary:")
print(f"  Status: {validation_report['validation_status']}")
print(f"  Quality Score: {validation_report['quality_score']:.1f}%")
print(f"  Quality Grade: {validation_report['quality_grade']}")
print(f"  Errors: {validation_report['total_errors']}")
print(f"  Warnings: {validation_report['total_warnings']}")

# Run custom checks
custom_checks = custom_data_checks(loader)

print("\nCustom Data Checks:")
print(f"  Passed: {len(custom_checks['passed'])}")
print(f"  Warnings: {len(custom_checks['warnings'])}")
print(f"  Errors: {len(custom_checks['errors'])}")

if custom_checks['warnings']:
    print("\nWarnings:")
    for warning in custom_checks['warnings']:
        print(f"  - {warning}")

if custom_checks['errors']:
    print("\nErrors:")
    for error in custom_checks['errors']:
        print(f"  - {error}")
```

## Conclusion

These usage examples demonstrate the versatility and depth of the Building DNA Dataset for various applications in building performance analysis, digital twin development, and machine learning. The dataset's comprehensive structure and high data quality make it suitable for:

1. **Energy Analysis**: Detailed energy consumption breakdown and efficiency analysis
2. **Thermal Performance**: Building envelope analysis and thermal modeling
3. **System Optimization**: HVAC, lighting, and renewable energy system analysis
4. **Digital Twin Development**: IoT integration and real-time monitoring setup
5. **Machine Learning**: Feature engineering and predictive modeling
6. **Retrofit Planning**: Scenario analysis and impact prediction
7. **Quality Assurance**: Automated validation and consistency checking

The examples provided can be adapted and extended for specific use cases and research applications. The dataset's standardized format and comprehensive documentation make it easy to integrate with existing building performance tools and develop new analytical capabilities.