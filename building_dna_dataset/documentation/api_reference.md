# Building DNA Dataset - API Reference

## Overview

This API reference provides detailed information for programmatic access to the Building DNA Dataset. The dataset is designed to be easily integrated into various applications, tools, and frameworks for building performance analysis, digital twin development, and machine learning applications.

## Data Access Patterns

### File-Based Access

The primary access method is through direct file system access to JSON files:

```python
import json
from pathlib import Path

# Load specific data file
def load_building_data(dataset_path, data_category):
    """
    Load specific category of building data
    
    Args:
        dataset_path (str): Path to dataset root directory
        data_category (str): Category of data to load
        
    Returns:
        dict: Loaded data structure
    """
    file_mapping = {
        'metadata': 'building_metadata.json',
        'geometry': 'geometric_data/bim_geometry.json',
        'floor_plans': 'geometric_data/floor_plans.json',
        'walls': 'construction_materials/wall_assemblies.json',
        'roof_floor': 'construction_materials/roof_floor_assemblies.json',
        'hvac': 'system_data/hvac_systems.json',
        'dhw': 'system_data/dhw_systems.json',
        'lighting': 'system_data/lighting_systems.json',
        'renewable': 'system_data/renewable_energy_systems.json',
        'lidar': 'environmental_data/lidar_aerial_data.json',
        'air_tightness': 'environmental_data/air_tightness_data.json'
    }
    
    if data_category not in file_mapping:
        raise ValueError(f"Unknown data category: {data_category}")
    
    file_path = Path(dataset_path) / file_mapping[data_category]
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
```

### REST API Interface (Optional)

For web applications, a REST API can be implemented:

```python
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)

DATASET_PATH = Path('/path/to/building_dna_dataset')

@app.route('/api/building/<building_id>')
def get_building_metadata(building_id):
    """Get building metadata"""
    try:
        with open(DATASET_PATH / 'building_metadata.json', 'r') as f:
            data = json.load(f)
        
        if data.get('building_id') == building_id:
            return jsonify(data)
        else:
            return jsonify({'error': 'Building not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/building/<building_id>/systems/<system_type>')
def get_building_systems(building_id, system_type):
    """Get building systems data"""
    system_files = {
        'hvac': 'system_data/hvac_systems.json',
        'dhw': 'system_data/dhw_systems.json',
        'lighting': 'system_data/lighting_systems.json',
        'renewable': 'system_data/renewable_energy_systems.json'
    }
    
    if system_type not in system_files:
        return jsonify({'error': 'Invalid system type'}), 400
    
    try:
        with open(DATASET_PATH / system_files[system_type], 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/building/<building_id>/performance')
def get_performance_metrics(building_id):
    """Calculate and return performance metrics"""
    try:
        # Load required data
        with open(DATASET_PATH / 'building_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        with open(DATASET_PATH / 'system_data/hvac_systems.json', 'r') as f:
            hvac_data = json.load(f)
        
        # Calculate metrics
        floor_area = metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0)
        
        total_hvac_energy = sum(
            s.get('performance_data', {}).get('annual_energy_consumption_kwh', 0)
            for s in hvac_data.get('hvac_systems', [])
        )
        
        eui = total_hvac_energy / floor_area if floor_area > 0 else 0
        
        return jsonify({
            'building_id': building_id,
            'floor_area_m2': floor_area,
            'annual_hvac_energy_kwh': total_hvac_energy,
            'hvac_eui_kwh_m2': eui,
            'timestamp': '2025-10-16T00:00:00Z'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Core Data Classes

### BuildingDNA Class

```python
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

class BuildingDNA:
    """
    Main class for accessing and manipulating building DNA data
    """
    
    def __init__(self, dataset_path: str):
        """
        Initialize BuildingDNA instance
        
        Args:
            dataset_path: Path to the building DNA dataset directory
        """
        self.dataset_path = Path(dataset_path)
        self._data_cache = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load building metadata"""
        metadata_path = self.dataset_path / 'building_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)
        else:
            raise FileNotFoundError("Building metadata not found")
    
    @property
    def building_id(self) -> str:
        """Get building identifier"""
        return self._metadata.get('building_id', '')
    
    @property
    def building_type(self) -> str:
        """Get building type"""
        return self._metadata.get('building_type', '')
    
    @property
    def floor_area_m2(self) -> float:
        """Get total floor area in square meters"""
        return self._metadata.get('building_characteristics', {}).get('total_floor_area_m2', 0.0)
    
    @property
    def climate_zone(self) -> str:
        """Get climate zone"""
        return self._metadata.get('location', {}).get('climate_zone', '')
    
    def get_systems(self, system_type: str) -> Dict[str, Any]:
        """
        Get building systems data
        
        Args:
            system_type: Type of system ('hvac', 'dhw', 'lighting', 'renewable')
            
        Returns:
            Dictionary containing system data
        """
        if system_type in self._data_cache:
            return self._data_cache[system_type]
        
        file_mapping = {
            'hvac': 'system_data/hvac_systems.json',
            'dhw': 'system_data/dhw_systems.json',
            'lighting': 'system_data/lighting_systems.json',
            'renewable': 'system_data/renewable_energy_systems.json'
        }
        
        if system_type not in file_mapping:
            raise ValueError(f"Unknown system type: {system_type}")
        
        file_path = self.dataset_path / file_mapping[system_type]
        
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._data_cache[system_type] = data
        return data
    
    def get_geometry(self) -> Dict[str, Any]:
        """Get building geometry data"""
        if 'geometry' in self._data_cache:
            return self._data_cache['geometry']
        
        geometry_path = self.dataset_path / 'geometric_data/bim_geometry.json'
        
        if not geometry_path.exists():
            return {}
        
        with open(geometry_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._data_cache['geometry'] = data
        return data
    
    def get_construction_data(self, component_type: str) -> Dict[str, Any]:
        """
        Get construction data
        
        Args:
            component_type: Type of component ('walls', 'roof_floor')
            
        Returns:
            Dictionary containing construction data
        """
        file_mapping = {
            'walls': 'construction_materials/wall_assemblies.json',
            'roof_floor': 'construction_materials/roof_floor_assemblies.json'
        }
        
        if component_type not in file_mapping:
            raise ValueError(f"Unknown component type: {component_type}")
        
        cache_key = f"construction_{component_type}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        file_path = self.dataset_path / file_mapping[component_type]
        
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._data_cache[cache_key] = data
        return data
    
    def get_environmental_data(self, data_type: str) -> Dict[str, Any]:
        """
        Get environmental data
        
        Args:
            data_type: Type of data ('lidar', 'air_tightness')
            
        Returns:
            Dictionary containing environmental data
        """
        file_mapping = {
            'lidar': 'environmental_data/lidar_aerial_data.json',
            'air_tightness': 'environmental_data/air_tightness_data.json'
        }
        
        if data_type not in file_mapping:
            raise ValueError(f"Unknown data type: {data_type}")
        
        cache_key = f"environmental_{data_type}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        file_path = self.dataset_path / file_mapping[data_type]
        
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._data_cache[cache_key] = data
        return data
    
    def calculate_eui(self) -> Dict[str, float]:
        """
        Calculate Energy Use Intensity
        
        Returns:
            Dictionary with EUI calculations
        """
        if self.floor_area_m2 == 0:
            return {'error': 'Floor area not available'}
        
        total_energy = 0
        
        # HVAC energy
        hvac_data = self.get_systems('hvac')
        for system in hvac_data.get('hvac_systems', []):
            performance = system.get('performance_data', {})
            total_energy += performance.get('annual_energy_consumption_kwh', 0)
        
        # Lighting energy
        lighting_data = self.get_systems('lighting')
        for system in lighting_data.get('lighting_systems', []):
            total_energy += system.get('annual_energy_consumption_kwh', 0)
        
        eui_kwh_m2 = total_energy / self.floor_area_m2
        eui_kbtu_sqft = eui_kwh_m2 * 0.316998  # Convert to imperial
        
        return {
            'total_energy_kwh': total_energy,
            'floor_area_m2': self.floor_area_m2,
            'eui_kwh_m2_year': eui_kwh_m2,
            'eui_kbtu_sqft_year': eui_kbtu_sqft
        }
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of all building systems"""
        summary = {}
        
        # HVAC systems
        hvac_data = self.get_systems('hvac')
        hvac_systems = hvac_data.get('hvac_systems', [])
        summary['hvac'] = {
            'count': len(hvac_systems),
            'total_cooling_capacity_kw': sum(
                s.get('capacity_data', {}).get('cooling_capacity_kw', 0) 
                for s in hvac_systems
            ),
            'total_heating_capacity_kw': sum(
                s.get('capacity_data', {}).get('heating_capacity_kw', 0) 
                for s in hvac_systems
            )
        }
        
        # Lighting systems
        lighting_data = self.get_systems('lighting')
        lighting_systems = lighting_data.get('lighting_systems', [])
        summary['lighting'] = {
            'count': len(lighting_systems),
            'total_power_w': sum(
                s.get('total_system_power', 0) 
                for s in lighting_systems
            )
        }
        
        # Renewable energy systems
        renewable_data = self.get_systems('renewable')
        renewable_systems = renewable_data.get('renewable_energy_systems', [])
        pv_systems = [s for s in renewable_systems if 'PV' in s.get('system_type', '')]
        
        summary['renewable'] = {
            'total_systems': len(renewable_systems),
            'pv_systems': len(pv_systems),
            'total_pv_capacity_kw': sum(
                s.get('system_capacity', {}).get('dc_capacity_kw', 0) 
                for s in pv_systems
            )
        }
        
        return summary
    
    def validate_data(self) -> Dict[str, Any]:
        """
        Run basic data validation
        
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'building_id': self.building_id,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        # Check required files exist
        required_files = [
            'building_metadata.json',
            'geometric_data/bim_geometry.json',
            'system_data/hvac_systems.json'
        ]
        
        for file_path in required_files:
            full_path = self.dataset_path / file_path
            if full_path.exists():
                validation_results['checks'].append(f"✓ {file_path} exists")
            else:
                validation_results['errors'].append(f"✗ {file_path} missing")
        
        # Check data consistency
        hvac_data = self.get_systems('hvac')
        hvac_systems = hvac_data.get('hvac_systems', [])
        
        if hvac_systems:
            total_capacity = sum(
                s.get('capacity_data', {}).get('cooling_capacity_kw', 0) 
                for s in hvac_systems
            )
            
            if self.floor_area_m2 > 0:
                capacity_per_m2 = (total_capacity * 1000) / self.floor_area_m2
                
                if capacity_per_m2 < 50:
                    validation_results['warnings'].append(
                        f"HVAC capacity may be low: {capacity_per_m2:.1f} W/m²"
                    )
                elif capacity_per_m2 > 200:
                    validation_results['warnings'].append(
                        f"HVAC capacity may be high: {capacity_per_m2:.1f} W/m²"
                    )
                else:
                    validation_results['checks'].append(
                        f"✓ HVAC capacity reasonable: {capacity_per_m2:.1f} W/m²"
                    )
        
        return validation_results
    
    def export_summary(self, format: str = 'json') -> str:
        """
        Export building summary
        
        Args:
            format: Export format ('json', 'csv', 'yaml')
            
        Returns:
            Formatted string with building summary
        """
        summary = {
            'building_info': {
                'building_id': self.building_id,
                'building_type': self.building_type,
                'floor_area_m2': self.floor_area_m2,
                'climate_zone': self.climate_zone
            },
            'systems_summary': self.get_system_summary(),
            'energy_performance': self.calculate_eui(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format.lower() == 'json':
            return json.dumps(summary, indent=2)
        elif format.lower() == 'yaml':
            import yaml
            return yaml.dump(summary, default_flow_style=False)
        elif format.lower() == 'csv':
            # Flatten for CSV export
            import pandas as pd
            flattened = {}
            
            def flatten_dict(d, parent_key='', sep='_'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            flattened = flatten_dict(summary)
            df = pd.DataFrame([flattened])
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
```

## Utility Functions

### Data Conversion Utilities

```python
def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """
    Convert between different units
    
    Args:
        value: Value to convert
        from_unit: Source unit
        to_unit: Target unit
        
    Returns:
        Converted value
    """
    conversions = {
        # Length
        ('m', 'ft'): 3.28084,
        ('ft', 'm'): 0.3048,
        ('mm', 'in'): 0.0393701,
        ('in', 'mm'): 25.4,
        
        # Area
        ('m2', 'ft2'): 10.7639,
        ('ft2', 'm2'): 0.092903,
        
        # Volume
        ('m3', 'ft3'): 35.3147,
        ('ft3', 'm3'): 0.0283168,
        ('l', 'gal'): 0.264172,
        ('gal', 'l'): 3.78541,
        
        # Temperature (requires special handling)
        # Energy
        ('kwh', 'btu'): 3412.14,
        ('btu', 'kwh'): 0.000293071,
        ('kwh', 'mj'): 3.6,
        ('mj', 'kwh'): 0.277778,
        
        # Power
        ('kw', 'btu_hr'): 3412.14,
        ('btu_hr', 'kw'): 0.000293071,
        ('ton', 'kw'): 3.51685,
        ('kw', 'ton'): 0.284345,
        
        # Pressure
        ('pa', 'psi'): 0.000145038,
        ('psi', 'pa'): 6894.76,
        ('kpa', 'psi'): 0.145038,
        ('psi', 'kpa'): 6.89476
    }
    
    # Handle temperature conversions separately
    if from_unit == 'c' and to_unit == 'f':
        return (value * 9/5) + 32
    elif from_unit == 'f' and to_unit == 'c':
        return (value - 32) * 5/9
    elif from_unit == 'c' and to_unit == 'k':
        return value + 273.15
    elif from_unit == 'k' and to_unit == 'c':
        return value - 273.15
    
    # Standard conversions
    conversion_key = (from_unit.lower(), to_unit.lower())
    if conversion_key in conversions:
        return value * conversions[conversion_key]
    else:
        raise ValueError(f"Conversion from {from_unit} to {to_unit} not supported")

def validate_data_types(data: Dict[str, Any], schema: Dict[str, type]) -> List[str]:
    """
    Validate data types against schema
    
    Args:
        data: Data to validate
        schema: Schema with expected types
        
    Returns:
        List of validation errors
    """
    errors = []
    
    for field, expected_type in schema.items():
        if field in data:
            if not isinstance(data[field], expected_type):
                errors.append(
                    f"Field '{field}' expected {expected_type.__name__}, "
                    f"got {type(data[field]).__name__}"
                )
        else:
            errors.append(f"Required field '{field}' missing")
    
    return errors

def calculate_weighted_average(values: List[float], weights: List[float]) -> float:
    """
    Calculate weighted average
    
    Args:
        values: List of values
        weights: List of weights
        
    Returns:
        Weighted average
    """
    if len(values) != len(weights):
        raise ValueError("Values and weights must have same length")
    
    if sum(weights) == 0:
        raise ValueError("Sum of weights cannot be zero")
    
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)
```

### Integration Helpers

```python
class EnergyPlusIntegration:
    """Helper class for EnergyPlus integration"""
    
    def __init__(self, building_dna: BuildingDNA):
        self.building_dna = building_dna
    
    def generate_idf_geometry(self) -> str:
        """Generate EnergyPlus IDF geometry section"""
        geometry = self.building_dna.get_geometry()
        
        idf_content = []
        
        # Building object
        idf_content.append(f"""
Building,
    {self.building_dna.building_id},    !- Name
    0.0,                                !- North Axis {{deg}}
    Suburbs,                           !- Terrain
    0.04,                              !- Loads Convergence Tolerance Value
    0.4,                               !- Temperature Convergence Tolerance Value {{deltaC}}
    FullInteriorAndExterior,           !- Solar Distribution
    25,                                !- Maximum Number of Warmup Days
    6;                                 !- Minimum Number of Warmup Days
        """)
        
        # Zones from interior spaces
        interior_spaces = geometry.get('interior_spaces', [])
        for space in interior_spaces:
            space_id = space.get('space_id', '')
            space_name = space.get('space_name', '')
            
            idf_content.append(f"""
Zone,
    {space_id},                        !- Name
    0.0,                               !- Direction of Relative North {{deg}}
    0.0,                               !- X Origin {{m}}
    0.0,                               !- Y Origin {{m}}
    0.0,                               !- Z Origin {{m}}
    1,                                 !- Type
    1,                                 !- Multiplier
    {space.get('ceiling_height_m', 3.0)}, !- Ceiling Height {{m}}
    {space.get('volume_m3', 0)};       !- Volume {{m3}}
            """)
        
        return '\n'.join(idf_content)
    
    def generate_idf_constructions(self) -> str:
        """Generate EnergyPlus construction definitions"""
        wall_data = self.building_dna.get_construction_data('walls')
        
        idf_content = []
        
        for assembly in wall_data.get('wall_assemblies', []):
            assembly_id = assembly.get('assembly_id', '')
            layers = assembly.get('layers', [])
            
            # Material definitions
            for layer in layers:
                layer_id = layer.get('layer_id', '')
                material_name = layer.get('material', '')
                
                idf_content.append(f"""
Material,
    {layer_id},                        !- Name
    MediumRough,                       !- Roughness
    {layer.get('thickness_m', 0.1)},   !- Thickness {{m}}
    {layer.get('thermal_conductivity_w_mk', 0.1)}, !- Conductivity {{W/m-K}}
    {layer.get('density_kg_m3', 1000)}, !- Density {{kg/m3}}
    {layer.get('specific_heat_j_kgk', 1000)}, !- Specific Heat {{J/kg-K}}
    0.9,                               !- Thermal Absorptance
    0.7,                               !- Solar Absorptance
    0.7;                               !- Visible Absorptance
                """)
            
            # Construction definition
            layer_names = [layer.get('layer_id', '') for layer in layers]
            
            idf_content.append(f"""
Construction,
    {assembly_id},                     !- Name
    {','.join(layer_names)};           !- Layer names
            """)
        
        return '\n'.join(idf_content)

class OpenStudioIntegration:
    """Helper class for OpenStudio integration"""
    
    def __init__(self, building_dna: BuildingDNA):
        self.building_dna = building_dna
    
    def generate_osm_model(self) -> Dict[str, Any]:
        """Generate OpenStudio Model (OSM) structure"""
        
        model = {
            'version': '3.4.0',
            'objects': []
        }
        
        # Building object
        model['objects'].append({
            'type': 'OS:Building',
            'handle': '{building-handle}',
            'name': self.building_dna.building_id,
            'building_type': self.building_dna.building_type,
            'north_axis': 0.0,
            'relocatable': False
        })
        
        # Thermal zones from geometry
        geometry = self.building_dna.get_geometry()
        for space in geometry.get('interior_spaces', []):
            model['objects'].append({
                'type': 'OS:ThermalZone',
                'handle': f"{{zone-{space.get('space_id')}-handle}}",
                'name': space.get('space_name', ''),
                'multiplier': 1,
                'ceiling_height': space.get('ceiling_height_m', 3.0),
                'volume': space.get('volume_m3', 0)
            })
        
        return model

class DigitalTwinIntegration:
    """Helper class for digital twin integration"""
    
    def __init__(self, building_dna: BuildingDNA):
        self.building_dna = building_dna
    
    def generate_sensor_config(self) -> Dict[str, Any]:
        """Generate sensor configuration for digital twin"""
        
        config = {
            'building_id': self.building_dna.building_id,
            'sensors': [],
            'data_streams': [],
            'analytics': []
        }
        
        # HVAC system sensors
        hvac_data = self.building_dna.get_systems('hvac')
        for system in hvac_data.get('hvac_systems', []):
            system_id = system.get('system_id', '')
            
            config['sensors'].append({
                'sensor_id': f"{system_id}_performance",
                'sensor_type': 'HVAC_Monitor',
                'location': system.get('system_name', ''),
                'parameters': [
                    'supply_air_temperature',
                    'return_air_temperature',
                    'power_consumption',
                    'efficiency'
                ],
                'sampling_rate': '1_minute',
                'communication': 'BACnet'
            })
        
        # Energy meters for renewable systems
        renewable_data = self.building_dna.get_systems('renewable')
        for system in renewable_data.get('renewable_energy_systems', []):
            if 'PV' in system.get('system_type', ''):
                config['sensors'].append({
                    'sensor_id': f"{system.get('system_id')}_production",
                    'sensor_type': 'Energy_Meter',
                    'location': 'Solar_Array',
                    'parameters': [
                        'dc_power',
                        'ac_power',
                        'energy_production',
                        'irradiance'
                    ],
                    'sampling_rate': '1_minute',
                    'communication': 'Modbus_TCP'
                })
        
        return config
    
    def generate_kpi_definitions(self) -> Dict[str, Any]:
        """Generate KPI definitions for monitoring"""
        
        kpis = {
            'energy_performance': {
                'real_time_eui': {
                    'description': 'Real-time Energy Use Intensity',
                    'unit': 'kWh/m²',
                    'calculation': 'total_power / floor_area',
                    'target_range': [50, 150],
                    'alert_thresholds': {'low': 30, 'high': 200}
                },
                'renewable_fraction': {
                    'description': 'Renewable energy fraction',
                    'unit': 'percent',
                    'calculation': 'renewable_production / total_consumption * 100',
                    'target_range': [20, 100],
                    'alert_thresholds': {'low': 10, 'high': None}
                }
            },
            'comfort_metrics': {
                'thermal_comfort': {
                    'description': 'Thermal comfort index',
                    'unit': 'PMV',
                    'calculation': 'pmv_calculation(temp, humidity, air_speed)',
                    'target_range': [-0.5, 0.5],
                    'alert_thresholds': {'low': -1.0, 'high': 1.0}
                }
            },
            'system_performance': {
                'hvac_efficiency': {
                    'description': 'Real-time HVAC efficiency',
                    'unit': 'COP',
                    'calculation': 'cooling_output / power_input',
                    'target_range': [3.0, 5.0],
                    'alert_thresholds': {'low': 2.0, 'high': None}
                }
            }
        }
        
        return kpis
```

## Error Handling

### Custom Exceptions

```python
class BuildingDNAError(Exception):
    """Base exception for Building DNA operations"""
    pass

class DataNotFoundError(BuildingDNAError):
    """Raised when required data is not found"""
    pass

class ValidationError(BuildingDNAError):
    """Raised when data validation fails"""
    pass

class ConversionError(BuildingDNAError):
    """Raised when unit conversion fails"""
    pass

# Usage example with error handling
try:
    building = BuildingDNA('/path/to/dataset')
    hvac_data = building.get_systems('hvac')
    eui = building.calculate_eui()
    
except FileNotFoundError as e:
    print(f"Dataset not found: {e}")
except DataNotFoundError as e:
    print(f"Required data missing: {e}")
except ValidationError as e:
    print(f"Data validation failed: {e}")
except BuildingDNAError as e:
    print(f"Building DNA error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Configuration

### Configuration File Format

```yaml
# building_dna_config.yaml
dataset:
  path: "/path/to/building_dna_dataset"
  cache_enabled: true
  cache_size_mb: 100
  validation_on_load: true

api:
  host: "localhost"
  port: 5000
  cors_enabled: true
  rate_limiting: true
  max_requests_per_minute: 100

integrations:
  energyplus:
    enabled: true
    idf_output_path: "/tmp/building.idf"
  
  openstudio:
    enabled: true
    osm_output_path: "/tmp/building.osm"
  
  digital_twin:
    enabled: true
    mqtt_broker: "localhost:1883"
    data_retention_days: 365

logging:
  level: "INFO"
  file: "/var/log/building_dna.log"
  max_size_mb: 10
  backup_count: 5
```

### Configuration Loading

```python
import yaml
from pathlib import Path

class BuildingDNAConfig:
    """Configuration management for Building DNA"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path.home() / '.building_dna_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'dataset': {
                'path': './building_dna_dataset',
                'cache_enabled': True,
                'cache_size_mb': 100,
                'validation_on_load': True
            },
            'api': {
                'host': 'localhost',
                'port': 5000,
                'cors_enabled': True
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def save_config(self):
        """Save current configuration to file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
```

This API reference provides comprehensive documentation for programmatic access to the Building DNA Dataset, including core classes, utility functions, integration helpers, error handling, and configuration management. The API is designed to be flexible and extensible for various building performance analysis applications.