#!/usr/bin/env python3
"""
Documentation Generator for Rubberized Concrete Dataset
Creates comprehensive documentation, metadata, and usage examples

Author: AI Assistant
Date: 2025-10-18
"""

import json
import os
from datetime import datetime
import pandas as pd

def create_comprehensive_documentation():
    """Create comprehensive documentation for the dataset"""
    
    output_dir = "rubberized_concrete_dataset"
    
    # Main README
    readme_content = """# Fire-Resistant Rubberized Concrete Dataset

## Overview

This comprehensive dataset supports the development and validation of thermo-mechanical models for fire-resistant structural elements utilizing high-performance rubberized concrete. The dataset includes both material property data for model calibration and experimental validation data for model verification.

## Dataset Description

**Topic**: Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete

**Generated**: {generation_date}

**Scope**: Temperature range 20-1000°C, Rubber content 0-20% by volume

## Dataset Structure

### 1. Material Properties (Model Input Data)

#### 1.1 Thermal Properties
- **File**: `thermal_properties.json`, `thermal_properties.csv`
- **Parameters**: 
  - Thermal conductivity (W/m·K)
  - Specific heat (J/kg·K) 
  - Density (kg/m³)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.2 Mechanical Properties
- **File**: `mechanical_properties.json`, `mechanical_properties.csv`
- **Parameters**:
  - Compressive strength (MPa)
  - Tensile strength (MPa)
  - Elastic modulus (GPa)
  - Poisson's ratio
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.3 Deformation Properties
- **File**: `deformation_properties.json`, `deformation_properties.csv`
- **Parameters**:
  - Coefficient of thermal expansion (1/K)
  - Transient thermal strain
  - Creep strain data (time-dependent)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

#### 1.4 Poro-mechanical Properties
- **File**: `poromechanical_properties.json`, `poromechanical_properties.csv`
- **Parameters**:
  - Porosity (temperature and damage dependent)
  - Permeability (m²)
  - Damage parameter (0-1)
- **Temperature Range**: 20-1000°C
- **Rubber Content**: 0, 5, 10, 15, 20% by volume

### 2. Validation Data (Experimental Measurements)

#### 2.1 Temperature Evolution Data
- **File**: `temperature_evolution_validation.json`, `temperature_evolution_validation.csv`
- **Description**: Thermocouple measurements during fire exposure
- **Specimen Types**: Small cube (100mm), Cylinder (φ100×200mm), Beam (100×100×400mm), Slab (500×500×100mm)
- **Fire Curves**: ISO834, ASTM E119, Hydrocarbon, Parametric
- **Measurement Points**: Surface, 25mm depth, center
- **Duration**: 0-4 hours

#### 2.2 Deformation/Strain History
- **File**: `deformation_strain_validation.json`, `deformation_strain_validation.csv`
- **Description**: Strain measurements under combined thermal-mechanical loading
- **Loading Scenarios**: 
  - Thermal only (free expansion)
  - Low load (20% of strength)
  - Medium load (40% of strength)
  - High load (60% of strength)
- **Strain Components**: Total, thermal, mechanical, creep, transient
- **Duration**: 0-3 hours

#### 2.3 Spalling and Failure Data
- **File**: `spalling_failure_validation.json`, `spalling_failure_summary.csv`, `spalling_failure_detailed.csv`
- **Description**: Spalling patterns and failure analysis
- **Test Conditions**:
  - Standard fire (ISO834, moderate load, 4.5% moisture)
  - Rapid heating (hydrocarbon curve, higher load, 6% moisture)
  - High load (ISO834, 60% load, 4% moisture)
  - High moisture (ISO834, moderate load, 8% moisture)
- **Measurements**: Spalling occurrence, depth progression, mass loss, failure time

## Key Features

### Realistic Material Behavior
- Temperature-dependent property degradation
- Rubber content effects on all properties
- Moisture-related phenomena (transient strain, spalling)
- Damage evolution with temperature

### Comprehensive Validation Data
- Multi-point temperature measurements
- Strain decomposition (thermal, mechanical, creep, transient)
- Spalling risk assessment
- Failure mode identification

### Multiple Data Formats
- JSON for hierarchical data structure
- CSV for tabular analysis
- Comprehensive metadata
- Visualization tools included

## Usage Examples

### Loading Thermal Properties
```python
import json
import pandas as pd

# Load JSON data
with open('thermal_properties.json', 'r') as f:
    thermal_data = json.load(f)

# Access data for 10% rubber content
rubber_10_data = thermal_data['rubber_10pct']
temperatures = rubber_10_data['temperature']
conductivity = rubber_10_data['thermal_conductivity']

# Or load CSV data
df = pd.read_csv('thermal_properties.csv')
rubber_10_df = df[df['rubber_content_pct'] == 10]
```

### Analyzing Validation Data
```python
# Load temperature evolution data
df_temp = pd.read_csv('temperature_evolution_validation.csv')

# Filter for specific conditions
iso834_data = df_temp[
    (df_temp['fire_curve'] == 'ISO834') & 
    (df_temp['specimen_type'] == 'small_cube') &
    (df_temp['rubber_content_pct'] == 10)
]

# Plot temperature evolution
import matplotlib.pyplot as plt
for location in ['surface', '25mm', 'center']:
    data = iso834_data[iso834_data['thermocouple_location'] == location]
    plt.plot(data['time_hours'], data['temperature_C'], label=location)
plt.legend()
plt.show()
```

### Spalling Analysis
```python
# Load spalling data
df_spall = pd.read_csv('spalling_failure_summary.csv')

# Analyze spalling occurrence by rubber content
spall_rate = df_spall.groupby('rubber_content_pct')['spalling_occurred'].mean()
print("Spalling occurrence rate by rubber content:")
print(spall_rate)
```

## Model Implementation Guidelines

### Material Property Functions
The dataset can be used to develop temperature-dependent material property functions:

```python
def thermal_conductivity(T, rubber_content):
    # Fit polynomial or exponential functions to data
    # Example: k(T) = k0 * (1 - a*rubber_content/100) * exp(-b*T)
    pass

def compressive_strength(T, rubber_content):
    # Implement strength degradation model
    # Consider both temperature and rubber content effects
    pass
```

### Validation Methodology
1. **Calibration**: Use material property data to determine model parameters
2. **Validation**: Compare model predictions with experimental validation data
3. **Verification**: Ensure model captures key phenomena (spalling, strain evolution)

## Data Quality and Limitations

### Strengths
- Comprehensive temperature range (20-1000°C)
- Multiple rubber content levels (0-20%)
- Realistic material behavior modeling
- Multiple validation scenarios
- Consistent data structure

### Limitations
- Synthetic data based on literature and engineering judgment
- Limited to specific rubber type and concrete matrix
- No consideration of aggregate type variations
- Simplified fire exposure conditions

## File Structure
```
rubberized_concrete_dataset/
├── thermal_properties.json
├── thermal_properties.csv
├── mechanical_properties.json
├── mechanical_properties.csv
├── deformation_properties.json
├── deformation_properties.csv
├── poromechanical_properties.json
├── poromechanical_properties.csv
├── temperature_evolution_validation.json
├── temperature_evolution_validation.csv
├── deformation_strain_validation.json
├── deformation_strain_validation.csv
├── spalling_failure_validation.json
├── spalling_failure_summary.csv
├── spalling_failure_detailed.csv
├── visualizations/
│   ├── thermal_properties.png
│   ├── mechanical_properties.png
│   ├── temperature_evolution.png
│   ├── strain_evolution.png
│   ├── spalling_analysis.png
│   └── summary_dashboard.png
├── metadata.json
├── README.md
├── data_dictionary.json
└── usage_examples.py
```

## Citation

If you use this dataset in your research, please cite:

```
Fire-Resistant Rubberized Concrete Dataset for Thermo-Mechanical Modeling
Generated: {generation_date}
Topic: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
```

## Contact and Support

This dataset was generated for research and educational purposes. For questions about the data structure or usage, refer to the included documentation and example scripts.

## Version History

- v1.0 ({generation_date}): Initial dataset release
  - Complete material property database
  - Comprehensive validation datasets
  - Visualization tools
  - Documentation and examples
""".format(generation_date=datetime.now().strftime("%Y-%m-%d"))

    # Data Dictionary
    data_dictionary = {
        "thermal_properties": {
            "description": "Temperature-dependent thermal properties",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Temperature"},
                "thermal_conductivity_W_m_K": {"type": "float", "unit": "W/m·K", "description": "Thermal conductivity"},
                "specific_heat_J_kg_K": {"type": "float", "unit": "J/kg·K", "description": "Specific heat capacity"},
                "density_kg_m3": {"type": "float", "unit": "kg/m³", "description": "Density"}
            }
        },
        "mechanical_properties": {
            "description": "Temperature-dependent mechanical properties",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Temperature"},
                "compressive_strength_MPa": {"type": "float", "unit": "MPa", "description": "Compressive strength"},
                "tensile_strength_MPa": {"type": "float", "unit": "MPa", "description": "Tensile strength"},
                "elastic_modulus_GPa": {"type": "float", "unit": "GPa", "description": "Elastic modulus"},
                "poisson_ratio": {"type": "float", "unit": "dimensionless", "description": "Poisson's ratio"}
            }
        },
        "deformation_properties": {
            "description": "Deformation-related properties",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Temperature"},
                "thermal_expansion_coeff_per_K": {"type": "float", "unit": "1/K", "description": "Coefficient of thermal expansion"},
                "transient_thermal_strain": {"type": "float", "unit": "dimensionless", "description": "Transient thermal strain"}
            }
        },
        "poromechanical_properties": {
            "description": "Porosity and permeability properties",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Temperature"},
                "porosity": {"type": "float", "unit": "dimensionless", "description": "Porosity (0-1)"},
                "permeability_m2": {"type": "float", "unit": "m²", "description": "Permeability"},
                "damage_parameter": {"type": "float", "unit": "dimensionless", "description": "Damage parameter (0-1)"}
            }
        },
        "temperature_evolution_validation": {
            "description": "Temperature evolution during fire exposure",
            "fields": {
                "specimen_type": {"type": "string", "description": "Type of test specimen"},
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "fire_curve": {"type": "string", "description": "Fire exposure curve"},
                "time_hours": {"type": "float", "unit": "hours", "description": "Time since start of test"},
                "time_minutes": {"type": "float", "unit": "minutes", "description": "Time since start of test"},
                "thermocouple_location": {"type": "string", "description": "Thermocouple position"},
                "depth_mm": {"type": "float", "unit": "mm", "description": "Depth from surface"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Measured temperature"},
                "furnace_temperature_C": {"type": "float", "unit": "°C", "description": "Furnace temperature"}
            }
        },
        "deformation_strain_validation": {
            "description": "Strain evolution under thermal-mechanical loading",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "loading_scenario": {"type": "string", "description": "Loading condition"},
                "mechanical_load_ratio": {"type": "float", "unit": "dimensionless", "description": "Load as fraction of strength"},
                "time_hours": {"type": "float", "unit": "hours", "description": "Time since start of test"},
                "temperature_C": {"type": "float", "unit": "°C", "description": "Temperature"},
                "total_strain": {"type": "float", "unit": "dimensionless", "description": "Total strain"},
                "thermal_strain": {"type": "float", "unit": "dimensionless", "description": "Thermal strain component"},
                "mechanical_strain": {"type": "float", "unit": "dimensionless", "description": "Mechanical strain component"},
                "creep_strain": {"type": "float", "unit": "dimensionless", "description": "Creep strain component"},
                "transient_strain": {"type": "float", "unit": "dimensionless", "description": "Transient strain component"},
                "displacement_mm": {"type": "float", "unit": "mm", "description": "Displacement over 100mm gauge length"}
            }
        },
        "spalling_failure_summary": {
            "description": "Summary of spalling and failure behavior",
            "fields": {
                "rubber_content_pct": {"type": "integer", "unit": "% by volume", "description": "Rubber content percentage"},
                "test_condition": {"type": "string", "description": "Test condition name"},
                "heating_rate": {"type": "string", "description": "Fire curve used"},
                "load_ratio": {"type": "float", "unit": "dimensionless", "description": "Load as fraction of strength"},
                "moisture_content_pct": {"type": "float", "unit": "%", "description": "Initial moisture content"},
                "spalling_occurred": {"type": "boolean", "description": "Whether spalling occurred"},
                "time_to_spalling_min": {"type": "float", "unit": "minutes", "description": "Time to first spalling"},
                "spalling_temperature_C": {"type": "float", "unit": "°C", "description": "Temperature at which spalling occurred"},
                "max_spalling_depth_mm": {"type": "float", "unit": "mm", "description": "Maximum spalling depth"},
                "failure_time_min": {"type": "float", "unit": "minutes", "description": "Time to structural failure"},
                "failure_mode": {"type": "string", "description": "Mode of failure"}
            }
        }
    }

    # Usage Examples
    usage_examples = '''#!/usr/bin/env python3
"""
Usage Examples for Rubberized Concrete Dataset
Demonstrates how to load, analyze, and visualize the dataset

Author: AI Assistant
Date: 2025-10-18
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def example_1_load_material_properties():
    """Example 1: Load and interpolate material properties"""
    print("Example 1: Loading Material Properties")
    
    # Load thermal properties
    with open('thermal_properties.json', 'r') as f:
        thermal_data = json.load(f)
    
    # Extract data for 10% rubber content
    rubber_10 = thermal_data['rubber_10pct']
    temperatures = np.array(rubber_10['temperature'])
    conductivity = np.array(rubber_10['thermal_conductivity'])
    
    # Create interpolation function
    k_interp = interpolate.interp1d(temperatures, conductivity, kind='cubic')
    
    # Get thermal conductivity at specific temperature
    T_query = 300  # °C
    k_at_300 = k_interp(T_query)
    print(f"Thermal conductivity at {T_query}°C: {k_at_300:.3f} W/m·K")
    
    return k_interp

def example_2_compare_rubber_effects():
    """Example 2: Compare effects of rubber content"""
    print("\\nExample 2: Comparing Rubber Content Effects")
    
    # Load mechanical properties CSV
    df = pd.read_csv('mechanical_properties.csv')
    
    # Compare compressive strength at 400°C for different rubber contents
    temp_target = 400
    df_400 = df[df['temperature_C'].round() == temp_target]
    
    print(f"Compressive strength at {temp_target}°C:")
    for rubber in sorted(df_400['rubber_content_pct'].unique()):
        strength = df_400[df_400['rubber_content_pct'] == rubber]['compressive_strength_MPa'].iloc[0]
        print(f"  {rubber}% rubber: {strength:.1f} MPa")

def example_3_analyze_temperature_evolution():
    """Example 3: Analyze temperature evolution validation data"""
    print("\\nExample 3: Temperature Evolution Analysis")
    
    # Load temperature evolution data
    df = pd.read_csv('temperature_evolution_validation.csv')
    
    # Filter for specific conditions
    conditions = (
        (df['specimen_type'] == 'small_cube') &
        (df['fire_curve'] == 'ISO834') &
        (df['rubber_content_pct'] == 10) &
        (df['thermocouple_location'] == 'center')
    )
    data = df[conditions]
    
    # Find time to reach 500°C at center
    temp_500_data = data[data['temperature_C'] >= 500]
    if not temp_500_data.empty:
        time_to_500 = temp_500_data['time_hours'].iloc[0]
        print(f"Time to reach 500°C at center (10% rubber): {time_to_500:.2f} hours")

def example_4_strain_decomposition():
    """Example 4: Analyze strain components"""
    print("\\nExample 4: Strain Component Analysis")
    
    # Load strain data
    df = pd.read_csv('deformation_strain_validation.csv')
    
    # Filter for medium load scenario with 15% rubber
    conditions = (
        (df['rubber_content_pct'] == 15) &
        (df['loading_scenario'] == 'medium_load')
    )
    data = df[conditions]
    
    # Find maximum strain components at end of test
    final_data = data.iloc[-1]
    print(f"Final strain components (15% rubber, medium load):")
    print(f"  Total strain: {final_data['total_strain']*1000:.2f} ×10⁻³")
    print(f"  Thermal strain: {final_data['thermal_strain']*1000:.2f} ×10⁻³")
    print(f"  Mechanical strain: {final_data['mechanical_strain']*1000:.2f} ×10⁻³")
    print(f"  Creep strain: {final_data['creep_strain']*1000:.2f} ×10⁻³")

def example_5_spalling_analysis():
    """Example 5: Spalling behavior analysis"""
    print("\\nExample 5: Spalling Analysis")
    
    # Load spalling summary data
    df = pd.read_csv('spalling_failure_summary.csv')
    
    # Calculate spalling occurrence rate by rubber content
    spall_rates = df.groupby('rubber_content_pct')['spalling_occurred'].mean()
    
    print("Spalling occurrence rates:")
    for rubber, rate in spall_rates.items():
        print(f"  {rubber}% rubber: {rate:.1%}")
    
    # Find conditions with highest spalling risk
    high_risk = df[df['spalling_occurred'] == True]
    if not high_risk.empty:
        worst_condition = high_risk.loc[high_risk['max_spalling_depth_mm'].idxmax()]
        print(f"\\nWorst spalling case:")
        print(f"  Rubber content: {worst_condition['rubber_content_pct']}%")
        print(f"  Test condition: {worst_condition['test_condition']}")
        print(f"  Max depth: {worst_condition['max_spalling_depth_mm']:.1f} mm")

def example_6_create_material_model():
    """Example 6: Create simple material property model"""
    print("\\nExample 6: Material Property Modeling")
    
    # Load thermal properties
    df = pd.read_csv('thermal_properties.csv')
    
    def thermal_conductivity_model(T, rubber_content):
        """
        Simple model for thermal conductivity
        k(T, rubber) = k0 * (1 - a*rubber/100) * exp(-b*T/1000)
        """
        k0 = 1.8  # Base conductivity
        a = 0.3   # Rubber reduction factor
        b = 0.8   # Temperature degradation factor
        
        return k0 * (1 - a * rubber_content / 100) * np.exp(-b * T / 1000)
    
    # Test the model
    T_test = 500  # °C
    rubber_test = 10  # %
    k_model = thermal_conductivity_model(T_test, rubber_test)
    
    # Compare with dataset
    data_point = df[(df['temperature_C'].round() == T_test) & 
                   (df['rubber_content_pct'] == rubber_test)]
    if not data_point.empty:
        k_data = data_point['thermal_conductivity_W_m_K'].iloc[0]
        error = abs(k_model - k_data) / k_data * 100
        print(f"Model validation at {T_test}°C, {rubber_test}% rubber:")
        print(f"  Model prediction: {k_model:.3f} W/m·K")
        print(f"  Dataset value: {k_data:.3f} W/m·K")
        print(f"  Error: {error:.1f}%")

def example_7_visualization():
    """Example 7: Create custom visualizations"""
    print("\\nExample 7: Custom Visualization")
    
    # Load mechanical properties
    df = pd.read_csv('mechanical_properties.csv')
    
    # Create strength degradation plot
    plt.figure(figsize=(10, 6))
    
    rubber_contents = [0, 10, 20]
    colors = ['red', 'blue', 'green']
    
    for i, rubber in enumerate(rubber_contents):
        data = df[df['rubber_content_pct'] == rubber]
        plt.plot(data['temperature_C'], data['compressive_strength_MPa'], 
                label=f'{rubber}% rubber', color=colors[i], linewidth=2)
    
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Compressive Strength (MPa)')
    plt.title('Compressive Strength Degradation with Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('custom_strength_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Custom plot saved as 'custom_strength_plot.png'")

if __name__ == "__main__":
    print("Rubberized Concrete Dataset - Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_1_load_material_properties()
    example_2_compare_rubber_effects()
    example_3_analyze_temperature_evolution()
    example_4_strain_decomposition()
    example_5_spalling_analysis()
    example_6_create_material_model()
    example_7_visualization()
    
    print("\\nAll examples completed successfully!")
'''

    # Metadata
    metadata = {
        "dataset_info": {
            "name": "Fire-Resistant Rubberized Concrete Dataset",
            "version": "1.0",
            "generation_date": datetime.now().isoformat(),
            "topic": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
            "description": "Comprehensive dataset for thermo-mechanical modeling of rubberized concrete under fire conditions"
        },
        "scope": {
            "temperature_range": {"min": 20, "max": 1000, "unit": "°C"},
            "rubber_content_range": {"min": 0, "max": 20, "unit": "% by volume"},
            "time_range": {"max": 4, "unit": "hours"},
            "specimen_types": ["small_cube", "cylinder", "beam", "slab"],
            "fire_curves": ["ISO834", "ASTM_E119", "hydrocarbon", "parametric"]
        },
        "data_categories": {
            "material_properties": {
                "thermal": ["thermal_conductivity", "specific_heat", "density"],
                "mechanical": ["compressive_strength", "tensile_strength", "elastic_modulus", "poisson_ratio"],
                "deformation": ["thermal_expansion_coefficient", "transient_strain", "creep_strain"],
                "poromechanical": ["porosity", "permeability", "damage_parameter"]
            },
            "validation_data": {
                "temperature_evolution": ["thermocouple_measurements", "fire_exposure"],
                "strain_evolution": ["thermal_mechanical_loading", "strain_components"],
                "spalling_failure": ["spalling_patterns", "failure_analysis"]
            }
        },
        "file_formats": {
            "json": "Hierarchical data structure with metadata",
            "csv": "Tabular format for analysis",
            "png": "Visualization plots"
        },
        "quality_metrics": {
            "data_points": "~50,000 total data points",
            "temperature_resolution": "~10°C intervals",
            "time_resolution": "1 minute intervals",
            "noise_level": "Realistic measurement uncertainty added"
        },
        "applications": [
            "Finite element model calibration",
            "Fire resistance prediction",
            "Material optimization",
            "Structural design verification",
            "Research and education"
        ]
    }

    # Write all documentation files
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(readme_content)
    
    with open(os.path.join(output_dir, "data_dictionary.json"), "w") as f:
        json.dump(data_dictionary, f, indent=2)
    
    with open(os.path.join(output_dir, "usage_examples.py"), "w") as f:
        f.write(usage_examples)
    
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Documentation created in {output_dir}/")
    print("Files created:")
    print("- README.md")
    print("- data_dictionary.json") 
    print("- usage_examples.py")
    print("- metadata.json")

if __name__ == "__main__":
    create_comprehensive_documentation()