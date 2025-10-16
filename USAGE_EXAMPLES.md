# Usage Examples for IoT Building Dataset Generator

This document provides comprehensive examples of how to use the IoT Building Dataset Generator for various applications in building digital twin development and retrofit optimization.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Custom Building Configuration](#custom-building-configuration)
3. [Large-Scale Dataset Generation](#large-scale-dataset-generation)
4. [Data Analysis Examples](#data-analysis-examples)
5. [Deep Reinforcement Learning Integration](#deep-reinforcement-learning-integration)
6. [Building Performance Analysis](#building-performance-analysis)
7. [Digital Twin Calibration](#digital-twin-calibration)

## Basic Usage

### Generate a One-Week Dataset

```python
from comprehensive_dataset_generator import ComprehensiveDatasetGenerator, BuildingConfig

# Basic configuration
config = BuildingConfig()

# Generate dataset
generator = ComprehensiveDatasetGenerator(config)
dataset = generator.generate_complete_dataset(
    start_date="2023-01-01 00:00:00",
    end_date="2023-01-07 23:45:00",
    freq="15T"
)

print(f"Generated {len(dataset['weather'])} data points")
```

### Command Line Usage

```bash
# Generate 1 year of data
python main.py --duration 365 --output ./annual_dataset

# Generate with custom configuration
python main.py --config my_building.json --duration 30 --output ./monthly_data

# Generate without visualizations (faster)
python main.py --duration 7 --no-visualizations --output ./weekly_data
```

## Custom Building Configuration

### Research Facility Configuration

```python
config = BuildingConfig(
    building_type="Research Facility",
    floor_area=12000.0,  # m²
    num_floors=4,
    num_zones=25,
    occupancy_capacity=300,
    
    # Location: MIT, Cambridge, MA
    latitude=42.3601,
    longitude=-71.0942,
    timezone="America/New_York",
    elevation=15.0,
    
    # Building envelope
    window_to_wall_ratio=0.35,
    building_orientation=15.0,  # 15° from north
    
    # HVAC system
    hvac_type="Dedicated Outdoor Air System (DOAS)",
    chiller_capacity=800.0,  # kW
    boiler_capacity=500.0,   # kW
    num_ahu=6,
    
    # Renewable energy
    has_solar_panels=True,
    solar_capacity=250.0,    # kW
    has_energy_storage=True,
    battery_capacity=500.0   # kWh
)
```

### Data Center Configuration

```python
config = BuildingConfig(
    building_type="Data Center",
    floor_area=3000.0,
    num_floors=1,
    num_zones=8,
    occupancy_capacity=50,
    
    # Location: Northern Virginia
    latitude=38.9072,
    longitude=-77.0369,
    timezone="America/New_York",
    
    # High cooling requirements
    chiller_capacity=2000.0,  # kW
    boiler_capacity=100.0,    # kW (minimal heating)
    
    # Specialized parameters
    server_load_density=500.0,  # W/m² (would need custom implementation)
    cooling_redundancy=2.0      # N+1 redundancy
)
```

## Large-Scale Dataset Generation

### Multi-Year Dataset with Batch Processing

```python
import pandas as pd
from datetime import datetime, timedelta

def generate_multi_year_dataset(start_year, num_years, config):
    """Generate multi-year dataset in annual batches."""
    
    all_datasets = []
    
    for year in range(start_year, start_year + num_years):
        print(f"Generating data for year {year}...")
        
        generator = ComprehensiveDatasetGenerator(config)
        
        # Generate one year at a time
        dataset = generator.generate_complete_dataset(
            start_date=f"{year}-01-01 00:00:00",
            end_date=f"{year+1}-01-01 00:00:00",
            freq="15T"
        )
        
        # Export immediately to save memory
        exporter = DataExporter(f"./data_{year}")
        exporter.export_all_formats(dataset, generator.metadata)
        
        all_datasets.append(dataset)
    
    return all_datasets

# Generate 5 years of data
config = BuildingConfig()
datasets = generate_multi_year_dataset(2019, 5, config)
```

### Parallel Generation for Multiple Buildings

```python
from multiprocessing import Pool
import json

def generate_building_dataset(building_config):
    """Generate dataset for a single building."""
    config_dict, building_id = building_config
    
    config = BuildingConfig(**config_dict)
    generator = ComprehensiveDatasetGenerator(config)
    
    dataset = generator.generate_complete_dataset(
        start_date="2023-01-01 00:00:00",
        end_date="2023-12-31 23:45:00"
    )
    
    # Export
    exporter = DataExporter(f"./building_{building_id}")
    exporter.export_all_formats(dataset, generator.metadata)
    
    return building_id

# Define multiple building configurations
building_configs = [
    ({"building_type": "Office", "floor_area": 5000.0}, "office_1"),
    ({"building_type": "Retail", "floor_area": 3000.0}, "retail_1"),
    ({"building_type": "School", "floor_area": 8000.0}, "school_1"),
]

# Generate in parallel
with Pool(processes=3) as pool:
    results = pool.map(generate_building_dataset, building_configs)

print(f"Generated datasets for buildings: {results}")
```

## Data Analysis Examples

### Energy Performance Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_energy_performance(dataset):
    """Comprehensive energy performance analysis."""
    
    energy_df = dataset['energy']
    weather_df = dataset['weather']
    
    # Merge weather and energy data
    analysis_df = energy_df.merge(weather_df[['timestamp', 'ambient_temperature_c']], 
                                 on='timestamp')
    
    # Calculate key metrics
    analysis_df['total_consumption'] = (analysis_df['total_electricity_kw'] + 
                                      analysis_df['total_gas_kw'])
    
    analysis_df['hvac_fraction'] = ((analysis_df['hvac_cooling_kw'] + 
                                   analysis_df['hvac_heating_kw']) / 
                                  analysis_df['total_consumption'])
    
    # Daily profiles
    analysis_df['hour'] = analysis_df['timestamp'].dt.hour
    daily_profile = analysis_df.groupby('hour').agg({
        'total_consumption': 'mean',
        'hvac_cooling_kw': 'mean',
        'hvac_heating_kw': 'mean'
    })
    
    # Temperature correlation
    temp_corr = analysis_df['ambient_temperature_c'].corr(
        analysis_df['hvac_cooling_kw']
    )
    
    print(f"Temperature-Cooling Correlation: {temp_corr:.3f}")
    
    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily profile
    daily_profile.plot(ax=axes[0,0])
    axes[0,0].set_title('Average Daily Energy Profile')
    axes[0,0].set_xlabel('Hour of Day')
    axes[0,0].set_ylabel('Power (kW)')
    
    # Temperature vs cooling
    axes[0,1].scatter(analysis_df['ambient_temperature_c'], 
                     analysis_df['hvac_cooling_kw'], alpha=0.6)
    axes[0,1].set_xlabel('Outdoor Temperature (°C)')
    axes[0,1].set_ylabel('Cooling Load (kW)')
    axes[0,1].set_title('Cooling Load vs Temperature')
    
    # HVAC fraction over time
    axes[1,0].plot(analysis_df['timestamp'], analysis_df['hvac_fraction'])
    axes[1,0].set_title('HVAC Fraction of Total Energy')
    axes[1,0].set_ylabel('HVAC Fraction')
    
    # Energy distribution
    energy_cols = ['hvac_cooling_kw', 'hvac_heating_kw', 'lighting_zone_1_kw']
    analysis_df[energy_cols].hist(ax=axes[1,1], bins=30)
    axes[1,1].set_title('Energy Distribution')
    
    plt.tight_layout()
    plt.savefig('energy_analysis.png', dpi=300, bbox_inches='tight')
    
    return analysis_df

# Run analysis
dataset = generator.generate_complete_dataset(...)
analysis_results = analyze_energy_performance(dataset)
```

### Indoor Air Quality Assessment

```python
def analyze_air_quality(dataset):
    """Analyze indoor air quality patterns."""
    
    ieq_df = dataset['ieq']
    occupancy_df = dataset['occupancy']
    
    # Merge IEQ and occupancy data
    analysis_df = ieq_df.merge(occupancy_df[['timestamp', 'total_occupancy']], 
                              on='timestamp')
    
    # CO2 analysis
    co2_cols = [col for col in ieq_df.columns if 'co2_ppm' in col]
    analysis_df['avg_co2'] = ieq_df[co2_cols].mean(axis=1)
    
    # Comfort assessment
    analysis_df['temp_comfort'] = analysis_df['building_avg_temp_c'].apply(
        lambda x: 1 if 20 <= x <= 26 else 0
    )
    
    analysis_df['co2_acceptable'] = analysis_df['avg_co2'].apply(
        lambda x: 1 if x <= 1000 else 0
    )
    
    # Calculate comfort metrics
    temp_comfort_pct = analysis_df['temp_comfort'].mean() * 100
    co2_acceptable_pct = analysis_df['co2_acceptable'].mean() * 100
    
    print(f"Temperature Comfort: {temp_comfort_pct:.1f}% of time")
    print(f"CO2 Acceptable: {co2_acceptable_pct:.1f}% of time")
    
    # Occupancy-CO2 correlation
    occ_co2_corr = analysis_df['total_occupancy'].corr(analysis_df['avg_co2'])
    print(f"Occupancy-CO2 Correlation: {occ_co2_corr:.3f}")
    
    return analysis_df

# Run air quality analysis
air_quality_results = analyze_air_quality(dataset)
```

## Deep Reinforcement Learning Integration

### State Space Definition

```python
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BuildingEnvironment:
    """Building environment for RL training."""
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_step = 0
        self.scaler = MinMaxScaler()
        
        # Define state space
        self.state_features = [
            # Weather
            'ambient_temperature_c',
            'relative_humidity_pct',
            'global_horizontal_irradiance_w_m2',
            'wind_speed_m_s',
            
            # Occupancy
            'total_occupancy',
            'building_utilization_pct',
            
            # IEQ
            'building_avg_temp_c',
            'building_avg_co2_ppm',
            'building_avg_rh_pct',
            
            # Energy
            'total_electricity_kw',
            'hvac_cooling_kw',
            'hvac_heating_kw',
            
            # Systems
            'building_cooling_setpoint_c',
            'building_heating_setpoint_c',
            'chiller_capacity_pct'
        ]
        
        # Prepare state data
        self._prepare_state_data()
    
    def _prepare_state_data(self):
        """Prepare normalized state data."""
        # Combine all relevant data
        combined_df = self.dataset['weather'].merge(
            self.dataset['occupancy'], on='timestamp'
        ).merge(
            self.dataset['ieq'], on='timestamp'
        ).merge(
            self.dataset['energy'], on='timestamp'
        ).merge(
            self.dataset['systems'], on='timestamp'
        )
        
        # Extract state features
        self.state_data = combined_df[self.state_features].fillna(0)
        
        # Normalize
        self.normalized_states = self.scaler.fit_transform(self.state_data)
    
    def get_state(self):
        """Get current state."""
        if self.current_step >= len(self.normalized_states):
            return None
        return self.normalized_states[self.current_step]
    
    def step(self, action):
        """Execute action and return next state, reward, done."""
        # Action: [cooling_setpoint_delta, heating_setpoint_delta, ventilation_rate]
        
        current_state = self.get_state()
        if current_state is None:
            return None, 0, True, {}
        
        # Calculate reward (multi-objective)
        energy_reward = -self.state_data.iloc[self.current_step]['total_electricity_kw'] / 1000
        comfort_reward = self._calculate_comfort_reward()
        
        total_reward = 0.6 * energy_reward + 0.4 * comfort_reward
        
        self.current_step += 1
        next_state = self.get_state()
        done = next_state is None
        
        return next_state, total_reward, done, {}
    
    def _calculate_comfort_reward(self):
        """Calculate comfort reward based on temperature and CO2."""
        temp = self.state_data.iloc[self.current_step]['building_avg_temp_c']
        co2 = self.state_data.iloc[self.current_step]['building_avg_co2_ppm']
        
        # Temperature comfort (22°C ± 2°C)
        temp_reward = 1 - min(abs(temp - 22) / 2, 1)
        
        # CO2 comfort (< 1000 ppm)
        co2_reward = 1 if co2 <= 1000 else max(0, 1 - (co2 - 1000) / 500)
        
        return (temp_reward + co2_reward) / 2

# Usage with RL framework (e.g., Stable Baselines3)
env = BuildingEnvironment(dataset)

# Example training loop structure
for episode in range(1000):
    state = env.get_state()
    total_reward = 0
    
    while state is not None:
        # RL agent selects action
        action = agent.predict(state)
        
        # Environment step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Train agent
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        if done:
            break
    
    print(f"Episode {episode}: Total Reward = {total_reward}")
```

### Multi-Objective Optimization

```python
def multi_objective_analysis(dataset):
    """Analyze trade-offs between energy, comfort, and cost."""
    
    # Combine datasets
    combined_df = dataset['energy'].merge(dataset['ieq'], on='timestamp')
    
    # Define objectives
    combined_df['energy_cost'] = (
        combined_df['total_electricity_kw'] * 0.12 +  # $/kWh
        combined_df['total_gas_kw'] * 0.08             # $/kWh equivalent
    )
    
    combined_df['comfort_score'] = calculate_comfort_score(combined_df)
    
    # Pareto frontier analysis
    pareto_points = find_pareto_frontier(
        combined_df[['energy_cost', 'comfort_score']].values
    )
    
    return pareto_points

def calculate_comfort_score(df):
    """Calculate overall comfort score."""
    # Temperature comfort
    temp_comfort = df['building_avg_temp_c'].apply(
        lambda x: max(0, 1 - abs(x - 22) / 5)
    )
    
    # CO2 comfort
    co2_comfort = df['building_avg_co2_ppm'].apply(
        lambda x: max(0, 1 - max(0, x - 1000) / 500)
    )
    
    return (temp_comfort + co2_comfort) / 2
```

## Building Performance Analysis

### Baseline vs Retrofit Scenarios

```python
def compare_retrofit_scenarios(baseline_dataset, retrofit_dataset):
    """Compare baseline vs retrofit performance."""
    
    baseline_energy = baseline_dataset['energy']['total_electricity_kw'].sum()
    retrofit_energy = retrofit_dataset['energy']['total_electricity_kw'].sum()
    
    energy_savings = (baseline_energy - retrofit_energy) / baseline_energy * 100
    
    # Comfort comparison
    baseline_comfort = calculate_comfort_score(baseline_dataset['ieq'])
    retrofit_comfort = calculate_comfort_score(retrofit_dataset['ieq'])
    
    comfort_improvement = (retrofit_comfort.mean() - baseline_comfort.mean()) * 100
    
    print(f"Energy Savings: {energy_savings:.1f}%")
    print(f"Comfort Improvement: {comfort_improvement:.1f}%")
    
    return {
        'energy_savings_pct': energy_savings,
        'comfort_improvement_pct': comfort_improvement
    }
```

### Fault Detection and Diagnostics

```python
def detect_system_faults(dataset):
    """Detect potential system faults in the data."""
    
    systems_df = dataset['systems']
    energy_df = dataset['energy']
    
    faults = []
    
    # Check for chiller efficiency issues
    if 'chiller_cop' in systems_df.columns:
        low_cop = systems_df['chiller_cop'] < 3.0
        if low_cop.sum() > len(systems_df) * 0.1:  # More than 10% of time
            faults.append("Low chiller efficiency detected")
    
    # Check for simultaneous heating and cooling
    simultaneous_hc = (
        (energy_df['hvac_heating_kw'] > 10) & 
        (energy_df['hvac_cooling_kw'] > 10)
    )
    if simultaneous_hc.sum() > 0:
        faults.append(f"Simultaneous heating/cooling: {simultaneous_hc.sum()} instances")
    
    # Check for excessive energy consumption
    energy_threshold = energy_df['total_electricity_kw'].quantile(0.95)
    excessive_energy = energy_df['total_electricity_kw'] > energy_threshold * 1.5
    if excessive_energy.sum() > 0:
        faults.append(f"Excessive energy consumption: {excessive_energy.sum()} instances")
    
    return faults

# Run fault detection
faults = detect_system_faults(dataset)
for fault in faults:
    print(f"⚠️ {fault}")
```

## Digital Twin Calibration

### Model Validation

```python
def validate_model_accuracy(generated_dataset, measured_data):
    """Validate generated data against measured building data."""
    
    from sklearn.metrics import mean_absolute_error, r2_score
    
    # Align timestamps
    generated_df = generated_dataset['energy']
    measured_df = measured_data  # Assumed to have same structure
    
    # Calculate validation metrics
    metrics = {}
    
    for column in ['total_electricity_kw', 'hvac_cooling_kw', 'hvac_heating_kw']:
        if column in both dataframes:
            mae = mean_absolute_error(measured_df[column], generated_df[column])
            r2 = r2_score(measured_df[column], generated_df[column])
            
            metrics[column] = {
                'MAE': mae,
                'R²': r2,
                'MAPE': np.mean(np.abs((measured_df[column] - generated_df[column]) / measured_df[column])) * 100
            }
    
    return metrics

# Example validation
validation_results = validate_model_accuracy(dataset, measured_building_data)
for param, metrics in validation_results.items():
    print(f"{param}: R² = {metrics['R²']:.3f}, MAPE = {metrics['MAPE']:.1f}%")
```

### Parameter Sensitivity Analysis

```python
def sensitivity_analysis(base_config, parameter_ranges):
    """Perform sensitivity analysis on building parameters."""
    
    results = []
    
    for param_name, param_range in parameter_ranges.items():
        for param_value in param_range:
            # Create modified configuration
            config = base_config.copy()
            setattr(config, param_name, param_value)
            
            # Generate dataset
            generator = ComprehensiveDatasetGenerator(config)
            dataset = generator.generate_complete_dataset(
                start_date="2023-01-01 00:00:00",
                end_date="2023-01-07 23:45:00"
            )
            
            # Calculate key metrics
            total_energy = dataset['energy']['total_electricity_kw'].sum()
            avg_comfort = calculate_comfort_score(dataset['ieq']).mean()
            
            results.append({
                'parameter': param_name,
                'value': param_value,
                'total_energy': total_energy,
                'avg_comfort': avg_comfort
            })
    
    return pd.DataFrame(results)

# Run sensitivity analysis
param_ranges = {
    'window_to_wall_ratio': [0.2, 0.3, 0.4, 0.5, 0.6],
    'chiller_capacity': [400, 500, 600, 700, 800],
    'solar_capacity': [50, 100, 150, 200, 250]
}

sensitivity_results = sensitivity_analysis(config, param_ranges)
print(sensitivity_results.groupby('parameter').agg({
    'total_energy': ['min', 'max', 'std'],
    'avg_comfort': ['min', 'max', 'std']
}))
```

These examples demonstrate the versatility and power of the IoT Building Dataset Generator for various applications in building science, energy optimization, and digital twin development. The generated datasets provide a solid foundation for developing and testing advanced building control strategies, retrofit optimization algorithms, and machine learning models for smart building applications.