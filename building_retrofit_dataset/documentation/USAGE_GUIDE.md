# Building Retrofit Dataset - Usage Guide

## Quick Start

### 1. Explore the Dataset Structure
```bash
# Navigate to the dataset directory
cd building_retrofit_dataset

# View the main README
cat README.md

# Explore the data structure
ls -la processed_data/analysis_ready/
```

### 2. Load Data in Python
```python
import pandas as pd
import numpy as np

# Load the main analysis-ready datasets
building_master = pd.read_csv('processed_data/analysis_ready/building_master.csv')
energy_analysis = pd.read_csv('processed_data/analysis_ready/energy_analysis.csv')
retrofit_analysis = pd.read_csv('processed_data/analysis_ready/retrofit_analysis.csv')
ml_ready = pd.read_csv('processed_data/analysis_ready/ml_ready.csv')

# Load IoT sensor data for time series analysis
iot_energy = pd.read_csv('raw_data/iot_sensors/B001_energy.csv')
iot_environmental = pd.read_csv('raw_data/iot_sensors/B001_environmental.csv')
```

### 3. Basic Data Exploration
```python
# Dataset overview
print("Dataset Overview:")
print(f"Total buildings: {len(building_master)}")
print(f"Building types: {building_master['building_type'].value_counts()}")
print(f"Construction periods: {building_master['construction_period'].value_counts()}")

# Energy performance summary
print("\nEnergy Performance Summary:")
print(f"Average energy intensity: {energy_analysis['energy_intensity_kwh_m2'].mean():.2f} kWh/m²")
print(f"EU rating distribution: {energy_analysis['eu_rating'].value_counts()}")

# Retrofit analysis
print("\nRetrofit Analysis:")
retrofit_buildings = retrofit_analysis[retrofit_analysis['has_retrofit'] == True]
print(f"Buildings with retrofits: {len(retrofit_buildings)}")
print(f"Average energy savings: {retrofit_buildings['energy_savings_percent'].mean():.2f}%")
```

## Analysis Use Cases

### 1. Energy Performance Analysis

#### Building Energy Benchmarking
```python
# Compare energy performance across building types
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(data=energy_analysis, x='building_type', y='energy_intensity_kwh_m2')
plt.title('Energy Intensity by Building Type')
plt.ylabel('Energy Intensity (kWh/m²)')
plt.xticks(rotation=45)
plt.show()

# Correlation analysis
correlation_matrix = energy_analysis[['energy_intensity_kwh_m2', 'building_age_years', 
                                    'floor_area_m2', 'u_value_wall_w_m2k']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Energy Performance Correlations')
plt.show()
```

#### Historical Energy Trends
```python
# Load historical consumption data
historical = pd.read_csv('raw_data/energy_performance/historical_consumption.csv')

# Plot energy trends over time
plt.figure(figsize=(14, 8))
for building_type in historical['building_type'].unique():
    data = historical[historical['building_type'] == building_type]
    plt.plot(data['year'], data['energy_intensity_kwh_m2'], 
             marker='o', label=building_type)

plt.xlabel('Year')
plt.ylabel('Energy Intensity (kWh/m²)')
plt.title('Energy Intensity Trends by Building Type')
plt.legend()
plt.grid(True)
plt.show()
```

### 2. Retrofit Optimization Analysis

#### Retrofit Cost-Benefit Analysis
```python
# Analyze retrofit scenarios
retrofit_scenarios = pd.read_csv('raw_data/lca_data/retrofit_lca_scenarios.csv')

# Filter for comprehensive retrofits
comprehensive = retrofit_scenarios[retrofit_scenarios['retrofit_scenario'] == 'comprehensive_retrofit']

# Cost vs. savings analysis
plt.figure(figsize=(10, 6))
plt.scatter(comprehensive['retrofit_cost_eur_m2'], 
           comprehensive['energy_savings_percent'], 
           alpha=0.7)
plt.xlabel('Retrofit Cost (EUR/m²)')
plt.ylabel('Energy Savings (%)')
plt.title('Retrofit Cost vs. Energy Savings')
plt.grid(True)

# Add building ID labels
for i, row in comprehensive.iterrows():
    plt.annotate(row['building_id'], 
                (row['retrofit_cost_eur_m2'], row['energy_savings_percent']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
plt.show()
```

#### Payback Period Analysis
```python
# Analyze payback periods
payback_data = comprehensive[comprehensive['payback_period_years'].notna()]

plt.figure(figsize=(10, 6))
plt.hist(payback_data['payback_period_years'], bins=10, alpha=0.7, edgecolor='black')
plt.xlabel('Payback Period (years)')
plt.ylabel('Number of Buildings')
plt.title('Distribution of Retrofit Payback Periods')
plt.axvline(payback_data['payback_period_years'].mean(), color='red', 
           linestyle='--', label=f'Mean: {payback_data["payback_period_years"].mean():.1f} years')
plt.legend()
plt.grid(True)
plt.show()
```

### 3. Machine Learning Applications

#### Feature Engineering
```python
# Use the ML-ready dataset
ml_data = pd.read_csv('processed_data/analysis_ready/ml_ready.csv')

# Select features for modeling
feature_columns = ['building_age_years', 'floor_area_m2', 'num_floors', 
                  'u_value_wall_w_m2k', 'u_value_roof_w_m2k', 'u_value_window_w_m2k',
                  'air_tightness_ach50', 'thermal_mass_kj_m2k', 'quality_score']

X = ml_data[feature_columns]
y = ml_data['energy_intensity_kwh_m2']

# Handle missing values
X = X.fillna(X.median())

# Split data for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Energy Prediction Model
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Train a random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {np.sqrt(mse):.2f} kWh/m²")
print(f"R²: {r2:.3f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='importance', y='feature')
plt.title('Feature Importance for Energy Prediction')
plt.xlabel('Importance')
plt.show()
```

### 4. Time Series Analysis

#### IoT Sensor Data Analysis
```python
# Load IoT data
iot_data = pd.read_csv('raw_data/iot_sensors/B001_energy.csv')
iot_data['timestamp'] = pd.to_datetime(iot_data['timestamp'])

# Set timestamp as index
iot_data.set_index('timestamp', inplace=True)

# Plot energy consumption patterns
plt.figure(figsize=(15, 8))
plt.subplot(2, 1, 1)
plt.plot(iot_data.index, iot_data['total_consumption'])
plt.title('Total Energy Consumption Over Time')
plt.ylabel('Consumption (kWh)')

plt.subplot(2, 1, 2)
plt.plot(iot_data.index, iot_data['heating_consumption'], label='Heating')
plt.plot(iot_data.index, iot_data['cooling_consumption'], label='Cooling')
plt.plot(iot_data.index, iot_data['lighting_consumption'], label='Lighting')
plt.title('End-Use Energy Consumption')
plt.ylabel('Consumption (kWh)')
plt.legend()
plt.tight_layout()
plt.show()

# Seasonal analysis
iot_data['month'] = iot_data.index.month
monthly_consumption = iot_data.groupby('month')['total_consumption'].mean()

plt.figure(figsize=(10, 6))
plt.plot(monthly_consumption.index, monthly_consumption.values, marker='o')
plt.title('Average Monthly Energy Consumption')
plt.xlabel('Month')
plt.ylabel('Average Consumption (kWh)')
plt.grid(True)
plt.show()
```

### 5. LCA Analysis

#### Environmental Impact Assessment
```python
# Load LCA data
building_lca = pd.read_csv('raw_data/lca_data/building_lca_database.csv')

# Environmental impact by building type
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(data=building_lca, x='building_type', y='gwp_per_m2_kg_co2e')
plt.title('GWP per m² by Building Type')
plt.ylabel('GWP (kg CO2e/m²)')
plt.xticks(rotation=45)

plt.subplot(2, 2, 2)
sns.boxplot(data=building_lca, x='building_type', y='energy_per_m2_mj')
plt.title('Energy per m² by Building Type')
plt.ylabel('Energy (MJ/m²)')
plt.xticks(rotation=45)

plt.subplot(2, 2, 3)
sns.boxplot(data=building_lca, x='building_type', y='water_per_m2_l')
plt.title('Water per m² by Building Type')
plt.ylabel('Water (L/m²)')
plt.xticks(rotation=45)

plt.subplot(2, 2, 4)
sns.boxplot(data=building_lca, x='building_type', y='renewable_energy_percent')
plt.title('Renewable Energy % by Building Type')
plt.ylabel('Renewable Energy (%)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

## Data Validation and Quality Checks

### Check Data Completeness
```python
# Load validation results
import json
with open('processed_data/validated/validation_results.json', 'r') as f:
    validation_results = json.load(f)

# Print validation summary
for dataset, results in validation_results.items():
    print(f"\n{dataset}:")
    print(f"  Total records: {results['total_records']}")
    print(f"  Missing values: {sum(results['missing_values'].values())}")
    print(f"  Duplicate records: {results['duplicate_records']}")
```

### Data Quality Assessment
```python
# Check for outliers in energy data
energy_data = pd.read_csv('processed_data/analysis_ready/energy_analysis.csv')

# Identify outliers using IQR method
Q1 = energy_data['energy_intensity_kwh_m2'].quantile(0.25)
Q3 = energy_data['energy_intensity_kwh_m2'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = energy_data[(energy_data['energy_intensity_kwh_m2'] < lower_bound) | 
                      (energy_data['energy_intensity_kwh_m2'] > upper_bound)]

print(f"Outliers detected: {len(outliers)}")
if len(outliers) > 0:
    print("Outlier building IDs:", outliers['building_id'].tolist())
```

## Best Practices

### 1. Data Preprocessing
- Always check for missing values before analysis
- Handle outliers appropriately based on domain knowledge
- Normalize features when comparing different scales
- Use appropriate data types for categorical variables

### 2. Time Series Analysis
- Resample data to appropriate frequencies for your analysis
- Consider seasonal decomposition for trend analysis
- Account for time zone differences in IoT data
- Use proper time series validation techniques

### 3. Machine Learning
- Split data chronologically for time series problems
- Use cross-validation appropriate for your data structure
- Consider feature scaling for algorithms sensitive to scale
- Validate model assumptions and check residuals

### 4. Visualization
- Use appropriate chart types for your data
- Include proper labels and legends
- Consider color accessibility
- Save high-resolution figures for publications

## Troubleshooting

### Common Issues
1. **Memory issues with large IoT datasets**: Use chunking or sampling
2. **Missing data in older buildings**: Use appropriate imputation methods
3. **Time zone issues**: Convert all timestamps to UTC
4. **Categorical encoding**: Use proper encoding for ML algorithms

### Performance Tips
1. Use `pd.read_csv()` with `chunksize` for large files
2. Consider using `dask` for very large datasets
3. Cache frequently used data in memory
4. Use vectorized operations instead of loops

## Support and Contributing

For questions about the dataset or to report issues:
1. Check the documentation first
2. Review the data dictionary for field definitions
3. Examine the validation results for data quality
4. Contact the dataset maintainers for specific questions

## Citation

When using this dataset in your research, please cite:

```
Building Retrofit Dataset for AI- and IoT-driven Optimization
[Your Name], [Institution], 2024
```