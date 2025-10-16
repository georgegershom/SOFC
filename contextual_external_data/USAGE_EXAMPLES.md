# Usage Examples - Dynamic Digital Twin Framework Dataset

This document provides comprehensive examples of how to use the Contextual & External Data for building retrofit optimization in your Dynamic Digital Twin Framework.

## 🚀 Quick Start Examples

### Example 1: Basic Data Loading and Exploration

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Load all major datasets
def load_all_data():
    """Load all contextual and external datasets"""
    
    data = {}
    
    # Weather data
    data['weather'] = pd.read_csv('weather_climate/tmy_data_new_york_city_2023.csv')
    data['weather']['datetime'] = pd.to_datetime(data['weather']['datetime'])
    
    # Energy prices
    data['energy_prices'] = pd.read_csv('economic_market/energy_prices_new_york/historical_prices_2015_2023.csv')
    data['tou_rates'] = pd.read_csv('economic_market/energy_prices_new_york/tou_rates_2023.csv')
    
    # Material costs
    data['material_costs'] = pd.read_csv('economic_market/material_technology_costs/detailed_material_costs_2023.csv')
    
    # Labor costs
    data['labor_costs'] = pd.read_csv('economic_market/labor_costs_national/trade_labor_rates_2023.csv')
    
    # Carbon intensity
    data['carbon_intensity'] = pd.read_csv('geospatial_regulatory/carbon_intensity/pjm/hourly_carbon_intensity_2023.csv')
    data['carbon_intensity']['datetime'] = pd.to_datetime(data['carbon_intensity']['datetime'])
    
    # Location data
    data['locations'] = pd.read_csv('geospatial_regulatory/location_data/city_location_data.csv')
    
    # Building codes
    data['building_codes'] = pd.read_csv('geospatial_regulatory/building_codes/iecc_2021_requirements.csv')
    
    return data

# Load and explore data
data = load_all_data()

print("Dataset Overview:")
for key, df in data.items():
    print(f"- {key}: {len(df)} records, {len(df.columns)} columns")

# Quick visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Temperature profile
data['weather'].set_index('datetime')['dry_bulb_temp_c'].plot(ax=axes[0,0], title='Annual Temperature Profile')
axes[0,0].set_ylabel('Temperature (°C)')

# Energy price trends
data['energy_prices'].groupby('year')['electricity_price_kwh'].mean().plot(ax=axes[0,1], title='Electricity Price Trends')
axes[0,1].set_ylabel('Price ($/kWh)')

# Carbon intensity daily pattern
hourly_carbon = data['carbon_intensity'].groupby('hour')['average_intensity'].mean()
hourly_carbon.plot(ax=axes[1,0], title='Daily Carbon Intensity Pattern')
axes[1,0].set_ylabel('Carbon Intensity (kg CO2e/kWh)')

# Material costs by category
material_costs_by_category = data['material_costs'].groupby('category')['cost_per_unit'].mean()
material_costs_by_category.plot(kind='bar', ax=axes[1,1], title='Material Costs by Category')
axes[1,1].set_ylabel('Average Cost per Unit')

plt.tight_layout()
plt.show()
```

### Example 2: Building Energy Simulation with Weather Data

```python
class SimpleBuildingEnergyModel:
    """Simplified building energy model using weather data"""
    
    def __init__(self, building_params):
        self.building_params = building_params
        
    def calculate_heating_load(self, outdoor_temp, indoor_setpoint=21):
        """Calculate heating load based on outdoor temperature"""
        if outdoor_temp < indoor_setpoint:
            # Simplified heating load calculation
            temp_diff = indoor_setpoint - outdoor_temp
            heating_load = (temp_diff * self.building_params['ua_value'] * 
                          self.building_params['floor_area'] / 1000)  # kW
            return max(0, heating_load)
        return 0
    
    def calculate_cooling_load(self, outdoor_temp, solar_radiation, indoor_setpoint=24):
        """Calculate cooling load based on outdoor temperature and solar gains"""
        if outdoor_temp > indoor_setpoint:
            # Temperature-driven cooling load
            temp_diff = outdoor_temp - indoor_setpoint
            temp_load = (temp_diff * self.building_params['ua_value'] * 
                        self.building_params['floor_area'] / 1000)
            
            # Solar-driven cooling load
            solar_load = (solar_radiation * self.building_params['window_area'] * 
                         self.building_params['shgc'] / 1000)
            
            return max(0, temp_load + solar_load)
        return 0
    
    def simulate_annual_energy(self, weather_data):
        """Simulate annual energy consumption"""
        
        results = []
        
        for _, hour in weather_data.iterrows():
            heating_load = self.calculate_heating_load(hour['dry_bulb_temp_c'])
            cooling_load = self.calculate_cooling_load(
                hour['dry_bulb_temp_c'], 
                hour['global_horizontal_irradiance']
            )
            
            # Convert loads to energy consumption (accounting for equipment efficiency)
            heating_energy = heating_load / self.building_params['heating_efficiency']
            cooling_energy = cooling_load / self.building_params['cooling_efficiency']
            
            # Base electrical load (lighting, equipment, etc.)
            base_load = self.building_params['base_electrical_load']
            
            total_energy = heating_energy + cooling_energy + base_load
            
            results.append({
                'datetime': hour['datetime'],
                'outdoor_temp': hour['dry_bulb_temp_c'],
                'solar_radiation': hour['global_horizontal_irradiance'],
                'heating_load_kw': heating_load,
                'cooling_load_kw': cooling_load,
                'heating_energy_kwh': heating_energy,
                'cooling_energy_kwh': cooling_energy,
                'base_energy_kwh': base_load,
                'total_energy_kwh': total_energy
            })
        
        return pd.DataFrame(results)

# Example usage
building_params = {
    'floor_area': 1000,  # m²
    'ua_value': 0.3,     # W/m²K (building thermal conductance)
    'window_area': 200,  # m²
    'shgc': 0.4,         # Solar Heat Gain Coefficient
    'heating_efficiency': 0.85,  # Heating system efficiency
    'cooling_efficiency': 3.0,   # Cooling system COP
    'base_electrical_load': 10   # kW base electrical load
}

# Load weather data
weather_data = pd.read_csv('weather_climate/tmy_data_new_york_city_2023.csv')
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])

# Create and run energy model
energy_model = SimpleBuildingEnergyModel(building_params)
energy_results = energy_model.simulate_annual_energy(weather_data)

# Analyze results
annual_energy = energy_results['total_energy_kwh'].sum()
peak_demand = energy_results['total_energy_kwh'].max()
heating_fraction = energy_results['heating_energy_kwh'].sum() / annual_energy
cooling_fraction = energy_results['cooling_energy_kwh'].sum() / annual_energy

print(f"Annual Energy Consumption: {annual_energy:,.0f} kWh")
print(f"Peak Demand: {peak_demand:.1f} kW")
print(f"Heating Fraction: {heating_fraction:.1%}")
print(f"Cooling Fraction: {cooling_fraction:.1%}")
```

### Example 3: Retrofit Cost-Benefit Analysis

```python
class RetrofitAnalyzer:
    """Comprehensive retrofit cost-benefit analysis"""
    
    def __init__(self, material_costs, labor_costs, energy_prices):
        self.material_costs = material_costs
        self.labor_costs = labor_costs
        self.energy_prices = energy_prices
    
    def calculate_retrofit_cost(self, retrofit_measures):
        """Calculate total retrofit cost including materials and labor"""
        
        total_cost = 0
        cost_breakdown = {}
        
        for measure in retrofit_measures:
            # Material costs
            material_cost = self._get_material_cost(measure)
            
            # Labor costs
            labor_cost = self._get_labor_cost(measure)
            
            # Total measure cost
            measure_cost = material_cost + labor_cost
            total_cost += measure_cost
            
            cost_breakdown[measure['name']] = {
                'material_cost': material_cost,
                'labor_cost': labor_cost,
                'total_cost': measure_cost
            }
        
        return total_cost, cost_breakdown
    
    def _get_material_cost(self, measure):
        """Get material cost for a retrofit measure"""
        
        material_data = self.material_costs[
            (self.material_costs['category'] == measure['category']) &
            (self.material_costs['material_type'] == measure['material_type'])
        ]
        
        if len(material_data) > 0:
            cost_per_unit = material_data.iloc[0]['cost_per_unit']
            return cost_per_unit * measure['quantity']
        
        return 0
    
    def _get_labor_cost(self, measure):
        """Get labor cost for a retrofit measure"""
        
        labor_data = self.labor_costs[
            self.labor_costs['trade'] == measure['trade']
        ]
        
        if len(labor_data) > 0:
            hourly_rate = labor_data.iloc[0]['fully_burdened_non_union']
            return hourly_rate * measure['labor_hours']
        
        return 0
    
    def calculate_energy_savings_value(self, energy_savings_kwh, analysis_period=20):
        """Calculate the financial value of energy savings"""
        
        # Use average electricity price
        avg_price = self.energy_prices['electricity_price_kwh'].mean()
        
        # Calculate annual savings value
        annual_savings = energy_savings_kwh * avg_price
        
        # Calculate net present value of savings over analysis period
        discount_rate = 0.06  # 6% discount rate
        npv_savings = 0
        
        for year in range(1, analysis_period + 1):
            # Escalate energy prices at 3% per year
            escalated_savings = annual_savings * (1.03 ** year)
            # Discount to present value
            pv_savings = escalated_savings / (1 + discount_rate) ** year
            npv_savings += pv_savings
        
        return annual_savings, npv_savings
    
    def perform_retrofit_analysis(self, retrofit_measures, energy_savings_kwh):
        """Perform complete retrofit cost-benefit analysis"""
        
        # Calculate costs
        total_cost, cost_breakdown = self.calculate_retrofit_cost(retrofit_measures)
        
        # Calculate energy savings value
        annual_savings, npv_savings = self.calculate_energy_savings_value(energy_savings_kwh)
        
        # Calculate financial metrics
        simple_payback = total_cost / annual_savings if annual_savings > 0 else float('inf')
        net_present_value = npv_savings - total_cost
        benefit_cost_ratio = npv_savings / total_cost if total_cost > 0 else 0
        
        return {
            'total_cost': total_cost,
            'cost_breakdown': cost_breakdown,
            'annual_energy_savings_kwh': energy_savings_kwh,
            'annual_cost_savings': annual_savings,
            'npv_savings': npv_savings,
            'net_present_value': net_present_value,
            'simple_payback_years': simple_payback,
            'benefit_cost_ratio': benefit_cost_ratio
        }

# Example retrofit analysis
material_costs = pd.read_csv('economic_market/material_technology_costs/detailed_material_costs_2023.csv')
labor_costs = pd.read_csv('economic_market/labor_costs_national/trade_labor_rates_2023.csv')
energy_prices = pd.read_csv('economic_market/energy_prices_new_york/historical_prices_2015_2023.csv')

# Define retrofit measures
retrofit_measures = [
    {
        'name': 'Wall Insulation',
        'category': 'insulation',
        'material_type': 'spray_foam_closed',
        'quantity': 500,  # sq ft
        'trade': 'insulation_installer',
        'labor_hours': 40
    },
    {
        'name': 'High-Performance Windows',
        'category': 'windows',
        'material_type': 'triple_pane_high_performance',
        'quantity': 200,  # sq ft
        'trade': 'general_contractor',
        'labor_hours': 80
    },
    {
        'name': 'Heat Pump System',
        'category': 'hvac',
        'material_type': 'heat_pump_air_source',
        'quantity': 48000,  # BTU
        'trade': 'hvac_technician',
        'labor_hours': 24
    }
]

# Estimated energy savings (would come from energy simulation)
estimated_energy_savings = 15000  # kWh/year

# Perform analysis
analyzer = RetrofitAnalyzer(material_costs, labor_costs, energy_prices)
results = analyzer.perform_retrofit_analysis(retrofit_measures, estimated_energy_savings)

print("Retrofit Analysis Results:")
print(f"Total Cost: ${results['total_cost']:,.0f}")
print(f"Annual Energy Savings: {results['annual_energy_savings_kwh']:,.0f} kWh")
print(f"Annual Cost Savings: ${results['annual_cost_savings']:,.0f}")
print(f"Simple Payback: {results['simple_payback_years']:.1f} years")
print(f"Net Present Value: ${results['net_present_value']:,.0f}")
print(f"Benefit-Cost Ratio: {results['benefit_cost_ratio']:.2f}")
```

### Example 4: Real-Time Carbon Optimization

```python
class CarbonOptimizer:
    """Real-time carbon emission optimization"""
    
    def __init__(self, carbon_intensity_data):
        self.carbon_data = carbon_intensity_data
        self.carbon_data['datetime'] = pd.to_datetime(self.carbon_data['datetime'])
        
    def get_current_carbon_intensity(self, timestamp):
        """Get carbon intensity for a specific timestamp"""
        
        # Find the closest timestamp in the data
        time_diff = abs(self.carbon_data['datetime'] - timestamp)
        closest_idx = time_diff.idxmin()
        
        return {
            'average_intensity': self.carbon_data.loc[closest_idx, 'average_intensity'],
            'marginal_intensity': self.carbon_data.loc[closest_idx, 'marginal_intensity']
        }
    
    def predict_carbon_intensity(self, timestamp, hours_ahead=24):
        """Predict carbon intensity for the next N hours"""
        
        # Simple prediction based on historical patterns
        hour = timestamp.hour
        month = timestamp.month
        
        predictions = []
        
        for h in range(hours_ahead):
            future_hour = (hour + h) % 24
            future_timestamp = timestamp + timedelta(hours=h)
            
            # Get historical pattern for this hour and month
            pattern_data = self.carbon_data[
                (self.carbon_data['hour'] == future_hour) &
                (self.carbon_data['month'] == month)
            ]
            
            if len(pattern_data) > 0:
                avg_intensity = pattern_data['average_intensity'].mean()
                marginal_intensity = pattern_data['marginal_intensity'].mean()
            else:
                # Fallback to overall averages
                avg_intensity = self.carbon_data['average_intensity'].mean()
                marginal_intensity = self.carbon_data['marginal_intensity'].mean()
            
            predictions.append({
                'timestamp': future_timestamp,
                'hour': future_hour,
                'predicted_average_intensity': avg_intensity,
                'predicted_marginal_intensity': marginal_intensity
            })
        
        return pd.DataFrame(predictions)
    
    def optimize_energy_schedule(self, energy_demand_schedule, flexibility_hours=4):
        """Optimize energy consumption schedule to minimize carbon emissions"""
        
        optimized_schedule = []
        
        for _, demand in energy_demand_schedule.iterrows():
            base_timestamp = demand['timestamp']
            base_energy = demand['energy_kwh']
            
            # Check if this load can be shifted
            if demand.get('flexible', False):
                # Get carbon intensity predictions for flexibility window
                predictions = self.predict_carbon_intensity(base_timestamp, flexibility_hours)
                
                # Find the hour with lowest carbon intensity
                best_hour_idx = predictions['predicted_marginal_intensity'].idxmin()
                best_timestamp = predictions.loc[best_hour_idx, 'timestamp']
                best_intensity = predictions.loc[best_hour_idx, 'predicted_marginal_intensity']
                
                optimized_schedule.append({
                    'original_timestamp': base_timestamp,
                    'optimized_timestamp': best_timestamp,
                    'energy_kwh': base_energy,
                    'original_carbon_intensity': self.get_current_carbon_intensity(base_timestamp)['marginal_intensity'],
                    'optimized_carbon_intensity': best_intensity,
                    'carbon_savings_kg': base_energy * (self.get_current_carbon_intensity(base_timestamp)['marginal_intensity'] - best_intensity)
                })
            else:
                # Non-flexible load - keep at original time
                current_intensity = self.get_current_carbon_intensity(base_timestamp)['marginal_intensity']
                
                optimized_schedule.append({
                    'original_timestamp': base_timestamp,
                    'optimized_timestamp': base_timestamp,
                    'energy_kwh': base_energy,
                    'original_carbon_intensity': current_intensity,
                    'optimized_carbon_intensity': current_intensity,
                    'carbon_savings_kg': 0
                })
        
        return pd.DataFrame(optimized_schedule)

# Example carbon optimization
carbon_data = pd.read_csv('geospatial_regulatory/carbon_intensity/pjm/hourly_carbon_intensity_2023.csv')
optimizer = CarbonOptimizer(carbon_data)

# Example energy demand schedule
energy_schedule = pd.DataFrame([
    {'timestamp': pd.Timestamp('2023-06-15 14:00'), 'energy_kwh': 100, 'flexible': False, 'description': 'HVAC - Peak cooling'},
    {'timestamp': pd.Timestamp('2023-06-15 20:00'), 'energy_kwh': 50, 'flexible': True, 'description': 'Water heating'},
    {'timestamp': pd.Timestamp('2023-06-15 22:00'), 'energy_kwh': 25, 'flexible': True, 'description': 'EV charging'},
    {'timestamp': pd.Timestamp('2023-06-16 02:00'), 'energy_kwh': 75, 'flexible': True, 'description': 'Thermal storage charging'}
])

# Optimize schedule
optimized = optimizer.optimize_energy_schedule(energy_schedule)

print("Carbon Optimization Results:")
total_original_emissions = (optimized['energy_kwh'] * optimized['original_carbon_intensity']).sum()
total_optimized_emissions = (optimized['energy_kwh'] * optimized['optimized_carbon_intensity']).sum()
total_savings = optimized['carbon_savings_kg'].sum()

print(f"Original emissions: {total_original_emissions:.1f} kg CO2e")
print(f"Optimized emissions: {total_optimized_emissions:.1f} kg CO2e")
print(f"Carbon savings: {total_savings:.1f} kg CO2e ({total_savings/total_original_emissions:.1%} reduction)")
```

### Example 5: Multi-Objective Retrofit Optimization

```python
class MultiObjectiveRetrofitOptimizer:
    """Multi-objective optimization for building retrofits"""
    
    def __init__(self, external_data):
        self.weather_data = external_data['weather']
        self.energy_prices = external_data['energy_prices']
        self.carbon_data = external_data['carbon_intensity']
        self.material_costs = external_data['material_costs']
        self.labor_costs = external_data['labor_costs']
        self.building_codes = external_data['building_codes']
    
    def evaluate_retrofit_solution(self, retrofit_solution, building_params):
        """Evaluate a retrofit solution across multiple objectives"""
        
        # Objective 1: Energy Performance
        energy_metrics = self._calculate_energy_performance(retrofit_solution, building_params)
        
        # Objective 2: Economic Performance
        economic_metrics = self._calculate_economic_performance(retrofit_solution, energy_metrics)
        
        # Objective 3: Environmental Performance
        environmental_metrics = self._calculate_environmental_performance(energy_metrics)
        
        # Objective 4: Regulatory Compliance
        compliance_metrics = self._calculate_compliance_performance(retrofit_solution, building_params)
        
        return {
            'energy': energy_metrics,
            'economic': economic_metrics,
            'environmental': environmental_metrics,
            'compliance': compliance_metrics,
            'overall_score': self._calculate_overall_score(
                energy_metrics, economic_metrics, environmental_metrics, compliance_metrics
            )
        }
    
    def _calculate_energy_performance(self, retrofit_solution, building_params):
        """Calculate energy performance metrics"""
        
        # Simulate building energy with retrofits
        modified_params = building_params.copy()
        
        # Apply retrofit modifications
        for retrofit in retrofit_solution:
            if retrofit['type'] == 'insulation':
                # Improve building thermal performance
                modified_params['ua_value'] *= (1 - retrofit['improvement_factor'])
            elif retrofit['type'] == 'windows':
                # Improve window performance
                modified_params['window_u_factor'] = retrofit['u_factor']
                modified_params['shgc'] = retrofit['shgc']
            elif retrofit['type'] == 'hvac':
                # Improve HVAC efficiency
                modified_params['heating_efficiency'] = retrofit['heating_efficiency']
                modified_params['cooling_efficiency'] = retrofit['cooling_efficiency']
        
        # Simulate energy consumption
        energy_model = SimpleBuildingEnergyModel(modified_params)
        energy_results = energy_model.simulate_annual_energy(self.weather_data)
        
        return {
            'annual_energy_kwh': energy_results['total_energy_kwh'].sum(),
            'peak_demand_kw': energy_results['total_energy_kwh'].max(),
            'energy_intensity_kwh_m2': energy_results['total_energy_kwh'].sum() / building_params['floor_area'],
            'heating_energy_kwh': energy_results['heating_energy_kwh'].sum(),
            'cooling_energy_kwh': energy_results['cooling_energy_kwh'].sum()
        }
    
    def _calculate_economic_performance(self, retrofit_solution, energy_metrics):
        """Calculate economic performance metrics"""
        
        # Calculate retrofit costs
        total_cost = 0
        for retrofit in retrofit_solution:
            material_cost = self._get_retrofit_material_cost(retrofit)
            labor_cost = self._get_retrofit_labor_cost(retrofit)
            total_cost += material_cost + labor_cost
        
        # Calculate energy cost savings
        avg_energy_price = self.energy_prices['electricity_price_kwh'].mean()
        annual_energy_cost = energy_metrics['annual_energy_kwh'] * avg_energy_price
        
        # Baseline energy cost (estimated)
        baseline_energy_cost = annual_energy_cost * 1.3  # Assume 30% savings
        annual_savings = baseline_energy_cost - annual_energy_cost
        
        # Calculate financial metrics
        simple_payback = total_cost / annual_savings if annual_savings > 0 else float('inf')
        
        # Net present value over 20 years
        discount_rate = 0.06
        npv = -total_cost
        for year in range(1, 21):
            escalated_savings = annual_savings * (1.03 ** year)
            pv_savings = escalated_savings / (1 + discount_rate) ** year
            npv += pv_savings
        
        return {
            'total_retrofit_cost': total_cost,
            'annual_energy_cost': annual_energy_cost,
            'annual_savings': annual_savings,
            'simple_payback_years': simple_payback,
            'net_present_value': npv,
            'benefit_cost_ratio': (npv + total_cost) / total_cost if total_cost > 0 else 0
        }
    
    def _calculate_environmental_performance(self, energy_metrics):
        """Calculate environmental performance metrics"""
        
        # Calculate carbon emissions
        avg_carbon_intensity = self.carbon_data['average_intensity'].mean()
        annual_emissions = energy_metrics['annual_energy_kwh'] * avg_carbon_intensity
        
        # Baseline emissions (estimated)
        baseline_emissions = annual_emissions * 1.3  # Assume 30% reduction
        emissions_reduction = baseline_emissions - annual_emissions
        
        return {
            'annual_carbon_emissions_kg': annual_emissions,
            'carbon_intensity_kg_kwh': avg_carbon_intensity,
            'emissions_reduction_kg': emissions_reduction,
            'emissions_reduction_percent': emissions_reduction / baseline_emissions if baseline_emissions > 0 else 0
        }
    
    def _calculate_compliance_performance(self, retrofit_solution, building_params):
        """Calculate regulatory compliance performance"""
        
        # Check building code compliance
        climate_zone = building_params.get('climate_zone', '4A')
        building_type = building_params.get('building_type', 'commercial')
        
        code_requirements = self.building_codes[
            (self.building_codes['climate_zone'] == climate_zone) &
            (self.building_codes['building_type'] == building_type)
        ]
        
        compliance_score = 0
        if len(code_requirements) > 0:
            req = code_requirements.iloc[0]
            
            # Check various compliance criteria
            compliance_checks = []
            
            # Window U-factor compliance
            if 'window_u_factor' in req and not pd.isna(req['window_u_factor']):
                actual_u_factor = self._get_retrofit_window_u_factor(retrofit_solution)
                if actual_u_factor <= req['window_u_factor']:
                    compliance_checks.append(1)
                else:
                    compliance_checks.append(0)
            
            # Add more compliance checks as needed
            compliance_score = np.mean(compliance_checks) if compliance_checks else 0.5
        
        return {
            'code_compliance_score': compliance_score,
            'climate_zone': climate_zone,
            'applicable_codes': 'IECC 2021'
        }
    
    def _calculate_overall_score(self, energy, economic, environmental, compliance):
        """Calculate weighted overall performance score"""
        
        # Normalize metrics to 0-1 scale
        energy_score = 1 / (1 + energy['energy_intensity_kwh_m2'] / 100)  # Lower is better
        economic_score = max(0, min(1, economic['benefit_cost_ratio'] / 2))  # Higher is better
        environmental_score = environmental['emissions_reduction_percent']  # Higher is better
        compliance_score = compliance['code_compliance_score']  # Higher is better
        
        # Weighted average (adjust weights as needed)
        weights = {'energy': 0.3, 'economic': 0.3, 'environmental': 0.25, 'compliance': 0.15}
        
        overall_score = (
            weights['energy'] * energy_score +
            weights['economic'] * economic_score +
            weights['environmental'] * environmental_score +
            weights['compliance'] * compliance_score
        )
        
        return overall_score
    
    def _get_retrofit_material_cost(self, retrofit):
        """Get material cost for a retrofit measure"""
        material_data = self.material_costs[
            (self.material_costs['category'] == retrofit['type']) &
            (self.material_costs['material_type'] == retrofit.get('material_type', ''))
        ]
        
        if len(material_data) > 0:
            return material_data.iloc[0]['cost_per_unit'] * retrofit.get('quantity', 1)
        return 1000  # Default cost
    
    def _get_retrofit_labor_cost(self, retrofit):
        """Get labor cost for a retrofit measure"""
        trade_map = {'insulation': 'insulation_installer', 'windows': 'general_contractor', 'hvac': 'hvac_technician'}
        trade = trade_map.get(retrofit['type'], 'general_contractor')
        
        labor_data = self.labor_costs[self.labor_costs['trade'] == trade]
        
        if len(labor_data) > 0:
            return labor_data.iloc[0]['fully_burdened_non_union'] * retrofit.get('labor_hours', 8)
        return 800  # Default cost
    
    def _get_retrofit_window_u_factor(self, retrofit_solution):
        """Get window U-factor from retrofit solution"""
        for retrofit in retrofit_solution:
            if retrofit['type'] == 'windows':
                return retrofit.get('u_factor', 0.5)
        return 0.5  # Default value

# Example multi-objective optimization
data = load_all_data()
optimizer = MultiObjectiveRetrofitOptimizer(data)

# Define building parameters
building_params = {
    'floor_area': 2000,  # m²
    'ua_value': 0.4,     # W/m²K
    'window_area': 400,  # m²
    'window_u_factor': 0.6,
    'shgc': 0.5,
    'heating_efficiency': 0.8,
    'cooling_efficiency': 2.5,
    'base_electrical_load': 20,
    'climate_zone': '4A',
    'building_type': 'commercial'
}

# Define retrofit solutions to compare
retrofit_solutions = [
    # Solution 1: Basic retrofits
    [
        {'type': 'insulation', 'improvement_factor': 0.2, 'quantity': 1000, 'labor_hours': 40},
        {'type': 'windows', 'u_factor': 0.35, 'shgc': 0.4, 'quantity': 400, 'labor_hours': 80}
    ],
    # Solution 2: Comprehensive retrofits
    [
        {'type': 'insulation', 'improvement_factor': 0.4, 'quantity': 1000, 'labor_hours': 60},
        {'type': 'windows', 'u_factor': 0.25, 'shgc': 0.3, 'quantity': 400, 'labor_hours': 100},
        {'type': 'hvac', 'heating_efficiency': 0.95, 'cooling_efficiency': 4.0, 'quantity': 1, 'labor_hours': 40}
    ],
    # Solution 3: High-performance retrofits
    [
        {'type': 'insulation', 'improvement_factor': 0.6, 'quantity': 1000, 'labor_hours': 80},
        {'type': 'windows', 'u_factor': 0.15, 'shgc': 0.25, 'quantity': 400, 'labor_hours': 120},
        {'type': 'hvac', 'heating_efficiency': 0.98, 'cooling_efficiency': 5.0, 'quantity': 1, 'labor_hours': 60}
    ]
]

# Evaluate all solutions
results = []
for i, solution in enumerate(retrofit_solutions):
    result = optimizer.evaluate_retrofit_solution(solution, building_params)
    result['solution_id'] = i + 1
    results.append(result)

# Compare solutions
print("Multi-Objective Retrofit Optimization Results:")
print("=" * 60)

for result in results:
    print(f"\nSolution {result['solution_id']}:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  Energy Intensity: {result['energy']['energy_intensity_kwh_m2']:.1f} kWh/m²")
    print(f"  Payback Period: {result['economic']['simple_payback_years']:.1f} years")
    print(f"  NPV: ${result['economic']['net_present_value']:,.0f}")
    print(f"  Emissions Reduction: {result['environmental']['emissions_reduction_percent']:.1%}")
    print(f"  Code Compliance: {result['compliance']['code_compliance_score']:.1%}")

# Find best solution
best_solution = max(results, key=lambda x: x['overall_score'])
print(f"\nBest Solution: Solution {best_solution['solution_id']} (Score: {best_solution['overall_score']:.3f})")
```

## 🔄 Advanced Integration Patterns

### Real-Time Data Integration

```python
class RealTimeDataIntegrator:
    """Integrate real-time data with historical patterns"""
    
    def __init__(self, historical_data):
        self.historical_data = historical_data
        
    def blend_real_time_weather(self, real_time_weather, forecast_hours=24):
        """Blend real-time weather with historical patterns"""
        
        current_time = datetime.now()
        
        # Use real-time data for current conditions
        blended_data = [real_time_weather]
        
        # Use historical patterns for forecast
        for h in range(1, forecast_hours):
            future_time = current_time + timedelta(hours=h)
            historical_pattern = self._get_historical_pattern(future_time)
            
            # Adjust historical pattern based on current conditions
            adjusted_forecast = self._adjust_forecast(historical_pattern, real_time_weather)
            blended_data.append(adjusted_forecast)
        
        return pd.DataFrame(blended_data)
    
    def _get_historical_pattern(self, timestamp):
        """Get historical weather pattern for a specific time"""
        
        hour = timestamp.hour
        month = timestamp.month
        
        pattern = self.historical_data[
            (self.historical_data['hour'] == hour) &
            (self.historical_data['month'] == month)
        ]
        
        if len(pattern) > 0:
            return pattern.mean().to_dict()
        
        return self.historical_data.mean().to_dict()
    
    def _adjust_forecast(self, historical_pattern, current_conditions):
        """Adjust historical pattern based on current conditions"""
        
        # Simple adjustment - blend historical with current trend
        adjustment_factor = 0.3
        
        adjusted = {}
        for key in historical_pattern:
            if key in current_conditions:
                adjusted[key] = (
                    historical_pattern[key] * (1 - adjustment_factor) +
                    current_conditions[key] * adjustment_factor
                )
            else:
                adjusted[key] = historical_pattern[key]
        
        return adjusted

# Example real-time integration
historical_weather = pd.read_csv('weather_climate/tmy_data_new_york_city_2023.csv')
integrator = RealTimeDataIntegrator(historical_weather)

# Simulate real-time weather data
real_time_weather = {
    'datetime': datetime.now(),
    'dry_bulb_temp_c': 25.5,
    'relative_humidity': 65.0,
    'global_horizontal_irradiance': 450.0,
    'wind_speed_ms': 3.2
}

# Get blended forecast
forecast = integrator.blend_real_time_weather(real_time_weather)
print(f"Generated {len(forecast)} hours of blended weather forecast")
```

This comprehensive set of examples demonstrates how to leverage the contextual and external data for various aspects of building retrofit optimization within your Dynamic Digital Twin Framework. The examples progress from basic data loading to sophisticated multi-objective optimization scenarios, showing the full potential of the dataset for real-world applications.