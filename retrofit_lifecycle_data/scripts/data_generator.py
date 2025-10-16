"""
Retrofit Intervention & Lifecycle Data Generator
Generates synthetic and parametric data for building retrofit optimization models
Part of the Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RetrofitDataGenerator:
    """Generate synthetic retrofit intervention and lifecycle data"""
    
    def __init__(self, seed: int = 42):
        """Initialize data generator with random seed for reproducibility"""
        np.random.seed(seed)
        self.seed = seed
        
    def generate_degradation_curve(self, 
                                   measure_type: str,
                                   lifespan: int,
                                   initial_performance: float = 100.0,
                                   degradation_type: str = 'linear',
                                   degradation_rate: float = 1.0) -> pd.DataFrame:
        """
        Generate performance degradation curve over lifecycle
        
        Args:
            measure_type: Type of retrofit measure
            lifespan: Expected lifespan in years
            initial_performance: Initial performance (0-100%)
            degradation_type: 'linear', 'exponential', or 's-curve'
            degradation_rate: Annual degradation rate percentage
            
        Returns:
            DataFrame with year, performance, efficiency columns
        """
        years = np.arange(0, lifespan + 1)
        
        if degradation_type == 'linear':
            performance = initial_performance - (degradation_rate * years)
        elif degradation_type == 'exponential':
            performance = initial_performance * np.exp(-degradation_rate * years / 100)
        elif degradation_type == 's-curve':
            # S-curve: slow initial degradation, rapid middle, slow end
            midpoint = lifespan / 2
            steepness = 0.5
            performance = initial_performance / (1 + np.exp(steepness * (years - midpoint)))
            # Normalize to reach end-of-life at expected lifespan
            performance = performance * (100 - degradation_rate * lifespan) / performance[-1]
        else:
            raise ValueError(f"Unknown degradation type: {degradation_type}")
        
        # Add realistic noise
        noise = np.random.normal(0, 1.5, len(years))
        performance = np.clip(performance + noise, 0, 100)
        
        df = pd.DataFrame({
            'year': years,
            'performance_percent': performance,
            'efficiency_ratio': performance / initial_performance,
            'degradation_type': degradation_type,
            'measure_type': measure_type
        })
        
        return df
    
    def generate_operational_data(self,
                                 measure_id: str,
                                 simulation_years: int = 30,
                                 base_consumption: float = 1000.0,
                                 seasonal_variation: float = 0.3) -> pd.DataFrame:
        """
        Generate synthetic operational energy data with realistic patterns
        
        Args:
            measure_id: Retrofit measure identifier
            simulation_years: Number of years to simulate
            base_consumption: Base annual energy consumption (kWh)
            seasonal_variation: Amplitude of seasonal variation (0-1)
            
        Returns:
            DataFrame with monthly operational data
        """
        dates = pd.date_range(start='2020-01-01', 
                            periods=simulation_years * 12, 
                            freq='M')
        
        # Generate monthly energy consumption with seasonal patterns
        months = np.tile(np.arange(12), simulation_years)
        seasonal_factor = 1 + seasonal_variation * np.sin(2 * np.pi * months / 12)
        
        # Add trend (slight increase over time due to usage)
        trend = 1 + 0.005 * np.arange(len(dates))
        
        # Add random variation
        random_factor = np.random.normal(1, 0.1, len(dates))
        
        energy_consumption = base_consumption / 12 * seasonal_factor * trend * random_factor
        
        # Calculate operational carbon based on grid intensity
        carbon_intensity = 0.45  # kg CO2e/kWh (average)
        operational_carbon = energy_consumption * carbon_intensity
        
        # Calculate cost (assuming $0.12/kWh average)
        energy_cost = energy_consumption * 0.12
        
        df = pd.DataFrame({
            'date': dates,
            'measure_id': measure_id,
            'year': dates.year,
            'month': dates.month,
            'energy_consumption_kwh': energy_consumption,
            'operational_carbon_kg_co2e': operational_carbon,
            'energy_cost_usd': energy_cost,
            'heating_load_kwh': energy_consumption * 0.6 * (seasonal_factor > 1).astype(float),
            'cooling_load_kwh': energy_consumption * 0.4 * (seasonal_factor < 1).astype(float)
        })
        
        return df
    
    def generate_maintenance_schedule(self,
                                     measure_id: str,
                                     lifespan: int,
                                     routine_freq_years: float,
                                     major_freq_years: int) -> pd.DataFrame:
        """
        Generate detailed maintenance schedule over lifecycle
        
        Args:
            measure_id: Retrofit measure identifier
            lifespan: Expected lifespan in years
            routine_freq_years: Frequency of routine maintenance
            major_freq_years: Frequency of major maintenance
            
        Returns:
            DataFrame with maintenance schedule
        """
        schedule = []
        current_date = datetime.now()
        
        # Routine maintenance
        if routine_freq_years > 0:
            routine_intervals = int(lifespan / routine_freq_years)
            for i in range(routine_intervals):
                date = current_date + timedelta(days=int(365 * routine_freq_years * i))
                schedule.append({
                    'measure_id': measure_id,
                    'maintenance_date': date,
                    'maintenance_type': 'Routine',
                    'estimated_cost_usd': np.random.normal(200, 50),
                    'estimated_duration_hours': np.random.normal(4, 1),
                    'required_downtime_hours': np.random.normal(2, 0.5),
                    'year_in_service': routine_freq_years * i
                })
        
        # Major maintenance
        if major_freq_years > 0:
            major_intervals = int(lifespan / major_freq_years)
            for i in range(1, major_intervals + 1):
                date = current_date + timedelta(days=int(365 * major_freq_years * i))
                schedule.append({
                    'measure_id': measure_id,
                    'maintenance_date': date,
                    'maintenance_type': 'Major',
                    'estimated_cost_usd': np.random.normal(2000, 500),
                    'estimated_duration_hours': np.random.normal(16, 4),
                    'required_downtime_hours': np.random.normal(8, 2),
                    'year_in_service': major_freq_years * i
                })
        
        df = pd.DataFrame(schedule)
        return df
    
    def generate_cost_scenarios(self,
                                measure_name: str,
                                base_capex: float,
                                base_opex: float,
                                n_scenarios: int = 1000) -> pd.DataFrame:
        """
        Generate Monte Carlo cost scenarios for uncertainty analysis
        
        Args:
            measure_name: Name of retrofit measure
            base_capex: Base capital expenditure
            base_opex: Base operational expenditure (annual)
            n_scenarios: Number of scenarios to generate
            
        Returns:
            DataFrame with cost scenarios
        """
        scenarios = []
        
        for i in range(n_scenarios):
            # CAPEX uncertainty: ±20% typical
            capex_multiplier = np.random.lognormal(0, 0.20)
            capex = base_capex * capex_multiplier
            
            # OPEX uncertainty: ±15% typical
            opex_multiplier = np.random.lognormal(0, 0.15)
            opex = base_opex * opex_multiplier
            
            # Installation cost uncertainty: ±25%
            install_multiplier = np.random.lognormal(0, 0.25)
            
            # Energy price escalation (annual)
            energy_escalation = np.random.uniform(0.02, 0.05)
            
            scenarios.append({
                'scenario_id': i,
                'measure_name': measure_name,
                'capex_usd': capex,
                'annual_opex_usd': opex,
                'installation_cost_multiplier': install_multiplier,
                'energy_price_escalation_rate': energy_escalation,
                'discount_rate': np.random.uniform(0.03, 0.08),
                'project_lifetime_years': np.random.choice([20, 25, 30]),
                'npv_calculation': None  # To be calculated
            })
        
        df = pd.DataFrame(scenarios)
        
        # Calculate NPV for each scenario
        df['npv_usd'] = df.apply(
            lambda row: self._calculate_npv(
                row['capex_usd'],
                row['annual_opex_usd'],
                row['discount_rate'],
                row['project_lifetime_years']
            ),
            axis=1
        )
        
        return df
    
    def _calculate_npv(self, capex: float, annual_opex: float, 
                      discount_rate: float, lifetime: int) -> float:
        """Calculate Net Present Value"""
        npv = -capex
        for year in range(1, lifetime + 1):
            npv += annual_opex / ((1 + discount_rate) ** year)
        return npv
    
    def generate_energy_model_inputs(self,
                                    building_type: str,
                                    floor_area_m2: float,
                                    climate_zone: str) -> Dict:
        """
        Generate building energy model input parameters
        
        Args:
            building_type: 'Residential', 'Commercial', 'Industrial'
            floor_area_m2: Gross floor area in square meters
            climate_zone: ASHRAE climate zone
            
        Returns:
            Dictionary with energy model parameters
        """
        # Base loads by building type (W/m²)
        base_loads = {
            'Residential': {'lighting': 5, 'equipment': 8, 'people': 10},
            'Commercial': {'lighting': 12, 'equipment': 15, 'people': 20},
            'Industrial': {'lighting': 10, 'equipment': 35, 'people': 5}
        }
        
        loads = base_loads.get(building_type, base_loads['Commercial'])
        
        # Climate-dependent parameters
        hdd_cdd = {
            '1': (0, 7500), '2': (1500, 4500), '3': (2500, 2500),
            '4': (4500, 1500), '5': (6000, 1000), '6': (7000, 600),
            '7': (9000, 300), '8': (12000, 100)
        }
        
        hdd, cdd = hdd_cdd.get(climate_zone[0], (5000, 1000))
        
        return {
            'building_type': building_type,
            'floor_area_m2': floor_area_m2,
            'climate_zone': climate_zone,
            'heating_degree_days': hdd,
            'cooling_degree_days': cdd,
            'lighting_power_density_w_m2': loads['lighting'],
            'equipment_power_density_w_m2': loads['equipment'],
            'occupancy_density_people_m2': loads['people'] / 100,
            'infiltration_ach': np.random.uniform(0.3, 1.5),
            'window_to_wall_ratio': np.random.uniform(0.20, 0.40),
            'roof_absorptance': np.random.uniform(0.20, 0.90),
            'wall_u_value_w_m2k': np.random.uniform(0.30, 0.80),
            'roof_u_value_w_m2k': np.random.uniform(0.20, 0.50),
            'window_u_value_w_m2k': np.random.uniform(1.50, 5.00),
            'window_shgc': np.random.uniform(0.25, 0.70),
            'hvac_cop_heating': np.random.uniform(0.75, 0.95),
            'hvac_eer_cooling': np.random.uniform(2.5, 4.0),
            'dhw_efficiency': np.random.uniform(0.60, 0.90),
            'ventilation_rate_l_s_person': np.random.uniform(8, 15)
        }
    
    def generate_synergy_matrix(self, measures: List[str]) -> pd.DataFrame:
        """
        Generate interaction/synergy matrix between retrofit measures
        
        Args:
            measures: List of measure IDs
            
        Returns:
            DataFrame with synergy coefficients
        """
        n = len(measures)
        synergy_matrix = np.ones((n, n))
        
        # Define synergy rules
        synergy_patterns = {
            ('insulation', 'hvac'): 1.15,  # Better insulation improves HVAC efficiency
            ('insulation', 'window'): 1.20,  # Combined envelope improvements
            ('solar', 'battery'): 1.30,  # PV with storage
            ('lighting', 'controls'): 1.25,  # LED with smart controls
            ('hvac', 'erv'): 1.18,  # Heat pump with ERV
            ('hvac', 'controls'): 1.22,  # HVAC with smart controls
            ('window', 'shading'): 1.15,  # High-performance windows with shading
        }
        
        for i, measure_i in enumerate(measures):
            for j, measure_j in enumerate(measures):
                if i != j:
                    # Check for synergy patterns
                    for (cat1, cat2), factor in synergy_patterns.items():
                        if (cat1 in measure_i.lower() and cat2 in measure_j.lower()) or \
                           (cat2 in measure_i.lower() and cat1 in measure_j.lower()):
                            synergy_matrix[i, j] = factor
                    
                    # Add random variation
                    synergy_matrix[i, j] *= np.random.uniform(0.95, 1.05)
        
        df = pd.DataFrame(synergy_matrix, index=measures, columns=measures)
        return df
    
    def generate_retrofit_packages(self, 
                                   n_packages: int = 50) -> pd.DataFrame:
        """
        Generate pre-configured retrofit packages with multiple measures
        
        Args:
            n_packages: Number of packages to generate
            
        Returns:
            DataFrame with retrofit package configurations
        """
        packages = []
        
        package_templates = [
            {
                'name': 'Deep Energy Retrofit - Premium',
                'measures': ['RM002', 'RM004', 'RM006', 'RM010', 'RM018', 'RM024'],
                'target_savings': 0.70,
                'cost_range': (150000, 300000)
            },
            {
                'name': 'Deep Energy Retrofit - Standard',
                'measures': ['RM001', 'RM004', 'RM007', 'RM009', 'RM024'],
                'target_savings': 0.50,
                'cost_range': (80000, 150000)
            },
            {
                'name': 'HVAC Optimization',
                'measures': ['RM009', 'RM015', 'RM017'],
                'target_savings': 0.30,
                'cost_range': (25000, 60000)
            },
            {
                'name': 'Envelope Enhancement',
                'measures': ['RM001', 'RM004', 'RM007', 'RM027'],
                'target_savings': 0.35,
                'cost_range': (40000, 90000)
            },
            {
                'name': 'Renewable Energy Integration',
                'measures': ['RM018', 'RM021', 'RM022'],
                'target_savings': 0.60,
                'cost_range': (50000, 120000)
            },
            {
                'name': 'Lighting & Controls',
                'measures': ['RM024', 'RM025', 'RM017'],
                'target_savings': 0.25,
                'cost_range': (15000, 40000)
            },
            {
                'name': 'Smart Building Upgrade',
                'measures': ['RM037', 'RM038', 'RM039', 'RM017'],
                'target_savings': 0.20,
                'cost_range': (70000, 150000)
            },
            {
                'name': 'Net Zero Ready',
                'measures': ['RM002', 'RM005', 'RM006', 'RM010', 'RM020', 'RM021', 'RM024', 'RM037'],
                'target_savings': 0.85,
                'cost_range': (250000, 500000)
            }
        ]
        
        for template in package_templates:
            for variant in range(n_packages // len(package_templates)):
                cost_min, cost_max = template['cost_range']
                estimated_cost = np.random.uniform(cost_min, cost_max)
                
                packages.append({
                    'package_id': f"PKG_{len(packages):03d}",
                    'package_name': f"{template['name']} - Variant {variant+1}",
                    'measure_ids': ','.join(template['measures']),
                    'num_measures': len(template['measures']),
                    'target_energy_savings_percent': template['target_savings'] * 100,
                    'estimated_total_cost_usd': estimated_cost,
                    'estimated_payback_years': estimated_cost / (template['target_savings'] * 10000),
                    'carbon_reduction_kg_co2e_per_year': template['target_savings'] * 50000,
                    'complexity_score': len(template['measures']) * np.random.uniform(0.8, 1.2),
                    'implementation_sequence': ','.join([str(i+1) for i in range(len(template['measures']))])
                })
        
        df = pd.DataFrame(packages)
        return df


def main():
    """Generate all datasets"""
    print("=" * 80)
    print("Retrofit Intervention & Lifecycle Data Generator")
    print("Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization")
    print("=" * 80)
    print()
    
    generator = RetrofitDataGenerator(seed=42)
    
    # Generate degradation curves for common measures
    print("Generating degradation curves...")
    degradation_data = []
    measure_types = [
        ('Insulation', 30, 'linear', 0.5),
        ('HVAC', 18, 's-curve', 1.2),
        ('Windows', 25, 's-curve', 0.3),
        ('Solar PV', 30, 'linear', 0.5),
        ('LED Lighting', 15, 'exponential', 2.0),
        ('Controls', 10, 'linear', 1.0)
    ]
    
    for mtype, lifespan, deg_type, deg_rate in measure_types:
        df = generator.generate_degradation_curve(mtype, lifespan, 100, deg_type, deg_rate)
        degradation_data.append(df)
    
    degradation_df = pd.concat(degradation_data, ignore_index=True)
    degradation_df.to_csv('../databases/degradation_curves.csv', index=False)
    print(f"  ✓ Generated {len(degradation_df)} degradation curve data points")
    
    # Generate operational data
    print("Generating operational data...")
    operational_data = []
    sample_measures = ['RM001', 'RM009', 'RM018', 'RM024']
    
    for measure_id in sample_measures:
        df = generator.generate_operational_data(measure_id, simulation_years=10)
        operational_data.append(df)
    
    operational_df = pd.concat(operational_data, ignore_index=True)
    operational_df.to_csv('../databases/operational_data.csv', index=False)
    print(f"  ✓ Generated {len(operational_df)} monthly operational records")
    
    # Generate maintenance schedules
    print("Generating maintenance schedules...")
    maintenance_data = []
    
    for measure_id in sample_measures:
        df = generator.generate_maintenance_schedule(measure_id, 20, 1.0, 5)
        maintenance_data.append(df)
    
    maintenance_df = pd.concat(maintenance_data, ignore_index=True)
    maintenance_df.to_csv('../databases/maintenance_schedules.csv', index=False)
    print(f"  ✓ Generated {len(maintenance_df)} maintenance events")
    
    # Generate cost scenarios
    print("Generating cost scenarios for uncertainty analysis...")
    cost_scenarios_df = generator.generate_cost_scenarios('RM009', 6000, 420, n_scenarios=1000)
    cost_scenarios_df.to_csv('../databases/cost_scenarios_monte_carlo.csv', index=False)
    print(f"  ✓ Generated {len(cost_scenarios_df)} cost scenarios")
    
    # Generate energy model inputs
    print("Generating building energy model inputs...")
    building_configs = []
    
    for btype in ['Residential', 'Commercial']:
        for zone in ['3', '4', '5', '6']:
            for _ in range(5):
                config = generator.generate_energy_model_inputs(
                    btype, 
                    np.random.uniform(1000, 10000),
                    zone
                )
                building_configs.append(config)
    
    building_df = pd.DataFrame(building_configs)
    building_df.to_csv('../databases/building_energy_model_inputs.csv', index=False)
    print(f"  ✓ Generated {len(building_df)} building configurations")
    
    # Generate synergy matrix
    print("Generating measure synergy matrix...")
    sample_measures_expanded = [f'RM{i:03d}' for i in range(1, 21)]
    synergy_df = generator.generate_synergy_matrix(sample_measures_expanded)
    synergy_df.to_csv('../databases/measure_synergy_matrix.csv')
    print(f"  ✓ Generated {synergy_df.shape[0]}x{synergy_df.shape[1]} synergy matrix")
    
    # Generate retrofit packages
    print("Generating pre-configured retrofit packages...")
    packages_df = generator.generate_retrofit_packages(n_packages=50)
    packages_df.to_csv('../databases/retrofit_packages.csv', index=False)
    print(f"  ✓ Generated {len(packages_df)} retrofit packages")
    
    print()
    print("=" * 80)
    print("Data generation complete!")
    print("=" * 80)
    print()
    print("Generated files:")
    print("  - degradation_curves.csv")
    print("  - operational_data.csv")
    print("  - maintenance_schedules.csv")
    print("  - cost_scenarios_monte_carlo.csv")
    print("  - building_energy_model_inputs.csv")
    print("  - measure_synergy_matrix.csv")
    print("  - retrofit_packages.csv")
    print()


if __name__ == "__main__":
    main()
