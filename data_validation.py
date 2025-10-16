#!/usr/bin/env python3
"""
Data Validation and Quality Checks for IoT Building Dataset

This module provides comprehensive validation and quality assessment
for the generated IoT building sensor dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

class IoTDataValidator:
    """
    Comprehensive data validation and quality assessment for IoT building datasets.
    """
    
    def __init__(self, dataset_path):
        """
        Initialize validator with dataset path.
        
        Args:
            dataset_path: Path to the dataset file (CSV, Parquet, or Excel)
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.validation_results = {}
        
    def load_dataset(self):
        """Load the dataset from file."""
        print(f"Loading dataset from {self.dataset_path}...")
        
        if self.dataset_path.endswith('.csv'):
            self.dataset = pd.read_csv(self.dataset_path)
        elif self.dataset_path.endswith('.parquet'):
            self.dataset = pd.read_parquet(self.dataset_path)
        elif self.dataset_path.endswith('.xlsx'):
            self.dataset = pd.read_excel(self.dataset_path)
        else:
            raise ValueError("Unsupported file format. Use CSV, Parquet, or Excel.")
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.dataset.columns:
            self.dataset['timestamp'] = pd.to_datetime(self.dataset['timestamp'], utc=True)
        
        print(f"Dataset loaded successfully. Shape: {self.dataset.shape}")
        return self.dataset
    
    def validate_data_completeness(self):
        """Check for missing values and data completeness."""
        print("\n=== DATA COMPLETENESS VALIDATION ===")
        
        missing_data = self.dataset.isnull().sum()
        missing_percentage = (missing_data / len(self.dataset)) * 100
        
        completeness_results = {
            'total_records': len(self.dataset),
            'missing_values': missing_data.to_dict(),
            'missing_percentage': missing_percentage.to_dict(),
            'completeness_score': (1 - missing_data.sum() / (len(self.dataset) * len(self.dataset.columns))) * 100
        }
        
        print(f"Total records: {completeness_results['total_records']:,}")
        print(f"Overall completeness: {completeness_results['completeness_score']:.2f}%")
        
        # Check for columns with significant missing data
        high_missing = missing_percentage[missing_percentage > 5]
        if len(high_missing) > 0:
            print(f"\nColumns with >5% missing data:")
            for col, pct in high_missing.items():
                print(f"  {col}: {pct:.2f}%")
        else:
            print("✓ No columns with significant missing data")
        
        self.validation_results['completeness'] = completeness_results
        return completeness_results
    
    def validate_data_ranges(self):
        """Validate that data values are within expected ranges."""
        print("\n=== DATA RANGE VALIDATION ===")
        
        range_checks = {
            # Weather data
            'outdoor_temperature_f': (-50, 120),
            'outdoor_humidity_rh': (0, 100),
            'solar_irradiance_wm2': (0, 1200),
            'wind_speed_ms': (0, 50),
            'wind_direction_deg': (0, 360),
            'precipitation_mm': (0, 100),
            
            # Energy data
            'total_electricity_kwh': (0, 10000),
            'hvac_electricity_kwh': (0, 5000),
            'lighting_electricity_kwh': (0, 1000),
            'plug_loads_kwh': (0, 2000),
            
            # IEQ data
            'indoor_humidity_rh': (0, 100),
            'co2_ppm': (300, 2000),
            'pm25_ugm3': (0, 100),
            'tvoc_ppb': (0, 500),
            'illuminance_lux': (0, 2000),
            'noise_level_db': (20, 100),
            
            # Occupancy data
            'occupant_count': (0, 500),
            'space_utilization_pct': (0, 100),
            'window_operation': (0, 1),
            'blind_operation': (0, 1),
            
            # HVAC data
            'supply_air_temp_f': (50, 90),
            'return_air_temp_f': (50, 90),
            'damper_position_pct': (0, 100),
            'fan_speed_pct': (0, 100),
            'valve_position_pct': (0, 100),
            'chiller_status': (0, 1),
            'boiler_status': (0, 1),
            'heating_setpoint_f': (60, 80),
            'cooling_setpoint_f': (70, 85)
        }
        
        range_violations = {}
        
        for column, (min_val, max_val) in range_checks.items():
            if column in self.dataset.columns:
                violations = ((self.dataset[column] < min_val) | (self.dataset[column] > max_val)).sum()
                if violations > 0:
                    range_violations[column] = {
                        'violations': violations,
                        'percentage': (violations / len(self.dataset)) * 100,
                        'min_found': self.dataset[column].min(),
                        'max_found': self.dataset[column].max(),
                        'expected_range': (min_val, max_val)
                    }
                    print(f"⚠️  {column}: {violations} violations ({violations/len(self.dataset)*100:.2f}%)")
                    print(f"    Range: {self.dataset[column].min():.2f} - {self.dataset[column].max():.2f} (expected: {min_val} - {max_val})")
                else:
                    print(f"✓ {column}: All values within expected range")
        
        if not range_violations:
            print("✓ All data within expected ranges")
        
        self.validation_results['range_validation'] = range_violations
        return range_violations
    
    def validate_temporal_consistency(self):
        """Validate temporal consistency and time series properties."""
        print("\n=== TEMPORAL CONSISTENCY VALIDATION ===")
        
        temporal_results = {}
        
        # Check for duplicate timestamps
        duplicate_timestamps = self.dataset['timestamp'].duplicated().sum()
        temporal_results['duplicate_timestamps'] = duplicate_timestamps
        
        if duplicate_timestamps > 0:
            print(f"⚠️  Found {duplicate_timestamps} duplicate timestamps")
        else:
            print("✓ No duplicate timestamps")
        
        # Check for gaps in time series
        expected_freq = pd.Timedelta(minutes=15)
        time_diffs = self.dataset['timestamp'].diff().dropna()
        gaps = (time_diffs != expected_freq).sum()
        temporal_results['time_gaps'] = gaps
        
        if gaps > 0:
            print(f"⚠️  Found {gaps} time gaps in the series")
        else:
            print("✓ No time gaps in the series")
        
        # Check for unrealistic jumps in sensor values
        jump_thresholds = {
            'outdoor_temperature_f': 20,  # 20°F change in 15 minutes
            'total_electricity_kwh': 1000,  # 1000 kWh change in 15 minutes
            'co2_ppm': 200,  # 200 ppm change in 15 minutes
        }
        
        jump_violations = {}
        for column, threshold in jump_thresholds.items():
            if column in self.dataset.columns:
                jumps = (self.dataset[column].diff().abs() > threshold).sum()
                if jumps > 0:
                    jump_violations[column] = jumps
                    print(f"⚠️  {column}: {jumps} unrealistic jumps detected")
                else:
                    print(f"✓ {column}: No unrealistic jumps")
        
        temporal_results['jump_violations'] = jump_violations
        
        self.validation_results['temporal_consistency'] = temporal_results
        return temporal_results
    
    def validate_correlations(self):
        """Validate expected correlations between sensor data."""
        print("\n=== CORRELATION VALIDATION ===")
        
        correlation_results = {}
        
        # Expected correlations
        expected_correlations = [
            ('outdoor_temperature_f', 'hvac_electricity_kwh', 0.3),  # Positive correlation
            ('outdoor_temperature_f', 'indoor_humidity_rh', 0.2),    # Positive correlation
            ('occupant_count', 'co2_ppm', 0.4),                     # Positive correlation
            ('occupant_count', 'total_electricity_kwh', 0.2),       # Positive correlation
            ('solar_irradiance_wm2', 'illuminance_lux', 0.5),       # Positive correlation
            ('outdoor_temperature_f', 'supply_air_temp_f', 0.1),    # Weak positive correlation
        ]
        
        for var1, var2, expected_corr in expected_correlations:
            if var1 in self.dataset.columns and var2 in self.dataset.columns:
                actual_corr = self.dataset[var1].corr(self.dataset[var2])
                correlation_results[f"{var1}_vs_{var2}"] = {
                    'expected': expected_corr,
                    'actual': actual_corr,
                    'difference': abs(actual_corr - expected_corr)
                }
                
                if abs(actual_corr - expected_corr) < 0.2:  # Within 0.2 of expected
                    print(f"✓ {var1} vs {var2}: {actual_corr:.3f} (expected: {expected_corr:.3f})")
                else:
                    print(f"⚠️  {var1} vs {var2}: {actual_corr:.3f} (expected: {expected_corr:.3f})")
        
        self.validation_results['correlations'] = correlation_results
        return correlation_results
    
    def validate_seasonal_patterns(self):
        """Validate seasonal patterns in the data."""
        print("\n=== SEASONAL PATTERN VALIDATION ===")
        
        seasonal_results = {}
        
        # Add month column for analysis
        # Convert to timezone-naive for month extraction
        timestamp_naive = self.dataset['timestamp'].dt.tz_localize(None)
        self.dataset['month'] = timestamp_naive.dt.month
        
        # Check seasonal patterns
        monthly_stats = self.dataset.groupby('month').agg({
            'outdoor_temperature_f': ['mean', 'std'],
            'total_electricity_kwh': ['mean', 'std'],
            'hvac_electricity_kwh': ['mean', 'std']
        }).round(2)
        
        seasonal_results['monthly_statistics'] = monthly_stats.to_dict()
        # Convert tuple keys to strings for JSON serialization
        if 'monthly_statistics' in seasonal_results:
            monthly_stats_dict = {}
            for key, value in seasonal_results['monthly_statistics'].items():
                if isinstance(key, tuple):
                    monthly_stats_dict[f"{key[0]}_{key[1]}"] = value
                else:
                    monthly_stats_dict[str(key)] = value
            seasonal_results['monthly_statistics'] = monthly_stats_dict
        
        # Check for expected seasonal patterns
        temp_by_month = self.dataset.groupby('month')['outdoor_temperature_f'].mean()
        energy_by_month = self.dataset.groupby('month')['total_electricity_kwh'].mean()
        
        # Temperature should be lowest in winter (Dec-Feb) and highest in summer (Jun-Aug)
        winter_months = [12, 1, 2]
        summer_months = [6, 7, 8]
        
        winter_temp = temp_by_month[winter_months].mean()
        summer_temp = temp_by_month[summer_months].mean()
        
        if summer_temp > winter_temp + 10:  # At least 10°F difference
            print(f"✓ Temperature seasonal pattern: Winter {winter_temp:.1f}°F, Summer {summer_temp:.1f}°F")
        else:
            print(f"⚠️  Weak temperature seasonal pattern: Winter {winter_temp:.1f}°F, Summer {summer_temp:.1f}°F")
        
        seasonal_results['temperature_seasonality'] = {
            'winter_avg': winter_temp,
            'summer_avg': summer_temp,
            'seasonal_range': summer_temp - winter_temp
        }
        
        self.validation_results['seasonal_patterns'] = seasonal_results
        return seasonal_results
    
    def generate_quality_report(self):
        """Generate a comprehensive quality report."""
        print("\n=== GENERATING QUALITY REPORT ===")
        
        # Run all validations
        self.load_dataset()
        self.validate_data_completeness()
        self.validate_data_ranges()
        self.validate_temporal_consistency()
        self.validate_correlations()
        self.validate_seasonal_patterns()
        
        # Calculate overall quality score
        completeness_score = self.validation_results['completeness']['completeness_score']
        range_violations = len(self.validation_results['range_validation'])
        temporal_issues = (self.validation_results['temporal_consistency']['duplicate_timestamps'] + 
                          self.validation_results['temporal_consistency']['time_gaps'])
        
        # Quality score calculation (0-100)
        quality_score = completeness_score
        if range_violations > 0:
            quality_score -= range_violations * 5
        if temporal_issues > 0:
            quality_score -= temporal_issues * 0.1
        
        quality_score = max(0, min(100, quality_score))
        
        # Generate report
        report = {
            'dataset_info': {
                'file_path': self.dataset_path,
                'total_records': len(self.dataset),
                'total_columns': len(self.dataset.columns),
                'date_range': {
                    'start': str(self.dataset['timestamp'].min()),
                    'end': str(self.dataset['timestamp'].max())
                },
                'sampling_frequency': '15 minutes'
            },
            'quality_score': quality_score,
            'validation_results': self.validation_results,
            'recommendations': []
        }
        
        # Add recommendations based on validation results
        if completeness_score < 95:
            report['recommendations'].append("Consider investigating missing data sources")
        
        if range_violations > 0:
            report['recommendations'].append("Review data generation parameters for out-of-range values")
        
        if temporal_issues > 0:
            report['recommendations'].append("Check time series generation logic for gaps or duplicates")
        
        if quality_score < 80:
            report['recommendations'].append("Overall data quality needs improvement")
        elif quality_score >= 95:
            report['recommendations'].append("Excellent data quality - ready for AI model training")
        
        # Save report
        with open('iot_dataset/quality_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n=== QUALITY REPORT SUMMARY ===")
        print(f"Overall Quality Score: {quality_score:.1f}/100")
        print(f"Data Completeness: {completeness_score:.1f}%")
        print(f"Range Violations: {range_violations}")
        print(f"Temporal Issues: {temporal_issues}")
        print(f"\nRecommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print(f"\nDetailed report saved to: iot_dataset/quality_report.json")
        
        return report

def main():
    """Run data validation on the generated dataset."""
    print("=" * 60)
    print("IoT Dataset Validation and Quality Assessment")
    print("=" * 60)
    
    # Validate the generated dataset
    validator = IoTDataValidator('iot_dataset/iot_building_dataset.csv')
    report = validator.generate_quality_report()
    
    return report

if __name__ == "__main__":
    report = main()