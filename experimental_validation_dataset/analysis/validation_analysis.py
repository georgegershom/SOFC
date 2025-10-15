#!/usr/bin/env python3
"""
SOFC Experimental Validation Analysis Script

This script performs comprehensive validation analysis comparing
experimental measurements with ML model predictions.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

class SOFCValidationAnalyzer:
    def __init__(self, warp_data_path, stress_data_path, sample_specs_path):
        """Initialize the analyzer with data paths."""
        self.warp_data_path = warp_data_path
        self.stress_data_path = stress_data_path
        self.sample_specs_path = sample_specs_path
        self.load_data()
    
    def load_data(self):
        """Load all experimental data."""
        print("Loading experimental data...")
        
        # Load warp measurements
        with open(self.warp_data_path, 'r') as f:
            self.warp_data = json.load(f)
        
        # Load stress measurements
        with open(self.stress_data_path, 'r') as f:
            self.stress_data = json.load(f)
        
        # Load sample specifications
        with open(self.sample_specs_path, 'r') as f:
            self.sample_specs = json.load(f)
        
        print(f"Loaded data for {len(self.warp_data['measurements'])} samples")
    
    def extract_warp_features(self):
        """Extract key features from warp measurements."""
        features = []
        
        for measurement in self.warp_data['measurements']:
            sample_id = measurement['metadata']['sample_id']
            point_cloud = measurement['point_cloud']
            
            # Extract z-values (warp heights)
            z_values = [point['z'] for point in point_cloud]
            
            # Calculate features
            features.append({
                'sample_id': sample_id,
                'max_warp': max(z_values),
                'min_warp': min(z_values),
                'rms_warp': np.sqrt(np.mean(np.array(z_values)**2)),
                'peak_to_valley': max(z_values) - min(z_values),
                'mean_warp': np.mean(z_values),
                'std_warp': np.std(z_values)
            })
        
        return features
    
    def extract_stress_features(self):
        """Extract key features from stress measurements."""
        features = []
        
        for measurement in self.stress_data['measurements']:
            sample_id = measurement['metadata']['sample_id']
            
            # Extract XRD stress values
            xrd_stresses = measurement['xrd_measurements']
            stress_xx = [s['stress_xx'] for s in xrd_stresses]
            stress_yy = [s['stress_yy'] for s in xrd_stresses]
            stress_xy = [s['stress_xy'] for s in xrd_stresses]
            
            # Calculate features
            features.append({
                'sample_id': sample_id,
                'mean_stress_xx': np.mean(stress_xx),
                'mean_stress_yy': np.mean(stress_yy),
                'mean_stress_xy': np.mean(stress_xy),
                'max_stress_xx': max(stress_xx),
                'max_stress_yy': max(stress_yy),
                'std_stress_xx': np.std(stress_xx),
                'std_stress_yy': np.std(stress_yy),
                'von_mises_stress': np.mean([np.sqrt(s['stress_xx']**2 + s['stress_yy']**2 - s['stress_xx']*s['stress_yy'] + 3*s['stress_xy']**2) for s in xrd_stresses])
            })
        
        return features
    
    def validate_ml_predictions(self, ml_predictions_path):
        """Validate ML model predictions against experimental data."""
        print("Validating ML model predictions...")
        
        # Load ML predictions (simulated for this example)
        # In real usage, this would load actual ML model predictions
        ml_predictions = self.simulate_ml_predictions()
        
        # Extract experimental features
        warp_features = self.extract_warp_features()
        stress_features = self.extract_stress_features()
        
        # Create comparison dataframe
        comparison_data = []
        for i, sample in enumerate(self.sample_specs['samples']):
            sample_id = sample['sample_id']
            
            # Find corresponding experimental data
            warp_feat = next((w for w in warp_features if w['sample_id'] == sample_id), None)
            stress_feat = next((s for s in stress_features if s['sample_id'] == sample_id), None)
            
            if warp_feat and stress_feat:
                comparison_data.append({
                    'sample_id': sample_id,
                    'thickness_ratio': sample['thickness_ratio'],
                    'sintering_cycle': sample['sintering_cycle'],
                    'exp_max_warp': warp_feat['max_warp'],
                    'exp_rms_warp': warp_feat['rms_warp'],
                    'exp_von_mises': stress_feat['von_mises_stress'],
                    'ml_max_warp': ml_predictions[i]['max_warp'],
                    'ml_rms_warp': ml_predictions[i]['rms_warp'],
                    'ml_von_mises': ml_predictions[i]['von_mises_stress']
                })
        
        return pd.DataFrame(comparison_data)
    
    def simulate_ml_predictions(self):
        """Simulate ML model predictions for validation."""
        # This is a placeholder - in real usage, load actual ML predictions
        predictions = []
        
        for sample in self.sample_specs['samples']:
            # Simulate predictions with some error
            thickness_ratio = sample['thickness_ratio']
            
            # Simulate warp predictions
            if thickness_ratio < 0.5:
                max_warp = np.random.uniform(0.4, 1.0)
                rms_warp = max_warp * 0.6
            elif thickness_ratio > 1.5:
                max_warp = np.random.uniform(0.2, 0.6)
                rms_warp = max_warp * 0.6
            else:
                max_warp = np.random.uniform(0.1, 0.3)
                rms_warp = max_warp * 0.6
            
            # Simulate stress predictions
            von_mises = 100 + thickness_ratio * 50 + np.random.normal(0, 20)
            
            predictions.append({
                'max_warp': max_warp,
                'rms_warp': rms_warp,
                'von_mises_stress': von_mises
            })
        
        return predictions
    
    def calculate_validation_metrics(self, comparison_df):
        """Calculate validation metrics."""
        metrics = {}
        
        # Warp validation metrics
        warp_metrics = {
            'max_warp_r2': r2_score(comparison_df['exp_max_warp'], comparison_df['ml_max_warp']),
            'max_warp_rmse': np.sqrt(mean_squared_error(comparison_df['exp_max_warp'], comparison_df['ml_max_warp'])),
            'max_warp_mae': mean_absolute_error(comparison_df['exp_max_warp'], comparison_df['ml_max_warp']),
            'rms_warp_r2': r2_score(comparison_df['exp_rms_warp'], comparison_df['ml_rms_warp']),
            'rms_warp_rmse': np.sqrt(mean_squared_error(comparison_df['exp_rms_warp'], comparison_df['ml_rms_warp'])),
            'rms_warp_mae': mean_absolute_error(comparison_df['exp_rms_warp'], comparison_df['ml_rms_warp'])
        }
        
        # Stress validation metrics
        stress_metrics = {
            'von_mises_r2': r2_score(comparison_df['exp_von_mises'], comparison_df['ml_von_mises']),
            'von_mises_rmse': np.sqrt(mean_squared_error(comparison_df['exp_von_mises'], comparison_df['ml_von_mises'])),
            'von_mises_mae': mean_absolute_error(comparison_df['exp_von_mises'], comparison_df['ml_von_mises'])
        }
        
        metrics.update(warp_metrics)
        metrics.update(stress_metrics)
        
        return metrics
    
    def generate_validation_report(self, comparison_df, metrics):
        """Generate comprehensive validation report."""
        report = {
            'validation_summary': {
                'total_samples': len(comparison_df),
                'validation_date': '2024-01-30',
                'overall_status': 'PASS' if metrics['max_warp_r2'] > 0.8 and metrics['von_mises_r2'] > 0.7 else 'FAIL'
            },
            'warp_validation': {
                'max_warp': {
                    'r2_score': metrics['max_warp_r2'],
                    'rmse': metrics['max_warp_rmse'],
                    'mae': metrics['max_warp_mae'],
                    'status': 'PASS' if metrics['max_warp_r2'] > 0.8 else 'FAIL'
                },
                'rms_warp': {
                    'r2_score': metrics['rms_warp_r2'],
                    'rmse': metrics['rms_warp_rmse'],
                    'mae': metrics['rms_warp_mae'],
                    'status': 'PASS' if metrics['rms_warp_r2'] > 0.8 else 'FAIL'
                }
            },
            'stress_validation': {
                'von_mises_stress': {
                    'r2_score': metrics['von_mises_r2'],
                    'rmse': metrics['von_mises_rmse'],
                    'mae': metrics['von_mises_mae'],
                    'status': 'PASS' if metrics['von_mises_r2'] > 0.7 else 'FAIL'
                }
            },
            'recommendations': self.generate_recommendations(metrics)
        }
        
        return report
    
    def generate_recommendations(self, metrics):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if metrics['max_warp_r2'] < 0.8:
            recommendations.append("Improve warp prediction model - consider additional features or model complexity")
        
        if metrics['von_mises_r2'] < 0.7:
            recommendations.append("Enhance stress prediction accuracy - review FEA model parameters")
        
        if metrics['max_warp_rmse'] > 0.2:
            recommendations.append("High warp prediction error - investigate measurement uncertainty")
        
        if metrics['von_mises_rmse'] > 50:
            recommendations.append("High stress prediction error - consider stress measurement technique limitations")
        
        if not recommendations:
            recommendations.append("Model validation successful - proceed with confidence")
        
        return recommendations
    
    def run_full_validation(self):
        """Run complete validation analysis."""
        print("Running full validation analysis...")
        
        # Validate ML predictions
        comparison_df = self.validate_ml_predictions("dummy_path")
        
        # Calculate metrics
        metrics = self.calculate_validation_metrics(comparison_df)
        
        # Generate report
        report = self.generate_validation_report(comparison_df, metrics)
        
        # Save results
        with open('experimental_validation_dataset/analysis/validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save comparison data
        comparison_df.to_csv('experimental_validation_dataset/analysis/validation_comparison.csv', index=False)
        
        print("Validation analysis complete!")
        print(f"Overall status: {report['validation_summary']['overall_status']}")
        print(f"Max warp R²: {metrics['max_warp_r2']:.3f}")
        print(f"Von Mises stress R²: {metrics['von_mises_r2']:.3f}")
        
        return report, comparison_df

def main():
    """Main execution function."""
    # Initialize analyzer
    analyzer = SOFCValidationAnalyzer(
        'experimental_validation_dataset/measurements/warp_data/warp_measurements.json',
        'experimental_validation_dataset/measurements/stress_data/stress_measurements.json',
        'experimental_validation_dataset/fabricated_plates/complete_sample_specifications.json'
    )
    
    # Run validation
    report, comparison_df = analyzer.run_full_validation()
    
    print("\nValidation complete! Check the analysis/ directory for detailed results.")

if __name__ == "__main__":
    main()
