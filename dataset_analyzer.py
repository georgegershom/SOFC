#!/usr/bin/env python3
"""
Dataset Analysis and Visualization Tools
Provides comprehensive analysis of the generated ML training dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DatasetAnalyzer:
    """Analyze and visualize the generated ML training dataset."""
    
    def __init__(self, dataset_path='ml_sintering_dataset'):
        """Initialize analyzer with dataset path."""
        self.dataset_path = Path(dataset_path)
        self.load_data()
        
    def load_data(self):
        """Load all dataset components."""
        print("Loading dataset...")
        
        # Load main dataset
        self.dataset = pd.read_csv(self.dataset_path / 'ml_training_dataset.csv')
        
        # Load validation data
        self.validation_data = pd.read_csv(self.dataset_path / 'experimental_validation_data.csv')
        
        # Load metadata
        with open(self.dataset_path / 'dataset_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load spatial data (sample a few for analysis)
        self.spatial_samples = {}
        try:
            with h5py.File(self.dataset_path / 'spatial_fields_data.h5', 'r') as f:
                # Load first 5 spatial samples
                for i, key in enumerate(list(f.keys())[:5]):
                    sample_data = {}
                    for field_name in f[key].keys():
                        sample_data[field_name] = f[key][field_name][:]
                    self.spatial_samples[key] = sample_data
        except FileNotFoundError:
            print("Spatial data file not found, skipping spatial analysis.")
            self.spatial_samples = {}
        
        print(f"Loaded dataset with {len(self.dataset)} samples")
        
    def analyze_feature_distributions(self):
        """Analyze and visualize feature distributions."""
        print("\nAnalyzing feature distributions...")
        
        # Identify feature and label columns
        feature_cols = [col for col in self.dataset.columns 
                       if col not in ['sample_id', 'stress_hotspot_count', 'max_hotspot_intensity',
                                    'avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                                    'max_delamination_prob', 'failure_risk_score']]
        
        label_cols = ['avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                     'max_delamination_prob', 'failure_risk_score']
        
        # Create distribution plots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        for i, col in enumerate(feature_cols[:12]):  # Plot first 12 features
            if i < len(axes):
                self.dataset[col].hist(bins=50, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col}\nMean: {self.dataset[col].mean():.3f}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Label distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, col in enumerate(label_cols):
            if i < len(axes):
                self.dataset[col].hist(bins=50, ax=axes[i], alpha=0.7, color='red')
                axes[i].set_title(f'{col}\nMean: {self.dataset[col].mean():.3f}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'label_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return feature_cols, label_cols
    
    def analyze_correlations(self, feature_cols, label_cols):
        """Analyze correlations between features and labels."""
        print("Analyzing correlations...")
        
        # Feature-feature correlations
        feature_corr = self.dataset[feature_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(feature_corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature-Feature Correlations')
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature-label correlations
        all_cols = feature_cols + label_cols
        full_corr = self.dataset[all_cols].corr()
        feature_label_corr = full_corr.loc[feature_cols, label_cols]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(feature_label_corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f')
        plt.title('Feature-Label Correlations')
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'feature_label_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print strongest correlations
        print("\nStrongest Feature-Label Correlations:")
        for label in label_cols:
            correlations = [(feat, abs(feature_label_corr.loc[feat, label])) 
                          for feat in feature_cols]
            correlations.sort(key=lambda x: x[1], reverse=True)
            print(f"\n{label}:")
            for feat, corr in correlations[:3]:
                print(f"  {feat}: {feature_label_corr.loc[feat, label]:.3f}")
    
    def analyze_spatial_patterns(self):
        """Analyze spatial field patterns."""
        if not self.spatial_samples:
            print("No spatial data available for analysis.")
            return
        
        print("Analyzing spatial patterns...")
        
        # Plot spatial fields for first sample
        sample_key = list(self.spatial_samples.keys())[0]
        sample_data = self.spatial_samples[sample_key]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        fields_to_plot = ['temperature_field', 'porosity_field', 'stress_field',
                         'hotspots', 'crack_risk', 'delamination_prob']
        
        for i, field in enumerate(fields_to_plot):
            if field in sample_data:
                row, col = i // 3, i % 3
                im = axes[row, col].imshow(sample_data[field], cmap='viridis', origin='lower')
                axes[row, col].set_title(f'{field.replace("_", " ").title()}')
                plt.colorbar(im, ax=axes[row, col])
        
        plt.suptitle(f'Spatial Fields - {sample_key}')
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'spatial_patterns.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def perform_pca_analysis(self, feature_cols):
        """Perform PCA analysis on features."""
        print("Performing PCA analysis...")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(self.dataset[feature_cols])
        
        # Perform PCA
        pca = PCA()
        pca_features = pca.fit_transform(features_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_, 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(range(1, len(cumsum) + 1), cumsum, 'ro-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.grid(True)
        plt.axhline(y=0.95, color='k', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print PCA summary
        print(f"\nPCA Summary:")
        print(f"Components needed for 95% variance: {np.argmax(cumsum >= 0.95) + 1}")
        print(f"Components needed for 99% variance: {np.argmax(cumsum >= 0.99) + 1}")
        
        return pca, scaler
    
    def analyze_validation_data(self):
        """Analyze experimental validation data."""
        print("Analyzing validation data...")
        
        # Basic statistics
        print("\nValidation Data Statistics:")
        numeric_cols = self.validation_data.select_dtypes(include=[np.number]).columns
        print(self.validation_data[numeric_cols].describe())
        
        # Plot validation data distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        plot_cols = ['sintering_temp', 'cooling_rate', 'dic_strain_xx', 
                    'dic_strain_yy', 'xrd_residual_stress']
        
        for i, col in enumerate(plot_cols):
            if i < len(axes) and col in self.validation_data.columns:
                self.validation_data[col].hist(bins=20, ax=axes[i], alpha=0.7, color='green')
                axes[i].set_title(f'{col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'validation_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_data_quality_report(self):
        """Generate comprehensive data quality report."""
        print("Generating data quality report...")
        
        report = {
            'dataset_info': {
                'total_samples': len(self.dataset),
                'total_features': len([col for col in self.dataset.columns if col != 'sample_id']),
                'generation_date': self.metadata.get('generation_date', 'Unknown'),
            },
            'data_quality': {
                'missing_values': self.dataset.isnull().sum().sum(),
                'duplicate_rows': self.dataset.duplicated().sum(),
                'infinite_values': np.isinf(self.dataset.select_dtypes(include=[np.number])).sum().sum(),
            },
            'feature_ranges': {},
            'label_ranges': {},
        }
        
        # Feature ranges
        feature_cols = [col for col in self.dataset.columns 
                       if col not in ['sample_id', 'stress_hotspot_count', 'max_hotspot_intensity',
                                    'avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                                    'max_delamination_prob', 'failure_risk_score']]
        
        for col in feature_cols:
            report['feature_ranges'][col] = {
                'min': float(self.dataset[col].min()),
                'max': float(self.dataset[col].max()),
                'mean': float(self.dataset[col].mean()),
                'std': float(self.dataset[col].std()),
            }
        
        # Label ranges
        label_cols = ['avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                     'max_delamination_prob', 'failure_risk_score']
        
        for col in label_cols:
            report['label_ranges'][col] = {
                'min': float(self.dataset[col].min()),
                'max': float(self.dataset[col].max()),
                'mean': float(self.dataset[col].mean()),
                'std': float(self.dataset[col].std()),
            }
        
        # Save report
        with open(self.dataset_path / 'data_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\nData Quality Summary:")
        print(f"Total samples: {report['dataset_info']['total_samples']}")
        print(f"Total features: {report['dataset_info']['total_features']}")
        print(f"Missing values: {report['data_quality']['missing_values']}")
        print(f"Duplicate rows: {report['data_quality']['duplicate_rows']}")
        print(f"Infinite values: {report['data_quality']['infinite_values']}")
        
        return report
    
    def create_train_test_splits(self, test_size=0.2, val_size=0.1):
        """Create train/validation/test splits and save them."""
        print(f"Creating train/validation/test splits...")
        
        # Separate features and labels
        feature_cols = [col for col in self.dataset.columns 
                       if col not in ['sample_id', 'stress_hotspot_count', 'max_hotspot_intensity',
                                    'avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                                    'max_delamination_prob', 'failure_risk_score']]
        
        label_cols = ['avg_crack_risk', 'max_crack_risk', 'avg_delamination_prob',
                     'max_delamination_prob', 'failure_risk_score']
        
        X = self.dataset[feature_cols]
        y = self.dataset[label_cols]
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=None
        )
        
        # Save splits
        splits = {
            'train': {'features': X_train, 'labels': y_train},
            'val': {'features': X_val, 'labels': y_val},
            'test': {'features': X_test, 'labels': y_test},
        }
        
        for split_name, split_data in splits.items():
            # Combine features and labels
            split_df = pd.concat([split_data['features'], split_data['labels']], axis=1)
            split_df.to_csv(self.dataset_path / f'{split_name}_split.csv', index=False)
        
        print(f"Train set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        return splits
    
    def run_complete_analysis(self):
        """Run complete dataset analysis."""
        print("="*60)
        print("COMPREHENSIVE DATASET ANALYSIS")
        print("="*60)
        
        # Feature and label analysis
        feature_cols, label_cols = self.analyze_feature_distributions()
        
        # Correlation analysis
        self.analyze_correlations(feature_cols, label_cols)
        
        # Spatial pattern analysis
        self.analyze_spatial_patterns()
        
        # PCA analysis
        pca, scaler = self.perform_pca_analysis(feature_cols)
        
        # Validation data analysis
        self.analyze_validation_data()
        
        # Data quality report
        quality_report = self.generate_data_quality_report()
        
        # Create train/test splits
        splits = self.create_train_test_splits()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print("Generated files:")
        print("  - feature_distributions.png")
        print("  - label_distributions.png")
        print("  - feature_correlations.png")
        print("  - feature_label_correlations.png")
        print("  - spatial_patterns.png")
        print("  - pca_analysis.png")
        print("  - validation_data_analysis.png")
        print("  - data_quality_report.json")
        print("  - train_split.csv, val_split.csv, test_split.csv")


def main():
    """Main function to run dataset analysis."""
    analyzer = DatasetAnalyzer('ml_sintering_dataset')
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()