#!/usr/bin/env python3
"""
SOFC Dataset Validation and Summary Tool
========================================

This module validates the generated SOFC dataset and provides summary statistics
without requiring matplotlib (suitable for background execution).

Author: Generated for PhD Thesis - Multi-Fidelity Digital Twin for SOFCs
Date: 2025-10-15
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any

class SOFCDatasetValidator:
    """
    Validator for SOFC multi-fidelity datasets.
    """
    
    def __init__(self, dataset_dir: str = "sofc_dataset"):
        """Initialize the validator with dataset directory."""
        self.dataset_dir = Path(dataset_dir)
        self.datasets = {}
        self.microstructural_data = {}
        self.transient_profiles = {}
        self.metadata = {}
        
        self._load_datasets()
    
    def _load_datasets(self):
        """Load all dataset files."""
        try:
            # Load parameter datasets
            for fidelity in ['lf', 'mf', 'hf']:
                file_path = self.dataset_dir / f"sofc_parameters_{fidelity}_fidelity.csv"
                if file_path.exists():
                    self.datasets[fidelity.upper()] = pd.read_csv(file_path)
            
            # Load combined dataset
            combined_path = self.dataset_dir / "sofc_parameters_combined.csv"
            if combined_path.exists():
                self.datasets['combined'] = pd.read_csv(combined_path)
            
            # Load microstructural data
            micro_path = self.dataset_dir / "sofc_microstructural_data.json"
            if micro_path.exists():
                with open(micro_path, 'r') as f:
                    self.microstructural_data = json.load(f)
            
            # Load transient profiles
            transient_path = self.dataset_dir / "sofc_transient_profiles.json"
            if transient_path.exists():
                with open(transient_path, 'r') as f:
                    self.transient_profiles = json.load(f)
            
            # Load metadata
            metadata_path = self.dataset_dir / "dataset_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            print("✅ Successfully loaded all dataset files")
            
        except Exception as e:
            print(f"❌ Error loading datasets: {e}")
    
    def validate_parameter_ranges(self):
        """Validate that all parameters are within expected ranges."""
        print("\n🔍 Validating Parameter Ranges...")
        print("=" * 60)
        
        validation_results = {}
        
        # Define expected ranges for key parameters
        expected_ranges = {
            'fuel_utilization': (0.6, 0.95),
            'oxidant_utilization': (0.15, 0.4),
            'current_density': (0.1, 1.5),
            'temperature': (973, 1273),
            'pressure': (1.0, 10.0),
            'h2_fraction': (0.3, 0.97),
        }
        
        for fidelity, df in self.datasets.items():
            if fidelity == 'combined':
                continue
                
            print(f"\n📊 Validating {fidelity} fidelity dataset:")
            fidelity_results = {}
            
            for param, (min_val, max_val) in expected_ranges.items():
                if param in df.columns:
                    actual_min = df[param].min()
                    actual_max = df[param].max()
                    
                    within_range = (actual_min >= min_val) and (actual_max <= max_val)
                    fidelity_results[param] = {
                        'expected_range': (min_val, max_val),
                        'actual_range': (actual_min, actual_max),
                        'valid': within_range
                    }
                    
                    status = "✅" if within_range else "❌"
                    print(f"  {status} {param}: [{actual_min:.3f}, {actual_max:.3f}] (expected: [{min_val}, {max_val}])")
            
            validation_results[fidelity] = fidelity_results
        
        return validation_results
    
    def analyze_dataset_statistics(self):
        """Generate comprehensive dataset statistics."""
        print("\n📊 Dataset Statistics Analysis")
        print("=" * 60)
        
        # Basic statistics
        print(f"\n📈 Basic Statistics:")
        print(f"{'Fidelity':<10} {'Samples':<8} {'Parameters':<12} {'Memory (KB)':<12}")
        print("-" * 45)
        
        total_samples = 0
        for fidelity, df in self.datasets.items():
            if fidelity != 'combined':
                memory_kb = df.memory_usage(deep=True).sum() / 1024
                print(f"{fidelity:<10} {len(df):<8} {len(df.columns):<12} {memory_kb:.1f}")
                total_samples += len(df)
        
        print(f"{'TOTAL':<10} {total_samples:<8} {'-':<12} {'-'}")
        
        # Parameter distribution analysis
        if 'HF' in self.datasets:
            df = self.datasets['HF']
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            print(f"\n📊 Parameter Statistics (HF Dataset):")
            print(f"{'Parameter':<25} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
            print("-" * 65)
            
            # Show statistics for key parameters
            key_params = ['fuel_utilization', 'current_density', 'temperature', 'pressure']
            for param in key_params:
                if param in numeric_cols:
                    mean_val = df[param].mean()
                    std_val = df[param].std()
                    min_val = df[param].min()
                    max_val = df[param].max()
                    print(f"{param:<25} {mean_val:<10.3f} {std_val:<10.3f} {min_val:<10.3f} {max_val:<10.3f}")
    
    def analyze_microstructural_data(self):
        """Analyze microstructural dataset."""
        print(f"\n🔬 Microstructural Data Analysis")
        print("=" * 60)
        
        if not self.microstructural_data:
            print("❌ No microstructural data available")
            return
        
        # Extract statistics
        voxel_sizes = []
        ni_fractions = []
        ysz_fractions = []
        pore_fractions = []
        
        for sample_data in self.microstructural_data.values():
            voxel_sizes.append(sample_data['voxel_size_nm'])
            ni_fractions.append(sample_data['phase_fractions']['ni'])
            ysz_fractions.append(sample_data['phase_fractions']['ysz'])
            pore_fractions.append(sample_data['phase_fractions']['pore'])
        
        print(f"📊 Microstructural Statistics:")
        print(f"- Total samples: {len(self.microstructural_data)}")
        print(f"- Voxel size range: {min(voxel_sizes):.1f} - {max(voxel_sizes):.1f} nm")
        print(f"- Ni fraction: {min(ni_fractions):.3f} - {max(ni_fractions):.3f}")
        print(f"- YSZ fraction: {min(ysz_fractions):.3f} - {max(ysz_fractions):.3f}")
        print(f"- Pore fraction: {min(pore_fractions):.3f} - {max(pore_fractions):.3f}")
        
        # Check phase fraction consistency
        total_fractions = [ni + ysz + pore for ni, ysz, pore in 
                          zip(ni_fractions, ysz_fractions, pore_fractions)]
        
        fraction_errors = [abs(total - 1.0) for total in total_fractions]
        max_error = max(fraction_errors)
        
        if max_error < 0.1:
            print(f"✅ Phase fractions are consistent (max error: {max_error:.4f})")
        else:
            print(f"❌ Phase fractions have issues (max error: {max_error:.4f})")
    
    def analyze_transient_profiles(self):
        """Analyze transient profile data."""
        print(f"\n⏱️  Transient Profiles Analysis")
        print("=" * 60)
        
        if not self.transient_profiles:
            print("❌ No transient profiles available")
            return
        
        # Group by type
        profile_types = {}
        for key, profile in self.transient_profiles.items():
            profile_type = profile['type']
            if profile_type not in profile_types:
                profile_types[profile_type] = []
            profile_types[profile_type].append(profile)
        
        print(f"📊 Profile Statistics:")
        for profile_type, profiles in profile_types.items():
            print(f"- {profile_type.replace('_', ' ').title()}: {len(profiles)} profiles")
            
            # Analyze temperature and current ranges
            temp_ranges = []
            current_ranges = []
            
            for profile in profiles[:5]:  # Sample first 5
                temps = np.array(profile['temperature'])
                currents = np.array(profile['current_density'])
                temp_ranges.append((temps.min(), temps.max()))
                current_ranges.append((currents.min(), currents.max()))
            
            if temp_ranges:
                temp_min = min(r[0] for r in temp_ranges)
                temp_max = max(r[1] for r in temp_ranges)
                current_min = min(r[0] for r in current_ranges)
                current_max = max(r[1] for r in current_ranges)
                
                print(f"  Temperature range: {temp_min:.1f} - {temp_max:.1f} K")
                print(f"  Current range: {current_min:.3f} - {current_max:.3f} A/cm²")
    
    def check_data_quality(self):
        """Perform comprehensive data quality checks."""
        print(f"\n🔍 Data Quality Assessment")
        print("=" * 60)
        
        issues = []
        
        # Check for missing values
        for fidelity, df in self.datasets.items():
            if fidelity == 'combined':
                continue
            
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                issues.append(f"{fidelity} dataset has {missing_count} missing values")
            else:
                print(f"✅ {fidelity} dataset: No missing values")
        
        # Check for duplicate rows
        for fidelity, df in self.datasets.items():
            if fidelity == 'combined':
                continue
            
            # Check duplicates in numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            duplicate_count = df[numeric_cols].duplicated().sum()
            
            if duplicate_count > 0:
                issues.append(f"{fidelity} dataset has {duplicate_count} duplicate rows")
            else:
                print(f"✅ {fidelity} dataset: No duplicate rows")
        
        # Check parameter correlations
        if 'HF' in self.datasets:
            df = self.datasets['HF']
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Check for highly correlated parameters (>0.95)
            corr_matrix = df[numeric_cols].corr()
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.95:
                        param1 = corr_matrix.columns[i]
                        param2 = corr_matrix.columns[j]
                        high_corr_pairs.append((param1, param2, corr_val))
            
            if high_corr_pairs:
                print(f"⚠️  Found {len(high_corr_pairs)} highly correlated parameter pairs (>0.95)")
                for param1, param2, corr in high_corr_pairs[:5]:  # Show first 5
                    print(f"   {param1} - {param2}: {corr:.3f}")
            else:
                print(f"✅ No highly correlated parameters found")
        
        # Summary
        if issues:
            print(f"\n❌ Found {len(issues)} data quality issues:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print(f"\n✅ All data quality checks passed!")
        
        return issues
    
    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*80)
        print("🔬 SOFC DATASET VALIDATION REPORT")
        print("="*80)
        
        # Basic dataset info
        print(f"\n📁 Dataset Location: {self.dataset_dir}")
        print(f"📅 Generated: {self.metadata.get('created_date', 'Unknown')}")
        print(f"📊 Total Datasets: {len(self.datasets)}")
        print(f"🔬 Microstructural Samples: {len(self.microstructural_data)}")
        print(f"⏱️  Transient Profiles: {len(self.transient_profiles)}")
        
        # Run all validations
        self.analyze_dataset_statistics()
        validation_results = self.validate_parameter_ranges()
        self.analyze_microstructural_data()
        self.analyze_transient_profiles()
        quality_issues = self.check_data_quality()
        
        # Final assessment
        print(f"\n" + "="*60)
        print("🎯 VALIDATION SUMMARY")
        print("="*60)
        
        total_validations = sum(len(results) for results in validation_results.values())
        passed_validations = sum(
            sum(1 for result in results.values() if result['valid'])
            for results in validation_results.values()
        )
        
        print(f"✅ Parameter validations passed: {passed_validations}/{total_validations}")
        print(f"❌ Data quality issues: {len(quality_issues)}")
        
        if len(quality_issues) == 0 and passed_validations == total_validations:
            print(f"\n🎉 DATASET VALIDATION SUCCESSFUL!")
            print(f"   All checks passed - dataset is ready for use!")
        else:
            print(f"\n⚠️  DATASET VALIDATION COMPLETED WITH WARNINGS")
            print(f"   Please review the issues above before using the dataset.")
        
        return validation_results, quality_issues


if __name__ == "__main__":
    # Run validation
    validator = SOFCDatasetValidator()
    validation_results, quality_issues = validator.generate_validation_report()