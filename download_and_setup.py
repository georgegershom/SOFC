#!/usr/bin/env python3
"""
SOFC Dataset Download and Setup Script
=====================================

This script provides easy access to the complete SOFC "In-The-Wild" dataset
and sets up the environment for ML model development.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import zipfile
import shutil

class SOFCDatasetDownloader:
    """
    Easy-to-use interface for accessing the SOFC dataset
    """
    
    def __init__(self, dataset_path='sofc_in_the_wild_dataset'):
        self.dataset_path = Path(dataset_path)
        self.verify_dataset()
    
    def verify_dataset(self):
        """Verify that the dataset is complete"""
        required_files = [
            'manufacturing_parameters.csv',
            'quality_analysis.csv', 
            'measurement_summary.csv',
            'dataset_statistics.json',
            'README.md'
        ]
        
        required_dirs = [
            'plates',
            'analysis_figures',
            'validation_report',
            'ml_analysis'
        ]
        
        print("Verifying dataset completeness...")
        
        missing_files = []
        for file in required_files:
            if not (self.dataset_path / file).exists():
                missing_files.append(file)
        
        missing_dirs = []
        for dir_name in required_dirs:
            if not (self.dataset_path / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_files or missing_dirs:
            print("❌ Dataset incomplete!")
            if missing_files:
                print(f"Missing files: {missing_files}")
            if missing_dirs:
                print(f"Missing directories: {missing_dirs}")
            return False
        
        # Count plates
        plates_dir = self.dataset_path / 'plates'
        n_plates = len(list(plates_dir.glob('*.json')))
        
        print(f"✅ Dataset verified: {n_plates} plates found")
        return True
    
    def load_summary_data(self):
        """Load all summary CSV files"""
        print("Loading summary data...")
        
        data = {}
        data['manufacturing'] = pd.read_csv(self.dataset_path / 'manufacturing_parameters.csv')
        data['quality'] = pd.read_csv(self.dataset_path / 'quality_analysis.csv')
        data['measurements'] = pd.read_csv(self.dataset_path / 'measurement_summary.csv')
        
        # Load statistics
        with open(self.dataset_path / 'dataset_statistics.json', 'r') as f:
            data['statistics'] = json.load(f)
        
        print(f"✅ Loaded summary data for {len(data['manufacturing'])} plates")
        return data
    
    def load_plate_data(self, plate_ids=None, max_plates=None):
        """
        Load individual plate data
        
        Args:
            plate_ids: List of specific plate IDs to load
            max_plates: Maximum number of plates to load (for memory management)
        """
        plates_dir = self.dataset_path / 'plates'
        
        if plate_ids is None:
            plate_files = list(plates_dir.glob('*.json'))
            if max_plates:
                plate_files = plate_files[:max_plates]
        else:
            plate_files = [plates_dir / f'{plate_id}.json' for plate_id in plate_ids]
        
        print(f"Loading {len(plate_files)} individual plates...")
        
        plates_data = {}
        for plate_file in plate_files:
            plate_id = plate_file.stem
            with open(plate_file, 'r') as f:
                plates_data[plate_id] = json.load(f)
        
        print(f"✅ Loaded {len(plates_data)} individual plates")
        return plates_data
    
    def load_ml_features(self):
        """Load pre-processed ML features"""
        ml_dir = self.dataset_path / 'ml_analysis'
        
        if not ml_dir.exists():
            print("❌ ML analysis not found. Run ml_analysis_tools.py first.")
            return None
        
        print("Loading ML features...")
        
        features = pd.read_csv(ml_dir / 'features.csv')
        targets = pd.read_csv(ml_dir / 'targets.csv')
        metadata = pd.read_csv(ml_dir / 'metadata.csv')
        
        print(f"✅ Loaded ML features: {len(features)} samples, {len(features.columns)-1} features")
        return features, targets, metadata
    
    def get_sample_plates(self, n_samples=10, criteria='random'):
        """
        Get sample plates for quick exploration
        
        Args:
            n_samples: Number of samples to return
            criteria: 'random', 'high_risk', 'low_risk', 'diverse'
        """
        summary_data = self.load_summary_data()
        
        if criteria == 'random':
            sample_ids = summary_data['quality']['plate_id'].sample(n_samples).tolist()
        elif criteria == 'high_risk':
            high_risk = summary_data['quality'].nlargest(n_samples, 'overall_failure_risk')
            sample_ids = high_risk['plate_id'].tolist()
        elif criteria == 'low_risk':
            low_risk = summary_data['quality'].nsmallest(n_samples, 'overall_failure_risk')
            sample_ids = low_risk['plate_id'].tolist()
        elif criteria == 'diverse':
            # Select plates with diverse manufacturing conditions
            mfg_data = summary_data['manufacturing']
            
            # Normalize manufacturing parameters
            params = ['sintering_temp', 'sintering_time', 'cooling_rate', 'green_density', 'humidity']
            normalized = mfg_data[params].apply(lambda x: (x - x.mean()) / x.std())
            
            # Calculate diversity score (distance from mean)
            diversity_scores = np.sqrt((normalized ** 2).sum(axis=1))
            diverse_plates = mfg_data.loc[diversity_scores.nlargest(n_samples).index]
            sample_ids = diverse_plates['plate_id'].tolist()
        
        return self.load_plate_data(sample_ids)
    
    def create_quick_visualization(self, output_dir='quick_viz'):
        """Create quick visualization of the dataset"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print("Creating quick visualizations...")
        
        # Load data
        summary_data = self.load_summary_data()
        sample_plates = self.get_sample_plates(6, 'diverse')
        
        # 1. Dataset overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Manufacturing parameter distributions
        mfg_data = summary_data['manufacturing']
        axes[0,0].hist(mfg_data['sintering_temp'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,0].set_title('Sintering Temperature Distribution')
        axes[0,0].set_xlabel('Temperature (°C)')
        axes[0,0].set_ylabel('Count')
        
        # Failure risk distribution
        quality_data = summary_data['quality']
        axes[0,1].hist(quality_data['overall_failure_risk'], bins=30, alpha=0.7, edgecolor='black')
        axes[0,1].set_title('Failure Risk Distribution')
        axes[0,1].set_xlabel('Failure Risk')
        axes[0,1].set_ylabel('Count')
        
        # Displacement vs failure risk
        merged = mfg_data.merge(quality_data, on='plate_id')
        merged = merged.merge(summary_data['measurements'], on='plate_id')
        
        axes[1,0].scatter(merged['rms_displacement_um'], merged['overall_failure_risk'], alpha=0.6)
        axes[1,0].set_xlabel('RMS Displacement (μm)')
        axes[1,0].set_ylabel('Failure Risk')
        axes[1,0].set_title('Displacement vs Failure Risk')
        
        # Production timeline
        mfg_data['production_date'] = pd.to_datetime(mfg_data['production_date'], format='ISO8601')
        axes[1,1].plot(mfg_data['production_date'], mfg_data['sintering_temp'], 'b-', alpha=0.7)
        axes[1,1].set_title('Temperature Drift Over Time')
        axes[1,1].set_xlabel('Production Date')
        axes[1,1].set_ylabel('Sintering Temperature (°C)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample displacement fields
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (plate_id, plate_data) in enumerate(list(sample_plates.items())[:6]):
            displacement = np.array(plate_data['measurements']['displacement_um'])
            
            im = axes[i].imshow(displacement, cmap='RdBu_r', 
                               extent=[0, 150, 0, 150])  # 150mm plate
            axes[i].set_title(f'{plate_id}\nFailure Risk: {plate_data["failure_analysis"]["overall_failure_risk"]:.2f}')
            axes[i].set_xlabel('X (mm)')
            axes[i].set_ylabel('Y (mm)')
            plt.colorbar(im, ax=axes[i], label='Displacement (μm)')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'sample_displacement_fields.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Quick visualizations saved to {output_dir}")
    
    def export_for_ml(self, output_file='sofc_ml_ready.npz'):
        """Export dataset in ML-ready format"""
        print("Exporting ML-ready dataset...")
        
        features, targets, metadata = self.load_ml_features()
        
        if features is None:
            return None
        
        # Prepare arrays
        X = features.drop('plate_id', axis=1).values
        y_max_stress = targets['stress_max'].values
        y_edge_stress = targets['max_edge_stress'].values
        y_concentration = targets['stress_concentration_factor'].values
        
        # Save as compressed numpy format
        np.savez_compressed(
            output_file,
            X=X,
            y_max_stress=y_max_stress,
            y_edge_stress=y_edge_stress,
            y_concentration=y_concentration,
            feature_names=features.drop('plate_id', axis=1).columns.tolist(),
            plate_ids=features['plate_id'].values,
            production_dates=metadata['production_date'].values
        )
        
        print(f"✅ ML-ready dataset exported to {output_file}")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {len(features.columns)-1}")
        print(f"   Samples: {len(X)}")
        
        return output_file
    
    def create_dataset_package(self, package_name='sofc_in_the_wild_complete'):
        """Create a complete dataset package"""
        print(f"Creating complete dataset package: {package_name}")
        
        # Create package directory
        package_dir = Path(package_name)
        if package_dir.exists():
            shutil.rmtree(package_dir)
        package_dir.mkdir()
        
        # Copy main dataset
        shutil.copytree(self.dataset_path, package_dir / 'dataset')
        
        # Create quick start files
        self.create_quick_visualization(package_dir / 'quick_viz')
        self.export_for_ml(package_dir / 'sofc_ml_ready.npz')
        
        # Create quick start script
        quick_start_script = f'''#!/usr/bin/env python3
"""
SOFC Dataset Quick Start
========================

Quick start script for the SOFC "In-The-Wild" dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def load_ml_data():
    """Load ML-ready data"""
    data = np.load('sofc_ml_ready.npz')
    return data

def load_summary():
    """Load summary data"""
    manufacturing = pd.read_csv('dataset/manufacturing_parameters.csv')
    quality = pd.read_csv('dataset/quality_analysis.csv')
    measurements = pd.read_csv('dataset/measurement_summary.csv')
    return manufacturing, quality, measurements

def quick_analysis():
    """Run quick analysis"""
    print("SOFC Dataset Quick Analysis")
    print("=" * 30)
    
    # Load data
    ml_data = load_ml_data()
    manufacturing, quality, measurements = load_summary()
    
    print(f"Dataset size: {{ml_data['X'].shape[0]}} plates")
    print(f"Features: {{ml_data['X'].shape[1]}}")
    print(f"Date range: {{manufacturing['production_date'].min()}} to {{manufacturing['production_date'].max()}}")
    
    # Basic statistics
    print("\\nManufacturing Parameters:")
    print(manufacturing[['sintering_temp', 'sintering_time', 'cooling_rate']].describe())
    
    print("\\nFailure Analysis:")
    print(quality[['overall_failure_risk', 'edge_crack_risk', 'delamination_risk']].describe())
    
    # Simple visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(manufacturing['sintering_temp'], bins=20, alpha=0.7)
    plt.title('Sintering Temperature')
    plt.xlabel('Temperature (°C)')
    
    plt.subplot(1, 3, 2)
    plt.hist(quality['overall_failure_risk'], bins=20, alpha=0.7)
    plt.title('Failure Risk')
    plt.xlabel('Risk')
    
    plt.subplot(1, 3, 3)
    plt.scatter(measurements['rms_displacement_um'], quality['overall_failure_risk'], alpha=0.6)
    plt.xlabel('RMS Displacement (μm)')
    plt.ylabel('Failure Risk')
    plt.title('Displacement vs Risk')
    
    plt.tight_layout()
    plt.savefig('quick_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\\n✅ Quick analysis complete!")
    print("Check 'quick_analysis.png' for visualizations")

if __name__ == "__main__":
    quick_analysis()
'''
        
        with open(package_dir / 'quick_start.py', 'w') as f:
            f.write(quick_start_script)
        
        # Create README for package
        package_readme = f'''# SOFC "In-The-Wild" Dataset Package

Complete dataset package for ML-Augmented Inverse Modeling of Residual Stress Quantification from Warped SOFC Plates.

## Quick Start

1. Run quick analysis:
   ```bash
   python quick_start.py
   ```

2. Load ML-ready data:
   ```python
   import numpy as np
   data = np.load('sofc_ml_ready.npz')
   X = data['X']  # Features
   y = data['y_max_stress']  # Target stress
   ```

3. Explore visualizations:
   - Check `quick_viz/` directory for overview plots
   - See `dataset/analysis_figures/` for detailed analysis

## Contents

- `dataset/` - Complete dataset with all files
- `sofc_ml_ready.npz` - ML-ready features and targets
- `quick_viz/` - Quick visualization plots
- `quick_start.py` - Quick start analysis script

## Dataset Structure

- **500 SOFC plates** with realistic manufacturing variations
- **Manufacturing parameters**: Temperature, time, cooling rate, density, humidity
- **Measurements**: High-resolution displacement fields (31×31 grid)
- **Ground truth**: Stress fields for validation
- **Failure analysis**: Risk indicators for various failure modes
- **Temporal data**: Production timeline with parameter drift

## Key Features

✅ **Realistic Manufacturing Variations**: Natural parameter drift and batch effects
✅ **Measurement Noise**: Multiple noise sources and uncertainties
✅ **Known Failure Modes**: Edge cracking, delamination, thermal shock patterns
✅ **ML-Ready**: Pre-processed features and comprehensive analysis
✅ **Validated**: Extensive validation for physical plausibility

## Citation

If you use this dataset in your research, please cite:

```
SOFC "In-The-Wild" Operational Dataset for ML-Augmented Inverse Modeling 
of Residual Stress Quantification from Warped Plates
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}
```

For detailed documentation, see `dataset/README.md`
'''
        
        with open(package_dir / 'README.md', 'w') as f:
            f.write(package_readme)
        
        # Create zip archive
        zip_file = f'{package_name}.zip'
        with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zipf.write(file_path, arcname)
        
        print(f"✅ Complete dataset package created:")
        print(f"   Directory: {package_dir}")
        print(f"   Archive: {zip_file}")
        print(f"   Size: {sum(f.stat().st_size for f in package_dir.rglob('*') if f.is_file()) / 1024**2:.1f} MB")
        
        return package_dir, zip_file

def main():
    """Main function to demonstrate dataset access"""
    print("SOFC Dataset Download and Setup")
    print("=" * 35)
    
    # Initialize downloader
    downloader = SOFCDatasetDownloader()
    
    # Load summary data
    summary_data = downloader.load_summary_data()
    
    # Get sample plates
    print("\nGetting sample plates...")
    sample_plates = downloader.get_sample_plates(5, 'diverse')
    
    # Create visualizations
    downloader.create_quick_visualization()
    
    # Export ML data
    downloader.export_for_ml()
    
    # Create complete package
    package_dir, zip_file = downloader.create_dataset_package()
    
    print("\n" + "=" * 35)
    print("✅ Dataset setup completed!")
    print(f"✅ {len(summary_data['manufacturing'])} plates available")
    print(f"✅ Complete package: {zip_file}")
    print("\nNext steps:")
    print("1. Explore quick_viz/ for dataset overview")
    print("2. Use sofc_ml_ready.npz for ML development")
    print("3. Check validation_report/ for data quality")
    print("4. See ml_analysis/ for baseline results")

if __name__ == "__main__":
    main()