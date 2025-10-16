#!/usr/bin/env python3
"""
Example script for loading and exploring the In-The-Wild SOFC Plate Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_metadata(dataset_dir='in_the_wild_dataset'):
    """Load the dataset metadata."""
    metadata_path = Path(dataset_dir) / 'dataset_metadata.csv'
    metadata = pd.read_csv(metadata_path)
    metadata['date'] = pd.to_datetime(metadata['date'])
    return metadata

def load_plate_data(plate_info, dataset_dir='in_the_wild_dataset'):
    """
    Load complete data for a single plate.
    
    Parameters:
    -----------
    plate_info : pandas.Series
        Row from metadata DataFrame
    dataset_dir : str
        Path to dataset directory
        
    Returns:
    --------
    dict with keys:
        - 'metadata': plate metadata
        - 'warp_df': warp measurements (DataFrame)
        - 'stress_xx': stress in X direction (ndarray)
        - 'stress_yy': stress in Y direction (ndarray)
        - 'warp_true': true warp without noise (ndarray)
        - 'warp_measured': measured warp with noise (ndarray)
        - 'X': X coordinate grid (ndarray)
        - 'Y': Y coordinate grid (ndarray)
    """
    dataset_path = Path(dataset_dir)
    
    # Load warp measurements
    warp_path = dataset_path / plate_info['measurement_file']
    warp_df = pd.read_csv(warp_path)
    
    # Load stress fields
    stress_path = dataset_path / plate_info['stress_file']
    stress_data = np.load(stress_path)
    
    return {
        'metadata': plate_info,
        'warp_df': warp_df,
        'stress_xx': stress_data['stress_xx'],
        'stress_yy': stress_data['stress_yy'],
        'warp_true': stress_data['warp_true'],
        'warp_measured': stress_data['warp_measured'],
        'X': stress_data['X'],
        'Y': stress_data['Y'],
    }

def visualize_plate(plate_data, save_path=None):
    """Visualize a plate's warp and stress fields."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Warp measurement
    im1 = axes[0, 0].contourf(plate_data['X'], plate_data['Y'], 
                             plate_data['warp_measured'], 
                             levels=20, cmap='RdYlBu_r')
    axes[0, 0].set_title(f"Measured Warp - {plate_data['metadata']['plate_id']}")
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 0], label='Warp (mm)')
    
    # True warp (without noise)
    im2 = axes[0, 1].contourf(plate_data['X'], plate_data['Y'], 
                             plate_data['warp_true'], 
                             levels=20, cmap='RdYlBu_r')
    axes[0, 1].set_title('True Warp (Ground Truth)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[0, 1], label='Warp (mm)')
    
    # Stress XX
    im3 = axes[1, 0].contourf(plate_data['X'], plate_data['Y'], 
                             plate_data['stress_xx'], 
                             levels=20, cmap='plasma')
    axes[1, 0].set_title('Residual Stress σ_xx')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[1, 0], label='Stress (MPa)')
    
    # Stress YY
    im4 = axes[1, 1].contourf(plate_data['X'], plate_data['Y'], 
                             plate_data['stress_yy'], 
                             levels=20, cmap='plasma')
    axes[1, 1].set_title('Residual Stress σ_yy')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im4, ax=axes[1, 1], label='Stress (MPa)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def print_dataset_summary(metadata):
    """Print dataset statistics."""
    print("=" * 80)
    print("IN-THE-WILD SOFC PLATE DATASET SUMMARY")
    print("=" * 80)
    print(f"\nTotal plates: {len(metadata)}")
    print(f"Date range: {metadata['date'].min().date()} to {metadata['date'].max().date()}")
    print(f"Production batches: {metadata['batch_id'].nunique()}")
    print(f"Material batches: {metadata['material_batch'].nunique()}")
    
    print("\n--- Quality Distribution ---")
    print(metadata['quality_class'].value_counts().sort_index())
    
    print("\n--- Failure Distribution ---")
    print(metadata['failure_type'].value_counts())
    
    print("\n--- Stress Pattern Distribution ---")
    print(metadata['stress_pattern'].value_counts())
    
    print("\n--- Warp Statistics ---")
    print(f"Mean warp: {metadata['mean_warp_mm'].mean():.3f} ± {metadata['mean_warp_mm'].std():.3f} mm")
    print(f"Max warp (across all plates): {metadata['max_warp_mm'].max():.3f} mm")
    print(f"Min warp (across all plates): {metadata['max_warp_mm'].min():.3f} mm")
    
    print("\n--- Stress Statistics ---")
    print(f"Mean max stress: {metadata['max_stress_MPa'].mean():.1f} ± {metadata['max_stress_MPa'].std():.1f} MPa")
    print(f"Max stress (across all plates): {metadata['max_stress_MPa'].max():.1f} MPa")
    
    print("\n--- Production Parameters ---")
    print(f"Furnace temp: {metadata['furnace_temp_C'].mean():.1f} ± {metadata['furnace_temp_C'].std():.1f} °C")
    print(f"Cooling rate: {metadata['cooling_rate_C_per_min'].mean():.2f} ± {metadata['cooling_rate_C_per_min'].std():.2f} °C/min")
    print(f"Furnace age: 0 to {metadata['furnace_age_days'].max()} days")
    print("=" * 80)

def example_filtering(metadata):
    """Demonstrate various filtering operations."""
    print("\n" + "=" * 80)
    print("EXAMPLE FILTERING OPERATIONS")
    print("=" * 80)
    
    # Filter by quality
    good = metadata[metadata['quality_class'] == 'good']
    print(f"\nGood quality plates: {len(good)} ({len(good)/len(metadata)*100:.1f}%)")
    
    # Filter by failure
    failed = metadata[metadata['failed'] == True]
    print(f"Failed plates: {len(failed)} ({len(failed)/len(metadata)*100:.1f}%)")
    if len(failed) > 0:
        print("\nFailed plate details:")
        print(failed[['plate_id', 'failure_type', 'max_stress_MPa', 'max_warp_mm']].to_string(index=False))
    
    # Filter by stress pattern
    edge_dominated = metadata[metadata['stress_pattern'] == 'edge_dominated']
    print(f"\nEdge-dominated stress pattern: {len(edge_dominated)} plates")
    
    # Filter by high stress
    high_stress = metadata[metadata['max_stress_MPa'] > 100]
    print(f"High stress (>100 MPa): {len(high_stress)} plates")
    
    # Filter by production time
    early = metadata[metadata['furnace_age_days'] < 100]
    late = metadata[metadata['furnace_age_days'] >= 200]
    print(f"\nEarly production (age < 100 days): {len(early)} plates")
    print(f"Late production (age >= 200 days): {len(late)} plates")
    
    print("=" * 80)

def example_ml_split(metadata):
    """Demonstrate train/test splits for ML."""
    try:
        from sklearn.model_selection import train_test_split
        has_sklearn = True
    except ImportError:
        has_sklearn = False
    
    print("\n" + "=" * 80)
    print("EXAMPLE ML TRAIN/TEST SPLITS")
    print("=" * 80)
    
    # Random split
    if has_sklearn:
        train, test = train_test_split(metadata, test_size=0.2, random_state=42)
        print(f"\nRandom split: {len(train)} train, {len(test)} test")
    else:
        print("\nRandom split: (sklearn not installed, skipping)")
        print("Install sklearn with: pip install scikit-learn")
    
    # Temporal split (train on early, test on late)
    cutoff_date = metadata['date'].quantile(0.8)
    train_temporal = metadata[metadata['date'] < cutoff_date]
    test_temporal = metadata[metadata['date'] >= cutoff_date]
    print(f"Temporal split: {len(train_temporal)} train, {len(test_temporal)} test")
    print(f"  Train date range: {train_temporal['date'].min().date()} to {train_temporal['date'].max().date()}")
    print(f"  Test date range: {test_temporal['date'].min().date()} to {test_temporal['date'].max().date()}")
    
    # Batch-based split
    n_batches = metadata['batch_id'].nunique()
    train_batches = int(n_batches * 0.8)
    unique_batches = sorted(metadata['batch_id'].unique())
    train_batch_ids = unique_batches[:train_batches]
    train_batch = metadata[metadata['batch_id'].isin(train_batch_ids)]
    test_batch = metadata[~metadata['batch_id'].isin(train_batch_ids)]
    print(f"Batch split: {len(train_batch)} train, {len(test_batch)} test")
    
    print("=" * 80)


if __name__ == '__main__':
    print("\n🔬 Loading In-The-Wild SOFC Plate Dataset...\n")
    
    # Load metadata
    metadata = load_metadata()
    
    # Print summary
    print_dataset_summary(metadata)
    
    # Example filtering
    example_filtering(metadata)
    
    # Example ML splits
    example_ml_split(metadata)
    
    # Load and visualize a sample plate
    print("\n" + "=" * 80)
    print("LOADING SAMPLE PLATE")
    print("=" * 80)
    
    # Pick an interesting plate (e.g., a failed one or high stress one)
    failed_plates = metadata[metadata['failed'] == True]
    if len(failed_plates) > 0:
        sample_plate = failed_plates.iloc[0]
        print(f"\nSelected: {sample_plate['plate_id']} (FAILED)")
    else:
        sample_plate = metadata.iloc[0]
        print(f"\nSelected: {sample_plate['plate_id']}")
    
    print(f"Quality: {sample_plate['quality_class']}")
    print(f"Stress pattern: {sample_plate['stress_pattern']}")
    print(f"Max warp: {sample_plate['max_warp_mm']:.3f} mm")
    print(f"Max stress: {sample_plate['max_stress_MPa']:.1f} MPa")
    print(f"Failure: {sample_plate['failure_type']}")
    
    print("\nLoading plate data...")
    plate_data = load_plate_data(sample_plate)
    
    print(f"Warp measurements shape: {plate_data['warp_df'].shape}")
    print(f"Stress field shape: {plate_data['stress_xx'].shape}")
    print(f"Missing data points: {plate_data['warp_df']['z_mm'].isna().sum()}")
    
    print("\n✅ Dataset loaded successfully!")
    print("\nTo visualize this plate, uncomment the line below:")
    print("# visualize_plate(plate_data)")
    
    print("\n" + "=" * 80)
    print("Ready for ML model training and testing!")
    print("=" * 80 + "\n")
