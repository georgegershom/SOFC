#!/usr/bin/env python3
"""
Complete Dataset Generation and Analysis Pipeline
Generates the ML training dataset and performs comprehensive analysis
"""

import sys
import time
from pathlib import Path

# Import our modules
from ml_training_dataset_generator import SinteringDatasetGenerator
from dataset_analyzer import DatasetAnalyzer

def main():
    """Run the complete dataset generation and analysis pipeline."""
    
    print("🧠 MACHINE LEARNING TRAINING DATASET GENERATOR")
    print("=" * 60)
    print("Generating comprehensive dataset for ANN and PINN models")
    print("Focus: Sintering process analysis with thermal stress modeling")
    print("=" * 60)
    
    # Configuration
    n_samples = 10000
    output_dir = 'ml_sintering_dataset'
    
    print(f"\nConfiguration:")
    print(f"  - Number of samples: {n_samples:,}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Sintering temperatures: 1200–1500°C")
    print(f"  - Cooling rates: 1–10°C/min")
    print(f"  - TEC mismatch: Δα = 2.3×10⁻⁶ K⁻¹")
    print(f"  - Variable porosity levels")
    
    # Step 1: Generate the dataset
    print(f"\n{'='*20} STEP 1: DATASET GENERATION {'='*20}")
    
    start_time = time.time()
    
    try:
        generator = SinteringDatasetGenerator(random_seed=42)
        dataset, spatial_data, validation_data = generator.generate_complete_dataset(
            n_samples=n_samples,
            output_dir=output_dir
        )
        
        generation_time = time.time() - start_time
        print(f"\n✅ Dataset generation completed in {generation_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Error during dataset generation: {e}")
        sys.exit(1)
    
    # Step 2: Analyze the dataset
    print(f"\n{'='*20} STEP 2: DATASET ANALYSIS {'='*20}")
    
    try:
        analyzer = DatasetAnalyzer(output_dir)
        analyzer.run_complete_analysis()
        
        analysis_time = time.time() - start_time - generation_time
        print(f"\n✅ Dataset analysis completed in {analysis_time:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Error during dataset analysis: {e}")
        print("Dataset generation was successful, but analysis failed.")
    
    # Step 3: Summary and next steps
    print(f"\n{'='*20} SUMMARY AND NEXT STEPS {'='*20}")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"Total execution time: {total_time:.1f} seconds")
    
    # Check if output directory exists and list files
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\n📁 Generated files in '{output_dir}':")
        for file_path in sorted(output_path.iterdir()):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  - {file_path.name} ({size_mb:.1f} MB)")
    
    print(f"\n📊 DATASET SPECIFICATIONS:")
    print(f"  ✓ {n_samples:,} simulated samples")
    print(f"  ✓ Physics-based thermal stress modeling")
    print(f"  ✓ Variable process parameters (temperature, cooling rate, porosity)")
    print(f"  ✓ Spatial field data (50x50 grids)")
    print(f"  ✓ Multiple output labels (stress hotspots, crack risk, delamination)")
    print(f"  ✓ Experimental validation data (DIC/XRD measurements)")
    print(f"  ✓ Multiple export formats (CSV, HDF5, NPZ)")
    print(f"  ✓ Train/validation/test splits")
    
    print(f"\n🚀 READY FOR MACHINE LEARNING:")
    print(f"  - Load data: pandas.read_csv('{output_dir}/ml_training_dataset.csv')")
    print(f"  - Use splits: '{output_dir}/train_split.csv', etc.")
    print(f"  - Spatial data: '{output_dir}/spatial_fields_data.h5'")
    print(f"  - Validation: '{output_dir}/experimental_validation_data.csv'")
    
    print(f"\n📈 RECOMMENDED ML APPROACHES:")
    print(f"  - Neural Networks (ANN): Use tabular features for regression")
    print(f"  - Physics-Informed Neural Networks (PINN): Incorporate spatial fields")
    print(f"  - Convolutional Neural Networks: Process spatial field images")
    print(f"  - Ensemble methods: Combine multiple model predictions")
    
    print(f"\n🔬 VALIDATION STRATEGY:")
    print(f"  - Cross-validate with experimental DIC/XRD data")
    print(f"  - Compare predicted vs. actual stress distributions")
    print(f"  - Validate crack initiation predictions")
    print(f"  - Test delamination probability accuracy")


if __name__ == "__main__":
    main()