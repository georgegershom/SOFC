"""
Main script to generate, analyze, and prepare welding dataset for inverse design
"""

import os
import sys
import time
from welding_dataset_generator import WeldingDatasetGenerator
from dataset_analysis import WeldingDatasetAnalyzer
import warnings
warnings.filterwarnings('ignore')


def main():
    """Main execution function"""
    
    print("=" * 80)
    print("WELDING INVERSE DESIGN DATASET GENERATOR")
    print("For Extreme-Temperature Performance Research")
    print("=" * 80)
    
    # Step 1: Generate Dataset
    print("\n[STEP 1] Generating Multi-Tier Welding Dataset...")
    print("-" * 60)
    
    start_time = time.time()
    generator = WeldingDatasetGenerator(seed=42)
    
    # Generate complete dataset with all tiers
    dataset = generator.generate_complete_dataset(
        tier1_samples=500,      # High-fidelity experimental data
        tier2_samples=10000,    # Computational simulation data
        tier3_samples=1000      # Literature-based data
    )
    
    # Save dataset
    generator.save_dataset(dataset, 'welding_dataset')
    
    generation_time = time.time() - start_time
    print(f"✓ Dataset generation completed in {generation_time:.1f} seconds")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Features: {len(dataset.columns)}")
    
    # Step 2: Analyze Dataset
    print("\n[STEP 2] Performing Dataset Analysis...")
    print("-" * 60)
    
    analyzer = WeldingDatasetAnalyzer('welding_dataset/complete_dataset.csv')
    
    # Generate analysis report
    print("Generating comprehensive analysis report...")
    analyzer.generate_analysis_report('analysis_report')
    
    # Perform PCA analysis
    print("Performing PCA analysis...")
    pca, X_pca = analyzer.perform_pca_analysis(n_components=5)
    
    # Analyze extreme temperature performance
    print("Analyzing extreme temperature performance...")
    analyzer.analyze_extreme_temperature_performance()
    
    # Export ML-ready data
    print("Exporting ML-ready data...")
    analyzer.export_for_ml_training('ml_ready_data')
    
    print("✓ Dataset analysis completed")
    
    # Step 3: Train Inverse Design Models (Optional)
    print("\n[STEP 3] Training Inverse Design Models (Optional)...")
    print("-" * 60)
    
    response = input("Do you want to train the inverse design models? (y/n): ")
    
    if response.lower() == 'y':
        try:
            import torch
            from inverse_design_model import demonstrate_inverse_design
            print("PyTorch is available. Training models...")
            demonstrate_inverse_design()
            print("✓ Model training completed")
        except ImportError:
            print("⚠ PyTorch not installed. Skipping model training.")
            print("  To train models, install PyTorch: pip install torch")
    else:
        print("Skipping model training.")
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    
    print("\n📁 Generated Files:")
    print("  • welding_dataset/")
    print("    - complete_dataset.csv (Main dataset)")
    print("    - complete_dataset.parquet (Efficient format)")
    print("    - metadata.json (Dataset metadata)")
    print("    - summary_statistics.csv")
    
    print("\n  • analysis_report/")
    print("    - analysis_overview.png (Comprehensive visualizations)")
    print("    - pca_analysis.png")
    print("    - extreme_temperature_analysis.png")
    
    print("\n  • ml_ready_data/")
    print("    - X_train.npy, Y_train.npy (Training data)")
    print("    - X_val.npy, Y_val.npy (Validation data)")
    print("    - X_test.npy, Y_test.npy (Test data)")
    print("    - feature_names.json")
    print("    - normalization_params.json")
    
    print("\n📊 Dataset Statistics:")
    print(f"  • Total Samples: {len(dataset)}")
    print(f"  • Tier 1 (Experimental): {len(dataset[dataset['data_tier'] == 1])}")
    print(f"  • Tier 2 (Simulation): {len(dataset[dataset['data_tier'] == 2])}")
    print(f"  • Tier 3 (Literature): {len(dataset[dataset['data_tier'] == 3])}")
    
    print("\n🔍 Key Features:")
    print("  Input Parameters (13):")
    print("    - Energy: laser_power, welding_speed, pulse_frequency, pulse_duration")
    print("    - Beam: beam_focus_position, beam_spot_size")
    print("    - Setup: clamping_pressure, gas_flow_rate, sheet_thickness")
    print("    - Material: material_combination, shield_gas_type, joint_type")
    
    print("\n  Output Parameters (19):")
    print("    - Morphology: nugget_width, penetration_depth, haz_width")
    print("    - Quality: has_cracks, has_porosity, spatter_count")
    print("    - Mechanical: tensile_strength, peel_strength")
    print("    - Electrical: contact_resistance, resistance_increase_percent")
    print("    - Thermal: cycles_to_failure, strength_degradation_percent")
    print("    - Microstructure: imc_thickness, grain_size evolution")
    
    print("\n✅ Dataset is ready for inverse design research!")
    print("\n💡 Next Steps:")
    print("  1. Review the analysis reports in 'analysis_report/' folder")
    print("  2. Use ML-ready data in 'ml_ready_data/' for model training")
    print("  3. Train inverse design models using 'inverse_design_model.py'")
    print("  4. Customize parameters in 'welding_dataset_generator.py' as needed")
    
    return dataset


if __name__ == "__main__":
    dataset = main()