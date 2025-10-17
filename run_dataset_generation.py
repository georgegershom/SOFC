#!/usr/bin/env python3
"""
Main execution script for Building DNA Dataset Generation
Run this script to generate the complete building DNA dataset
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from comprehensive_dataset_generator import ComprehensiveDatasetGenerator
import json
from datetime import datetime

def main():
    """Main execution function"""
    print("=" * 60)
    print("BUILDING DNA DATASET GENERATOR")
    print("Dynamic Digital Twin Framework for Multi-Objective Building Retrofit Optimization")
    print("=" * 60)
    print()
    
    # Configuration
    BUILDING_TYPES = ['residential', 'commercial', 'office', 'industrial']
    NUM_BUILDINGS_PER_TYPE = 3
    OUTPUT_DIR = "building_dna_dataset"
    SEED = 42
    
    print(f"Configuration:")
    print(f"  Building Types: {', '.join(BUILDING_TYPES)}")
    print(f"  Buildings per Type: {NUM_BUILDINGS_PER_TYPE}")
    print(f"  Total Buildings: {len(BUILDING_TYPES) * NUM_BUILDINGS_PER_TYPE}")
    print(f"  Output Directory: {OUTPUT_DIR}")
    print(f"  Random Seed: {SEED}")
    print()
    
    # Initialize generator
    print("Initializing dataset generator...")
    generator = ComprehensiveDatasetGenerator(seed=SEED)
    
    # Generate datasets
    print("Generating building datasets...")
    all_buildings = generator.generate_multiple_buildings(
        BUILDING_TYPES, 
        NUM_BUILDINGS_PER_TYPE
    )
    
    print(f"\nGenerated {len(all_buildings)} complete building datasets")
    
    # Export datasets
    print("Exporting datasets...")
    exported_paths = []
    
    for i, building_dataset in enumerate(all_buildings):
        building_type = building_dataset['metadata']['building_type']
        building_id = building_dataset['metadata']['building_id']
        
        print(f"  Exporting {building_type} building {i+1}/{len(all_buildings)} (ID: {building_id[:8]}...)")
        
        building_path = generator.export_dataset(building_dataset, OUTPUT_DIR)
        exported_paths.append(building_path)
    
    # Create summary dataset
    print("Creating summary dataset...")
    summary_data = {
        'metadata': {
            'total_buildings': len(all_buildings),
            'building_types': BUILDING_TYPES,
            'buildings_per_type': NUM_BUILDINGS_PER_TYPE,
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0.0',
            'random_seed': SEED
        },
        'buildings': all_buildings
    }
    
    summary_path = f"{OUTPUT_DIR}/complete_dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Create dataset archive
    print("Creating dataset archive...")
    archive_path = generator.create_dataset_archive(OUTPUT_DIR)
    
    # Print completion summary
    print("\n" + "=" * 60)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 60)
    print()
    
    print(f"Total buildings generated: {len(all_buildings)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Archive created: {archive_path}")
    print(f"Summary file: {summary_path}")
    print()
    
    # Print building type breakdown
    print("Building Type Breakdown:")
    print("-" * 30)
    for building_type in BUILDING_TYPES:
        count = sum(1 for b in all_buildings if b['metadata']['building_type'] == building_type)
        print(f"  {building_type.capitalize()}: {count} buildings")
    
    print()
    
    # Print dataset contents
    print("Each building dataset contains:")
    print("-" * 40)
    print("  📊 building_dna.json - Static & fabric data")
    print("  🏗️  bim_model.json - 3D BIM model")
    print("  📡 lidar_point_cloud.json - LiDAR point cloud")
    print("  🔌 iot_sensor_data.json - IoT sensor data")
    print("  🌱 lca_assessment.json - Life-cycle assessment")
    print("  📈 performance_metrics.json - Performance indicators")
    print("  📋 complete_dataset.json - Complete integrated dataset")
    print("  🎨 Multiple HTML visualizations")
    print("  📖 README.md - Documentation")
    
    print()
    
    # Print performance statistics
    print("Performance Statistics:")
    print("-" * 30)
    total_area = sum(b['building_dna']['geometry']['total_area'] for b in all_buildings)
    avg_area = total_area / len(all_buildings)
    print(f"  Total building area: {total_area:.0f} m²")
    print(f"  Average building area: {avg_area:.0f} m²")
    
    # Calculate average performance scores
    performance_scores = [b['performance_metrics']['performance_score'] for b in all_buildings]
    avg_performance = sum(performance_scores) / len(performance_scores)
    print(f"  Average performance score: {avg_performance:.1f}/100")
    
    # Calculate average carbon intensity
    carbon_intensities = [b['lca_assessment']['carbon_intensity'] for b in all_buildings]
    avg_carbon = sum(carbon_intensities) / len(carbon_intensities)
    print(f"  Average carbon intensity: {avg_carbon:.1f} kg CO2/m²/year")
    
    print()
    print("Dataset ready for use in Dynamic Digital Twin Framework!")
    print("=" * 60)

if __name__ == "__main__":
    main()