#!/usr/bin/env python3
"""
Main Execution Script for IoT Building Dataset Generator
=======================================================

Complete system for generating and exporting realistic IoT building datasets
for digital twin applications and building retrofit optimization.

Usage:
    python main.py --config config.json --output ./output --duration 365

Author: AI Assistant
Date: 2025-10-16
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional

# Import our generators
from iot_building_dataset_generator import BuildingConfig, WeatherGenerator, EnergyConsumptionGenerator
from ieq_occupancy_generators import OccupancyPatternGenerator, IndoorEnvironmentalQualityGenerator
from building_systems_generator import BuildingSystemsGenerator
from comprehensive_dataset_generator import ComprehensiveDatasetGenerator
from data_export_visualization import DataExporter, DataVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> BuildingConfig:
    """Load building configuration from file or use defaults."""
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dict to BuildingConfig
        config = BuildingConfig(**config_dict)
    else:
        logger.info("Using default configuration")
        config = BuildingConfig(
            building_type="Commercial Office Building",
            floor_area=5000.0,
            num_floors=5,
            num_zones=20,
            occupancy_capacity=250,
            latitude=40.7128,   # New York City
            longitude=-74.0060,
            timezone="America/New_York",
            elevation=10.0,
            window_to_wall_ratio=0.4,
            building_orientation=0.0,
            hvac_type="VAV with Reheat",
            chiller_capacity=500.0,
            boiler_capacity=300.0,
            num_ahu=4,
            has_solar_panels=True,
            solar_capacity=100.0,
            has_energy_storage=True,
            battery_capacity=200.0
        )
    
    return config

def save_config_template(output_path: str):
    """Save a configuration template file."""
    
    template_config = {
        "building_type": "Commercial Office Building",
        "floor_area": 5000.0,
        "num_floors": 5,
        "num_zones": 20,
        "occupancy_capacity": 250,
        "latitude": 40.7128,
        "longitude": -74.0060,
        "timezone": "America/New_York",
        "elevation": 10.0,
        "window_to_wall_ratio": 0.4,
        "building_orientation": 0.0,
        "hvac_type": "VAV with Reheat",
        "chiller_capacity": 500.0,
        "boiler_capacity": 300.0,
        "num_ahu": 4,
        "has_solar_panels": True,
        "solar_capacity": 100.0,
        "has_energy_storage": True,
        "battery_capacity": 200.0
    }
    
    with open(output_path, 'w') as f:
        json.dump(template_config, f, indent=2)
    
    logger.info(f"Configuration template saved to {output_path}")

def calculate_dates(start_date: Optional[str] = None, duration_days: int = 365) -> tuple:
    """Calculate start and end dates for dataset generation."""
    
    if start_date:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
    else:
        # Default to start of current year
        current_year = datetime.now().year
        start = datetime(current_year, 1, 1)
    
    end = start + timedelta(days=duration_days)
    
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")
    
    return start_str, end_str

def main():
    """Main execution function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate comprehensive IoT building dataset for digital twin applications"
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to building configuration JSON file'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./iot_building_dataset',
        help='Output directory for generated dataset (default: ./iot_building_dataset)'
    )
    
    parser.add_argument(
        '--start-date', '-s',
        type=str,
        help='Start date for dataset (YYYY-MM-DD HH:MM:SS format, default: current year start)'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=int,
        default=365,
        help='Duration in days (default: 365)'
    )
    
    parser.add_argument(
        '--frequency', '-f',
        type=str,
        default='15T',
        help='Data frequency (default: 15T for 15-minute intervals)'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Skip data export (generate only)'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--create-config-template',
        type=str,
        help='Create a configuration template file at specified path and exit'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle config template creation
    if args.create_config_template:
        save_config_template(args.create_config_template)
        return
    
    # Print header
    print("🏢" + "=" * 70)
    print("  IoT Building Dataset Generator for Digital Twin Framework")
    print("  A Dynamic Digital Twin Framework for Multi-Objective")
    print("  Building Retrofit Optimization")
    print("=" * 72)
    print()
    
    try:
        # Load configuration
        logger.info("Loading building configuration...")
        config = load_config(args.config)
        
        # Calculate dates
        start_date, end_date = calculate_dates(args.start_date, args.duration)
        
        # Log configuration
        logger.info("Dataset Generation Configuration:")
        logger.info(f"  Building Type: {config.building_type}")
        logger.info(f"  Floor Area: {config.floor_area:,.0f} m²")
        logger.info(f"  Number of Zones: {config.num_zones}")
        logger.info(f"  Location: {config.latitude:.4f}°N, {config.longitude:.4f}°W")
        logger.info(f"  Start Date: {start_date}")
        logger.info(f"  End Date: {end_date}")
        logger.info(f"  Duration: {args.duration} days")
        logger.info(f"  Frequency: {args.frequency}")
        logger.info(f"  Output Directory: {args.output}")
        print()
        
        # Initialize generator
        logger.info("Initializing comprehensive dataset generator...")
        generator = ComprehensiveDatasetGenerator(config)
        
        # Generate dataset
        logger.info("Starting dataset generation...")
        start_time = datetime.now()
        
        dataset = generator.generate_complete_dataset(
            start_date=start_date,
            end_date=end_date,
            freq=args.frequency
        )
        
        generation_time = datetime.now() - start_time
        logger.info(f"Dataset generation completed in {generation_time}")
        
        # Export data
        if not args.no_export:
            logger.info("Exporting dataset to multiple formats...")
            export_start_time = datetime.now()
            
            exporter = DataExporter(args.output)
            export_paths = exporter.export_all_formats(dataset, generator.metadata)
            
            export_time = datetime.now() - export_start_time
            logger.info(f"Data export completed in {export_time}")
            
            # Log export paths
            logger.info("Export Summary:")
            for format_type, paths in export_paths.items():
                if isinstance(paths, dict):
                    logger.info(f"  {format_type.upper()}:")
                    for name, path in paths.items():
                        logger.info(f"    {name}: {path}")
                else:
                    logger.info(f"  {format_type.upper()}: {paths}")
        
        # Generate visualizations
        if not args.no_visualizations:
            logger.info("Generating visualizations...")
            viz_start_time = datetime.now()
            
            visualizer = DataVisualizer()
            viz_output_dir = os.path.join(args.output, "visualizations")
            viz_paths = visualizer.create_static_visualizations(dataset, viz_output_dir)
            
            viz_time = datetime.now() - viz_start_time
            logger.info(f"Visualization generation completed in {viz_time}")
            
            # Log visualization paths
            logger.info("Visualization Summary:")
            for viz_type, path in viz_paths.items():
                logger.info(f"  {viz_type}: {path}")
        
        # Final summary
        total_time = datetime.now() - start_time
        total_points = sum(len(df) for df in dataset.values())
        total_parameters = sum(len(df.columns) - 1 for df in dataset.values())  # -1 for timestamp
        
        print("\n" + "=" * 72)
        print("🎉 DATASET GENERATION COMPLETE!")
        print("=" * 72)
        print(f"📊 Total Data Points: {total_points:,}")
        print(f"📈 Total Parameters: {total_parameters:,}")
        print(f"⏱️  Total Processing Time: {total_time}")
        print(f"💾 Output Directory: {args.output}")
        
        # Calculate estimated file sizes
        estimated_size_mb = (total_points * total_parameters * 8) / (1024 * 1024)
        print(f"💽 Estimated Dataset Size: {estimated_size_mb:.1f} MB")
        
        print("\n📋 Dataset Components:")
        for name, df in dataset.items():
            print(f"  {name.upper()}: {len(df.columns)-1} parameters, {len(df):,} records")
        
        print("\n🔍 Key Metrics:")
        
        # Weather summary
        weather_df = dataset['weather']
        print(f"🌡️  Temperature Range: {weather_df['ambient_temperature_c'].min():.1f}°C to {weather_df['ambient_temperature_c'].max():.1f}°C")
        
        # Energy summary
        energy_df = dataset['energy']
        print(f"⚡ Peak Electricity: {energy_df['total_electricity_kw'].max():.1f} kW")
        annual_consumption = energy_df['total_electricity_kw'].sum() * (15/60)  # Convert 15-min to hours
        print(f"🔋 Annual Consumption: {annual_consumption:,.0f} kWh")
        
        # Occupancy summary
        occupancy_df = dataset['occupancy']
        print(f"👥 Peak Occupancy: {occupancy_df['total_occupancy'].max()} people")
        
        # IEQ summary
        ieq_df = dataset['ieq']
        print(f"🏢 Indoor Temp Range: {ieq_df['building_avg_temp_c'].min():.1f}°C to {ieq_df['building_avg_temp_c'].max():.1f}°C")
        
        print("\n✅ Dataset ready for digital twin applications!")
        print("📚 See README.md for usage examples and API documentation")
        print("=" * 72)
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()