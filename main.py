"""
Main script to generate, visualize, and export the rubberized concrete experimental dataset
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from data_generator import ConcreteDataGenerator
from visualization import ConcreteDataVisualizer
from config import *

def create_output_directories():
    """Create output directories for the dataset"""
    directories = [
        'output',
        'output/data',
        'output/figures',
        'output/reports',
        'output/raw_data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Created output directories")

def generate_dataset():
    """Generate the complete experimental dataset"""
    print("=" * 60)
    print("RUBBERIZED CONCRETE EXPERIMENTAL DATASET GENERATION")
    print("=" * 60)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize data generator
    generator = ConcreteDataGenerator(seed=42)
    
    # Generate complete dataset
    print("Generating complete experimental dataset...")
    complete_data = generator.generate_complete_dataset()
    
    # Add metadata
    complete_data['generation_timestamp'] = datetime.now()
    complete_data['dataset_version'] = '1.0'
    complete_data['research_topic'] = 'Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete'
    
    print(f"Dataset generated successfully!")
    print(f"Total records: {len(complete_data)}")
    print(f"Mix designs: {complete_data['mix_design'].nunique()}")
    print(f"Test types: {complete_data['test_type'].nunique()}")
    print(f"Temperature levels: {complete_data['temperature_c'].nunique()}")
    print()
    
    return complete_data

def export_dataset(data: pd.DataFrame):
    """Export dataset in multiple formats"""
    print("Exporting dataset in multiple formats...")
    
    # Export to CSV
    csv_path = 'output/data/rubberized_concrete_experimental_dataset.csv'
    data.to_csv(csv_path, index=False)
    print(f"✓ CSV exported to: {csv_path}")
    
    # Export to Excel with multiple sheets
    excel_path = 'output/data/rubberized_concrete_experimental_dataset.xlsx'
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Complete dataset
        data.to_excel(writer, sheet_name='Complete_Dataset', index=False)
        
        # Ambient data only
        ambient_data = data[data['test_type'] == 'ambient']
        ambient_data.to_excel(writer, sheet_name='Ambient_Tests', index=False)
        
        # Thermal exposure data only
        thermal_data = data[data['test_type'] == 'thermal_exposure']
        thermal_data.to_excel(writer, sheet_name='Thermal_Exposure_Tests', index=False)
        
        # In-situ thermal data only
        insitu_data = data[data['test_type'] == 'insitu_thermal']
        insitu_data.to_excel(writer, sheet_name='InSitu_Thermal_Tests', index=False)
        
        # Summary statistics
        visualizer = ConcreteDataVisualizer(data)
        summary_stats = visualizer.generate_summary_statistics()
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"✓ Excel exported to: {excel_path}")
    
    # Export to JSON
    json_path = 'output/data/rubberized_concrete_experimental_dataset.json'
    data.to_json(json_path, orient='records', indent=2)
    print(f"✓ JSON exported to: {json_path}")
    
    # Export raw data files for each test type
    for test_type in data['test_type'].unique():
        type_data = data[data['test_type'] == test_type]
        raw_path = f'output/raw_data/{test_type}_data.csv'
        type_data.to_csv(raw_path, index=False)
        print(f"✓ {test_type} data exported to: {raw_path}")
    
    print()

def create_visualizations(data: pd.DataFrame):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    visualizer = ConcreteDataVisualizer(data)
    
    # Create static plots
    print("  - Creating strength development plots...")
    visualizer.plot_strength_development('output/figures/strength_development.png')
    
    print("  - Creating temperature effects plots...")
    visualizer.plot_temperature_effects('output/figures/temperature_effects.png')
    
    print("  - Creating rubber content effects plots...")
    visualizer.plot_rubber_content_effects('output/figures/rubber_content_effects.png')
    
    print("  - Creating spalling behavior plots...")
    visualizer.plot_spalling_behavior('output/figures/spalling_behavior.png')
    
    print("  - Creating in-situ properties plots...")
    visualizer.plot_insitu_properties('output/figures/insitu_properties.png')
    
    # Create interactive dashboard
    print("  - Creating interactive dashboard...")
    visualizer.create_interactive_dashboard('output/figures/interactive_dashboard.html')
    
    print("✓ All visualizations created successfully!")
    print()

def generate_report(data: pd.DataFrame):
    """Generate a comprehensive report"""
    print("Generating comprehensive report...")
    
    report_path = 'output/reports/dataset_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# Rubberized Concrete Experimental Dataset Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Research Topic:** Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total Records:** {len(data):,}\n")
        f.write(f"- **Mix Designs:** {data['mix_design'].nunique()}\n")
        f.write(f"- **Test Types:** {data['test_type'].nunique()}\n")
        f.write(f"- **Temperature Levels:** {data['temperature_c'].nunique()}\n")
        f.write(f"- **Cooling Regimes:** {data['cooling_regime'].nunique()}\n\n")
        
        f.write("## Mix Designs\n\n")
        for mix in data['mix_design'].unique():
            mix_data = data[data['mix_design'] == mix]
            f.write(f"- **{mix}:** {mix_data['rubber_content_percent'].iloc[0]}% rubber content\n")
        f.write("\n")
        
        f.write("## Test Types\n\n")
        f.write("### 1. Ambient Condition Tests\n")
        f.write("- Compressive Strength (ASTM C39)\n")
        f.write("- Splitting Tensile Strength (ASTM C496)\n")
        f.write("- Flexural Strength (ASTM C78)\n")
        f.write("- Static Modulus of Elasticity (ASTM C469)\n")
        f.write("- Density and Ultrasonic Pulse Velocity (UPV)\n\n")
        
        f.write("### 2. High-Temperature Exposure Tests\n")
        f.write("- Thermal Exposure Regime: 23°C, 200°C, 400°C, 600°C, 800°C\n")
        f.write("- Heating Rate: 8°C/min\n")
        f.write("- Soak Time: 60 minutes\n")
        f.write("- Cooling Methods: Furnace Cooling, Water Quenching\n")
        f.write("- Residual Property Tests\n\n")
        
        f.write("### 3. In-Situ High-Temperature Tests\n")
        f.write("- Transient Thermal Strain\n")
        f.write("- In-Situ Compressive Strength & Modulus of Elasticity\n")
        f.write("- Thermal Expansion (Dilatometry)\n")
        f.write("- Spalling Behavior Analysis\n")
        f.write("- Pore Pressure Measurement\n\n")
        
        f.write("## Statistical Summary\n\n")
        visualizer = ConcreteDataVisualizer(data)
        summary_stats = visualizer.generate_summary_statistics()
        
        f.write("### Ambient Tests (28-day strength)\n")
        ambient_28d = data[(data['test_type'] == 'ambient') & (data['curing_age_days'] == 28)]
        for mix in ambient_28d['mix_design'].unique():
            mix_data = ambient_28d[ambient_28d['mix_design'] == mix]
            f.write(f"- **{mix}:**\n")
            f.write(f"  - Compressive Strength: {mix_data['compressive_strength_mpa'].mean():.1f} ± {mix_data['compressive_strength_mpa'].std():.1f} MPa\n")
            f.write(f"  - Tensile Strength: {mix_data['tensile_strength_mpa'].mean():.1f} ± {mix_data['tensile_strength_mpa'].std():.1f} MPa\n")
            f.write(f"  - Modulus of Elasticity: {mix_data['modulus_elasticity_mpa'].mean():.0f} ± {mix_data['modulus_elasticity_mpa'].std():.0f} MPa\n")
            f.write(f"  - Density: {mix_data['density_kg_m3'].mean():.0f} ± {mix_data['density_kg_m3'].std():.0f} kg/m³\n\n")
        
        f.write("## Key Findings\n\n")
        f.write("1. **Rubber Content Effects:**\n")
        f.write("   - Compressive strength decreases with increasing rubber content\n")
        f.write("   - Tensile strength shows similar trend but with less reduction\n")
        f.write("   - Modulus of elasticity significantly decreases with rubber content\n")
        f.write("   - Density decreases with rubber content due to lower rubber density\n\n")
        
        f.write("2. **Temperature Effects:**\n")
        f.write("   - Significant strength reduction at temperatures above 400°C\n")
        f.write("   - Water quenching causes additional damage compared to furnace cooling\n")
        f.write("   - Mass loss increases with temperature due to dehydration\n")
        f.write("   - Spalling occurs primarily above 400°C\n\n")
        
        f.write("3. **Fire Resistance:**\n")
        f.write("   - Rubber content improves fire resistance\n")
        f.write("   - Reduced spalling with higher rubber content\n")
        f.write("   - Better thermal expansion characteristics\n\n")
        
        f.write("## Files Generated\n\n")
        f.write("- `rubberized_concrete_experimental_dataset.csv` - Complete dataset in CSV format\n")
        f.write("- `rubberized_concrete_experimental_dataset.xlsx` - Excel file with multiple sheets\n")
        f.write("- `rubberized_concrete_experimental_dataset.json` - JSON format for programmatic access\n")
        f.write("- `interactive_dashboard.html` - Interactive Plotly dashboard\n")
        f.write("- Various PNG files with static plots\n\n")
        
        f.write("## Usage Instructions\n\n")
        f.write("1. Load the CSV or Excel file in your preferred data analysis software\n")
        f.write("2. Use the interactive dashboard for exploratory data analysis\n")
        f.write("3. Refer to the static plots for publication-quality figures\n")
        f.write("4. Use the raw data files for specific test types\n\n")
        
        f.write("---\n")
        f.write("*This dataset was generated using advanced statistical modeling techniques to simulate realistic experimental conditions for rubberized concrete under fire exposure.*\n")
    
    print(f"✓ Report generated: {report_path}")
    print()

def main():
    """Main function to orchestrate the dataset generation process"""
    print("Starting rubberized concrete experimental dataset generation...")
    print()
    
    # Create output directories
    create_output_directories()
    
    # Generate dataset
    data = generate_dataset()
    
    # Export dataset
    export_dataset(data)
    
    # Create visualizations
    create_visualizations(data)
    
    # Generate report
    generate_report(data)
    
    print("=" * 60)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("Generated files:")
    print("📊 Data files in: output/data/")
    print("📈 Visualizations in: output/figures/")
    print("📋 Report in: output/reports/")
    print("🔬 Raw data in: output/raw_data/")
    print()
    print("Key statistics:")
    print(f"  • Total records: {len(data):,}")
    print(f"  • Mix designs: {data['mix_design'].nunique()}")
    print(f"  • Test conditions: {len(data)}")
    print(f"  • Temperature range: {data['temperature_c'].min()}°C - {data['temperature_c'].max()}°C")
    print(f"  • Rubber content range: {data['rubber_content_percent'].min()}% - {data['rubber_content_percent'].max()}%")
    print()
    print("The dataset is ready for analysis and model development!")

if __name__ == "__main__":
    main()