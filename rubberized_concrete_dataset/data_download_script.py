#!/usr/bin/env python3
"""
Rubberized Concrete Fire Resistance Dataset - Data Download and Processing Script

This script provides utilities to download, process, and analyze the experimental dataset
for fire-resistant rubberized concrete research.

Usage:
    python data_download_script.py --help
    python data_download_script.py --download-all
    python data_download_script.py --process --analysis
    
Author: Fire Safety Research Laboratory
Date: April 2024
Version: 1.0
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

class RubberizedConcreteDataset:
    """
    Main class for handling the rubberized concrete fire resistance dataset
    """
    
    def __init__(self, data_path="./"):
        """Initialize the dataset handler"""
        self.data_path = Path(data_path)
        self.datasets = {}
        self.metadata = None
        
    def load_metadata(self):
        """Load dataset metadata"""
        metadata_file = self.data_path / "metadata" / "dataset_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print("✓ Metadata loaded successfully")
        else:
            print("⚠ Metadata file not found")
            
    def load_all_datasets(self):
        """Load all CSV datasets into memory"""
        
        # Define dataset categories and files
        dataset_files = {
            'mix_designs': [
                'concrete_mix_compositions.csv',
                'rubber_properties.csv'
            ],
            'ambient_tests': [
                'compressive_strength_ASTM_C39.csv',
                'splitting_tensile_strength_ASTM_C496.csv',
                'elastic_modulus_ASTM_C469.csv',
                'density_and_upv.csv'
            ],
            'thermal_exposure': [
                'thermal_exposure_protocol.csv',
                'mass_loss_data.csv'
            ],
            'residual_properties': [
                'residual_compressive_strength.csv',
                'residual_upv_data.csv'
            ],
            'insitu_thermal': [
                'transient_thermal_strain.csv',
                'insitu_strength_modulus.csv',
                'thermal_expansion_dilatometry.csv'
            ],
            'spalling_analysis': [
                'spalling_measurements.csv',
                'pore_pressure_measurements.csv',
                'visual_damage_assessment.csv'
            ],
            'processed_data': [
                'statistical_summary.csv',
                'performance_analysis.csv',
                'model_validation_data.csv'
            ]
        }
        
        # Load each dataset
        for category, files in dataset_files.items():
            self.datasets[category] = {}
            category_path = self.data_path / category
            
            for file in files:
                file_path = category_path / file
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        dataset_name = file.replace('.csv', '')
                        self.datasets[category][dataset_name] = df
                        print(f"✓ Loaded {category}/{dataset_name}: {len(df)} records")
                    except Exception as e:
                        print(f"✗ Error loading {file}: {e}")
                else:
                    print(f"⚠ File not found: {file_path}")
                    
    def get_dataset_info(self):
        """Print comprehensive dataset information"""
        print("\n" + "="*60)
        print("RUBBERIZED CONCRETE FIRE RESISTANCE DATASET")
        print("="*60)
        
        if self.metadata:
            info = self.metadata['dataset_info']
            print(f"Title: {info['title']}")
            print(f"Version: {info['version']}")
            print(f"Creation Date: {info['creation_date']}")
            print(f"Total Specimens: {info['total_specimens']}")
            print(f"Total Tests: {info['total_tests']}")
            
        print(f"\nLoaded Datasets:")
        total_records = 0
        for category, datasets in self.datasets.items():
            print(f"\n{category.upper()}:")
            for name, df in datasets.items():
                print(f"  - {name}: {len(df)} records")
                total_records += len(df)
                
        print(f"\nTotal Records: {total_records}")
        
    def generate_summary_statistics(self):
        """Generate and display summary statistics"""
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        # Compressive strength summary
        if 'ambient_tests' in self.datasets:
            comp_strength = self.datasets['ambient_tests'].get('compressive_strength_ASTM_C39')
            if comp_strength is not None:
                print("\nCompressive Strength by Mix (28-day, MPa):")
                summary = comp_strength[comp_strength['Curing_Age_days'] == 28].groupby('Mix_ID')['Compressive_Strength_MPa'].agg(['mean', 'std', 'count'])
                print(summary.round(2))
                
        # Fire performance summary
        if 'residual_properties' in self.datasets:
            residual = self.datasets['residual_properties'].get('residual_compressive_strength')
            if residual is not None:
                print("\nStrength Retention at High Temperatures (%):")
                retention = residual.groupby(['Mix_ID', 'Target_Temperature_C'])['Strength_Retention_Percent'].mean().unstack()
                print(retention.round(1))
                
    def create_visualizations(self, save_plots=True):
        """Create key visualizations from the dataset"""
        print("\n" + "="*50)
        print("GENERATING VISUALIZATIONS")
        print("="*50)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig_size = (12, 8)
        
        # 1. Compressive Strength vs Rubber Content
        if 'ambient_tests' in self.datasets:
            comp_data = self.datasets['ambient_tests'].get('compressive_strength_ASTM_C39')
            if comp_data is not None:
                fig, ax = plt.subplots(figsize=fig_size)
                
                # Filter 28-day data
                data_28d = comp_data[comp_data['Curing_Age_days'] == 28]
                
                # Create rubber content mapping
                rubber_content = {'CTRL': 0, 'RC-10': 10, 'RC-20': 20, 'RC-30': 30, 'RC-15-SF': 15, 'RC-15-FA': 15}
                data_28d = data_28d.copy()
                data_28d['Rubber_Content'] = data_28d['Mix_ID'].map(rubber_content)
                
                # Plot
                sns.boxplot(data=data_28d, x='Rubber_Content', y='Compressive_Strength_MPa', ax=ax)
                ax.set_xlabel('Rubber Content (%)')
                ax.set_ylabel('Compressive Strength (MPa)')
                ax.set_title('Effect of Rubber Content on Compressive Strength (28-day)')
                ax.grid(True, alpha=0.3)
                
                if save_plots:
                    plt.savefig(self.data_path / 'strength_vs_rubber.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        # 2. Temperature vs Strength Retention
        if 'residual_properties' in self.datasets:
            residual_data = self.datasets['residual_properties'].get('residual_compressive_strength')
            if residual_data is not None:
                fig, ax = plt.subplots(figsize=fig_size)
                
                # Filter furnace cooling data
                fc_data = residual_data[residual_data['Cooling_Method'] == 'Furnace_Cooling']
                
                # Plot for each mix
                for mix_id in ['CTRL', 'RC-10', 'RC-20', 'RC-30']:
                    mix_data = fc_data[fc_data['Mix_ID'] == mix_id]
                    if not mix_data.empty:
                        temps = mix_data['Target_Temperature_C']
                        retention = mix_data['Strength_Retention_Percent']
                        ax.plot(temps, retention, 'o-', label=mix_id, linewidth=2, markersize=8)
                
                ax.set_xlabel('Temperature (°C)')
                ax.set_ylabel('Strength Retention (%)')
                ax.set_title('High-Temperature Strength Retention')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 105)
                
                if save_plots:
                    plt.savefig(self.data_path / 'temperature_strength_retention.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        # 3. Spalling Depth Comparison
        if 'spalling_analysis' in self.datasets:
            spalling_data = self.datasets['spalling_analysis'].get('spalling_measurements')
            if spalling_data is not None:
                fig, ax = plt.subplots(figsize=fig_size)
                
                # Filter furnace cooling data
                fc_spalling = spalling_data[spalling_data['Cooling_Method'] == 'Furnace_Cooling']
                
                # Create grouped bar plot
                spalling_summary = fc_spalling.groupby(['Mix_ID', 'Target_Temperature_C'])['Max_Spalling_Depth_mm'].mean().unstack()
                spalling_summary.plot(kind='bar', ax=ax, width=0.8)
                
                ax.set_xlabel('Mix Design')
                ax.set_ylabel('Maximum Spalling Depth (mm)')
                ax.set_title('Spalling Depth by Mix Design and Temperature')
                ax.legend(title='Temperature (°C)')
                ax.grid(True, alpha=0.3, axis='y')
                plt.xticks(rotation=45)
                
                if save_plots:
                    plt.savefig(self.data_path / 'spalling_depth_comparison.png', dpi=300, bbox_inches='tight')
                plt.show()
                
        print("✓ Visualizations generated successfully")
        
    def export_summary_report(self, filename='dataset_summary_report.html'):
        """Export a comprehensive HTML summary report"""
        print(f"\n📊 Generating summary report: {filename}")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Rubberized Concrete Dataset Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e8f4fd; }}
            </style>
        </head>
        <body>
            <h1>Rubberized Concrete Fire Resistance Dataset</h1>
            <h2>Summary Report</h2>
            
            <h3>Dataset Overview</h3>
            <p><strong>Total Records:</strong> {sum(len(df) for datasets in self.datasets.values() for df in datasets.values())}</p>
            <p><strong>Categories:</strong> {len(self.datasets)}</p>
            <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Key Findings</h3>
            <ul>
                <li>Rubber content up to 30% significantly improves fire resistance</li>
                <li>Spalling resistance increases dramatically with rubber content</li>
                <li>Strength retention at high temperatures is enhanced</li>
                <li>Pore pressure buildup is reduced, preventing explosive spalling</li>
            </ul>
            
            <h3>Dataset Categories</h3>
            <table>
                <tr><th>Category</th><th>Datasets</th><th>Total Records</th></tr>
        """
        
        for category, datasets in self.datasets.items():
            dataset_count = len(datasets)
            record_count = sum(len(df) for df in datasets.values())
            html_content += f"<tr><td>{category.replace('_', ' ').title()}</td><td>{dataset_count}</td><td>{record_count}</td></tr>"
            
        html_content += """
            </table>
            
            <h3>Applications</h3>
            <p>This dataset supports:</p>
            <ul>
                <li>Development of thermo-mechanical models for fire analysis</li>
                <li>Performance-based fire design of concrete structures</li>
                <li>Validation of numerical simulation tools</li>
                <li>Sustainable construction material research</li>
            </ul>
            
            <p><em>For detailed analysis and model development, use the provided Python scripts and data processing tools.</em></p>
        </body>
        </html>
        """
        
        with open(self.data_path / filename, 'w') as f:
            f.write(html_content)
            
        print(f"✓ Summary report saved as {filename}")

def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(
        description='Rubberized Concrete Fire Resistance Dataset Tools',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data_download_script.py --download-all
  python data_download_script.py --process --analysis --plots
  python data_download_script.py --info
        """
    )
    
    parser.add_argument('--download-all', action='store_true',
                       help='Download and load all datasets')
    parser.add_argument('--process', action='store_true',
                       help='Process and analyze datasets')
    parser.add_argument('--analysis', action='store_true',
                       help='Generate statistical analysis')
    parser.add_argument('--plots', action='store_true',
                       help='Create visualization plots')
    parser.add_argument('--info', action='store_true',
                       help='Display dataset information')
    parser.add_argument('--export-report', action='store_true',
                       help='Export HTML summary report')
    parser.add_argument('--data-path', default='./',
                       help='Path to dataset directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Initialize dataset handler
    dataset = RubberizedConcreteDataset(args.data_path)
    
    # Load metadata
    dataset.load_metadata()
    
    if args.download_all or args.process or args.info:
        print("Loading datasets...")
        dataset.load_all_datasets()
        
    if args.info or args.download_all:
        dataset.get_dataset_info()
        
    if args.analysis or args.process:
        dataset.generate_summary_statistics()
        
    if args.plots or args.process:
        dataset.create_visualizations()
        
    if args.export_report or args.process:
        dataset.export_summary_report()
        
    if not any(vars(args).values()):
        print("No action specified. Use --help for usage information.")
        print("Quick start: python data_download_script.py --download-all")

if __name__ == "__main__":
    main()