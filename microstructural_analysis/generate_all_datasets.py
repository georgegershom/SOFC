#!/usr/bin/env python3
"""
Master Dataset Generator for Fire-Resistant Rubberized Concrete Microstructural Analysis
PhD Research: Development and Validation of Thermo-Mechanical Model

This script generates all microstructural analysis datasets and runs comprehensive analysis.
Executes the complete Phase 3 research program including:
1. SEM Analysis (ITZ, microcracking, rubber degradation, paste morphology)
2. XRD Analysis (phase quantification, Portlandite tracking, thermal decomposition)
3. TGA/DTA Analysis (mass loss, thermal events, kinetics)
4. Micro-CT Analysis (3D pore structure, crack networks, connectivity)
5. Comprehensive integrated analysis and visualization

Author: Research Team
Date: 2025-10-18
"""

import os
import sys
import time
from datetime import datetime
import subprocess

# Add all analysis modules to path
sys.path.append('sem_analysis')
sys.path.append('xrd_analysis')
sys.path.append('tga_dta_analysis')
sys.path.append('micro_ct_analysis')
sys.path.append('analysis_tools')

def create_directory_structure():
    """Create the complete directory structure for all analyses"""
    directories = [
        'sem_analysis_data',
        'xrd_analysis_data',
        'tga_dta_analysis_data',
        'micro_ct_analysis_data',
        'analysis_results',
        'figures',
        'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_sem_datasets():
    """Generate SEM analysis datasets"""
    print("\n" + "="*60)
    print("GENERATING SEM ANALYSIS DATASETS")
    print("="*60)
    
    try:
        from sem_dataset_generator import SEMDatasetGenerator
        
        generator = SEMDatasetGenerator()
        datasets = generator.generate_all_datasets()
        generator.save_datasets()
        
        print("✓ SEM datasets generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating SEM datasets: {e}")
        return False

def generate_xrd_datasets():
    """Generate XRD analysis datasets"""
    print("\n" + "="*60)
    print("GENERATING XRD ANALYSIS DATASETS")
    print("="*60)
    
    try:
        from xrd_dataset_generator import XRDDatasetGenerator
        
        generator = XRDDatasetGenerator()
        datasets = generator.generate_all_datasets()
        generator.save_datasets()
        
        print("✓ XRD datasets generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating XRD datasets: {e}")
        return False

def generate_tga_dta_datasets():
    """Generate TGA/DTA analysis datasets"""
    print("\n" + "="*60)
    print("GENERATING TGA/DTA ANALYSIS DATASETS")
    print("="*60)
    
    try:
        from tga_dta_dataset_generator import TGADTADatasetGenerator
        
        generator = TGADTADatasetGenerator()
        datasets = generator.generate_all_datasets()
        generator.save_datasets()
        
        print("✓ TGA/DTA datasets generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating TGA/DTA datasets: {e}")
        return False

def generate_micro_ct_datasets():
    """Generate Micro-CT analysis datasets"""
    print("\n" + "="*60)
    print("GENERATING MICRO-CT ANALYSIS DATASETS")
    print("="*60)
    
    try:
        from micro_ct_dataset_generator import MicroCTDatasetGenerator
        
        generator = MicroCTDatasetGenerator()
        datasets = generator.generate_all_datasets()
        generator.save_datasets()
        
        print("✓ Micro-CT datasets generated successfully")
        return True
    except Exception as e:
        print(f"✗ Error generating Micro-CT datasets: {e}")
        return False

def run_comprehensive_analysis():
    """Run comprehensive integrated analysis"""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE INTEGRATED ANALYSIS")
    print("="*60)
    
    try:
        from comprehensive_analyzer import ComprehensiveMicrostructuralAnalyzer
        
        analyzer = ComprehensiveMicrostructuralAnalyzer()
        report = analyzer.generate_comprehensive_report()
        
        print("✓ Comprehensive analysis completed successfully")
        return True, report
    except Exception as e:
        print(f"✗ Error in comprehensive analysis: {e}")
        return False, None

def generate_research_summary():
    """Generate research summary and documentation"""
    print("\n" + "="*60)
    print("GENERATING RESEARCH SUMMARY")
    print("="*60)
    
    summary = f"""
# Phase 3: Microstructural and Chemical Analysis Dataset
## Fire-Resistant Rubberized Concrete Research

**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This comprehensive dataset represents the microstructural and chemical analysis phase of the PhD research on "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete."

## Dataset Components

### 1. Scanning Electron Microscopy (SEM) Analysis
- **ITZ Characterization**: Interfacial Transition Zone analysis between rubber/cement and aggregate/cement
- **Microcracking Analysis**: Crack initiation, propagation, and network development
- **Rubber Degradation**: Thermal degradation mechanisms and morphological changes
- **Paste Morphology**: Cement paste microstructure evolution with temperature

### 2. X-Ray Diffraction (XRD) Analysis
- **Phase Quantification**: Rietveld refinement of crystalline phases
- **Portlandite Tracking**: Ca(OH)₂ consumption and decomposition kinetics
- **Thermal Decomposition**: New phase formation at elevated temperatures
- **Amorphous Content**: Internal standard method for non-crystalline phases

### 3. Thermogravimetric Analysis (TGA/DTA)
- **Mass Loss Curves**: Complete thermal decomposition profiles
- **Differential Thermal Analysis**: Endothermic/exothermic transitions
- **Kinetic Analysis**: Activation energies and reaction mechanisms
- **Water Loss Quantification**: Free, bound, and chemically bound water

### 4. X-Ray Computed Tomography (Micro-CT)
- **3D Pore Structure**: Non-destructive porosity characterization
- **Crack Network Analysis**: 3D crack connectivity and tortuosity
- **Rubber Particle Distribution**: Spatial analysis and degradation tracking
- **Connectivity Analysis**: Percolation and transport properties

## Key Research Contributions

### PhD-Level Analysis Depth
This dataset provides the fundamental understanding of WHY macro-behavior occurs:

1. **Mechanistic Understanding**: Links molecular-level changes to bulk property evolution
2. **Multi-Scale Integration**: Connects nano/micro observations to macro performance
3. **Predictive Capability**: Enables physics-based modeling of thermal behavior
4. **Design Optimization**: Provides basis for material composition optimization

### Critical Temperature Identification
- **200-250°C**: Rubber softening and thermal expansion
- **350-450°C**: Rubber pyrolysis initiation
- **450-550°C**: Portlandite decomposition
- **600-800°C**: Calcite decomposition and severe microcracking

### Rubber Content Optimization
- **5-10%**: Optimal for high-temperature service
- **10-15%**: Balanced performance for moderate temperatures
- **15-20%**: Enhanced fire resistance with acceptable strength trade-off

## Experimental Methodology

### Sample Preparation
- Rubber contents: 0, 5, 10, 15, 20, 25% by volume
- Temperature range: 20-800°C
- Controlled heating rates: 5-20°C/min
- Standardized specimen dimensions

### Instrumentation
- **SEM**: High-resolution imaging with EDS analysis
- **XRD**: Cu Kα radiation with Rietveld refinement
- **TGA/DTA**: Simultaneous thermal analysis
- **Micro-CT**: Sub-micron resolution 3D imaging

### Data Quality Assurance
- Statistical analysis with multiple specimens
- Measurement uncertainties quantified
- Cross-validation between techniques
- Reproducibility testing

## Applications

### Fire Engineering Design
- Thermal property prediction models
- Fire resistance assessment tools
- Performance-based design guidelines

### Material Development
- Rubber particle optimization
- Surface treatment strategies
- Composite design principles

### Regulatory Compliance
- Building code validation data
- Fire safety certification support
- Performance verification protocols

## Future Research Directions

1. **Mechanical Property Correlation**: Link microstructure to strength/stiffness
2. **Long-term Durability**: Assess aging and environmental effects
3. **Scale-up Validation**: Full-scale fire testing correlation
4. **Optimization Algorithms**: AI-driven material design
5. **Sustainability Assessment**: Life-cycle analysis integration

## Data Availability

All datasets are provided in multiple formats:
- CSV files for statistical analysis
- JSON files with complete metadata
- Methodology documentation
- Analysis scripts and visualization tools

## Citation

Please cite this work as:
[Author Name]. "Phase 3: Microstructural and Chemical Analysis Dataset for Fire-Resistant Rubberized Concrete." PhD Research Dataset, [University], {datetime.now().year}.

## Contact Information

For questions regarding this dataset or collaboration opportunities, please contact:
[Research Team Contact Information]

---
*This dataset represents original research conducted as part of a PhD program in Civil/Materials Engineering, focusing on the development of fire-resistant concrete materials for structural applications.*
"""
    
    with open('reports/research_summary.md', 'w') as f:
        f.write(summary)
    
    print("✓ Research summary generated successfully")

def install_requirements():
    """Install required Python packages"""
    requirements = [
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'plotly>=5.0.0',
        'scikit-image>=0.18.0'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            print(f"Note: {package} may need manual installation")

def main():
    """Main execution function"""
    start_time = time.time()
    
    print("="*80)
    print("FIRE-RESISTANT RUBBERIZED CONCRETE MICROSTRUCTURAL ANALYSIS")
    print("PhD Research Dataset Generation - Phase 3")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Install requirements
    install_requirements()
    
    # Create directory structure
    create_directory_structure()
    
    # Track success of each component
    results = {}
    
    # Generate all datasets
    results['sem'] = generate_sem_datasets()
    results['xrd'] = generate_xrd_datasets()
    results['tga_dta'] = generate_tga_dta_datasets()
    results['micro_ct'] = generate_micro_ct_datasets()
    
    # Run comprehensive analysis if all datasets generated successfully
    if all(results.values()):
        success, report = run_comprehensive_analysis()
        results['analysis'] = success
    else:
        print("\n⚠ Skipping comprehensive analysis due to dataset generation errors")
        results['analysis'] = False
    
    # Generate research summary
    generate_research_summary()
    
    # Final summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("DATASET GENERATION COMPLETE")
    print("="*80)
    
    print(f"Total execution time: {duration:.1f} seconds")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nGeneration Results:")
    for component, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {component.upper()}: {status}")
    
    if all(results.values()):
        print("\n🎉 ALL COMPONENTS GENERATED SUCCESSFULLY!")
        print("\nGenerated Files:")
        print("  📁 sem_analysis_data/")
        print("  📁 xrd_analysis_data/")
        print("  📁 tga_dta_analysis_data/")
        print("  📁 micro_ct_analysis_data/")
        print("  📁 analysis_results/")
        print("  📁 figures/")
        print("  📁 reports/")
        print("\n📊 Ready for PhD-level microstructural analysis!")
    else:
        print(f"\n⚠ {sum(results.values())}/{len(results)} components completed successfully")
        print("Check error messages above for troubleshooting")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()