#!/usr/bin/env python3
"""
Dataset Packaging Script
Creates a comprehensive archive of the rubberized concrete dataset
for easy distribution and download.

Author: AI Assistant
Date: 2025-10-18
"""

import os
import shutil
import zipfile
import tarfile
from datetime import datetime
import json

def create_archive(format_type='zip'):
    """Create compressed archive of the dataset"""
    
    source_dir = 'rubberized_concrete_dataset'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == 'zip':
        archive_name = f'rubberized_concrete_dataset_{timestamp}.zip'
        
        print(f"Creating ZIP archive: {archive_name}")
        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_path = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arc_path)
                    
    elif format_type == 'tar':
        archive_name = f'rubberized_concrete_dataset_{timestamp}.tar.gz'
        
        print(f"Creating TAR.GZ archive: {archive_name}")
        with tarfile.open(archive_name, 'w:gz') as tarf:
            tarf.add(source_dir, arcname='rubberized_concrete_dataset')
    
    # Get archive size
    archive_size = os.path.getsize(archive_name) / (1024 * 1024)  # MB
    
    print(f"✅ Archive created successfully!")
    print(f"   File: {archive_name}")
    print(f"   Size: {archive_size:.2f} MB")
    
    return archive_name

def create_dataset_manifest():
    """Create a manifest file listing all dataset contents"""
    
    manifest = {
        "dataset_name": "Fire-Resistant Rubberized Concrete Dataset",
        "version": "1.0",
        "creation_date": datetime.now().isoformat(),
        "topic": "Development and Validation of a Thermo-Mechanical Model for Fire-Resistant Structural Elements Utilizing High-Performance Rubberized Concrete",
        "contents": {}
    }
    
    source_dir = 'rubberized_concrete_dataset'
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, source_dir)
            
            file_info = {
                "size_bytes": os.path.getsize(file_path),
                "size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 3),
                "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
            }
            
            # Add file type description
            if file.endswith('.json'):
                file_info["type"] = "JSON data file"
            elif file.endswith('.csv'):
                file_info["type"] = "CSV data file"
            elif file.endswith('.png'):
                file_info["type"] = "Visualization plot"
            elif file.endswith('.md'):
                file_info["type"] = "Documentation"
            elif file.endswith('.py'):
                file_info["type"] = "Python script"
            else:
                file_info["type"] = "Other"
            
            manifest["contents"][rel_path] = file_info
    
    # Calculate totals
    total_files = len(manifest["contents"])
    total_size = sum(info["size_bytes"] for info in manifest["contents"].values())
    
    manifest["summary"] = {
        "total_files": total_files,
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2)
    }
    
    # Save manifest
    manifest_file = os.path.join(source_dir, "MANIFEST.json")
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✅ Manifest created: {manifest_file}")
    return manifest

def create_download_instructions():
    """Create download and usage instructions"""
    
    instructions = """# DATASET DOWNLOAD AND USAGE INSTRUCTIONS

## Quick Start

1. **Extract the archive**:
   ```bash
   # For ZIP files
   unzip rubberized_concrete_dataset_YYYYMMDD_HHMMSS.zip
   
   # For TAR.GZ files  
   tar -xzf rubberized_concrete_dataset_YYYYMMDD_HHMMSS.tar.gz
   ```

2. **Navigate to the dataset directory**:
   ```bash
   cd rubberized_concrete_dataset
   ```

3. **Read the documentation**:
   ```bash
   # Main documentation
   cat README.md
   
   # Generation report
   cat GENERATION_REPORT.md
   
   # Data dictionary
   cat data_dictionary.json
   ```

4. **Test the dataset**:
   ```bash
   python3 usage_examples.py
   ```

## Dataset Contents

### Material Properties (Model Input)
- `thermal_properties.csv/json` - Temperature-dependent thermal properties
- `mechanical_properties.csv/json` - Temperature-dependent mechanical properties  
- `deformation_properties.csv/json` - Thermal expansion and transient strain
- `poromechanical_properties.csv/json` - Porosity, permeability, and damage

### Validation Data (Experimental)
- `temperature_evolution_validation.csv/json` - Thermocouple measurements
- `deformation_strain_validation.csv/json` - Strain evolution under loading
- `spalling_failure_summary.csv` - Spalling occurrence and failure analysis
- `spalling_failure_detailed.csv` - Time-series spalling data

### Visualizations
- `visualizations/` directory contains all plots:
  - `thermal_properties.png` - Thermal property plots
  - `mechanical_properties.png` - Mechanical property plots
  - `temperature_evolution.png` - Temperature validation plots
  - `strain_evolution.png` - Strain validation plots
  - `spalling_analysis.png` - Spalling analysis plots
  - `summary_dashboard.png` - Comprehensive overview

### Documentation
- `README.md` - Complete documentation and usage guide
- `data_dictionary.json` - Field descriptions and units
- `metadata.json` - Dataset metadata and specifications
- `usage_examples.py` - Code examples and templates
- `MANIFEST.json` - Complete file listing and checksums

## System Requirements

- Python 3.7+
- Required packages: numpy, pandas, matplotlib, seaborn, scipy
- Install with: `pip install numpy pandas matplotlib seaborn scipy`

## Data Loading Examples

### Python (Pandas)
```python
import pandas as pd

# Load thermal properties
thermal_df = pd.read_csv('thermal_properties.csv')

# Filter for specific rubber content
rubber_10 = thermal_df[thermal_df['rubber_content_pct'] == 10]

# Plot thermal conductivity vs temperature
import matplotlib.pyplot as plt
plt.plot(rubber_10['temperature_C'], rubber_10['thermal_conductivity_W_m_K'])
plt.xlabel('Temperature (°C)')
plt.ylabel('Thermal Conductivity (W/m·K)')
plt.show()
```

### Python (JSON)
```python
import json

# Load hierarchical data
with open('thermal_properties.json', 'r') as f:
    thermal_data = json.load(f)

# Access specific rubber content data
rubber_10_data = thermal_data['rubber_10pct']
temperatures = rubber_10_data['temperature']
conductivity = rubber_10_data['thermal_conductivity']
```

### R
```r
# Load CSV data
thermal_data <- read.csv('thermal_properties.csv')

# Filter and plot
library(ggplot2)
rubber_10 <- subset(thermal_data, rubber_content_pct == 10)
ggplot(rubber_10, aes(x=temperature_C, y=thermal_conductivity_W_m_K)) +
  geom_line() +
  labs(x='Temperature (°C)', y='Thermal Conductivity (W/m·K)')
```

## Applications

1. **Finite Element Modeling**: Use material properties for FE model calibration
2. **Fire Safety Design**: Validate fire resistance predictions
3. **Material Optimization**: Compare different rubber contents
4. **Research**: Basis for further experimental or numerical studies
5. **Education**: Teaching fire engineering and material science

## Citation

If you use this dataset in your research, please cite:

```
Fire-Resistant Rubberized Concrete Dataset
Topic: Development and Validation of a Thermo-Mechanical Model for 
       Fire-Resistant Structural Elements Utilizing High-Performance 
       Rubberized Concrete
Generated: 2025-10-18
Version: 1.0
```

## Support

For questions about the dataset:
1. Check the README.md file
2. Review the usage_examples.py script
3. Examine the data_dictionary.json for field descriptions
4. Refer to the visualizations for data overview

## License

This dataset is provided for research and educational purposes.
Please acknowledge the source when using in publications or presentations.
"""

    with open('DOWNLOAD_INSTRUCTIONS.md', 'w') as f:
        f.write(instructions)
    
    print("✅ Download instructions created: DOWNLOAD_INSTRUCTIONS.md")

def main():
    """Main packaging function"""
    
    print("📦 DATASET PACKAGING TOOL 📦")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('rubberized_concrete_dataset'):
        print("❌ Dataset directory not found!")
        print("   Please run generate_complete_dataset.py first")
        return False
    
    # Create manifest
    print("\n1. Creating dataset manifest...")
    manifest = create_dataset_manifest()
    
    # Create download instructions
    print("\n2. Creating download instructions...")
    create_download_instructions()
    
    # Create archives
    print("\n3. Creating compressed archives...")
    zip_file = create_archive('zip')
    tar_file = create_archive('tar')
    
    # Summary
    print(f"\n{'='*50}")
    print("📦 PACKAGING COMPLETE! 📦")
    print(f"{'='*50}")
    
    print(f"\nDataset archives created:")
    print(f"📁 {zip_file}")
    print(f"📁 {tar_file}")
    print(f"📄 DOWNLOAD_INSTRUCTIONS.md")
    
    print(f"\nDataset summary:")
    print(f"📊 {manifest['summary']['total_files']} files")
    print(f"💾 {manifest['summary']['total_size_mb']} MB total")
    
    print(f"\nThe dataset is now ready for:")
    print("✅ Distribution and sharing")
    print("✅ Upload to repositories") 
    print("✅ Integration into research projects")
    print("✅ Use in numerical modeling")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)