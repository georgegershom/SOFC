"""
Export and package the welding dataset for download and distribution
"""

import os
import shutil
import tarfile
import zipfile
import json
from datetime import datetime


def create_dataset_package(format='zip'):
    """
    Package the complete dataset for easy download and distribution
    
    Parameters:
    -----------
    format: str
        'zip' or 'tar.gz'
    """
    
    # Create export directory
    export_dir = 'welding_dataset_export'
    os.makedirs(export_dir, exist_ok=True)
    
    # Define what to include
    include_items = [
        'welding_dataset',           # Main dataset files
        'ml_ready_data',             # ML-ready preprocessed data
        'analysis_report',           # Analysis visualizations
        'README.md',                 # Documentation
        'requirements.txt',          # Dependencies
        'welding_dataset_generator.py',  # Generator script
        'dataset_analysis.py',      # Analysis tools
        'inverse_design_model.py',  # Model implementations
        'main.py'                    # Main execution script
    ]
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if format == 'zip':
        archive_name = f'welding_inverse_design_dataset_{timestamp}.zip'
        archive_path = os.path.join(export_dir, archive_name)
        
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for item in include_items:
                if os.path.exists(item):
                    if os.path.isdir(item):
                        # Add directory and all its contents
                        for root, dirs, files in os.walk(item):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path)
                                zipf.write(file_path, arcname)
                                print(f"  Added: {arcname}")
                    else:
                        # Add single file
                        zipf.write(item, item)
                        print(f"  Added: {item}")
    
    elif format == 'tar.gz':
        archive_name = f'welding_inverse_design_dataset_{timestamp}.tar.gz'
        archive_path = os.path.join(export_dir, archive_name)
        
        with tarfile.open(archive_path, 'w:gz') as tarf:
            for item in include_items:
                if os.path.exists(item):
                    tarf.add(item, arcname=item)
                    print(f"  Added: {item}")
    
    else:
        raise ValueError("Format must be 'zip' or 'tar.gz'")
    
    # Get archive size
    archive_size = os.path.getsize(archive_path) / (1024 * 1024)  # in MB
    
    # Create metadata file
    metadata = {
        'archive_name': archive_name,
        'format': format,
        'size_mb': round(archive_size, 2),
        'created': timestamp,
        'contents': include_items,
        'dataset_stats': {
            'total_samples': 11500,
            'tier1_experimental': 500,
            'tier2_simulation': 10000,
            'tier3_literature': 1000,
            'input_features': 13,
            'output_features': 19,
            'total_features': 67
        }
    }
    
    metadata_path = os.path.join(export_dir, f'package_metadata_{timestamp}.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Dataset package created successfully!")
    print(f"📦 Archive: {archive_path}")
    print(f"📏 Size: {archive_size:.2f} MB")
    print(f"📋 Metadata: {metadata_path}")
    
    # Create download instructions
    instructions = f"""
# Dataset Download Instructions

## Package Information
- **File**: {archive_name}
- **Size**: {archive_size:.2f} MB
- **Format**: {format}
- **Created**: {timestamp}

## Contents
- `welding_dataset/`: Complete dataset files (CSV, Parquet)
- `ml_ready_data/`: Preprocessed ML-ready numpy arrays
- `analysis_report/`: Analysis visualizations
- Python scripts for generation and analysis
- Documentation and requirements

## How to Use

1. **Extract the archive**:
   ```bash
   {'unzip ' + archive_name if format == 'zip' else 'tar -xzf ' + archive_name}
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Load the dataset**:
   ```python
   import pandas as pd
   
   # Load complete dataset
   df = pd.read_csv('welding_dataset/complete_dataset.csv')
   
   # Or use the efficient Parquet format
   df = pd.read_parquet('welding_dataset/complete_dataset.parquet')
   ```

4. **Use ML-ready data**:
   ```python
   import numpy as np
   
   # Load preprocessed data
   X_train = np.load('ml_ready_data/X_train.npy')
   Y_train = np.load('ml_ready_data/Y_train.npy')
   ```

5. **Run analysis**:
   ```python
   from dataset_analysis import WeldingDatasetAnalyzer
   
   analyzer = WeldingDatasetAnalyzer('welding_dataset/complete_dataset.csv')
   analyzer.generate_analysis_report()
   ```

## Dataset Statistics
- Total Samples: 11,500
- Tier 1 (Experimental): 500 samples
- Tier 2 (Simulation): 10,000 samples
- Tier 3 (Literature): 1,000 samples
- Input Features: 13
- Output Features: 19
- Total Features: 67

## Support
For questions or issues, refer to the README.md file included in the package.
"""
    
    instructions_path = os.path.join(export_dir, 'DOWNLOAD_INSTRUCTIONS.md')
    with open(instructions_path, 'w') as f:
        f.write(instructions)
    
    print(f"📄 Instructions: {instructions_path}")
    
    return archive_path, metadata


def create_minimal_dataset(output_dir='minimal_dataset'):
    """
    Create a minimal version of the dataset for quick testing
    """
    import pandas as pd
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load full dataset
    df = pd.read_csv('welding_dataset/complete_dataset.csv')
    
    # Sample smaller subset
    df_minimal = pd.concat([
        df[df['data_tier'] == 1].sample(n=50, random_state=42),  # 50 experimental
        df[df['data_tier'] == 2].sample(n=400, random_state=42),  # 400 simulation
        df[df['data_tier'] == 3].sample(n=50, random_state=42),   # 50 literature
    ])
    
    # Save minimal dataset
    df_minimal.to_csv(f'{output_dir}/minimal_dataset.csv', index=False)
    
    print(f"\n✅ Minimal dataset created: {output_dir}/minimal_dataset.csv")
    print(f"   Samples: {len(df_minimal)} (reduced from {len(df)})")
    
    return df_minimal


if __name__ == "__main__":
    print("=" * 60)
    print("WELDING DATASET EXPORT TOOL")
    print("=" * 60)
    
    # Check if dataset exists
    if not os.path.exists('welding_dataset/complete_dataset.csv'):
        print("❌ Dataset not found! Please run main.py first to generate the dataset.")
        exit(1)
    
    print("\nSelect export format:")
    print("1. ZIP (recommended for Windows)")
    print("2. TAR.GZ (recommended for Linux/Mac)")
    print("3. Create minimal dataset (for testing)")
    print("4. Both ZIP and TAR.GZ")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        create_dataset_package('zip')
    elif choice == '2':
        create_dataset_package('tar.gz')
    elif choice == '3':
        create_minimal_dataset()
    elif choice == '4':
        print("\nCreating ZIP package...")
        create_dataset_package('zip')
        print("\nCreating TAR.GZ package...")
        create_dataset_package('tar.gz')
    else:
        print("Invalid choice!")
    
    print("\n✨ Export complete!")