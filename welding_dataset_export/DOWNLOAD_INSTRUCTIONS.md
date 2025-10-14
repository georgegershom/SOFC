
# Dataset Download Instructions

## Package Information
- **File**: welding_inverse_design_dataset_20251014_070603.zip
- **Size**: 9.85 MB
- **Format**: zip
- **Created**: 20251014_070603

## Contents
- `welding_dataset/`: Complete dataset files (CSV, Parquet)
- `ml_ready_data/`: Preprocessed ML-ready numpy arrays
- `analysis_report/`: Analysis visualizations
- Python scripts for generation and analysis
- Documentation and requirements

## How to Use

1. **Extract the archive**:
   ```bash
   unzip welding_inverse_design_dataset_20251014_070603.zip
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
