
import pandas as pd
import numpy as np
import json
from pathlib import Path

def validate_datasets():
    """Validate the generated datasets"""
    base_path = Path("/workspace/data")
    
    # Load all datasets
    datasets = {}
    for category in ['weather_climate', 'economic_market', 'geospatial_regulatory']:
        category_path = base_path / category
        for file in category_path.glob('*.csv'):
            datasets[file.stem] = pd.read_csv(file)
    
    # Validation results
    validation_results = {}
    
    for name, df in datasets.items():
        results = {
            'total_records': len(df),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'numeric_ranges': {}
        }
        
        # Check numeric ranges
        for col in df.select_dtypes(include=[np.number]).columns:
            try:
                results['numeric_ranges'][col] = {
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else None
                }
            except (ValueError, TypeError):
                results['numeric_ranges'][col] = {
                    'min': None, 'max': None, 'mean': None
                }
        
        validation_results[name] = results
    
    # Save validation results
    with open(base_path / 'integrated' / 'validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print("Dataset validation completed!")
    return validation_results

if __name__ == "__main__":
    validate_datasets()
