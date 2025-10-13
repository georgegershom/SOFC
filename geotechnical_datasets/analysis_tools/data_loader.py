#!/usr/bin/env python3
"""
Data Loader Module for Geotechnical Datasets
Provides functions to load and preprocess all dataset types
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class GeotechnicalDataLoader:
    """Main class for loading geotechnical datasets"""
    
    def __init__(self, base_path='..'):
        """Initialize with base path to datasets"""
        self.base_path = Path(base_path)
        self.sandy_path = self.base_path / 'sandy_soils'
        self.clay_path = self.base_path / 'clay_soils'
        self.case_path = self.base_path / 'case_studies'
        self.spatial_path = self.base_path / 'spatial_data'
        
    def load_sandy_soils(self):
        """Load all sandy soil datasets"""
        datasets = {}
        
        # Load basic properties
        datasets['basic'] = pd.read_csv(self.sandy_path / 'sand_basic_properties.csv')
        
        # Load mechanical properties
        datasets['mechanical'] = pd.read_csv(self.sandy_path / 'sand_mechanical_properties.csv')
        
        # Load liquefaction data
        datasets['liquefaction'] = pd.read_csv(self.sandy_path / 'liquefaction_data.csv')
        
        # Merge datasets on sample_id
        merged = datasets['basic'].merge(
            datasets['mechanical'], on='sample_id', how='outer'
        ).merge(
            datasets['liquefaction'], on='sample_id', how='outer'
        )
        
        return datasets, merged
    
    def load_clay_soils(self):
        """Load all clay soil datasets"""
        datasets = {}
        
        # Load basic properties
        datasets['basic'] = pd.read_csv(self.clay_path / 'clay_basic_properties.csv')
        
        # Load mechanical properties
        datasets['mechanical'] = pd.read_csv(self.clay_path / 'clay_mechanical_properties.csv')
        
        # Load mineralogy
        datasets['mineralogy'] = pd.read_csv(self.clay_path / 'clay_mineralogy.csv')
        
        # Load slip surface data
        datasets['slip_surface'] = pd.read_csv(self.clay_path / 'slip_surface_data.csv')
        
        # Merge datasets
        merged = datasets['basic'].merge(
            datasets['mechanical'], on='sample_id', how='outer'
        ).merge(
            datasets['mineralogy'], on='sample_id', how='outer'
        ).merge(
            datasets['slip_surface'], on='sample_id', how='outer'
        )
        
        return datasets, merged
    
    def load_case_studies(self):
        """Load case study datasets"""
        datasets = {}
        
        # Load failure cases
        datasets['failures'] = pd.read_csv(self.case_path / 'underground_structure_failures.csv')
        
        # Load monitoring data
        datasets['monitoring'] = pd.read_csv(self.case_path / 'structural_response_monitoring.csv')
        
        # Convert timestamp to datetime
        datasets['monitoring']['timestamp'] = pd.to_datetime(datasets['monitoring']['timestamp'])
        
        return datasets
    
    def load_spatial_data(self):
        """Load spatial datasets"""
        datasets = {}
        
        # Load GeoJSON
        with open(self.spatial_path / 'regional_soil_properties.geojson', 'r') as f:
            datasets['geojson'] = json.load(f)
        
        # Load grid data
        datasets['grid'] = pd.read_csv(self.spatial_path / 'grid_soil_data.csv')
        
        return datasets
    
    def get_summary_statistics(self, df):
        """Generate summary statistics for a dataframe"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': df[numeric_cols].describe().to_dict()
        }
        
        return summary
    
    def filter_by_property(self, df, property_name, min_val=None, max_val=None):
        """Filter dataframe by property range"""
        if property_name not in df.columns:
            raise ValueError(f"Property {property_name} not found in dataframe")
        
        filtered = df.copy()
        
        if min_val is not None:
            filtered = filtered[filtered[property_name] >= min_val]
        
        if max_val is not None:
            filtered = filtered[filtered[property_name] <= max_val]
        
        return filtered
    
    def calculate_correlations(self, df, target_col):
        """Calculate correlations with target column"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if target_col not in numeric_cols:
            raise ValueError(f"Target column {target_col} is not numeric")
        
        correlations = {}
        for col in numeric_cols:
            if col != target_col:
                corr = df[col].corr(df[target_col])
                if not np.isnan(corr):
                    correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_corr = dict(sorted(correlations.items(), 
                                 key=lambda x: abs(x[1]), 
                                 reverse=True))
        
        return sorted_corr

# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = GeotechnicalDataLoader()
    
    # Load sandy soils
    print("Loading Sandy Soil Data...")
    sandy_data, sandy_merged = loader.load_sandy_soils()
    print(f"Loaded {len(sandy_merged)} sandy soil samples")
    
    # Load clay soils
    print("\nLoading Clay Soil Data...")
    clay_data, clay_merged = loader.load_clay_soils()
    print(f"Loaded {len(clay_merged)} clay soil samples")
    
    # Load case studies
    print("\nLoading Case Studies...")
    case_data = loader.load_case_studies()
    print(f"Loaded {len(case_data['failures'])} failure cases")
    print(f"Loaded {len(case_data['monitoring'])} monitoring records")
    
    # Load spatial data
    print("\nLoading Spatial Data...")
    spatial_data = loader.load_spatial_data()
    print(f"Loaded {len(spatial_data['geojson']['features'])} geographic features")
    print(f"Loaded {len(spatial_data['grid'])} grid points")
    
    # Example: Get correlations for sandy soils
    print("\nTop 5 correlations with liquefaction potential (CSR):")
    sandy_corr = loader.calculate_correlations(sandy_merged, 'CSR')
    for param, corr in list(sandy_corr.items())[:5]:
        print(f"  {param}: {corr:.3f}")