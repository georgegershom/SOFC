"""
Data utility functions for SOFC digital twin dataset.
"""

import numpy as np
import h5py
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import os


class DataUtils:
    """Utility class for data processing and manipulation."""
    
    @staticmethod
    def create_dataframe_from_sensors(sensor_data: Dict[str, Any]) -> pd.DataFrame:
        """Convert sensor data to pandas DataFrame."""
        data = []
        
        for sensor_id, sensor_info in sensor_data.items():
            if 'temperature' in sensor_info:
                for i, (time, temp) in enumerate(zip(sensor_info['time_points'], sensor_info['temperature'])):
                    data.append({
                        'sensor_id': sensor_id,
                        'time': time,
                        'temperature': temp,
                        'sensor_type': 'thermocouple'
                    })
            elif 'strain' in sensor_info:
                for i, (time, strain) in enumerate(zip(sensor_info['time_points'], sensor_info['strain'])):
                    data.append({
                        'sensor_id': sensor_id,
                        'time': time,
                        'strain': strain,
                        'sensor_type': 'strain_gauge'
                    })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def interpolate_data(data: np.ndarray, old_time: np.ndarray, new_time: np.ndarray) -> np.ndarray:
        """Interpolate data to new time points."""
        return np.interp(new_time, old_time, data)
    
    @staticmethod
    def resample_data(data: np.ndarray, old_sampling_rate: float, new_sampling_rate: float) -> np.ndarray:
        """Resample data to new sampling rate."""
        if old_sampling_rate == new_sampling_rate:
            return data
        
        # Calculate new length
        old_length = len(data)
        new_length = int(old_length * new_sampling_rate / old_sampling_rate)
        
        # Create new time arrays
        old_time = np.arange(old_length) / old_sampling_rate
        new_time = np.arange(new_length) / new_sampling_rate
        
        # Interpolate
        return np.interp(new_time, old_time, data)
    
    @staticmethod
    def apply_filter(data: np.ndarray, filter_type: str = 'lowpass', cutoff: float = 0.1) -> np.ndarray:
        """Apply digital filter to data."""
        from scipy import signal
        
        if filter_type == 'lowpass':
            b, a = signal.butter(4, cutoff, btype='low')
            return signal.filtfilt(b, a, data)
        elif filter_type == 'highpass':
            b, a = signal.butter(4, cutoff, btype='high')
            return signal.filtfilt(b, a, data)
        elif filter_type == 'bandpass':
            b, a = signal.butter(4, [cutoff*0.5, cutoff*1.5], btype='band')
            return signal.filtfilt(b, a, data)
        else:
            return data
    
    @staticmethod
    def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """Detect outliers in data."""
        if method == 'iqr':
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (data < lower_bound) | (data > upper_bound)
        elif method == 'zscore':
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            return z_scores > threshold
        else:
            return np.zeros_like(data, dtype=bool)
    
    @staticmethod
    def remove_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 1.5) -> np.ndarray:
        """Remove outliers from data."""
        outliers = DataUtils.detect_outliers(data, method, threshold)
        return data[~outliers]
    
    @staticmethod
    def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics for data."""
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'skewness': DataUtils._calculate_skewness(data),
            'kurtosis': DataUtils._calculate_kurtosis(data)
        }
    
    @staticmethod
    def _calculate_skewness(data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    @staticmethod
    def normalize_data(data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize data."""
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        elif method == 'robust':
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            return (data - median) / (1.4826 * mad)
        else:
            return data
    
    @staticmethod
    def denormalize_data(normalized_data: np.ndarray, original_data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Denormalize data back to original scale."""
        if method == 'minmax':
            return normalized_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)
        elif method == 'zscore':
            return normalized_data * np.std(original_data) + np.mean(original_data)
        elif method == 'robust':
            median = np.median(original_data)
            mad = np.median(np.abs(original_data - median))
            return normalized_data * (1.4826 * mad) + median
        else:
            return normalized_data
    
    @staticmethod
    def create_time_series_features(data: np.ndarray, window_size: int = 10) -> np.ndarray:
        """Create time series features from data."""
        features = []
        
        for i in range(len(data) - window_size + 1):
            window = data[i:i + window_size]
            
            # Basic features
            feature_vector = [
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                np.median(window)
            ]
            
            # Trend features
            if len(window) > 1:
                feature_vector.append(np.polyfit(range(len(window)), window, 1)[0])  # Slope
                feature_vector.append(np.corrcoef(range(len(window)), window)[0, 1])  # Correlation
            
            # Frequency features
            fft = np.fft.fft(window)
            feature_vector.append(np.max(np.abs(fft[1:len(fft)//2])))  # Max frequency component
            
            features.append(feature_vector)
        
        return np.array(features)
    
    @staticmethod
    def merge_datasets(datasets: List[Dict[str, Any]], merge_key: str = 'time_points') -> Dict[str, Any]:
        """Merge multiple datasets on a common key."""
        if not datasets:
            return {}
        
        # Find common time points
        common_time = datasets[0][merge_key]
        for dataset in datasets[1:]:
            common_time = np.intersect1d(common_time, dataset[merge_key])
        
        # Merge data
        merged_data = {merge_key: common_time}
        
        for dataset in datasets:
            for key, value in dataset.items():
                if key != merge_key:
                    if isinstance(value, np.ndarray) and len(value) == len(dataset[merge_key]):
                        # Interpolate to common time points
                        interpolated = np.interp(common_time, dataset[merge_key], value)
                        merged_data[key] = interpolated
                    else:
                        merged_data[key] = value
        
        return merged_data
    
    @staticmethod
    def split_data(data: Dict[str, Any], train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Split data into train, validation, and test sets."""
        # Get time points
        time_points = data.get('time_points', np.arange(len(list(data.values())[0])))
        
        # Calculate split indices
        n_total = len(time_points)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_indices = slice(0, n_train)
        val_indices = slice(n_train, n_train + n_val)
        test_indices = slice(n_train + n_val, n_total)
        
        # Split data
        train_data = {}
        val_data = {}
        test_data = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray) and len(value) == n_total:
                train_data[key] = value[train_indices]
                val_data[key] = value[val_indices]
                test_data[key] = value[test_indices]
            else:
                train_data[key] = value
                val_data[key] = value
                test_data[key] = value
        
        return train_data, val_data, test_data
    
    @staticmethod
    def export_to_csv(data: Dict[str, Any], filename: str, index: bool = True) -> str:
        """Export data to CSV file."""
        df = pd.DataFrame(data)
        df.to_csv(filename, index=index)
        return filename
    
    @staticmethod
    def export_to_excel(data: Dict[str, Any], filename: str, sheet_name: str = 'Sheet1') -> str:
        """Export data to Excel file."""
        df = pd.DataFrame(data)
        df.to_excel(filename, sheet_name=sheet_name, index=False)
        return filename
    
    @staticmethod
    def load_from_hdf5(filename: str, group: Optional[str] = None) -> Dict[str, Any]:
        """Load data from HDF5 file."""
        data = {}
        
        with h5py.File(filename, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            
            DataUtils._load_dict_from_h5(g, data)
        
        return data
    
    @staticmethod
    def _load_dict_from_h5(group: h5py.Group, data: Dict[str, Any]):
        """Recursively load dictionary from HDF5 group."""
        for key in group.keys():
            if isinstance(group[key], h5py.Group):
                data[key] = {}
                DataUtils._load_dict_from_h5(group[key], data[key])
            else:
                data[key] = group[key][:]
        
        # Load attributes
        for key, value in group.attrs.items():
            data[f"_{key}"] = value
    
    @staticmethod
    def validate_data_quality(data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data quality and return quality metrics."""
        quality_metrics = {}
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                metrics = {
                    'missing_values': np.isnan(value).sum(),
                    'infinite_values': np.isinf(value).sum(),
                    'zero_values': (value == 0).sum(),
                    'negative_values': (value < 0).sum(),
                    'data_range': [np.min(value), np.max(value)],
                    'data_type': str(value.dtype),
                    'shape': value.shape
                }
                
                # Calculate additional metrics
                if len(value) > 0:
                    metrics['mean'] = np.mean(value)
                    metrics['std'] = np.std(value)
                    metrics['cv'] = metrics['std'] / metrics['mean'] if metrics['mean'] != 0 else 0
                
                quality_metrics[key] = metrics
        
        return quality_metrics
    
    @staticmethod
    def create_data_summary(data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the dataset."""
        summary = {
            'total_sensors': 0,
            'total_time_points': 0,
            'data_types': [],
            'time_range': [0, 0],
            'sensor_locations': [],
            'data_quality': DataUtils.validate_data_quality(data)
        }
        
        # Count sensors and time points
        for key, value in data.items():
            if isinstance(value, dict) and 'time_points' in value:
                summary['total_sensors'] += 1
                if summary['total_time_points'] == 0:
                    summary['total_time_points'] = len(value['time_points'])
                    summary['time_range'] = [value['time_points'][0], value['time_points'][-1]]
                
                if 'location' in value:
                    summary['sensor_locations'].append(value['location'])
            
            elif isinstance(value, np.ndarray):
                summary['data_types'].append(key)
        
        return summary