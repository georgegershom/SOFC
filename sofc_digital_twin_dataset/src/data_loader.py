"""
Data Loader for Physics-Informed Machine Learning
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import pandas as pd
import json
from pathlib import Path
from scipy.interpolate import griddata
from tqdm import tqdm


class SOFCSimulationDataset(Dataset):
    """
    PyTorch Dataset for SOFC simulation data
    """
    
    def __init__(self, data_path, split='train', normalize=True, fields=None):
        """
        Initialize dataset
        
        Parameters:
        -----------
        data_path : str or Path
            Path to dataset directory
        split : str
            Data split ('train', 'validation', 'test')
        normalize : bool
            Whether to normalize the data
        fields : list
            List of fields to load (None for all)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.normalize = normalize
        self.fields = fields or ['temperature', 'von_mises_stress', 'current_density']
        
        # Load data splits
        with open(self.data_path / 'data_splits.json', 'r') as f:
            splits = json.load(f)
        self.indices = splits[split]
        
        # Load normalization parameters
        if self.normalize:
            self.load_normalization_params()
        
        # Cache for faster loading
        self.cache = {}
        self.use_cache = True
    
    def load_normalization_params(self):
        """
        Calculate or load normalization parameters
        """
        norm_file = self.data_path / 'normalization_params.json'
        
        if norm_file.exists():
            with open(norm_file, 'r') as f:
                self.norm_params = json.load(f)
        else:
            # Calculate normalization parameters from training set
            print("Calculating normalization parameters...")
            self.norm_params = self.calculate_normalization_params()
            
            # Save for future use
            with open(norm_file, 'w') as f:
                json.dump(self.norm_params, f)
    
    def calculate_normalization_params(self):
        """
        Calculate mean and std for each field
        """
        params = {}
        
        with h5py.File(self.data_path / 'simulation' / 'high_fidelity_simulations.h5', 'r') as hf:
            # Use only training indices
            with open(self.data_path / 'data_splits.json', 'r') as f:
                train_indices = json.load(f)['train']
            
            for field in self.fields:
                field_data = []
                for idx in tqdm(train_indices[:50], desc=f"Calculating stats for {field}"):
                    sim = hf[f'simulation_{idx:04d}']
                    if field in sim:
                        data = sim[field][:]
                        field_data.append(data.flatten())
                
                if field_data:
                    all_data = np.concatenate(field_data)
                    params[field] = {
                        'mean': float(np.mean(all_data)),
                        'std': float(np.std(all_data))
                    }
        
        # Add input parameter normalization
        params['inputs'] = {
            'current_density': {'min': 1000, 'max': 10000},
            'fuel_utilization': {'min': 0.3, 'max': 0.85},
            'air_utilization': {'min': 0.2, 'max': 0.5},
            'fuel_temperature': {'min': 600, 'max': 800},
            'air_temperature': {'min': 500, 'max': 700},
            'crack_length': {'min': 0, 'max': 5},
            'porosity_factor': {'min': 0.9, 'max': 1.2}
        }
        
        return params
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Returns:
        --------
        dict : Contains inputs, outputs, and metadata
        """
        sim_idx = self.indices[idx]
        
        # Check cache
        if self.use_cache and sim_idx in self.cache:
            return self.cache[sim_idx]
        
        with h5py.File(self.data_path / 'simulation' / 'high_fidelity_simulations.h5', 'r') as hf:
            sim = hf[f'simulation_{sim_idx:04d}']
            
            # Load input parameters
            inputs = torch.tensor([
                sim.attrs['current_density'],
                sim.attrs['fuel_utilization'],
                sim.attrs['air_utilization'],
                sim.attrs['fuel_temperature'],
                sim.attrs['air_temperature'],
                sim.attrs['crack_length'],
                sim.attrs['porosity_factor']
            ], dtype=torch.float32)
            
            # Normalize inputs
            if self.normalize:
                for i, param in enumerate(['current_density', 'fuel_utilization', 
                                          'air_utilization', 'fuel_temperature',
                                          'air_temperature', 'crack_length', 
                                          'porosity_factor']):
                    min_val = self.norm_params['inputs'][param]['min']
                    max_val = self.norm_params['inputs'][param]['max']
                    inputs[i] = (inputs[i] - min_val) / (max_val - min_val)
            
            # Load field data
            fields_data = {}
            for field in self.fields:
                if field in sim:
                    data = torch.tensor(sim[field][:], dtype=torch.float32)
                    
                    # Normalize field data
                    if self.normalize and field in self.norm_params:
                        mean = self.norm_params[field]['mean']
                        std = self.norm_params[field]['std']
                        data = (data - mean) / (std + 1e-8)
                    
                    fields_data[field] = data
            
            # Load scalar outputs
            outputs = {
                'voltage': torch.tensor(sim.attrs['voltage'], dtype=torch.float32),
                'max_stress': torch.tensor(sim.attrs['max_stress'], dtype=torch.float32),
                'creep_damage': torch.tensor(sim.attrs['creep_damage'], dtype=torch.float32)
            }
            
            sample = {
                'inputs': inputs,
                'fields': fields_data,
                'outputs': outputs,
                'sim_idx': sim_idx
            }
            
            # Cache the sample
            if self.use_cache:
                self.cache[sim_idx] = sample
            
            return sample
    
    def get_dataloader(self, batch_size=32, shuffle=True, num_workers=4):
        """
        Create a DataLoader for this dataset
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )


class ExperimentalDataLoader:
    """
    Loader for experimental validation data
    """
    
    def __init__(self, data_path):
        """
        Initialize experimental data loader
        
        Parameters:
        -----------
        data_path : str or Path
            Path to dataset directory
        """
        self.data_path = Path(data_path)
        self.load_all_data()
    
    def load_all_data(self):
        """
        Load all experimental data into memory
        """
        # Operational data
        self.operational_data = pd.read_csv(
            self.data_path / 'experimental' / 'operational_data.csv'
        )
        
        # EIS data
        with open(self.data_path / 'experimental' / 'eis_data.json', 'r') as f:
            self.eis_data = json.load(f)
        
        # Thermal images
        self.thermal_images = np.load(
            self.data_path / 'experimental' / 'thermal_images.npy'
        )
        
        # Strain gauge data
        self.strain_data = pd.read_csv(
            self.data_path / 'experimental' / 'strain_gauge_data.csv'
        )
        
        # AE events
        self.ae_events = pd.read_csv(
            self.data_path / 'experimental' / 'acoustic_emission_events.csv'
        )
    
    def get_time_window(self, start_hour, end_hour):
        """
        Get experimental data within a time window
        
        Parameters:
        -----------
        start_hour : float
            Start time in hours
        end_hour : float
            End time in hours
        
        Returns:
        --------
        dict : Data within the time window
        """
        # Filter operational data
        op_mask = (self.operational_data['time_hours'] >= start_hour) & \
                 (self.operational_data['time_hours'] <= end_hour)
        op_window = self.operational_data[op_mask]
        
        # Filter strain data
        strain_mask = (self.strain_data['time_hours'] >= start_hour) & \
                     (self.strain_data['time_hours'] <= end_hour)
        strain_window = self.strain_data[strain_mask]
        
        # Filter AE events
        ae_mask = (self.ae_events['time_hours'] >= start_hour) & \
                 (self.ae_events['time_hours'] <= end_hour)
        ae_window = self.ae_events[ae_mask]
        
        # Get corresponding EIS measurements
        eis_window = [m for m in self.eis_data 
                     if start_hour <= m['time_hours'] <= end_hour]
        
        # Get thermal images (assuming uniform time spacing)
        n_images = len(self.thermal_images)
        total_hours = self.operational_data['time_hours'].max()
        start_idx = int(n_images * start_hour / total_hours)
        end_idx = int(n_images * end_hour / total_hours)
        thermal_window = self.thermal_images[start_idx:end_idx]
        
        return {
            'operational': op_window,
            'strain': strain_window,
            'acoustic_emission': ae_window,
            'eis': eis_window,
            'thermal_images': thermal_window
        }
    
    def get_degradation_indicators(self):
        """
        Extract degradation indicators from experimental data
        
        Returns:
        --------
        pd.DataFrame : Degradation indicators over time
        """
        indicators = pd.DataFrame()
        
        # Voltage degradation rate
        indicators['time_hours'] = self.operational_data['time_hours']
        indicators['voltage'] = self.operational_data['voltage_V']
        indicators['voltage_degradation_rate'] = np.gradient(
            self.operational_data['voltage_V'],
            self.operational_data['time_hours']
        )
        
        # Power degradation
        indicators['power'] = self.operational_data['power_W']
        initial_power = self.operational_data['power_W'].iloc[:100].mean()
        indicators['power_degradation'] = (initial_power - self.operational_data['power_W']) / initial_power
        
        # Efficiency drop
        indicators['efficiency'] = self.operational_data['efficiency']
        initial_efficiency = self.operational_data['efficiency'].iloc[:100].mean()
        indicators['efficiency_drop'] = initial_efficiency - self.operational_data['efficiency']
        
        # Cumulative AE events (indicator of damage accumulation)
        ae_counts = []
        for t in indicators['time_hours']:
            count = len(self.ae_events[self.ae_events['time_hours'] <= t])
            ae_counts.append(count)
        indicators['cumulative_ae_events'] = ae_counts
        
        # Average strain (from all sensors)
        strain_cols = [col for col in self.strain_data.columns if 'strain_sensor' in col]
        if strain_cols:
            avg_strain = self.strain_data[strain_cols].mean(axis=1)
            # Interpolate to match operational data time points
            indicators['avg_strain'] = np.interp(
                indicators['time_hours'],
                self.strain_data['time_hours'],
                avg_strain
            )
        
        return indicators


class MonitoringStreamLoader:
    """
    Loader for real-time monitoring stream data
    """
    
    def __init__(self, data_path):
        """
        Initialize monitoring stream loader
        
        Parameters:
        -----------
        data_path : str or Path
            Path to dataset directory
        """
        self.data_path = Path(data_path)
        self.load_stream_data()
    
    def load_stream_data(self):
        """
        Load monitoring stream data
        """
        self.stream_data = pd.read_csv(
            self.data_path / 'monitoring' / 'realtime_stream.csv'
        )
        
        # Convert timestamp to datetime
        self.stream_data['timestamp'] = pd.to_datetime(self.stream_data['timestamp'])
        
        # Load triggers if available
        trigger_file = self.data_path / 'monitoring' / 'adaptive_triggers.csv'
        if trigger_file.exists():
            self.triggers = pd.read_csv(trigger_file)
        else:
            self.triggers = pd.DataFrame()
    
    def get_stream_batch(self, start_idx, batch_size):
        """
        Get a batch of stream data
        
        Parameters:
        -----------
        start_idx : int
            Starting index
        batch_size : int
            Batch size
        
        Returns:
        --------
        pd.DataFrame : Batch of stream data
        """
        end_idx = min(start_idx + batch_size, len(self.stream_data))
        return self.stream_data.iloc[start_idx:end_idx]
    
    def simulate_realtime_stream(self, callback_fn, speed_factor=1.0):
        """
        Simulate real-time data streaming
        
        Parameters:
        -----------
        callback_fn : function
            Function to call with each data point
        speed_factor : float
            Speed multiplier (>1 for faster, <1 for slower)
        """
        import time
        
        for idx, row in self.stream_data.iterrows():
            # Call the callback function with the data
            callback_fn(row)
            
            # Wait for the appropriate time
            if idx < len(self.stream_data) - 1:
                next_time = self.stream_data.iloc[idx + 1]['timestamp']
                current_time = row['timestamp']
                wait_time = (next_time - current_time).total_seconds() / speed_factor
                time.sleep(max(0, wait_time))
    
    def get_anomalies(self, threshold_factor=3.0):
        """
        Detect anomalies in the stream data
        
        Parameters:
        -----------
        threshold_factor : float
            Number of standard deviations for anomaly threshold
        
        Returns:
        --------
        pd.DataFrame : Detected anomalies
        """
        anomalies = []
        
        # Calculate rolling statistics
        window_size = 100
        
        for col in ['voltage_V', 'current_A', 'temperature_C']:
            rolling_mean = self.stream_data[col].rolling(window_size).mean()
            rolling_std = self.stream_data[col].rolling(window_size).std()
            
            # Detect anomalies
            upper_bound = rolling_mean + threshold_factor * rolling_std
            lower_bound = rolling_mean - threshold_factor * rolling_std
            
            anomaly_mask = (self.stream_data[col] > upper_bound) | \
                          (self.stream_data[col] < lower_bound)
            
            for idx in self.stream_data[anomaly_mask].index:
                anomalies.append({
                    'timestamp': self.stream_data.loc[idx, 'timestamp'],
                    'variable': col,
                    'value': self.stream_data.loc[idx, col],
                    'expected_range': (lower_bound[idx], upper_bound[idx])
                })
        
        return pd.DataFrame(anomalies)


def create_physics_informed_batch(simulation_batch, grid_size=(10, 10, 5)):
    """
    Create a batch with physics-informed features
    
    Parameters:
    -----------
    simulation_batch : dict
        Batch from SOFCSimulationDataset
    grid_size : tuple
        Size for downsampled grid
    
    Returns:
    --------
    dict : Physics-informed batch
    """
    batch_size = simulation_batch['inputs'].shape[0]
    device = simulation_batch['inputs'].device
    
    # Create spatial coordinates
    x = torch.linspace(0, 1, grid_size[0], device=device)
    y = torch.linspace(0, 1, grid_size[1], device=device)
    z = torch.linspace(0, 1, grid_size[2], device=device)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    
    # Flatten spatial coordinates
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
    
    # Repeat for batch
    coords_batch = coords.unsqueeze(0).repeat(batch_size, 1, 1)
    
    # Combine with operating conditions
    inputs_expanded = simulation_batch['inputs'].unsqueeze(1).repeat(1, coords.shape[0], 1)
    
    # Full input tensor
    full_inputs = torch.cat([coords_batch, inputs_expanded], dim=2)
    
    # Downsample field data to match grid_size
    downsampled_fields = {}
    for field_name, field_data in simulation_batch['fields'].items():
        # Interpolate to smaller grid
        field_resized = torch.nn.functional.interpolate(
            field_data.unsqueeze(1),  # Add channel dimension
            size=grid_size,
            mode='trilinear',
            align_corners=False
        ).squeeze(1)
        
        downsampled_fields[field_name] = field_resized.flatten(1)
    
    return {
        'inputs': full_inputs,
        'targets': downsampled_fields,
        'scalars': simulation_batch['outputs'],
        'coords': coords_batch,
        'original_batch': simulation_batch
    }


def main():
    """
    Example usage of data loaders
    """
    data_path = Path("../data")
    
    if not data_path.exists():
        print(f"Data directory not found at {data_path}")
        print("Please run the dataset generator first.")
        return
    
    print("SOFC Digital Twin Data Loading Examples")
    print("=" * 50)
    
    # 1. Load simulation dataset
    print("\n1. Loading simulation dataset...")
    train_dataset = SOFCSimulationDataset(data_path, split='train')
    print(f"Training samples: {len(train_dataset)}")
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Input shape: {sample['inputs'].shape}")
    print(f"Field shapes: {[f'{k}: {v.shape}' for k, v in sample['fields'].items()]}")
    
    # Create dataloader
    train_loader = train_dataset.get_dataloader(batch_size=8)
    batch = next(iter(train_loader))
    print(f"Batch input shape: {batch['inputs'].shape}")
    
    # 2. Load experimental data
    print("\n2. Loading experimental data...")
    exp_loader = ExperimentalDataLoader(data_path)
    
    # Get data window
    window_data = exp_loader.get_time_window(100, 200)
    print(f"Operational data points in window: {len(window_data['operational'])}")
    print(f"Thermal images in window: {len(window_data['thermal_images'])}")
    
    # Get degradation indicators
    indicators = exp_loader.get_degradation_indicators()
    print(f"Degradation indicators shape: {indicators.shape}")
    print(f"Columns: {list(indicators.columns)}")
    
    # 3. Load monitoring streams
    print("\n3. Loading monitoring streams...")
    monitor_loader = MonitoringStreamLoader(data_path)
    
    # Get a batch
    stream_batch = monitor_loader.get_stream_batch(0, 100)
    print(f"Stream batch shape: {stream_batch.shape}")
    
    # Detect anomalies
    anomalies = monitor_loader.get_anomalies()
    if not anomalies.empty:
        print(f"Detected {len(anomalies)} anomalies")
        print(f"Anomaly types: {anomalies['variable'].value_counts().to_dict()}")
    
    # 4. Create physics-informed batch
    print("\n4. Creating physics-informed batch...")
    pi_batch = create_physics_informed_batch(batch, grid_size=(10, 10, 5))
    print(f"PI input shape: {pi_batch['inputs'].shape}")
    print(f"PI target shapes: {[f'{k}: {v.shape}' for k, v in pi_batch['targets'].items()]}")
    
    print("\nData loading examples complete!")


if __name__ == "__main__":
    main()