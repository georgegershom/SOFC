"""
Dataset Export Utilities

Provides various export formats for the SOFC synthetic dataset,
optimized for different use cases and ML frameworks.
"""

import os
import h5py
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Any
import vtk
import meshio
from pathlib import Path


class DatasetExporter:
    """Exports SOFC dataset in various formats"""
    
    def __init__(self, dataset: Dict[str, Any], output_dir: str):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_hdf5(self, filename: str = 'sofc_dataset.h5') -> str:
        """Export dataset as HDF5 file (recommended for large datasets)"""
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Create main groups
            f.create_group('warp_data')
            f.create_group('stress_data')
            f.create_group('parameters')
            f.create_group('metadata')
            f.create_group('dataset_info')
            
            # Export warp data
            self._export_warp_data_hdf5(f['warp_data'])
            
            # Export stress data
            self._export_stress_data_hdf5(f['stress_data'])
            
            # Export parameters
            self._export_parameters_hdf5(f['parameters'])
            
            # Export metadata
            self._export_metadata_hdf5(f['metadata'])
            
            # Export dataset info
            self._export_dataset_info_hdf5(f['dataset_info'])
        
        print(f"HDF5 dataset exported to: {filepath}")
        return str(filepath)
    
    def _export_warp_data_hdf5(self, group):
        """Export warp data to HDF5 group"""
        for i, sample in enumerate(self.dataset['warp_data']):
            sample_group = group.create_group(f'sample_{i:04d}')
            
            # Height maps
            sample_group.create_dataset('top_height_map', data=sample['top_height_map'])
            sample_group.create_dataset('top_height_x', data=sample['top_height_x'])
            sample_group.create_dataset('top_height_y', data=sample['top_height_y'])
            sample_group.create_dataset('bottom_height_map', data=sample['bottom_height_map'])
            sample_group.create_dataset('bottom_height_x', data=sample['bottom_height_x'])
            sample_group.create_dataset('bottom_height_y', data=sample['bottom_height_y'])
            
            # Point clouds
            sample_group.create_dataset('top_point_cloud', data=sample['top_point_cloud'])
            sample_group.create_dataset('bottom_point_cloud', data=sample['bottom_point_cloud'])
            
            # Metrics as attributes
            for key, value in sample['top_warp_metrics'].items():
                sample_group.attrs[f'top_{key}'] = value
            for key, value in sample['bottom_warp_metrics'].items():
                sample_group.attrs[f'bottom_{key}'] = value
    
    def _export_stress_data_hdf5(self, group):
        """Export stress data to HDF5 group"""
        for i, sample in enumerate(self.dataset['stress_data']):
            sample_group = group.create_group(f'sample_{i:04d}')
            
            # Stress tensors
            sample_group.create_dataset('electrolyte_stress_tensor', data=sample['electrolyte_stress_tensor'])
            sample_group.create_dataset('electrolyte_von_mises', data=sample['electrolyte_von_mises'])
            sample_group.create_dataset('electrolyte_principal_stresses', data=sample['electrolyte_principal_stresses'])
            sample_group.create_dataset('anode_stress_tensor', data=sample['anode_stress_tensor'])
            sample_group.create_dataset('cathode_stress_tensor', data=sample['cathode_stress_tensor'])
            
            # Stress maps
            stress_maps = sample['electrolyte_stress_maps']
            sample_group.create_dataset('von_mises_map', data=stress_maps['von_mises_map'])
            sample_group.create_dataset('principal_1_map', data=stress_maps['principal_1_map'])
            sample_group.create_dataset('x_grid', data=stress_maps['x_grid'])
            sample_group.create_dataset('y_grid', data=stress_maps['y_grid'])
            
            # Stress metrics as attributes
            for component, metrics in sample['stress_metrics'].items():
                for key, value in metrics.items():
                    sample_group.attrs[f'{component}_{key}'] = value
    
    def _export_parameters_hdf5(self, group):
        """Export parameters to HDF5 group"""
        param_df = pd.DataFrame(self.dataset['parameters'])
        param_df.to_hdf(group.file, group.name + '/dataframe', mode='a')
    
    def _export_metadata_hdf5(self, group):
        """Export metadata to HDF5 group"""
        for i, meta in enumerate(self.dataset['metadata']):
            sample_group = group.create_group(f'sample_{i:04d}')
            for key, value in meta.items():
                if isinstance(value, (int, float, str, bool)):
                    sample_group.attrs[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            sample_group.attrs[f'{key}_{subkey}'] = subvalue
    
    def _export_dataset_info_hdf5(self, group):
        """Export dataset information to HDF5 group"""
        if 'dataset_stats' in self.dataset:
            stats = self.dataset['dataset_stats']
            for key, value in stats.items():
                if isinstance(value, (int, float, str, bool)):
                    group.attrs[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (int, float, str, bool)):
                            group.attrs[f'{key}_{subkey}'] = subvalue
    
    def export_npz(self, filename: str = 'sofc_dataset.npz') -> str:
        """Export dataset as NPZ file (NumPy compressed format)"""
        filepath = self.output_dir / filename
        
        # Prepare data for NPZ export
        data = {}
        
        # Warp data
        for i, sample in enumerate(self.dataset['warp_data']):
            data[f'warp_top_height_{i}'] = sample['top_height_map']
            data[f'warp_bottom_height_{i}'] = sample['bottom_height_map']
            data[f'warp_top_point_cloud_{i}'] = sample['top_point_cloud']
            data[f'warp_bottom_point_cloud_{i}'] = sample['bottom_point_cloud']
        
        # Stress data
        for i, sample in enumerate(self.dataset['stress_data']):
            data[f'stress_electrolyte_tensor_{i}'] = sample['electrolyte_stress_tensor']
            data[f'stress_electrolyte_von_mises_{i}'] = sample['electrolyte_von_mises']
            data[f'stress_electrolyte_principal_{i}'] = sample['electrolyte_principal_stresses']
        
        # Parameters
        param_df = pd.DataFrame(self.dataset['parameters'])
        data['parameters'] = param_df.values
        data['parameter_names'] = param_df.columns.values
        
        # Save
        np.savez_compressed(filepath, **data)
        print(f"NPZ dataset exported to: {filepath}")
        return str(filepath)
    
    def export_vtk(self, sample_indices: Optional[List[int]] = None) -> List[str]:
        """Export samples as VTK files for visualization"""
        if sample_indices is None:
            sample_indices = list(range(len(self.dataset['warp_data'])))
        
        vtk_files = []
        
        for i in sample_indices:
            if i >= len(self.dataset['warp_data']):
                continue
            
            # Create VTK file for warp data
            warp_file = self.output_dir / f'warp_sample_{i:04d}.vtk'
            self._export_warp_vtk(i, warp_file)
            vtk_files.append(str(warp_file))
            
            # Create VTK file for stress data
            stress_file = self.output_dir / f'stress_sample_{i:04d}.vtk'
            self._export_stress_vtk(i, stress_file)
            vtk_files.append(str(stress_file))
        
        print(f"VTK files exported: {len(vtk_files)} files")
        return vtk_files
    
    def _export_warp_vtk(self, sample_idx: int, filepath: Path):
        """Export warp data as VTK file"""
        sample = self.dataset['warp_data'][sample_idx]
        
        # Create unstructured grid
        grid = vtk.vtkUnstructuredGrid()
        
        # Add points (top surface)
        points = vtk.vtkPoints()
        top_cloud = sample['top_point_cloud']
        for point in top_cloud:
            points.InsertNextPoint(point)
        grid.SetPoints(points)
        
        # Add cells (simplified - just points for now)
        for i in range(len(top_cloud)):
            cell = vtk.vtkVertex()
            cell.GetPointIds().SetId(0, i)
            grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        
        # Add data arrays
        # Height data
        height_data = vtk.vtkFloatArray()
        height_data.SetName('Height')
        height_data.SetNumberOfComponents(1)
        for point in top_cloud:
            height_data.InsertNextValue(point[2])  # Z coordinate
        grid.GetPointData().AddArray(height_data)
        
        # Displacement data
        displacement_data = vtk.vtkFloatArray()
        displacement_data.SetName('Displacement')
        displacement_data.SetNumberOfComponents(3)
        for point in top_cloud:
            displacement_data.InsertNextTuple3(point[0], point[1], point[2])
        grid.GetPointData().AddArray(displacement_data)
        
        # Write file
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(str(filepath))
        writer.SetInputData(grid)
        writer.Write()
    
    def _export_stress_vtk(self, sample_idx: int, filepath: Path):
        """Export stress data as VTK file"""
        sample = self.dataset['stress_data'][sample_idx]
        
        # Create structured grid
        grid = vtk.vtkStructuredGrid()
        
        # Get stress map dimensions
        von_mises_map = sample['electrolyte_stress_maps']['von_mises_map']
        x_grid = sample['electrolyte_stress_maps']['x_grid']
        y_grid = sample['electrolyte_stress_maps']['y_grid']
        
        # Set dimensions
        grid.SetDimensions(len(x_grid), len(y_grid), 1)
        
        # Create points
        points = vtk.vtkPoints()
        for j in range(len(y_grid)):
            for i in range(len(x_grid)):
                points.InsertNextPoint(x_grid[i], y_grid[j], 0.0)
        grid.SetPoints(points)
        
        # Add stress data
        stress_data = vtk.vtkFloatArray()
        stress_data.SetName('VonMisesStress')
        stress_data.SetNumberOfComponents(1)
        for row in von_mises_map:
            for value in row:
                stress_data.InsertNextValue(value)
        grid.GetPointData().AddArray(stress_data)
        
        # Write file
        writer = vtk.vtkStructuredGridWriter()
        writer.SetFileName(str(filepath))
        writer.SetInputData(grid)
        writer.Write()
    
    def export_csv_summary(self, filename: str = 'dataset_summary.csv') -> str:
        """Export dataset summary as CSV"""
        filepath = self.output_dir / filename
        
        # Create summary data
        summary_data = []
        
        for i in range(len(self.dataset['warp_data'])):
            row = {}
            
            # Sample ID
            row['sample_id'] = i
            
            # Parameters
            params = self.dataset['parameters'][i]
            for key, value in params.items():
                row[f'param_{key}'] = value
            
            # Warp metrics
            warp_metrics = self.dataset['warp_data'][i]['top_warp_metrics']
            for key, value in warp_metrics.items():
                row[f'warp_{key}'] = value
            
            # Stress metrics
            stress_metrics = self.dataset['stress_data'][i]['stress_metrics']['electrolyte']
            for key, value in stress_metrics.items():
                row[f'stress_{key}'] = value
            
            # Metadata
            metadata = self.dataset['metadata'][i]
            row['simulation_time'] = metadata.get('simulation_time', 0.0)
            
            summary_data.append(row)
        
        # Create DataFrame and save
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(filepath, index=False)
        
        print(f"Dataset summary exported to: {filepath}")
        return str(filepath)
    
    def export_json_metadata(self, filename: str = 'dataset_metadata.json') -> str:
        """Export dataset metadata as JSON"""
        filepath = self.output_dir / filename
        
        metadata = {
            'dataset_info': self.dataset.get('dataset_stats', {}),
            'export_info': {
                'export_time': pd.Timestamp.now().isoformat(),
                'n_samples': len(self.dataset['warp_data']),
                'formats_available': ['hdf5', 'npz', 'vtk', 'csv']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset metadata exported to: {filepath}")
        return str(filepath)
    
    def export_all_formats(self) -> Dict[str, str]:
        """Export dataset in all available formats"""
        exported_files = {}
        
        # HDF5 (recommended)
        exported_files['hdf5'] = self.export_hdf5()
        
        # NPZ
        exported_files['npz'] = self.export_npz()
        
        # VTK (first 10 samples)
        exported_files['vtk'] = self.export_vtk(list(range(min(10, len(self.dataset['warp_data'])))))
        
        # CSV summary
        exported_files['csv'] = self.export_csv_summary()
        
        # JSON metadata
        exported_files['json'] = self.export_json_metadata()
        
        return exported_files


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('..')
    
    # Load a sample dataset (this would be loaded from actual data)
    sample_dataset = {
        'warp_data': [],
        'stress_data': [],
        'parameters': [],
        'metadata': [],
        'dataset_stats': {}
    }
    
    exporter = DatasetExporter(sample_dataset, './exports')
    exported_files = exporter.export_all_formats()
    
    print("Exported files:")
    for format_name, files in exported_files.items():
        print(f"  {format_name}: {files}")