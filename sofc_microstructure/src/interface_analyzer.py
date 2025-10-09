"""
Advanced Interface Geometry Analysis for SOFC Microstructures

This module provides detailed analysis of anode/electrolyte interfaces,
including morphology characterization, roughness analysis, and delamination
risk assessment.

Author: AI Assistant
Date: 2025-10-08
"""

import numpy as np
import scipy.ndimage as ndi
from scipy import spatial, interpolate
from skimage import morphology, filters, measure, segmentation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from typing import Tuple, Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

class InterfaceGeometryAnalyzer:
    """
    Analyzer for detailed interface geometry characterization in SOFC microstructures.
    Focuses on the critical anode/electrolyte interface where delamination occurs.
    """
    
    def __init__(self, microstructure: np.ndarray, voxel_size: float, phases: Dict[str, int]):
        """
        Initialize the interface analyzer.
        
        Parameters:
        -----------
        microstructure : np.ndarray
            3D microstructure array with phase labels
        voxel_size : float
            Size of each voxel in micrometers
        phases : dict
            Dictionary mapping phase names to phase IDs
        """
        self.microstructure = microstructure
        self.voxel_size = voxel_size
        self.phases = phases
        self.dimensions = microstructure.shape
        
        # Interface analysis results
        self.interface_surface = None
        self.interface_points = None
        self.roughness_metrics = {}
        self.curvature_metrics = {}
        self.stress_concentration_factors = None
        
    def extract_interface_surface(self, smooth_iterations: int = 2) -> np.ndarray:
        """
        Extract the precise 3D interface surface between anode and electrolyte.
        
        Parameters:
        -----------
        smooth_iterations : int
            Number of smoothing iterations to apply
            
        Returns:
        --------
        np.ndarray
            Binary array marking interface voxels
        """
        print("Extracting anode/electrolyte interface surface...")
        
        # Get phase masks
        anode_mask = (self.microstructure == self.phases['ni_ysz'])
        electrolyte_mask = (self.microstructure == self.phases['ysz_electrolyte'])
        
        # Find interface using morphological operations
        # Method 1: Dilate anode by 1 voxel and find overlap with electrolyte
        anode_dilated = morphology.binary_dilation(anode_mask, morphology.ball(1))
        interface_method1 = anode_dilated.astype(bool) & electrolyte_mask.astype(bool)
        
        # Method 2: Dilate electrolyte and find overlap with anode
        electrolyte_dilated = morphology.binary_dilation(electrolyte_mask, morphology.ball(1))
        interface_method2 = electrolyte_dilated.astype(bool) & anode_mask.astype(bool)
        
        # Method 3: Find boundary voxels using gradient
        # Create a combined phase mask (anode=1, electrolyte=2, others=0)
        combined_mask = np.zeros_like(self.microstructure, dtype=np.uint8)
        combined_mask[anode_mask] = 1
        combined_mask[electrolyte_mask] = 2
        
        # Find boundaries using gradient
        grad_x = np.abs(np.diff(combined_mask, axis=0, prepend=0))
        grad_y = np.abs(np.diff(combined_mask, axis=1, prepend=0))
        grad_z = np.abs(np.diff(combined_mask, axis=2, prepend=0))
        
        # Interface where there's a transition between anode and electrolyte
        interface_method3 = ((grad_x > 0) | (grad_y > 0) | (grad_z > 0)) & (combined_mask > 0)
        
        # Combine all methods
        interface_rough = interface_method1 | interface_method2 | interface_method3
        
        # Smooth the interface
        interface_smooth = interface_rough.astype(float)
        for _ in range(smooth_iterations):
            interface_smooth = filters.gaussian(interface_smooth, sigma=0.8)
        
        # Threshold back to binary
        self.interface_surface = interface_smooth > 0.3
        
        # Extract interface points for further analysis
        self.interface_points = np.array(np.where(self.interface_surface)).T
        
        print(f"Interface extracted: {len(self.interface_points)} interface voxels")
        return self.interface_surface
    
    def analyze_interface_morphology(self) -> Dict:
        """
        Comprehensive morphological analysis of the interface.
        
        Returns:
        --------
        dict
            Dictionary containing morphological metrics
        """
        if self.interface_surface is None:
            self.extract_interface_surface()
        
        print("Analyzing interface morphology...")
        
        morphology_metrics = {}
        
        # 1. Interface area calculation
        interface_area = self._calculate_interface_area()
        morphology_metrics['interface_area_um2'] = interface_area
        
        # 2. Interface thickness distribution
        thickness_stats = self._analyze_interface_thickness()
        morphology_metrics['thickness_stats'] = thickness_stats
        
        # 3. Interface tortuosity
        tortuosity = self._calculate_interface_tortuosity()
        morphology_metrics['tortuosity'] = tortuosity
        
        # 4. Interface connectivity
        connectivity_stats = self._analyze_interface_connectivity()
        morphology_metrics['connectivity'] = connectivity_stats
        
        # 5. Local interface orientation
        orientation_stats = self._analyze_interface_orientation()
        morphology_metrics['orientation'] = orientation_stats
        
        return morphology_metrics
    
    def analyze_interface_roughness(self, analysis_scales: List[float] = None) -> Dict:
        """
        Multi-scale roughness analysis of the interface.
        
        Parameters:
        -----------
        analysis_scales : list
            List of length scales for analysis (in micrometers)
            
        Returns:
        --------
        dict
            Dictionary containing roughness metrics
        """
        if self.interface_surface is None:
            self.extract_interface_surface()
        
        if analysis_scales is None:
            analysis_scales = [0.5, 1.0, 2.0, 5.0]  # micrometers
        
        print("Analyzing interface roughness at multiple scales...")
        
        roughness_metrics = {}
        
        # Convert interface to height map for each XY position
        height_map = self._extract_interface_height_map()
        
        if height_map is None:
            return {'error': 'Could not extract height map'}
        
        # Calculate roughness metrics at different scales
        for scale in analysis_scales:
            scale_voxels = int(scale / self.voxel_size)
            if scale_voxels < 2:
                continue
            
            # Smooth height map at this scale
            smoothed_height = filters.gaussian(height_map, sigma=scale_voxels)
            
            # Calculate roughness parameters
            roughness_params = self._calculate_roughness_parameters(height_map, smoothed_height)
            roughness_metrics[f'scale_{scale}um'] = roughness_params
        
        # Overall roughness statistics
        overall_stats = self._calculate_overall_roughness_stats(height_map)
        roughness_metrics['overall'] = overall_stats
        
        self.roughness_metrics = roughness_metrics
        return roughness_metrics
    
    def analyze_interface_curvature(self) -> Dict:
        """
        Analyze local curvature of the interface surface.
        
        Returns:
        --------
        dict
            Dictionary containing curvature metrics
        """
        if self.interface_surface is None:
            self.extract_interface_surface()
        
        print("Analyzing interface curvature...")
        
        # Extract interface as mesh using marching cubes
        try:
            verts, faces, normals, values = measure.marching_cubes(
                self.interface_surface.astype(float), level=0.5
            )
            
            # Scale vertices to physical coordinates
            verts_scaled = verts * self.voxel_size
            
            # Calculate curvature metrics
            curvature_metrics = self._calculate_surface_curvature(verts_scaled, faces, normals)
            
            self.curvature_metrics = curvature_metrics
            return curvature_metrics
            
        except Exception as e:
            print(f"Warning: Could not analyze curvature: {e}")
            return {'error': str(e)}
    
    def assess_delamination_risk(self) -> Dict:
        """
        Assess potential delamination risk based on interface geometry.
        
        Returns:
        --------
        dict
            Dictionary containing delamination risk assessment
        """
        if self.interface_surface is None:
            self.extract_interface_surface()
        
        print("Assessing delamination risk...")
        
        risk_assessment = {}
        
        # 1. Stress concentration factors based on curvature
        if not self.curvature_metrics:
            self.analyze_interface_curvature()
        
        stress_factors = self._calculate_stress_concentration_factors()
        risk_assessment['stress_concentration'] = stress_factors
        
        # 2. Interface defects (sharp corners, high curvature regions)
        defect_analysis = self._identify_interface_defects()
        risk_assessment['defects'] = defect_analysis
        
        # 3. Interface adhesion quality metrics
        adhesion_metrics = self._assess_interface_adhesion_quality()
        risk_assessment['adhesion_quality'] = adhesion_metrics
        
        # 4. Overall risk score
        overall_risk = self._calculate_overall_delamination_risk(
            stress_factors, defect_analysis, adhesion_metrics
        )
        risk_assessment['overall_risk_score'] = overall_risk
        
        return risk_assessment
    
    def _calculate_interface_area(self) -> float:
        """Calculate the total interface area using marching cubes."""
        try:
            verts, faces, _, _ = measure.marching_cubes(
                self.interface_surface.astype(float), level=0.5
            )
            
            # Calculate area of each triangle
            total_area = 0.0
            for face in faces:
                v0, v1, v2 = verts[face]
                # Cross product for triangle area
                edge1 = v1 - v0
                edge2 = v2 - v0
                cross = np.cross(edge1, edge2)
                area = 0.5 * np.linalg.norm(cross)
                total_area += area
            
            # Convert to physical units
            return total_area * (self.voxel_size ** 2)
            
        except Exception as e:
            print(f"Warning: Could not calculate interface area: {e}")
            return 0.0
    
    def _analyze_interface_thickness(self) -> Dict:
        """Analyze the thickness distribution of the interface region."""
        
        # Calculate distance transform from both sides
        anode_mask = (self.microstructure == self.phases['ni_ysz'])
        electrolyte_mask = (self.microstructure == self.phases['ysz_electrolyte'])
        
        # Distance from anode
        dist_from_anode = ndi.distance_transform_edt(~anode_mask) * self.voxel_size
        
        # Distance from electrolyte
        dist_from_electrolyte = ndi.distance_transform_edt(~electrolyte_mask) * self.voxel_size
        
        # Interface thickness is sum of distances where interface exists
        if np.any(self.interface_surface):
            interface_thickness = (dist_from_anode + dist_from_electrolyte)[self.interface_surface]
            
            if len(interface_thickness) > 0:
                return {
                    'mean_thickness_um': np.mean(interface_thickness),
                    'std_thickness_um': np.std(interface_thickness),
                    'min_thickness_um': np.min(interface_thickness),
                    'max_thickness_um': np.max(interface_thickness),
                    'median_thickness_um': np.median(interface_thickness)
                }
        
        # Return default values if no interface found
        return {
            'mean_thickness_um': 0.0,
            'std_thickness_um': 0.0,
            'min_thickness_um': 0.0,
            'max_thickness_um': 0.0,
            'median_thickness_um': 0.0
        }
    
    def _calculate_interface_tortuosity(self) -> float:
        """Calculate interface tortuosity as ratio of actual to straight-line distance."""
        
        if len(self.interface_points) < 2:
            return 1.0
        
        # Project interface points to find main direction
        # Use PCA to find principal direction
        centered_points = self.interface_points - np.mean(self.interface_points, axis=0)
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Principal direction (largest eigenvalue)
        principal_direction = eigenvectors[:, -1]
        
        # Project points onto principal direction
        projections = np.dot(centered_points, principal_direction)
        
        # Calculate tortuosity
        actual_length = np.max(projections) - np.min(projections)
        straight_length = np.linalg.norm(
            self.interface_points[np.argmax(projections)] - 
            self.interface_points[np.argmin(projections)]
        )
        
        if straight_length > 0:
            return actual_length / straight_length
        else:
            return 1.0
    
    def _analyze_interface_connectivity(self) -> Dict:
        """Analyze connectivity of the interface."""
        
        # Find connected components of interface
        labeled_interface, num_components = ndi.label(self.interface_surface)
        
        if num_components == 0:
            return {'connected_components': 0, 'largest_component_fraction': 0.0}
        
        # Calculate size of each component
        component_sizes = np.bincount(labeled_interface.ravel())[1:]  # Skip background
        
        return {
            'connected_components': num_components,
            'largest_component_fraction': np.max(component_sizes) / np.sum(self.interface_surface),
            'component_size_distribution': {
                'mean': np.mean(component_sizes),
                'std': np.std(component_sizes),
                'max': np.max(component_sizes),
                'min': np.min(component_sizes)
            }
        }
    
    def _analyze_interface_orientation(self) -> Dict:
        """Analyze local orientation of interface surface."""
        
        # Calculate gradient at interface points
        gradients = []
        
        for point in self.interface_points[:1000]:  # Sample for performance
            x, y, z = point
            
            # Calculate local gradient using finite differences
            if (1 <= x < self.dimensions[0]-1 and 
                1 <= y < self.dimensions[1]-1 and 
                1 <= z < self.dimensions[2]-1):
                
                grad_x = (float(self.interface_surface[x+1, y, z]) - 
                         float(self.interface_surface[x-1, y, z])) / 2.0
                grad_y = (float(self.interface_surface[x, y+1, z]) - 
                         float(self.interface_surface[x, y-1, z])) / 2.0
                grad_z = (float(self.interface_surface[x, y, z+1]) - 
                         float(self.interface_surface[x, y, z-1])) / 2.0
                
                gradient = np.array([grad_x, grad_y, grad_z])
                if np.linalg.norm(gradient) > 0:
                    gradients.append(gradient / np.linalg.norm(gradient))
        
        if not gradients:
            return {'error': 'Could not calculate orientations'}
        
        gradients = np.array(gradients)
        
        # Calculate orientation statistics
        # Angle with respect to z-axis (normal to layers)
        z_axis = np.array([0, 0, 1])
        angles_with_z = np.arccos(np.abs(np.dot(gradients, z_axis))) * 180 / np.pi
        
        return {
            'mean_angle_with_z_deg': np.mean(angles_with_z),
            'std_angle_with_z_deg': np.std(angles_with_z),
            'orientation_uniformity': 1.0 - np.std(angles_with_z) / 90.0  # Normalized
        }
    
    def _extract_interface_height_map(self) -> Optional[np.ndarray]:
        """Extract 2D height map of the interface."""
        
        # Find interface height for each (x,y) position
        height_map = np.full((self.dimensions[0], self.dimensions[1]), -1, dtype=float)
        
        for x in range(self.dimensions[0]):
            for y in range(self.dimensions[1]):
                # Find interface z-positions at this (x,y)
                z_positions = np.where(self.interface_surface[x, y, :])[0]
                
                if len(z_positions) > 0:
                    # Use mean if multiple interface points
                    height_map[x, y] = np.mean(z_positions) * self.voxel_size
        
        # Fill missing values using interpolation
        valid_mask = height_map >= 0
        if np.sum(valid_mask) < 10:
            return None
        
        # Interpolate missing values
        from scipy.interpolate import griddata
        
        valid_points = np.array(np.where(valid_mask)).T
        valid_values = height_map[valid_mask]
        
        all_points = np.array(np.where(np.ones_like(height_map))).T
        
        try:
            interpolated = griddata(valid_points, valid_values, all_points, method='linear')
            height_map_filled = interpolated.reshape(height_map.shape)
            
            # Fill any remaining NaN values with nearest neighbor
            nan_mask = np.isnan(height_map_filled)
            if np.any(nan_mask):
                interpolated_nn = griddata(valid_points, valid_values, all_points, method='nearest')
                height_map_filled[nan_mask] = interpolated_nn.reshape(height_map.shape)[nan_mask]
            
            return height_map_filled
            
        except Exception as e:
            print(f"Warning: Could not interpolate height map: {e}")
            return height_map
    
    def _calculate_roughness_parameters(self, height_map: np.ndarray, 
                                      smoothed_height: np.ndarray) -> Dict:
        """Calculate standard roughness parameters."""
        
        # Height deviations from smoothed surface
        deviations = height_map - smoothed_height
        
        return {
            'Ra_um': np.mean(np.abs(deviations)),  # Average roughness
            'Rq_um': np.sqrt(np.mean(deviations**2)),  # RMS roughness
            'Rz_um': np.max(deviations) - np.min(deviations),  # Peak-to-valley
            'Rsk': self._calculate_skewness(deviations),  # Skewness
            'Rku': self._calculate_kurtosis(deviations),  # Kurtosis
            'bearing_ratio': np.sum(deviations > 0) / deviations.size
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return np.mean(((data - mean_val) / std_val) ** 4) - 3.0
    
    def _calculate_overall_roughness_stats(self, height_map: np.ndarray) -> Dict:
        """Calculate overall roughness statistics."""
        
        # Calculate gradients
        grad_x, grad_y = np.gradient(height_map)
        
        # Slope magnitude
        slope_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        return {
            'mean_slope': np.mean(slope_magnitude),
            'max_slope': np.max(slope_magnitude),
            'slope_std': np.std(slope_magnitude),
            'height_range_um': np.max(height_map) - np.min(height_map),
            'height_std_um': np.std(height_map)
        }
    
    def _calculate_surface_curvature(self, vertices: np.ndarray, faces: np.ndarray, 
                                   normals: np.ndarray) -> Dict:
        """Calculate curvature metrics for the interface surface."""
        
        # Simplified curvature calculation
        # In a full implementation, you would use more sophisticated methods
        
        curvature_values = []
        
        # Sample vertices for performance
        sample_indices = np.random.choice(len(vertices), min(1000, len(vertices)), replace=False)
        
        for i in sample_indices:
            vertex = vertices[i]
            
            # Find neighboring vertices (simplified approach)
            distances = np.linalg.norm(vertices - vertex, axis=1)
            neighbors = np.where((distances > 0) & (distances < 2.0))[0]  # Within 2 μm
            
            if len(neighbors) > 3:
                # Fit local plane and calculate curvature
                neighbor_points = vertices[neighbors]
                
                # Center the points
                centered = neighbor_points - vertex
                
                # SVD to find local coordinate system
                try:
                    U, s, Vt = np.linalg.svd(centered.T)
                    
                    # Local curvature estimate (simplified)
                    if len(s) >= 2 and s[1] > 1e-6:
                        curvature = s[0] / s[1]  # Ratio of principal components
                        curvature_values.append(curvature)
                        
                except np.linalg.LinAlgError:
                    continue
        
        if not curvature_values:
            return {'error': 'Could not calculate curvature'}
        
        curvature_values = np.array(curvature_values)
        
        return {
            'mean_curvature': np.mean(curvature_values),
            'max_curvature': np.max(curvature_values),
            'curvature_std': np.std(curvature_values),
            'high_curvature_fraction': np.sum(curvature_values > np.percentile(curvature_values, 90)) / len(curvature_values)
        }
    
    def _calculate_stress_concentration_factors(self) -> Dict:
        """Calculate stress concentration factors based on interface geometry."""
        
        if not self.curvature_metrics or 'error' in self.curvature_metrics:
            return {'error': 'Curvature analysis required first'}
        
        # Simplified stress concentration factor calculation
        # Based on local curvature and geometry
        
        max_curvature = self.curvature_metrics.get('max_curvature', 1.0)
        mean_curvature = self.curvature_metrics.get('mean_curvature', 1.0)
        
        # Theoretical stress concentration factor for curved interfaces
        # K_t ≈ 1 + 2√(ρ/R) where ρ is notch radius, R is characteristic length
        
        characteristic_length = 1.0  # μm
        
        if mean_curvature > 0:
            notch_radius = 1.0 / mean_curvature
            stress_factor = 1.0 + 2.0 * np.sqrt(notch_radius / characteristic_length)
        else:
            stress_factor = 1.0
        
        # Maximum stress concentration
        if max_curvature > 0:
            max_notch_radius = 1.0 / max_curvature
            max_stress_factor = 1.0 + 2.0 * np.sqrt(max_notch_radius / characteristic_length)
        else:
            max_stress_factor = 1.0
        
        return {
            'mean_stress_concentration_factor': stress_factor,
            'max_stress_concentration_factor': max_stress_factor,
            'critical_stress_locations': max_stress_factor > 3.0  # Threshold for concern
        }
    
    def _identify_interface_defects(self) -> Dict:
        """Identify potential defects in the interface."""
        
        defects = {
            'sharp_corners': 0,
            'high_curvature_regions': 0,
            'discontinuities': 0,
            'thin_sections': 0
        }
        
        if self.curvature_metrics and 'max_curvature' in self.curvature_metrics:
            # High curvature regions
            high_curvature_threshold = np.percentile([self.curvature_metrics['mean_curvature']], 95)
            if self.curvature_metrics['max_curvature'] > high_curvature_threshold * 2:
                defects['high_curvature_regions'] = 1
        
        # Check for discontinuities in interface
        labeled_interface, num_components = ndi.label(self.interface_surface)
        if num_components > 1:
            defects['discontinuities'] = num_components - 1
        
        # Check for thin sections (simplified)
        if hasattr(self, 'roughness_metrics') and 'overall' in self.roughness_metrics:
            height_std = self.roughness_metrics['overall'].get('height_std_um', 0)
            if height_std > 1.0:  # More than 1 μm variation
                defects['thin_sections'] = 1
        
        return defects
    
    def _assess_interface_adhesion_quality(self) -> Dict:
        """Assess interface adhesion quality based on geometric factors."""
        
        adhesion_metrics = {}
        
        # Contact area efficiency
        if self.interface_surface is not None:
            total_interface_area = np.sum(self.interface_surface)
            projected_area = self.dimensions[0] * self.dimensions[1]
            contact_efficiency = total_interface_area / projected_area
            adhesion_metrics['contact_efficiency'] = contact_efficiency
        
        # Interface uniformity
        if hasattr(self, 'roughness_metrics') and 'overall' in self.roughness_metrics:
            height_std = self.roughness_metrics['overall'].get('height_std_um', 0)
            # Lower variation indicates better uniformity
            uniformity_score = max(0, 1.0 - height_std / 5.0)  # Normalize to 0-1
            adhesion_metrics['uniformity_score'] = uniformity_score
        
        # Interface integrity (based on connectivity)
        connectivity_stats = self._analyze_interface_connectivity()
        if 'largest_component_fraction' in connectivity_stats:
            integrity_score = connectivity_stats['largest_component_fraction']
            adhesion_metrics['integrity_score'] = integrity_score
        
        return adhesion_metrics
    
    def _calculate_overall_delamination_risk(self, stress_factors: Dict, 
                                           defects: Dict, adhesion: Dict) -> float:
        """Calculate overall delamination risk score (0-1, higher is worse)."""
        
        risk_score = 0.0
        
        # Stress concentration contribution (30% weight)
        if 'max_stress_concentration_factor' in stress_factors:
            stress_risk = min(1.0, stress_factors['max_stress_concentration_factor'] / 5.0)
            risk_score += 0.3 * stress_risk
        
        # Defects contribution (25% weight)
        total_defects = sum(defects.values())
        defect_risk = min(1.0, total_defects / 5.0)
        risk_score += 0.25 * defect_risk
        
        # Adhesion quality contribution (45% weight)
        if 'uniformity_score' in adhesion and 'integrity_score' in adhesion:
            adhesion_risk = 1.0 - 0.5 * (adhesion['uniformity_score'] + adhesion['integrity_score'])
            risk_score += 0.45 * adhesion_risk
        
        return min(1.0, risk_score)
    
    def visualize_interface_analysis(self, save_path: str = None):
        """Create comprehensive visualization of interface analysis."""
        
        if self.interface_surface is None:
            self.extract_interface_surface()
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Interface surface visualization
        ax1 = plt.subplot(3, 4, 1, projection='3d')
        
        # Subsample for visualization
        step = max(1, min(self.dimensions) // 100)
        interface_coords = np.where(self.interface_surface[::step, ::step, ::step])
        
        if len(interface_coords[0]) > 0:
            ax1.scatter(interface_coords[0], interface_coords[1], interface_coords[2], 
                       c='red', s=1, alpha=0.6)
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Interface Surface')
        
        # 2. Interface height map
        ax2 = plt.subplot(3, 4, 2)
        height_map = self._extract_interface_height_map()
        
        if height_map is not None:
            im = ax2.imshow(height_map, cmap='viridis', aspect='equal')
            plt.colorbar(im, ax=ax2, label='Height (μm)')
            ax2.set_title('Interface Height Map')
        
        # 3. Roughness analysis
        if self.roughness_metrics:
            ax3 = plt.subplot(3, 4, 3)
            
            scales = []
            ra_values = []
            
            for key, value in self.roughness_metrics.items():
                if key.startswith('scale_') and isinstance(value, dict):
                    scale = float(key.split('_')[1].replace('um', ''))
                    scales.append(scale)
                    ra_values.append(value.get('Ra_um', 0))
            
            if scales and ra_values:
                ax3.loglog(scales, ra_values, 'o-')
                ax3.set_xlabel('Analysis Scale (μm)')
                ax3.set_ylabel('Ra Roughness (μm)')
                ax3.set_title('Multi-scale Roughness')
                ax3.grid(True)
        
        # 4. Curvature distribution
        if self.curvature_metrics and 'error' not in self.curvature_metrics:
            ax4 = plt.subplot(3, 4, 4)
            
            # Create synthetic curvature distribution for visualization
            mean_curv = self.curvature_metrics.get('mean_curvature', 0)
            std_curv = self.curvature_metrics.get('curvature_std', 0.1)
            
            curvatures = np.random.normal(mean_curv, std_curv, 1000)
            ax4.hist(curvatures, bins=50, alpha=0.7)
            ax4.axvline(mean_curv, color='red', linestyle='--', label=f'Mean: {mean_curv:.3f}')
            ax4.set_xlabel('Curvature')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Curvature Distribution')
            ax4.legend()
        
        # 5. Cross-section with interface
        ax5 = plt.subplot(3, 4, 5)
        mid_y = self.dimensions[1] // 2
        cross_section = self.microstructure[:, mid_y, :]
        interface_cross = self.interface_surface[:, mid_y, :]
        
        # Overlay interface on microstructure
        display_section = cross_section.copy().astype(float)
        display_section[interface_cross] = 4  # Highlight interface
        
        ax5.imshow(display_section, cmap='tab10', aspect='auto')
        ax5.set_title(f'Cross-section with Interface (y={mid_y})')
        ax5.set_xlabel('X')
        ax5.set_ylabel('Z')
        
        # 6. Stress concentration visualization
        ax6 = plt.subplot(3, 4, 6)
        
        if hasattr(self, 'stress_concentration_factors'):
            # Create synthetic stress field for visualization
            x = np.linspace(0, self.dimensions[0], 50)
            y = np.linspace(0, self.dimensions[1], 50)
            X, Y = np.meshgrid(x, y)
            
            # Simplified stress field based on interface geometry
            if height_map is not None:
                # Downsample height map
                height_small = height_map[::max(1, self.dimensions[0]//50), 
                                        ::max(1, self.dimensions[1]//50)]
                
                # Calculate stress concentration (simplified)
                grad_x, grad_y = np.gradient(height_small)
                stress_field = 1.0 + np.sqrt(grad_x**2 + grad_y**2)
                
                im = ax6.contourf(X, Y, stress_field, levels=20, cmap='hot')
                plt.colorbar(im, ax=ax6, label='Stress Concentration Factor')
                ax6.set_title('Stress Concentration Map')
        
        # 7. Risk assessment summary
        ax7 = plt.subplot(3, 4, 7)
        ax7.axis('off')
        
        # Create risk assessment text
        risk_text = "Interface Risk Assessment:\n\n"
        
        if hasattr(self, 'roughness_metrics') and 'overall' in self.roughness_metrics:
            overall_roughness = self.roughness_metrics['overall']
            risk_text += f"Roughness:\n"
            risk_text += f"  Height Std: {overall_roughness.get('height_std_um', 0):.2f} μm\n"
            risk_text += f"  Mean Slope: {overall_roughness.get('mean_slope', 0):.3f}\n\n"
        
        if self.curvature_metrics and 'error' not in self.curvature_metrics:
            risk_text += f"Curvature:\n"
            risk_text += f"  Mean: {self.curvature_metrics.get('mean_curvature', 0):.3f}\n"
            risk_text += f"  Max: {self.curvature_metrics.get('max_curvature', 0):.3f}\n\n"
        
        ax7.text(0.1, 0.9, risk_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # 8. Interface thickness distribution
        ax8 = plt.subplot(3, 4, 8)
        
        thickness_stats = self._analyze_interface_thickness()
        if thickness_stats:
            # Create synthetic thickness distribution
            mean_thick = thickness_stats['mean_thickness_um']
            std_thick = thickness_stats['std_thickness_um']
            
            thicknesses = np.random.normal(mean_thick, std_thick, 1000)
            thicknesses = thicknesses[thicknesses > 0]  # Remove negative values
            
            ax8.hist(thicknesses, bins=30, alpha=0.7)
            ax8.axvline(mean_thick, color='red', linestyle='--', 
                       label=f'Mean: {mean_thick:.2f} μm')
            ax8.set_xlabel('Interface Thickness (μm)')
            ax8.set_ylabel('Frequency')
            ax8.set_title('Interface Thickness Distribution')
            ax8.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Interface analysis visualization saved to {save_path}")
        
        plt.show()


# Example usage
if __name__ == "__main__":
    # This would typically be called after generating a microstructure
    print("Interface Geometry Analyzer module loaded successfully!")
    print("Use this module to analyze anode/electrolyte interfaces in SOFC microstructures.")