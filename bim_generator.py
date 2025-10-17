#!/usr/bin/env python3
"""
3D BIM and LiDAR Data Generator
Generates detailed 3D building information models and LiDAR point clouds
for the Building DNA Dataset
"""

import numpy as np
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from scipy.spatial import cKDTree
import warnings
warnings.filterwarnings('ignore')

@dataclass
class BIMElement:
    """Individual BIM element with geometric and material properties"""
    id: str
    element_type: str  # wall, floor, roof, window, door, column, beam
    material_id: str
    geometry: Dict[str, Any]  # vertices, faces, normals
    properties: Dict[str, Any]  # material, thermal, structural properties
    location: Tuple[float, float, float]  # x, y, z coordinates
    orientation: Tuple[float, float, float]  # rotation angles
    dimensions: Tuple[float, float, float]  # length, width, height
    level: int  # floor level
    zone: str  # thermal zone
    construction_id: str

@dataclass
class LiDARPoint:
    """Individual LiDAR point with coordinates and properties"""
    x: float
    y: float
    z: float
    intensity: float
    return_number: int
    number_of_returns: int
    classification: int  # 1=unclassified, 2=ground, 6=building, etc.
    scan_angle: float
    gps_time: float
    red: int = 0
    green: int = 0
    blue: int = 0

@dataclass
class BIMModel:
    """Complete 3D BIM model"""
    model_id: str
    building_id: str
    elements: List[BIMElement]
    materials: Dict[str, Any]
    spaces: List[Dict[str, Any]]
    systems: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class BIMGenerator:
    """Generator for 3D BIM models and LiDAR data"""
    
    def __init__(self, seed: int = 42):
        """Initialize the BIM generator"""
        np.random.seed(seed)
        self.materials_db = self._initialize_bim_materials()
        
    def _initialize_bim_materials(self) -> Dict[str, Dict[str, Any]]:
        """Initialize materials database for BIM elements"""
        return {
            'concrete': {
                'name': 'Concrete',
                'density': 2400,  # kg/m³
                'thermal_conductivity': 1.7,  # W/m·K
                'specific_heat': 1000,  # J/kg·K
                'youngs_modulus': 30000,  # MPa
                'poisson_ratio': 0.2,
                'thermal_expansion': 12e-6,  # 1/K
                'color': [0.7, 0.7, 0.7],
                'texture': 'concrete'
            },
            'steel': {
                'name': 'Steel',
                'density': 7850,  # kg/m³
                'thermal_conductivity': 50,  # W/m·K
                'specific_heat': 460,  # J/kg·K
                'youngs_modulus': 200000,  # MPa
                'poisson_ratio': 0.3,
                'thermal_expansion': 12e-6,  # 1/K
                'color': [0.5, 0.5, 0.5],
                'texture': 'steel'
            },
            'brick': {
                'name': 'Brick',
                'density': 1800,  # kg/m³
                'thermal_conductivity': 0.77,  # W/m·K
                'specific_heat': 840,  # J/kg·K
                'youngs_modulus': 10000,  # MPa
                'poisson_ratio': 0.2,
                'thermal_expansion': 5e-6,  # 1/K
                'color': [0.8, 0.4, 0.2],
                'texture': 'brick'
            },
            'glass': {
                'name': 'Glass',
                'density': 2500,  # kg/m³
                'thermal_conductivity': 1.0,  # W/m·K
                'specific_heat': 840,  # J/kg·K
                'youngs_modulus': 70000,  # MPa
                'poisson_ratio': 0.23,
                'thermal_expansion': 9e-6,  # 1/K
                'color': [0.9, 0.9, 1.0],
                'texture': 'glass'
            },
            'wood': {
                'name': 'Wood',
                'density': 600,  # kg/m³
                'thermal_conductivity': 0.14,  # W/m·K
                'specific_heat': 1200,  # J/kg·K
                'youngs_modulus': 12000,  # MPa
                'poisson_ratio': 0.3,
                'thermal_expansion': 5e-6,  # 1/K
                'color': [0.6, 0.4, 0.2],
                'texture': 'wood'
            },
            'insulation': {
                'name': 'Insulation',
                'density': 20,  # kg/m³
                'thermal_conductivity': 0.04,  # W/m·K
                'specific_heat': 1000,  # J/kg·K
                'youngs_modulus': 1,  # MPa
                'poisson_ratio': 0.3,
                'thermal_expansion': 20e-6,  # 1/K
                'color': [0.9, 0.9, 0.5],
                'texture': 'insulation'
            }
        }
    
    def generate_rectangular_geometry(self, length: float, width: float, height: float, 
                                    center: Tuple[float, float, float] = (0, 0, 0)) -> Dict[str, Any]:
        """Generate rectangular geometry for BIM elements"""
        x, y, z = center
        
        # Define vertices of rectangular box
        vertices = np.array([
            [x - length/2, y - width/2, z - height/2],  # 0
            [x + length/2, y - width/2, z - height/2],  # 1
            [x + length/2, y + width/2, z - height/2],  # 2
            [x - length/2, y + width/2, z - height/2],  # 3
            [x - length/2, y - width/2, z + height/2],  # 4
            [x + length/2, y - width/2, z + height/2],  # 5
            [x + length/2, y + width/2, z + height/2],  # 6
            [x - length/2, y + width/2, z + height/2],  # 7
        ])
        
        # Define faces (triangles)
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ])
        
        # Calculate normals
        normals = []
        for face in faces:
            v1 = vertices[face[1]] - vertices[face[0]]
            v2 = vertices[face[2]] - vertices[face[0]]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            normals.append(normal)
        
        return {
            'vertices': vertices.tolist(),
            'faces': faces.tolist(),
            'normals': normals,
            'type': 'rectangular',
            'dimensions': [length, width, height]
        }
    
    def generate_wall_element(self, start_point: Tuple[float, float, float], 
                            end_point: Tuple[float, float, float], 
                            height: float, thickness: float, 
                            material: str = 'concrete') -> BIMElement:
        """Generate wall BIM element"""
        # Calculate wall properties
        length = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        center_x = (start_point[0] + end_point[0]) / 2
        center_y = (start_point[1] + end_point[1]) / 2
        center_z = start_point[2] + height / 2
        
        # Generate geometry
        geometry = self.generate_rectangular_geometry(length, thickness, height, 
                                                    (center_x, center_y, center_z))
        
        # Calculate orientation
        angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
        orientation = (0, 0, np.degrees(angle))
        
        # Get material properties
        material_props = self.materials_db[material]
        
        return BIMElement(
            id=str(uuid.uuid4()),
            element_type='wall',
            material_id=material,
            geometry=geometry,
            properties={
                'material': material_props,
                'thermal': {
                    'u_value': 1.0 / (thickness / 1000 / material_props['thermal_conductivity']),
                    'r_value': thickness / 1000 / material_props['thermal_conductivity'],
                    'thermal_mass': material_props['density'] * material_props['specific_heat'] * thickness / 1000
                },
                'structural': {
                    'load_capacity': material_props['youngs_modulus'] * thickness / 1000,
                    'flexural_strength': material_props['youngs_modulus'] * 0.1
                }
            },
            location=(center_x, center_y, center_z),
            orientation=orientation,
            dimensions=(length, thickness, height),
            level=int(center_z // 3),  # Assume 3m per floor
            zone=f'zone_{int(center_z // 3)}',
            construction_id=f'wall_{material}_{thickness}mm'
        )
    
    def generate_floor_element(self, corners: List[Tuple[float, float]], 
                             level: int, thickness: float = 0.2,
                             material: str = 'concrete') -> BIMElement:
        """Generate floor BIM element"""
        # Calculate floor properties
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        length = max(x_coords) - min(x_coords)
        width = max(y_coords) - min(y_coords)
        center_x = (max(x_coords) + min(x_coords)) / 2
        center_y = (max(y_coords) + min(y_coords)) / 2
        center_z = level * 3.0  # Assume 3m per floor
        
        # Generate geometry
        geometry = self.generate_rectangular_geometry(length, width, thickness, 
                                                    (center_x, center_y, center_z))
        
        # Get material properties
        material_props = self.materials_db[material]
        
        return BIMElement(
            id=str(uuid.uuid4()),
            element_type='floor',
            material_id=material,
            geometry=geometry,
            properties={
                'material': material_props,
                'thermal': {
                    'u_value': 1.0 / (thickness / material_props['thermal_conductivity']),
                    'r_value': thickness / material_props['thermal_conductivity'],
                    'thermal_mass': material_props['density'] * material_props['specific_heat'] * thickness
                },
                'structural': {
                    'load_capacity': material_props['youngs_modulus'] * thickness,
                    'flexural_strength': material_props['youngs_modulus'] * 0.1
                }
            },
            location=(center_x, center_y, center_z),
            orientation=(0, 0, 0),
            dimensions=(length, width, thickness),
            level=level,
            zone=f'zone_{level}',
            construction_id=f'floor_{material}_{thickness}m'
        )
    
    def generate_window_element(self, wall_element: BIMElement, 
                              window_width: float, window_height: float,
                              sill_height: float = 1.0) -> BIMElement:
        """Generate window BIM element in a wall"""
        # Calculate window position
        wall_center = wall_element.location
        wall_length = wall_element.dimensions[0]
        wall_height = wall_element.dimensions[2]
        
        # Random position along wall
        window_x = wall_center[0] + np.random.uniform(-wall_length/4, wall_length/4)
        window_y = wall_center[1] + np.random.uniform(-wall_length/4, wall_length/4)
        window_z = wall_center[2] - wall_height/2 + sill_height + window_height/2
        
        # Generate geometry
        geometry = self.generate_rectangular_geometry(window_width, wall_element.dimensions[1], 
                                                    window_height, (window_x, window_y, window_z))
        
        # Get glass material properties
        material_props = self.materials_db['glass']
        
        return BIMElement(
            id=str(uuid.uuid4()),
            element_type='window',
            material_id='glass',
            geometry=geometry,
            properties={
                'material': material_props,
                'thermal': {
                    'u_value': 2.5,  # W/m²·K
                    'shgc': 0.6,  # Solar Heat Gain Coefficient
                    'visible_transmittance': 0.8
                },
                'optical': {
                    'transmittance': 0.8,
                    'reflectance': 0.1,
                    'absorptance': 0.1
                }
            },
            location=(window_x, window_y, window_z),
            orientation=wall_element.orientation,
            dimensions=(window_width, wall_element.dimensions[1], window_height),
            level=wall_element.level,
            zone=wall_element.zone,
            construction_id='window_double_glazed'
        )
    
    def generate_column_element(self, location: Tuple[float, float, float], 
                              height: float, cross_section: Tuple[float, float],
                              material: str = 'concrete') -> BIMElement:
        """Generate column BIM element"""
        # Generate geometry
        geometry = self.generate_rectangular_geometry(cross_section[0], cross_section[1], height, location)
        
        # Get material properties
        material_props = self.materials_db[material]
        
        return BIMElement(
            id=str(uuid.uuid4()),
            element_type='column',
            material_id=material,
            geometry=geometry,
            properties={
                'material': material_props,
                'structural': {
                    'load_capacity': material_props['youngs_modulus'] * cross_section[0] * cross_section[1],
                    'buckling_load': material_props['youngs_modulus'] * cross_section[0] * cross_section[1] * 0.1
                }
            },
            location=location,
            orientation=(0, 0, 0),
            dimensions=(cross_section[0], cross_section[1], height),
            level=int(location[2] // 3),
            zone=f'zone_{int(location[2] // 3)}',
            construction_id=f'column_{material}_{cross_section[0]}x{cross_section[1]}'
        )
    
    def generate_complete_bim_model(self, building_geometry: Dict[str, Any]) -> BIMModel:
        """Generate complete 3D BIM model from building geometry"""
        print("Generating 3D BIM model...")
        
        elements = []
        building_id = building_geometry['building_id']
        building_type = building_geometry['building_type']
        
        # Extract building dimensions
        floor_areas = building_geometry['floor_areas']
        floor_heights = building_geometry['floor_heights']
        num_floors = building_geometry['number_of_floors']
        
        # Calculate building footprint
        total_area = sum(floor_areas)
        avg_floor_area = total_area / num_floors
        building_length = np.sqrt(avg_floor_area * building_geometry['aspect_ratio'])
        building_width = avg_floor_area / building_length
        
        # Generate floors
        for level in range(num_floors):
            floor_corners = [
                (-building_length/2, -building_width/2),
                (building_length/2, -building_width/2),
                (building_length/2, building_width/2),
                (-building_length/2, building_width/2)
            ]
            floor_element = self.generate_floor_element(floor_corners, level)
            elements.append(floor_element)
        
        # Generate walls
        wall_height = sum(floor_heights)
        wall_thickness = 0.2  # 200mm
        
        # Exterior walls
        wall_positions = [
            # North wall
            ((-building_length/2, -building_width/2, 0), (building_length/2, -building_width/2, 0)),
            # South wall
            ((building_length/2, building_width/2, 0), (-building_length/2, building_width/2, 0)),
            # East wall
            ((building_length/2, -building_width/2, 0), (building_length/2, building_width/2, 0)),
            # West wall
            ((-building_length/2, building_width/2, 0), (-building_length/2, -building_width/2, 0))
        ]
        
        for start, end in wall_positions:
            wall_element = self.generate_wall_element(start, end, wall_height, wall_thickness)
            elements.append(wall_element)
            
            # Add windows to walls
            num_windows = np.random.randint(2, 6)
            for _ in range(num_windows):
                window_width = np.random.uniform(1.0, 2.0)
                window_height = np.random.uniform(1.2, 1.8)
                window_element = self.generate_window_element(wall_element, window_width, window_height)
                elements.append(window_element)
        
        # Generate structural columns
        if building_type in ['commercial', 'office', 'industrial']:
            column_spacing = 6.0  # 6m spacing
            num_columns_x = int(building_length / column_spacing) + 1
            num_columns_y = int(building_width / column_spacing) + 1
            
            for i in range(num_columns_x):
                for j in range(num_columns_y):
                    x = -building_length/2 + i * column_spacing
                    y = -building_width/2 + j * column_spacing
                    if x <= building_length/2 and y <= building_width/2:
                        column_element = self.generate_column_element(
                            (x, y, 0), wall_height, (0.4, 0.4)
                        )
                        elements.append(column_element)
        
        # Generate spaces/zones
        spaces = []
        for level in range(num_floors):
            spaces.append({
                'id': f'space_level_{level}',
                'name': f'Level {level}',
                'level': level,
                'area': floor_areas[level],
                'volume': floor_areas[level] * floor_heights[level],
                'zone_type': 'conditioned' if level > 0 else 'unconditioned',
                'occupancy': np.random.randint(10, 100) if building_type != 'residential' else np.random.randint(2, 6)
            })
        
        # Generate systems
        systems = []
        for level in range(num_floors):
            systems.append({
                'id': f'hvac_level_{level}',
                'type': 'HVAC',
                'level': level,
                'capacity': np.random.uniform(10, 50),  # kW
                'efficiency': np.random.uniform(0.7, 0.9)
            })
            
            systems.append({
                'id': f'lighting_level_{level}',
                'type': 'Lighting',
                'level': level,
                'power_density': np.random.uniform(8, 15),  # W/m²
                'efficiency': np.random.uniform(0.8, 0.95)
            })
        
        return BIMModel(
            model_id=str(uuid.uuid4()),
            building_id=building_id,
            elements=elements,
            materials=self.materials_db,
            spaces=spaces,
            systems=systems,
            metadata={
                'generated_at': datetime.now().isoformat(),
                'building_type': building_type,
                'total_elements': len(elements),
                'total_volume': sum(space['volume'] for space in spaces),
                'total_area': sum(space['area'] for space in spaces)
            }
        )
    
    def generate_lidar_point_cloud(self, bim_model: BIMModel, 
                                 point_density: float = 1000) -> List[LiDARPoint]:
        """Generate LiDAR point cloud from BIM model"""
        print("Generating LiDAR point cloud...")
        
        points = []
        
        # Generate points for each element
        for element in bim_model.elements:
            if element.element_type in ['wall', 'floor', 'roof']:
                # Generate points on element surfaces
                vertices = np.array(element.geometry['vertices'])
                faces = np.array(element.geometry['faces'])
                
                # Calculate element area
                element_area = self._calculate_element_area(vertices, faces)
                num_points = int(element_area * point_density)
                
                # Generate random points on element surface
                for _ in range(num_points):
                    # Select random face
                    face_idx = np.random.randint(0, len(faces))
                    face = faces[face_idx]
                    
                    # Generate random point on face using barycentric coordinates
                    u, v = np.random.random(2)
                    if u + v > 1:
                        u, v = 1 - u, 1 - v
                    w = 1 - u - v
                    
                    point = (u * vertices[face[0]] + v * vertices[face[1]] + w * vertices[face[2]])
                    
                    # Add noise
                    noise = np.random.normal(0, 0.01, 3)
                    point += noise
                    
                    # Generate LiDAR properties
                    intensity = np.random.uniform(0.1, 1.0)
                    return_number = np.random.randint(1, 4)
                    number_of_returns = np.random.randint(1, 5)
                    classification = 6 if element.element_type == 'wall' else 2  # building or ground
                    scan_angle = np.random.uniform(-30, 30)
                    gps_time = np.random.uniform(0, 86400)  # seconds in day
                    
                    # Generate color based on material
                    material = bim_model.materials[element.material_id]
                    color = material['color']
                    red = int(color[0] * 255)
                    green = int(color[1] * 255)
                    blue = int(color[2] * 255)
                    
                    lidar_point = LiDARPoint(
                        x=point[0], y=point[1], z=point[2],
                        intensity=intensity,
                        return_number=return_number,
                        number_of_returns=number_of_returns,
                        classification=classification,
                        scan_angle=scan_angle,
                        gps_time=gps_time,
                        red=red, green=green, blue=blue
                    )
                    points.append(lidar_point)
        
        return points
    
    def _calculate_element_area(self, vertices: np.ndarray, faces: np.ndarray) -> float:
        """Calculate surface area of element"""
        total_area = 0
        for face in faces:
            v1 = vertices[face[1]] - vertices[face[0]]
            v2 = vertices[face[2]] - vertices[face[0]]
            area = 0.5 * np.linalg.norm(np.cross(v1, v2))
            total_area += area
        return total_area
    
    def visualize_bim_model(self, bim_model: BIMModel, save_path: str = None):
        """Visualize 3D BIM model using Plotly"""
        print("Generating 3D visualization...")
        
        fig = go.Figure()
        
        # Add elements
        for element in bim_model.elements:
            vertices = np.array(element.geometry['vertices'])
            faces = np.array(element.geometry['faces'])
            
            # Get material color
            material = bim_model.materials[element.material_id]
            color = f'rgb({int(material["color"][0]*255)}, {int(material["color"][1]*255)}, {int(material["color"][2]*255)})'
            
            # Create mesh
            fig.add_trace(go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=0.7,
                name=f'{element.element_type}_{element.id[:8]}'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'3D BIM Model - {bim_model.metadata["building_type"].title()} Building',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"3D visualization saved to: {save_path}")
        else:
            fig.show()
    
    def visualize_lidar_points(self, points: List[LiDARPoint], save_path: str = None):
        """Visualize LiDAR point cloud"""
        print("Generating LiDAR visualization...")
        
        # Convert points to arrays
        x = [p.x for p in points]
        y = [p.y for p in points]
        z = [p.z for p in points]
        colors = [f'rgb({p.red}, {p.green}, {p.blue})' for p in points]
        intensities = [p.intensity for p in points]
        
        fig = go.Figure(data=go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=2,
                color=colors,
                opacity=0.6
            ),
            text=[f'Intensity: {i:.2f}' for i in intensities],
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title='LiDAR Point Cloud',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            )
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"LiDAR visualization saved to: {save_path}")
        else:
            fig.show()
    
    def export_bim_to_json(self, bim_model: BIMModel, filename: str = None) -> str:
        """Export BIM model to JSON format"""
        if filename is None:
            filename = f"bim_model_{bim_model.building_id}.json"
        
        # Convert to serializable format
        bim_data = {
            'model_id': bim_model.model_id,
            'building_id': bim_model.building_id,
            'elements': [asdict(element) for element in bim_model.elements],
            'materials': bim_model.materials,
            'spaces': bim_model.spaces,
            'systems': bim_model.systems,
            'metadata': bim_model.metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(bim_data, f, indent=2, default=str)
        
        print(f"BIM model exported to: {filename}")
        return filename
    
    def export_lidar_to_las(self, points: List[LiDARPoint], filename: str = None) -> str:
        """Export LiDAR points to LAS format (simplified JSON for now)"""
        if filename is None:
            filename = f"lidar_points_{len(points)}.json"
        
        # Convert to serializable format
        lidar_data = {
            'points': [asdict(point) for point in points],
            'metadata': {
                'total_points': len(points),
                'generated_at': datetime.now().isoformat(),
                'coordinate_system': 'WGS84',
                'units': 'meters'
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(lidar_data, f, indent=2, default=str)
        
        print(f"LiDAR data exported to: {filename}")
        return filename

def main():
    """Main function to demonstrate BIM and LiDAR generation"""
    print("3D BIM and LiDAR Data Generator")
    print("=" * 40)
    
    # Initialize generator
    generator = BIMGenerator(seed=42)
    
    # Create sample building geometry
    from building_dna_generator import BuildingDNAGenerator
    dna_generator = BuildingDNAGenerator(seed=42)
    building_geometry = dna_generator.generate_building_geometry('office')
    
    # Generate BIM model
    bim_model = generator.generate_complete_bim_model(building_geometry)
    
    # Generate LiDAR point cloud
    lidar_points = generator.generate_lidar_point_cloud(bim_model, point_density=500)
    
    # Export data
    bim_file = generator.export_bim_to_json(bim_model)
    lidar_file = generator.export_lidar_to_las(lidar_points)
    
    # Generate visualizations
    generator.visualize_bim_model(bim_model, 'bim_visualization.html')
    generator.visualize_lidar_points(lidar_points, 'lidar_visualization.html')
    
    print(f"\nGenerated BIM model with {len(bim_model.elements)} elements")
    print(f"Generated LiDAR point cloud with {len(lidar_points)} points")
    print("Visualizations saved as HTML files")

if __name__ == "__main__":
    main()